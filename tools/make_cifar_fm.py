"""
Flow-matching baseline trajectories for the CIFAR-10 viewer on the project page.

The baseline is our time-conditioned flow-matching model on the same U-Net
(/pv/flow_matching/output_dir/20251113_072621baseline, checkpoint-1899, EMA),
evaluated in /pv/flow_matching/output_dir/gt_2.36 with midpoint on the EDM time
grid (nfe 50), seed 0, batch 128: FID 2.36. Its first batch shares the initial
noise of the paper's uncurated Distance Marching batch when drawn on the same
GPU model (A10).

This script uses the original codebase (/pv/flow_matching/examples/image) and its
sampling wrapper (CFGScaledModel: fp16 autocast, cfg 0) and

1. reruns the evaluated recipe (midpoint, EDM grid, nfe 50) and compares the grid
   with the snapshot of gt_2.36,
2. integrates the same ODE with dopri5 (atol = rtol = 1e-5, the codebase defaults)
   and keeps the states at 251 uniform times on [t_0, 1] (t_0 = 0.0124, the start
   of the evaluated EDM grid), to pair
   with the 251 states of the 250-step Distance Marching trajectories,
3. writes fm_states.npy (251, 128, 3, 32, 32), fm/final.png (8 x 16 atlas) and
   fm/<index>_sheet.png (16 x 16 sprite sheet of 32 px tiles).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def to_uint8(x):
    x = torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
    return torch.floor(x * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def grid_png(imgs, nrow=8, pad=2):
    from torchvision.utils import make_grid
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    g = make_grid(t, nrow=nrow, padding=pad)
    return g.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()


def main(args):
    sys.path.insert(0, args.code)
    sys.path.insert(0, os.path.dirname(os.path.dirname(args.code.rstrip("/"))))  # flow_matching package
    from flow_matching.solver.ode_solver import ODESolver
    from models.model_configs import instantiate_model
    from training.edm_time_discretization import get_time_discretization

    class CFGScaledModel(torch.nn.Module):
        """Same forward as training/eval_loop.py of the fork with cfg_scale 0 (that module
        also imports torchmetrics, which this image lacks)."""

        def __init__(self, model):
            super().__init__()
            self.model = model
            self.nfe_counter = 0

        def forward(self, x, t, cfg_scale, label):
            assert cfg_scale == 0.0
            t = torch.zeros(x.shape[0], device=x.device) + t
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model(x, t, extra={"label": label})
            self.nfe_counter += 1
            return result.to(dtype=torch.float32)

        def get_nfe(self):
            return self.nfe_counter

    device = "cuda"
    report = {}
    os.makedirs(os.path.join(args.out, "fm"), exist_ok=True)
    model = instantiate_model(architechture="cifar10", is_discrete=False, use_ema=True, max_period=10000)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    report["ckpt_epoch"] = ck.get("epoch")
    model.to(device)
    wrapped = CFGScaledModel(model=model)
    wrapped.train(False)  # EMA weights
    solver = ODESolver(velocity_model=wrapped)

    if args.x0:  # initial noise of another GPU model (cuda_randn.py)
        x0 = torch.from_numpy(np.load(args.x0)).to(device, torch.float32)
    else:
        torch.manual_seed(args.seed)
        x0 = torch.randn((args.n, 3, 32, 32), dtype=torch.float32, device=device)
    label = torch.zeros(args.n, dtype=torch.long, device=device)  # unused: unconditional U-Net
    extras = {"label": label, "cfg_scale": 0.0}

    # 1) the evaluated recipe
    tg = get_time_discretization(nfes=50).to(device)
    xm = solver.sample(time_grid=tg, x_init=x0, method="midpoint", return_intermediates=False,
                       atol=1e-5, rtol=1e-5, step_size=None, **extras)
    gm = grid_png(to_uint8(xm))
    Image.fromarray(gm).save(os.path.join(args.out, "fm_midpoint_grid.png"))
    ref = np.asarray(Image.open(args.ref).convert("RGB"))
    report["midpoint_vs_gt_2.36_snapshot_mae"] = float(np.abs(gm.astype(np.float32) - ref.astype(np.float32)).mean())
    print("midpoint vs gt_2.36 snapshot MAE:", report["midpoint_vs_gt_2.36_snapshot_mae"], flush=True)

    # 2) dense trajectory
    # same interval as the evaluated EDM grid (which starts at t = 0.0124, not 0)
    times = torch.linspace(float(tg[0]), 1.0, args.steps + 1, device=device, dtype=torch.float32)
    report["time_interval"] = [float(tg[0]), 1.0]
    sol = solver.sample(time_grid=times, x_init=x0, method="dopri5", return_intermediates=True,
                        atol=1e-5, rtol=1e-5, step_size=None, **extras)  # (251, n, 3, 32, 32)
    final = to_uint8(sol[-1])
    report["dopri5_vs_midpoint_final_mae"] = float(np.abs(final.astype(np.float32) - to_uint8(xm).astype(np.float32)).mean())
    report["nfe_dopri5"] = wrapped.get_nfe()
    print("dopri5 vs midpoint final MAE:", report["dopri5_vs_midpoint_final_mae"], flush=True)
    np.save(os.path.join(args.out, "fm_states.npy"), sol.float().cpu().numpy())
    atlas = final.reshape(args.n // 16, 16, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(args.n // 16 * 32, 16 * 32, 3)
    Image.fromarray(atlas).save(os.path.join(args.out, "fm", "final.png"))
    T = sol.shape[0]
    for j in range(args.n):
        ims = to_uint8(sol[:, j])
        sheet = Image.new("RGB", (16 * 32, ((T + 15) // 16) * 32), "white")
        for t in range(T):
            sheet.paste(Image.fromarray(ims[t]), ((t % 16) * 32, (t // 16) * 32))
        sheet.save(os.path.join(args.out, "fm", f"{j:03d}_sheet.png"))
    with open(os.path.join(args.out, "fm_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("FM DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, help="examples/image of the original flow_matching fork")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ref", required=True, help="snapshot of the gt_2.36 evaluation (first batch)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x0", default="", help="initial noise (.npy) from cuda_randn.py instead of torch.randn")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--steps", type=int, default=250)
    main(p.parse_args())
