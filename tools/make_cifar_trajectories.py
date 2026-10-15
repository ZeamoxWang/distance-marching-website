"""
Generate the data for the CIFAR-10 trajectory viewer on the project page.

The paper's uncurated CIFAR-10 figure is the first sampling batch (128 images,
saved by torchvision's save_image with nrow=8) of the released checkpoint
(cifar10-checkpoint-1809.pth). This script

1. compares the paper grid with every snapshot PNG found under --search
   (old sampling/training runs) to identify the run it came from,
2. replays the first batch of cifar10/sample.py (seed 0, one process:
   torch.manual_seed(0), then x_0 = randn(128, 3, 32, 32) on the GPU) for
   sphere tracing (0.031) and gradient descent (0.46), 250 steps, keeping all
   251 states (step 0 = noise, step 250 = output),
3. compares the regenerated grids with the paper grid,
4. per trajectory: PCA of its 251 states in pixel space (3072-D), first three
   principal directions P, origin at the generated sample x_end, equal axis
   scales; u_theta on a 3D grid x_end + P z; u_theta along the path; all
   states as a 16 x 16 sprite sheet of 32 px tiles,
5. with --fm-states (from make_cifar_fm.py, same initial noise): projects the
   flow-matching trajectory on the same axes (the PCA is fitted on ours only,
   the cube covers both), and stores the displacement |x_end - x_0| and the
   path length (sum of segment lengths) of both trajectories in pixel space.

Outputs (under --out): manifest.json, report.json, <sampler>/final.png (8 x 16
atlas, 32 px, no padding), <sampler>/<index>/{meta.json,grid.bin,sheet.png}.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

SAMPLERS = {"st": 0.031, "gd": 0.46}


def to_uint8(x):
    """[-1, 1] -> uint8 HWC, same quantization as training/eval_loop.py."""
    x = torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
    x = torch.floor(x * 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def grid_png(imgs, nrow=8, pad=2):
    """Same layout as torchvision.utils.save_image(nrow=8, padding=2)."""
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    from torchvision.utils import make_grid
    g = make_grid(t, nrow=nrow, padding=pad)
    return (g.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy())


def mae(a, b):
    if a.shape != b.shape:
        return None
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


@torch.no_grad()
def u_of(model, x, bs=8192):
    out = []
    for s in range(0, x.shape[0], bs):
        out.append(model.t_predictor(x[s:s + bs]).float().cpu())
    return torch.cat(out)


def main(args):
    sys.path.insert(0, args.code)
    from models.model_configs import instantiate_model
    from dm.solver import DistanceMarchingSolver

    t0 = time.time()
    report = {}
    os.makedirs(args.out, exist_ok=True)
    ref = np.asarray(Image.open(args.ref).convert("RGB"))
    report["ref_shape"] = list(ref.shape)

    fm = np.load(args.fm_states).astype(np.float64) if args.fm_states else None  # (251, n, 3, 32, 32)
    # 1) which old snapshot is the paper grid?
    found = []
    for pat in args.search:
        for f in sorted(glob.glob(pat, recursive=True)):
            try:
                im = np.asarray(Image.open(f).convert("RGB"))
            except Exception:
                continue
            m = mae(im, ref)
            if m is not None:
                found.append((m, f))
    found.sort()
    report["snapshot_matches"] = found[:20]
    print("closest snapshots:", found[:10], flush=True)

    device = "cuda"
    model = instantiate_model("cifar10", use_ema=True, max_period=1000000.0, unified_scale=120.0)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    report["ckpt_epoch"] = ck.get("epoch")
    model.to(device)
    model.train(False)  # copies the EMA weights into the model
    net = model.model
    solver = DistanceMarchingSolver(velocity_model=model)

    if args.x0:  # initial noise of another GPU model (cuda_randn.py)
        x0 = torch.from_numpy(np.load(args.x0)).to(device, torch.float32)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        x0 = torch.randn((args.n, 3, 32, 32), dtype=torch.float32, device=device)

    manifest = {"n": args.n, "samplers": {}, "steps": args.steps, "grid_res": args.res,
                "seed": args.seed, "ckpt": os.path.basename(args.ckpt), "sheet": {"tile": 32, "cols": 16}}
    report["compare"] = {}
    for sampler, eta in SAMPLERS.items():
        traj = solver.sample(x_init=x0, step_size=eta, method=sampler, num_steps=args.steps,
                             return_intermediates=True)  # (251, n, 3, 32, 32)
        T = traj.shape[0]
        final = to_uint8(traj[-1])
        g = grid_png(final)
        Image.fromarray(g).save(os.path.join(args.out, f"{sampler}_grid.png"))
        report["compare"][sampler] = mae(g, ref)
        print(sampler, "grid vs paper MAE:", report["compare"][sampler], flush=True)
        d0 = os.path.join(args.out, sampler)
        os.makedirs(d0, exist_ok=True)
        atlas = final.reshape(args.n // 16, 16, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(args.n // 16 * 32, 16 * 32, 3)
        Image.fromarray(atlas).save(os.path.join(d0, "final.png"))
        manifest["samplers"][sampler] = {"stepsize": eta}

        U = u_of(net, traj.reshape(-1, 3, 32, 32)).reshape(T, args.n)
        for j in range(args.n):
            X = traj[:, j].reshape(T, -1).double().cpu().numpy()
            x_end = X[-1]
            _, S, Vt = np.linalg.svd(X - X.mean(0, keepdims=True), full_matrices=False)
            P = Vt[:3].T
            C = (X - x_end) @ P
            sign = np.where(C[0] < 0, -1.0, 1.0)
            P, C = P * sign, C * sign
            ratio = (S ** 2 / (S ** 2).sum())[:3]
            both = C
            if fm is not None:
                Xf = fm[:, j].reshape(fm.shape[0], -1)
                Cf = (Xf - x_end) @ P
                both = np.concatenate([C, Cf], 0)
            lo, hi = both.min(0), both.max(0)
            half = (1 + 2 * args.margin) * (hi - lo).max() / 2
            mid = (lo + hi) / 2
            axes = [np.linspace(mid[a] - half, mid[a] + half, args.res) for a in range(3)]
            Z = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            xg = torch.from_numpy(x_end[None] + Z @ P.T).to(device, torch.float32).reshape(-1, 3, 32, 32)
            grid = u_of(net, xg).numpy().reshape(args.res, args.res, args.res)
            d = os.path.join(d0, f"{j:03d}")
            os.makedirs(d, exist_ok=True)
            grid.astype(np.float32).tofile(os.path.join(d, "grid.bin"))
            ims = to_uint8(traj[:, j])
            sheet = Image.new("RGB", (16 * 32, ((T + 15) // 16) * 32), "white")
            for t in range(T):
                sheet.paste(Image.fromarray(ims[t]), ((t % 16) * 32, (t // 16) * 32))
            sheet.save(os.path.join(d, "sheet.png"))
            meta = {"index": j, "sampler": sampler, "stepsize": eta,
                    "explained_variance_ratio": ratio.tolist(),
                    "axes": [[float(a[0]), float(a[-1])] for a in axes],
                    "grid_res": args.res, "coords": C.astype(np.float32).round(4).tolist(),
                    "u": U[:, j].numpy().round(4).tolist()}
            if fm is not None:
                geo = lambda Y: {"chord": float(np.linalg.norm(Y[-1] - Y[0])),
                                 "length": float(np.linalg.norm(np.diff(Y, axis=0), axis=1).sum())}
                meta["coords_fm"] = Cf.astype(np.float32).round(4).tolist()
                meta["u_fm"] = u_of(net, torch.from_numpy(Xf).to(device, torch.float32).reshape(-1, 3, 32, 32)).numpy().round(4).tolist()
                meta["geo"] = {"ours": geo(X), "fm": geo(Xf)}
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f)
        print(sampler, "done", time.time() - t0, flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    report["seconds"] = time.time() - t0
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("DONE", json.dumps(report["compare"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, help="cifar10/ folder of the official codebase")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ref", required=True, help="the paper's original 16 x 8 sample grid (PNG)")
    p.add_argument("--search", nargs="*", default=[], help="glob patterns of old snapshot PNGs")
    p.add_argument("--fm-states", default="", help="fm_states.npy from make_cifar_fm.py")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x0", default="", help="initial noise (.npy) from cuda_randn.py instead of torch.randn")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--res", type=int, default=32)
    p.add_argument("--margin", type=float, default=0.15)
    main(p.parse_args())
