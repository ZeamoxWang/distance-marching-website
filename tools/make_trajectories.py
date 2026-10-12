"""
Generate the data for the interactive trajectory viewer on the project page.

For the 16 curated ImageNet samples of the paper (DM-XL/2, CFG 4.0, 250 steps;
sphere tracing with step 0.015 and gradient descent with step 0.5), this script

1. recovers the initial noise and class label of each sample by replaying the
   RNG of the original sampler (torch.manual_seed(global_seed * W + rank), then
   z = randn(n, 4, 32, 32), y = randint(0, 1000, (n,)) per rank) and picks the
   (world size, per-rank batch) layout whose labels match the curated classes,
2. reruns both samplers and records all 250 latent states of each trajectory,
3. applies PCA to the states of each trajectory (as in the paper appendix,
   "Local image-space geometry"), with the first three principal directions
   P = [p1, p2, p3] and the generated endpoint x_end as origin,
4. evaluates the distance head u_theta(x, y) on a 3D grid x_end + P z,
5. decodes the states every 50 steps (0, 50, ..., 200, 249) to thumbnails,
6. compares the regenerated final images with the curated PNGs.

Outputs (under --out): manifest.json, <sampler>/<index>/{meta.json,grid.bin,
step_XXX.jpg}, <sampler>/<index>.jpg (final image), report.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CURATED = [1, 4, 19, 33, 44, 56, 58, 60, 63, 71, 75, 85, 87, 92, 123, 125]
# ImageNet classes recognizable in the curated grid, used to identify the RNG layout.
EXPECTED = {1: 0, 4: 853, 19: 548, 33: 291, 44: 378, 56: 324, 58: 873, 71: 105,
            75: 933, 85: 261, 92: 812, 123: 84, 125: 518}
SAMPLERS = {"st": 0.015, "gd": 0.5}
THUMB_STEPS = [0, 50, 100, 150, 200, 249]


def draw(seed, n, predraw, device):
    g = torch.manual_seed(seed)
    del g
    if predraw:  # sample_gd_cfg.py draws one extra (y, z) batch before the loop
        torch.randint(1000, size=(n,), device=device)
        torch.randn(n, 4, 32, 32, device=device)
    z = torch.randn(n, 4, 32, 32, device=device)
    y = torch.randint(0, 1000, (n,), device=device)
    return z, y


def find_layout(device, global_seed):
    results = []
    for world in [1, 2, 4, 8]:
        for n in [16, 32, 64, 128, 256]:
            if max(CURATED) // world >= n:
                continue
            for predraw in [False, True]:
                cache = {}
                labels = {}
                for k in CURATED:
                    r, i = k % world, k // world
                    if r not in cache:
                        cache[r] = draw(global_seed * world + r, n, predraw, device)
                    labels[k] = int(cache[r][1][i])
                score = sum(labels[k] == c for k, c in EXPECTED.items())
                results.append(dict(world=world, n=n, predraw=predraw, score=score, labels=labels))
    results.sort(key=lambda d: -d["score"])
    return results


def gather_init(layout, device, global_seed):
    world, n, predraw = layout["world"], layout["n"], layout["predraw"]
    zs, ys = [], []
    for k in CURATED:
        r, i = k % world, k // world
        z, y = draw(global_seed * world + r, n, predraw, device)
        zs.append(z[i])
        ys.append(y[i])
    return torch.stack(zs), torch.stack(ys)


@torch.no_grad()
def run_sampler(model, z, y, sampler, stepsize, cfg_scale, steps):
    """Same update as imagenet/sample.py; returns the cond-half states and u along the path."""
    n = z.shape[0]
    t = torch.ones((2 * n,), device=z.device)
    xt = torch.cat([z, z], 0)
    yy = torch.cat([y, torch.full_like(y, 1000)], 0)
    states = [z.clone().cpu()]
    us = []
    for _ in range(steps - 1):
        cond_out, cond_e, uncond_out, uncond_e = model.forward_with_cfg(xt, t, yy, cfg_scale, get_energy=True)
        us.append((-cond_e).float().cpu())
        if sampler == "gd":
            out = uncond_out + cfg_scale * (cond_out - uncond_out)
        else:
            uf = uncond_out * uncond_e.reshape(-1, 1, 1, 1)
            cf = cond_out * cond_e.reshape(-1, 1, 1, 1)
            out = uf + cfg_scale * (cf - uf)
        out = torch.cat([out, out], 0)
        xt = (xt + out * stepsize).detach()
        states.append(xt[:n].clone().cpu())
    u_last = model.noise_level_predictor(xt[:n], y, False)
    us.append(u_last.float().cpu())
    return torch.stack(states, 1), torch.stack(us, 1)  # (n, 250, 4, 32, 32), (n, 250)


@torch.no_grad()
def decode(vae, lat):
    img = vae.decode(lat / 0.18215).sample
    img = torch.clamp(127.5 * img + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return img


@torch.no_grad()
def u_grid(model, x_end, P, axes, y, device, bs=4096):
    """u_theta(x_end + P z, y) on the grid axes[0] x axes[1] x axes[2] (index order i, j, k)."""
    Z = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    Zt = torch.from_numpy(Z).to(device, torch.float32)
    Pt = torch.from_numpy(P).to(device, torch.float32)  # (D, 3)
    xe = torch.from_numpy(x_end).to(device, torch.float32)
    out = []
    for s in range(0, len(Zt), bs):
        x = (xe[None] + Zt[s:s + bs] @ Pt.T).reshape(-1, 4, 32, 32)
        yy = torch.full((x.shape[0],), int(y), device=device, dtype=torch.long)
        out.append(model.noise_level_predictor(x, yy, False).float().cpu())
    return torch.cat(out).numpy().reshape([len(a) for a in axes])


def main(args):
    sys.path.insert(0, args.code)
    from diffusers.models import AutoencoderKL
    from models import DM_models

    device = "cuda"
    t0 = time.time()
    report = {}

    layouts = find_layout(device, args.global_seed)
    best = layouts[0]
    report["layout_candidates"] = [{k: v for k, v in d.items() if k != "labels"} for d in layouts[:8]]
    report["layout"] = best
    print("best layout", best, flush=True)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = sd.get("args")
    rescale = getattr(ck_args, "rescale_energy", 2.78e-3)
    npc = getattr(ck_args, "noise_predictor_condition", True)
    report["ckpt_args"] = {k: str(v) for k, v in vars(ck_args).items()} if ck_args is not None else None
    model = DM_models[args.model](input_size=32, num_classes=1000, rescale_energy=rescale,
                                  noise_predictor_condition=npc)
    missing, unexpected = model.load_state_dict(sd["ema"], strict=False)
    report["missing_keys"], report["unexpected_keys"] = list(missing), list(unexpected)
    print("missing", missing, "unexpected", unexpected, flush=True)
    assert not missing, missing
    del sd
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    z, y = gather_init(best, device, args.global_seed)
    labels = y.tolist()
    manifest = {"indices": CURATED, "labels": labels, "thumb_steps": THUMB_STEPS, "samplers": {},
                "grid_res": args.res, "model": args.model, "cfg_scale": args.cfg_scale,
                "num_sampling_steps": args.steps, "layout": {k: best[k] for k in ["world", "n", "predraw"]}}
    report["compare"] = {}
    for sampler, eta in SAMPLERS.items():
        print(f"sampling {sampler} eta={eta}", flush=True)
        states, us = run_sampler(model, z, y, sampler, eta, args.cfg_scale, args.steps)
        finals = decode(vae, states[:, -1].to(device))
        manifest["samplers"][sampler] = {"stepsize": eta}
        report["compare"][sampler] = {}
        os.makedirs(os.path.join(args.out, sampler), exist_ok=True)
        for j, k in enumerate(CURATED):
            Image.fromarray(finals[j]).save(os.path.join(args.out, sampler, f"{k:06d}.png"))
            ref = os.path.join(args.ref, sampler, f"{k:06d}.png")
            if os.path.exists(ref):
                r = np.asarray(Image.open(ref).convert("RGB")).astype(np.float32)
                report["compare"][sampler][k] = float(np.abs(r - finals[j].astype(np.float32)).mean())

            X = states[j].reshape(args.steps, -1).double().numpy()  # (250, 4096)
            x_end = X[-1]
            Xc = X - X.mean(0, keepdims=True)
            _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            P = Vt[:3].T  # (4096, 3), unit norm, orthogonal
            # orient each axis so that the trajectory starts at positive coordinates
            C = (X - x_end) @ P
            sign = np.where(C[0] < 0, -1.0, 1.0)
            P, C = P * sign, C * sign
            ratio = (S ** 2 / (S ** 2).sum())[:3]
            lo, hi = C.min(0), C.max(0)
            span = hi - lo
            m = args.margin * span
            axes = [np.linspace(lo[a] - m[a], hi[a] + m[a], args.res) for a in range(3)]
            grid = u_grid(model, x_end, P, axes, labels[j], device)
            d = os.path.join(args.out, sampler, f"{k:06d}")
            os.makedirs(d, exist_ok=True)
            grid.astype(np.float32).tofile(os.path.join(d, "grid.bin"))
            thumbs = decode(vae, states[j, THUMB_STEPS].to(device))
            for s, im in zip(THUMB_STEPS, thumbs):
                Image.fromarray(im).resize((160, 160), Image.BICUBIC).save(
                    os.path.join(d, f"step_{s:03d}.jpg"), quality=88)
            meta = {
                "index": k, "label": labels[j], "sampler": sampler, "stepsize": eta,
                "explained_variance_ratio": ratio.tolist(),
                "axes": [[float(a[0]), float(a[-1])] for a in axes],
                "grid_res": args.res, "grid_min": float(grid.min()), "grid_max": float(grid.max()),
                "coords": C.astype(np.float32).round(4).tolist(),
                "u": us[j].numpy().astype(np.float32).round(4).tolist(),
            }
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f)
            print(sampler, k, "label", labels[j], "ratio", np.round(ratio, 4), "u", float(us[j, 0]), "->",
                  float(us[j, -1]), "grid", float(grid.min()), float(grid.max()), flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    report["seconds"] = time.time() - t0
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("compare", json.dumps(report["compare"]), flush=True)
    print("DONE", report["seconds"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--code", required=True, help="imagenet/ folder of the official codebase (models.py)")
    p.add_argument("--ref", required=True, help="folder with st/ and gd/ curated PNGs")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="DM-XL/2")
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--res", type=int, default=40)
    p.add_argument("--margin", type=float, default=0.15)
    main(p.parse_args())
