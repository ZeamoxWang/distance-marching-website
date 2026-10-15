"""
Package the output of make_cifar_trajectories.py for the CIFAR-10 viewer:
quantize each u_theta grid to uint16 (range stored in meta.json as grid_q),
round the coordinates, re-encode the sprite sheets as WebP (the 251 states of
a trajectory, mostly noise, are ~330 KB as PNG and ~80-120 KB as WebP), and
copy the final-image atlases; the flow-matching sprite sheets (shared by both
samplers) go to fm/<index>_sheet.webp.
The CIFAR-10 distance head outputs u_theta directly (no sign flip, unlike ImageNet).

Usage: python3 tools/package_cifar_web.py <job out dir> static/traj_cifar [samplers...]
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image


def main(src, dst, samplers=("st", "gd"), quality=88):
    mf = os.path.join(src, "manifest.json")
    if os.path.exists(mf):
        manifest = json.load(open(mf))
    else:  # partial output (UI testing only)
        manifest = {"n": len([d for d in os.listdir(os.path.join(src, "st")) if d.isdigit()]), "steps": 250,
                    "grid_res": 32, "seed": 0, "sheet": {"tile": 32, "cols": 16},
                    "samplers": {"st": {"stepsize": 0.031}}, "partial": True}
    manifest["samplers"] = {k: v for k, v in manifest["samplers"].items() if k in samplers}
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    N = manifest["grid_res"]
    for sampler in manifest["samplers"]:
        os.makedirs(os.path.join(dst, sampler))
        shutil.copy(os.path.join(src, sampler, "final.png"), os.path.join(dst, sampler))
        for j in range(manifest["n"]):
            sd, dd = os.path.join(src, sampler, f"{j:03d}"), os.path.join(dst, sampler, f"{j:03d}")
            os.makedirs(dd)
            meta = json.load(open(os.path.join(sd, "meta.json")))
            grid = np.fromfile(os.path.join(sd, "grid.bin"), dtype=np.float32).reshape(N, N, N)
            lo, hi = float(grid.min()), float(grid.max())
            np.round((grid - lo) / (hi - lo) * 65535).astype("<u2").tofile(os.path.join(dd, "grid.bin"))
            meta["grid_q"] = [lo, hi]
            meta["coords"] = [[round(v, 3) for v in c] for c in meta["coords"]]
            meta["u"] = [round(v, 3) for v in meta["u"]]
            meta["sheet"] = manifest["sheet"]
            json.dump(meta, open(os.path.join(dd, "meta.json"), "w"), separators=(",", ":"))
            Image.open(os.path.join(sd, "sheet.png")).convert("RGB").save(
                os.path.join(dd, "sheet.webp"), "WEBP", quality=quality, method=6)
    if os.path.isdir(os.path.join(src, "fm")):
        os.makedirs(os.path.join(dst, "fm"))
        for j in range(manifest["n"]):
            Image.open(os.path.join(src, "fm", f"{j:03d}_sheet.png")).convert("RGB").save(
                os.path.join(dst, "fm", f"{j:03d}_sheet.webp"), "WEBP", quality=quality, method=6)
        rep = os.path.join(src, "fm_report.json")
        manifest["fm"] = {"ckpt": "output_dir/20251113_072621baseline/checkpoint-1899.pth", "solver": "dopri5",
                          "atol": 1e-5, "rtol": 1e-5, "report": json.load(open(rep)) if os.path.exists(rep) else None}
    json.dump(manifest, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    print("packaged", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], tuple(sys.argv[3:]) or ("st", "gd"))
