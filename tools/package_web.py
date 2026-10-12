"""
Package the output of make_trajectories.py for the web page:
quantize each u_theta grid to uint16 (range stored in meta.json as grid_q),
compress thumbnails, and add class names to the manifest.

Usage: python3 tools/package_web.py <job out dir> static/traj
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

# ImageNet class names of the curated samples (short form).
CLASS_NAMES = {
    0: "tench", 30: "bullfrog", 31: "tree frog", 32: "tailed frog", 84: "peacock", 105: "koala",
    261: "keeshond", 291: "lion", 324: "cabbage butterfly", 378: "capuchin", 518: "crash helmet",
    548: "entertainment center", 812: "space shuttle", 853: "thatched roof", 873: "triumphal arch",
    933: "cheeseburger", 935: "mashed potato", 962: "meat loaf", 970: "alp", 972: "cliff",
    975: "lakeside", 978: "seashore",
}


def main(src, dst):
    manifest = json.load(open(os.path.join(src, "manifest.json")))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    N = manifest["grid_res"]
    for sampler in manifest["samplers"]:
        os.makedirs(os.path.join(dst, sampler))
        for idx in manifest["indices"]:
            name = f"{idx:06d}"
            Image.open(os.path.join(src, sampler, f"{name}.png")).convert("RGB").resize(
                (192, 192), Image.BICUBIC).save(os.path.join(dst, sampler, f"{name}.jpg"), quality=88)
            sd, dd = os.path.join(src, sampler, name), os.path.join(dst, sampler, name)
            os.makedirs(dd)
            meta = json.load(open(os.path.join(sd, "meta.json")))
            grid = np.fromfile(os.path.join(sd, "grid.bin"), dtype=np.float32).reshape(N, N, N)
            lo, hi = float(grid.min()), float(grid.max())
            q = np.round((grid - lo) / (hi - lo) * 65535).astype("<u2")
            q.tofile(os.path.join(dd, "grid.bin"))
            meta["grid_q"] = [lo, hi]
            meta["coords"] = [[round(v, 3) for v in c] for c in meta["coords"]]
            meta["u"] = [round(v, 3) for v in meta["u"]]
            json.dump(meta, open(os.path.join(dd, "meta.json"), "w"), separators=(",", ":"))
            for s in manifest["thumb_steps"]:
                shutil.copy(os.path.join(sd, f"step_{s:03d}.jpg"), dd)
    manifest["class_names"] = {str(l): CLASS_NAMES.get(l, f"class {l}") for l in manifest["labels"]}
    json.dump(manifest, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    missing = [l for l in manifest["labels"] if l not in CLASS_NAMES]
    print("packaged", dst, "labels without names:", missing)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
