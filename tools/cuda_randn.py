"""
Reproduce torch.randn(shape, device='cuda') (float32, right after torch.manual_seed)
for a GPU model other than the one we run on.

PyTorch fills a float normal tensor with distribution_nullary_kernel: block size
256, grid = min(ceil(numel / 256), SMs * maxThreadsPerSM / 256), and every thread
`idx` (of T = 256 * grid) calls curand_init(seed, idx, offset) and then
curand_normal4, writing its i-th normal (i = 0..3) to element idx + i * T (and
repeating with a further call if numel > 4 T). The values therefore depend on T,
i.e. on the GPU model: the paper's CIFAR-10 batch was drawn on an A10 (72 SMs x 1536
threads), and an A100 or a GTX 1080 Ti reproduce only its first T elements.

Here Philox4x32-10 is evaluated exactly with integers and the Box-Muller step in
float32 on the GPU (curand uses __sincosf, so values can differ in the last bits).
`python3 cuda_randn.py --out x0.npy` checks the emulation against torch.randn on
the local GPU, then writes the A10 tensor.
"""
import argparse
import json

import numpy as np
import torch

M0, M1, W0, W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
MASK = 0xFFFFFFFF


def philox(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on uint64 arrays holding 32-bit values."""
    for r in range(10):
        if r:
            k0, k1 = (k0 + W0) & MASK, (k1 + W1) & MASK
        p0 = c0 * M0
        p1 = c2 * M1
        hi0, lo0 = p0 >> 32, p0 & MASK
        hi1, lo1 = p1 >> 32, p1 & MASK
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def emulate(shape, seed, threads, device="cuda"):
    numel = int(np.prod(shape))
    calls = (numel - 1) // (4 * threads) + 1
    idx = np.arange(threads, dtype=np.uint64)
    out = np.empty(calls * 4 * threads, dtype=np.float32)
    k0, k1 = np.uint64(seed & MASK), np.uint64((seed >> 32) & MASK)
    inv = torch.tensor(2.3283064e-10, dtype=torch.float32, device=device)
    inv2pi = torch.tensor(2.3283064e-10 * 6.2831855, dtype=torch.float32, device=device)
    for c in range(calls):
        # curand_init(seed, subsequence=idx, offset=0): counter (0, 0, idx, 0); call c uses counter c
        z = np.zeros_like(idx)
        x = philox(z + np.uint64(c), z, idx & MASK, idx >> 32, z + k0, z + k1)
        xs = [torch.from_numpy(v.astype(np.float64)).to(device, torch.float32) for v in x]  # uint32 -> float (as in C)

        def box_muller(a, b):
            u = torch.addcmul(inv / 2, a, inv)
            v = torch.addcmul(inv2pi / 2, b, inv2pi)
            s = torch.sqrt(-2.0 * torch.log(u))
            return torch.sin(v) * s, torch.cos(v) * s

        n0, n1 = box_muller(xs[0], xs[1])
        n2, n3 = box_muller(xs[2], xs[3])
        for i, n in enumerate([n0, n1, n2, n3]):
            out[(4 * c + i) * threads:(4 * c + i + 1) * threads] = n.cpu().numpy()
    return out[:numel].reshape(shape)


def threads_of(props):
    grid_cap = props.multi_processor_count * (props.max_threads_per_multi_processor // 256)
    return grid_cap * 256


def main(args):
    shape = (args.n, 3, 32, 32)
    numel = int(np.prod(shape))
    props = torch.cuda.get_device_properties(0)
    local_t = min(threads_of(props), -(-numel // 256) * 256)
    torch.manual_seed(args.seed)
    ref = torch.randn(shape, dtype=torch.float32, device="cuda").cpu().numpy()
    emu = emulate(shape, args.seed, local_t)
    d = np.abs(emu - ref)
    report = {"gpu": props.name, "local_threads": local_t, "local_max_abs_diff": float(d.max()),
              "local_exact_fraction": float((d == 0).mean())}
    a10_t = min(72 * (1536 // 256) * 256, -(-numel // 256) * 256)
    x0 = emulate(shape, args.seed, a10_t)
    report["a10_threads"] = a10_t
    report["a10_vs_local_equal_prefix"] = int(np.argmax(np.abs(x0.ravel() - ref.ravel()) > 1e-4))
    if local_t == a10_t:  # same thread count as the A10: the real values are exact
        x0 = ref
        report["used"] = "torch.randn on this GPU"
    else:
        report["used"] = "emulation"
    np.save(args.out, x0)
    print(json.dumps(report), flush=True)
    with open(args.out + ".json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=128)
    main(p.parse_args())
