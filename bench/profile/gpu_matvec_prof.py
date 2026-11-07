"""Two-site matvec timing on one GPU: tnkit's apply_two_site against a fused, transpose-free
variant, on the Heisenberg-cylinder MPO (w = 3 Ly + 2) and on the XXZ chain MPO (w = 5).

usage: python gpu_matvec_prof.py OUT.json
Needs torch with CUDA and tnkit importable. No pytest. Runs in a few minutes on a MIG 1g slice.
"""
import json, sys, time

import numpy as np
import torch

from tnkit._dmrg import apply_two_site
from tnkit.backend import get_backend


def cyl_mpo_site(Ly):
    """Bulk W of H = sum (S+S- + S-S+)/2 + SzSz over the bonds of an Ly-periodic cylinder on the
    column-major snake: channel 0 idle, (o, k) operator o placed k sites ago, last = done."""
    SP = np.array([[0, 1], [0, 0]], float); SM = SP.T.copy(); SZ = np.diag([0.5, -0.5]); I2 = np.eye(2)
    R = Ly
    ops = [(SP, 0.5 * SM), (SM, 0.5 * SP), (SZ, SZ)]
    D = 2 + 3 * R
    ch = lambda o, k: 1 + o * R + (k - 1)
    W = np.zeros((D, D, 2, 2))
    W[0, 0] = W[-1, -1] = I2
    for o, (a, b) in enumerate(ops):
        W[0, ch(o, 1)] = a
        for k in range(1, R + 1):
            if k < R:
                W[ch(o, k), ch(o, k + 1)] = I2
        W[ch(o, 1), D - 1] = b            # vertical-neighbour (k = 1) and
        W[ch(o, R), D - 1] = b            # horizontal-neighbour (k = Ly) couplings
        if R > 1:
            W[ch(o, R - 1), D - 1] = b    # periodic wrap (k = Ly - 1)
    return W


def xxz_site():
    SP = np.array([[0, 1], [0, 0]], float); SM = SP.T.copy(); SZ = np.diag([0.5, -0.5]); I2 = np.eye(2)
    W = np.zeros((5, 5, 2, 2))
    W[0, 0], W[0, 1], W[0, 2], W[0, 3] = I2, SP, SM, SZ
    W[1, 4], W[2, 4], W[3, 4], W[4, 4] = 0.5 * SM, 0.5 * SP, SZ, I2
    return W


def fused_op(W1, W2):
    wl, wr, d = W1.shape[0], W2.shape[1], W1.shape[2]
    return torch.einsum("wmsi,mrtj->strwij", W1, W2).reshape(d * d * wr, wl * d * d).contiguous()


def apply_fused(theta, L, S, R):
    p, wl, q = L.shape
    _, s, t, b = theta.shape
    dr, wr = R.shape[0], R.shape[1]
    x1 = (L.reshape(p * wl, q) @ theta.reshape(q, s * t * b)).reshape(p, wl * s * t, b)
    x3 = S @ x1
    return (x3.reshape(p * s * t, wr * b) @ R.reshape(dr, wr * b).T).reshape(p, s, t, dr)


def timeit(f, reps):
    f(); torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(reps):
        y = f()
    ev1.record(); torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps, y


def main(out_path):
    be = get_backend("torch-cuda-real")
    dev = torch.device("cuda")
    a = torch.randn(4096, 4096, dtype=torch.float64, device=dev)
    ms, _ = timeit(lambda: a @ a, 10)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "dgemm4096_TFs": 2 * 4096 ** 3 / ms / 1e9, "rows": []}
    for name, W in (("cyl_Ly6", cyl_mpo_site(6)), ("chain_xxz", xxz_site())):
        Wt = torch.as_tensor(W, device=dev)
        w = W.shape[0]
        S = fused_op(Wt, Wt)
        for chi in (256, 512, 1024):
            if chi ** 2 * w * 4 * 8 * 4 > 0.6 * torch.cuda.get_device_properties(0).total_memory:
                continue
            g = torch.Generator(device=dev).manual_seed(0)
            L = torch.randn(chi, w, chi, dtype=torch.float64, device=dev, generator=g)
            R = torch.randn(chi, w, chi, dtype=torch.float64, device=dev, generator=g)
            th = torch.randn(chi, 2, 2, chi, dtype=torch.float64, device=dev, generator=g)
            reps = 20 if chi <= 512 else 5
            t0, y0 = timeit(lambda: apply_two_site(th, L, Wt, Wt, R, be), reps)
            t1, y1 = timeit(lambda: apply_fused(th, L, S, R), reps)
            err = float((y1 - y0).abs().max() / y0.abs().max())
            flops = 2 * 2 * chi ** 3 * w * 4 + 2 * 2 * chi ** 2 * w * w * 8
            res["rows"].append({"mpo": name, "w": w, "chi": chi, "ms_apply_two_site": t0,
                                "ms_fused": t1, "speedup": t0 / t1, "rel_err": err,
                                "TFs_apply_two_site": flops / t0 / 1e9})
            print(res["rows"][-1], flush=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
