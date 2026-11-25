"""Memory and time per real-time step of a purified Schwinger chain, block tensors against dense.

usage: python purified_blocks.py [--backend numpy|torch-cuda] [--N 16 20 24] [--chi 64 128 256]
                                 [--cache-mb MB] [--out OUT.json]

Model: x = 4, m/g = 0.05 (mu = 0.2), T/g = 0.5 (beta = 0.5 in units of W = 2H/(a g^2)). The
thermal state is cooled in U(1) x U(1) blocks (dbeta 0.01, svd_tol 1e-10). A neutral Gaussian
pair-creation packet (width 1.5/g, momentum 0.6 g, centred) is applied at svd_tol 1e-8, and the
chain is evolved in real time with backward twins (g dt = 0.1, svd_tol 1e-8) until its largest
bond reaches each cap in turn. At each cap the same state is timed for one step on blocks and,
converted, on dense tensors: the first step (environments, contraction cache), then a steady one.
Peak memory of one more step is the torch allocator peak on CUDA, or the tracemalloc peak on
numpy. Also recorded: the sectors on every bond and their sizes, and the block contraction cache.
Standalone: numpy, optionally torch, and tnkit (no pytest).
"""
import argparse
import json
import time
import tracemalloc

import numpy as np

from tnkit.backend import get_backend
from tnkit.block_backend import BlockBackend, apply_block_mpo, to_dense_mps
from tnkit.blocks import cache_info, set_cache_limit
from tnkit.models_lgt import schwinger_charges, schwinger_mpo
from tnkit.opsum import OpSum, compile_mpo, spin_half
from tnkit.tdvp import TDVPOptions
from tnkit.tdvp_evolver import TDVPEvolver
from tnkit.thermal import purified_charges, purify_mpo, thermal_state

X, MU, BETA, DBETA = 4.0, 0.2, 0.5, 0.01
GA = 1 / np.sqrt(X)                     # g a
DT = 0.1 * GA / 2                       # g dt = 0.1 in units of W


def packet(N, width=1.5, p=0.6):
    """sum_n f(z_n) h+_n, h+_n = s+_n s-_{n+1} (n even) or s-_n s+_{n+1} (n odd): it creates a
    pair on the staggered vacuum. z_n is the link centre in units of 1/g."""
    os = OpSum()
    for n in range(N - 1):
        z, c = (n + 0.5) * GA, (N - 1) / 2 * GA
        f = np.exp(-0.5 * ((z - c) / width) ** 2 + 1j * p * z)
        os += (f, "Sp", n, "Sm", n + 1) if n % 2 == 0 else (f, "Sm", n, "Sp", n + 1)
    return compile_mpo(os, [spin_half()] * N)[0]


def sectors(mps):
    """Per bond: the sector labels and their sizes."""
    return [{"charges": [list(c) for c in t.legs[2].charges], "dims": list(t.legs[2].dims)}
            for t in mps[:-1]]


class Meter:
    def __init__(self, be):
        self.cuda = getattr(be, "device", None) is not None and "cuda" in str(be.device)
        self.be = be

    def sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    def peak(self, fn):
        if self.cuda:
            import torch
            self.sync()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn()
            self.sync()
            return torch.cuda.max_memory_allocated() - base
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak


def time_steps(mps, W, cap, be, meter):
    ev = TDVPEvolver(mps, W, DT, cap, TDVPOptions(backend=be, svd_tol=1e-8, graph="off"))
    meter.sync()
    t0 = time.perf_counter()
    ev.run(1, "2site")
    meter.sync()
    t1 = time.perf_counter()
    ev.run(1, "2site")
    meter.sync()
    t2 = time.perf_counter()
    peak = meter.peak(lambda: ev.run(1, "2site"))
    out = {"first_s": t1 - t0, "steady_s": t2 - t1, "peak_bytes": int(peak),
           "chi_after": max(int(t.shape[2]) for t in ev.mps)}
    if isinstance(be, BlockBackend):
        info = cache_info()
        out["cache"] = {k: info[k] for k in ("bytes", "entries", "build_s", "misses")}
    return out


def run(N, chis, backend, max_steps=400):
    dense_be = get_backend(backend)
    bb = BlockBackend(dense_be)
    meter = Meter(dense_be)
    H, q = schwinger_mpo(N, X, MU), schwinger_charges(N)
    t = time.perf_counter()
    rho, log_z = thermal_state(H, BETA, 512, DBETA, charges=q, blocks=True, svd_tol=1e-10,
                               backend=bb)
    out = {"N": N, "cool_s": time.perf_counter() - t, "log_z": log_z,
           "chi_thermal": max(int(t.shape[2]) for t in rho), "caps": []}
    psi, _, _ = apply_block_mpo(list(rho), purify_mpo(packet(N), charges=q), max(chis),
                                svd_tol=1e-8)
    out["chi_packet"] = max(int(t.shape[2]) for t in psi)
    Wb, Wd = purify_mpo(H, twin=True, charges=q), purify_mpo(H, twin=True)
    pq = purified_charges(q)
    steps = 0
    for cap in chis:
        ev = TDVPEvolver(psi, Wb, DT, cap, TDVPOptions(backend=bb, svd_tol=1e-8, graph="off"))
        t = time.perf_counter()
        while True:                                     # at least one step: cut to the cap
            ev.run(1, "2site")
            steps += 1
            if max(int(x.shape[2]) for x in ev.mps) >= cap or steps >= max_steps:
                break
        psi = list(ev.mps)
        chi = max(int(x.shape[2]) for x in psi)
        rec = {"cap": cap, "chi": chi, "g_t": steps * 0.1, "grow_s": time.perf_counter() - t,
               "sectors": sectors(psi),
               "stored_block": int(sum(x.struct.size for x in psi)),
               "stored_dense": int(sum(int(np.prod(x.shape)) for x in psi))}
        rec["block"] = time_steps(psi, Wb, cap, bb, meter)
        dense = [dense_be.array(x) for x in to_dense_mps(psi, pq)]
        rec["dense"] = time_steps(dense, Wd, cap, dense_be, meter)
        out["caps"].append(rec)
        b, d = rec["block"], rec["dense"]
        print(f"N {N} cap {cap} chi {chi} g t {rec['g_t']:.1f}: steady step "
              f"{d['steady_s']:.2f} s dense, {b['steady_s']:.2f} s block "
              f"({d['steady_s'] / b['steady_s']:.2f}x); peak {d['peak_bytes'] / 1e6:.0f} MB "
              f"dense, {b['peak_bytes'] / 1e6:.0f} MB block "
              f"({d['peak_bytes'] / max(b['peak_bytes'], 1):.2f}x)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--N", type=int, nargs="+", default=[16, 20, 24])
    ap.add_argument("--chi", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--cache-mb", type=float, default=None, help="block contraction cache limit")
    ap.add_argument("--out", default="purified_blocks.json")
    a = ap.parse_args()
    if a.cache_mb is not None:
        set_cache_limit(int(a.cache_mb * 2 ** 20))
    res = {"backend": a.backend, "cache_mb": a.cache_mb, "runs": [run(N, sorted(a.chi), a.backend) for N in a.N]}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
