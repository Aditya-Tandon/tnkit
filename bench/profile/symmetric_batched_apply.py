"""Batched recoupling on one device: warm permute and H_eff theta times, the batched products
against a loop over groups, peak memory and index bytes. Standalone: numpy, torch and tnkit.

usage: python symmetric_batched_apply.py DEVICE [OUT.json]
DEVICE is `numpy` (numpy backend) or a torch device (`cpu`, `cuda`, `cuda:1`); complex128.
Bonds V:
  su2   SU(2), 13 spins 2j = 0..12, degeneracies peaked at 2j = 4 (dense bond about 1000);
        physical leg spin 1/2, MPO bond {0: 2, spin 1: 1}.
  u1su2 U(1) x SU(2), charges -3..3 times 2j = 0..5 (42 sectors), degeneracies peaked at
        charge 0, 2j = 1; physical leg {(0, 0), (1, 1), (2, 0)} (a spinful fermion site), MPO
        bond {(0, 0): 2, (1, 1): 1, (-1, 1): 1, (0, 2): 1}.
Tensors: theta (V, P <- P, V), L (V <- V, W), the MPO tensor (W, P <- P, W), R (V, W <- V).
Printed per bond: warm permute of theta (three permutations) and of the intermediate
(V, W <- P, P, V); warm H_eff theta (three contractions); for the permutation with the most
groups, the batched apply against one product per group (same device arrays, results compared);
peak memory over one H_eff and one apply, after the timings (CUDA allocator; tracemalloc for
numpy; not measured for torch on the CPU); index bytes and
their ratio to the bytes of the buffers they index; cache sizes.
"""
import json
import statistics
import sys
import time
import tracemalloc

import numpy as np

from tnkit import symmetric_tensor as st
from tnkit.backend import NumpyBackend
from tnkit.symmetric_tensor import Space, contract, permute, random
from tnkit.symmetry import SU2, U1, ProductSymmetry


def backend(device):
    if device == "numpy":
        return NumpyBackend(), None
    import torch
    from tnkit.backend import TorchBackend
    return TorchBackend(device, torch.complex128), torch


def space(sym, degs):
    items = sorted(degs.items(), key=lambda kv: sym.sort_key(kv[0]))
    return Space(sym, [a for a, _ in items], [n for _, n in items])


def bonds():
    s = SU2()
    yield "su2", s, (
        space(s, {j: max(1, round(40 * np.exp(-((j - 4) / 3.2) ** 2))) for j in range(13)}),
        space(s, {1: 1}), space(s, {0: 2, 2: 1}))
    u = ProductSymmetry((U1(), SU2()))
    yield "u1su2", u, (
        space(u, {(q, j): max(1, round(12 * np.exp(-q * q / 3 - (j - 1) ** 2 / 3)))
                  for q in range(-3, 4) for j in range(6)}),
        space(u, {(0, 0): 1, (1, 1): 1, (2, 0): 1}),
        space(u, {(0, 0): 2, (1, 1): 1, (-1, 1): 1, (0, 2): 1}))


def clock(f, sync, reps):
    f()
    sync()
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        sync()
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


def looped(op, data):
    """The recoupler's apply with one product per group in place of one per shape."""
    x = data[op.src]
    if op.scale is not None:
        x[:op.n_scalar] = x[:op.n_scalar] * op.scale
    for o, size, (k, p, d), M in op.stacks:
        for g in range(k):
            a = o + g * p * d
            x[a:a + p * d] = (M[g] @ x[a:a + p * d].reshape(p, d)).reshape(-1)
    return x[op.inv]


def run(device, name, sym, legs, rng, reps):
    be, torch = backend(device)
    cuda = torch is not None and torch.device(device).type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    for c in (st.RECOUPLING_CACHE, st.INDEX_CACHE, st.STRUCTURE_CACHE, st.RECENT_LAYOUTS):
        c.clear()
    V, P, W = legs
    theta = random((V, P), (P, V), be, rng)
    L, M, R = (random((V,), (V, W), be, rng), random((W, P), (P, W), be, rng),
               random((V, W), (V,), be, rng))
    x = random((V, W), (P, P, V), be, rng)
    out = {"sectors": len(V.sectors), "dense_bond": V.dim, "theta_stored": theta.layout.size}
    perms = [(theta, "theta_bend", ((0,), (1, 2, 3))),
             (theta, "theta_bend_back", ((0, 1, 2), (3,))),
             (theta, "theta_swap_sides", ((1, 3), (0, 2))), (x, "x_bend", ((0, 2), (1, 3, 4)))]
    for t, label, p in perms:
        out[f"permute_{label}_ms"] = 1e3 * clock(lambda: permute(t, *p), sync, reps)

    def heff():
        y = contract(L, theta, [1], [0])
        y = contract(y, M, [1, 2], [0, 2])
        return contract(y, R, [2, 4], [0, 1])

    out["heff_ms"] = 1e3 * clock(heff, sync, reps)
    ops = [(op, t) for t, _, p in perms
           for op in [t.layout.map(be, ("permute",) + p, None)]]
    op, t = max(ops, key=lambda ot: sum(s[2][0] for s in ot[0].stacks))
    out["groups"] = sum(s[2][0] for s in op.stacks)
    out["shapes"] = len(op.stacks)
    out["batched_ms"] = 1e3 * clock(lambda: op.apply(t.data), sync, reps)
    out["per_group_ms"] = 1e3 * clock(lambda: looped(op, t.data), sync, reps)
    diff = op.apply(t.data) - looped(op, t.data)
    out["batched_vs_loop_max_abs"] = float(abs(diff).max()) if len(diff) else 0.0
    if cuda:                                   # peak over one H_eff and one apply, timed apart
        torch.cuda.reset_peak_memory_stats()
    elif torch is None:
        tracemalloc.start()
    heff()
    op.apply(t.data)
    sync()
    if cuda:
        out["peak_device_bytes"] = int(torch.cuda.max_memory_allocated())
    elif torch is None:
        out["peak_host_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    entry = 16 if "complex128" in str(be.dtype) else 8
    maps = st.INDEX_CACHE.values()
    index = st.INDEX_CACHE.bytes
    indexed = sum(entry * (m.src.shape[0] if m.inv is None else m.inv.shape[0])
                  for m in maps if hasattr(m, "stacks"))
    out["index_and_matrix_bytes"] = index
    out["index_bytes_per_buffer_byte"] = index / indexed if indexed else 0.0
    out["caches"] = {k: {f: v[f] for f in ("entries", "bytes") if f in v}
                     for k, v in st.cache_info().items()}
    return out


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "numpy"
    rng = np.random.default_rng(0)
    res = {"device": device}
    for name, sym, legs in bonds():
        res[name] = run(device, name, sym, legs, rng, reps=50)
        r = res[name]
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                     if k != "caches"})
        print("  caches", {k: (v.get("entries"), v.get("bytes")) for k, v in r["caches"].items()})
    if len(sys.argv) > 2:
        json.dump(res, open(sys.argv[2], "w"), indent=1)
