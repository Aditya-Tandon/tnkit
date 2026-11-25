"""Layout, recoupling and contraction times of SU(2) symmetric tensors on a production-sized bond.

usage: python symmetric_su2_timing.py OUT.json
Bonds (all 2j present from 0, each with the degeneracy given):
  s30  30 spins, 2j = 0..29, degeneracy 1    (dense bond 465)
  s40  40 spins, 2j = 0..39, degeneracy 1    (dense bond 820)
  s13  13 spins, 2j = 0..12, degeneracies peaked at 2j = 4 (dense bond about 1000)
Physical leg spin 1/2; MPO bond spins {0: 2, 1 (spin 1): 1}, dense size 5. Tensors: the two-site
tensor theta (V, P <- P, V), the left environment L (V <- V, W), the MPO tensor (W, P <- P, W) and
the right environment R (V, W <- V). Times, numpy backend, complex128:
  layout     building the layout of theta: cold (no cached trees) and with new degeneracies only;
  permute    theta to (V <- P, P, V), (V, P, P <- V) and (P, V <- V, P): first call (index build,
             with and without cached recoupling matrices) and the median of later calls;
  heff       L theta W R, the two-site effective Hamiltonian applied once (three contractions);
  fuse       theta's (V, P) legs fused.
Also the cache sizes and the stored and dense sizes of theta.
"""
import json
import statistics
import sys
import time

import numpy as np

from tnkit import symmetric_tensor as st
from tnkit.backend import NumpyBackend
from tnkit.symmetric_tensor import Space, contract, fuse, permute, random, split
from tnkit.symmetry import SU2

S, BE = SU2(), NumpyBackend()


def bond(name):
    if name == "s30":
        return Space(S, tuple(range(30)), (1,) * 30)
    if name == "s40":
        return Space(S, tuple(range(40)), (1,) * 40)
    degs = [max(1, round(40 * np.exp(-((j - 4) / 3.2) ** 2))) for j in range(13)]
    return Space(S, tuple(range(13)), degs)


def clock(f, reps=5):
    t0 = time.perf_counter()
    out = f()
    first = time.perf_counter() - t0
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        runs.append(time.perf_counter() - t0)
    return out, first, statistics.median(runs)


def clear():
    for c in (st.RECOUPLING_CACHE, st.INDEX_CACHE, st.STRUCTURE_CACHE, st.RECENT_LAYOUTS):
        c.clear()


def run(name, rng):
    V, P, W = bond(name), Space(S, (1,), (1,)), Space(S, (0, 2), (2, 1))
    out = {"sectors": len(V.sectors), "dense_bond": V.dim}
    clear()
    t0 = time.perf_counter()
    theta = random((V, P), (P, V), BE, rng)
    out["layout_cold_s"] = time.perf_counter() - t0
    V2 = Space(S, V.sectors, tuple(n + 1 for n in V.degeneracies))
    t0 = time.perf_counter()
    random((V2, P), (P, V2), BE, rng)
    out["layout_new_degeneracies_s"] = time.perf_counter() - t0
    out["theta_stored"], out["theta_dense"] = theta.layout.size, int(np.prod(theta.shape))
    for label, p in [("bend", ((0,), (1, 2, 3))), ("bend_back", ((0, 1, 2), (3,))),
                     ("swap_sides", ((1, 3), (0, 2)))]:
        u, first, warm = clock(lambda: permute(theta, *p))
        out[f"permute_{label}_first_s"], out[f"permute_{label}_s"] = first, warm
        st.INDEX_CACHE.clear()
        t0 = time.perf_counter()
        permute(theta, *p)
        out[f"permute_{label}_rebuilt_indices_cached_matrices_s"] = time.perf_counter() - t0
    L = random((V,), (V, W), BE, rng)
    M = random((W, P), (P, W), BE, rng)
    R = random((V, W), (V,), BE, rng)

    def heff():
        x = contract(L, theta, [1], [0])           # legs V, W | P, P, V
        y = contract(x, M, [1, 2], [0, 2])         # legs V, P, V | P, W
        return contract(y, R, [2, 4], [0, 1])      # legs V, P, P | V

    y, first, warm = clock(heff)
    out["heff_first_s"], out["heff_s"] = first, warm
    out["heff_result_shape"] = list(y.shape)
    f, first, warm = clock(lambda: fuse(theta, [0, 1]))
    out["fuse_first_s"], out["fuse_s"] = first, warm
    assert np.allclose(split(f, 0, (V, P)).data, theta.data)
    out["caches"] = st.cache_info()
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    res = {name: run(name, rng) for name in ("s30", "s40", "s13")}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for name, r in res.items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                     if k != "caches"})
        print("  caches", {k: (v["entries"], v.get("bytes")) for k, v in r["caches"].items()})
