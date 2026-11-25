"""Recoupling maps built on a GPU against the same maps built on the host.

usage: TNKIT_CHECK=debug python bench/gpu_check_index_maps.py [device]     (default cuda)

Standalone (numpy, torch and tnkit from this checkout; no pytest). For U(1), SU(2), U(1) x SU(2)
and A4 x SU(2):
  * every permute of small rank-3 and rank-4 layouts, and some permutes of large rank-4 ones
    (5e5 to 2e7 entries; U(1) past 2^24, where float32 stops being exact): src, inv and scale from
    the device path equal those from the host path, bit for bit;
  * permute and contract on the device give the same bits with either map, and permute agrees
    with numpy (bit for bit on U(1), where it is a gather).
With TNKIT_CHECK=debug the library also checks that every src and inv is a permutation. Prints
one line per failure and a last line `IDXMAP_<DEVICE> PASS` or `IDXMAP_<DEVICE> FAIL`.
"""
import itertools
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

import numpy as np
import torch

import tnkit
from tnkit import _recoupling as rc
from tnkit import _symmetric_layout as sl
from tnkit.backend import NumpyBackend, TorchBackend
from tnkit.symmetric_tensor import Space, contract, permute, random, zeros
from tnkit.symmetry import SU2, U1, ProductSymmetry
from oracles.a4 import A4

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
SYMS = {"U1": U1(), "SU2": SU2(), "U1xSU2": ProductSymmetry((U1(), SU2())),
        "A4xSU2": ProductSymmetry((A4(), SU2()))}
SMALL = {                    # three legs per symmetry, {sector: degeneracy}
    "U1": [{-1: 2, 0: 3, 1: 2}, {-2: 3, 1: 4}, {-1: 5, 1: 4}],
    "SU2": [{0: 2, 1: 3}, {1: 3, 2: 3}, {0: 4, 1: 4, 2: 4}],
    "U1xSU2": [{(0, 0): 2, (1, 1): 2}, {(-1, 1): 3, (0, 0): 3, (0, 2): 3}, {(1, 1): 4, (2, 0): 4}],
    "A4xSU2": [{(0, 0): 2, (3, 1): 2}, {(1, 1): 3, (3, 0): 3}, {(0, 0): 4, (2, 1): 4}]}
LARGE = {                    # four legs per symmetry
    "U1": [{-1: 40, 0: 40, 1: 40}, {-1: 30, 0: 50, 1: 36}, {-2: 18, 0: 60, 2: 22},
           {-1: 34, 1: 44}],
    "SU2": [{0: 12, 1: 20, 2: 9}, {1: 25, 3: 6}, {0: 14, 2: 16}, {1: 18, 3: 7}],
    "U1xSU2": [{(0, 0): 30, (1, 1): 24}, {(-1, 1): 27, (0, 0): 36, (0, 2): 15},
               {(1, 1): 30, (0, 0): 33}, {(0, 0): 27, (-1, 1): 24, (0, 2): 18}],
    "A4xSU2": [{(0, 0): 24, (3, 1): 18}, {(1, 1): 21, (3, 0): 18}, {(0, 0): 24, (3, 1): 15},
               {(3, 0): 21, (2, 1): 18}]}
failures = []


def fail(what):
    failures.append(what)
    print("FAIL", what, flush=True)


def spaces(sym, legs):
    return [Space(sym, *zip(*sorted(leg.items(), key=lambda kv: sym.sort_key(kv[0]))))
            for leg in legs]


def splits(r):
    return [(p[:k], p[k:]) for p in itertools.permutations(range(r)) for k in range(r + 1)]


def small_shapes(A, B, C):
    a, b, c = A.dual(), B.dual(), C.dual()
    return [((A, B), (C,)), ((a,), (B, c)), ((A, b, C), ()), ((), (a, B, c)), ((A, B), (c, A)),
            ((a, B, C), (b,)), ((A,), (B, c, a)), ((A, b, C, a), ()), ((), (A, B, c, a))]


def host(x):
    return None if x is None else np.asarray(x.cpu() if hasattr(x, "cpu") else x)


def same_maps(tag, a, b):
    for what, x, y in (("src", a.src, b.src), ("inv", a.inv, b.inv), ("scale", a.scale, b.scale)):
        x, y = host(x), host(y)
        if (x is None) != (y is None):
            return fail(f"{tag} {what}: present on one path only")
        if x is not None and (x.dtype != y.dtype or not np.array_equal(x, y)):
            return fail(f"{tag} {what}: {x.dtype} vs {y.dtype}, "
                        f"{int(np.sum(x != y)) if x.shape == y.shape else 'shapes differ'}")
    if [s[:3] for s in a.stacks] != [s[:3] for s in b.stacks]:
        fail(f"{tag} stacks differ")


def clear():
    sl.INDEX_CACHE.clear()
    sl.DEVICE_INDEX_CACHE.clear()


ON_DEVICE = rc._on_device


def results(t, u, perms, on_device):
    """permute(t) for each split and one contraction of t with u, with maps from one path."""
    rc._on_device = ON_DEVICE if on_device else (lambda be: False)
    clear()
    out = [permute(t, p1, p2).data.cpu().numpy() for p1, p2 in perms]
    out.append(contract(t, u, (2, 3), (0, 1)).data.cpu().numpy())
    rc._on_device = ON_DEVICE
    clear()
    return out


def main():
    t0 = time.time()
    if DEVICE == "cuda" and not torch.cuda.is_available():
        fail("no CUDA device")
        return
    dev, cpu = TorchBackend(DEVICE), NumpyBackend()
    print(f"tnkit {tnkit.__file__}, torch {torch.__version__}, numpy {np.__version__}, "
          f"device {torch.cuda.get_device_name() if DEVICE == 'cuda' else DEVICE}, "
          f"TNKIT_CHECK={os.environ.get('TNKIT_CHECK', 'entry')}", flush=True)
    for name, sym in SYMS.items():                      # small layouts, every permute
        n_maps = 0
        for cod, dom in small_shapes(*spaces(sym, SMALL[name])):
            lay = zeros(cod, dom, cpu, sym).layout
            for p1, p2 in splits(len(cod) + len(dom)):
                same_maps(f"{name} small {lay.size} {p1}{p2}", rc.Recoupler(lay, p1, p2, cpu),
                          rc.Recoupler(lay, p1, p2, dev))
                n_maps += 1
        print(f"{name}: {n_maps} small maps compared ({time.time() - t0:.0f} s)", flush=True)
    dt = torch.complex64 if DEVICE == "mps" else torch.complex128      # A4 needs complex
    tol = 1e-5 if dt == torch.complex64 else 1e-12
    for name, sym in SYMS.items():                      # large layouts
        A, B, C, D = spaces(sym, LARGE[name])
        lay = zeros((A, B), (C, D), cpu, sym).layout
        perms = splits(4)[::17]
        maps = 0
        for p1, p2 in perms:
            same_maps(f"{name} large {lay.size} {p1}{p2}", rc.Recoupler(lay, p1, p2, cpu),
                      rc.Recoupler(lay, p1, p2, dev))
            maps += 1
        npdt = np.complex64 if dt == torch.complex64 else np.complex128
        t = random((A, B), (C, D), NumpyBackend(npdt), np.random.default_rng(0), sym=sym)
        u = random((C, D), (B,), NumpyBackend(npdt), np.random.default_rng(1), sym=sym)
        td, ud = (zeros(x.codomain, x.domain, TorchBackend(DEVICE, dtype=dt), sym)._like(
            torch.as_tensor(x.data, device=DEVICE)) for x in (t, u))
        a, b = results(td, ud, perms, True), results(td, ud, perms, False)
        for k, (x, y) in enumerate(zip(a, b)):
            if x.dtype != y.dtype or x.tobytes() != y.tobytes():
                fail(f"{name} large result {k}: device-built and host-built maps differ by "
                     f"{float(np.abs(x - y).max()) if x.shape == y.shape else 'shape'}")
        for (p1, p2), x in zip(perms, a):
            r = permute(t, p1, p2).data
            if name == "U1":
                if r.tobytes() != x.tobytes():
                    fail(f"{name} large permute {p1}{p2}: differs from numpy")
            elif not np.allclose(r, x, rtol=0, atol=tol * np.abs(r).max()):
                fail(f"{name} large permute {p1}{p2}: differs from numpy by "
                     f"{float(np.abs(r - x).max())}")
        print(f"{name}: large layout of {lay.size} entries, {maps} maps and {len(a)} results "
              f"compared ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                                  # report, then the one line
        import traceback
        traceback.print_exc()
        fail(f"exception: {type(e).__name__}: {e}")
    print(f"IDXMAP_{DEVICE.upper()} {'FAIL' if failures else 'PASS'}"
          + (f" ({len(failures)} failures)" if failures else ""), flush=True)
