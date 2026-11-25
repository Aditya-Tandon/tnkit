"""U(1) x SU(2) symmetric tensors on realistic bonds, against SU(2) alone, U(1) alone and dense.
Standalone: numpy, torch and tnkit.

usage: python symmetric_product_bonds.py DEVICE [OUT.json] [--dims=500,1000,2000] [--dense=500]
                                          [--reps=10]
DEVICE is `numpy` (numpy backend) or a torch device (`cpu`, `cuda`); complex128.

Sites hold two-colour fermions in spin language: sectors (charge, 2j) = (0, 0), (1, 1), (2, 0),
once each. A bond after n sites keeps the multiplets of n fused sites with weight
exp(-(q - n)^2 / (2 sigma^2) - j (j + 1) / tau), scaled to a target dense dimension; the right
bond of a two-site tensor is the one after n + 2 sites. The MPO bond is {(0, 0): 2, (1, 1): 1,
(-1, 1): 1, (0, 2): 1} (dense 9). SU(2) alone sums the degeneracies over charges, U(1) alone
over spins and multiplet states (the abelian `BlockTensor`).

Printed per target dimension:
  structure  cold layout times of tensors of ranks 3 to 5 (one to three bond legs per side);
  per representation: stored entries of theta (V, P, P <- V'), first and warm (median, with
             quartiles in the JSON) two-site H_eff theta (L, two MPO tensors, R), warm
             exchange of the two physical legs of theta, peak memory of one H_eff
             (tracemalloc, numpy only);
  caches     recoupling matrices built and reused, and per permute of the H_eff the sector
             groups, the matrix products they take and their distinct shapes.
Dense runs only up to --dense (default 500): its H_eff intermediates exceed laptop memory beyond.
"""
import json
import statistics
import sys
import time
import tracemalloc

import numpy as np

from tnkit import _symmetric_layout as sl
from tnkit import blocks
from tnkit import symmetric_tensor as st
from tnkit.backend import NumpyBackend
from tnkit.symmetric_tensor import Space, contract, fuse_spaces, permute, random
from tnkit.symmetry import SU2, U1, ProductSymmetry

U1SU2 = ProductSymmetry((U1(), SU2()))
SITE = {(0, 0): 1, (1, 1): 1, (2, 0): 1}
MPO = {(0, 0): 2, (1, 1): 1, (-1, 1): 1, (0, 2): 1}


def backend(device):
    if device == "numpy":
        return NumpyBackend(), None
    import torch
    from tnkit.backend import TorchBackend
    return TorchBackend(device, torch.complex128), torch


def space(sym, degs):
    items = sorted(degs.items(), key=lambda kv: sym.sort_key(kv[0]))
    return Space(sym, [a for a, _ in items], [n for _, n in items])


def multiplets(n):
    """{(q, 2j): count} of n fused sites."""
    P = space(U1SU2, SITE)
    F = P
    for _ in range(n - 1):
        F = fuse_spaces([F, P])
    return dict(zip(F.sectors, F.degeneracies))


def bond(n, target, sigma=3.5, tau=10.0):
    """{(q, 2j): degeneracy} of the bond after n sites, of dense dimension about `target`."""
    m = multiplets(n)
    w = {a: k * np.exp(-(a[0] - n) ** 2 / (2 * sigma ** 2) - a[1] * (a[1] + 2) / (4 * tau))
         for a, k in m.items()}
    lo, hi = 1e-6, 1e6
    for _ in range(100):
        s = (lo * hi) ** 0.5
        d = {a: min(m[a], int(round(s * x))) for a, x in w.items()}
        dim = sum(v * (a[1] + 1) for a, v in d.items())
        lo, hi = (s, hi) if dim < target else (lo, s)
    return {a: v for a, v in d.items() if v > 0}


def project(degs, keep):
    """Degeneracies of one factor: `keep` 0 sums over spins and multiplet states (U(1)), 1 over
    charges (SU(2))."""
    out = {}
    for (q, j), n in degs.items():
        k, v = (q, n * (j + 1)) if keep == 0 else (j, n)
        out[k] = out.get(k, 0) + v
    return out


def clock(f, sync, reps):
    t0 = time.perf_counter()
    f()
    sync()
    first = time.perf_counter() - t0
    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        sync()
        runs.append(time.perf_counter() - t0)
    q = statistics.quantiles(runs, n=4) if len(runs) > 1 else runs * 3
    return first, statistics.median(runs), (q[0], q[2])


def peak(f, torch):
    if torch is not None:
        return None
    tracemalloc.start()
    f()
    p = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return p


def clear():
    for c in (st.RECOUPLING_CACHE, st.INDEX_CACHE, st.STRUCTURE_CACHE, st.RECENT_LAYOUTS):
        c.clear()
    sl.fusions.cache_clear()


# --------------------------------------------------------------------------- representations
def symmetric(Vl, Vr, P, W, be, rng):
    theta = random((Vl, P, P), (Vr,), be, rng)
    L, M, R = (random((Vl,), (Vl, W), be, rng), random((W, P), (P, W), be, rng),
               random((Vr, W), (Vr,), be, rng))

    def heff():
        y = contract(L, theta, [1], [0])                  # (a', w; s1, s2, b)
        y = contract(y, M, [1, 2], [0, 2])                # (a', s2, b; s1', w1)
        y = contract(y, M, [1, 4], [2, 0])                # (a', b, s1'; s2', w2)
        return contract(y, R, [1, 4], [0, 1])             # (a', s1', s2'; b')

    return theta, heff, lambda: permute(theta, (0, 2, 1), (3,)), theta.layout.size


def u1_blocks(Vl, Vr, P, W, be, rng):
    leg = (lambda d, s: blocks.Leg(tuple(sorted(d)), tuple(d[k] for k in sorted(d)), s))
    vl, vr, p, w = (project(x, 0) for x in (Vl, Vr, P, W))
    theta = blocks.random([leg(vl, 1), leg(p, 1), leg(p, 1), leg(vr, -1)], 0, be, rng)
    L = blocks.random([leg(vl, 1), leg(vl, -1), leg(w, -1)], 0, be, rng)
    M = blocks.random([leg(w, 1), leg(p, 1), leg(p, -1), leg(w, -1)], 0, be, rng)
    R = blocks.random([leg(vr, 1), leg(w, 1), leg(vr, -1)], 0, be, rng)
    td = blocks.tensordot

    def heff():
        y = td(L, theta, ([1], [0]))                      # (a', w, s1, s2, b)
        y = td(y, M, ([1, 2], [0, 2]))                    # (a', s2, b, s1', w1)
        y = td(y, M, ([1, 4], [2, 0]))                    # (a', b, s1', s2', w2)
        return td(y, R, ([1, 4], [0, 1]))                 # (a', s1', s2', b')

    return theta, heff, lambda: theta.transpose((0, 2, 1, 3)).canonical(), theta.struct.size


def dense(Vl, Vr, P, W, be, torch, rng):
    dim = (lambda d: sum(n * (a[1] + 1) for a, n in d.items()))
    a, b, p, w = dim(Vl), dim(Vr), dim(P), dim(W)
    theta, L, M, R = (be.randn(rng, s) for s in [(a, p, p, b), (a, a, w), (w, p, p, w),
                                                  (b, w, b)])
    td = torch.tensordot if torch is not None else np.tensordot
    cont = (lambda x: x.contiguous()) if torch is not None else np.ascontiguousarray

    def heff():
        y = td(L, theta, ([1], [0]))
        y = td(y, M, ([1, 2], [0, 2]))
        y = td(y, M, ([1, 4], [2, 0]))
        return td(y, R, ([1, 4], [0, 1]))

    return theta, heff, lambda: cont(theta.permute(0, 2, 1, 3) if torch is not None
                                     else theta.transpose(0, 2, 1, 3)), a * b * p * p


# --------------------------------------------------------------------------- report
def structure_times(Vl, Vr, P, W, Vc):
    """Cold layout times (best of three), ranks 3 to 5, with one to three bond legs on a side.
    Vc is V with charges counted from half filling, so that several bonds can meet."""
    shapes = {"rank3 (V,P<-V')": ((Vl, P), (Vr,)), "rank3 (Vc,Vc<-Vc)": ((Vc, Vc), (Vc,)),
              "rank4 (V,P,P<-V')": ((Vl, P, P), (Vr,)),
              "rank4 (Vc,Vc<-Vc,Vc)": ((Vc, Vc), (Vc, Vc)),
              "rank5 (V,W,P,P<-V')": ((Vl, W, P, P), (Vr,)),
              "rank5 (Vc,Vc,Vc<-Vc,Vc)": ((Vc, Vc, Vc), (Vc, Vc))}
    out = {}
    for name, (cod, dom) in shapes.items():
        best = float("inf")
        for _ in range(3):
            clear()
            t0 = time.perf_counter()
            lay = sl.Layout(U1SU2, cod, dom)
            best = min(best, time.perf_counter() - t0)
        out[name] = {"ms": 1e3 * best, "stored": lay.size,
                     "row_trees": len(lay.row.trees.coupled),
                     "col_trees": len(lay.col.trees.coupled)}
    return out


def grouping():
    """Sector groups, matrix products and distinct shapes per recoupler met by one H_eff."""
    rows = []
    for m in st.INDEX_CACHE.values():
        if hasattr(m, "stacks"):
            rows.append({"groups": m.n_groups, "products": sum(s[2][0] for s in m.stacks),
                         "shapes": len(m.stacks)})
    return rows


def run(device, target, dense_max, rng, reps):
    be, torch = backend(device)
    sync = (torch.cuda.synchronize if torch is not None and torch.device(device).type == "cuda"
            else (lambda: None))
    n = 12
    Vl, Vr = bond(n, target), bond(n + 2, target)
    res = {"bond": {"sectors": len(Vl), "dense": sum(k * (a[1] + 1) for a, k in Vl.items()),
                    "multiplets": sum(Vl.values()), "right_sectors": len(Vr),
                    "charges": len({a[0] for a in Vl}), "max_2j": max(a[1] for a in Vl)}}
    centred = (lambda d, q0: {(q - q0, j): k for (q, j), k in d.items()})
    res["structure"] = structure_times(*(space(U1SU2, x) for x in (
        Vl, Vr, SITE, MPO, centred(Vl, n))))
    reps_of = {}
    clear()
    s = (lambda d: space(U1SU2, d))
    reps_of["u1su2"] = symmetric(s(Vl), s(Vr), s(SITE), s(MPO), be, rng)
    su2 = SU2()
    s2 = (lambda d: space(su2, project(d, 1)))
    reps_of["su2"] = symmetric(s2(Vl), s2(Vr), s2(SITE), s2(MPO), be, rng)
    reps_of["u1"] = u1_blocks(Vl, Vr, SITE, MPO, be, rng)
    if target <= dense_max:
        reps_of["dense"] = dense(Vl, Vr, SITE, MPO, be, torch, rng)
    for name, (theta, heff, perm, stored) in reps_of.items():
        first, warm, (lo, hi) = clock(heff, sync, reps)
        r = {"theta_stored": int(stored), "heff_first_ms": 1e3 * first, "heff_ms": 1e3 * warm,
             "heff_quartiles_ms": (1e3 * lo, 1e3 * hi),
             "permute_ms": 1e3 * clock(perm, sync, reps)[1], "heff_peak_bytes": peak(heff, torch)}
        if name == "u1su2":
            r["caches"] = {k: st.cache_info()[k] for k in ("recoupling", "structure", "indices")}
            r["recouplers"] = grouping()
        res[name] = r
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
    device = args[0] if args else "numpy"
    dims = [int(x) for x in opts.get("dims", "500,1000,2000").split(",")]
    dense_max = int(opts.get("dense", 500))
    rng = np.random.default_rng(0)
    out = {"device": device}
    for target in dims:
        r = out[target] = run(device, target, dense_max, rng, reps=int(opts.get("reps", 10)))
        print(f"target {target}: bond {r['bond']}")
        for k, v in r["structure"].items():
            print(f"  structure {k}: {v['ms']:.1f} ms, {v['stored']} stored, "
                  f"{v['row_trees']} + {v['col_trees']} trees")
        for name in ("u1su2", "su2", "u1", "dense"):
            if name in r:
                x = r[name]
                pk = x["heff_peak_bytes"]
                print(f"  {name:6s} theta {x['theta_stored']:>10d}  H_eff first "
                      f"{x['heff_first_ms']:8.1f} ms  warm {x['heff_ms']:8.2f} ms  permute "
                      f"{x['permute_ms']:7.3f} ms  peak "
                      + ("-" if pk is None else f"{pk / 2 ** 20:.1f} MB"))
        c = r["u1su2"]["caches"]["recoupling"]
        print(f"  recoupling matrices: {c['entries']} built, {c['hits']} reused; "
              f"recouplers {r['u1su2']['recouplers']}")
    if len(args) > 1:
        json.dump(out, open(args[1], "w"), indent=1, default=str)
