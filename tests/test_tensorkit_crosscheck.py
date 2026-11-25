"""tnkit's symmetric code against numbers stored from TensorKit and MPSKit.

tests/data/tensorkit_crosscheck.json holds TensorKit.jl and MPSKit.jl reference values, with the
versions recorded in the file's `provenance` (Julia is not run here): SU(2) Clebsch-Gordan, F and R symbols up to twice the spin 4;
the sector ground-state energies and middle-bond singular values of open Heisenberg chains
(SU(2)) and two-flavour Schwinger chains (U(1) x SU(2)); and full-rank two-site TDVP after a
quench. Each test rebuilds the problem with tnkit and compares within a fixed tolerance.
"""
import json
import os

import numpy as np
import pytest

from tnkit import su2
from tnkit import symmetric_linalg as sl
from tnkit._dmrg import right_canonicalize, update_left_env
from tnkit.backend import get_backend
from tnkit.dmrg import dmrg
from tnkit.legs import fuse, split
from tnkit.models_lgt2f import (D, U, compile2f, merge_pairs, qubit, schwinger2f_mpo,
                                site_spaces_2f)
from tnkit.opsum import OpSum, compile_mpo, spin_half
from tnkit.symmetric_backend import (SymmetricBackend, edge_environments, spin_half_site,
                                     symmetric_mpo)
from tnkit.symmetric_tensor import from_dense
from tnkit.tdvp import tdvp2_evolve

GOLD = json.load(open(os.path.join(os.path.dirname(__file__), "data", "tensorkit_crosscheck.json")))
SLOW = {"heis20", "schw12"}
DMRG = dict(n_sweeps=30, tol=1e-12, rtol=0.0, eig_tol=1e-12)
FLOOR = 1.01e-6        # the floor the stored singular values were cut at (make_golden.py)
TOL_SU2, TOL_E, TOL_S, TOL_NORM, TOL_TDVP = 1e-12, 1e-10, 1e-8, 1e-10, 1e-6


def params(names, slow=SLOW):
    return [pytest.param(n, marks=pytest.mark.slow) if n in slow else n for n in names]


# ----------------------------------------------------------------------------- models
def heisenberg_dense(N, J):
    os_ = OpSum()
    for b in range(N - 1):
        if J[b]:
            os_ += (0.5 * J[b], "Sp", b, "Sm", b + 1)
            os_ += (0.5 * J[b], "Sm", b, "Sp", b + 1)
            os_ += (0.25 * J[b], "Z", b, "Z", b + 1)
    return [np.real(w) for w in compile_mpo(os_, [spin_half()] * N)[0]]


def build(case, which=""):
    """Symmetric MPO and sites; `which` = "0" picks the initial couplings of a quench."""
    N = case["N"]
    if case["model"] == "heisenberg":
        d = case["delta" + which]
        W = heisenberg_dense(N, [1 + d if b % 2 == 0 else 1 - d for b in range(N - 1)])
        sites = [spin_half_site()] * N
    else:
        W = [np.real(w) for w in merge_pairs(schwinger2f_mpo(N, case["x"], case["mu" + which]))]
        sites = site_spaces_2f(N)
    return symmetric_mpo(W, sites)[0], sites


def local_observable(case, sites):
    N = case["N"]
    if case["model"] == "heisenberg":
        J = [0.0] * (N - 1)
        J[N // 2 - 1] = 1.0
        W = heisenberg_dense(N, J)
    else:
        os_ = OpSum()
        os_ += (1.0, "N", qubit(N // 2, U))
        os_ += (1.0, "N", qubit(N // 2, D))
        W = [np.real(w) for w in merge_pairs(compile2f(os_, N)[0])]
    return symmetric_mpo(W, sites)[0]


def charge(case, sec):
    return sec[0] if case["model"] == "heisenberg" else tuple(sec)


def label(c):
    return [int(v) for v in c] if isinstance(c, tuple) else [int(c)]


# ----------------------------------------------------------------------------- measurements
def host(x):
    return x.detach().resolve_conj().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def scalar(x, be):
    return complex(host(be.dense.to_numpy(x)).reshape(-1)[0])


def expectation(M, Ws, be):
    """<M|W|M> of a normalised MPS (the norm is checked in the spectrum test)."""
    L, R = edge_environments(M, Ws, be)
    for A, W in zip(M, Ws):
        L = update_left_env(L, A, W, be)
    return scalar(be.einsum("pwq,pwq->", L, R), be).real


def bond_spectrum(M, b, be):
    """Middle-bond Schmidt values per sector, sum_c dim(c) s^2 and the entropy."""
    sym = M[0].sym
    M = right_canonicalize([be.array(t) for t in M], be)
    C = M[0]
    for i in range(b):
        f, rec = fuse(C, ((0, 1), (2,)), be)
        _, S, Vh, _ = sl.truncated_svd(f, None)
        if i == b - 1:
            out, tot, ent = {}, 0.0, 0.0
            for c in S.sectors:
                s = np.sort(np.abs(np.diag(host(be.dense.to_numpy(S.block(c))))))[::-1]
                d = sym.dim(c)
                out[tuple(label(c))] = (d, s)
                tot += d * float(np.sum(s ** 2))
                p = s[s > 0] ** 2
                ent -= d * float(np.sum(p * np.log(p)))
            return out, tot, ent
        SV = split(be.einsum("ab,bc->ac", S, Vh), rec, be, legs=(1,))
        C = be.einsum("rb,bsc->rsc", SV, M[i + 1])


# ----------------------------------------------------------------------------- SU(2) symbols
def test_golden_file_is_the_expected_run():
    p = GOLD["provenance"]
    assert (p["julia"], p["TensorKit"], p["MPSKit"]) == ("1.13.3", "0.17.2", "0.14.0")
    assert {c["name"] for c in GOLD["cases"]["spectra"]} == set(GOLD["levels"]) == set(GOLD["spectra"])
    assert {c["name"] for c in GOLD["cases"]["tdvp"]} == set(GOLD["tdvp"])


def test_su2_symbols_match_tensorkit():
    g = GOLD["su2"]
    jm = GOLD["cases"]["su2_max"]
    rng = range(jm + 1)
    cg, f, r = {}, {}, {}
    for a in rng:
        for b in rng:
            for c in rng:
                if su2.triangle(a, b, c):
                    r[(a, b, c)] = su2.rsymbol(a, b, c)
                    arr = su2.cg_array(a, b, c)
                    for idx in np.ndindex(arr.shape):
                        cg[(a, b, c) + idx] = arr[idx]
                for d in rng:
                    for e in rng:
                        for h in rng:
                            if (su2.triangle(a, b, e) and su2.triangle(e, c, d)
                                    and su2.triangle(b, c, h) and su2.triangle(a, h, d)):
                                f[(a, b, c, d, e, h)] = su2.fsymbol(a, b, c, d, e, h)
    for mine, theirs in ((cg, g["CG"]), (f, g["F"]), (r, g["R"])):
        gold = {tuple(row[:-1]): row[-1] for row in theirs}
        assert set(mine) == set(gold)
        assert max(abs(mine[k] - gold[k]) for k in gold) < TOL_SU2


# ----------------------------------------------------------------------------- statics
@pytest.fixture(scope="module")
def ground_states():
    cache = {}

    def get(name):
        if name not in cache:
            case = next(c for c in GOLD["cases"]["spectra"] if c["name"] == name)
            Ws, _ = build(case)
            cache[name] = {}
            for sec in case["sectors"]:
                E, M, _ = dmrg(Ws, case["chi"], charge=charge(case, sec), backend="numpy-real", **DMRG)
                cache[name][tuple(sec)] = (E, list(M))
        return cache[name]
    return get


@pytest.mark.parametrize("name", params(GOLD["levels"]))
def test_levels_match_tensorkit(name, ground_states):
    case = next(c for c in GOLD["cases"]["spectra"] if c["name"] == name)
    mine = ground_states(name)
    for lev in GOLD["levels"][name]:
        sec = tuple(lev["sector"])
        E = mine[sec][0]
        assert abs(E - lev["E"]) <= TOL_E * max(1.0, abs(lev["E"])), (sec, E, lev["E"])
        sym = mine[sec][1][0].sym
        assert sym.dim(charge(case, sec)) == lev["dim"]
    order = lambda en: sorted(en, key=en.get)
    assert order({s: mine[s][0] for s in mine}) == order({tuple(l["sector"]): l["E"] for l in GOLD["levels"][name]})


@pytest.mark.parametrize("name", params(GOLD["spectra"]))
def test_entanglement_spectra_match_tensorkit(name, ground_states):
    case = next(c for c in GOLD["cases"]["spectra"] if c["name"] == name)
    be = SymmetricBackend(get_backend("numpy-real"))
    for es in GOLD["spectra"][name]:
        sec = tuple(es["sector"])
        spec, tot, ent = bond_spectrum(ground_states(name)[sec][1], case["N"] // 2, be)
        assert abs(tot - 1.0) < TOL_NORM
        assert abs(ent - es["S"]) < TOL_S, (sec, ent, es["S"])
        mine = {c: (d, s[s > FLOOR]) for c, (d, s) in spec.items() if np.any(s > FLOOR)}
        assert set(mine) == {tuple(s["sector"]) for s in es["sectors"]}, sec
        for s in es["sectors"]:
            d, v = mine[tuple(s["sector"])]
            assert d == s["dim"]
            assert len(v) == len(s["s"]), (sec, s["sector"])
            assert np.max(np.abs(v - np.array(s["s"]))) < TOL_S, (sec, s["sector"])


# ----------------------------------------------------------------------------- TDVP
@pytest.mark.parametrize("name", list(GOLD["tdvp"]))
def test_full_rank_tdvp_matches_tensorkit(name):
    case = next(c for c in GOLD["cases"]["tdvp"] if c["name"] == name)
    gold = GOLD["tdvp"][name]
    N, chi = case["N"], case["chi"]
    W0, sites = build(case, "0")
    W1, _ = build(case)
    obs = local_observable(case, sites)
    start = charge(case, [0] if case["model"] == "heisenberg" else [0, 0])
    _, psi, _ = dmrg(W0, chi, charge=start, backend="numpy-real", **DMRG)
    be = SymmetricBackend(get_backend("numpy"))
    got = {"E": [], "obs": [], "S": []}

    def measure(M):
        M = [be.array(t) for t in M]
        got["E"].append(expectation(M, W1, be))
        got["obs"].append(expectation(M, obs, be))
        got["S"].append(bond_spectrum(M, N // 2, be)[2])

    measure(list(psi))
    _, recs = tdvp2_evolve(list(psi), W1, chi, case["dt"], case["steps"], backend=be,
                           measure=measure, meas_every=case["meas_every"])
    times = [0.0] + [t for t, _ in recs]
    if len(times) > 1 and times[1] == 0.0:              # the integrator also measured t = 0
        for v in got.values():
            del v[1]
        del times[1]
    assert np.allclose(times, gold["t"], atol=1e-12)
    for key in got:
        assert np.max(np.abs(np.array(got[key]) - np.array(gold[key]))) < TOL_TDVP, key
