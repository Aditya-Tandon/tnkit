"""Factorisations of symmetric tensors (`tnkit.symmetric_linalg`) against dense linear algebra.

On U(1), Z_3, U(1) x U(1), SU(2), U(1) x SU(2), A4 and A4 x SU(2), with several legs on each side
and mixed dual flags:
  * the full SVD reconstructs the tensor, its factors are isometries, and its singular values,
    each repeated dim(c) times, are the dense singular values of the dense matrix;
  * a truncated SVD keeps the multiplets of largest weight dim(c) s^2, reports the squared
    dense norm of what it drops, and its product has whole multiplets as its dense spectrum;
  * `svd_tol` counts dense weight; `degenerate_tol` ties equal weights dim(c) s^2, which are
    equal dense values only between sectors of equal dimension; on U(1) x U(1) the cut equals
    the one of `tnkit.blocks.truncated_svd`;
  * `zero_tol` acts on s, keeps the largest value and counts what it drops against `svd_tol`;
  * charged and zero tensors, QR and LQ, the Gram route, and torch (batched, real, complex64)
    agree with the above; the singular values cross to the host once.
"""
import numpy as np
import pytest

from oracles.a4 import A4
from tnkit import blocks
from tnkit.backend import NumpyBackend
from tnkit.symmetric_linalg import qr, svd, truncated_svd
from tnkit.symmetric_tensor import Space, contract, from_blocks, random
from tnkit.symmetry import SU2, U1, ProductSymmetry, Zn

NP = NumpyBackend()
TOL = 1e-12


def _space(sym, leg, dual=False):
    secs = sorted(leg, key=sym.sort_key)
    return Space(sym, secs, tuple(leg[a] for a in secs), dual)


SYMS = {
    "U1": (U1(), [{-1: 2, 0: 3, 1: 2}, {0: 2, 1: 3}, {-1: 1, 1: 2}]),
    "Z3": (Zn(3), [{0: 2, 1: 2, 2: 1}, {1: 3, 2: 2}, {0: 1, 2: 2}]),
    "U1xU1": (ProductSymmetry((U1(), U1())),
              [{(0, 0): 2, (1, 0): 2, (0, 1): 2}, {(-1, 0): 2, (0, -1): 1, (0, 0): 2},
               {(0, 0): 1, (1, -1): 2}]),
    "SU2": (SU2(), [{0: 2, 1: 2, 2: 1}, {1: 2, 2: 1}, {0: 1, 1: 1, 2: 2}]),
    "U1xSU2": (ProductSymmetry((U1(), SU2())),
               [{(0, 0): 2, (1, 1): 2}, {(-1, 1): 1, (0, 0): 2, (0, 2): 1},
                {(1, 1): 2, (0, 0): 1}]),
    "A4": (A4(), [{0: 1, 3: 2}, {1: 1, 2: 1, 3: 2}, {0: 2, 1: 1, 3: 1}]),
    "A4xSU2": (ProductSymmetry((A4(), SU2())),
               [{(0, 0): 1, (3, 1): 2}, {(1, 1): 1, (3, 1): 1, (0, 0): 2}, {(0, 0): 2, (3, 1): 1}]),
}


def _shapes(A, B, C):
    """(codomain, domain) pairs: one or two legs on a side, mixed dual flags."""
    a, b, c = A.dual(), B.dual(), C.dual()
    return [((A,), (B,)), ((A, B), (C,)), ((a,), (B, c)), ((A, b), (c, A)), ((A, B, C), (a,)),
            ((b,), (A, C, a))]


CASES = [(name, i) for name in SYMS for i in range(6)]
IDS = [f"{n}-{i}" for n, i in CASES]


def _case(name, i):
    sym, legs = SYMS[name]
    return (sym,) + _shapes(*[_space(sym, leg) for leg in legs])[i]


def _rng(name, i):
    return np.random.default_rng([sorted(SYMS).index(name), i])


def _mat(t):
    """The dense matrix of a symmetric tensor: codomain legs as rows, domain legs as columns."""
    x = np.asarray(t.to_dense())
    rows = int(np.prod([V.dim for V in t.codomain]))
    return x.reshape(rows, -1)


def _spectrum(S):
    """The dense values of the diagonal bond matrix S, each reduced value dim(c) times."""
    out = [np.repeat(np.diag(np.asarray(S.block(c))).real, S.sym.dim(c)) for c in S.sectors]
    return np.sort(np.concatenate(out))[::-1]


def _product(U, S, Vh):
    US = contract(U, S, [len(U.codomain)], [0])
    return contract(US, Vh, [len(US.codomain)], [0])


def _err(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def _designed(sym, cod, dom, rng, spectrum):
    """A tensor whose block c has the singular values `spectrum(c, n)` and random vectors."""
    t = random(cod, dom, NP, rng)
    out = {}
    for c in t.sectors:
        r, k = t.layout.shapes[c]
        n = min(r, k)
        q1 = np.linalg.qr(rng.normal(size=(r, n)) + 1j * rng.normal(size=(r, n)))[0]
        q2 = np.linalg.qr(rng.normal(size=(k, n)) + 1j * rng.normal(size=(k, n)))[0]
        out[c] = (q1 * np.asarray(spectrum(c, n), float)) @ q2.conj().T
    return from_blocks(out, cod, dom, NP, sym)


def _check_svd(t, tol=TOL):
    U, S, Vh = svd(t)
    x, u, vh = _mat(t), _mat(U), _mat(Vh)
    assert _err(_mat(_product(U, S, Vh)), x) < tol
    assert _err(u.conj().T @ u, np.eye(u.shape[1])) < tol
    assert _err(vh @ vh.conj().T, np.eye(vh.shape[0])) < tol
    dense = np.linalg.svd(x, compute_uv=False)
    ours = _spectrum(S)
    assert _err(ours, dense[:len(ours)]) < tol
    assert np.abs(dense[len(ours):]).max(initial=0) < tol * max(dense[0], 1.0)
    assert U.domain[0].dim == len(ours)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_svd_against_dense(case):
    sym, cod, dom = _case(*case)
    _check_svd(random(cod, dom, NP, _rng(*case)))


def _weights(S):
    return {c: S.sym.dim(c) * np.diag(np.asarray(S.block(c))).real ** 2 for c in S.sectors}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_truncation_keeps_whole_multiplets_and_reports_the_dense_loss(case):
    sym, cod, dom = _case(*case)
    t = random(cod, dom, NP, _rng(*case))
    _, Sfull, _ = svd(t)
    w_all = np.sort(np.concatenate(list(_weights(Sfull).values())))[::-1]
    x = _mat(t)
    tot = float(np.linalg.norm(x) ** 2)
    for chi in sorted({1, 2, 3, len(w_all) // 2, len(w_all) - 1, len(w_all)} - {0}):
        info = {}
        U, S, Vh, disc = truncated_svd(t, chi, info=info)
        p = _mat(_product(U, S, Vh))
        assert abs(np.linalg.norm(x - p) ** 2 - disc) < 1e-10 * tot
        assert sum(info["kept"].values()) == min(chi, len(w_all))
        assert info["dim"] == U.domain[0].dim
        kept = np.sort(np.concatenate(list(_weights(S).values())))[::-1]
        assert _err(kept, w_all[:len(kept)]) < 1e-10            # the largest weights
        assert abs(disc - w_all[len(kept):].sum()) < 1e-10 * tot
        dense = np.linalg.svd(p, compute_uv=False)
        ours = _spectrum(S)
        assert _err(ours, dense[:len(ours)]) < 1e-10            # whole multiplets only
        assert np.abs(dense[len(ours):]).max(initial=0) < 1e-10 * dense[0]
        assert U.domain[0].dim == len(ours)
        assert _err(_mat(U).conj().T @ x @ _mat(Vh).conj().T, np.asarray(S.to_dense())) < 1e-10


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_svd_tol_counts_dense_weight(case):
    sym, cod, dom = _case(*case)
    t = random(cod, dom, NP, _rng(*case))
    tot = float(np.linalg.norm(_mat(t)) ** 2)
    _, Sfull, _ = svd(t)
    w = np.sort(np.concatenate(list(_weights(Sfull).values())))[::-1]
    for tol in (1e-3, 0.05, 0.3):
        _, _, _, disc = truncated_svd(t, None, svd_tol=tol)
        n = len(w) - int(np.sum(np.cumsum(w[::-1]) / tot <= tol))
        n = max(n, 1)
        assert abs(disc - w[n:].sum()) < 1e-10 * tot
        assert disc / tot <= tol or n == 1


def test_ranking_is_by_dimension_weighted_square():
    """A spin-1 multiplet at s = 0.7 (weight 1.47) outranks a singlet at s = 1 (weight 1)."""
    sym = SU2()
    V = _space(sym, {0: 1, 2: 1})
    t = from_blocks({0: np.array([[1.0]]), 2: np.array([[0.7]])}, (V,), (V,), NP, sym)
    U, S, Vh, disc = truncated_svd(t, 1)
    assert S.sectors == (2,)
    assert abs(disc - 1.0) < TOL
    U, S, Vh, disc = truncated_svd(t, None, svd_tol=1.0 / 2.47 + 1e-9)
    assert S.sectors == (2,) and abs(disc - 1.0) < TOL


def _tied(sym, cod, dom, rng):
    """Blocks with values 1, 0.9, 0.5, 0.5 (1 + 1e-9), 0.2, ... scaled by 1 / sqrt(dim c), so
    that ties in weight occur across sectors as well as inside one."""
    base = np.array([1.0, 0.9, 0.5, 0.5 * (1 + 1e-9), 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])

    def spec(c, n):
        return np.resize(base, n) / np.sqrt(sym.dim(c))
    return _designed(sym, cod, dom, rng, spec)


@pytest.mark.parametrize("name", ["SU2", "U1xSU2", "A4"])
def test_degenerate_tol_keeps_or_drops_tied_multiplets_whole(name):
    """Every sector holds two multiplets of equal weight next to each other: a cap between them
    moves down below the pair, an svd_tol cut moves up above it, as `_cut` does for one
    spectrum. The ties here are ties in weight."""
    sym, legs = SYMS[name]
    cod, dom = (_space(sym, legs[0]), _space(sym, legs[1])), (_space(sym, legs[2]),)
    t = _tied(sym, cod, dom, np.random.default_rng(3))
    _, Sfull, _ = svd(t)
    w = np.sort(np.concatenate(list(_weights(Sfull).values())))[::-1]
    groups = np.split(w, np.flatnonzero(np.abs(np.diff(w)) > 1e-6 * w[1:]) + 1)
    assert len(t.sectors) >= 2 and max(len(g) for g in groups) >= 2
    edges = np.cumsum([len(g) for g in groups])
    tot = w.sum()
    tail = np.r_[np.cumsum(w[::-1])[::-1], 0.0]               # tail[m]: weight from m on
    for j, g in enumerate(groups[:-1]):
        if len(g) < 2:
            continue
        start = edges[j] - len(g)
        for chi in range(start + 1, edges[j]):
            info = {}
            truncated_svd(t, chi, degenerate_tol=1e-6, info=info)
            assert sum(info["kept"].values()) == (start or chi)    # a leading group is split
            assert info["pinned"] == bool(start)
            info0 = {}
            truncated_svd(t, chi, info=info0)
            assert sum(info0["kept"].values()) == chi and not info0["pinned"]
        tol = (tail[start] + tail[start + 1]) / 2 / tot          # the plain cut keeps start + 1
        for dtol, n in ((0.0, start + 1), (1e-6, edges[j])):
            info = {}
            truncated_svd(t, None, svd_tol=tol, degenerate_tol=dtol, info=info)
            assert sum(info["kept"].values()) == n


def _block_twin(t):
    """The `BlockTensor` of an abelian two-leg symmetric tensor (codomain one leg, domain one)."""
    (A,), (B,) = t.codomain, t.domain
    legs = (blocks.Leg(A.sectors, A.degeneracies, -1 if A.isdual else 1, t.sym),
            blocks.Leg(B.sectors, B.degeneracies, 1 if B.isdual else -1, t.sym))
    bt, dropped = blocks.from_dense(np.asarray(t.to_dense()), legs)
    assert dropped < TOL
    return bt


def test_cut_equals_block_tensors_on_u1xu1():
    sym, legs = SYMS["U1xU1"]
    A = _space(sym, {(0, 0): 4, (1, 0): 3, (0, 1): 3, (1, 1): 2})
    t = _tied(sym, (A,), (A,), np.random.default_rng(5))
    bt = _block_twin(t)
    for chi in range(1, 13):
        for tol in (0.0, 1e-3, 0.02, 0.2):
            for dtol in (0.0, 1e-6, 0.15):
                for zt in (0.0, 0.03, 0.3):
                    i1, i2 = {}, {}
                    kw = dict(svd_tol=tol, degenerate_tol=dtol, zero_tol=zt)
                    _, S, _, d1 = truncated_svd(t, chi, info=i1, **kw)
                    _, S2, _, d2 = blocks.truncated_svd(bt, chi, info=i2, **kw)
                    assert abs(d1 - d2) < 1e-12 and i1["pinned"] == i2["pinned"]
                    assert _err(np.sort(_spectrum(S)), np.sort(np.asarray(S2))) < 1e-12


def test_zero_tol_drops_small_values():
    sym = SU2()
    V = _space(sym, {0: 2, 1: 1})
    blk = {0: np.diag([1.0, 1e-15]), 1: np.array([[0.5]])}
    t = from_blocks(blk, (V,), (V,), NP, sym)
    info = {}
    _, S, _, disc = truncated_svd(t, 10, zero_tol=1e-12, info=info)
    assert info["kept"] == {0: 1, 1: 1}
    assert abs(disc - 1e-30) < 1e-40


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_qr_and_lq(case):
    sym, cod, dom = _case(*case)
    t = random(cod, dom, NP, _rng(*case))
    x = _mat(t)
    Q, R = qr(t)
    q = _mat(Q)
    assert _err(_mat(contract(Q, R, [len(Q.codomain)], [0])), x) < TOL
    assert _err(q.conj().T @ q, np.eye(q.shape[1])) < TOL
    assert all(Q.layout.shapes[c][1] == min(t.layout.shapes[c]) for c in t.sectors)
    L, Q = qr(t, right=True)
    q = _mat(Q)
    assert _err(_mat(contract(L, Q, [len(L.codomain)], [0])), x) < TOL
    assert _err(q @ q.conj().T, np.eye(q.shape[0])) < TOL


@pytest.mark.parametrize("name", ["SU2", "U1xSU2", "A4"])
def test_gram_route(name):
    sym, legs = SYMS[name]
    cod, dom = (_space(sym, legs[0]), _space(sym, legs[1])), (_space(sym, legs[2]),)
    t = random(cod, dom, NP, np.random.default_rng(7))
    for chi in (2, 5, None):
        U, S, Vh, d = truncated_svd(t, chi)
        U2, S2, Vh2, d2 = truncated_svd(t, chi, method="gram")
        assert abs(d - d2) < 1e-10 * float(np.linalg.norm(_mat(t))) ** 2
        assert _err(_spectrum(S2), _spectrum(S)) < 1e-10
        assert _err(_mat(_product(U2, S2, Vh2)), _mat(_product(U, S, Vh))) < 1e-10


@pytest.mark.parametrize("name", ["U1xU1", "SU2", "U1xSU2"])
def test_torch_matches_numpy_with_batched_blocks(name, monkeypatch):
    torch = pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    sym, legs = SYMS[name]
    A, B, C = (_space(sym, leg) for leg in legs)
    tb = TorchBackend("cpu", torch.complex128)
    calls = []
    real = tb.svd_batched
    monkeypatch.setattr(tb, "svd_batched", lambda X: calls.append(X.shape) or real(X))
    for cod, dom in (((A,), (A,)), ((A, B), (C,))):
        t = random(cod, dom, NP, np.random.default_rng(11))
        tt = from_blocks({c: tb.array(b) for c, b in t.blocks().items()}, cod, dom, tb, sym)
        for chi in (3, None):
            U, S, Vh, d = truncated_svd(t, chi)
            Ut, St, Vht, dt = truncated_svd(tt, chi)
            assert abs(d - dt) < 1e-12 * max(1.0, d)
            assert _err(_spectrum(St), _spectrum(S)) < 1e-12
            assert _err(_mat(_product(Ut, St, Vht)), _mat(_product(U, S, Vh))) < 1e-12
        Q, R = qr(tt)
        assert _err(_mat(contract(Q, R, [len(cod)], [0])), _mat(t)) < TOL
    assert any(s[0] > 1 for s in calls)                                 # (A,) -> (A,) batches


def test_real_backend_and_errors():
    from tnkit.backend import get_backend
    sym, legs = SYMS["SU2"]
    cod, dom = (_space(sym, legs[0]),), (_space(sym, legs[1]),)
    be = get_backend("numpy-real")
    t = random(cod, dom, be, np.random.default_rng(1))
    U, S, Vh, _ = truncated_svd(t, 2)
    assert np.asarray(U.data).dtype == np.float64
    with pytest.raises(ValueError, match="chi"):
        truncated_svd(t, 0)


def _kept(blk, legs, sym, **kw):
    """`info` of a truncation of the one-leg map with these diagonal blocks."""
    V = _space(sym, legs)
    t = from_blocks({c: np.diag(np.asarray(b, float)) for c, b in blk.items()}, (V,), (V,), NP,
                    sym)
    info = {}
    out = truncated_svd(t, info=info, **kw)
    return info, out[3]


def test_degenerate_tol_ties_equal_weights_only():
    """Ties are in dim(c) s^2. Equal dense values in sectors of different dimension have
    different weights and can be split; values of equal weight that differ in dense value are
    treated as tied."""
    sym = SU2()
    # dense values 1, then 0.5 eight times (two doublets, one triplet, one singlet)
    info, _ = _kept({0: [1, .5], 1: [.5, .5], 2: [.5]}, {0: 2, 1: 2, 2: 1}, sym, chi=2,
                    degenerate_tol=1e-6)
    assert info["kept"] == {0: 1, 2: 1} and not info["pinned"] and info["dim"] == 4
    # a singlet at sqrt(3)/2 and a triplet at 1/2 have the same weight 3/4
    info, _ = _kept({0: [1, 3 ** .5 / 2], 2: [.5]}, {0: 2, 2: 1}, sym, chi=2,
                    degenerate_tol=1e-6)
    assert info["kept"] == {0: 1} and info["pinned"]


def test_density_ranks_eigenvalues_by_dimension_times_value():
    """A density matrix: a triplet at 0.4 (weight 1.2) outranks a singlet at 1, which a ranking
    by dim(c) s^2 (0.48 against 1) would keep."""
    sym = SU2()
    info, disc = _kept({0: [1.0], 2: [0.4]}, {0: 1, 2: 1}, sym, chi=1, density=True)
    assert info["kept"] == {2: 1} and abs(disc - 1.0) < TOL
    info, _ = _kept({0: [1.0], 2: [0.4]}, {0: 1, 2: 1}, sym, chi=1)
    assert info["kept"] == {0: 1}


def test_zero_tol_acts_on_s_and_keeps_the_largest():
    sym = SU2()
    # the triplet's weight 0.75 is above 0.6 of the largest, its s = 0.5 is not
    info, disc = _kept({0: [1.0], 2: [0.5]}, {0: 1, 2: 1}, sym, chi=5, zero_tol=0.6)
    assert info["kept"] == {0: 1} and abs(disc - 0.75) < TOL
    # nothing passes zero_tol >= 1, the largest s (not the largest weight) stays
    info, disc = _kept({0: [1.0], 2: [0.9]}, {0: 1, 2: 1}, sym, chi=5, zero_tol=2.0)
    assert info["kept"] == {0: 1} and abs(disc - 3 * 0.81) < TOL


def test_zero_tol_weight_counts_against_svd_tol():
    """Values 1, 0.3, 0.1 with 0.1 below zero_tol: dropping 0.3 as well loses 0.1 / 1.1 > 0.085
    of the weight, so svd_tol = 0.085 keeps two values."""
    sym = SU2()
    info, disc = _kept({0: [1, .3, .1]}, {0: 3}, sym, zero_tol=0.2, svd_tol=0.085)
    assert info["kept"] == {0: 2} and abs(disc - 0.01) < TOL
    info, _ = _kept({0: [1, .3, .1]}, {0: 3}, sym, zero_tol=0.2, svd_tol=0.095)
    assert info["kept"] == {0: 1}


@pytest.mark.parametrize("name", ["U1", "SU2", "U1xSU2", "A4xSU2"])
def test_charged_tensor(name):
    """A tensor of total charge q: an extra domain leg holding only q."""
    sym, legs = SYMS[name]
    A, B, C = (_space(sym, leg) for leg in legs)
    q = max(legs[0], key=sym.dim) if name != "U1" else 1
    Q = Space(sym, (q,), (1,))
    t = random((A, B), (C, Q), NP, np.random.default_rng(2))
    assert sym.dim(q) > 1 or name == "U1"
    _check_svd(t)
    x = _mat(t)
    for chi in (1, 3):
        U, S, Vh, disc = truncated_svd(t, chi)
        assert abs(np.linalg.norm(x - _mat(_product(U, S, Vh))) ** 2 - disc) < 1e-10
    Q_, R = qr(t)
    assert _err(_mat(contract(Q_, R, [2], [0])), x) < TOL


def test_zero_tensor():
    sym, legs = SYMS["SU2"]
    A, B = _space(sym, legs[0]), _space(sym, legs[1])
    t = from_blocks({}, (A,), (B,), NP, sym)
    info = {}
    U, S, Vh, disc = truncated_svd(t, 3, info=info)
    assert disc == 0 and sum(info["kept"].values()) == 3
    assert np.abs(_mat(_product(U, S, Vh))).max() == 0
    truncated_svd(t, 3, zero_tol=1e-12, info=info)
    assert sum(info["kept"].values()) == 1
    Q, R = qr(t)
    assert np.abs(_mat(contract(Q, R, [1], [0]))).max() == 0


def test_no_shared_sector_is_an_error():
    sym = U1()
    t = from_blocks({}, (_space(sym, {0: 2}),), (_space(sym, {1: 2}),), NP, sym)
    for f in (svd, qr, lambda x: qr(x, right=True), lambda x: truncated_svd(x, 2)):
        with pytest.raises(ValueError, match="no coupled sectors"):
            f(t)


def test_gram_runs_the_gram_route(monkeypatch):
    import tnkit.symmetric_linalg as sl
    sym, legs = SYMS["SU2"]
    t = random((_space(sym, legs[0]),), (_space(sym, legs[1]),), NP, np.random.default_rng(4))
    calls = []
    real = sl._svd_block
    monkeypatch.setattr(sl, "_svd_block", lambda M, m, d: calls.append(m) or real(M, m, d))
    truncated_svd(t, 2, method="gram")
    assert calls == ["gram"] * len(t.sectors)
    calls.clear()
    truncated_svd(t, 2)
    assert not calls
    with pytest.raises(ValueError, match="method"):
        truncated_svd(t, 2, method="eig")


def test_gram_refuses_single_precision():
    sym, legs = SYMS["SU2"]
    be = NumpyBackend(np.complex64)
    t = random((_space(sym, legs[0]),), (_space(sym, legs[1]),), be, np.random.default_rng(4))
    with pytest.raises(ValueError, match="double precision"):
        truncated_svd(t, 2, method="gram")
    truncated_svd(t, 2)


def _torch_case(name, dtype, rng):
    torch = pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    sym, legs = SYMS[name]
    A, B, C = (_space(sym, leg) for leg in legs)
    tb = TorchBackend("cpu", getattr(torch, dtype))
    nb = NumpyBackend(np.float64 if dtype == "float64" else np.complex128)
    t = random((A, B), (C,), nb, rng)
    return t, from_blocks({c: tb.array(b) for c, b in t.blocks().items()}, t.codomain,
                          t.domain, tb, sym)


@pytest.mark.parametrize("dtype,tol", [("float64", 1e-12), ("complex64", 1e-5)])
def test_torch_real_and_single_precision(dtype, tol):
    t, tt = _torch_case("U1xSU2", dtype, np.random.default_rng(6))
    for chi in (2, None):
        U, S, Vh, d = truncated_svd(t, chi)
        Ut, St, Vht, dt = truncated_svd(tt, chi)
        assert St.data.dtype == tt.data.dtype and abs(d - dt) < tol * max(1.0, d)
        assert _err(_mat(_product(Ut, St, Vht)), _mat(_product(U, S, Vh))) < tol
    for right in (False, True):
        a, b = qr(tt, right)
        assert _err(_mat(contract(a, b, [len(a.codomain)], [0])), _mat(t)) < tol


@pytest.mark.parametrize("method", ["svd", "gram"])
def test_one_host_copy(method, monkeypatch):
    t, tt = _torch_case("U1xSU2", "complex128", np.random.default_rng(8))
    assert len(t.sectors) > 1
    calls = []
    real = tt.be.to_numpy
    monkeypatch.setattr(tt.be, "to_numpy", lambda x: calls.append(1) or real(x))
    truncated_svd(tt, 3, method=method)
    assert len(calls) == 1


@pytest.mark.parametrize("torch_", [False, True])
def test_stacked_svd_falls_back_per_block(torch_, monkeypatch):
    t, tt = _torch_case("SU2", "complex128", np.random.default_rng(9))
    x = tt if torch_ else t

    def fail(X):
        raise np.linalg.LinAlgError("SVD did not converge")
    monkeypatch.setattr(x.be, "svd_batched", fail)
    U, S, Vh, d = truncated_svd(x, 3)
    monkeypatch.undo()
    ref = truncated_svd(t, 3)
    assert abs(d - ref[3]) < 1e-12 * max(1.0, ref[3])
    assert _err(_mat(_product(U, S, Vh)), _mat(_product(*ref[:3]))) < 1e-12


@pytest.mark.cuda
def test_driver_reaches_the_cuda_svd(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from tnkit.backend import TorchBackend
    sym, legs = SYMS["SU2"]
    tb = TorchBackend("cuda")
    t = random((_space(sym, legs[0]),), (_space(sym, legs[1]),), tb, np.random.default_rng(4))
    seen = []
    real = torch.linalg.svd
    monkeypatch.setattr(torch.linalg, "svd",
                        lambda X, **kw: seen.append(kw.get("driver")) or real(X, **kw))
    truncated_svd(t, 2, driver="gesvdj")
    assert seen and set(seen) == {"gesvdj"}
