"""Global subspace expansion (`TDVPOptions.expand`) on dense and block tensors."""
from collections import Counter

import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

from tnkit import blocks as B
from tnkit.api import dmrg
from tnkit.backend import get_backend
from tnkit.block_backend import BlockBackend, to_block_mpo, to_block_mps, to_dense_mps
from tnkit.expansion import expand_right
from tnkit.models_lgt2f import (compile2f, gaussian_profile, packet_opsum, schwinger2f_mpo,
                                schwinger2f_sparse, site_charges_2f)
from tnkit.observables import apply_mpo, compress, to_numpy
from tnkit.tdvp import TDVPOptions, _edge_envs, tdvp2_evolve
from tnkit.symmetry import U1
from tnkit.tdvp_evolver import TDVPEvolver

X, MU, DT = 4.0, 0.3, 0.05
DENSE = get_backend("numpy")
BB = BlockBackend(DENSE)
BACKENDS = ["numpy"]
try:
    import torch
    BACKENDS += ["torch-cpu", pytest.param("torch-cuda", marks=[
        pytest.mark.cuda,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")])]
except ImportError:
    pass


def vector(mps):
    v = np.ones((1, 1), complex)
    for t in mps:
        v = np.tensordot(v, np.asarray(t), (v.ndim - 1, 0))
    return v.reshape(-1)


def fidelity(u, w):
    return abs(np.vdot(u, w)) ** 2 / (np.vdot(u, u).real * np.vdot(w, w).real)


def strong_coupling_vacuum(N):
    """The product state with even staggered sites empty and odd ones full: every bond is one
    state in one sector."""
    out = []
    for j in range(2 * N):
        t = np.zeros((1, 2, 1), complex)
        t[0, (j // 2) % 2, 0] = 1
        out.append(t)
    return out


def evolve(psi, W, qs, cap, expand, symmetry, n, method="12site", backend="numpy",
           profile=None):
    dbe = get_backend(backend)
    if symmetry == "blocks":
        psi, W = to_block_mps(psi, qs, dbe, sym=U1())[0], to_block_mpo(W, qs, dbe, sym=U1())[0]
        be = BlockBackend(dbe)
    else:
        be = backend
    ev = TDVPEvolver(psi, W, DT, cap, TDVPOptions(backend=be, svd_tol=1e-10, graph="off",
                                                  expand=expand, profile=profile))
    ev.run(n, method)
    M = to_dense_mps(ev.mps, qs) if symmetry == "blocks" else ev.mps
    return ev.mps, vector([dbe.to_numpy(t) for t in M])


def test_concat():
    rng = np.random.default_rng(1)
    a = B.random((B.Leg((0, 1), (2, 1)), B.Leg((0, 1), (1, 1)), B.Leg((0, 1, 2), (1, 2, 1), -1)),
                 0, DENSE, rng)
    b = B.random((B.Leg((1, 2), (2, 3)),) + a.legs[1:], 0, DENSE, rng)
    c = B.concat(a, b, 0)
    assert c.legs[0] == B.Leg((0, 1, 2), (2, 3, 3))
    d = np.asarray(c.to_dense())
    np.testing.assert_array_equal(d[[0, 1, 2]], a.to_dense())     # charges 0 and 1 of a
    np.testing.assert_array_equal(d[[3, 4, 5, 6, 7]], b.to_dense())
    with pytest.raises(ValueError):
        B.concat(a, b, 1)


def _held(v):
    """What a cache entry may share with others: its indices (plans included) and a plan's
    output layout (written apart from `B._shared`)."""
    if isinstance(v, B._Index):
        yield v
    elif isinstance(v, tuple):
        for x in v:
            yield from _held(x)
    elif isinstance(v, B._Plan):
        yield from (x for x in (v.ia, v.ib) if x is not None)
        yield v.out
    elif isinstance(v, B._Gather) and v.ix is not None:
        yield v.ix


def _nbytes(x):
    """An index's bytes; a layout's arrays plus the offsets and strides that intern it."""
    if isinstance(x, B.Struct):
        off, st, shp = x.arrays
        return len(off.tobytes()) + len(st.tobytes()) + off.nbytes + st.nbytes + shp.nbytes
    return x.nbytes


def _metadata(v):
    """Host arrays an entry holds besides what it shares: a gather's offsets."""
    return sum(a.nbytes for a in v.offs) if isinstance(v, B._Gather) else 0


def held_bytes():
    """Bytes of the distinct indices and layouts reachable from the cache's entries, plus their
    metadata. Checks on the way that each one's `refs` is the number of entries holding it."""
    entries = B._INDEX_CACHE._d.values()
    holders = Counter(id(x) for v in entries for x in _held(v))
    held = {id(x): x for v in entries for x in _held(v)}
    assert all(x.refs == holders[i] for i, x in held.items())
    return sum(_nbytes(x) for x in held.values()) + sum(_metadata(v) for v in entries)


def test_cache_byte_total():
    """The running byte total equals a recount over the distinct indices and the metadata the
    entries hold."""
    B.clear_cache()
    evolve(strong_coupling_vacuum(2), schwinger2f_mpo(2, X, MU), site_charges_2f(2), 8, 2,
           "blocks", 2)
    recount = held_bytes()
    assert B._INDEX_CACHE.nbytes() == recount > 0 and B.cache_info()["bytes"] == recount
    B.clear_cache()
    assert B.cache_info()["bytes"] == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_capped_cache_counts_what_it_holds(backend):
    """Under a limit of a fifth of a run's bytes, gathers are evicted while the plans that share
    their indices stay; the running total must still equal the bytes the entries hold."""
    def run():
        evolve(strong_coupling_vacuum(4), schwinger2f_mpo(4, X, MU), site_charges_2f(4), 16, 0,
               "blocks", 3, "2site", backend)
    limit = B.cache_info()["max_bytes"]
    try:
        B.clear_cache()
        run()
        full = B.cache_info()["bytes"]
        B.clear_cache()
        B.set_cache_limit(full // 5)
        run()
        info = B.cache_info()
        assert info["evictions"] > 0 and info["bytes"] == held_bytes() > 0
    finally:
        B.set_cache_limit(limit)
        B.clear_cache()


def test_expansion_fills_empty_sectors():
    N = 3
    W, qs = schwinger2f_mpo(N, X, MU), site_charges_2f(N)
    M = to_block_mps(strong_coupling_vacuum(N), qs, DENSE, sym=U1())[0]
    Wb = to_block_mpo(W, qs, DENSE, sym=U1())[0]
    before = [t.legs[2].charges for t in M[:-1]]
    v0 = vector(to_dense_mps(M, qs))
    L0, RN = _edge_envs(M, Wb, BB, None, None)
    added = expand_right(M, Wb, 4, 16, BB, L0, RN)
    after = [t.legs[2].charges for t in M[:-1]]
    assert all(len(b) == 1 for b in before)
    assert all(n > 0 for n in added) and all(set(b) < set(a) for b, a in zip(before, after))
    assert fidelity(vector(to_dense_mps(M, qs)), v0) == pytest.approx(1, abs=1e-14)
    for t in M[1:]:                                       # still right-orthonormal
        d = np.asarray(t.to_dense())
        np.testing.assert_allclose(np.einsum("asb,csb->ac", d, d.conj()), np.eye(d.shape[0]),
                                   atol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_expansion_unfreezes_a_product_state(backend):
    """Every hop spans three qubits, so TDVP from a product state cannot move, dense or blocks;
    with the expansion both follow exact evolution."""
    N, n = 3, 20
    W, qs = schwinger2f_mpo(N, X, MU), site_charges_2f(N)
    psi = strong_coupling_vacuum(N)
    exact = expm_multiply(-1j * DT * n * schwinger2f_sparse(N, X, MU), vector(psi))
    for symmetry in ("dense", "blocks"):
        assert fidelity(evolve(psi, W, qs, 64, 0, symmetry, n, backend=backend)[1],
                        vector(psi)) > 1 - 1e-12
        _, v = evolve(psi, W, qs, 64, 8, symmetry, n, backend=backend)
        assert fidelity(v, exact) > 1 - 1e-10


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["12site", "1site", "2site"])
def test_blocks_match_dense_with_expansion(method, backend):
    """A state with no isospin degeneracy (unequal flavour masses), bonds at a cap the evolution
    exceeds: the expansion's choices and the cuts back are the same in both storages."""
    N, cap = 4, 6
    W, qs = schwinger2f_mpo(N, X, (0.3, 0.5)), site_charges_2f(N)
    _, vac, _ = dmrg(W, chi=16, n_sweeps=20, tol=1e-12)
    st = to_numpy(vac)
    op, _ = compile2f(packet_opsum(N, "u+d", gaussian_profile(N, 1.5, 0.3, 1.0, 0, 3)), N)
    psi = [np.asarray(t) for t in to_numpy(compress(apply_mpo(st, op), cap, svd_tol=0)[0])]
    Md, vd = evolve(psi, W, qs, cap, 3, "dense", 4, method, backend)
    Mb, vb = evolve(psi, W, qs, cap, 3, "blocks", 4, method, backend)
    assert [int(t.shape[2]) for t in Md] == [int(t.shape[2]) for t in Mb]
    assert fidelity(vd, vb) > 1 - 1e-12
    assert fidelity(vd, evolve(psi, W, qs, cap, 0, "dense", 4, method)[1]) < 1 - 1e-8


def test_refusals():
    W = schwinger2f_mpo(2, X, MU)
    psi = strong_coupling_vacuum(2)
    with pytest.raises(ValueError, match="TDVPEvolver option"):
        tdvp2_evolve(psi, W, 4, DT, 1, TDVPOptions(expand=2))
    for bad in (-1, 2.5, True):
        with pytest.raises(ValueError, match="expand must be"):
            TDVPEvolver(psi, W, DT, 4, TDVPOptions(expand=bad))
    with pytest.raises(ValueError, match="open boundaries"):
        TDVPEvolver(psi, W, DT, 4, TDVPOptions(expand=2, graph="on"))
    with pytest.raises(ValueError, match="bond cap"):
        TDVPEvolver(psi, W, DT, None, TDVPOptions(expand=2)).run(1, "1site")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["12site", "1site"])
def test_expansion_keeps_the_cap_and_the_norm(method, backend):
    """Bonds end within the cap ('1site': at their sizes without the expansion), norm 1."""
    N, cap = 4, 6
    W, qs = schwinger2f_mpo(N, X, (0.3, 0.5)), site_charges_2f(N)
    _, vac, _ = dmrg(W, chi=16, n_sweeps=20, tol=1e-12)
    op, _ = compile2f(packet_opsum(N, "u+d", gaussian_profile(N, 1.5, 0.3, 1.0, 0, 3)), N)
    psi = [np.asarray(t)
           for t in to_numpy(compress(apply_mpo(to_numpy(vac), op), cap, svd_tol=0)[0])]
    for symmetry in ("dense", "blocks"):
        M0, _ = evolve(psi, W, qs, cap, 0, symmetry, 4, method, backend)
        prof = {}
        M, v = evolve(psi, W, qs, cap, 3, symmetry, 4, method, backend, prof)
        dims = [int(t.shape[2]) for t in M]
        assert max(dims) <= cap
        disc = [s["disc"] for s in prof["steps"]]
        assert len(disc) == 4 and min(disc) > 0       # the cut back's dropped weight is reported
        if method == "1site":
            assert dims == [int(t.shape[2]) for t in M0]
        assert np.vdot(v, v).real == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("method", ["12site", "2site"])
def test_blocks_report_the_dense_dropped_weight(method):
    N, cap = 4, 6
    W, qs = schwinger2f_mpo(N, X, (0.3, 0.5)), site_charges_2f(N)
    _, vac, _ = dmrg(W, chi=16, n_sweeps=20, tol=1e-12)
    op, _ = compile2f(packet_opsum(N, "u+d", gaussian_profile(N, 1.5, 0.3, 1.0, 0, 3)), N)
    psi = [np.asarray(t)
           for t in to_numpy(compress(apply_mpo(to_numpy(vac), op), cap, svd_tol=0)[0])]
    psi[0] = 2 * psi[0]                     # norm 2: relative and absolute dropped weights differ
    disc = {}
    for symmetry in ("dense", "blocks"):
        p, w, be = psi, W, "numpy"
        if symmetry == "blocks":
            p, w, be = to_block_mps(psi, qs, DENSE, sym=U1())[0], to_block_mpo(W, qs, DENSE, sym=U1())[0], BB
        prof = {}
        TDVPEvolver(p, w, DT, cap, TDVPOptions(backend=be, graph="off", expand=3, renorm=False,
                                               profile=prof)).run(3, method)
        disc[symmetry] = [s["disc"] for s in prof["steps"]]
    assert min(disc["dense"]) > 0
    np.testing.assert_allclose(disc["blocks"], disc["dense"], rtol=1e-8)


def _pin_evolver(cap, symmetry, backend):
    """Equal masses: isospin multiplets sit at the cut."""
    N = 4
    W, qs = schwinger2f_mpo(N, X, MU), site_charges_2f(N)
    _, vac, _ = dmrg(W, chi=32, n_sweeps=12, tol=1e-12)
    psi, be = [np.asarray(t) for t in to_numpy(vac)], backend
    if symmetry == "blocks":
        dbe = get_backend(backend)
        psi, W, be = to_block_mps(psi, qs, dbe, sym=U1())[0], to_block_mpo(W, qs, dbe, sym=U1())[0], BlockBackend(dbe)
    return TDVPEvolver(psi, W, DT, cap, TDVPOptions(backend=be, graph="off", svd_tol=1e-12,
                                                    degenerate_tol=1e-6, expand=3))


def _pins(cap, symmetry, backend, steps=3):
    ev, out = _pin_evolver(cap, symmetry, backend), []
    for _ in range(steps):
        ev.run(1, "12site")
        out.append(([int(t.shape[2]) for t in ev.mps], list(ev._pinned)))
    return out


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cap", [8, 11])
def test_expansion_cut_back_keeps_multiplets_whole(cap, backend):
    """The cut back keeps multiplets whole, a pinned bond goes back up to the cap when the
    spectrum allows, and blocks agree with dense."""
    d, b = _pins(cap, "dense", backend), _pins(cap, "blocks", backend)
    assert d == b
    dims, pins = d[-1]
    assert max(dims) <= cap
    if cap == 11:
        assert dims[3] == 10 and pins[3]        # held below the cap by a multiplet
    else:
        assert d[0][1][3] and d[0][0][3] == 6     # pinned below the cap by the two-site cut,
        assert d[1][0][3] == 8 and not any(d[1][1])   # back at the cap after one cut back


def test_concat_keeps_a_first_within_a_shared_sector():
    rng = np.random.default_rng(2)
    rest = (B.Leg((0, 1), (1, 1)), B.Leg((0, 1, 2), (1, 2, 1), -1))
    a = B.random((B.Leg((0, 1), (2, 2)),) + rest, 0, DENSE, rng)
    b = B.random((B.Leg((1,), (3,)),) + rest, 0, DENSE, rng)
    d = np.asarray(B.concat(a, b, 0).to_dense())
    np.testing.assert_array_equal(d[:4], a.to_dense())
    np.testing.assert_array_equal(d[4:], b.to_dense())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("symmetry", ["dense", "blocks"])
def test_one_site_expansion_keeps_pinned_bond_sizes(symmetry, backend):
    """A pin left by a '12site' step does not let a '1site' cut back grow the bond."""
    ev = _pin_evolver(8, symmetry, backend)
    ev.run(1, "12site")
    dims = [int(t.shape[2]) for t in ev.mps]
    assert ev._pinned[3] and dims[3] < 8
    ev.run(2, "1site")
    assert [int(t.shape[2]) for t in ev.mps] == dims


def test_expand_right_refuses_a_state_that_is_not_right_canonical(monkeypatch):
    monkeypatch.setenv("TNKIT_CHECK", "entry")
    N = 3
    W, qs = schwinger2f_mpo(N, X, MU), site_charges_2f(N)
    for symmetry in ("dense", "blocks"):
        M, be, Wb = strong_coupling_vacuum(N), DENSE, W
        if symmetry == "blocks":
            M, Wb, be = to_block_mps(M, qs, DENSE, sym=U1())[0], to_block_mpo(W, qs, DENSE, sym=U1())[0], BB
        M[2] = M[2] * 2
        L0, RN = _edge_envs(M, Wb, be, None, None)
        with pytest.raises(ValueError, match="right-canonical"):
            expand_right(list(M), Wb, 2, 8, be, L0, RN)
        monkeypatch.setenv("TNKIT_CHECK", "off")
        expand_right(list(M), Wb, 2, 8, be, L0, RN)
        monkeypatch.setenv("TNKIT_CHECK", "entry")
    W3 = schwinger2f_mpo(3, X, MU)
    _, vac, _ = dmrg(W3, chi=16, n_sweeps=10, tol=1e-12)
    M = [np.array(t) for t in to_numpy(vac)]
    j = next(j for j in range(1, len(M)) if M[j].shape[0] > 1)
    M[j][1] = M[j][0]                       # unit-norm rows, so the trace still equals a
    L0, RN = _edge_envs(M, W3, DENSE, None, None)
    with pytest.raises(ValueError, match="right-canonical"):
        expand_right(M, W3, 2, 8, DENSE, L0, RN)
