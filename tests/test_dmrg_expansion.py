"""DMRG subspace expansion (`expand`) and one-site sweeps (`sites=1`), dense and block."""
import numpy as np
import pytest

from tnkit import blocks as B
from tnkit._dmrg import _BLOCK_EXPAND
from tnkit.api import DMRGOptions, dmrg
from tnkit.block_backend import BlockBackend, to_block_mpo, to_block_mps, to_dense_mps
from tnkit.expansion import enriched_split
from tnkit.models import tfim_mpo
from tnkit.symmetry import U1

from test_block_dmrg import REAL, SZ2, ground, sz2, vector, xxz_opsum_mpo, xxz_sparse

KW = dict(n_sweeps=20, tol=1e-13, eig_tol=1e-12)
EXP = dict(expand=4, expand_alpha=3e-3)


def neel(N, be):
    """The Neel product state as block tensors (bond dimension 1)."""
    up, dn = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    dense = [(up if i % 2 == 0 else dn).reshape(1, 2, 1) for i in range(N)]
    return to_block_mps(dense, [SZ2] * N, be, sym=U1())[0]


def test_off_by_default_on_dense_and_alpha_ignored_when_off():
    o = DMRGOptions()
    assert (o.sites, o.expand) == (2, None)
    W = tfim_mpo(10, 0.9)
    E0, m0, i0 = dmrg(W, 8, n_sweeps=3)
    E1, m1, i1 = dmrg(W, 8, n_sweeps=3, expand=0, expand_alpha=0.7, expand_decay=0.1)
    assert E0 == E1 and all(np.array_equal(a, b) for a, b in zip(m0, m1))
    assert i0["expand"] == 0 and i0["expand_alphas"] == [0.0] * i0["sweeps"]


@pytest.mark.parametrize("n_sweeps", [1, 2, 3, 4, 5, 6, 8])
def test_block_default_expands_and_leaves_two_plain_sweeps(n_sweeps):
    N = 6
    W = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())[0]
    kw = dict(n_sweeps=n_sweeps, tol=0.0, rtol=0.0)
    E, m, info = dmrg(W, 4, charge=0, **kw)
    k = min(_BLOCK_EXPAND, max(0, n_sweeps - 2))
    assert info["expand"] == k and info["sweeps"] == n_sweeps
    assert [a > 0 for a in info["expand_alphas"]] == [s < k for s in range(n_sweeps)]
    E1, m1, _ = dmrg(W, 4, charge=0, expand=k, **kw)
    assert E1 == E and all(np.array_equal(a.data, b.data) for a, b in zip(m, m1))
    assert dmrg(W, 4, charge=0, expand=0, **kw)[2]["expand"] == 0


@pytest.mark.parametrize("opts", [EXP, dict(sites=1), dict(sites=1, **EXP)])
@pytest.mark.parametrize("block", [False, True])
def test_against_ed(opts, block):
    N = 10
    W = xxz_opsum_mpo(N, 0.5)
    E_ed = ground(xxz_sparse(N, 0.5), sz2(N) == 0)
    if block:
        W = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())[0]
        E, mps, info = dmrg(W, 32, charge=0, **KW, **opts)
    else:
        E, mps, info = dmrg(W, 32, **KW, **opts)
    assert abs(E - E_ed) < 1e-10
    v = vector(to_dense_mps(mps, [SZ2] * N) if block else mps)
    assert abs(np.linalg.norm(v) - 1) < 1e-10
    Hv = xxz_sparse(N, 0.5) @ v
    assert abs(np.vdot(v, Hv) / np.vdot(v, v) - E_ed) < 1e-10
    if opts.get("expand"):
        a = info["expand_alphas"]
        assert a[:4] == pytest.approx([3e-3 * 0.3 ** k for k in range(4)]) and a[-1] == 0.0


@pytest.mark.parametrize("backend", ["numpy-real", "numpy"])
def test_one_site_grows_from_a_product_state(backend):
    """One-site DMRG keeps every bond at dimension 1 and the Neel sectors without expansion;
    with it, the bonds grow, fill the other charge sectors, and reach the exact energy."""
    N = 10
    W = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())[0]
    E_ed = ground(xxz_sparse(N, 1.0), sz2(N) == 0)
    kw = dict(KW, sites=1, backend=backend)
    E0, m0, _ = dmrg(W, 32, init_mps=neel(N, REAL), expand=0, **kw)
    assert max(t.shape[2] for t in m0) == 1 and E0 > E_ed + 0.5
    E1, m1, _ = dmrg(W, 32, init_mps=neel(N, REAL), **kw, **EXP)
    assert max(t.shape[2] for t in m1) == 32 and abs(E1 - E_ed) < 1e-10
    assert len(m1[N // 2 - 1].legs[2].charges) > 1


def test_enriched_split_keeps_the_state_and_adds_sectors():
    """With room for every state, A C reproduces x exactly; the added states come from the
    expansion term's own sectors, which x lacks (rows of charge +-3)."""
    be = BlockBackend(REAL)
    rng = np.random.default_rng(1)
    phys = B.Leg((-1, 1), (1, 1))
    a = B.Leg((-2, 0, 2), (1, 2, 1))
    b = B.Leg((0,), (3,), -1)
    x = B.random((a, phys, phys, b), 0, REAL, rng)       # charge 0: only (s1, s2) = (+-, -+)
    w = B.Leg((-2, 0, 2), (1, 1, 1))
    extra = B.random((a, phys, w, phys, b), 0, REAL, rng)
    A, C, disc = enriched_split(x, extra, 0.1, 100, be, True)
    xa = be.einsum("abk,kcd->abcd", A, C)
    assert np.allclose(xa.to_dense(), x.to_dense(), atol=1e-12) and disc < 1e-12
    plain = set(enriched_split(x, extra, 0.0, 100, be, True)[0].legs[2].charges)
    assert set(A.legs[2].charges) > plain
    B_, C_, _ = enriched_split(x, extra, 0.1, 100, be, False)
    assert np.allclose(be.einsum("abk,kcd->abcd", C_, B_).to_dense(), x.to_dense(), atol=1e-12)


@pytest.mark.parametrize("right", [True, False])
def test_enriched_split_keeps_the_weights_of_x(right):
    """The expansion term competes only for states outside x's span: with exactly rank(x)
    states allowed and P weaker than x's smallest singular value, x is kept whole. The kept
    legs (the first two, or the last two) are wider than rank(x), so P has room to compete."""
    be = BlockBackend(REAL)
    rng = np.random.default_rng(2)
    phys = B.Leg((-1, 1), (1, 1))
    wide, thin = B.Leg((-2, 0, 2), (1, 2, 1)), B.Leg((0,), (1,))
    a, b = (wide, thin.flip()) if right else (thin, wide.flip())
    x = B.random((a, phys, phys, b), 0, REAL, rng)
    extra = B.random((a, phys, B.Leg((-2, 0, 2), (1, 1, 1)), phys, b), 0, REAL, rng)
    S = B.truncated_svd(x.fuse(((0, 1), (2, 3)))[0], 100)[1]
    alpha = 0.3 * float(S.min()) / float(be.norm(extra))
    _, _, disc = enriched_split(x, extra, alpha, len(S), be, right)
    assert disc < 1e-12 * float(be.norm(x)) ** 2


def test_block_and_dense_agree_with_expansion():
    N = 12
    W = xxz_opsum_mpo(N, 1.0)
    Wb = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())[0]
    E_d = dmrg(W, 24, **KW, **EXP)[0]
    E_b = dmrg(Wb, 24, charge=0, **KW, **EXP)[0]
    assert abs(E_b - E_d) < 1e-10


def test_refusals():
    W = tfim_mpo(6, 1.0)
    with pytest.raises(ValueError, match="sites"):
        dmrg(W, 4, sites=3)
    with pytest.raises(ValueError, match="unbiased"):
        dmrg(W, 4, n_sweeps=4, expand=4)
    with pytest.raises(ValueError, match="expand_alpha.*pass expand=0"):
        dmrg(W, 4, expand=2, expand_alpha=0.0)
    for bad in (2.5, True, "2"):
        with pytest.raises(TypeError, match="expand must be an int"):
            dmrg(W, 4, expand=bad)
    with pytest.raises(ValueError, match="lanczos-mixed"):
        dmrg(W, 4, sites=1, eig="lanczos-mixed")
