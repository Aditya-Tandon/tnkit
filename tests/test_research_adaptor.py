"""The field-theory helpers on the backend adaptor: torch runs agree with numpy, and the numpy
default reproduces the plain-numpy recipe bit for bit."""
from __future__ import annotations

import numpy as np
import pytest

from tnkit import SX, SZ
from tnkit.backend import NumpyBackend, PlainEinsumNumpyBackend, get_backend
from tnkit.idmrg import tfim_bulk_mpo
from tnkit.vumps import _left_environment, _right_environment, vumps
from tnkit import window as win

torch = pytest.importorskip("torch", reason="torch extra not installed")


@pytest.fixture(scope="module")
def bg():
    gs = vumps(1.5, 6, n_iter=60, tol=1e-10)
    be = get_backend("numpy")
    W = be.array(tfim_bulk_mpo(1.5))
    FL, _ = _left_environment(gs["AL"], gs["C"], W, be, env_tol=1e-10)
    FR, _ = _right_environment(gs["AR"], gs["C"], W, be, env_tol=1e-10)
    return gs, W, FL, FR


def test_plain_backend_keeps_np_einsum_layouts():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 3, 5)), rng.standard_normal((5, 2))
    ref = np.einsum("asx,xb->asb", a, b, optimize=True)
    got = PlainEinsumNumpyBackend().einsum("asx,xb->asb", a, b)
    assert np.array_equal(got, ref) and got.strides == ref.strides
    assert NumpyBackend().einsum("asx,xb->asb", a, b).strides != ref.strides   # what plans do


def _pair_blocks_plain_numpy(sep, n, npr, dmax):
    """The same blocks with plain numpy throughout: np.array, broadcast copies, np.einsum."""
    E = lambda sub, *ops: np.einsum(sub, *ops, optimize=True)            # noqa: E731
    L, L2, j0 = sep.ell, sep.ell ** 2, sep._anchor()
    mp, npp, mu, nu = sep._batch_index()
    Bst = np.array(sep.Bs)
    H = np.zeros((L2, L2), dtype=complex)
    for delta in range(-dmax, dmax + 1):
        pa1, pa2, pb1, pb2 = j0 + delta, j0 + delta + npr, j0, j0 + n
        Lh = np.broadcast_to(sep.FL, (L ** 4,) + sep.FL.shape).copy()
        Ln = np.broadcast_to(sep._eye, (L ** 4,) + sep._eye.shape).copy()
        for k in range(sep.NS):
            bra = Bst[mp] if k == pa1 else Bst[npp] if k == pa2 else (sep.AL if k < pa2 else sep.AR)
            ket = Bst[mu] if k == pb1 else Bst[nu] if k == pb2 else (sep.AL if k < pb2 else sep.AR)
            bb, kk = ("b" if bra.ndim == 4 else ""), ("b" if ket.ndim == 4 else "")
            t = E(f"bpwq,{bb}psd->bwqsd", Lh, bra.conj())
            t = E("bwqsd,wrsy->bqdry", t, sep.W)
            Lh = E(f"bqdry,{kk}qye->bdre", t, ket)
            Ln = E(f"bxsc,{bb}xsy->bcy", E(f"bax,{kk}asc->bxsc", Ln, ket), bra.conj())
        H += (E("bakc,akc->b", Lh, sep.FR) - sep.e_vac * np.einsum("baa->b", Ln)).reshape(L2, L2)
    return H


def test_pair_blocks_match_the_plain_numpy_recipe_exactly(bg):
    from tnkit.ift.two_particle import SeparationBasis
    gs, W, FL, FR = bg
    D = gs["AL"].shape[0]
    rng = np.random.default_rng(3)
    orbs = [rng.standard_normal(D * D) + 1j * rng.standard_normal(D * D) for _ in range(2)]
    sep = SeparationBasis(gs["AL"], gs["AR"], gs["C"], W, FL, FR, orbs, 16, dmax=1, margin=2)
    assert np.array_equal(sep.pair_blocks(1, 2)[0], _pair_blocks_plain_numpy(sep, 1, 2, 1))


def test_separation_basis_needs_double_precision(bg):
    from tnkit.backend import TorchBackend
    from tnkit.ift.two_particle import SeparationBasis
    gs, W, FL, FR = bg
    with pytest.raises(TypeError, match="complex128"):
        SeparationBasis(gs["AL"], gs["AR"], gs["C"], W, FL, FR, [], 10,
                        be=TorchBackend("cpu", torch.complex64))


def test_separation_basis_on_torch_matches_numpy(bg):
    from tnkit.ift.two_particle import SeparationBasis, compute_pair_blocks
    gs, W, FL, FR = bg
    D = gs["AL"].shape[0]
    rng = np.random.default_rng(7)
    orbs = [rng.standard_normal(D * D) + 1j * rng.standard_normal(D * D) for _ in range(2)]
    args = (torch.as_tensor(gs["AL"]), gs["AR"], gs["C"], W, FL, FR, orbs, 20)
    sn = SeparationBasis(*args, dmax=2, margin=3)
    st = SeparationBasis(*args, dmax=2, margin=3, be=get_backend("torch-cpu"))
    assert abs(sn.e_vac - st.e_vac) < 1e-12
    Hn, Nn, _ = compute_pair_blocks(sn, 3, verbose=False)
    Ht, Nt, _ = compute_pair_blocks(st, 3, verbose=False)
    for k in Hn:
        assert isinstance(Ht[k], np.ndarray)
        assert np.allclose(Hn[k], Ht[k], rtol=0, atol=1e-10 * np.abs(Hn[k]).max())
        assert np.allclose(Nn[k], Nt[k], rtol=0, atol=1e-12 * max(1.0, np.abs(Nn[k]).max()))
    Hc, Nc = sn.cross_blocks(1)
    Hct, Nct = st.cross_blocks(1)
    assert np.allclose(Hc, Hct, atol=1e-10 * np.abs(Hc).max()) and np.allclose(Nc, Nct, atol=1e-12)


def test_window_observables_on_torch_match_numpy(bg):
    gs = bg[0]
    w, FL, FR, W, _ = win.build_window(1.5, 6, 4, 4, op=SZ, gs=gs)
    wt = [torch.as_tensor(m) for m in w]
    for f, ops in ((win.window_expectations, (SZ,)),
                   (win.window_neighbor_correlations, (SX, SX))):
        a, b = f(w, *ops), f(wt, *ops)
        assert isinstance(b, np.ndarray) and np.allclose(a, b, rtol=0, atol=1e-13)
    g = win.grow_window(wt, gs["AL"], gs["AR"], 1, 2)
    assert len(g) == len(w) + 3
    assert all(isinstance(t, torch.Tensor) and t.dtype == torch.complex128 for t in g)
    assert win.window_neighbor_correlations(w[:1], SX, SX).shape == (0,)
