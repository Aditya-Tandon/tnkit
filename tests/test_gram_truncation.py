"""trunc='gram': agreement with a truncated SVD, exact isometries, end-to-end runs, the
single-precision, block and degenerate_tol refusals, the svd_tol warning and the torch output
kind."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tnkit import models as M  # noqa: E402
from tnkit.backend import NumpyBackend  # noqa: E402
from tnkit.dmrg import _truncate, dmrg  # noqa: E402
from tnkit.dmrg_excited import dmrg_excited  # noqa: E402
from tnkit.linalg import density_truncation, gram_truncation  # noqa: E402
from tnkit.tdvp import TDVPOptions, tdvp2_evolve  # noqa: E402
from tnkit.tdvp_evolver import TDVPEvolver  # noqa: E402

C128, C64, F64 = NumpyBackend(np.complex128), NumpyBackend(np.complex64), NumpyBackend(np.float64)
F32 = NumpyBackend(np.float32)


def spectral(m, n, sigma, rng, real=False):
    """m x n matrix with singular values `sigma` and random orthonormal bases."""
    r = len(sigma)
    def basis(k):
        Z = rng.standard_normal((k, r)) + (0 if real else 1j * rng.standard_normal((k, r)))
        return np.linalg.qr(Z)[0]
    return (basis(m) * sigma) @ basis(n).conj().T


def svd_cut(A, k):
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vh[:k], float(np.sum(S[k:] ** 2))


CASES = [(40, 40, "geometric"), (60, 24, "geometric"), (24, 60, "geometric"),
         (50, 50, "random"), (36, 64, "deficient")]


def matrix(m, n, kind, rng, real=False):
    r = min(m, n)
    if kind == "geometric":                     # 1 .. 1e-8, the production regime
        return spectral(m, n, np.geomspace(1, 1e-8, r), rng, real)
    if kind == "deficient":                     # rank 10 and exact zero rows
        A = spectral(m, n, np.geomspace(1, 1e-3, 10), rng, real)
        A[::3] = 0
        return A
    A = rng.standard_normal((m, n))
    return A if real else A + 1j * rng.standard_normal((m, n))


@pytest.mark.parametrize("m,n,kind", CASES)
@pytest.mark.parametrize("iso", ["left", "right"])
@pytest.mark.parametrize("real", [False, True])
def test_matches_truncated_svd(m, n, kind, iso, real):
    rng = np.random.default_rng(m * n)
    A = matrix(m, n, kind, rng, real)
    be = F64 if real else C128
    k = min(m, n) // 2
    U, S, Vh, disc = gram_truncation(be.array(A), k, be, iso)
    Ur, Sr, Vr, dr = svd_cut(A, k)
    nrm2 = np.linalg.norm(A) ** 2
    assert abs(disc - dr) <= 1e-12 * nrm2
    above = Sr ** 2 > 1e3 * np.finfo(float).eps * Sr[0] ** 2
    assert np.abs(np.sort(S)[::-1][above] - Sr[above]).max() <= 1e-10 * Sr[0]
    tx, tr = (U * S) @ Vh, (Ur * Sr) @ Vr
    assert abs(np.linalg.norm(A - tx) ** 2 - dr) <= 1e-12 * nrm2
    if kind != "deficient":                     # the kept subspace is unique
        fid = abs(np.vdot(tr, tx)) / (np.linalg.norm(tr) * np.linalg.norm(tx))
        assert 1 - fid <= 1e-10


@pytest.mark.parametrize("m,n,kind", CASES)
def test_isometry_and_carried_factor(m, n, kind):
    """The canonical factor is isometric to rounding, and S Vh (U S) is the carried
    projection U^H A (A V) itself, not a quotient by eigenvalue-derived singular values."""
    A = matrix(m, n, kind, np.random.default_rng(7))
    k = min(m, n) // 2
    U, S, Vh, _ = gram_truncation(A, k, C128, "left")
    assert np.abs(U.conj().T @ U - np.eye(k)).max() < 1e-13
    assert np.abs(S[:, None] * Vh - U.conj().T @ A).max() < 1e-12 * np.linalg.norm(A)
    U, S, Vh, _ = gram_truncation(A, k, C128, "right")
    assert np.abs(Vh @ Vh.conj().T - np.eye(k)).max() < 1e-13
    assert np.abs(U * S - A @ Vh.conj().T).max() < 1e-12 * np.linalg.norm(A)
    assert np.isfinite(U).all() and np.isfinite(Vh).all()


def test_density_route_breaks_the_isometry():
    """Why 'density' is deprecated: its Vh = U^H A / S is not orthonormal for small S."""
    A = spectral(64, 64, np.geomspace(1, 1e-10, 64), np.random.default_rng(3))
    _, _, Vh, _ = density_truncation(A, 48)
    assert np.abs(Vh @ Vh.conj().T - np.eye(Vh.shape[0])).max() > 1e-6
    _, _, Vh, _ = gram_truncation(A, 48, C128, "right")
    assert np.abs(Vh @ Vh.conj().T - np.eye(48)).max() < 1e-13


@pytest.mark.parametrize("be", [F64, C128])
@pytest.mark.parametrize("iso", ["left", "right"])
def test_zero_rows_stay_finite(be, iso):
    A = np.zeros((6, 8))
    A[:3, :3] = np.random.default_rng(0).standard_normal((3, 3))
    U, S, Vh, _ = gram_truncation(be.array(A), 6, be, iso)
    assert np.isfinite(U).all() and np.isfinite(Vh).all()


@pytest.mark.parametrize("be", [C64, F32])
def test_single_precision_is_refused(be):
    """Gram squares the condition number: in single precision the discarded weight is wrong by
    ~ eps32 ||A||_F^2 ~ 1e-7, far above a 1e-8 budget, so the kernel refuses rather than lies."""
    A = spectral(40, 40, np.geomspace(1, 1e-8, 40), np.random.default_rng(5), real=True)
    with pytest.raises(ValueError, match="double precision"):
        gram_truncation(be.array(A), 20, be, "left")


def test_single_precision_is_refused_on_torch():
    torch = pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    be = TorchBackend("cpu", torch.complex64)
    with pytest.raises(ValueError, match="double precision"):
        gram_truncation(be.array(np.eye(8)), 4, be, "right")


def test_invalid_iso_raises():
    with pytest.raises(ValueError, match="iso"):
        gram_truncation(C128.array(np.eye(8)), 4, C128, "up")


def test_dispatch():
    A = np.random.default_rng(1).standard_normal((8, 8)) + 0j
    with pytest.raises(ValueError, match="iso"):
        _truncate(A, 4, "gram", C128)
    with pytest.raises(ValueError, match="iso"):
        _truncate(A, 4, "gram", C128, iso="up")
    for iso in ("left", "right"):
        assert _truncate(A, 4, "gram", C128, iso=iso)[0].shape[1] == 4


def test_density_deprecation_is_raised_once_at_the_entry_points():
    """One warning per call, attributed to the caller, not one per bond step."""
    mpo = M.tfim_mpo(8, 1.0)
    with pytest.warns(DeprecationWarning, match="trunc='svd'") as w:
        dmrg(mpo, chi=4, n_sweeps=2, trunc="density", backend="numpy")
    assert len(w) == 1 and w[0].filename == __file__
    _, psi, _ = dmrg(mpo, chi=4, n_sweeps=2, backend="numpy")
    with pytest.warns(DeprecationWarning, match="trunc='svd'") as w:
        dmrg_excited(mpo, 4, prev_states=[psi], n_sweeps=1, trunc="density", backend="numpy")
    assert len(w) == 1 and w[0].filename == __file__
    with pytest.warns(DeprecationWarning, match="trunc='svd'") as w:
        tdvp2_evolve(psi, mpo, 4, 0.05, 2, trunc="density", backend="numpy")
    assert len(w) == 1 and w[0].filename == __file__


def test_backends_agree():
    torch = pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    be = TorchBackend("cpu", torch.complex128)
    A = spectral(30, 20, np.geomspace(1, 1e-8, 20), np.random.default_rng(4))
    for iso in ("left", "right"):
        a = gram_truncation(A, 10, C128, iso)
        b = gram_truncation(be.array(A), 10, be, iso)
        ra, rb = (a[0] * a[1]) @ a[2], be.to_numpy((b[0] * b[1]) @ b[2])
        assert np.abs(ra - rb).max() < 1e-12 and abs(a[3] - b[3]) < 1e-14


def _canon_right(mps):
    return max(np.abs((B.reshape(B.shape[0], -1) @ B.reshape(B.shape[0], -1).conj().T)
                      - np.eye(B.shape[0])).max() for B in mps[1:])


@pytest.mark.parametrize("sites", [1, 2])
def test_dmrg_gram_equals_svd(sites):
    mpo = M.tfim_mpo(16, 1.0)
    kw = dict(chi=8, n_sweeps=6, tol=0.0, rtol=0.0, sites=sites, backend="numpy", seed=0)
    Es, _, _ = dmrg(mpo, trunc="svd", **kw)
    Eg, mps, _ = dmrg(mpo, trunc="gram", **kw)
    assert abs(Eg - Es) < 1e-10 * abs(Es)
    assert _canon_right([np.asarray(t) for t in mps]) < 1e-12


@pytest.mark.parametrize("sites", [1, 2])
def test_dmrg_gram_keeps_the_canonical_form_in_the_sweep(sites, monkeypatch):
    """Every tensor that enters an environment update is an exact isometry: left sites A
    (A^+ A = 1), right sites B (B B^+ = 1). The energy cannot see which factor a cut keeps."""
    import tnkit._dmrg as dm
    seen = {"left": 0, "right": 0}

    def wrap(name, side, gram):
        orig = getattr(dm, name)

        def spy(env, T, W, be=None):
            Tn = np.asarray(T)
            m = Tn.reshape(-1, Tn.shape[2]) if side == "left" else Tn.reshape(Tn.shape[0], -1)
            assert np.abs(gram(m) - np.eye(m.shape[1 if side == "left" else 0])).max() < 1e-12
            seen[side] += 1
            return orig(env, T, W, be)
        monkeypatch.setattr(dm, name, spy)

    wrap("update_left_env", "left", lambda m: m.conj().T @ m)
    wrap("update_right_env", "right", lambda m: m @ m.conj().T)
    dmrg(M.tfim_mpo(12, 1.0), chi=6, n_sweeps=4, tol=0.0, rtol=0.0, sites=sites,
         trunc="gram", backend="numpy", seed=0)
    assert seen["left"] > 20 and seen["right"] > 20


def test_tdvp2_gram_equals_svd():
    from tnkit.observables import local_expectations
    _, psi, _ = dmrg(M.tfim_mpo(12, 0.5), chi=8, n_sweeps=4, backend="numpy", seed=0)
    mpo = M.tfim_mpo(12, 1.0, 0.3)
    out = {}
    for trunc in ("svd", "gram"):
        fin, _ = tdvp2_evolve(psi, mpo, 16, 0.05, 20, trunc=trunc, svd_tol=1e-10,
                              backend="numpy")
        out[trunc] = [np.asarray(t) for t in fin]
    za, zb = (np.real(local_expectations(out[t], M.SZ)) for t in ("svd", "gram"))
    assert np.abs(za - zb).max() < 1e-9
    assert _canon_right(out["gram"]) < 1e-12


def test_dmrg_excited_gram_equals_svd():
    mpo = M.tfim_mpo(12, 1.0)
    _, psi0, _ = dmrg(mpo, chi=8, n_sweeps=6, tol=0.0, rtol=0.0, backend="numpy", seed=0)
    kw = dict(prev_states=[psi0], penalty=10.0, n_sweeps=8, tol=0.0, backend="numpy", seed=1)
    for deflate in ("penalty", "project"):
        Es, _, _ = dmrg_excited(mpo, 8, trunc="svd", deflate=deflate, **kw)
        Eg, mps, _ = dmrg_excited(mpo, 8, trunc="gram", deflate=deflate, **kw)
        assert abs(Eg - Es) < 1e-12 * abs(Es)
        assert _canon_right([np.asarray(t) for t in mps]) < 1e-12


def test_torch_outputs_convert_to_numpy():
    """The returned tensors must match 'svd' in kind: no lazily conjugated view, so `.numpy()`
    works on every site."""
    torch = pytest.importorskip("torch")
    mpo = M.tfim_mpo(10, 1.0)
    kw = dict(chi=8, backend="torch-cpu", seed=0)
    E0, psi, _ = dmrg(mpo, n_sweeps=4, trunc="gram", **kw)
    fin, _ = tdvp2_evolve(psi, M.tfim_mpo(10, 1.0, 0.3), 8, 0.05, 4, trunc="gram", svd_tol=1e-10,
                          backend="torch-cpu")
    Ex, exc, _ = dmrg_excited(mpo, 8, prev_states=[psi], penalty=10.0, n_sweeps=4, trunc="gram",
                              backend="torch-cpu")
    for mps in (psi, fin, exc):
        assert not any(t.is_conj() for t in mps)
        assert all(isinstance(t.numpy(), np.ndarray) for t in mps)


def test_discarded_weight_is_nonnegative_when_rank_deficient():
    """At full chi the dropped Gram eigenvalues of a rank-deficient matrix are rounding noise,
    some negative; the reported weight must not go below zero."""
    A = matrix(36, 64, "deficient", np.random.default_rng(2))
    for iso in ("left", "right"):
        assert gram_truncation(A, 30, C128, iso)[3] >= 0.0


def test_degenerate_tol_is_refused():
    _, psi, _ = dmrg(M.tfim_mpo(6, 1.0), chi=4, n_sweeps=2, backend="numpy", seed=0)
    o = TDVPOptions(trunc="gram", degenerate_tol=1e-6, backend="numpy", graph="off")
    with pytest.raises(ValueError, match="degenerate_tol needs trunc='svd'"):
        tdvp2_evolve(psi, M.tfim_mpo(6, 1.0), 4, 0.05, 1, o)
    with pytest.raises(ValueError, match="degenerate_tol needs trunc='svd'"):
        TDVPEvolver(psi, M.tfim_mpo(6, 1.0), 0.05, 4, o)


def _xxz_blocks(N=6):
    from tnkit.backend import get_backend
    from tnkit.block_backend import BlockBackend, random_block_mps, to_block_mpo
    from tnkit.opsum import OpSum, compile_mpo, spin_half
    from tnkit.symmetry import U1
    os_ = OpSum()
    for i in range(N - 1):
        os_ += (0.5, "Sp", i, "Sm", i + 1)
        os_ += (0.5, "Sm", i, "Sp", i + 1)
        os_ += (0.25, "Z", i, "Z", i + 1)
    W = [np.ascontiguousarray(w.real) for w in compile_mpo(os_, [spin_half()] * N)[0]]
    real = get_backend("numpy-real")
    Wb, _ = to_block_mpo(W, [[1, -1]] * N, real, sym=U1())
    return Wb, random_block_mps(Wb, 4, real, 0, np.random.default_rng(0)), BlockBackend(real)


def test_block_tensors_are_refused():
    Wb, Mb, bb = _xxz_blocks()
    with pytest.raises(ValueError, match="trunc='svd'"):
        dmrg(Wb, 4, charge=0, trunc="gram")
    o = TDVPOptions(backend=bb, trunc="gram", graph="off")
    with pytest.raises(ValueError, match="trunc='svd'"):
        tdvp2_evolve(list(Mb), Wb, 4, 0.05, 1, o)
    with pytest.raises(ValueError, match="trunc='svd'"):
        TDVPEvolver(list(Mb), Wb, 0.05, 4, o)              # at construction, before any step


def test_svd_tol_below_1e_12_warns():
    _, psi, _ = dmrg(M.tfim_mpo(6, 1.0), chi=4, n_sweeps=2, backend="numpy", seed=0)
    mpo = M.tfim_mpo(6, 1.0, 0.3)
    for tol in (1e-13, 1e-12, 0.0):
        o = TDVPOptions(trunc="gram", svd_tol=tol, backend="numpy", graph="off")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tdvp2_evolve(psi, mpo, 4, 0.05, 1, o)
            TDVPEvolver(psi, mpo, 0.05, 4, o)
        w = [x for x in w if "noise floor" in str(x.message)]
        assert len(w) == (2 if tol == 1e-13 else 0)
        assert all(x.category is UserWarning and x.filename == __file__ for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tdvp2_evolve(psi, mpo, 4, 0.05, 1, svd_tol=1e-13, backend="numpy")
    assert not [x for x in w if "noise floor" in str(x.message)]


@pytest.mark.parametrize("kw", [dict(svd_tol=1e-10), dict(svd_tol=1e-10, expand=2)])
def test_evolver_12site_gram_equals_svd(kw):
    from tnkit.observables import local_expectations
    _, psi, _ = dmrg(M.tfim_mpo(12, 0.5), chi=8, n_sweeps=4, backend="numpy", seed=0)
    mpo = M.tfim_mpo(12, 1.0, 0.3)
    out = {}
    for trunc in ("svd", "gram"):
        ev = TDVPEvolver(psi, mpo, 0.05, 16, TDVPOptions(trunc=trunc, backend="numpy",
                                                          graph="off", **kw))
        ev.run(12, "12site")
        out[trunc] = [np.asarray(t) for t in ev.mps]
    za, zb = (np.real(local_expectations(out[t], M.SZ)) for t in ("svd", "gram"))
    assert np.abs(za - zb).max() < 1e-12                   # measured 1.6e-14
    assert _canon_right(out["gram"]) < 1e-12
