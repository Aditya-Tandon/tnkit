"""`mpo_sq_expectation` and `mpo_variance` against dense linear algebra at small N."""
import numpy as np
import pytest

from tnkit import dmrg
from tnkit.dmrg_excited import mpo_expectation, mpo_sq_expectation, mpo_variance
from tnkit.models import mpo_to_dense, tfim_mpo
from tnkit.models_lgt import schwinger_dense, schwinger_mpo

N = 8


def _rand_mps(n, chi, seed):
    rng = np.random.default_rng(seed)
    dims = [1] + [min(chi, 2 ** min(i, n - i)) for i in range(1, n)] + [1]
    return [rng.normal(size=(dims[i], 2, dims[i + 1]))
            + 1j * rng.normal(size=(dims[i], 2, dims[i + 1])) for i in range(n)]


def _dense(mps):
    v = np.ones((1, 1))
    for A in mps:
        v = np.einsum("xa,asb->xsb", v, A).reshape(-1, A.shape[2])
    return v[:, 0] / np.linalg.norm(v[:, 0])


def _mps_from_vector(v, n):
    """Exact MPS of a state vector (site 0 most significant), by untruncated SVDs."""
    mps, rest, left = [], v.reshape(1, -1), 1
    for _ in range(n - 1):
        U, S, Vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
        mps.append(U.reshape(left, 2, -1))
        rest, left = S[:, None] * Vh, U.shape[1]
    mps.append(rest.reshape(left, 2, 1))
    return mps


def _models():
    rng = np.random.default_rng(7)
    x, mu, ell = 1.3, rng.normal(size=N), rng.normal(size=N - 1)
    W_tfim = tfim_mpo(N, 1.2, 0.3)
    return {"tfim": (W_tfim, mpo_to_dense(W_tfim)),
            "schwinger": (schwinger_mpo(N, x, mu, ell, 2.0), schwinger_dense(N, x, mu, ell, 2.0))}


MODELS = _models()


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("seed", [1, 2])
def test_matches_dense_on_random_mps(model, seed):
    W, H = MODELS[model]
    psi = _rand_mps(N, 6, seed)
    v = _dense(psi)
    Hv = H @ v
    e1, e2 = np.vdot(v, Hv), np.vdot(Hv, Hv)
    assert abs(mpo_sq_expectation(psi, W) - e2) <= 1e-12 * abs(e2)
    var, floor = mpo_variance(psi, W)
    ref = (e2 - e1 * e1).real
    assert ref > 0.1                                   # a random state is far from an eigenstate
    assert abs(var - ref) <= 1e-11 * abs(e2)
    assert floor == pytest.approx(np.finfo(float).eps * abs(e2), rel=1e-9, abs=0)


def test_sq_expectation_on_a_complex_nonsymmetric_mpo():
    """Pins the bra/ket orientation of the MPO's physical legs, which a real symmetric H cannot."""
    rng = np.random.default_rng(11)
    D = [1] + [3] * (N - 1) + [1]
    W = [rng.normal(size=(D[i], D[i + 1], 2, 2)) + 1j * rng.normal(size=(D[i], D[i + 1], 2, 2))
         for i in range(N)]
    H = mpo_to_dense(W)
    assert np.linalg.norm(H - H.T) > 1 and np.linalg.norm(H - H.conj().T) > 1
    psi = _rand_mps(N, 6, 4)
    v = _dense(psi)
    ref = np.vdot(v, H @ (H @ v))
    assert abs(mpo_sq_expectation(psi, W) - ref) <= 1e-12 * abs(ref)


def test_floor_scales_with_the_second_moment_down_to_one():
    """For <W^2> < 1 the floor is eps, not eps * <W^2>."""
    W = [A * (0.1 if i == 0 else 1.0) for i, A in enumerate(MODELS["tfim"][0])]
    psi = _rand_mps(N, 6, 5)
    var, floor = mpo_variance(psi, W)
    assert mpo_sq_expectation(psi, W).real < 1
    assert floor == np.finfo(float).eps


@pytest.mark.parametrize("model", list(MODELS))
def test_zero_on_an_exact_eigenstate(model):
    W, H = MODELS[model]
    w, V = np.linalg.eigh(H)
    for k in (0, 5):
        psi = _mps_from_vector(V[:, k], N)
        var, floor = mpo_variance(psi, W)
        assert abs(mpo_expectation(psi, W) - w[k]) <= 1e-12 * max(1.0, abs(w[k]))
        assert var <= 100 * floor


@pytest.mark.parametrize("model", list(MODELS))
def test_weinstein_bound(model):
    """Some exact eigenvalue lies within sqrt(var) of <H>, for converged and unconverged states."""
    W, H = MODELS[model]
    spec = np.linalg.eigvalsh(H)
    states = [_rand_mps(N, 4, s) for s in range(3)]
    states += [dmrg(W, chi, n_sweeps=sw, backend="numpy")[1] for chi, sw in ((2, 2), (16, 10))]
    for psi in states:
        e = mpo_expectation(psi, W).real
        var, floor = mpo_variance(psi, W)
        assert np.min(np.abs(spec - e)) <= np.sqrt(var + floor) * (1 + 1e-9)


@pytest.mark.parametrize("model", list(MODELS))
def test_torch_cpu_equals_numpy(model):
    torch = pytest.importorskip("torch", reason="torch extra not installed")
    W, _ = MODELS[model]
    psi = _rand_mps(N, 6, 3)
    var_np, floor_np = mpo_variance(psi, W)
    var_t, floor_t = mpo_variance([torch.as_tensor(A) for A in psi], W)
    assert abs(var_t - var_np) <= 1e-12 * max(1.0, var_np)
    assert floor_t == pytest.approx(floor_np, rel=1e-12, abs=0)
