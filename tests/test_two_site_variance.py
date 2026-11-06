"""`two_site_variance` against dense projectors built from the null spaces at small N."""
import numpy as np
import pytest

from tnkit.dmrg_excited import mpo_variance, two_site_variance
from tnkit.models import mpo_to_dense, tfim_mpo
from tnkit.models_lgt import schwinger_dense, schwinger_mpo

N = 7


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
    mps, rest, left = [], v.reshape(1, -1), 1
    for _ in range(n - 1):
        U, S, Vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
        mps.append(U.reshape(left, 2, -1))
        rest, left = S[:, None] * Vh, U.shape[1]
    mps.append(rest.reshape(left, 2, 1))
    return mps


def _brute_force(mps, H):
    """var1, var2 and the full variance from explicit orthonormal bases of the one- and
    two-site spaces (full QRs give F and G), as dense vectors."""
    n = len(mps)
    A, c = [], np.ones((1, 1))
    for T in mps:                                       # left-canonical sweep
        T = np.einsum("ab,bsc->asc", c, T)
        Q, c = np.linalg.qr(T.reshape(-1, T.shape[2]))
        A.append(Q.reshape(T.shape[0], 2, -1))
    B, c = [None] * n, np.ones((1, 1))
    for i in range(n - 1, -1, -1):                      # right-canonical sweep
        T = np.einsum("asb,bc->asc", mps[i], c)
        Q, c = np.linalg.qr(T.reshape(T.shape[0], -1).T)
        B[i], c = Q.T.reshape(-1, 2, T.shape[2]), c.T
    left = [np.ones((1, 1))]                            # (2^i, D_i): sites < i
    for T in A:
        left.append(np.einsum("xa,asb->xsb", left[-1], T).reshape(-1, T.shape[2]))
    right = [None] * (n + 1)                            # (D, 2^(n-i)): sites >= i
    right[n] = np.ones((1, 1))
    for i in range(n - 1, -1, -1):
        right[i] = np.einsum("asb,by->asy", B[i], right[i + 1]).reshape(B[i].shape[0], -1)

    def F(i):                                           # (2^(i+1), m'): complement of A_i
        Q = np.linalg.qr(A[i].reshape(-1, A[i].shape[2]), mode="complete")[0]
        return np.kron(left[i], np.eye(2)) @ Q[:, A[i].shape[2]:]

    def G(i):                                           # (m', 2^(n-i)): complement of B_i
        Bm = B[i].reshape(B[i].shape[0], -1)
        Q = np.linalg.qr(Bm.T, mode="complete")[0]
        return Q[:, Bm.shape[0]:].T @ np.kron(np.eye(2), right[i + 1])

    v = _dense(mps)
    Hv = H @ v
    e = np.vdot(v, Hv)

    def weight(U):                                      # ||U^dag H v||^2, U orthonormal columns
        return float(np.linalg.norm(U.conj().T @ Hv) ** 2)

    var1 = sum(weight(np.kron(F(i), right[i + 1].T)) for i in range(n))
    var2 = sum(weight(np.kron(F(i), G(i + 1).T)) for i in range(n - 1))
    return var1, var2, float(np.linalg.norm(Hv - e * v) ** 2)


def _models():
    rng = np.random.default_rng(7)
    x, mu, ell = 1.3, rng.normal(size=N), rng.normal(size=N - 1)
    W_tfim = tfim_mpo(N, 1.2, 0.3)
    return {"tfim": (W_tfim, mpo_to_dense(W_tfim)),
            "schwinger": (schwinger_mpo(N, x, mu, ell, 2.0), schwinger_dense(N, x, mu, ell, 2.0))}


MODELS = _models()


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("chi", [2, 3])
def test_matches_brute_force(model, chi):
    W, H = MODELS[model]
    psi = _rand_mps(N, chi, chi)
    var1, var2 = two_site_variance(psi, W)
    ref1, ref2, full = _brute_force(psi, H)
    assert ref2 > 1e-2 * full                           # the two-site part is not trivially zero
    assert var1 == pytest.approx(ref1, rel=1e-10)
    assert var2 == pytest.approx(ref2, rel=1e-10)
    if model == "tfim":                                 # nearest neighbour: P0 + P1 + P2 is exact
        assert var1 + var2 == pytest.approx(full, rel=1e-10)
    else:                                               # long range: three-site and wider parts
        assert full - var1 - var2 > 1e-3 * full


@pytest.mark.parametrize("model", list(MODELS))
def test_bounded_by_full_variance(model):
    W, _ = MODELS[model]
    for seed in range(3):
        psi = _rand_mps(N, 3, seed)
        var1, var2 = two_site_variance(psi, W)
        var, floor = mpo_variance(psi, W)
        assert var1 + var2 <= var + 10 * floor


@pytest.mark.parametrize("model", list(MODELS))
def test_zero_on_an_exact_eigenstate(model):
    W, H = MODELS[model]
    w, V = np.linalg.eigh(H)
    for k in (0, 5):
        var1, var2 = two_site_variance(_mps_from_vector(V[:, k], N), W)
        scale = max(1.0, w[k] ** 2)
        assert var1 <= 1e-24 * scale and var2 <= 1e-24 * scale


@pytest.mark.parametrize("model", list(MODELS))
def test_torch_cpu_equals_numpy(model):
    torch = pytest.importorskip("torch", reason="torch extra not installed")
    W, _ = MODELS[model]
    psi = _rand_mps(N, 3, 4)
    ref = two_site_variance(psi, W)
    got = two_site_variance([torch.as_tensor(A) for A in psi], W)
    assert got == pytest.approx(ref, rel=1e-12)


def _tangent_basis(mps, width):
    """Orthonormal basis of the span of all derivatives of the state with respect to blocks of
    `width` neighbouring tensors: the tangent space, with no reference to canonical forms."""
    n, vecs = len(mps), []
    for i in range(n - width + 1):
        blk = mps[i]
        for j in range(1, width):
            blk = np.einsum("asb,btc->astc", blk, mps[i + j]).reshape(blk.shape[0], -1,
                                                                    mps[i + j].shape[2])
        for idx in np.ndindex(*blk.shape):
            e = np.zeros(blk.shape, complex)
            e[idx] = 1
            vecs.append(_dense_raw(list(mps[:i]) + [e] + list(mps[i + width:])))
    U, s, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return U[:, s > 1e-9 * s[0]]


def _dense_raw(mps):
    v = np.ones((1, 1))
    for A in mps:
        v = np.einsum("xa,asb->xsb", v, A).reshape(-1, A.shape[2])
    return v[:, 0]


def _opsum_mpo(long_range):
    """Complex Hermitian: i (S+_n S-_n+1 - S-_n S+_n+1) and random Z, Y fields, plus Z Z at
    every distance >= 2 when `long_range`."""
    from tnkit.opsum import OpSum, compile_mpo, spin_half
    rng, os_ = np.random.default_rng(5), OpSum()
    for n in range(N - 1):
        os_ += (1j, "Sp", n, "Sm", n + 1)
        os_ += (-1j, "Sm", n, "Sp", n + 1)
    for n in range(N):
        os_ += (rng.normal(), "Z", n)
        os_ += (rng.normal(), "Y", n)
    if long_range:
        for a in range(N):
            for b in range(a + 2, N):
                os_ += (0.3 * rng.normal(), "Z", a, "Z", b)
    return compile_mpo(os_, [spin_half()] * N)[0]


@pytest.mark.parametrize("model", ["tfim", "schwinger", "complex-nn", "complex-long"])
@pytest.mark.parametrize("chi", [2, 3])
def test_matches_the_tangent_space_projectors(model, chi):
    """var1 and var2 against projectors onto the span of the one-site and of the one- and
    two-site parameter derivatives, for a random complex state, including complex Hermitian
    Hamiltonians. Nearest-neighbour models close the sum exactly; long-range ones do not."""
    W = {"tfim": lambda: MODELS["tfim"][0], "schwinger": lambda: MODELS["schwinger"][0],
         "complex-nn": lambda: _opsum_mpo(False), "complex-long": lambda: _opsum_mpo(True)}[model]()
    H = mpo_to_dense(W)
    psi = _rand_mps(N, chi, 3)
    v = _dense_raw(psi)
    v = v / np.linalg.norm(v)
    r = H @ v - np.vdot(v, H @ v) * v
    full = np.linalg.norm(r) ** 2
    b1 = np.linalg.norm(_tangent_basis(psi, 1).conj().T @ r) ** 2
    b12 = np.linalg.norm(_tangent_basis_union(psi).conj().T @ r) ** 2
    var1, var2 = two_site_variance(psi, W)
    assert abs(var1 - b1) < 1e-10 * full and abs(var2 - (b12 - b1)) < 1e-10 * full
    gap = full - var1 - var2
    assert gap >= -1e-10 * full
    assert (abs(gap) < 1e-10 * full) == (model in ("tfim", "complex-nn"))


def _tangent_basis_union(mps):
    vecs = np.hstack([_tangent_basis(mps, 1), _tangent_basis(mps, 2)])
    U, s, _ = np.linalg.svd(vecs, full_matrices=False)
    return U[:, s > 1e-9 * s[0]]
