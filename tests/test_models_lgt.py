"""The lattice Schwinger model: MPO against its dense definition, physics limits, the
profile helpers, the C-parity tag, and the MPS observables against dense vectors."""
from __future__ import annotations

import time
from functools import reduce

import numpy as np
import pytest

from oracles.schwinger_rank_check import dense_H, hand_mpo
from tnkit.dmrg_excited import mpo_expectation
from tnkit.models import SZ, translation_mpo
from tnkit.models_lgt import (chiral_condensate, charge_conjugation_mpo, energy_density,
                              flux_correlator, flux_profile, particle_number, schwinger_dense,
                              schwinger_mpo, schwinger_opsum, site_charge)
from tnkit.observables import local_expectations
from tnkit.opsum import compile_mpo, spin_half, to_dense

CASES = [(1.0, 2e-5, 0.0, 0.0), (1.3, 0.5, 0.3, 0.0), (0.7, 0.4, 0.0, 10.0)]


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(b).max(), 1e-300)


def _zbits(N):
    """z_n of every basis state, site 0 most significant; index 0 is Z = +1."""
    return 1.0 - 2.0 * ((np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1)


def _random_mps(N, chi, seed):
    rng = np.random.default_rng(seed)
    dims = [1] + [min(chi, 2 ** min(i, N - i)) for i in range(1, N)] + [1]
    return [rng.normal(size=(dims[i], 2, dims[i + 1])) + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))
            for i in range(N)]


def _vec(mps):
    v = np.ones((1, 1))
    for A in mps:
        v = np.einsum("xa,asb->xsb", v, A).reshape(-1, A.shape[2])
    v = v[:, 0]
    return v / np.linalg.norm(v)


def _product(z):
    """Product-state MPS with Z eigenvalues z_n."""
    return [np.array([1.0, 0.0] if zn > 0 else [0.0, 1.0], dtype=complex).reshape(1, 2, 1)
            for zn in z]


def _torch(mps):
    torch = pytest.importorskip("torch")
    return [torch.as_tensor(A) for A in mps]


# --------------------------------------------------------------------------- the operator
@pytest.mark.parametrize("N", [2, 3, 4, 7, 8, 10])
@pytest.mark.parametrize("x,mu,ell,lam", CASES)
def test_mpo_equals_dense_definition(N, x, mu, ell, lam):
    H = schwinger_dense(N, x, mu, ell, lam)
    assert _rel(to_dense(schwinger_mpo(N, x, mu, ell, lam)), H) <= 1e-12


def test_mpo_equals_dense_n12_and_arrays():
    assert _rel(to_dense(schwinger_mpo(12, 1.3, 0.5, 0.3)), schwinger_dense(12, 1.3, 0.5, 0.3)) <= 1e-12
    rng = np.random.default_rng(8)
    mu, ell = rng.normal(size=8), rng.normal(size=7)
    assert _rel(to_dense(schwinger_mpo(8, 0.9, mu, ell, 5.0)), schwinger_dense(8, 0.9, mu, ell, 5.0)) <= 1e-12


@pytest.mark.parametrize("x,mu,ell,lam", CASES)
def test_dense_builder_equals_oracle(x, mu, ell, lam):
    assert _rel(schwinger_dense(8, x, mu, ell, lam), dense_H(8, x, mu, ell, lam=lam)) <= 1e-12


def test_opsum_is_public_and_extendable():
    os = schwinger_opsum(6, 1.1, 0.3, 0.2)
    os += (0.25, "X", 2)
    mpo, _ = compile_mpo(os, [spin_half()] * 6)
    X2 = reduce(np.kron, [np.array([[0, 1], [1, 0]]) if k == 2 else np.eye(2) for k in range(6)])
    assert _rel(to_dense(mpo), schwinger_dense(6, 1.1, 0.3, 0.2) + 0.25 * X2) <= 1e-12


@pytest.mark.parametrize("N,lam,widths", [(2, 0, [4]), (3, 0, [5, 4]), (8, 0, [5] * 6 + [4]),
                                          (2, 1, [5]), (3, 1, [5, 5]), (8, 1, [5] * 7)])
def test_widths(N, lam, widths):
    assert [W.shape[1] for W in schwinger_mpo(N, 1.0, 0.3, 0.0, lam)[:-1]] == widths


@pytest.mark.parametrize("lam", [0.0, 2.0])
def test_widths_with_arrays(lam):
    rng = np.random.default_rng(1)
    mpo = schwinger_mpo(8, 1.0, rng.normal(size=8), rng.normal(size=7), lam)
    widths = [W.shape[1] for W in mpo[:-1]]
    assert max(widths) <= 5 and (lam == 0 or widths == [5] * 7)


@pytest.mark.parametrize("bad", [
    lambda: schwinger_mpo(1, 1.0, 0.1),
    lambda: schwinger_mpo(4, 1.0, 0.1, charge_penalty=-1.0),
    lambda: schwinger_mpo(4, 1.0, 0.1, charge_penalty=float("nan")),
    lambda: schwinger_mpo(4, 1.0, 0.1, charge_penalty=float("inf")),
    lambda: schwinger_opsum(4, 1.0, 0.1, charge_penalty=float("nan")),
    lambda: schwinger_mpo(4, 1.0, np.ones(3)),
    lambda: schwinger_mpo(4, 1.0, 0.1, ell=np.ones(4)),
    lambda: charge_conjugation_mpo(5),
])
def test_bad_input_raises(bad):
    with pytest.raises(ValueError):
        bad()


@pytest.mark.parametrize("N", [8, 9, 10])
def test_free_fermion_spectrum(N):
    """W(x, mu) - W(0, 0) is quadratic in n_k = (1 + Z_k)/2: one-particle matrix
    h = x * adjacency + mu * diag((-1)^k), plus mu/2 * sum_k (1 - (-1)^k)."""
    x, mu = 0.8, 0.45
    s = (-1.0) ** np.arange(N)
    eps = np.linalg.eigvalsh(x * (np.eye(N, k=1) + np.eye(N, k=-1)) + mu * np.diag(s))
    occ = (np.arange(2 ** N)[:, None] >> np.arange(N)) & 1
    many = np.sort(occ @ eps + 0.5 * mu * (1 - s).sum())
    H = to_dense(schwinger_mpo(N, x, mu)) - to_dense(schwinger_mpo(N, 0.0, 0.0))
    spec = np.linalg.eigvalsh(H)
    assert np.abs(spec - many).max() <= 1e-12 * np.abs(many).max()


def test_strong_coupling_levels():
    N, mu = 10, 0.3
    H = to_dense(schwinger_mpo(N, 0.0, mu))
    assert np.count_nonzero(H - np.diag(np.diag(H))) == 0
    E, Q = np.diag(H).real, 0.5 * (_zbits(N) + (-1.0) ** np.arange(N)).sum(1)
    e0 = np.sort(E[Q == 0])
    assert np.abs(e0[:10] - np.array([0.0] + [1 + 2 * mu] * 9)).max() <= 1e-12
    assert e0[10] > 1 + 2 * mu + 0.1
    em = np.sort(E[Q == -1])
    assert abs(em[0] - mu) <= 1e-12 and em[1] > mu + 0.1


def test_charge_penalty_removes_the_charged_level():
    N, x, mu = 10, 1.0, 2e-5
    Q = 0.5 * (_zbits(N) + (-1.0) ** np.arange(N)).sum(1)
    E, V = np.linalg.eigh(to_dense(schwinger_mpo(N, x, mu)))
    assert abs((np.abs(V[:, 1]) ** 2) @ Q + 1) <= 1e-10
    H = to_dense(schwinger_mpo(N, x, mu)).real
    sector = np.linalg.eigvalsh(H[np.ix_(Q == 0, Q == 0)])[:3]
    pen = np.linalg.eigvalsh(to_dense(schwinger_mpo(N, x, mu, charge_penalty=10.0)))[:3]
    assert np.abs(pen - sector).max() <= 1e-10


def test_large_n_against_hand_automaton():
    t0 = time.perf_counter()
    mpo = schwinger_mpo(300, 100.0, 2.5)
    assert time.perf_counter() - t0 < 2.0
    rng = np.random.default_rng(0)
    v = rng.normal(size=(300, 2)) + 1j * rng.normal(size=(300, 2))
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    def expval(m):
        E = np.ones((1,))
        for W, u in zip(m, v):
            E = E @ np.einsum("abst,s,t->ab", W, u.conj(), u)
        return E[0]

    ref = expval(hand_mpo(300, 100.0, 2.5, 0.0))
    assert abs(expval(mpo) - ref) <= 1e-12 * abs(ref)


# --------------------------------------------------------------------------- profile helpers
def test_profiles_on_product_states():
    N, ell = 8, np.linspace(-0.3, 0.4, 7)
    vac = -(-1.0) ** np.arange(N)                      # even sites Z = -1, odd sites Z = +1
    assert np.array_equal(site_charge(vac), np.zeros(N))
    assert np.array_equal(flux_profile(vac, ell), ell)
    assert particle_number(vac).sum() == 0
    for n, sign in [(2, 1.0), (3, -1.0)]:
        z = vac.copy()
        z[[n, n + 1]] *= -1
        L = np.zeros(N - 1)
        L[n] = sign
        assert np.array_equal(flux_profile(z), L)
        assert np.array_equal(flux_profile(z, ell), ell + L)
        assert particle_number(z).mean() == 2 / N
        assert site_charge(z).sum() == 0


def test_profiles_match_mps_measurement():
    psi = _random_mps(6, 4, 3)
    z = local_expectations(psi, SZ)
    zd = (np.abs(_vec(psi)) ** 2) @ _zbits(6)
    assert np.abs(z - zd).max() <= 1e-12


def test_chiral_condensate_against_dense():
    N, x = 8, 2.3
    psi = _random_mps(N, 6, 4)
    v = _vec(psi)
    sigma = np.sqrt(x) / N * (0.5 * (1 + _zbits(N)) * (-1.0) ** np.arange(N)).sum(1)
    assert abs(chiral_condensate(local_expectations(psi, SZ), x) - (np.abs(v) ** 2) @ sigma) <= 1e-12


def test_chiral_condensate_free_ground_state_is_minus_banuls_eq_3_1():
    """Free ground state (gauge term removed) against arXiv:1310.4118 Eq. (3.1)."""
    N, x, mu = 10, 1.7, 0.6
    H = to_dense(schwinger_mpo(N, x, mu)) - to_dense(schwinger_mpo(N, 0.0, 0.0))
    v = np.linalg.eigh(H)[1][:, 0]
    z = (np.abs(v) ** 2) @ _zbits(N)
    q = np.arange(1, N // 2 + 1)
    eq31 = np.sqrt(x) / N * np.sum(mu / np.sqrt(mu ** 2 + 4 * x ** 2 * np.cos(q * np.pi / (N + 1)) ** 2))
    assert abs(chiral_condensate(z, x) + eq31) <= 1e-12


# --------------------------------------------------------------------------- C-parity tag
def _dense_SR(N):
    """(prod X) T(+1) on bitstrings: T|s_0..s_{N-1}> = |s_{N-1}, s_0, .., s_{N-2}>."""
    bits = ((np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1)
    out = 1 - np.roll(bits, 1, axis=1)
    S = np.zeros((2 ** N, 2 ** N))
    S[out @ (1 << np.arange(N - 1, -1, -1)), np.arange(2 ** N)] = 1
    return S


@pytest.mark.parametrize("N", [2, 4, 8, 12])
def test_charge_conjugation_is_the_dense_permutation(N):
    mpo = charge_conjugation_mpo(N)
    assert [W.shape[1] for W in mpo[:-1]] == [4] * (N - 1)
    assert np.array_equal(to_dense(mpo), _dense_SR(N))


@pytest.mark.parametrize("N", [4, 8])
def test_charge_conjugation_algebra(N):
    S = _dense_SR(N)
    T = to_dense(translation_mpo(N, +1)).real
    assert np.array_equal(S.T @ S, np.eye(2 ** N)) and np.array_equal(S @ S, T @ T)
    P = np.eye(2 ** N)
    for k in range(1, N + 1):
        P = P @ S
        assert np.array_equal(P, np.eye(2 ** N)) == (k == N)
    vac = (_zbits(N) == -(-1.0) ** np.arange(N)).all(1).astype(float)
    assert np.array_equal(S @ vac, vac)


def test_charge_conjugation_tags_vector_and_ground_state():
    N, x, mu = 12, 0.3, 0.1
    Q = 0.5 * (_zbits(N) + (-1.0) ** np.arange(N)).sum(1)
    H = schwinger_dense(N, x, mu)[np.ix_(Q == 0, Q == 0)]
    V = np.linalg.eigh(H)[1][:, :4]
    S = _dense_SR(N)[np.ix_(Q == 0, Q == 0)]
    tag = np.einsum("ik,ij,jk->k", V, S, V)
    assert tag[0] > 0.5 and (tag[1:] < -0.5).all()


# --------------------------------------------------------------------------- MPS observables
def _dense_flux(N, v):
    Ls = np.cumsum(0.5 * (_zbits(N) + (-1.0) ** np.arange(N)), axis=1)[:, :-1]
    p = np.abs(v) ** 2
    mean = p @ Ls
    return np.einsum("k,km,kn->mn", p, Ls, Ls) - np.outer(mean, mean)


def test_flux_correlator_against_dense():
    psi = _random_mps(8, 6, 5)
    assert np.abs(flux_correlator(psi) - _dense_flux(8, _vec(psi))).max() <= 1e-12
    assert np.abs(flux_correlator(_product(-(-1.0) ** np.arange(6)))).max() <= 1e-15


def _dense_pieces(N, v, x, mu, ell, lam):
    mu = np.broadcast_to(mu, (N,))
    ell = np.broadcast_to(ell, (N - 1,))
    z, s, p = _zbits(N), (-1.0) ** np.arange(N), np.abs(v) ** 2
    q = 0.5 * (z + s)
    hop = []
    for n in range(N - 1):
        flip = np.arange(2 ** N) ^ (3 << (N - n - 2))
        differ = z[:, n] != z[:, n + 1]
        hop.append(x * np.sum(np.conj(v[flip[differ]]) * v[differ]).real)
    electric = p @ (ell + np.cumsum(q, axis=1)[:, :-1]) ** 2
    return np.array(hop), 0.5 * mu * (1 + s * (p @ z)), electric, lam * p @ q.sum(1) ** 2


@pytest.mark.parametrize("N,lam", [(8, 0.0), (10, 3.0)])
def test_energy_density_sum_rule_and_pieces(N, lam):
    rng = np.random.default_rng(N)
    x, mu, ell = 1.3, rng.normal(size=N), rng.normal(size=N - 1)
    psi = _random_mps(N, 6, N)
    pieces = energy_density(psi, x, mu, ell, lam)
    total = sum(np.sum(p) for p in pieces)
    E = mpo_expectation(psi, schwinger_mpo(N, x, mu, ell, lam))
    assert abs(total - E) <= 1e-12 * abs(E) and abs(E.imag) <= 1e-12 * abs(E)
    for got, ref in zip(pieces, _dense_pieces(N, _vec(psi), x, mu, ell, lam)):
        assert np.abs(got - ref).max() <= 1e-12 * max(1.0, np.abs(ref).max())


def test_energy_density_vanishes_on_the_bare_vacuum():
    pieces = energy_density(_product(-(-1.0) ** np.arange(8)), 0.0, 0.7, 0.0, 2.0)
    assert all(np.abs(p).max() <= 1e-15 for p in pieces)


@pytest.mark.parametrize("fn", [flux_correlator, lambda m: energy_density(m, 1.0, 0.1)])
def test_window_state_is_refused(fn):
    psi = _random_mps(6, 4, 1)
    psi[0] = np.ones((3, 2, psi[0].shape[2]))
    with pytest.raises(ValueError, match="open chain"):
        fn(psi)


@pytest.mark.parametrize("fn", [flux_correlator,
                                lambda m: np.concatenate([np.ravel(p) for p in energy_density(
                                    m, 1.2, np.linspace(0.1, 0.5, 8), 0.3, 2.0)])])
def test_torch_cpu_equals_numpy(fn):
    psi = _random_mps(8, 6, 9)
    assert np.abs(fn(_torch(psi)) - fn(psi)).max() <= 1e-12


# --------------------------------------------------------------------------- GPU smoke
def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.cuda
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device on this runner")
def test_cuda_schwinger_pipeline():
    """Vacuum, packet, compression, 2TDVP then graphed 1TDVP, and every observable, on the
    device. The graph's Krylov length is 24: Schwinger blocks need more than the default 12
    (up to 17 for this state on the CPU path), and each shortfall re-runs the step eagerly."""
    import torch
    from tnkit import (all_entropies, apply_mpo, compress, dmrg, get_backend, tdvp1_evolve,
                       tdvp2_evolve, two_site_entropy)
    from tnkit.opsum import OpSum
    from tnkit.prefix_graph_impl import PrefixSiteStepGraphCache

    N, x, mu, lam, dt = 20, 1.0, 2e-5, 10.0, 0.05
    W = schwinger_mpo(N, x, mu, charge_penalty=lam)
    _, vac, _ = dmrg(W, chi=16, n_sweeps=10, backend="torch-cuda")
    os = OpSum()
    for n in range(N - 1):
        a = np.exp(-(n - N / 2) ** 2 / 16) * np.exp(1.13j * n)
        os += (a, "Sp", n, "Sm", n + 1)
        os += (-a, "Sp", n + 1, "Sm", n)
    psi, disc = compress(apply_mpo(vac, compile_mpo(os, [spin_half()] * N)[0]), 32)
    assert psi[0].device.type == "cuda" and np.isfinite(disc).all()
    psi, _ = tdvp2_evolve(psi, W, 32, dt, 2, backend="torch-cuda")
    w0 = mpo_expectation(psi, W).real
    cache = PrefixSiteStepGraphCache(get_backend("torch-cuda"), torch, dt, 24)
    psi, _ = tdvp1_evolve(psi, W, dt, 5, backend="torch-cuda", prefix_graph=cache)
    stats = cache.summary()
    assert stats["short"] == 0 and stats["replayed_frac"] >= 0.9, stats
    assert psi[0].device.type == "cuda"
    z = local_expectations(psi, SZ)
    assert abs(site_charge(z).sum()) <= 1e-8
    assert abs(mpo_expectation(psi, W).real - w0) <= 1e-8 * abs(w0)
    pieces = energy_density(psi, x, mu, 0.0, lam)
    assert abs(sum(np.sum(p) for p in pieces) - w0) <= 1e-8 * abs(w0)
    for arr in (flux_correlator(psi), two_site_entropy(psi), all_entropies(psi),
                chiral_condensate(z, x)):
        assert np.isfinite(arr).all()
