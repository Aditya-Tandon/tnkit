"""Purified thermal states against exact diagonalisation."""
import numpy as np
import pytest

from tnkit import (SX, SY, SZ, apply_mpo, local_expectations, mpo_expectation, tdvp2_evolve,
                   tfim_mpo)
from tnkit.models_lgt import schwinger_charges, schwinger_dense, schwinger_mpo
from tnkit.thermal import (cool, infinite_temperature_state, purify, purify_mpo,
                           thermal_state)


def dense(mpo):
    T = np.asarray(mpo[0])[0]
    for W in mpo[1:]:
        T = np.einsum("bst,bcuv->csutv", T, W)
        c, s, u, t, v = T.shape
        T = T.reshape(c, s * u, t * v)
    return T[0]


def fused_order(N, d):
    """Permutation from (s_1..s_N, a_1..a_N) to the fused order (s_1 a_1, ..., s_N a_N)."""
    idx = np.arange(d ** (2 * N)).reshape([d] * (2 * N))
    return idx.transpose([k for n in range(N) for k in (n, N + n)]).reshape(-1)


def gibbs(Hd, beta, ops=()):
    """Energy, log Z and <op> of exp(-beta H)/Z."""
    w, V = np.linalg.eigh(Hd)
    p = np.exp(-beta * (w - w[0]))
    Z = p.sum()
    rho = (V * (p / Z)) @ V.conj().T
    return ((w * p).sum() / Z, np.log(Z) - beta * w[0],
            [np.trace(rho @ op).real for op in ops])


def site_ops(op, N):
    return [np.kron(np.kron(np.eye(2 ** j), op), np.eye(2 ** (N - j - 1))) for j in range(N)]


def complex_mpo(N, g=0.8, h=0.4):
    """-sum X_j Y_{j+1} - g sum Z_j - h sum Y_j: complex, so H^T != H."""
    W = np.zeros((3, 3, 2, 2), dtype=complex)
    W[0, 0] = W[2, 2] = np.eye(2)
    W[0, 1], W[0, 2], W[1, 2] = SX, -g * SZ - h * SY, -SY
    return [W[:1]] + [W.copy() for _ in range(N - 2)] + [W[:, 2:]]


def sector(N):
    """Indices of the Q = 0 states of the Schwinger chain in the dense basis."""
    q = schwinger_charges(N)
    bits = (np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1
    return np.nonzero(q[np.arange(N), bits].sum(1) == 0)[0]


@pytest.mark.parametrize("twin", [False, True])
def test_purify_mpo_is_the_fused_kronecker_product(twin):
    N = 4
    for H in (tfim_mpo(N, 0.8, 0.3), schwinger_mpo(N, 4.0, 0.2, ell=0.3), complex_mpo(N)):
        Hd, I = dense(H), np.eye(2 ** N)
        ref = np.kron(Hd, I) - (np.kron(I, Hd.T) if twin else 0)
        p = fused_order(N, 2)
        assert np.abs(dense(purify_mpo(H, twin=twin)) - ref[np.ix_(p, p)]).max() < 1e-12


def test_infinite_temperature_state():
    N = 6
    psi, log_dim = infinite_temperature_state(N)
    assert log_dim == pytest.approx(N * np.log(2))
    assert np.allclose(local_expectations(psi, purify(SZ)), 0, atol=1e-14)
    assert mpo_expectation(psi, purify_mpo(tfim_mpo(N, 0.7))) == pytest.approx(0, abs=1e-13)

    psi, log_dim = infinite_temperature_state(N, 2, schwinger_charges(N), 0)
    keep = sector(N)
    assert log_dim == pytest.approx(np.log(len(keep)))
    assert max(int(A.shape[2]) for A in psi) <= N // 2 + 1
    Hd = schwinger_dense(N, 4.0, 0.2)
    E = mpo_expectation(psi, purify_mpo(schwinger_mpo(N, 4.0, 0.2))).real
    assert E == pytest.approx(np.trace(Hd[np.ix_(keep, keep)]).real / len(keep), abs=1e-12)
    with pytest.raises(ValueError):
        infinite_temperature_state(N, 2, schwinger_charges(N), N)


def test_tfim_energy_and_log_z_match_ed():
    N, g, h = 6, 0.8, 0.3                       # longitudinal field: not integrable
    H = tfim_mpo(N, g, h)
    Hd, Z = dense(H), site_ops(SZ, N)
    psi, log_dim = infinite_temperature_state(N)
    PH, PZ = purify_mpo(H), purify(SZ)
    _, rec = cool(psi, H, 64, 0.05, 40, log_z=log_dim, meas_every=10,
                  measure=lambda m: (mpo_expectation(m, PH).real,
                                     local_expectations(m, PZ).real))
    assert [round(r[0], 12) for r in rec] == [0.0, 0.5, 1.0, 1.5, 2.0]
    for beta, log_z, (E, z) in rec:
        E0, lz0, z0 = gibbs(Hd, beta, Z)
        assert abs(E - E0) < 1e-8 and abs(log_z - lz0) < 1e-8
        assert np.abs(z - z0).max() < 1e-8


def test_schwinger_sector_matches_ed():
    N, x, mu = 8, 4.0, 0.2
    H = schwinger_mpo(N, x, mu)
    keep = sector(N)
    Hd = schwinger_dense(N, x, mu)[np.ix_(keep, keep)]
    Z = [np.diag(op)[keep] for op in site_ops(SZ, N)]
    PH, PZ = purify_mpo(H), purify(SZ)
    psi, log_dim = infinite_temperature_state(N, 2, schwinger_charges(N), 0)
    _, rec = cool(psi, H, 256, 0.05, 10, log_z=log_dim, warmup=8, meas_every=5,
                  measure=lambda m: (mpo_expectation(m, PH).real,
                                     local_expectations(m, PZ).real))
    w, V = np.linalg.eigh(Hd)
    for beta, log_z, (E, z) in rec[1:]:
        E0, lz0, _ = gibbs(Hd, beta)
        p = np.exp(-beta * (w - w[0]))
        diag = (np.abs(V) ** 2 * p).sum(1) / p.sum()
        assert abs(E - E0) < 1e-8 and abs(log_z - lz0) < 1e-8
        assert np.abs(z - [diag @ zn for zn in Z]).max() < 1e-8


def test_warmup_removes_the_long_range_start_error():
    """Without warm-up, the first steps from bond dimension 1 miss the long-range terms."""
    N, beta = 6, 0.25
    H = schwinger_mpo(N, 4.0, 0.2)
    keep = sector(N)
    E0 = gibbs(schwinger_dense(N, 4.0, 0.2)[np.ix_(keep, keep)], beta)[0]
    errs = []
    for warmup in (0, 8):
        rho, _ = thermal_state(H, beta, 64, 0.05, charges=schwinger_charges(N), warmup=warmup)
        errs.append(abs(mpo_expectation(rho, purify_mpo(H)).real - E0))
    assert errs[0] > 1e-6 and errs[1] < 1e-9


def test_low_temperature_limit():
    N = 6
    H = tfim_mpo(N, 1.5)
    w = np.linalg.eigvalsh(dense(H))
    rho, log_z = thermal_state(H, 6.0, 64, 0.1)
    E = mpo_expectation(rho, purify_mpo(H)).real
    assert E - w[0] < 1e-3 and E == pytest.approx(gibbs(dense(H), 6.0)[0], abs=1e-8)
    assert log_z == pytest.approx(gibbs(dense(H), 6.0)[1], abs=1e-8)


def test_backward_twins_keep_the_thermal_state_static():
    N, beta = 6, 0.7
    H = tfim_mpo(N, 0.8, 0.3)
    rho, _ = thermal_state(H, beta, 64, 0.05)
    PH = purify_mpo(H)
    chi = {}
    for twin in (True, False):
        _, rec = tdvp2_evolve(rho, purify_mpo(H, twin=twin), 64, 0.05, 40, svd_tol=1e-12,
                              meas_every=40, measure=lambda m: (mpo_expectation(m, PH).real,
                                                                max(A.shape[2] for A in m)))
        (_, (E0, _)), (_, (E1, c1)) = rec
        assert abs(E1 - E0) < 1e-9
        chi[twin] = c1
    assert chi[False] > 2 * chi[True]


def test_real_time_after_a_local_perturbation_matches_ed():
    N, beta = 6, 0.7
    H = tfim_mpo(N, 0.8, 0.3)
    Hd = dense(H)
    rho, _ = thermal_state(H, beta, 64, 0.05)
    B = [np.eye(2, dtype=complex).reshape(1, 1, 2, 2) for _ in range(N)]
    B[2] = SX.reshape(1, 1, 2, 2)
    _, rec = tdvp2_evolve(apply_mpo(rho, purify_mpo(B)), purify_mpo(H, twin=True), 64, 0.05, 40,
                          meas_every=20, measure=lambda m: local_expectations(m, purify(SZ)).real)
    w, V = np.linalg.eigh(Hd)
    Bd = site_ops(SX, N)[2]
    r0 = Bd @ (V * np.exp(-beta * (w - w[0]))) @ V.conj().T @ Bd
    r0 /= np.trace(r0)
    for t, z in rec:
        U = (V * np.exp(-1j * w * t)) @ V.conj().T
        rt = U @ r0 @ U.conj().T
        assert np.abs(z - [np.trace(rt @ Zj).real for Zj in site_ops(SZ, N)]).max() < 1e-8


def test_cool_refuses_graphs_and_leaves_options_alone():
    from tnkit import TDVPOptions
    H = tfim_mpo(4, 1.0)
    psi, _ = infinite_temperature_state(4)
    o = TDVPOptions(svd_tol=1e-10)
    cool(psi, H, 16, 0.1, 2, options=o)
    assert o.renorm is True and o.graph == "auto"
    with pytest.raises(ValueError):
        cool(psi, H, 16, 0.1, 1, graph="on")
    with pytest.raises(ValueError):
        cool(psi, H, 16, 0.1, 1, canonicalize=False)
    with pytest.raises(ValueError):
        thermal_state(H, 0.25, 16, 0.1)


def test_torch_matches_numpy():
    pytest.importorskip("torch")
    N = 4
    H = schwinger_mpo(N, 4.0, 0.2)
    out = []
    for backend in ("numpy", "torch-cpu"):
        rho, log_z = thermal_state(H, 0.5, 16, 0.05, charges=schwinger_charges(N),
                                   backend=backend)
        out.append((mpo_expectation(rho, purify_mpo(H)).real, log_z))
    assert np.allclose(out[0], out[1], atol=1e-10, rtol=0)
