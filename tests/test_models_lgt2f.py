"""The two-flavour lattice Schwinger model: the compiled MPO against the Pauli-string exact
diagonalisation, isospin, the lattice P and CP tags, and the d = 4 regrouping."""
from __future__ import annotations

import numpy as np
import pytest

from tnkit.models_lgt2f import (charge_conjugation_2f_mpo, charge_sparse_diag, compile2f,
                                isospin_opsum, isospin_sparse, merge_pairs, merged,
                                reflection_expectation, schwinger2f_mpo, schwinger2f_opsum,
                                schwinger2f_sparse)
from tnkit.opsum import to_dense

CASES = [(1.0, -0.3, 0.0, 0.0), (4.0, -0.1, 0.0, 20.0), (1.3, (0.5, 0.2), 0.3, 2.0),
         (0.7, 0.1, -0.6, 10.0)]


def _exact_mps(v, n_qubits):
    mps, chi, rest = [], 1, v.reshape(1, -1)
    for _ in range(n_qubits - 1):
        U, S, Vh = np.linalg.svd(rest.reshape(chi * 2, -1), full_matrices=False)
        k = int(np.sum(S > 1e-14 * S[0]))
        mps.append(U[:, :k].reshape(chi, 2, k))
        rest, chi = S[:k, None] * Vh[:k], k
    mps.append(rest.reshape(chi, 2, 1))
    return mps


def _q0_levels(H, N, k):
    q = charge_sparse_diag(N)
    idx = np.flatnonzero(q == 0)
    e, u = np.linalg.eigh(H[np.ix_(idx, idx)])
    V = np.zeros((H.shape[0], k))
    V[idx] = u[:, :k]
    return e[:k], V


@pytest.mark.parametrize("N", [2, 3, 4])
@pytest.mark.parametrize("x,mu,ell,lam", CASES)
def test_mpo_equals_pauli_string_ed(N, x, mu, ell, lam):
    H = schwinger2f_sparse(N, x, mu, ell, lam).toarray()
    W = to_dense(schwinger2f_mpo(N, x, mu, ell, lam))
    assert np.abs(W - H).max() <= 1e-12 * max(1.0, np.abs(H).max())


def test_site_dependent_arrays():
    rng = np.random.default_rng(3)
    N = 4
    mu, ell = rng.normal(size=(N, 2)), rng.normal(size=N - 1)
    H = schwinger2f_sparse(N, 0.9, mu, ell, 3.0).toarray()
    assert np.abs(to_dense(schwinger2f_mpo(N, 0.9, mu, ell, 3.0)) - H).max() <= 1e-12 * np.abs(H).max()


def test_q0_spectrum_n5():
    """Ground state and the lowest gaps of the Q = 0 sector, ED against the MPO, N = 5."""
    N = 5
    H = schwinger2f_sparse(N, 1.0, -0.3).toarray()
    e_ed, _ = _q0_levels(H, N, 8)
    e_mpo, _ = _q0_levels(to_dense(schwinger2f_mpo(N, 1.0, -0.3)).real, N, 8)
    assert abs(e_ed[0] - e_mpo[0]) <= 1e-10
    assert np.abs((e_ed[1:] - e_ed[0]) - (e_mpo[1:] - e_mpo[0])).max() <= 1e-10


def test_strong_coupling_vacuum():
    """x = 0: the bare vacuum (even sites empty, odd sites full) has W = 0 and the cheapest
    Q = 0 excitation is one flavour hopped across one link, 2 mu + 1."""
    N, mu = 4, 0.3
    H = schwinger2f_sparse(N, 0.0, mu)
    e, _ = _q0_levels(H.toarray(), N, 9)
    assert abs(e[0]) <= 1e-12
    assert np.abs(e[1:] - (2 * mu + 1)).min() <= 1e-12


def test_isospin():
    N = 4
    H = schwinger2f_sparse(N, 1.0, -0.3, 0.0, 5.0)
    Iz, Ip = isospin_sparse(N)
    assert abs(H @ Ip - Ip @ H).max() <= 1e-12 and abs(H @ Iz - Iz @ H).max() <= 1e-12
    Hs = schwinger2f_sparse(N, 1.0, (-0.3, 0.2))
    assert abs(Hs @ Ip - Ip @ Hs).max() > 0.1
    I2 = (Iz @ Iz + 0.5 * (Ip @ Ip.T + Ip.T @ Ip)).toarray()
    for which, ref in (("I2", I2), ("Iz", Iz.toarray()),
                       ("Iz2", ((Iz - np.eye(4 ** N)) @ (Iz - np.eye(4 ** N))))):
        mpo, _ = compile2f(isospin_opsum(N, which, target=1.0), N)
        assert np.abs(to_dense(mpo) - np.asarray(ref)).max() <= 1e-12
    e, V = _q0_levels(H.toarray(), N, 4)
    assert abs(V[:, 0] @ I2 @ V[:, 0]) <= 1e-10                   # isosinglet vacuum
    assert np.ptp(e[1:4]) <= 1e-10                                # pion triplet
    assert all(abs(V[:, k] @ I2 @ V[:, k] - 2.0) <= 1e-10 for k in (1, 2, 3))


def test_merged_opsum():
    N = 3
    a = schwinger2f_opsum(N, 1.0, -0.3)
    b = isospin_opsum(N, "I2")
    m, _ = compile2f(merged(a, b, weights=[1.0, 2.5]), N)
    ref = to_dense(compile2f(a, N)[0]) + 2.5 * to_dense(compile2f(b, N)[0])
    assert np.abs(to_dense(m) - ref).max() <= 1e-12


@pytest.mark.parametrize("N", [3, 4])
def test_isospin_penalty(N):
    """isospin_penalty adds exactly lambda I^2, at most 4 MPO channels wider; 0 changes nothing."""
    lam = 16.0
    W0, Wp = schwinger2f_mpo(N, 4.0, 0.3), schwinger2f_mpo(N, 4.0, 0.3, isospin_penalty=lam)
    Iz, Ip = isospin_sparse(N)
    I2 = (Iz @ Iz + 0.5 * (Ip @ Ip.T + Ip.T @ Ip)).toarray()
    Dp = to_dense(Wp)
    assert np.abs(Dp - to_dense(W0) - lam * I2).max() <= 1e-15 * np.abs(Dp).max()
    assert max(w.shape[1] for w in Wp) <= max(w.shape[1] for w in W0) + 4
    assert all(a.tobytes() == b.tobytes()
               for a, b in zip(W0, schwinger2f_mpo(N, 4.0, 0.3, isospin_penalty=0.0)))


@pytest.mark.parametrize("N,ph", [(5, False), (4, True)])
def test_reflection_is_exact(N, ph):
    """P (odd N) and CP (even N) commute with W on Q = 0: eigenstates give +-1."""
    H = schwinger2f_sparse(N, 1.0, -0.3, 0.0 if not ph else 0.25).toarray()
    _, V = _q0_levels(H, N, 10)
    vals = [reflection_expectation(_exact_mps(V[:, k], 2 * N), particle_hole=ph).real
            for k in range(10)]
    assert abs(vals[0] - 1) <= 1e-10
    nondeg = [k for k in range(10) if abs(abs(vals[k]) - 1) > 1e-8]
    # only members of degenerate multiplets may be mixed
    e, _ = _q0_levels(H, N, 11)
    for k in nondeg:
        assert min(abs(e[k] - e[k - 1]), abs(e[k + 1] - e[k])) <= 1e-9


def test_charge_conjugation_tag_signs():
    """Vacuum C-even, neutral pion C-odd (G = +1, I = 1), at N = 6."""
    N = 6
    H = schwinger2f_sparse(N, 1.0, -0.3, 0.0, 20.0)
    e, V = _q0_levels(H.toarray(), N, 4)
    C = to_dense(charge_conjugation_2f_mpo(N)).real
    Iz, _ = isospin_sparse(N)
    assert V[:, 0] @ C @ V[:, 0] > 0.3
    # rotate the triplet to I_z eigenstates and take the I_z = 0 member
    T = V[:, 1:4]
    w, u = np.linalg.eigh(T.T @ (Iz @ T))
    pi0 = T @ u[:, np.argmin(abs(w))]
    assert pi0 @ C @ pi0 < -0.3
    with pytest.raises(ValueError):
        charge_conjugation_2f_mpo(5)


def test_merge_pairs():
    N = 3
    mpo = schwinger2f_mpo(N, 1.0, -0.3)
    m4 = merge_pairs(mpo)
    assert [w.shape[2] for w in m4] == [4] * N
    assert np.abs(to_dense(m4) - to_dense(mpo)).max() <= 1e-13


def test_isospin_commutes_with_H_only_for_equal_masses():
    """The total isospin I^2 commutes with H for equal flavour masses and not for unequal ones."""
    N = 3
    Iz, Ip = isospin_sparse(N)
    I2 = (Ip @ Ip.T + Ip.T @ Ip) / 2 + Iz @ Iz

    def comm(mu):
        H = schwinger2f_sparse(N, 1.0, mu, ell=0.4, charge_penalty=2.0)
        return abs(H @ I2 - I2 @ H).max()

    assert comm(0.3) < 1e-12
    assert comm((0.3, 0.5)) > 0.1


@pytest.mark.parametrize("bad", [
    lambda: schwinger2f_mpo(1, 1.0, 0.1),
    lambda: schwinger2f_mpo(4, 1.0, 0.1, charge_penalty=-1.0),
    lambda: schwinger2f_mpo(4, 1.0, 0.1, charge_penalty=float("nan")),
    lambda: schwinger2f_mpo(4, 1.0, 0.1, charge_penalty=float("inf")),
    lambda: schwinger2f_opsum(4, 1.0, 0.1, charge_penalty=float("nan")),
    lambda: schwinger2f_mpo(4, 1.0, 0.1, isospin_penalty=-1.0),
    lambda: schwinger2f_mpo(4, 1.0, 0.1, isospin_penalty=float("nan")),
    lambda: schwinger2f_mpo(4, 1.0, 0.1, isospin_penalty=float("inf")),
    lambda: schwinger2f_opsum(4, 1.0, 0.1, isospin_penalty=float("nan")),
    lambda: schwinger2f_mpo(4, 1.0, np.ones(3)),
    lambda: schwinger2f_mpo(4, 1.0, 0.1, ell=np.ones(4)),
    lambda: isospin_opsum(4, "Ix"),
])
def test_bad_input_raises(bad):
    with pytest.raises(ValueError):
        bad()


# --------------------------------------------------------------------------- packets, observables
def _dense_os(os_, N):
    return to_dense(compile2f(os_, N)[0])


def test_energy_density_and_occupations_match_dense():
    from tnkit.models_lgt2f import energy_density, flavour_profiles, occupations, qubit
    from tnkit.opsum import OpSum
    N, x, mu, ell, lam = 4, 1.3, (0.4, -0.2), 0.3, 2.0
    rng = np.random.default_rng(1)
    v = rng.normal(size=4 ** N) + 1j * rng.normal(size=4 ** N)
    v /= np.linalg.norm(v)
    M = _exact_mps(v, 2 * N)
    h, m, e, p = energy_density(M, x, mu, ell, lam)
    H = to_dense(schwinger2f_mpo(N, x, mu, ell, lam))
    assert abs(h.sum() + m.sum() + e.sum() + p - np.vdot(v, H @ v).real) <= 1e-12
    for n in range(N - 1):
        for f in (0, 1):
            os_ = OpSum()
            os_ += (x, "Cdag", qubit(n, f), "C", qubit(n + 1, f))
            os_ += (x, "Cdag", qubit(n + 1, f), "C", qubit(n, f))
            assert abs(h[f, n] - np.vdot(v, _dense_os(os_, N) @ v).real) <= 1e-12
    occ = occupations(M)
    for n in range(N):
        for f in (0, 1):
            os_ = OpSum()
            os_ += (1.0, "N", qubit(n, f))
            assert abs(occ[n, f] - np.vdot(v, _dense_os(os_, N) @ v).real) <= 1e-12
    prof = flavour_profiles(occ, ell)
    s = (-1.0) ** np.arange(N)
    assert np.allclose(prof["q"], occ.sum(1) - (1 - s))
    assert np.allclose(prof["L"], ell + np.cumsum(prof["q"])[:-1])


def test_packet_operators_and_isospin_rotation():
    from tnkit.models_lgt2f import gaussian_profile, packet_opsum
    N = 4
    f = gaussian_profile(N, 1.5, 0.7, 1.0)
    B = {k: _dense_os(packet_opsum(N, k, f), N) for k in ("u-d", "u+d", "+", "-")}
    Iz, Ip = isospin_sparse(N)
    Ip, Iz = Ip.toarray(), Iz.toarray()
    assert np.abs(Ip @ B["u-d"] - B["u-d"] @ Ip + 2 * B["+"]).max() <= 1e-13
    assert np.abs(Ip @ B["u+d"] - B["u+d"] @ Ip).max() <= 1e-13          # isoscalar
    assert np.abs(Iz @ B["+"] - B["+"] @ Iz - B["+"]).max() <= 1e-13     # raises I_z
    Bc = _dense_os(packet_opsum(N, "+", np.conj(f)), N)
    assert np.abs(Bc.conj().T + B["-"]).max() <= 1e-13
    H = schwinger2f_sparse(N, 1.0, -0.3, 0.0, 20.0).toarray()
    vac = np.linalg.eigh(H)[1][:, 0]
    assert np.abs(B["+"] @ vac + 0.5 * Ip @ (B["u-d"] @ vac)).max() <= 1e-12
    with pytest.raises(ValueError):
        packet_opsum(N, "x", f)


def test_wide_packet_with_one_range_is_the_link_packet():
    from tnkit.models_lgt2f import (PACKET_KINDS, gaussian_profile, packet_opsum,
                                    wide_packet_opsum)
    N = 4
    for kind in PACKET_KINDS:
        for lo, hi in ((0, None), (1, 1)):
            a = _dense_os(wide_packet_opsum(N, kind, 1.5, 0.7, 1.0, (1.0,), lo, hi), N)
            b = _dense_os(packet_opsum(N, kind, gaussian_profile(N, 1.5, 0.7, 1.0, lo, hi)), N)
            assert np.abs(a - b).max() <= 1e-14
    with pytest.raises(ValueError):
        wide_packet_opsum(N, "x", 1.5, 0.7, 1.0, (1.0,))


def test_packed_charges_unpack_on_every_cut():
    from tnkit.models_lgt2f import site_charges_2f, unpack_charge
    N = 3
    qs = site_charges_2f(N)
    for bits in range(2 ** (2 * N)):
        occ = [(bits >> j) & 1 for j in range(2 * N)]
        c = np.cumsum([q[o] for q, o in zip(qs, occ)])
        q, t = unpack_charge(c, N)
        n = np.arange(2 * N) // 2
        assert (q == np.cumsum(np.array(occ) - n % 2)).all()
        assert (t == np.cumsum((1 - 2 * (np.arange(2 * N) % 2)) * np.array(occ))).all()
        q, t = unpack_charge(-c, N)                     # the flipped leg of a block tensor
        assert (q == -np.cumsum(np.array(occ) - n % 2)).all()


def test_compile_real():
    W = compile2f(schwinger2f_opsum(3, 1.0, 0.2), 3, real=True)[0]
    assert all(w.dtype == np.float64 for w in W)
    assert np.abs(to_dense(W) - to_dense(schwinger2f_mpo(3, 1.0, 0.2))).max() == 0


def test_merge_pairs_on_a_state():
    rng = np.random.default_rng(2)
    v = rng.normal(size=4 ** 3) + 1j * rng.normal(size=4 ** 3)
    m4 = merge_pairs(_exact_mps(v / np.linalg.norm(v), 6))
    w = m4[0]
    for t in m4[1:]:
        w = np.tensordot(w, t, (w.ndim - 1, 0))
    assert [t.shape[1] for t in m4] == [4] * 3
    assert np.abs(w.reshape(-1) - v / np.linalg.norm(v)).max() <= 1e-13


def test_wide_packet_ranges_and_window():
    from tnkit.models_lgt2f import qubit, wide_packet_opsum
    from tnkit.opsum import OpSum
    N, c, k, sig = 5, 1.0, 0.4, 1.2
    ref = OpSum()
    for n in range(0, 2):                       # hi = 2: a range-2 term needs n + 2 <= 3
        a = np.exp(-(n - c) ** 2 / (2 * sig ** 2) + 1j * k * n)
        ref += (0.5 * a, "Cdag", qubit(n, 0), "C", qubit(n + 2, 1))
        ref += (0.5 * a, "Cdag", qubit(n + 2, 0), "C", qubit(n, 1))   # (-1)^2 = +1
    got = wide_packet_opsum(N, "+", c, k, sig, (0.0, 0.5), lo=0, hi=2)
    assert np.abs(_dense_os(got, N) - _dense_os(ref, N)).max() <= 1e-14


def test_flavour_profiles_per_flavour():
    from tnkit.models_lgt2f import flavour_profiles
    occ = np.array([[0.2, 0.7], [0.9, 0.4], [0.1, 0.3]])
    p = flavour_profiles(occ)
    np.testing.assert_allclose(p["q_u"], [0.2, -0.1, 0.1])
    np.testing.assert_allclose(p["q_d"], [0.7, -0.6, 0.3])
    np.testing.assert_allclose(p["iz"], [-0.25, 0.25, -0.1])
