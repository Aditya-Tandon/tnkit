"""Regression tests for the tangent-space quasiparticle ansatz (E1).

The gate is the exact TFIM dispersion epsilon(k) = 2 sqrt(1 + g^2 - 2 g cos k), which the ansatz
must reproduce in the DISORDERED phase (where the elementary excitation really is one fermion)
and must NOT reproduce in the ORDERED phase (where a single kink is topologically forbidden).
"""
import copy
import functools

import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.excitations import (QuasiparticleSolver, null_space_VL, tfim_dispersion_exact,
                               tfim_excitations)
from tnkit.idmrg import mixed_field_bond_h, mixed_field_bulk_mpo, tfim_bulk_mpo
from tnkit.vumps import vumps


@functools.lru_cache(maxsize=None)
def _cached_vumps(g, chi, n_iter, tol, seed, gz):
    mpo = bond_h = None
    if gz is not None:                              # mixed-field Ising instead of the TFIM
        mpo, bond_h = mixed_field_bulk_mpo(g, gz), mixed_field_bond_h(g, gz)
    return vumps(g, chi, n_iter=n_iter, tol=tol, mpo=mpo, bond_h=bond_h, seed=seed)


def ground_state(g, chi, n_iter, tol, seed=0, gz=None):
    """VUMPS ground state, run once per module and copied to each caller.

    VUMPS is deterministic at a fixed seed, so sharing a run changes no tested value."""
    return copy.deepcopy(_cached_vumps(g, chi, n_iter, tol, seed, gz))


def test_null_space_isometry():
    """V_L must be an isometry annihilating AL -- everything downstream assumes it."""
    be = get_backend("numpy")
    gs = ground_state(1.5, 8, n_iter=200, tol=1e-12)
    VL = null_space_VL(gs["AL"], be)
    D, d, n = VL.shape
    assert n == D * (d - 1)
    VLm = np.asarray(VL).reshape(D * d, n)
    ALm = np.asarray(gs["AL"]).reshape(D * d, D)
    assert np.allclose(VLm.conj().T @ VLm, np.eye(n), atol=1e-10)
    assert np.abs(VLm.conj().T @ ALm).max() < 1e-10


@pytest.mark.parametrize("g", [1.5, 2.0])
def test_dispersion_matches_free_fermion(g):
    """Disordered phase: the ansatz is exact for the TFIM, so this is a tight gate."""
    ps = [0.0, 0.7, np.pi / 2, 2.4, np.pi]
    w, _ = tfim_excitations(g, 12, ps, verbose=False)
    exact = tfim_dispersion_exact(np.array(ps), g)
    assert np.abs(w - exact).max() < 1e-7


def test_p_zero_is_not_singular():
    """The mixed transfer has eigenvalue 1, so (1 - e^{ip} T) is singular at p=0.

    The left-gauge condition is what keeps the source orthogonal to that direction. This test
    exists because losing the gauge condition would still give plausible numbers everywhere
    EXCEPT p=0, which is exactly where the physical gap lives.
    """
    w, _ = tfim_excitations(1.5, 12, [0.0], verbose=False)
    assert abs(w[0] - 2.0 * abs(1.0 - 1.5)) < 1e-7      # gap = 2|1-g|


def test_energy_zero_is_subtracted():
    """The regularised VUMPS environments leave a finite offset lambda0 != 0.

    Without subtracting it the dispersion has the right SHAPE but a constant offset -- the exact
    way this bug first presented (err = 2.0279 identically at every momentum).
    """
    be = get_backend("numpy")
    g, chi = 1.5, 12
    gs = vumps(g, chi, n_iter=300, tol=1e-12)
    W = be.array(tfim_bulk_mpo(g))
    solver = QuasiparticleSolver(gs["AL"], gs["AR"], gs["C"], W, be)
    assert abs(solver.lambda0) > 1e-6, "lambda0 is meant to be a genuine nonzero offset"

    # and with it applied, two different momenta must BOTH be right -- a constant offset would
    # survive a single-momentum check
    for p in (0.4, 2.2):
        w = solver.dispersion_at(p)
        assert abs(w - tfim_dispersion_exact(p, g)) < 1e-7


def test_ordered_phase_gives_two_kink_continuum():
    """Ordered phase: a topologically trivial ansatz CANNOT make one domain wall.

    It must land on the 2-kink threshold 2*min_k epsilon(k), variationally from above -- not on
    the 1-kink dispersion. Reproducing epsilon(k) here would mean the ansatz is wrong.
    """
    g = 0.5
    w, _ = tfim_excitations(g, 16, [0.0], verbose=False)
    one_kink = tfim_dispersion_exact(0.0, g)            # 1.0
    two_kink = 2.0 * one_kink                           # 2.0
    assert w[0] > 1.5 * one_kink                        # nowhere near the 1-kink gap
    assert two_kink <= w[0] < 1.15 * two_kink           # at the 2-kink threshold, from above


def test_negative_omega_guard_fires():
    """The omega >= 0 guard must actually raise, tested deterministically.

    Motivation: one run returned omega = -2.5e10 at g_x=1, g_z=0.16, chi=32 on Linux,
    while the identical call on macOS returned a clean 2.0317. The mechanism was diagnosed later
    and was NOT in this solver at all -- see test_overparameterised_environment_is_sane below.
    "Platform dependent" was a red herring: the failure is a roundoff lottery over the trace of a
    degenerate transfer-operator fixed point, which both platforms sample and macOS happened to
    keep winning. The physical failure is therefore still not usable as a portable regression
    test for THIS guard, so the guard's MECHANISM is forced here instead. Without this, the guard
    would be shipped never having fired even once.
    """
    import numpy as np
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.excitations import QuasiparticleSolver
    from tnkit.idmrg import tfim_bulk_mpo

    be = get_backend("numpy")
    g, chi = 1.5, 8
    gs = ground_state(g, chi, n_iter=200, tol=1e-12)
    W = be.array(tfim_bulk_mpo(g))
    solver = QuasiparticleSolver(gs["AL"], gs["AR"], gs["C"], W, be)

    # shift the energy zero far positive so every excitation energy comes out negative
    solver.lambda0 += 1.0e6
    with _pytest.raises(RuntimeError, match="negative"):
        solver.dispersion_at(0.7)


def test_gmres_residual_guard_fires():
    """The geometric-series residual check must raise when the solve does not converge."""
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.excitations import _solve_channel

    be = get_backend("numpy")
    D, dW = 8, 3                                   # n = 192 unknowns
    rng = np.random.default_rng(0)
    src = be.array((rng.standard_normal(D * dW * D)
                    + 1j * rng.standard_normal(D * dW * D)).reshape(D, dW, D))

    # A scaled cyclic shift needs O(n) Krylov vectors to invert. Capping the solver at a single
    # outer iteration (48 vectors for 192 unknowns) therefore leaves a large residual, which is
    # exactly the "converged=False but returned anyway" situation the guard exists to catch.
    # NOTE: an EXACTLY singular operator is not a good test -- GMRES raises LinAlgError from its
    # own least-squares solve before the guard is reached.
    def shift(x):
        flat = np.asarray(x).reshape(-1)
        return be.array(0.99 * np.roll(flat, 1)).reshape(x.shape)

    with _pytest.raises(RuntimeError, match="did not converge"):
        _solve_channel(src, shift, 1.0 + 0j, D, dW, be, tol=1e-12, maxiter=1)


def test_two_particle_basis_is_orthogonal():
    """One- and two-particle basis states must be mutually orthogonal.

    This is the property the whole variational construction assumes, and it is the LEFT GAUGE
    CONDITION that supplies it -- states with a B insertion at different sites, and states in
    different particle-number sectors, are orthogonal because sum_s AL[s]^dag B[s] = 0 kills the
    connecting transfer. Worth pinning: if it silently failed, the eigenproblem would acquire a
    non-identity overlap matrix and every band energy would be quietly wrong.
    """
    import numpy as np
    from tnkit.backend import get_backend
    from tnkit.excitations import null_space_VL, one_particle_state, two_particle_state

    def ovl(a, b):
        E = np.eye(np.asarray(a[0]).shape[0], dtype=complex)
        for x, y in zip(a, b):
            E = np.einsum("ab,asc,bsd->cd", E, np.asarray(x).conj(), np.asarray(y), optimize=True)
        return abs(np.trace(E))

    be = get_backend("numpy")
    N = 9
    gs = ground_state(1.5, 8, n_iter=200, tol=1e-12)
    AL, AR, C = gs["AL"], gs["AR"], gs["C"]
    VL = null_space_VL(AL, be)
    D = AL.shape[0]
    rng = np.random.default_rng(0)
    X1 = be.array((rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / D)
    X2 = be.array((rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / D)

    s3 = one_particle_state(AL, AR, VL, X1, 3, N, be)
    s5 = one_particle_state(AL, AR, VL, X1, 5, N, be)
    t26 = two_particle_state(AL, AR, C, VL, X1, X2, 2, 6, N, be)
    t37 = two_particle_state(AL, AR, C, VL, X1, X2, 3, 7, N, be)

    assert ovl(s3, s3) > 1e-6                      # nonzero norm
    assert ovl(t26, t26) > 1e-6
    assert ovl(s3, s5) < 1e-12                     # different sites
    assert ovl(t26, t37) < 1e-12                   # different site pairs
    assert ovl(t26, s3) < 1e-12                    # different particle-number sectors


def test_two_particle_state_rejects_bad_indices():
    """n1 < n2 < n_sites is a precondition, not a suggestion."""
    import numpy as np
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.excitations import null_space_VL, two_particle_state

    be = get_backend("numpy")
    gs = ground_state(1.5, 8, n_iter=200, tol=1e-12)
    AL, AR, C = gs["AL"], gs["AR"], gs["C"]
    VL = null_space_VL(AL, be)
    D = AL.shape[0]
    X = be.array(np.eye(D, dtype=complex) / D)
    for n1, n2 in ((5, 3), (3, 3), (3, 20)):
        with _pytest.raises(ValueError):
            two_particle_state(AL, AR, C, VL, X, X, n1, n2, 9, be)


# --------------------------------------------------------------------------- #
# The transfer-operator fixed point that caused the "platform-dependent"
# g_z=0.16 blow-up. Root cause in tnkit.vumps, symptom in tnkit.excitations.
# --------------------------------------------------------------------------- #
def test_traceless_fixed_point_is_rejected():
    """A near-traceless "density matrix" must raise BEFORE it is divided by its own trace.

    This is the exact failure: at g_x=1, g_z=0.16, chi=32 the k=1 Arnoldi returned a unit-norm,
    near-traceless, indefinite matrix (tr = -7e-12 on Linux, +1.003 on macOS -- same state,
    same code, different rounding). `l / tr(l)` then amplified it by 1.4e11, e0 = (l|Y) came out
    2.3e11 instead of -1.401, and the MPO environment reached norm 1.4e15.
    """
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.vumps import _validate_fixed_point

    be = get_backend("numpy")
    D = 8
    rng = np.random.default_rng(0)
    M = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    M = 0.5 * (M + M.conj().T)
    M = M - np.eye(D) * (np.trace(M) / D)          # make it exactly traceless
    with _pytest.raises(RuntimeError, match="traceless"):
        _validate_fixed_point(be.array(M), lambda x, a, b: x, None, D, be, 1e-9, "test")


def test_indefinite_fixed_point_is_rejected():
    """A fixed point with an O(1) trace but negative eigenvalues is still not a density matrix.

    The trace test alone is not enough: an arbitrary mixture of the physical fixed point with the
    spurious null-block one can have a perfectly healthy trace and still be indefinite, in which
    case the regulariser |I)(l| projects out the wrong direction and the geometric series is
    solved in the wrong complement -- a wrong answer with no large numbers anywhere to notice.
    """
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.vumps import _validate_fixed_point

    be = get_backend("numpy")
    D = 8
    M = np.diag(np.array([2.0, -1.0] + [0.0] * (D - 2), dtype=np.complex128))   # trace 1
    with _pytest.raises(RuntimeError, match="positive semi-definite"):
        _validate_fixed_point(be.array(M), lambda x, a, b: x, None, D, be, 1e-9, "test")


def test_fixed_point_residual_is_checked():
    """An unconverged fixed point must raise rather than be used."""
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.vumps import _validate_fixed_point

    be = get_backend("numpy")
    D = 4
    rho = be.array(np.eye(D, dtype=np.complex128) / D)
    with _pytest.raises(RuntimeError, match="fixed-point residual"):
        _validate_fixed_point(rho, lambda x, a, b: 0.5 * x, None, D, be, 1e-9, "test")


def test_energy_channel_residual_guard_fires():
    """_energy_channel must refuse to return an unconverged MPO environment.

    Before this guard the residual was never looked at: the solve stagnated at a relative residual of
    7e-3 after all 300 restarts and the caller happily took the answer.
    """
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.vumps import _energy_channel

    be = get_backend("numpy")
    D = 8
    rng = np.random.default_rng(0)
    Y = be.array(rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D)))
    l = be.array(np.eye(D, dtype=np.complex128) / D)

    # Reproduce the STRUCTURE of the real failure rather than a generic hard system: give the
    # transfer operator a second eigenvalue-1 direction v that the regulariser |I)(l| does not
    # remove (v is orthogonal to both I and l). Then (1 - T + |I)(l|) is exactly singular along
    # v, the right-hand side has a component there, and no amount of restarting can converge.
    # That is precisely what an over-parameterised MPS does: the numerically null block carries
    # its own fixed point, and only one direction is ever projected out.
    v = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    v = 0.5 * (v + v.conj().T)
    v = v - np.eye(D) * (np.trace(v) / D)           # orthogonal to I, hence to l = I/D
    v = (v / np.linalg.norm(v)).reshape(-1)

    def T_extra_fixed_point(x, A, be_):
        flat = np.asarray(x).reshape(-1)
        return be_.array((v * np.vdot(v, flat)).reshape(x.shape))

    with _pytest.raises(RuntimeError, match="did not converge"):
        _energy_channel(Y, T_extra_fixed_point, None, l, D, be, gmres_tol=1e-10)


def test_overparameterised_environment_is_sane():
    """THE regression test: the point that returned m_1 = -2.5e10 on Linux.

    g_x=1, g_z=0.16 needs only ~9 Schmidt values, so chi=32 leaves 23 numerically null
    directions; the isometric AL then acts reducibly and a spurious transfer eigenvalue appears
    within ~1e-3 of 1. Everything asserted here was violated by the earlier code on at least one
    platform. The chi=16/chi=32 agreement is the physics check: the answer must not depend on
    padding the bond dimension past the Schmidt rank.
    """
    from tnkit.backend import get_backend
    from tnkit.excitations import QuasiparticleSolver
    from tnkit.idmrg import mixed_field_bulk_mpo
    from tnkit.vumps import _left_environment, _right_environment

    be = get_backend("numpy")
    gx, gz = 1.0, 0.16
    W_np = mixed_field_bulk_mpo(gx, gz)

    omegas = {}
    for chi in (16, 32):
        gs = ground_state(gx, chi, n_iter=1200, tol=1e-12, gz=gz)
        W = be.array(W_np)
        FL, e0L = _left_environment(gs["AL"], gs["C"], W, be, env_tol=1e-10)
        FR, e0R = _right_environment(gs["AR"], gs["C"], W, be, env_tol=1e-10)

        # the energy densities from the two channels must agree with the variational e0
        assert abs(e0L - gs["e0"]) < 1e-8, (chi, e0L, gs["e0"])
        assert abs(e0R - gs["e0"]) < 1e-8, (chi, e0R, gs["e0"])
        # environments stay bounded (they reached 1.4e15 when the fixed point was wrong)
        assert float(np.linalg.norm(np.asarray(FL))) < 1e6
        assert float(np.linalg.norm(np.asarray(FR))) < 1e6

        solver = QuasiparticleSolver(gs["AL"], gs["AR"], gs["C"], W, be, env_tol=1e-10)
        assert abs(solver.lambda0 + 2.1642428) < 1e-5, (chi, solver.lambda0)
        omegas[chi] = solver.dispersion_at(0.0)

    assert abs(omegas[32] - 2.0316895) < 1e-6, omegas
    assert abs(omegas[32] - omegas[16]) < 1e-6, omegas


# --------------------------------------------------------------------------- #
# XI2 -- a follow-up: the SAME degenerate transfer eigenvalue-1 eigenspace
# also corrupts the reported correlation length, silently and by an order of
# magnitude. The guards above fix the fixed point; these pin the transfer SPECTRUM.
# --------------------------------------------------------------------------- #
def _padded_state(AL8, C8, D, noise=1e-13, seed_sv=1e-15):
    """Embed a converged chi=8 state in a chi=D bond space the way over-parameterisation
    does it in the wild: an exact isometry (polar decomposition) whose extra block is pure
    rounding noise, and a bond matrix whose extra Schmidt values are below any real weight."""
    import numpy as _np
    from tnkit.vumps import _polar_np
    rng = _np.random.default_rng(7)
    B = noise * (rng.standard_normal((D, 2, D)) + 1j * rng.standard_normal((D, 2, D)))
    B[:8, :, :8] = AL8
    ALp = _polar_np(B.reshape(D * 2, D)).reshape(D, 2, D)
    Cp = _np.zeros((D, D), dtype=_np.complex128)
    Cp[:8, :8] = C8
    for i in range(8, D):
        Cp[i, i] = seed_sv
    return ALp, Cp / _np.linalg.norm(Cp)


def test_xi2_guard_recovers_the_unpadded_correlation_length():
    """Padding a converged state must not change its correlation length. Before the guard it
    changed it by tens of percent, upward, with no signal of any kind.

    AL stays an EXACT isometry under padding (it comes out of a polar decomposition), so
    nothing in the norm or the canonical form complains -- but an isometry on a padded space
    acts reducibly, the null block carries its own eigenvalue-1 fixed point, and the spurious
    spectrum just below it is what -1/ln|lambda_2| then reports as a correlation length.
    """
    import numpy as _np
    from tnkit.backend import get_backend
    from tnkit.vumps import _correlation_lengths

    be = get_backend("numpy")
    r8 = ground_state(1.5, 8, n_iter=300, tol=1e-12, seed=1)
    AL8, C8 = _np.asarray(r8["AL"]), _np.asarray(r8["C"])
    xi_true = float(r8["xi"])

    for D in (16, 24):
        ALp, Cp = _padded_state(AL8, C8, D)
        naive, _ = _correlation_lengths(be.array(ALp), D, be, C=be.array(Cp),
                                        on_padding="ignore")
        guarded, _, info = _correlation_lengths(be.array(ALp), D, be, C=be.array(Cp),
                                                on_padding="deflate", return_info=True)
        assert info["schmidt_rank"] == 8, (D, info["schmidt_rank"])
        assert info["deflated"] and info["padding_fraction"] > 0.4, info
        # the padding really does corrupt it: the naive answer is inflated by >20%
        assert naive > 1.2 * xi_true, (D, naive, xi_true)
        # ... and the guard recovers the unpadded value essentially exactly
        assert abs(guarded - xi_true) < 1e-8 * xi_true, (D, guarded, xi_true)
        # the deflation was legitimate: supp(rho) is invariant under AL
        assert info["support_leakage"] < 1e-9, info
        assert info["isometry_defect"] < 1e-9, info


def test_xi2_guard_can_raise_instead_of_deflating():
    """on_padding='raise' turns the silent wrong number into a named, loud error carrying the
    diagnostics -- the lesson above (a bad draw must not be able to return a number)."""
    import numpy as _np
    import pytest as _pytest
    from tnkit.backend import get_backend
    from tnkit.vumps import _correlation_lengths, TransferSpectrumPaddingError

    be = get_backend("numpy")
    r8 = ground_state(1.5, 8, n_iter=300, tol=1e-12, seed=1)
    ALp, Cp = _padded_state(_np.asarray(r8["AL"]), _np.asarray(r8["C"]), 24)
    with _pytest.raises(TransferSpectrumPaddingError, match="measuring the padding"):
        _correlation_lengths(be.array(ALp), 24, be, C=be.array(Cp), on_padding="raise")
    # the error must carry the numbers a human needs to act on
    try:
        _correlation_lengths(be.array(ALp), 24, be, C=be.array(Cp), on_padding="raise")
    except TransferSpectrumPaddingError as exc:
        assert exc.info["schmidt_rank"] == 8 and exc.info["chi"] == 24, exc.info
        assert exc.info["xi2_naive"] > exc.info["xi2"], exc.info


def test_xi2_guard_is_a_noop_when_chi_tracks_the_schmidt_rank():
    """The guard must be invisible on a well-parameterised state -- otherwise it would move
    every published number instead of only the wrong ones. At criticality the entanglement
    spectrum decays slowly and the state fills every one of its chi bond directions, so no
    deflation happens and xi2 is returned bit-identically to the pre-guard code path."""
    from tnkit.vumps import vumps

    for g, chi in ((1.0, 16), (1.5, 12)):
        r = vumps(g, chi=chi, n_iter=400, tol=1e-11, seed=1)
        info = r["xi_info"]
        assert info["schmidt_rank"] == chi, (g, chi, info["schmidt_rank"])
        assert info["padding_fraction"] == 0.0 and not info["deflated"], (g, info)
        assert r["xi"] == info["xi2_naive"], (g, r["xi"], info["xi2_naive"])


def test_reproducer_xi2_is_the_padding_not_the_state():
    """THE physical regression test, on the original failing state.

    g_x=1, g_z=0.16 has 9 Schmidt values above 1e-8 (19 above 1e-12); at chi=32 that leaves
    the transfer operator reducible. Measured on this state:

        chi = 8   rank 8    xi2 = 0.9497
        chi = 16  rank 16   xi2 = 0.9679
        chi = 32  rank 19   xi2 = 10.157 naive  ->  0.9710 guarded

    The naive chi=32 number is a factor of TEN above the physical one and diverging with chi,
    i.e. exactly the failure mode 'the correlation length grows with chi' is normally taken as
    evidence AGAINST. The guarded value continues the converging chi=8/16 sequence.
    """
    xi = {}
    for chi in (16, 32):
        r = ground_state(1.0, chi, n_iter=1200, tol=1e-12, gz=0.16)
        xi[chi] = (float(r["xi"]), r["xi_info"])

    x16, i16 = xi[16]
    x32, i32 = xi[32]
    # chi=16 still tracks the Schmidt rank -- untouched
    assert not i16["deflated"] and i16["schmidt_rank"] == 16, i16
    # chi=32 does not: ~19 of 32 directions carry weight, the rest are rounding. The exact
    # rank flickers by one because the 19th/20th Schmidt values straddle rank_tol -- which is
    # the whole point, and why the CUT is chosen by the isometry defect rather than fixed.
    assert i32["deflated"] and i32["schmidt_rank"] in (18, 19, 20), i32
    assert i32["cut_is_clean"], i32
    # the pre-guard number was at least an order of magnitude out, upward. It is not even
    # reproducible: the same state gives xi2_naive = 10.2 at OMP_NUM_THREADS=4 and 222 at 1.
    assert i32["xi2_naive"] > 9.0, i32["xi2_naive"]
    assert i32["xi2_shift"] > 5.0, i32["xi2_shift"]
    # the guarded chi=32 value agrees with chi=16 to better than 1%, as physics requires
    assert abs(x32 - x16) < 0.01 * x16, (x16, x32)
    assert abs(x32 - 0.9710) < 5e-3, x32
