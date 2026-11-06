"""Tangent-space excitation spectra for uniform iMPS (the quasiparticle ansatz).

Given a VUMPS ground state {AL, AR, C} of a translation-invariant MPO Hamiltonian, build the
momentum-resolved excitation ansatz

    |Phi_p(B)> = sum_n e^{ipn} |... AL AL  B_n  AR AR ...>

and solve the variational eigenproblem for omega(p), the quasiparticle dispersion.

WHY THIS EXISTS
---------------
Every dispersion relation in the project so far comes from the exact free-fermion solution, which
works only because the TFIM is integrable. The moment a longitudinal field or a Z2 gauge constraint
is switched on there is no epsilon(k), hence no v_max, hence no way to predict the Lieb-Robinson
light cone or size a finite-size window. This supplies epsilon(k)
variationally, from the state itself.

THE GAUGE CONDITION IS WHAT MAKES IT WORK
-----------------------------------------
B is parameterised as B = V_L X, where V_L spans the null space of AL (i.e. sum_s AL[s]^dag V_L[s]
= 0). Two things follow, and the second is not cosmetic:

  1. The ansatz has no zero modes -- the parameterisation is injective, so the overlap matrix is the
     identity and the generalised eigenproblem collapses to a plain Hermitian one.
  2. The geometric series below are FINITE. The mixed transfer operator T^{AL,AR} has eigenvalue
     exactly 1 (eigenvector C, since sum_s AL[s]^dag C AR[s] = sum_s AL[s]^dag AL[s] C = C), so
     (1 - e^{i phi} T)^{-1} is singular at phi=0. The gauge condition kills the source term in
     precisely that channel, so the physical answer stays finite.

CONVENTIONS (matched to tnkit.dmrg / tnkit.vumps, do not change independently)
  environments  F[bra, mpo, ket]
  MPO           W[w_left, w_right, s_bra, s_ket]
  MPS tensors   A[left_bond, phys, right_bond]
"""
from __future__ import annotations

import numpy as np

from .backend import get_backend, NumpyBackend
from .krylov import gmres
from .eigensolver import lanczos_ground
from ._idmrg import (tfim_bulk_mpo, mixed_field_bulk_mpo, mixed_field_bond_h,
                    improved_ising_bulk_mpo, improved_triple_h, improved_critical_g,
                    improved_dispersion_exact, SYMANZIK_LAMBDA)
from ._vumps import vumps, _left_environment, _right_environment

_NP = NumpyBackend()


# --------------------------------------------------------------------------- #
# generalised transfer maps: same contractions as update_{left,right}_env, but
# with independent bra and ket tensors (the ground-state versions are the
# special case bra is ket).
# --------------------------------------------------------------------------- #
def _TL(F, bra, ket, W, be):
    """Grow a left environment past one site with independent bra/ket tensors."""
    t1 = be.einsum("pwq,psd->wqsd", F, bra.conj())
    t2 = be.einsum("wqsd,wrsy->qdry", t1, W)
    return be.einsum("qdry,qye->dre", t2, ket)


def _TR(F, bra, ket, W, be):
    """Grow a right environment past one site with independent bra/ket tensors."""
    t1 = be.einsum("xry,usx->ryus", F, bra.conj())
    t2 = be.einsum("ryus,wrsp->yuwp", t1, W)
    return be.einsum("yuwp,vpy->uwv", t2, ket)


def _apply_HAC(L, x, R, W, be):
    """One-site effective Hamiltonian: L--W--R acting on the tensor in the open slot."""
    t1 = be.einsum("awq,qty->awty", L, x)
    t2 = be.einsum("awty,wrst->arsy", t1, W)
    return be.einsum("arsy,bry->asb", t2, R)


# --------------------------------------------------------------------------- #
# null-space isometry
# --------------------------------------------------------------------------- #
def null_space_VL(AL, be):
    """V_L: (D, d, D*(d-1)) isometry with sum_s AL[s]^dag V_L[s] = 0 and V_L^dag V_L = 1.

    Taken as the trailing columns of the full left-singular basis of AL reshaped to a
    (D*d, D) matrix: the leading D columns span its column space, the rest the complement.
    """
    D, d, _ = AL.shape
    Amat = np.asarray(_to_np(AL)).reshape(D * d, D)
    U, _, _ = np.linalg.svd(Amat, full_matrices=True)
    VL = U[:, D:]                                   # (D*d, D*(d-1))
    # guard: the whole construction is silently wrong if this is not a true null space
    resid = np.linalg.norm(VL.conj().T @ Amat)
    if resid > 1e-10:
        raise RuntimeError(f"null_space_VL: gauge condition violated, |V_L^dag A_L| = {resid:.3e}")
    return be.array(VL.reshape(D, d, D * (d - 1)))


def _to_np(x):
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)


def bond_pinv(C, rcond=1e-3):
    """Regularised inverse of the bond matrix. Returns (C^{-1}, number of directions dropped).

    The cutoff acts relative to the largest singular value. 1e-3 is the value validated in the
    two-particle off-shell fix, where it was scanned against two quantities at once: the
    two-packet energy residual (window energy above the vacuum minus omega(+k0) + omega(-k0))
    and the overlap |<R_in(k)|R_out(k)>| of the in- and out-references at the same momentum.
    Below it C^{-1} amplifies small Schmidt directions until every reference collapses onto
    one vector (the energy residual cannot see this; the overlap can); above it the fix stops
    removing the deformation.
    """
    Cn = np.asarray(_to_np(C), dtype=complex)
    U, s, Vh = np.linalg.svd(Cn)
    keep = s > rcond * s[0]
    si = np.zeros_like(s)
    si[keep] = 1.0 / s[keep]
    return (Vh.conj().T * si) @ U.conj().T, int((~keep).sum())


# --------------------------------------------------------------------------- #
# the momentum-dependent geometric sums
# --------------------------------------------------------------------------- #
def _solve_channel(source, apply_T, phase, D, dW, be, tol=1e-10, maxiter=400):
    """Solve (1 - phase * T) x = source by matrix-free GMRES.

    `phase` is e^{-ip} (left sums) or e^{+ip} (right sums). At p=0 the operator is singular in
    the MPO-diagonal identity channels; the gauge condition makes the source orthogonal to the
    offending direction, so GMRES still converges to the (unique, physical) solution in the
    complement.

    The relative residual is CHECKED here rather than assumed. A silently unconverged solve does
    not produce a slightly-wrong dispersion -- it produces an H_eff that is not even bounded, and
    the symptom is a huge negative "excitation energy" (measured -2.5e10 at g_x=1, g_z=0.16,
    chi=32, where chi=16 was clean). Failing loudly here is the difference between a caught bug
    and a corrupted data point.
    """
    def mv(v):
        x = v.reshape(D, dW, D)
        return (x - phase * apply_T(x)).reshape(-1)

    b = source.reshape(-1)
    sol = gmres(mv, b, be, tol=tol, atol=1e-12, maxiter=maxiter)

    # be.norm can hand back a complex-dtype scalar on the complex backends; abs() first.
    nb = abs(complex(be.to_scalar(be.norm(b))))
    if nb > 1e-300:
        rel = abs(complex(be.to_scalar(be.norm(mv(sol) - b)))) / nb
        if not np.isfinite(rel) or rel > max(1e-6, 1e3 * tol):
            raise RuntimeError(
                f"_solve_channel: geometric-series GMRES did not converge "
                f"(relative residual {rel:.3e}, phase={phase:.6g}). The effective excitation "
                f"Hamiltonian would be unbounded; refusing to return a dispersion.")
    return sol.reshape(D, dW, D)


class QuasiparticleSolver:
    """Effective excitation Hamiltonian H_eff(p) acting on the tangent-space parameter X."""

    def __init__(self, AL, AR, C, W, be, env_tol=1e-10, gmres_tol=1e-10):
        self.be = be
        self.AL, self.AR, self.C, self.W = AL, AR, C, W
        self.D, self.d, _ = AL.shape
        self.dW = W.shape[0]
        self.gmres_tol = gmres_tol

        self.LW, self.e0L = _left_environment(AL, C, W, be, env_tol=env_tol)
        self.RW, self.e0R = _right_environment(AR, C, W, be, env_tol=env_tol)
        self.VL = null_space_VL(AL, be)
        self.nX = self.D * (self.d - 1)

        # Energy zero. The VUMPS environments are regularised so the EXTENSIVE ground-state
        # energy is projected out of the identity channel, but that leaves a finite residual
        # offset: the ground state itself satisfies H_AC |AC> = lambda0 |AC> with lambda0 != 0.
        # Excitation energies are measured from that zero, so it must be subtracted. Without
        # this, omega(p) comes out with the right SHAPE but a constant offset -- which is exactly
        # how the bug presented (err = 2.0279 identically at every momentum).
        AC = be.einsum("asb,bc->asc", AL, C)
        HAC = _apply_HAC(self.LW, AC, self.RW, W, be)
        num = be.vdot(AC.reshape(-1), HAC.reshape(-1))
        den = be.vdot(AC.reshape(-1), AC.reshape(-1))
        self.lambda0 = float(np.real(be.to_scalar(num) / be.to_scalar(den)))

    # -- parameterisation ---------------------------------------------------
    def B_of_X(self, X):
        return self.be.einsum("asx,xb->asb", self.VL, X.reshape(self.nX, self.D))

    def X_of_G(self, G):
        return self.be.einsum("asx,asb->xb", self.VL.conj(), G)

    # -- the three diagrams -------------------------------------------------
    def apply(self, Xflat, p):
        be, W = self.be, self.W
        X = Xflat.reshape(self.nX, self.D)
        B = self.B_of_X(X)

        # (1) ket-B in the same slot as the bra derivative
        G = _apply_HAC(self.LW, B, self.RW, W, be)

        # (2) ket-B strictly to the LEFT: sum over separations d>=1 of
        #     e^{-ipd} (T^{AL,AR})^{d-1} applied to T^{AL,B}(L_W)
        src_L = _TL(self.LW, self.AL, B, W, be)
        phaseL = np.exp(-1j * p)
        LB = _solve_channel(src_L, lambda x: _TL(x, self.AL, self.AR, W, be),
                            phaseL, self.D, self.dW, be, tol=self.gmres_tol)
        LB = phaseL * LB
        G = G + _apply_HAC(LB, self.AR, self.RW, W, be)

        # (3) ket-B strictly to the RIGHT: mirror image
        src_R = _TR(self.RW, self.AR, B, W, be)
        phaseR = np.exp(+1j * p)
        RB = _solve_channel(src_R, lambda x: _TR(x, self.AR, self.AL, W, be),
                            phaseR, self.D, self.dW, be, tol=self.gmres_tol)
        RB = phaseR * RB
        G = G + _apply_HAC(self.LW, self.AL, RB, W, be)

        return (self.X_of_G(G) - self.lambda0 * X).reshape(-1)

    def dispersion_at(self, p, tol=1e-10, max_iter=200, seed=0, return_vec=False):
        """Lowest variational excitation energy at momentum p.

        return_vec=True additionally returns the minimising tangent-space vector X. It was already
        being computed by lanczos_ground and discarded; exposing it is what lets an INDEPENDENT
        check be made -- e.g. building the explicit one-particle window from this X and contracting
        its energy against the FL/FR environments, which never forms H_eff and so shares no code
        path with the value returned here.
        """
        be = self.be
        rng = np.random.default_rng(seed)
        n = self.nX * self.D
        x0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x0 = be.array(x0 / np.linalg.norm(x0))
        res = lanczos_ground(lambda v: self.apply(v, p), x0, be,
                             tol=tol, max_iter=max_iter, reorth="full")
        w = float(np.real(res[0]))

        # PHYSICAL INVARIANT, not a numerical tolerance. omega(p) is a variational estimate of an
        # excitation energy ABOVE the ground state, so it cannot be negative beyond solver noise,
        # and it cannot exceed the operator norm of the local Hamiltonian by orders of magnitude.
        # Checking this turns a silent corruption into an exception when nothing upstream noticed.
        #
        # BACKSTOP, NOT THE GUARD. This test was originally the ONLY thing standing between the
        # g_z=0.16 blow-up (m_1 = -2.5e10) and the data, which made a conditioning bug look like a
        # platform bug. It was masking, not diagnosing: the Lanczos had converged perfectly well --
        # to the ground state of a corrupted operator, because the VUMPS MPO environment fed to
        # this solver had norm 1.4e15. The actual defect was an unvalidated transfer-operator fixed
        # point in tnkit.vumps._left_fixed_point plus an unchecked residual in _energy_channel;
        # both now raise at the point of failure. Keep this check anyway -- a
        # cheap invariant on the final answer is worth having -- but do not read a pass here as
        # evidence that the environments are sound.
        if not np.isfinite(w):
            raise RuntimeError(f"dispersion_at(p={p:.6g}): omega is not finite ({w})")
        if w < -max(1e-8, 100 * tol):
            raise RuntimeError(
                f"dispersion_at(p={p:.6g}): omega = {w:.6e} is negative. An excitation energy "
                f"above the ground state cannot be negative; H_eff is corrupted (check the "
                f"geometric-series solves and the lambda0 energy zero).")
        if return_vec:
            return w, res[1]
        return w

    def bands_at(self, p, nbands=2, tol=1e-10, seed=0):
        """The `nbands` LOWEST excitation energies at momentum p, as a sorted array.

        Why not build H_eff densely: the eigenproblem is only D(d-1) x D, which for spin-1/2 is
        D^2 -- 1024 at D=32, trivially diagonalisable. But every matvec costs TWO matrix-free
        GMRES solves for the momentum-dependent geometric series, so forming the dense matrix
        costs D^2 matvecs = O(2000) GMRES solves per momentum. Implicitly-restarted Lanczos needs
        a few dozen instead.

        This is what unlocks the multi-particle spectrum: the paper's index j runs over every
        stable particle below the two-particle threshold, and m_2/m_1 = 1.618 at the E8 point is
        the headline check. It is also how the lattice-to-continuum constant beta is calibrated,
        by matching alpha_2(eta) and alpha_3(eta) to the measured spectrum.

        HARD LIMIT, inherited from the ansatz and not a bug: this is only meaningful BELOW the
        two-particle threshold, E < 2 m_1. Above it the target excitation sits inside a continuum
        of scattering states, the gap the variational method relies on closes, and convergence
        degrades. The paper hits the same wall and cannot resolve m_3 (= 1.989 at E8, within 0.6%
        of 2 m_1). Expect returned bands at or above 2*band[0] to be unreliable, and check rather
        than trust them.
        """
        from scipy.sparse.linalg import LinearOperator, eigsh

        be = self.be
        n = self.nX * self.D
        if nbands >= n:
            raise ValueError(f"bands_at: nbands={nbands} must be < problem size {n}")

        def mv(v):
            out = self.apply(be.array(np.asarray(v).reshape(-1)), p)
            return np.asarray(_to_np(out)).reshape(-1)

        op = LinearOperator((n, n), matvec=mv, dtype=complex)
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        vals = eigsh(op, k=nbands, which="SA", tol=tol, v0=v0,
                     return_eigenvectors=False, maxiter=5000)
        w = np.sort(np.real(np.asarray(vals)))

        # same physical invariant as dispersion_at, applied to every band
        if not np.all(np.isfinite(w)):
            raise RuntimeError(f"bands_at(p={p:.6g}): non-finite band energies {w}")
        if w[0] < -max(1e-8, 100 * tol):
            raise RuntimeError(
                f"bands_at(p={p:.6g}): lowest band {w[0]:.6e} is negative. An excitation energy "
                f"above the ground state cannot be negative; H_eff is corrupted.")
        return w


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def tfim_dispersion_exact(p, g):
    """Exact TFIM single-particle dispersion for H = -sum sx sx - g sum sz."""
    return 2.0 * np.sqrt(1.0 + g ** 2 - 2.0 * g * np.cos(p))


def tfim_excitations(g, chi, momenta, backend="numpy", vumps_iter=400, vumps_tol=1e-12,
                     env_tol=1e-10, gmres_tol=1e-10, verbose=True, *, mpo=None, bond_h=None,
                     triple_h=None):
    """VUMPS ground state at (g, chi), then omega(p) on the given momentum grid.

    `mpo` replaces the TFIM bulk tensor; `vumps` then needs the matching `bond_h` or
    `triple_h` for its energy readout. The exact TFIM comparison is printed only without
    `mpo`."""
    be = get_backend(backend)
    gs = vumps(g, chi, n_iter=vumps_iter, tol=vumps_tol, backend=backend, mpo=mpo,
               bond_h=bond_h, triple_h=triple_h)
    if verbose:
        print(f"  VUMPS: e0={gs['e0']:.12f}  eps={gs['eps']:.3e}  iters={gs['iters']}", flush=True)

    W = be.array(tfim_bulk_mpo(g) if mpo is None else mpo)
    solver = QuasiparticleSolver(gs["AL"], gs["AR"], gs["C"], W, be,
                                 env_tol=env_tol, gmres_tol=gmres_tol)
    out = []
    for p in momenta:
        w = solver.dispersion_at(p)
        out.append(w)
        if verbose and mpo is None:
            ex = tfim_dispersion_exact(p, g)
            print(f"  p={p:+.5f}  omega={w:+.10f}  exact={ex:.10f}  "
                  f"err={abs(w - ex):.3e}", flush=True)
    return np.array(out), gs


def mixed_field_excitations(g, h, chi, momenta, backend="numpy", vumps_iter=400,
                            vumps_tol=1e-12, env_tol=1e-10, gmres_tol=1e-10, verbose=True):
    """omega(p) for the MIXED-FIELD Ising chain H = -sum sx sx - g sum sz - h sum sx.

    A longitudinal field breaks integrability, so there is no exact epsilon(k) here. This is the
    case the tangent-space machinery exists for: the dispersion comes from the state itself.

    The two available checks are (a) continuity into the exact TFIM answer as h -> 0, and
    (b) convergence in chi.
    """
    return tfim_excitations(g, chi, momenta, backend=backend, vumps_iter=vumps_iter,
                            vumps_tol=vumps_tol, env_tol=env_tol, gmres_tol=gmres_tol,
                            verbose=verbose, mpo=mixed_field_bulk_mpo(g, h),
                            bond_h=mixed_field_bond_h(g, h))


def improved_excitations(g, h, lam, chi, momenta, nbands=1, backend="numpy",
                         vumps_iter=400, vumps_tol=1e-12, env_tol=1e-10,
                         gmres_tol=1e-10, verbose=True, init=None):
    """Bands of the tree-level Symanzik-IMPROVED mixed-field chain

        H = -sum sx sx - g sum sz - h sum sx - lam sum sx_i sz_{i+1} sx_{i+2}

    lam = 0 reduces exactly to mixed_field_excitations; lam = SYMANZIK_LAMBDA cancels the
    T^2 + Tbar^2 (spin-4, dimension-4) lattice operator, which is the dominant irrelevant
    correction for 2d Ising in a magnetic field (Caselle-Grinza-Magnoli hep-th/0103263).
    See tnkit.idmrg for the derivation of lam* = (sqrt(3)-2)/4.

    Returns (bands, gs) with bands of shape (len(momenta), nbands).
    """
    be = get_backend(backend)
    W_np = improved_ising_bulk_mpo(g, h, lam)
    th = improved_triple_h(g, h, lam)
    gs = vumps(g, chi, n_iter=vumps_iter, tol=vumps_tol, backend=backend,
               mpo=W_np, triple_h=th, init=init)
    if verbose:
        print(f"  VUMPS(g={g}, h={h}, lam={lam}): e0={gs['e0']:.12f}  "
              f"eps={gs['eps']:.3e}  iters={gs['iters']}", flush=True)

    solver = QuasiparticleSolver(gs["AL"], gs["AR"], gs["C"], be.array(W_np), be,
                                 env_tol=env_tol, gmres_tol=gmres_tol)
    out = np.array([solver.bands_at(p, nbands=nbands) for p in momenta])
    return out, gs


# --------------------------------------------------------------------------- #
# two-particle basis states
# --------------------------------------------------------------------------- #
def two_particle_state(AL, AR, C, VL, X1, X2, n1, n2, n_sites, be, fix_C=True, rcond=1e-3):
    """The basis state |(n1, B1), (n2, B2)> as an explicit finite window of tensors.

    THE LEFT INSERTION IS OFF SHELL UNLESS fix_C=True.
    Pushing the middle A_L region into the right gauge with A_L C = C A_R gives

        A_L^i B1 A_L^m B2 A_R^t = A_L^i (B1 C) A_R^m (C^{-1} B2) A_R^t,

    and only the right-hand form has each insertion in the standard one-particle slot (A_L left,
    A_R right, both environments the identity). So with the plain B1 the LEFT particle is built
    from the deformed tangent vector B1 C, whose Rayleigh quotient against H_eff(k) is above
    omega(k) by an amount set by the spread of C's singular values -- i.e. it gets WORSE with
    chi_vac while omega(k) does not move. Measured: that Rayleigh quotient reproduces the whole
    measured off-shell residual of the window to 0.35% over chi_vac = 8..48.

    fix_C=True applies B1 -> B1 C^{-1} (regularised, cutoff `rcond`), which drops the two-packet
    residual onto the one-packet envelope floor at every bond dimension. It is ON by default --
    the corrected construction is the correct one, and the (fix_C, rcond=1e-3) pair is the one
    validated against both the two-packet energy residual and the same-momentum in/out
    reference overlap (see `bond_pinv`). Pass fix_C=False only to reproduce the legacy
    construction for A/B comparison; older archived results used that legacy construction and
    are reproduced at their recorded code_commit, not at HEAD.

    Structure, for n1 < n2:

        AL ... AL  B1  AR ... AR  B2  AR ... AR
                   ^n1            ^n2

    AL to the left of the first insertion, AR everywhere to the right of it. Both B tensors are
    built from the SAME null-space isometry V_L, so each independently satisfies the left gauge
    condition sum_s AL[s]^dag B[s] = 0.

    TWO USES, and they are the same object:

      1. Scattering (their Eq. 15). Superposing these with Gaussian coefficients in n1, n2 gives a
         two-particle wave packet in-state. The paper notes such a state is representable at bond
         dimension 2*D_vac; this routine builds the individual basis states, not the superposition.

      2. Containing the continuum. The single-B ansatz spans only one-particle states, so once a
         would-be bound state rises above 2*m_1 the variational minimum in that space IS the
         two-particle continuum and the bound state is swallowed -- this shows up as band 2
         crossing 2*m_1 and tracking the continuum edge from there on. That is a property of
         the ansatz, not a tolerance to tighten. Including
         states like these in the basis represents the continuum explicitly, so a genuine
         near-threshold bound state can separate from it instead of being lost.

    GAUGE BETWEEN THE INSERTIONS -- this was wrong, and here is the correction.

    An earlier version used AR throughout the region right of n1, i.e. AL..AL B1 AR..AR B2 AR..AR,
    on the reasoning that it kept every tensor right-canonical there. That is inconsistent: BOTH B
    tensors are built from V_L, the null space of AL, so each satisfies sum_s AL[s]^dag B[s] = 0 --
    a condition referred to AL. Putting B2 in a region whose left neighbours are AR gauge-fixes it
    against a tensor that is not there.

    The symptom was decisive and is worth recording, because it looked like a physical result rather
    than a bug: with AR in the middle, the two-particle energy is INDEPENDENT OF SEPARATION to ten
    digits -- 4.8848208102 at separations 8, 12, 16, 20, 24, 28 and 32 alike (g=1.5, chi=16, 81-site
    window, X the variational minimum). Two excitations 32 sites apart in a gapped model cannot have
    exactly the energy they have at 8 sites. The second insertion was simply not contributing as an
    independent excitation.

    Correct is AL to the left of EVERY insertion and AR to the right of every insertion,
    AL..AL B1 AL..AL B2 AR..AR, which is the standard multi-particle tangent-space ansatz and is
    genuinely "the one-particle ansatz applied twice". With it the energy converges with separation
    and reaches 7.6567 against a non-interacting target of 2*E1 = 7.5285 (a 1.7% residual that is
    still under investigation -- a candidate is that using the SAME X for both particles is not a
    clean two-particle state, since Ising domain walls are fermionic).

    """
    if not (0 <= n1 < n2 < n_sites):
        raise ValueError(f"two_particle_state: need 0 <= n1 < n2 < n_sites, got {n1}, {n2}, {n_sites}")
    D = AL.shape[0]
    B1 = be.einsum("asx,xb->asb", VL, X1.reshape(-1, D))
    B2 = be.einsum("asx,xb->asb", VL, X2.reshape(-1, D))
    if fix_C:
        Cinv, _ = bond_pinv(C, rcond)
        B1 = be.array(np.einsum("asb,bc->asc", _to_np(B1), Cinv))
    win = []
    for j in range(n_sites):
        if j == n1:
            win.append(B1)
        elif j == n2:
            win.append(B2)
        elif j < n2:
            win.append(AL)          # AL to the left of EVERY insertion -- see the gauge note above
        else:
            win.append(AR)
    return win


def one_particle_state(AL, AR, VL, X, n, n_sites, be):
    """|(n, B)> as an explicit window: AL left of n, B at n, AR right of n."""
    if not (0 <= n < n_sites):
        raise ValueError(f"one_particle_state: need 0 <= n < n_sites, got {n}")
    D = AL.shape[0]
    B = be.einsum("asx,xb->asb", VL, X.reshape(-1, D))
    return [AL if j < n else (B if j == n else AR) for j in range(n_sites)]
