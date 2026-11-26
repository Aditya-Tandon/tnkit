"""VUMPS, the variational uniform MPS algorithm (Zauner-Stauber, Vanderstraeten, Fishman,
Verstraete and Haegeman, PRB 97, 045145 (2018); tangent-space notes arXiv:1810.07006).

VUMPS optimises directly on the manifold of uniform MPSs of fixed bond dimension, with
self-consistent environments, so it reaches the critical fixed point and gives the central
charge c and the finite-entanglement exponent kappa from the chi scaling. Pure-growth iDMRG
(`tnkit.idmrg`) is not variational on that manifold and converges poorly at criticality.

State: a single-site uniform MPS {AL, AR, AC, C} with
  AL left-canonical  (Σ_s AL_s† AL_s = I),   AR right-canonical (Σ_s AR_s AR_s† = I),
  AC = AL·C = C·AR   (centre tensor and bond matrix).
Index order: A[a,s,b] (left bond, physical, right bond); C[a,b]; MPO W[wl,wr,s,s'].

Each iteration:
  1. left and right MPO environments FL, FR by the triangular recursion; the energy
     (identity-diagonal) channel needs a regularised geometric-series solve with the leading
     transfer eigenvalue projected out. It is matrix-free: the fixed point is a k=1 Arnoldi
     eigenvector and the projected system is solved by GMRES, with the transfer applied as an
     O(D^3 d) einsum (never a dense D^2 x D^2 matrix).
  2. new AC = ground state of H_AC (`apply_one_site` with FL, W, FR); new C = ground state of
     the zero-site H_C; both by the matrix-free `lanczos_ground`.
  3. AL and AR from polar decompositions of AC and C.
Convergence: ε = max(‖AC − AL·C‖, ‖AC − C·AR‖) (the tangent-space gradient) → 0.

Every heavy contraction (the transfer einsums, the environment recursions, the two Lanczos
solves, the polar SVDs) goes through a backend (`backend=`: numpy, torch-cpu, torch-cuda), so
the optimisation runs on a GPU with no host round trip on the O(D^3) work. The Arnoldi fixed
point and the GMRES energy channel are the backend-native Krylov solvers of `tnkit.krylov`;
only their small projected problems run in host numpy.
"""
from __future__ import annotations
import dataclasses
from typing import Any, Dict, List, Optional, Union

import numpy as np

from . import _options
from ._idmrg import tfim_bulk_mpo
from .eigensolver import lanczos_ground
from ._dmrg import apply_one_site, update_left_env, update_right_env
from .backend import NumpyBackend, get_backend
from .krylov import eigs_dominant, gmres
from .models import SX, SZ

_NP = NumpyBackend()


def _real(be, x):
    """A real-valued backend scalar -> python float."""
    return float(be.to_scalar(x).real)


def _energy_density(AC, AR, g, be):
    """TFIM energy per site of the uniform state, from local expectation values:
    e0 = -<Sx_i Sx_{i+1}> - g<Sz_i>. This is exact given the (converged) state, and
    is the robust readout — the MPO energy *channel* of the left/right environments
    carries a subtle additive normalisation that does not affect the optimisation (a
    constant shift of H_AC leaves its ground state unchanged) but is unreliable as an
    absolute energy. Center tensor AC carries the physical one-site state; the bond
    <Sx Sx> uses the two-site block AC(site i)·AR(site i+1)."""
    sx, sz_op = be.array(SX), be.array(SZ)
    AC = AC / be.norm(AC)
    sz = _real(be, be.einsum("asb,st,atb->", AC.conj(), sz_op, AC))
    th = be.einsum("asc,ctb->astb", AC, AR)
    th = th / be.norm(th)
    sxsx = _real(be, be.einsum("astb,su,tv,auvb->", th.conj(), sx, sx, th))
    return float(-sxsx - g * sz)


def _bond_energy_density(AC, AR, bond_h, be):
    """Energy per site for a general nearest-neighbour model: <h> on the two-site block.

    Same reasoning as _energy_density above -- the MPO energy *channel* carries an additive
    normalisation that is harmless for the optimisation but unreliable as an absolute energy,
    so the readout goes through a local expectation value instead. `bond_h` is (d,d,d,d)
    indexed [s_out, t_out, s_in, t_in] and must already account for how the single-site terms
    are split across bonds (for a uniform chain, attributing each site's field wholly to the
    bond on its right is exact).
    """
    th = be.einsum("asc,ctb->astb", AC, AR)
    th = th / be.norm(th)
    val = be.einsum("astb,stuv,auvb->", th.conj(), be.array(bond_h), th)
    return float(_real(be, val))


def _triple_energy_density(AC, AR, triple_h, be):
    """Energy per site for a model with up to three-site terms (the Symanzik-improved
    chain, tnkit.idmrg.improved_triple_h). Same reasoning as _bond_energy_density, one
    site wider: `triple_h` is (d,d,d,d,d,d) as [s,t,u out | s,t,u in] and must already
    attribute each site's fields and right-going bonds to the triple that site starts."""
    th = be.einsum("asc,ctd,dub->astub", AC, AR, AR)
    th = th / be.norm(th)
    val = be.einsum("astub,stuvwx,avwxb->", th.conj(), be.array(triple_h), th)
    return float(_real(be, val))


# --------------------------------------------------------------------------- #
# matrix-free transfer operators + environment solves (no dense D^2 x D^2)
# --------------------------------------------------------------------------- #
# Each transfer maps a D x D matrix X to a D x D matrix in O(D^3 d) via a backend einsum,
# so the environments never build or factorise a D^2 x D^2 matrix, and the contraction
# runs on the chosen device. The fixed point is a k=1 Arnoldi eigenvector and the energy
# channel is a GMRES solve (tnkit/krylov.py) — the "matrix-free GMRES environment" that
# lets VUMPS reach large χ / run on-GPU (the transfer einsums are the same contraction
# kernels DMRG already dispatches through the backend).


def _tL(X, A, be):      # left transfer:  Σ_s A_s† X A_s  (right fixed point = I, left-canonical)
    return be.einsum("psd,pq,qse->de", A.conj(), X, A)


def _tLadj(Y, A, be):   # its adjoint:    Σ_s A_s Y A_s†  (right fixed point = density matrix l)
    return be.einsum("psd,de,qse->pq", A, Y, A.conj())


def _tR(X, A, be):      # right transfer, matching update_right_env's identity self-loop exactly:
    return be.einsum("usx,xy,vsy->uv", A.conj(), X, A)      # Σ_s conj(A[u,s,x]) X A[v,s,y]


def _tRadj(Y, A, be):   # adjoint of the right transfer
    return be.einsum("uv,usx,vsy->xy", Y, A, A.conj())


def _validate_fixed_point(l, apply_adj, A, D, be, tol, where):
    """Hermitise, trace-normalise and VERIFY that `l` really is the reduced density matrix.

    At a chi well above the state's Schmidt rank the eigensolver can return a near-traceless,
    indefinite matrix, which a bare `l / tr(l)` would amplify by orders of magnitude; each
    check below catches one form of that.

    Order matters. The trace test comes before the division, because dividing by a rounding
    residue is the amplification to be caught.
    """
    l = 0.5 * (l + be.einsum("ab->ba", l).conj())
    nrm = abs(complex(be.to_scalar(be.norm(l))))
    tr = complex(be.to_scalar(be.einsum("ii->", l)))
    if nrm <= 1e-300:
        raise RuntimeError(f"{where}: transfer fixed point is the zero matrix.")
    # A density matrix is PSD, so |tr| = ||l||_1 >= ||l||_F: the ratio is O(1) by construction.
    # |tr| << ||l||_F means the eigensolver returned something that is NOT a density matrix --
    # in practice an arbitrary mixture of the physical fixed point with a spurious, traceless
    # one living on the numerically null block of an over-parameterised MPS.
    if abs(tr) < 1e-8 * nrm:
        raise RuntimeError(
            f"{where}: the transfer-operator fixed point is (numerically) traceless -- "
            f"|tr| = {abs(tr):.3e} against ||l||_F = {nrm:.3e}. The eigenvalue-1 eigenspace is "
            f"degenerate, which happens when chi far exceeds the state's Schmidt rank: the "
            f"isometric A then acts reducibly and the null block carries its own spurious fixed "
            f"point. Normalising by this trace would amplify the answer by 1/|tr|. Reduce chi, "
            f"or supply the exact rho_hint (C C^dag / conj(C^dag C)).")
    l = l / tr

    # residual of the fixed-point equation itself
    res = abs(complex(be.to_scalar(be.norm(apply_adj(l, A, be) - l))))
    if not np.isfinite(res) or res > max(1e-6, 1e3 * tol):
        raise RuntimeError(
            f"{where}: fixed-point residual ||T*(l) - l|| = {res:.3e} exceeds tolerance; "
            f"the reduced density matrix has not converged.")

    # positivity: catches a mixture whose trace happens NOT to be small
    ev = np.linalg.eigvalsh(_to_np(0.5 * (l + be.einsum("ab->ba", l).conj())))
    if ev.min() < -1e-7 * max(abs(ev.max()), 1e-300):
        raise RuntimeError(
            f"{where}: fixed point is not positive semi-definite (eigenvalues in "
            f"[{ev.min():.3e}, {ev.max():.3e}]). It is therefore not a reduced density matrix; "
            f"same degenerate-eigenspace cause as above.")
    return l


def _left_fixed_point(apply_adj, A, D, be, tol=1e-9, rho_hint=None):
    """Left fixed point l (eigenvalue 1) of the transfer, as the k=1 dominant Arnoldi
    eigenvector of its adjoint; Hermitian, Tr=1 (the reduced density matrix). Used for
    the energy e0=(l|Y) and the projector that regularises the geometric series. A loose
    Arnoldi tol is ample — the fixed point feeds the H_AC eigenvector, not the reported
    energy (which comes from expectation values).

    `rho_hint` is the EXACT closed form of this fixed point, which the caller knows: for the
    left environment it is C C^dag, for the right environment conj(C^dag C). Supplying it is
    not an optimisation: without it the fixed point can be wrong with no warning (below).

    WHY THE ARNOLDI ALONE IS NOT SAFE
    ---------------------------------
    A is an exact isometry (it comes out of a polar decomposition), so eigenvalue 1 of the
    transfer operator is guaranteed to exist -- but its multiplicity equals the number of
    decoupled blocks of the channel, and is only one when A acts irreducibly. At a chi far above
    the state's Schmidt rank the numerically null block decouples: a spurious second eigenvalue
    appears close to 1, and a k=1 Arnoldi returns an arbitrary mixture of the two. The spurious
    fixed point is traceless, so tr(l) becomes a difference of near-equal numbers whose sign and
    size depend on rounding (and so on the machine). `l / tr(l)` then scales the "density
    matrix" by a huge factor, e0 = (l|Y) is wrong by as much, the energy channel stops
    converging, and anything built on the environments (such as the excitation solver) is
    corrupted.

    Starting the Arnoldi from rho_hint (rather than from the identity) keeps the Krylov space
    inside the physical block, and _validate_fixed_point then refuses anything that is not a
    density matrix. If the Arnoldi answer fails validation but rho_hint itself is a valid fixed
    point, rho_hint is used: it is exact, so there is nothing to gain from the iteration.
    """
    v0 = be.array(np.eye(D).reshape(-1)) if rho_hint is None else rho_hint.reshape(-1)
    apply = lambda x: apply_adj(x.reshape(D, D), A, be).reshape(-1)
    try:
        _, vecs = eigs_dominant(apply, v0, be, k=1, want_vectors=True, tol=tol)
        return _validate_fixed_point(vecs[0].reshape(D, D), apply_adj, A, D, be, tol,
                                     "_left_fixed_point")
    except RuntimeError:
        if rho_hint is None:
            raise
        return _validate_fixed_point(rho_hint, apply_adj, A, D, be, tol,
                                     "_left_fixed_point[rho_hint]")


def _energy_channel(Y, apply_T, A, l, D, be, gmres_tol=1e-9):
    """Solve the regularised geometric series for the MPO energy channel:
    x = Y + T[x] has no solution (T has eigenvalue 1), so extract e0=(l|Y) and solve the
    projected system (1 - T + |I)(l|) x = Y - e0 I by matrix-free GMRES. Returns (x, e0).

    The residual is CHECKED. It used not to be, and that is how a stagnating solve (relative
    residual 7e-3 after all 300 restarts) got to return an environment of norm 1.4e15 that
    every downstream energy and excitation silently inherited.
    """
    I = be.array(np.eye(D))
    vI, vl = I.reshape(-1), l.reshape(-1)
    e0 = _real(be, be.vdot(vl, Y.reshape(-1)))

    def mv(x):                                    # (1 - T)x + I·(l|x)
        x2 = x.reshape(D, D)
        s = be.vdot(vl, x)                        # (l|x) scalar
        return (x2 - apply_T(x2, A, be)).reshape(-1) + vI * s

    b = Y.reshape(-1) - e0 * vI
    x = gmres(mv, b, be, tol=gmres_tol, atol=1e-11, maxiter=300)
    nb = abs(complex(be.to_scalar(be.norm(b))))
    if nb > 1e-300:
        rel = abs(complex(be.to_scalar(be.norm(mv(x) - b)))) / nb
        if not np.isfinite(rel) or rel > max(1e-6, 1e3 * gmres_tol):
            raise RuntimeError(
                f"_energy_channel: the regularised geometric series did not converge "
                f"(relative residual {rel:.3e}). The MPO environment would be wrong by an "
                f"unbounded amount; refusing to return it. Usual cause: chi far above the "
                f"state's Schmidt rank, which puts a spurious transfer eigenvalue next to 1.")
    return x.reshape(D, D), e0


class EnergyReadoutError(RuntimeError):
    """`vumps` was given a Hamiltonian it cannot read an energy density off.

    Raised when a custom `mpo` comes without the matching `bond_h` or `triple_h`. The only
    readout left would be the TFIM formula −⟨sx_i sx_{i+1}⟩ − g⟨sz_i⟩ of `_energy_density`,
    which drops every term the custom MPO adds (for the mixed-field chain, h⟨sx⟩ per site)
    while the state itself is correct, so nothing else would show the error. The caller knows
    its Hamiltonian and must pass its local energy operator. See `_check_energy_readout`.
    """


def _check_energy_readout(g, mpo, bond_h, triple_h, on_mpo_mismatch):
    """Refuse to report a TFIM e₀ for a Hamiltonian that is not the TFIM at this g.

    Called once, before the optimisation loop, so a mis-specified run stops at once. If
    `bond_h` or `triple_h` is given the readout is the caller's own operator and nothing is
    checked. Otherwise the readout is `_energy_density(..., g)`, correct only if `mpo` is
    `tfim_bulk_mpo(g)`, so that is what is compared.

    on_mpo_mismatch: "raise" (default) | "ignore" (return the TFIM readout anyway).
    """
    if on_mpo_mismatch not in ("raise", "ignore"):
        raise ValueError(f"on_mpo_mismatch must be raise|ignore, got {on_mpo_mismatch!r}")
    if mpo is None or bond_h is not None or triple_h is not None:
        return
    W_tfim = tfim_bulk_mpo(g)
    # A device-resident MPO (a CUDA tensor) makes `np.asarray` raise a torch TypeError; such
    # an MPO is treated as a mismatch, with the reason in the message.
    try:
        W = np.asarray(mpo)
        same = bool(W.shape == W_tfim.shape
                    and np.allclose(W, W_tfim, atol=1e-12, rtol=0.0))
    except Exception as exc:                           # device tensor, ragged object, ...
        W, same = None, False
        why = f"could not be compared to tfim_bulk_mpo({g}) ({type(exc).__name__}: {exc})"
    else:
        why = None
    if same:
        return                                        # custom MPO that IS the TFIM: readout valid
    if on_mpo_mismatch == "ignore":
        return
    if why is not None:
        dev = float("nan")
    else:
        dev = np.max(np.abs(W - W_tfim)) if W.shape == W_tfim.shape else np.inf
    detail = why if why is not None else f"max |W - W_TFIM| = {dev:.3e}"
    raise EnergyReadoutError(
        f"vumps(g={g}) was given a custom `mpo` that is not tfim_bulk_mpo({g}) "
        f"({detail}) and neither `bond_h` nor `triple_h`. The only energy "
        f"readout available would be the TFIM formula -<sx sx> - g<sz>, which silently drops "
        f"every extra term (for the mixed-field chain, -h<sx> per site). Refusing to return "
        f"e0.\n"
        f"  Fix: vumps(g, chi, ..., mpo=mixed_field_bulk_mpo(g, h), "
        f"bond_h=mixed_field_bond_h(g, h))\n"
        f"  (three-site models: triple_h=improved_triple_h(g, h, lam); to return the TFIM "
        f"readout anyway: on_mpo_mismatch='ignore')")


#: The two internals a caller may replace (for instance to compare another fixed-point or
#: energy-channel solver), in one mutable registry. A dict rather than module globals: every
#: path to it (`tnkit._vumps._HOOKS`, `tnkit.vumps._HOOKS` through the shim and through the
#: exported function's attribute) is the same object, so a replacement is seen by the
#: functions below whichever path was used. Rebinding a global through `tnkit.vumps`, which
#: is the function (see `tnkit/_shim.py`), would not reach this module. `_EIGSOLVERS` in
#: `tnkit/_dmrg.py` has the same shape.
_HOOKS = {
    "left_fixed_point": _left_fixed_point,
    "energy_channel": _energy_channel,
}


class TransferSpectrumPaddingError(RuntimeError):
    """The transfer spectrum is not a property of the state: χ exceeds its Schmidt rank.

    Raised by `_correlation_lengths(..., on_padding="raise")`. See that function for the
    mechanism. Carries the diagnostics as `.info`.
    """

    def __init__(self, msg, info=None):
        super().__init__(msg)
        self.info = info or {}


def _support_projector(C, be, rank_tol):
    """Isometry P (D×r) onto the support of the reduced density matrix ρ = C C†, and r.

    In the VUMPS mixed gauge AC = AL·C, so the half-chain reduced density matrix expressed
    in AL's *right* bond basis is exactly ρ = C C† = U Σ² U†. Its support is span(U[:, :r])
    with r the number of singular values above `rank_tol` **relative to the largest**. For a
    uniform iMPS the left and right bond spaces are the same space in the same basis, so the
    same P deflates both legs of the transfer operator.
    """
    U, s, _ = be.svd(C)
    s = be.to_numpy(s).real
    s = s / s[0]
    r = int(np.sum(s > rank_tol))
    return U[:, :r], r, s


def _deflate(AL, P, r, be):
    """Restrict both bond legs of AL to the r-dimensional subspace spanned by P.

    Returns (ALr, leakage, isometry_defect). `leakage` = ‖(1 − PP†) AL P‖ measures whether the
    subspace is invariant under AL; `isometry_defect` = ‖Σ_s ÃL_s† ÃL_s − 1‖ measures whether
    the restricted tensor is still a left-canonical MPS tensor. Both are ~0 exactly when the
    cut separated real Schmidt weight from rounding, and both blow up when it did not — which
    is what makes the cut self-verifying rather than a magic constant.
    """
    ALr = be.einsum("ai,asb,bj->isj", P.conj(), AL, P)
    PPd = be.einsum("ai,bi->ab", P, P.conj())
    ALP = be.einsum("asb,bj->asj", AL, P)
    leak = float(abs(complex(be.to_scalar(
        be.norm(ALP - be.einsum("ab,bsj->asj", PPd, ALP))))))
    isod = float(abs(complex(be.to_scalar(
        be.norm(be.einsum("isj,isk->jk", ALr.conj(), ALr)
                - be.array(np.eye(r, dtype=np.complex128)))))))
    return ALr, leak, isod


def _transfer_spectrum(AL, D, be, k=4):
    """Leading k eigenvalues of x ↦ Σ_s AL_s† x AL_s, |λ| descending."""
    k = min(k, D * D - 2)
    apply = lambda x: _tL(x.reshape(D, D), AL, be).reshape(-1)
    # Start from a generic random vector, NOT the identity: for a left-canonical AL the
    # identity IS the eigenvalue-1 fixed point (Σ_s AL_s† I AL_s = I), so Arnoldi from I
    # breaks down at step 0 and never sees the subdominant λ₂,λ₃ that set ξ. A fixed seed
    # keeps ξ reproducible across backends (the start is host-generated, then moved to be).
    rng = np.random.default_rng(1234)
    v0 = (rng.standard_normal(D * D) + 1j * rng.standard_normal(D * D))
    # Krylov cap scales with D: a fixed m=64 under-resolves λ₂,λ₃ once D²≫64 (D≳16), giving
    # an erratic, non-monotonic ξ₂ at large χ even for a well-converged state. Cap at 256 and
    # allow more restarts so the sub-leading transfer spectrum is resolved.
    ev, _ = eigs_dominant(apply, be.array(v0), be,
                          k=k, want_vectors=False, m=min(D * D - 2, 256), tol=1e-11,
                          restarts=100)
    return np.sort(np.abs(ev))[::-1]              # descending; m[0]≈1


def _xi_from_moduli(m):
    """(ξ₂, ξ₃) from the |λ| moduli, normalised by the leading one."""
    if m.size < 2 or m[0] <= 0:
        return np.inf, np.nan
    m = m / m[0]
    xi2 = -1.0 / np.log(m[1]) if 0 < m[1] < 1 else np.inf
    xi3 = -1.0 / np.log(m[2]) if m.size > 2 and 0 < m[2] < 1 else np.nan
    return float(xi2), float(xi3)


def _correlation_lengths(AL, D, be, C=None, rank_tol=1e-12, on_padding="deflate",
                         shift_tol=0.02, cut_max=1e-8, iso_tol=1e-2, return_info=False):
    """The two leading transfer correlation lengths ξ₂,ξ₃ (from the 2nd and 3rd transfer
    eigenvalues; the largest, ≈1, is the norm). ξ₂ is the standard correlation length —
    biased LOW at finite χ (Rams–Czarnik–Cincio, PRX 2018). Returning ξ₃ as well lets the
    FES driver form the refinement parameter δ = 1/ξ₂ − 1/ξ₃ = ln(|λ₃|/|λ₂|) → 0 as χ→∞
    and extrapolate ξ₂(δ→0) to the unbiased value.

    WHY THIS NEEDS A GUARD
    ----------------------
    This reads λ₂, λ₃ off the SAME transfer operator whose eigenvalue-1 eigenspace stops
    being simple when χ far exceeds the state's Schmidt rank. AL is an exact isometry, but an
    isometry on a padded space acts **reducibly**: the numerically null block decouples and
    carries its own eigenvalue-1 fixed point, plus a whole spurious spectrum just below it.
    In that regime λ₂ is not the state's subleading transfer eigenvalue at all — it belongs
    to the padding, and ξ₂ = −1/ln|λ₂| measures the noise floor of the polar decomposition.

    The error is large and inflates the correlation length: once χ exceeds the Schmidt rank,
    the undeflated ξ₂ grows with χ while the physical one converges.

    A |λ₂| near 1 is not by itself a sign of padding: at criticality a genuine large ξ₂ also
    puts |λ₂| close to 1. Deflation tells them apart: a physical λ₂ lives inside supp(ρ) and
    survives the projection; a null-block λ₂ vanishes with its block.

    Parameters
    ----------
    C : bond matrix of the same VUMPS state, or None.
        REQUIRED to assess padding — the Schmidt rank comes from it. With C=None the
        function cannot tell padding from physics, computes the naive answer, and marks
        ``info["assessed"] = False``. Callers inside this module always pass it.
    rank_tol : relative singular-value cut that decides WHETHER the state is padded
        (default 1e-12, below the polar-decomposition noise floor of a converged VUMPS run
        and well below the smallest Schmidt values of a critical state).
    cut_max, iso_tol : where the cut is actually TAKEN is then chosen, not assumed — see the
        comment in the body. The ladder runs rank_tol → cut_max in decades and the finest cut
        whose deflated tensor is still isometric to `iso_tol` wins. `info["cut_is_clean"]` is
        False if none qualifies, and `info["cut_sensitivity"]` always reports how much the
        answer moves one decade coarser.
    on_padding : "deflate" (default) | "raise" | "ignore".
        "deflate" restricts the transfer operator to supp(ρ) and reports the deflated ξ.
        "raise" raises TransferSpectrumPaddingError when the deflation moves ξ₂ by more than
        `shift_tol` (relative). "ignore" returns the undeflated ξ.
    return_info : also return the diagnostics dict.
    """
    if on_padding not in ("deflate", "raise", "ignore"):
        raise ValueError(f"on_padding must be deflate|raise|ignore, got {on_padding!r}")

    m_naive = _transfer_spectrum(AL, D, be)
    xi2_n, xi3_n = _xi_from_moduli(m_naive)
    info = {"assessed": C is not None, "chi": int(D), "schmidt_rank": None,
            "padding_fraction": None, "rank_tol": rank_tol, "deflated": False,
            "xi2_naive": xi2_n, "xi3_naive": xi3_n, "xi2": xi2_n, "xi3": xi3_n,
            "xi2_shift": 0.0, "support_leakage": None, "isometry_defect": None,
            "lambda_naive": m_naive.tolist()}

    if C is None or on_padding == "ignore":
        if C is not None:
            _, r, _ = _support_projector(C, be, rank_tol)
            info["schmidt_rank"] = r
            info["padding_fraction"] = 1.0 - r / D
        return (xi2_n, xi3_n, info) if return_info else (xi2_n, xi3_n)

    P, r, _ = _support_projector(C, be, rank_tol)
    info["schmidt_rank"] = r
    info["padding_fraction"] = 1.0 - r / D
    if r >= D or r < 2:
        # χ tracks the Schmidt rank: the transfer operator is irreducible and λ₂ is physical.
        return (xi2_n, xi3_n, info) if return_info else (xi2_n, xi3_n)

    # ---- choose WHERE to cut, then deflate ----------------------------------------------
    # rank_tol only decides *whether* the state is padded. It is a bad choice of *where* to
    # cut, because a Schmidt tail that straddles it leaves residual near-null directions
    # inside the retained block and they carry the same spurious spectrum. Even a single
    # near-null direction left in the block can make xi2 several times too large, so the cut
    # cannot be a fixed constant.
    #
    # It does not have to be. The restricted tensor must still be a left-canonical MPS tensor
    # on its block: if the cut separated weight from rounding then supp(rho) is invariant
    # under AL and ||sum_s ALr_s^dag ALr_s - 1|| ~ 0; if it cut through real weight, or kept
    # directions that leak into the discarded ones, the defect is an order of magnitude or more
    # larger (good cuts give about 1e-3, bad ones 1e-2 to 1e-1, hence `iso_tol`). So walk the
    # ladder from rank_tol upward and take the FINEST cut that is self-consistent.
    # The isometry defect is an einsum, not an eigensolve, so the walk is nearly free.
    cuts, c = [rank_tol], rank_tol * 10.0
    while c <= cut_max * (1 + 1e-12):
        cuts.append(c)
        c *= 10.0
    tried = []
    chosen = None
    for c in cuts:
        Pc, rc, _ = _support_projector(C, be, c)
        if rc < 2 or rc >= D:
            continue
        ALc, leak, isod = _deflate(AL, Pc, rc, be)
        tried.append({"cut": c, "r": rc, "leak": leak, "iso": isod})
        if chosen is None and isod <= iso_tol:
            chosen = (c, rc, ALc, leak, isod)
    info["cut_ladder"] = tried
    if chosen is None:
        # nothing on the ladder is self-consistent -- fall back to the coarsest tried, and say so
        c, rc, ALc, leak, isod = (tried[-1]["cut"], tried[-1]["r"], None,
                                  tried[-1]["leak"], tried[-1]["iso"]) if tried else \
                                 (rank_tol, r, None, np.nan, np.nan)
        if ALc is None and tried:
            Pc, rc, _ = _support_projector(C, be, c)
            ALc, leak, isod = _deflate(AL, Pc, rc, be)
        info["cut_is_clean"] = False
    else:
        c, rc, ALc, leak, isod = chosen
        info["cut_is_clean"] = True
    if ALc is None:
        return (xi2_n, xi3_n, info) if return_info else (xi2_n, xi3_n)

    m_def = _transfer_spectrum(ALc, rc, be)
    xi2_d, xi3_d = _xi_from_moduli(m_def)
    shift = abs(xi2_n - xi2_d) / max(abs(xi2_d), 1e-300)
    info.update(deflated=True, support_leakage=leak, isometry_defect=isod,
                deflation_cut=c, deflation_rank=rc,
                xi2=xi2_d, xi3=xi3_d, xi2_shift=shift,
                lambda_deflated=m_def.tolist())

    # Is the ANSWER a property of the state, or of where the cut was taken? Re-deflate one
    # decade coarser and report the disagreement, so the number is never quoted without its
    # own systematic. (Cheap: one more k=4 Arnoldi on a smaller block.)
    nxt = next((t for t in tried if t["cut"] > c and t["r"] < rc and t["iso"] <= iso_tol), None)
    if nxt is not None:
        Pn, rn, _ = _support_projector(C, be, nxt["cut"])
        ALn, _, _ = _deflate(AL, Pn, rn, be)
        xi2_c, _ = _xi_from_moduli(_transfer_spectrum(ALn, rn, be))
        info["xi2_coarse_cut"] = xi2_c
        info["coarse_rank"] = rn
        info["cut_sensitivity"] = abs(xi2_d - xi2_c) / max(abs(xi2_d), 1e-300)
    else:
        info["xi2_coarse_cut"] = xi2_d
        info["coarse_rank"] = rc
        info["cut_sensitivity"] = 0.0

    if on_padding == "raise" and shift > shift_tol:
        raise TransferSpectrumPaddingError(
            f"_correlation_lengths: the transfer spectrum is measuring the padding, not the "
            f"state. chi={D} but the Schmidt rank is {r} (padding fraction "
            f"{info['padding_fraction']:.2f} at rank_tol={rank_tol:g}); AL is an isometry on "
            f"a padded space and therefore acts reducibly, so the null block contributes its "
            f"own transfer eigenvalues just below 1. Deflating onto supp(rho)=C C^dag at the "
            f"self-consistent cut {c:g} (r={rc}, isometry defect {isod:.1e}) moves xi2 from "
            f"{xi2_n:.6g} to {xi2_d:.6g} ({shift * 100:.1f}%); one decade coarser (r="
            f"{info['coarse_rank']}) gives {info['xi2_coarse_cut']:.6g}, i.e. "
            f"{info['cut_sensitivity'] * 100:.1f}% residual cut sensitivity. Re-run at "
            f"chi <~ {rc}, or accept the deflated value (on_padding='deflate').", info)
    return (xi2_d, xi3_d, info) if return_info else (xi2_d, xi3_d)


# --------------------------------------------------------------------------- #
# MPO environments for an upper-triangular MPO (TFIM: dW=3)
# --------------------------------------------------------------------------- #
def _left_environment(AL, C, W, be, env_tol=1e-9):
    """Left environment FL (D,dW,D) + energy density e0. Triangular recursion for the
    intermediate channels; the identity-diagonal energy channel via the matrix-free
    regularised solve above. env_tol sets both the Arnoldi fixed-point and the GMRES
    channel tolerances (tunable for the near-critical convergence diagnostic)."""
    D = AL.shape[0]
    dW = W.shape[0]
    FL = be.array(np.zeros((D, dW, D), dtype=np.complex128))
    FL[:, 0, :] = be.array(np.eye(D))            # "nothing yet" channel = identity
    for wr in range(1, dW - 1):                  # intermediate channels: finite range
        FL[:, wr, :] = update_left_env(FL, AL, W, be)[:, wr, :]
    Y = update_left_env(FL, AL, W, be)[:, dW - 1, :]     # energy source (FL[dW-1] still 0)
    # exact fixed point of _tLadj for a left-canonical AL: rho = C C^dag (AL C = AC = C AR).
    rho = be.einsum("ab,cb->ac", C, C.conj())
    l = _HOOKS["left_fixed_point"](_tLadj, AL, D, be, tol=env_tol,
                                   rho_hint=rho / be.einsum("ii->", rho))
    FL[:, dW - 1, :], e0 = _HOOKS["energy_channel"](Y, _tL, AL, l, D, be, gmres_tol=env_tol)
    return FL, e0


def _right_environment(AR, C, W, be, env_tol=1e-9):
    """Right environment FR (D,dW,D) + energy density e0 (mirror of the left)."""
    D = AR.shape[0]
    dW = W.shape[0]
    FR = be.array(np.zeros((D, dW, D), dtype=np.complex128))
    FR[:, dW - 1, :] = be.array(np.eye(D))       # done-identity channel
    for wl in range(dW - 2, 0, -1):
        FR[:, wl, :] = update_right_env(FR, AR, W, be)[:, wl, :]
    Y = update_right_env(FR, AR, W, be)[:, 0, :]
    # mirror closed form: the fixed point of _tRadj for a right-canonical AR is conj(C^dag C)
    rho = be.einsum("ab,ac->bc", C, C.conj())
    l = _HOOKS["left_fixed_point"](_tRadj, AR, D, be, tol=env_tol,
                                   rho_hint=rho / be.einsum("ii->", rho))  # left fp of right T
    FR[:, 0, :], e0 = _HOOKS["energy_channel"](Y, _tR, AR, l, D, be, gmres_tol=env_tol)
    return FR, e0


# --------------------------------------------------------------------------- #
# gauge fixing: AL, AR from AC and C via polar decompositions
# --------------------------------------------------------------------------- #
def _polar(M, be):
    U, _, Vh = be.svd(M)
    return be.einsum("ab,bc->ac", U, Vh)


def _gauge(AC, C, be):
    D, d, _ = AC.shape
    UL = _polar(AC.reshape(D * d, D), be)         # left polar of AC
    UC = _polar(C, be)
    AL = be.einsum("ab,cb->ac", UL, UC.conj()).reshape(D, d, D)   # UL @ UC†
    UR = _polar(AC.reshape(D, d * D), be)         # right polar of AC
    AR = be.einsum("ba,bc->ac", UC.conj(), UR).reshape(D, d, D)   # UC† @ UR
    epsL = _real(be, be.norm(AC - be.einsum("asb,bc->asc", AL, C)))
    epsR = _real(be, be.norm(AC - be.einsum("ab,bsc->asc", C, AR)))
    return AL, AR, max(epsL, epsR)                # both must vanish at the fixed point


# --------------------------------------------------------------------------- #
# VUMPS driver
# --------------------------------------------------------------------------- #
def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _polar_np(M):
    U, _, Vh = np.linalg.svd(M, full_matrices=False)
    return U @ Vh


def _expand(AL, AR, C, new_D, rng, noise=1e-6):
    """Enlarge a converged (AL,AR,C) to bond dimension new_D for χ-ramping (host/numpy;
    a cheap one-off per χ). Embed the old tensors in the top-left block, fill the new
    bond directions with small noise, re-canonicalise. Seeding a large-χ run from the
    converged smaller-χ state (instead of random) is what tames the near-critical
    stiffness — the state already sits near the fixed point, so the ill-conditioned
    environment solves start from a good point and the outer loop converges in far fewer
    iterations (and to the right fixed point, not a stalled one).

    Seed magnitude matters near criticality. Noise added to the whole C diagonal, or new bonds
    seeded with a weight above the true Schmidt tail (about 1e-4 to 1e-5 there), kicks the
    state off the fixed point and can stall the larger-χ run. So the converged block is kept
    exactly, and only the new bonds are seeded, with a tiny deterministic Schmidt weight just
    below the retained spectrum, so that VUMPS grows them from near machine level."""
    AL, AR, C = _to_np(AL), _to_np(AR), _to_np(C)
    D, d, _ = AL.shape
    if new_D <= D:
        return AL, AR, C

    def pad(A):
        B = noise * (rng.standard_normal((new_D, d, new_D)) + 1j * rng.standard_normal((new_D, d, new_D)))
        B[:D, :, :D] = A
        return B

    ALn = _polar_np(pad(AL).reshape(new_D * d, new_D)).reshape(new_D, d, new_D)
    ARn = _polar_np(pad(AR).reshape(new_D, d * new_D)).reshape(new_D, d, new_D)
    # new-bond Schmidt weight: a small fraction of the smallest retained singular value,
    # floored, so the new directions sit just under the real spectrum (never above it).
    s_old = np.linalg.svd(C, compute_uv=False)
    seed = max(1e-10, 0.1 * float(s_old.min()))
    Cn = np.zeros((new_D, new_D), dtype=np.complex128)
    Cn[:D, :D] = C                                # converged block preserved EXACTLY
    for i in range(D, new_D):                     # seed ONLY the new bonds
        Cn[i, i] = seed
    return ALn, ARn, Cn / np.linalg.norm(Cn)


@dataclasses.dataclass
class VUMPSOptions:
    """Settings of `vumps`, apart from `g` and `chi`. Every field can also be passed to `vumps`
    as a keyword, with the same default.

    Parameters
    ----------
    n_iter : int, default 200
        Maximum number of VUMPS iterations.
    tol : float, default 1e-10
        Stop once the tangent-space gradient eps falls below this.
    seed : int, default 0
        Seeds the random initial state (ignored when `init` is given).
    init : dict or None, default None
        {AL, AR, C} of a converged run at a smaller chi to start from (padded by `_expand`);
        None starts from a random left-canonical state.
    verbose : bool, default False
        Print e0 and eps every 20 iterations.
    backend : str or backend, default 'numpy'
        'numpy', 'torch-cpu' or 'torch-cuda' (complex), or a backend object; every O(D^3)
        contraction runs there.
    env_tol : float, default 1e-9
        Tolerance of the Arnoldi fixed point and the GMRES energy channel of the environments.
    local_tol : float, default 1e-10
        Tolerance of the H_AC and H_C Lanczos ground-state solves.
    m_local_cap : int or None, default None
        Krylov cap of those local solves. None: ``min(80, 4 chi)``. An int uses
        ``min(m_local_cap, D d D - 1)``, so it can exceed ``4 chi``; eps cannot fall below the
        local-solve error, and ``min(80, 4 chi)`` can under-resolve H_AC at large chi.
    trace : bool, default False
        Record a per-iteration trajectory (it, e0, eps, local H_AC residual, smallest singular
        value of C) under the key "trace".
    mpo : ndarray or None, default None
        A custom bulk MPO tensor ``(D, D, 2, 2)``; None is `tfim_bulk_mpo(g)`. Only spins 1/2
        (d = 2) are supported; another d raises `ValueError`. The state is
        optimised against it, but `e0` is read off local expectation values, so a custom MPO
        needs the matching `bond_h` or `triple_h`; without one `vumps` raises
        `EnergyReadoutError`.
    bond_h : ndarray or None, default None
        The two-site energy operator, shape ``(d, d, d, d)``, whose expectation is e0 (for
        example `mixed_field_bond_h`).
    triple_h : ndarray or None, default None
        The three-site energy operator, shape ``(d,) * 6``, for three-site models
        (`improved_triple_h`).
    on_mpo_mismatch : {'raise', 'ignore'}, default 'raise'
        'ignore' returns the TFIM readout for a custom `mpo` without `bond_h` or `triple_h`.
    xi_on_padding : {'deflate', 'raise', 'ignore'}, default 'deflate'
        How the correlation lengths treat a chi above the Schmidt rank: 'deflate' restricts the
        transfer operator to the support of C C† first, so xi measures the state and not the
        padding; 'raise' raises `TransferSpectrumPaddingError` when that changes xi; 'ignore'
        returns the undeflated values. See `_correlation_lengths`.
    xi_rank_tol : float, default 1e-12
        Relative singular-value cut defining the Schmidt rank.
    """

    n_iter: int = 200
    tol: float = 1e-10
    seed: int = 0
    init: Optional[Dict[str, Any]] = None
    verbose: bool = False
    backend: Any = "numpy"
    env_tol: float = 1e-9
    local_tol: float = 1e-10
    m_local_cap: Optional[int] = None
    trace: bool = False
    mpo: Optional[Any] = None
    bond_h: Optional[Any] = None
    triple_h: Optional[Any] = None
    on_mpo_mismatch: str = "raise"
    xi_on_padding: str = "deflate"
    xi_rank_tol: float = 1e-12


class UniformMPS(dict):
    """The result of `vumps`: a dict whose keys are also attributes.

    ``gs["AL"]``, ``gs.get("trace")`` and ``"trace" in gs`` keep working; ``gs.AL`` is the
    same object. Keys: e0, S, xi, xi3, svals, eps, iters, xi_info, AL, AR, AC, C, and
    trace when requested. AL/AR/AC/C are on the run's backend.
    """
    __slots__ = ()

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def vumps(g, chi: Optional[int] = None,
          options: Optional[Union[VUMPSOptions, Dict[str, Any]]] = None,
          **legacy_kwargs: Any) -> UniformMPS:
    """Single-site VUMPS ground state of an infinite chain: the TFIM at field g, or a custom
    bulk MPO.

    Parameters
    ----------
    g : float
        Transverse field of the TFIM ``H = -sum sx_i sx_{i+1} - g sum sz_i`` (Pauli
        matrices), critical at g = 1; with a custom `mpo`, the field the energy check
        compares against.
    chi : int
        Bond dimension of the uniform MPS, >= 1.
    options : VUMPSOptions or dict, optional
        Settings; see `VUMPSOptions` for every field, its default and its range.
    **legacy_kwargs
        Any field of `VUMPSOptions`, passed flat; a keyword overrides the same field of
        `options`.

    Returns
    -------
    UniformMPS
        A dict whose keys are also attributes: ``e0`` (energy per site), ``S`` (von Neumann
        entropy of a bond, natural log), ``xi`` and ``xi3`` (correlation lengths in sites from
        the second and third transfer eigenvalues; ``xi3`` with ``xi`` allows an extrapolation
        in chi), ``svals`` (normalised Schmidt values), ``eps`` (final tangent-space
        gradient), ``iters``, ``xi_info`` (diagnostics of the padding check, including the
        undeflated ``xi2_naive``), ``AL``, ``AR``, ``AC``, ``C`` (on the run's backend) and
        ``trace`` when requested.

    Notes
    -----
    Two call styles run the same code::

        gs = vumps(1.0, chi=16, n_iter=400, tol=1e-12)                  # flat
        gs = vumps(1.0, 16, VUMPSOptions(n_iter=400, tol=1e-12))        # grouped
    """
    # `chi` is a keyword for the drivers that call vumps(**kw) with chi in the dict
    # (older call sites). Both spellings at once was a TypeError from
    # the interpreter and stays one, rather than a silently chosen bond dimension.
    if chi is not None and "chi" in legacy_kwargs:
        raise TypeError("vumps() got multiple values for argument 'chi'")
    if chi is None and "chi" not in legacy_kwargs:
        raise TypeError("vumps() missing required argument 'chi' (bond dimension)")
    chi = legacy_kwargs.pop("chi", chi)
    return _vumps_run(g, chi, _options.resolve(VUMPSOptions, options, legacy_kwargs, "vumps"))


def _vumps_run(g, chi: int, o: VUMPSOptions) -> UniformMPS:
    """The optimisation itself; `vumps` documents it."""
    n_iter, tol, seed, init, verbose = o.n_iter, o.tol, o.seed, o.init, o.verbose
    backend, env_tol, local_tol = o.backend, o.env_tol, o.local_tol
    m_local_cap, trace = o.m_local_cap, o.trace
    mpo, bond_h, triple_h = o.mpo, o.bond_h, o.triple_h
    on_mpo_mismatch = o.on_mpo_mismatch
    xi_on_padding, xi_rank_tol = o.xi_on_padding, o.xi_rank_tol

    if mpo is not None:
        shape = tuple(int(n) for n in getattr(mpo, "shape", np.shape(mpo)))
        if len(shape) != 4 or shape[2:] != (2, 2):
            raise ValueError(f"vumps supports d = 2 (spin-1/2) only; got an MPO of shape "
                             f"{shape} (d = {shape[2] if len(shape) == 4 else '?'})")
    # Guard first: a mis-specified energy readout must cost a second, not 600 iterations.
    _check_energy_readout(g, mpo, bond_h, triple_h, on_mpo_mismatch)
    be = get_backend(backend)
    D, d = chi, 2
    rng = np.random.default_rng(seed)

    # seed / ramp is built on host (deterministic across backends), then moved to device
    if init is not None:
        AL_np, AR_np, C_np = _expand(init["AL"], init["AR"], init["C"], D, rng)
        AC_np = np.einsum("asb,bc->asc", AL_np, C_np)
    else:
        A = rng.standard_normal((D, d, D)) + 1j * rng.standard_normal((D, d, D))
        AL_np = _polar_np(A.reshape(D * d, D)).reshape(D, d, D)     # left-canonical seed
        C_np = (np.eye(D, dtype=np.complex128) / np.sqrt(D))
        AC_np = np.einsum("asb,bc->asc", AL_np, C_np)
        AR_np = AL_np.copy()

    AL, AR, C, AC = (be.array(AL_np), be.array(AR_np), be.array(C_np), be.array(AC_np))
    W = be.array(tfim_bulk_mpo(g) if mpo is None else mpo)

    # local eigensolves are warm-started from the current AC/C, so the ground state
    # converges in a modest Krylov space; a small cap keeps the per-iteration cost low.
    # A loose local tol suffices — the OUTER VUMPS loop is what must converge.
    m_local = min(80, 4 * chi) if m_local_cap is None else min(m_local_cap, D * d * D - 1)
    e0 = np.nan
    eps = np.nan
    tr: Optional[List[Any]] = [] if trace else None
    for it in range(n_iter):
        FL, _ = _left_environment(AL, C, W, be, env_tol=env_tol)
        FR, _ = _right_environment(AR, C, W, be, env_tol=env_tol)

        shape_ac = AC.shape
        mv_ac = lambda x: apply_one_site(x.reshape(shape_ac), FL, W, FR, be).reshape(-1)
        _, ac_vec, _ = lanczos_ground(mv_ac, AC.reshape(-1) / be.norm(AC),
                                      be, tol=local_tol, max_iter=m_local, reorth="full")
        AC = ac_vec.reshape(shape_ac)

        mv_c = lambda x: be.einsum("xwa,ab,ywb->xy", FL, x.reshape(D, D), FR).reshape(-1)
        _, c_vec, _ = lanczos_ground(mv_c, C.reshape(-1) / be.norm(C),
                                     be, tol=local_tol, max_iter=m_local, reorth="full")
        C = c_vec.reshape(D, D)

        AL, AR, eps = _gauge(AC, C, be)
        if triple_h is not None:
            e0 = _triple_energy_density(AC, AR, triple_h, be)
        elif bond_h is not None:
            e0 = _bond_energy_density(AC, AR, bond_h, be)
        else:
            e0 = _energy_density(AC, AR, g, be)
        if trace:
            # H_AC eigen-residual of the accepted AC (‖H·v − (v|H|v)v‖) exposes whether the
            # local solve — not the outer loop — is what floors ε; σ_min(C) flags an
            # ill-conditioned bond matrix (the other classic near-critical failure).
            r = mv_ac(AC.reshape(-1))
            lam = be.vdot(AC.reshape(-1), r)
            res_ac = _real(be, be.norm(r - lam * AC.reshape(-1)))
            smin = float(be.to_numpy(be.svd(C)[1]).real.min())
            tr.append((it, float(e0), float(eps), res_ac, smin))
        if verbose and it % 20 == 0:
            print(f"  it={it} e0={e0:.10f} eps={eps:.2e}", flush=True)
        if eps < tol:
            break

    sv = be.to_numpy(be.svd(C)[1]).real
    sv = sv / np.linalg.norm(sv)
    p = sv ** 2
    p = p[p > 1e-16]
    S = float(-np.sum(p * np.log(p)))
    # C is passed so the transfer spectrum can be checked against the Schmidt rank: at
    # chi >> rank the isometric AL acts reducibly and lambda_2 belongs to the padding, not the
    # state (see _correlation_lengths). xi_info carries the diagnostics.
    xi, xi3, xi_info = _correlation_lengths(AL, D, be, C=C, rank_tol=xi_rank_tol,
                                            on_padding=xi_on_padding, return_info=True)
    out = UniformMPS(e0=e0, S=S, xi=xi, xi3=xi3, svals=sv, eps=eps, iters=it + 1,
                     xi_info=xi_info, AL=AL, AR=AR, AC=AC, C=C)
    if trace:
        out["trace"] = tr
    return out
