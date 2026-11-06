"""Batched TDVP — evolve k independent MPS of identical geometry as ONE tensor program.

WHY THIS EXISTS. At small bond dimensions (chi <= 48, d = 2) a TDVP
site-step is not arithmetic-bound, it is *launch*-bound: the measured price of a
dispatched tensor operation is ~22 us against arithmetic that is tens of kFLOP, and
capturing the step as a CUDA graph removes the asking but not the doing -- 71-80% of a
captured site-step is the captured loop's own execution time at tiny sizes. Both numbers
say the same thing: the card is mostly idle, and the way to fill it is not to launch
fewer kernels but to give each kernel more work.

A scan is always many independent legs of identical geometry -- same site count, same
bond cap, same dt, same step count -- differing only in couplings, packet momenta and
field. Stack k of them on a LEADING axis and every contraction, QR, SVD and Krylov step
becomes one batched call on k times the data. The kernel COUNT per site-step is unchanged;
the work per kernel is k times larger. So per-leg cost should fall as ~1/k until the
device saturates, and where it flattens is a measurement, not a guess.

WHERE THE LEG AXIS GOES. Every tensor carries it first::

    M[i] : (k, left_bond, phys, right_bond)       site tensors
    L[i] : (k, bra, mpo, ket)                     left environments
    R[i] : (k, bra, mpo, ket)                     right environments
    W[i] : (k, mpo_left, mpo_right, phys, phys')  the per-leg MPO

so leg n's slice `M[i][n]` is exactly the tensor the unbatched integrator would hold, in
tnkit's own index order (see `tnkit.mps.MPS`). The contractions below are the unbatched
ones of `tnkit._dmrg` with an `n` prepended to every index group; the QR, SVD and
eigendecomposition are `linalg` calls on a leading batch axis.

THE KRYLOV SCHEDULE IS FIXED, AND THAT IS WHAT MAKES BATCHING POSSIBLE. The adaptive
solver stops when a per-block error estimate falls below tolerance, so k legs would stop
at k different iteration counts and the batched loop would have no shape. The fixed-depth
solver runs the same number of iterations for every block by construction, agrees with the
adaptive solver to ~3e-12 -- closer than an over-converged control solver sits to it --
and pays for that with extra matvecs whose price is exactly what batching cuts. Its
breakdown handling needs no host branch, so the batched loop has none either: `guard_denom`
divides safely, and a broken-down direction couples into the projected tridiagonal through
its own off-diagonal, which suppresses it to the order it broke down at.

ONE SCHEDULE, K DEMANDS. A fixed `m` is only as safe as the calibration that froze it, and
legs are defined by this module to differ in exactly the couplings, fields and momenta that
change a leg's own Krylov demand -- so `krylov_m="auto"` calibrates against EVERY leg, not
leg 0 (`_calibrate_krylov_m`), and takes the max. That still does not make the fixed
schedule provably safe for the whole trajectory: a leg's demand can drift as the state
evolves, past whatever the entry-state calibration measured. So every fixed-schedule solve
also carries the standard a-posteriori Krylov error estimate (Saad 1992, the same estimator
`eigensolver.lanczos_expm` uses for its own adaptive stop) computed from quantities the
batched loop already has -- the last Lanczos beta and the last row of the projected
tridiagonal's exponential -- and the per-step, per-leg runtime guard in `_run` checks it
against `krylov_tol`. An under-resolved leg grows `m` for the whole batch on the next step
(bounded by `krylov_m_max`), or the run stops with a loud error naming the leg: never a
quiet, finished, wrong trajectory.

RAGGED BOND DIMENSIONS -- the honest problem. Legs of identical geometry do NOT have
identical bond dimensions while chi is still growing: a two-site sweep with a
discarded-weight budget keeps a different rank in each leg at each bond, and shapes that
differ cannot be stacked. Three strategies are implemented and all three are EXACT; which
one is cheapest before saturation is a measurement, not a preference.

  ragged='pad' (the default) -- hold every stacked bond at the WIDEST rank the cap and the
      chain geometry could ever put there, min(chi_max, d^i * outer_left,
      d^(N-i) * outer_right), and carry a per-leg RANK MASK for the rest. The stacked
      shapes are then fixed for the whole run: no bond ever changes size, so one shape
      class covers the growth phase and the saturated phase alike and nothing would have to
      be re-planned, re-allocated or re-captured mid-run IF this module captured anything --
      see EAGER, NOT CAPTURED below. It pays for the fixed shape in arithmetic -- a bond
      holding rank 12 is computed at width 48 -- which is the trade to measure.

  ragged='tight' -- pad only to the widest rank any leg ACTUALLY keeps at that bond on
      that step, and re-cut it at every truncation. The wasted arithmetic is then only the
      spread BETWEEN the legs rather than the distance to the cap, which is far less; the
      price is that the stacked shapes move as the state grows, so allocations and any
      shape-keyed plan a FUTURE capture layer built on this module would move too.
      Identical trajectory, different bill.

EAGER, NOT CAPTURED. Every call in this module -- every contraction, QR, SVD and Krylov
step, for every ragged mode -- runs eagerly. There is no `torch.cuda.graph`, no
`make_graphed_callables`, no capture of any kind anywhere in this file: the throughput
numbers this module produces are timed against the ALREADY-CAPTURED single-leg production
path (`tnkit.tdvp`, `graph=...`) as the honest baseline, not against another capture. The
'shape class' language above describes what would make a FUTURE combination of batching
and capture possible -- one captured graph per (evolver, ragged, geometry, m) shape instead
of one per leg -- it is not a claim that this module builds or exercises that combination
itself.

  ragged='group' -- partition the legs by their full bond profile ON ENTRY and step each
      group as its own batch. Trivially exact, and it degrades to serial exactly when the
      entry profiles disagree. It is a partition of the ENTRY state only: raggedness that
      develops later inside a group is carried by the same masks as above. The temporal
      version of the same idea -- stay serial through the growth phase and batch only once
      every leg has reached the cap -- needs no mode of its own, being one serial call
      followed by one batched one.

Why the mask is not optional, and why it is what makes padding exact. Zero-padding a bond
changes nothing about the state -- a zero Schmidt value contributes nothing. But the QR
that moves the orthogonality centre returns an ARBITRARY orthonormal completion in the
padded columns, and the environment built from those columns is not zero; the bond solve
that follows then couples the true block to the padded one, and the leg has silently been
promoted onto a larger manifold. That is a different (finer) trajectory, not the one the
serial run takes. Zeroing the padded columns of Q, the padded rows of the bond matrix, and
the padded Schmidt values restores the block structure: every environment stays exactly
zero outside the true rank, every operand the true block sees is the operand it would have
seen alone, and the only residual difference is the ORDER of summation inside contractions
whose extra terms are exact zeros -- the measured reduction-order floor,
four orders below any tolerance here.

WHAT IS NOT CLAIMED. This is a second evolver, not a change to the first: `tdvp1_evolve`
and `tdvp2_evolve` are untouched and remain the reference. The batched path uses the
fixed-depth Krylov solver, so it is not bit-identical to the adaptive production path and
is not meant to be. Two different comparisons, with different numbers:

1. Batched against the SAME fixed-schedule code run one leg at a time (the question "does
   batching change the solver's own answer"). Measured at full collision scale (N = 501, chi 48,
   k = 8): state infidelity <= 2.4e-14; derived quantities are less forgiving because they
   subtract large numbers -- window energy and local observables deviate by up to ~1e-11
   relative and the worst two-site scattering amplitude by ~1.4e-10, i.e. a 1e-12 target on
   derived quantities is missed by roughly 7-9x while the states agree to round-off.

2. Fixed-depth Krylov against the adaptive solver the production integrators use (the
   question "does the fixed schedule match the adaptive one"). That was measured
   separately, on the fixed-depth solver alone: ~3e-12 at small sizes, growing to ~1e-8
   at N = 48, chi = 32 against a measured floor of ~2e-7 there.

A caller choosing this module should know which of the two they are relying on.
"""
from __future__ import annotations
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .backend import get_backend
from .legs import fuse, split
from . import invariants
from .mps import MPS, MPSLike


class BatchGeometryError(ValueError):
    """Raised when the states handed to `evolve_batch` do not share one geometry.

    Batching k legs is only meaningful when they are the SAME program on different data. A
    silent fallback to serial would hide a configuration mistake behind an unexplained
    slowdown, so every geometric mismatch -- site count, physical dimension, MPO bond
    structure, boundary environment shape, outer bonds -- is refused loudly and names the
    leg and the site where the disagreement is.
    """


class BatchKrylovError(RuntimeError):
    """Raised when the fixed Krylov schedule is under-resolved for a leg and cannot be grown.

    `krylov_m` is one FIXED dimension shared by every leg for the whole trajectory, and its
    per-leg adequacy is checked at runtime by a Saad (1992) a-posteriori residual estimate
    (see `_run`). When a leg's estimated residual exceeds `krylov_tol`, growing `m` for the
    next step (bounded by `krylov_m_max`) is tried first; this error is raised only when `m`
    is already at that ceiling and the leg is still under-resolved. Growing `m` further would
    change the batch's shape mid-run without limit, which is not a decision this module makes
    silently -- the caller must raise `krylov_m_max`, pass a larger explicit `krylov_m`, or
    accept a looser `krylov_tol`.
    """


RAGGED_MODES = ("pad", "tight", "group")
EVOLVERS = ("1tdvp", "2tdvp")
EXPM_WHERE = ("eigh", "host")
KRYLOV_HEADROOM = 2           # additive, matching `tdvp.FAST_EXPM_HEADROOM`
KRYLOV_FLOOR = 4
KRYLOV_GROWTH_STEP = KRYLOV_HEADROOM  # how much `m` grows per failed runtime guard check
# The runtime guard's default absolute residual tolerance. Two orders above the ~3e-12
# fixed-vs-adaptive agreement the module docstring measures for a PROPERLY calibrated `m`
# (so a correctly calibrated run has headroom and does not grow), and comfortably above the
# ~1e-10 borderline residual an under-calibrated but not badly-broken small `krylov_m` can
# show on a short, cheap test trajectory -- while still catching the many-orders-of-magnitude
# residual a genuinely too-small `m` produces (see the `krylov_m="auto"` and runtime-guard
# tests in `tests/test_batch_tdvp.py`).
KRYLOV_TOL_DEFAULT = 1e-8


@dataclasses.dataclass
class BatchTDVPOptions:
    """Everything `evolve_batch` can be told, apart from the states, the Hamiltonians and
    the time grid.

    backend      : as everywhere else in tnkit. 'torch-cuda' is the point of this module;
                   'numpy' and 'torch-cpu' work and are what the test suite uses.
    evolver      : '1tdvp' (bond-preserving) or '2tdvp' (grows bonds, truncates at chi_max).
    krylov_m     : the FIXED Krylov dimension, or 'auto' to calibrate it from one metered
                   adaptive sweep run on EVERY leg -- the largest dimension any of those
                   sweeps needed, plus the project's standing additive headroom of two. Not
                   leg 0 alone: legs are defined by this module to differ in exactly the
                   couplings, fields and momenta that change a leg's own Krylov demand, so
                   the max is taken over all of them (`_calibrate_krylov_m`). Whatever `m`
                   is calibrated or given, it is still only a fixed schedule frozen at the
                   ENTRY state -- see `krylov_tol` for the runtime check on the rest of the
                   trajectory.
    krylov_tol   : the runtime per-leg, per-step Krylov residual guard's absolute tolerance
                   (Saad 1992 a-posteriori estimate, computed from quantities the batched
                   Lanczos loop already has -- see `_run`). A leg whose estimated residual
                   exceeds this grows `m` for the whole batch on the next step, bounded by
                   `krylov_m_max`; if `m` is already at that bound, `evolve_batch` raises
                   `BatchKrylovError` naming the leg rather than continuing on a schedule
                   known to be too short. Checking costs one extra device-to-host transfer
                   per trajectory step (all legs and all Krylov solves in that step share
                   it), not one per solve.
    krylov_m_max : the ceiling the runtime guard may grow `krylov_m` to. Defaults to twice
                   the starting `krylov_m` (calibrated or given) when left at `None`; growth
                   is also capped by the local vector length, which `lanczos_expm_fixed_batched`
                   already clamps to.
    expm_where   : 'eigh' (default) exponentiates the k projected tridiagonals in ONE
                   batched eigendecomposition on the device; 'host' transfers them and
                   calls `scipy.linalg.expm` k times, which is the unbatched path's own
                   route and exists here as a correctness control.
    ragged       : 'pad' (default), 'tight' or 'group' -- see the module docstring.
    chi_max      : bond cap. Required for '2tdvp', refused for '1tdvp'.
    svd_tol      : relative discarded-weight budget per bond ('2tdvp' only).
    renorm       : rescale the retained Schmidt values to unit norm ('2tdvp' only).
    L0, RN       : boundary environments -- one shared tensor, or a list of k. A window on
                   an infinite background needs these and must NOT be re-canonicalised.
    canonicalize : right-canonicalise each input state first. False for a window.
    keep_device  : hand `measure` the on-device tensors instead of host copies.
    measure      : called with the k per-leg states every `meas_every` steps.
    stats        : optional dict, filled with the run's diagnostics -- the calibrated
                   Krylov dimension (and, when 'auto', the per-leg calibration sweep that
                   fed it), the per-leg max runtime Krylov residual, whether and how far `m`
                   grew, the group sizes, how full the padded bonds were, the mean
                   discarded weight ('disc') and the PER-LEG discarded weight
                   ('disc_per_leg'), in the order the states were given.
    """

    backend: Any = "numpy"
    evolver: str = "1tdvp"
    krylov_m: Union[int, str] = "auto"
    krylov_tol: float = KRYLOV_TOL_DEFAULT
    krylov_m_max: Optional[int] = None
    expm_where: str = "eigh"
    ragged: str = "pad"
    chi_max: Optional[int] = None
    svd_tol: float = 0.0
    renorm: bool = True
    L0: Optional[Any] = None
    RN: Optional[Any] = None
    canonicalize: bool = True
    keep_device: bool = False
    measure: Optional[Callable[[Any], Any]] = None
    meas_every: int = 1
    stats: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- contractions
#
# The contractions of `tnkit._dmrg` with an `n` prepended to every index group. They are
# spelled out rather than generated from the unbatched equation strings: the index letters
# ARE the interface (see `tnkit.mps.MPS`), and a generated string is one refactor away from
# silently transposing an MPO.

def apply_one_site_b(M, L, W, R, be):
    """(H_eff M)[n,a,s,b] for a batch of single-site tensors. Batched `apply_one_site`."""
    x1 = be.einsum("naky,nysb->naksb", L, M)
    x2 = be.einsum("nklst,naktb->nalsb", W, x1)
    return be.einsum("nbly,nalsy->nasb", R, x2)


def apply_zero_site_b(C, L, R, be):
    """(H_eff C)[n,a,b] for a batch of BOND matrices. Batched `apply_zero_site`."""
    x1 = be.einsum("naky,nyb->nakb", L, C)
    return be.einsum("nbkz,nakz->nab", R, x1)


def apply_two_site_b(theta, L, W1, W2, R, be):
    """(H_eff theta)[n,a,s1,s2,b] for a batch of two-site blocks. Batched `apply_two_site`."""
    x1 = be.einsum("npwq,nqijb->npwijb", L, theta)
    x2 = be.einsum("nwmsi,npwijb->npmsjb", W1, x1)
    x3 = be.einsum("nmrtj,npmsjb->npstrb", W2, x2)
    return be.einsum("ndrb,npstrb->npstd", R, x3)


def update_left_env_b(L, A, W, be):
    """Grow a batch of left environments past left-canonical A[n,a,s,c]."""
    t1 = be.einsum("npwq,npsd->nwqsd", L, be.conj(A))
    t2 = be.einsum("nwqsd,nwrsy->nqdry", t1, W)
    return be.einsum("nqdry,nqye->ndre", t2, A)


def update_right_env_b(R, B, W, be):
    """Grow a batch of right environments past right-canonical B[n,u,s,v]."""
    t1 = be.einsum("nxry,nusx->nryus", R, be.conj(B))
    t2 = be.einsum("nryus,nwrsp->nyuwp", t1, W)
    return be.einsum("nyuwp,nvpy->nuwv", t2, B)


# --------------------------------------------------------------------------- Krylov
def lanczos_expm_fixed_batched(matvec, v, tau, be, m, expm_where="eigh", resid_log=None):
    """exp(tau H_n) v_n for k INDEPENDENT Hermitian problems at once.

    `v` is (k, n) and `matvec` maps (k, n) -> (k, n), applying each leg's own effective
    Hamiltonian to that leg's own vector. Returns (k, n).

    This is `eigensolver.lanczos_expm_fixed` with a leading leg axis, and every choice in
    it carries over unchanged: a FIXED schedule so there is no host branch (and, here, so
    the k legs share one shape); the basis and its conjugate held in two preallocated
    buffers so both operands of the projection are contiguous materialised tensors; two
    batched classical-Gram-Schmidt passes rather than a sequential loop (twice is what
    makes CGS as stable as MGS); the first projection coefficient read out as the Lanczos
    alpha instead of a separate `vdot` and axpy; and `guard_denom` in place of a breakdown
    test.

    What the leg axis changes. The projections become batched matrix-vector products --
    `(k, j, n) @ (k, n, 1)` -- so the k legs' re-orthogonalisations are ONE call rather
    than k, and the k projected tridiagonals are exponentiated in one batched
    eigendecomposition. The kernel count per solve is therefore identical to the unbatched
    solver's; only the work per kernel grows, which is the whole hypothesis.

    `resid_log`, when given a list, gets one (k,) DEVICE tensor appended: the Saad (1992)
    a-posteriori residual estimate for this solve, `|beta_{m+1}| * |tau| * |[exp(tau T_m)]
    [m-1, 0]| * beta0` -- the same estimator `eigensolver.lanczos_expm`'s adaptive stop uses,
    evaluated post-hoc at the fixed `m` instead of driving a stopping test. `beta_{m+1}` is
    the residual norm the LAST iteration already computes (to decide whether it would have
    extended the basis) and normally discards; here it is kept. Nothing is transferred to
    the host here -- the caller reduces and syncs once per trajectory step, not once per
    solve, which is what keeps the runtime guard cheap (see `_run`).
    """
    k, n = int(v.shape[0]), int(v.shape[1])
    m = max(1, min(int(m), n))
    Vb = be.zeros((k, m, n))
    Vc = be.zeros((k, m, n))
    beta0 = be.norm_axis(v, axis=-1)                  # (k,) real, DEVICE, never read back
    v0 = v / be.guard_denom(beta0)[:, None]
    Vb[:, 0] = v0
    Vc[:, 0] = be.conj(v0)
    alphas, betas = [], []
    beta_last = None
    for j in range(m):
        w = matvec(Vb[:, j])
        if j > 0:
            w = w - betas[j - 1][:, None] * Vb[:, j - 1]
        Vl, Vcl = Vb[:, : j + 1], Vc[:, : j + 1]
        c = (Vcl @ w[:, :, None])[:, :, 0]            # (k, j+1) -- one batched GEMV
        alphas.append(c[:, j].real)
        w = w - (c[:, None, :] @ Vl)[:, 0, :]
        c = (Vcl @ w[:, :, None])[:, :, 0]            # the second CGS pass
        w = w - (c[:, None, :] @ Vl)[:, 0, :]
        b = be.norm_axis(w, axis=-1)                  # (k,) real, DEVICE
        if j < m - 1:
            betas.append(b)
            vn = w / be.guard_denom(b)[:, None]
            Vb[:, j + 1] = vn
            Vc[:, j + 1] = be.conj(vn)
        else:
            beta_last = b     # the (m+1)-th beta: not used to extend the basis (there is no
                               # m+1-th slot), but exactly what the residual estimate needs.
    da = be.stack_axis(alphas, axis=-1)               # (k, m)
    eb = be.stack_axis(betas, axis=-1) if betas else None
    if expm_where == "host":
        # The correctness control: k separate host exponentials, exactly the route the
        # unbatched solver takes. It is k times the Python that the batched
        # eigendecomposition removes, which is why it is a control and not the default.
        from scipy.linalg import expm as _expm
        T = _tridiag_host(be, da, eb)
        E = be.array(np.stack([_expm(tau * T[i]) for i in range(k)]))
    else:
        E = be.expm_sym_tridiag_eigh_batched(da, eb, tau)
    if resid_log is not None:
        # Stays on-device: a (k,) complex multiply, no `.item()`, no `to_numpy`. The caller
        # collects one of these per Krylov solve in a step and syncs the whole list once.
        # |beta_{m+1}| * |tau| * |[exp(tau T_m)][m-1,0]| * beta0 -- Saad (1992), matching
        # `eigensolver.lanczos_expm`'s own adaptive-stop estimator exactly.
        resid_log.append((tau * beta_last * E[:, -1, 0]) * beta0)
    c = beta0[:, None] * E[:, :, 0]
    return (c[:, None, :] @ Vb)[:, 0, :]


def _tridiag_host(be, d, e):
    """The k projected tridiagonals as one host (k, m, m) real array."""
    dn = np.asarray(be.to_numpy(d)).real
    k, m = dn.shape
    T = np.zeros((k, m, m), dtype=np.float64)
    idx = np.arange(m)
    T[:, idx, idx] = dn
    if e is not None:
        en = np.asarray(be.to_numpy(e)).real
        j = np.arange(m - 1)
        T[:, j, j + 1] = en
        T[:, j + 1, j] = en
    return T


# --------------------------------------------------------------------------- geometry
def _bond_profile(state) -> Tuple[int, ...]:
    """(left bond of site 0, then every bond to the right of each site)."""
    return tuple([int(state[0].shape[0])] + [int(A.shape[2]) for A in state])


def _shape(x):
    return tuple(int(v) for v in x.shape)


def _check_geometry(states, mpos, L0s, RNs):
    """Refuse anything that is not k copies of one program on different data."""
    k = len(states)
    if k == 0:
        raise BatchGeometryError("evolve_batch needs at least one state; got an empty list.")
    N, d = len(states[0]), int(states[0][0].shape[1])
    p0 = _bond_profile(states[0])
    for n, s in enumerate(states):
        if len(s) != N:
            raise BatchGeometryError(
                f"leg {n} has {len(s)} sites, leg 0 has {N}. Batching requires identical site "
                f"counts -- these are different programs, not one program on different data.")
        for i, A in enumerate(s):
            if int(A.shape[1]) != d:
                raise BatchGeometryError(
                    f"leg {n} site {i} has physical dimension {int(A.shape[1])} against {d} "
                    f"in leg 0.")
            if i and int(A.shape[0]) != int(s[i - 1].shape[2]):
                raise BatchGeometryError(
                    f"leg {n} is not a well-formed MPS: the bond between sites {i-1} and {i} "
                    f"is {int(s[i-1].shape[2])} on one side and {int(A.shape[0])} on the other.")
        # The OUTER bonds are part of the geometry: a window on an infinite background carries
        # the vacuum bond dimension there, and two legs built at different chi_vac are not one
        # program even if every internal bond happened to agree.
        pn = _bond_profile(s)
        if (pn[0], pn[-1]) != (p0[0], p0[-1]):
            raise BatchGeometryError(
                f"leg {n} has outer bonds ({pn[0]}, {pn[-1]}) against ({p0[0]}, {p0[-1]}) for "
                f"leg 0. Internal bonds may differ (that is the ragged case); the outer ones "
                f"are fixed by the background and may not.")
    for n, W in enumerate(mpos):
        if len(W) != N:
            raise BatchGeometryError(f"MPO {n} has {len(W)} sites against {N} in the states.")
        for i in range(N):
            if _shape(W[i]) != _shape(mpos[0][i]):
                raise BatchGeometryError(
                    f"MPO {n} site {i} has shape {_shape(W[i])} against {_shape(mpos[0][i])} in "
                    f"MPO 0. Legs may differ in the VALUES of their couplings and fields -- that "
                    f"is what a scan varies -- but not in the MPO's bond structure.")
    for name, envs in (("L0", L0s), ("RN", RNs)):
        if envs[0] is None:
            continue
        for n, E in enumerate(envs):
            if E is None or _shape(E) != _shape(envs[0]):
                raise BatchGeometryError(
                    f"boundary environment {name} of leg {n} has shape "
                    f"{None if E is None else _shape(E)} against {_shape(envs[0])} for leg 0.")
    return N, d


# --------------------------------------------------------------------------- the batch
class _Batch:
    """k legs of one geometry, stacked on a leading axis, with their rank masks.

    `dims[i]` is the STACKED dimension of bond i (there are N+1 bonds); `rank[i]` is the
    per-leg TRUE dimension of that bond; `mask[i]` is None when every leg fills the bond
    and a real (k, dims[i]) indicator otherwise. The mask is real, not complex, so that one
    array serves both the complex tensors and the real Schmidt values -- both backends
    promote real * complex correctly.
    """

    __slots__ = ("be", "k", "N", "d", "M", "W", "L", "R", "rank", "mask", "dims", "fixed")

    def __init__(self, be, states, mpos, dims, ranks, fixed=False):
        self.be, self.k, self.N = be, len(states), len(states[0])
        self.d = int(states[0][0].shape[1])
        self.dims = list(dims)
        # True when `dims` is already the widest profile the cap allows, so no truncation
        # may change a stacked shape: that is what makes 'pad' a single shape class.
        self.fixed = bool(fixed)
        self.rank = [np.asarray(r, dtype=np.int64) for r in ranks]
        self.M = [_stack_padded(be, [s[i] for s in states], dims[i], self.d, dims[i + 1])
                  for i in range(self.N)]
        self.W = [_stack_exact(be, [W[i] for W in mpos]) for i in range(self.N)]
        self.L = [None] * (self.N + 1)
        self.R = [None] * (self.N + 1)
        self.mask = [self.mask_for(i) for i in range(self.N + 1)]

    def mask_for(self, i):
        r, B = self.rank[i], self.dims[i]
        if int(r.min()) >= B:
            return None
        m = (np.arange(B)[None, :] < r[:, None]).astype(np.float64)
        return self.be.array(m).real                 # real view of a backend-dtype array

    def fill(self):
        """Mean rank / stacked dimension over every bond and leg -- how much of the padded
        program is real work. 1.0 means the padding costs nothing at all."""
        num = float(np.mean([np.mean(self.rank[i] / float(self.dims[i]))
                             for i in range(self.N + 1)]))
        return num

    def states(self, host=True):
        """The k legs back as ordinary MPS, each trimmed to its OWN bond dimensions."""
        out = []
        for n in range(self.k):
            legs = []
            for i in range(self.N):
                a, b = int(self.rank[i][n]), int(self.rank[i + 1][n])
                T = self.M[i][n, :a, :, :b]
                legs.append(np.ascontiguousarray(self.be.to_numpy(T)) if host else T)
            out.append(MPS(legs))
        return out


def _stack_exact(be, tensors):
    """Stack k tensors of identical shape on a new leading axis, on the device."""
    xs = [be.array(t) for t in tensors]
    out = be.zeros((len(xs),) + _shape(xs[0]))
    for n, x in enumerate(xs):
        out[n] = x
    return out


def _stack_padded(be, tensors, Bl, d, Br):
    """Stack k rank-3 tensors into (k, Bl, d, Br), zero-padding each to the batch maximum.

    Done on the device, slice by slice: a host round trip of the whole batch would be tens
    of megabytes of transfer to accomplish an assignment.
    """
    out = be.zeros((len(tensors), Bl, d, Br))
    for n, A in enumerate(tensors):
        x = be.array(A)
        out[n, : int(x.shape[0]), :, : int(x.shape[2])] = x
    return out


def _pad_env(be, envs, B):
    """Stack k environments into (k, B, w, B), zero-padded on the device."""
    w = int(envs[0].shape[1])
    out = be.zeros((len(envs), B, w, B))
    for n, E in enumerate(envs):
        x = be.array(E)
        out[n, : int(x.shape[0]), :, : int(x.shape[2])] = x
    return out


def _apply_mask(x, mask, axis):
    """Zero every slot of `x` past its leg's true rank along `axis`. `mask` is (k, B)."""
    if mask is None:
        return x
    shape = [1] * x.ndim
    shape[0] = int(mask.shape[0])
    shape[axis] = int(mask.shape[1])
    return x * mask.reshape(tuple(shape))


# --------------------------------------------------------------------------- one-site steps
def _site_step_right(B, i, dt, m, expm_where, resid_log=None):
    """Batched forward site-step: evolve site i, move the centre right, back-evolve the bond."""
    be, k = B.be, B.k
    a, s, b = _shape(B.M[i])[1:]
    Li, Wi, Ri1 = B.L[i], B.W[i], B.R[i + 1]

    def mv1(x):
        return apply_one_site_b(x.reshape(k, a, s, b), Li, Wi, Ri1, be).reshape(k, -1)

    B.M[i] = lanczos_expm_fixed_batched(
        mv1, B.M[i].reshape(k, -1), -1j * dt / 2.0, be, m, expm_where,
        resid_log).reshape(k, a, s, b)
    if i == B.N - 1:
        return
    Q, C = be.qr(B.M[i].reshape(k, a * s, b))
    r = int(C.shape[1])
    # THE MASK. Q's columns past a leg's true rank are the QR's arbitrary orthonormal
    # completion of a zero column. Left alone they build a NON-zero left environment, and
    # the bond solve below then couples the true block to them -- silently promoting that
    # leg onto a larger manifold, which is a different trajectory from the one it takes
    # alone. Zeroing them (and the matching rows and columns of C, which the QR has already
    # set to zero) keeps every environment exactly block-diagonal in the true rank.
    mk = B.mask[i + 1]
    Q = _apply_mask(Q, mk, 2)
    C = _apply_mask(_apply_mask(C, mk, 1), mk, 2)
    B.M[i] = Q.reshape(k, a, s, r)
    B.L[i + 1] = update_left_env_b(B.L[i], B.M[i], Wi, be)
    Li1 = B.L[i + 1]

    def mv0(x):
        return apply_zero_site_b(x.reshape(k, r, b), Li1, Ri1, be).reshape(k, -1)

    C = lanczos_expm_fixed_batched(
        mv0, C.reshape(k, -1), +1j * dt / 2.0, be, m, expm_where, resid_log).reshape(k, r, b)
    C = _apply_mask(_apply_mask(C, mk, 1), mk, 2)
    B.M[i + 1] = be.einsum("nrb,nbsc->nrsc", C, B.M[i + 1])


def _site_step_left(B, i, dt, m, expm_where, resid_log=None):
    """Batched backward site-step: the mirror image of `_site_step_right`."""
    be, k = B.be, B.k
    a, s, b = _shape(B.M[i])[1:]
    Li, Wi, Ri1 = B.L[i], B.W[i], B.R[i + 1]

    def mv1(x):
        return apply_one_site_b(x.reshape(k, a, s, b), Li, Wi, Ri1, be).reshape(k, -1)

    B.M[i] = lanczos_expm_fixed_batched(
        mv1, B.M[i].reshape(k, -1), -1j * dt / 2.0, be, m, expm_where,
        resid_log).reshape(k, a, s, b)
    if i == 0:
        return
    Qt, Ct = be.qr(be.einsum("nasb->nsba", B.M[i]).reshape(k, s * b, a))   # LQ via QR of the transpose
    r = int(Ct.shape[1])
    mk = B.mask[i]
    Qt = _apply_mask(Qt, mk, 2)
    Ct = _apply_mask(_apply_mask(Ct, mk, 1), mk, 2)
    B.M[i] = be.einsum("nsbr->nrsb", Qt.reshape(k, s, b, r))
    B.R[i] = update_right_env_b(B.R[i + 1], B.M[i], Wi, be)
    Ri = B.R[i]
    C = be.einsum("nra->nar", Ct)

    def mv0(x):
        return apply_zero_site_b(x.reshape(k, a, r), Li, Ri, be).reshape(k, -1)

    C = lanczos_expm_fixed_batched(
        mv0, C.reshape(k, -1), +1j * dt / 2.0, be, m, expm_where, resid_log).reshape(k, a, r)
    C = _apply_mask(_apply_mask(C, mk, 1), mk, 2)
    B.M[i - 1] = be.einsum("ncsa,nar->ncsr", B.M[i - 1], C)


# --------------------------------------------------------------------------- two-site step
def _ranks_from_spectrum(Sn, nk, svd_tol, ceiling=None):
    """Per-leg retained rank and relative discarded weight from a stacked spectrum.

    `Sn` is the host (k, p) array of singular values and `nk` the widest rank the batch will
    hold. For each leg the rank is the smallest one whose TOTAL relative dropped weight
    stays under the budget, capped at `nk`.

    `ceiling` is the per-leg STRUCTURAL rank bound -- d times the leg's own true rank on
    either neighbouring bond -- and it is what keeps zero-padding exact independently of
    the budget. A padded block's singular values past its true rank are zero in exact
    arithmetic but come back from LAPACK at ~1e-17, so a discarded-weight rule with a zero
    budget would decline to drop them and would silently promote the leg onto a larger
    manifold. The ceiling is a statement about the block's structure rather than about a
    tolerance, so it drops exactly the padding and nothing else, at any budget.
    """
    k, p = Sn.shape
    tot = np.sum(Sn ** 2, axis=1)
    tot = np.where(tot > 0.0, tot, 1.0)
    tail = np.cumsum((Sn ** 2)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty(k, dtype=np.int64)
    disc = np.empty(k, dtype=np.float64)
    for n in range(k):
        lim = nk if ceiling is None else max(1, min(int(nk), int(ceiling[n])))
        drop = np.append(tail[n] / tot[n], 0.0)       # drop[j] = weight lost keeping j vectors
        ok = np.nonzero(drop[1: lim + 1] <= svd_tol)[0]
        ranks[n] = int(ok[0]) + 1 if len(ok) else lim
        disc[n] = float(drop[ranks[n]])
    return ranks, disc


def _two_site_step(B, i, dt, m, expm_where, cap, svd_tol, renorm, moving, tight=False,
                   resid_log=None):
    """Batched two-site block: evolve, truncate, back-evolve the new one-site centre.

    The truncation is where the ragged problem is MADE: the retained rank is a property of
    each leg's own spectrum, and with a discarded-weight budget the k legs disagree from
    the first step. Each leg's rank is recorded and masked, so a leg that wanted fewer
    singular values carries exact zeros in the rest and its arithmetic is unchanged; the
    only question left is how WIDE to make the stacked bond that holds them, which is the
    three ragged strategies:

      B.fixed  the bond already sits at the widest the cap allows and stays there, so the
               stacked shape never changes -- 'pad'.
      tight    cut the bond back to the widest rank any leg actually kept, so the padding
               costs only the spread between the legs -- 'tight'.
      neither  keep whatever the spectrum and the cap allow, letting the bond grow with
               the state -- 'group', which starts from the entry profile.
    """
    be, k = B.be, B.k
    W1, W2 = B.W[i], B.W[i + 1]
    Li, Ri2 = B.L[i], B.R[i + 2]
    theta = be.einsum("nasc,nctb->nastb", B.M[i], B.M[i + 1])
    a, s1, s2, b = _shape(theta)[1:]

    def mv2(x):
        return apply_two_site_b(x.reshape(k, a, s1, s2, b), Li, W1, W2, Ri2, be).reshape(k, -1)

    theta = lanczos_expm_fixed_batched(
        mv2, theta.reshape(k, -1), -1j * dt / 2.0, be, m, expm_where,
        resid_log).reshape(k, a, s1, s2, b)
    mat, rec = fuse(theta, ((0,), (1, 2), (3, 4)), be)       # (k, a s1, s2 b)
    U, S, Vh = be.svd_batched(mat)
    Sn = np.asarray(be.to_numpy(S)).real              # ONE transfer for k legs
    pool = min(int(Sn.shape[1]), int(cap))
    if B.fixed:
        pool = min(pool, int(B.dims[i + 1]))
    ceiling = np.minimum(B.rank[i] * B.d, B.rank[i + 2] * B.d)
    ranks, disc = _ranks_from_spectrum(Sn, pool, svd_tol, ceiling)
    nk = max(1, int(ranks.max())) if tight else pool
    U, S, Vh = U[:, :, :nk], S[:, :nk], Vh[:, :nk, :]
    B.rank[i + 1] = ranks
    B.dims[i + 1] = nk
    mk = B.mask[i + 1] = B.mask_for(i + 1)
    S = _apply_mask(S, mk, 1)
    if renorm:
        S = S / be.norm_axis(S, axis=-1)[:, None]
    U = _apply_mask(U, mk, 2)
    Vh = _apply_mask(Vh, mk, 1)
    Scol = S.reshape(k, nk, 1)
    if moving == "right":
        B.M[i] = split(U, rec, be, legs=(1,))
        B.L[i + 1] = update_left_env_b(B.L[i], B.M[i], W1, be)
        C = split(Scol * Vh, rec, be, legs=(2,))
        if i < B.N - 2:
            Li1, Ri2b = B.L[i + 1], B.R[i + 2]

            def mv1r(x):
                return apply_one_site_b(x.reshape(k, nk, s2, b), Li1, W2, Ri2b, be).reshape(k, -1)

            C = lanczos_expm_fixed_batched(
                mv1r, C.reshape(k, -1), +1j * dt / 2.0, be, m, expm_where,
                resid_log).reshape(k, nk, s2, b)
            C = _apply_mask(C, mk, 1)
        B.M[i + 1] = C
    else:
        B.M[i + 1] = split(Vh, rec, be, legs=(2,))
        B.R[i + 1] = update_right_env_b(B.R[i + 2], B.M[i + 1], W2, be)
        C = split(U * S.reshape(k, 1, nk), rec, be, legs=(1,))
        if i > 0:
            Lib, Ri1 = B.L[i], B.R[i + 1]

            def mv1l(x):
                return apply_one_site_b(x.reshape(k, a, s1, nk), Lib, W1, Ri1, be).reshape(k, -1)

            C = lanczos_expm_fixed_batched(
                mv1l, C.reshape(k, -1), +1j * dt / 2.0, be, m, expm_where,
                resid_log).reshape(k, a, s1, nk)
            C = _apply_mask(C, mk, 3)
        B.M[i] = C
    return disc


# --------------------------------------------------------------------------- calibration
def _calibrate_krylov_m(states, mpos, L0s, RNs, dt, backend, evolver, chi_max=None):
    """Freeze the batched Krylov schedule from one metered adaptive sweep on EVERY leg.

    Same rule as the unbatched fast path: take the LARGEST dimension the adaptive solver
    needed anywhere in the sweep -- max, not mean, because the schedule has to satisfy the
    hardest block and a mean over ~4N blocks is dominated by the easy bulk. Run over ALL
    legs, not leg 0 alone: legs are defined by this module to differ in exactly the
    couplings, fields and momenta that change a leg's own Krylov demand, so leg 0's measured
    demand does not structurally upper-bound leg 5's. This is a PREROLL cost -- k adaptive
    single-step sweeps run once, before the batched loop starts -- not a per-step cost, so
    it stays cheap relative to the n_steps-long batched trajectory that follows.

    Adds the project's standing ADDITIVE headroom of two to the max over legs. Each sweep
    runs its leg through the ordinary integrator with a step-size controller pinned to no
    change at all, whose only job here is to fill the Krylov meter; it does not mutate the
    caller's states.

    Returns `(m, per_leg)`: the frozen dimension `m`, and the raw per-leg adaptive demand
    (before headroom) as an array, for the caller's `stats`.
    """
    from .tdvp import TDVPOptions, _tdvp1, _tdvp2
    per_leg = np.zeros(len(states), dtype=np.int64)
    for idx, (state, mpo, L0, RN) in enumerate(zip(states, mpos, L0s, RNs)):
        log: List[Dict[str, Any]] = []
        o = TDVPOptions(backend=backend, canonicalize=False, keep_device=True, L0=L0, RN=RN,
                        graph="off", adaptive_dt={"fmin": 1.0, "fmax": 1.0, "log": log})
        if evolver == "2tdvp":
            _tdvp2(list(state), mpo, int(chi_max), dt, 1, o)
        else:
            _tdvp1(list(state), mpo, dt, 1, o)
        per_leg[idx] = max([int(r.get("m_max", 0)) for r in log] + [0])
    m = max(KRYLOV_FLOOR, int(per_leg.max()) + KRYLOV_HEADROOM) if len(per_leg) else KRYLOV_FLOOR
    return m, per_leg


# --------------------------------------------------------------------------- entry point
def evolve_batch(states: Sequence[MPSLike], hams: Any, dt: float, n_steps: int,
                 options: Optional[Union[BatchTDVPOptions, Dict[str, Any]]] = None,
                 **legacy_kwargs: Any) -> Tuple[List[MPS], List[Any]]:
    """Evolve k independent MPS of identical geometry as ONE batched tensor program.

    Parameters
    ----------
    states : k MPS (or plain lists of rank-3 tensors) sharing a site count, a physical
        dimension and their outer bond dimensions. Internal bonds may differ -- that is the
        ragged case, and `ragged` says how it is handled.
    hams : one MPO shared by every leg, or a list of k MPOs. Legs may differ in the VALUES
        of their couplings and fields (which is what a scan varies); they may not
        differ in the MPO's bond structure.
    dt, n_steps : the time grid, shared by every leg. Different step counts per leg is not
        one program and is refused.
    options : a `BatchTDVPOptions`, or the same fields flat as keywords.

    Returns
    -------
    (states, records) : the k evolved states, in the order they were given and each trimmed
        back to its OWN bond dimensions, and the measurement records. With one batch group
        (always, under `ragged='pad'`) the records are `[(t, measure(states_at_t)), ...]`;
        with several groups they are one such list per group, since the groups are stepped
        independently.

    Raises
    ------
    BatchGeometryError : whenever the legs are not one program on different data. This is
        deliberately loud: a silent per-leg fallback would turn a configuration mistake into
        an unexplained slowdown.
    BatchKrylovError : whenever the runtime Krylov residual guard finds a leg under-resolved
        at the fixed schedule AND `krylov_m` is already at the configured `krylov_m_max`
        ceiling, so it cannot be grown to fix it. Also deliberately loud, for the same
        reason: the alternative is a normalised, plausible-looking, silently wrong leg.

    Examples
    --------
    >>> out, rec = evolve_batch(legs, mpos, 0.05, 40,              # doctest: +SKIP
    ...                         backend="torch-cuda", evolver="1tdvp",
    ...                         krylov_m=10, L0=FL, RN=FR, canonicalize=False)
    """
    o = _resolve(options, legacy_kwargs)
    if o.evolver not in EVOLVERS:
        raise ValueError(f"evolver must be one of {EVOLVERS}, got {o.evolver!r}")
    if o.ragged not in RAGGED_MODES:
        raise ValueError(f"ragged must be one of {RAGGED_MODES}, got {o.ragged!r}")
    if o.expm_where not in EXPM_WHERE:
        raise ValueError(f"expm_where must be one of {EXPM_WHERE}, got {o.expm_where!r}")
    if o.evolver == "2tdvp" and o.chi_max is None:
        raise ValueError("evolver='2tdvp' needs chi_max: the two-site update grows bonds and "
                         "the cap is what stops it.")
    if o.evolver == "1tdvp" and (o.chi_max is not None or o.svd_tol):
        raise ValueError("evolver='1tdvp' does not accept chi_max or svd_tol: the one-site "
                         "update is strictly bond-preserving and truncates nothing, so a "
                         "budget would have no effect.")

    be = get_backend(o.backend)
    k = len(states)
    mpos = list(hams) if _is_mpo_list(hams) else [hams] * k
    if len(mpos) != k:
        raise BatchGeometryError(f"got {len(mpos)} MPOs for {k} states.")
    L0s = list(o.L0) if isinstance(o.L0, (list, tuple)) and len(o.L0) == k else [o.L0] * k
    RNs = list(o.RN) if isinstance(o.RN, (list, tuple)) and len(o.RN) == k else [o.RN] * k
    _check_geometry(states, mpos, L0s, RNs)
    invariants.entry(states[0], mpos[0], L0s[0], RNs[0])     # legs agree, so leg 0 suffices

    if o.canonicalize:
        from ._dmrg import right_canonicalize
        states = [right_canonicalize([be.array(x) for x in s], be) for s in states]

    stats = o.stats if o.stats is not None else {}
    m = o.krylov_m
    if m == "auto":
        m, calib_per_leg = _calibrate_krylov_m(states, mpos, L0s, RNs, dt, o.backend,
                                               o.evolver, o.chi_max)
        stats["krylov_calibration_per_leg"] = calib_per_leg.tolist()
    m = int(m)
    if m < KRYLOV_FLOOR:
        # NOT clamped, refused. A Krylov space of one or two vectors still returns a
        # normalised state and a plausible-looking trajectory -- it simply is not the
        # exponential of anything, and downstream the two-site truncation eats the bond
        # dimension rather than raising. A caller that passes a sentinel it meant to have
        # resolved elsewhere gets an error here instead of a quiet, finished, wrong run.
        raise ValueError(
            f"krylov_m must be at least {KRYLOV_FLOOR} (got {m}). A Krylov dimension below "
            f"that does not approximate exp(tau H) at all, and nothing downstream will say so.")
    stats["krylov_m"] = m
    m_max = o.krylov_m_max
    m_max = m * 2 if m_max is None else int(m_max)
    m_max = max(m_max, m)
    stats["krylov_m_max"] = m_max
    stats["krylov_tol"] = float(o.krylov_tol)

    groups = _partition(states, o.ragged)
    stats["groups"] = [len(g) for g in groups]
    stats["n_groups"] = len(groups)
    # The size-weighted mean group size is what the throughput actually scales with: a leg
    # in a group of one is a serial leg however many groups there are.
    stats["mean_batch"] = sum(len(g) ** 2 for g in groups) / float(k)
    stats["serial_legs"] = sum(1 for g in groups if len(g) == 1)

    out_states: List[Optional[MPS]] = [None] * k
    centre = 0 if (o.canonicalize or n_steps > 0) else None   # every sweep ends at site 0
    all_records: List[Any] = []
    fills, discs = [], []
    disc_leg_all = np.zeros(k, dtype=np.float64)
    leg_resid = np.zeros(k, dtype=np.float64)
    m_grown_to = m
    for g in groups:
        B = _build(be, [states[n] for n in g], [mpos[n] for n in g],
                   [L0s[n] for n in g], [RNs[n] for n in g],
                   cap=o.chi_max if (o.ragged == "pad" and o.evolver == "2tdvp") else None)
        recs, fill, disc, group_resid, m_used = _run(B, dt, n_steps, m, o, g, m_max)
        all_records.append(recs)
        fills.append(fill)
        discs.append(float(np.mean(disc)))
        m_grown_to = max(m_grown_to, m_used)
        for slot, n in enumerate(g):
            leg_resid[n] = group_resid[slot]
            disc_leg_all[n] = float(disc[slot])
        fin = B.states(host=not o.keep_device)
        for slot, n in enumerate(g):
            fin[slot].center = centre
            out_states[n] = fin[slot]
    stats["fill"] = float(np.mean(fills)) if fills else 1.0
    stats["disc"] = float(np.mean(discs)) if discs else 0.0
    # The per-leg discarded-weight ledger, in the order the caller gave the states. `disc`
    # above is its mean and is kept because callers already read it. A health clause is written
    # per arm -- one leg's drift against the weight THAT leg threw away -- so a batch reporting
    # only the mean would make a batched arm unscoreable against a serial one.
    stats["disc_per_leg"] = [float(x) for x in disc_leg_all]
    # The a-posteriori residual estimate the runtime guard checked every step, maxed over
    # steps, ONE NUMBER PER LEG -- so a caller can see how close the fixed schedule ran to
    # `krylov_tol` even on a run that never tripped the guard.
    stats["krylov_residual_max"] = leg_resid.tolist()
    stats["krylov_m_grown_to"] = m_grown_to
    stats["krylov_grown"] = bool(m_grown_to > m)
    return out_states, (all_records[0] if len(groups) == 1 else all_records)


def _is_mpo_list(hams):
    """True when `hams` is a LIST of MPOs rather than one MPO (a list of site tensors).

    An MPO is a sequence of rank-4 arrays; a list of MPOs is a sequence of those. The
    discriminator is therefore whether the first element has a `shape` -- an array does, a
    nested MPO does not.
    """
    return isinstance(hams, (list, tuple)) and not hasattr(hams[0], "shape")


def _resolve(options, legacy):
    if isinstance(options, BatchTDVPOptions):
        o = dataclasses.replace(options)
    elif isinstance(options, dict):
        o = BatchTDVPOptions(**options)
    else:
        o = BatchTDVPOptions()
    known = {f.name for f in dataclasses.fields(BatchTDVPOptions)}
    bad = sorted(set(legacy) - known)
    if bad:
        raise TypeError(f"evolve_batch() got unexpected keyword(s) {bad}; it accepts "
                        f"{sorted(known)}.")
    for key, val in legacy.items():
        setattr(o, key, val)
    return o


def _partition(states, ragged):
    """Group leg indices that can share one shape. Only 'group' actually partitions; the two
    padding strategies put every leg in one batch and let the rank masks do the work."""
    if ragged != "group":
        return [list(range(len(states)))]
    seen: Dict[Tuple[int, ...], List[int]] = {}
    for n, s in enumerate(states):
        seen.setdefault(_bond_profile(s), []).append(n)
    return list(seen.values())


def _widest_profile(N, d, outer_l, outer_r, cap):
    """The widest each of the N+1 bonds can EVER be under a two-site sweep capped at `cap`.

    A bond `i` sites from the left boundary cannot exceed `outer_l * d**i`, nor
    `outer_r * d**(N-i)` from the right, nor the cap. Padding to this profile is what makes
    the stacked shapes constant for a whole run: no truncation can ask for more. The
    exponent is clamped before it is taken, since d**501 is a number and not a bond.
    """
    lim = int(cap)
    out = []
    for i in range(N + 1):
        e_l, e_r = min(i, 32), min(N - i, 32)
        out.append(int(min(lim, outer_l * d ** e_l, outer_r * d ** e_r)))
    return out


def _build(be, states, mpos, L0s, RNs, cap=None):
    """Stack the legs. `cap` set means pad every bond to the widest the cap allows, so the
    stacked shapes are fixed for the run; `cap` None means start from the entry profile."""
    N = len(states[0])
    profiles = [_bond_profile(s) for s in states]
    dims = [max(p[i] for p in profiles) for i in range(N + 1)]
    fixed = cap is not None
    if fixed:
        d = int(states[0][0].shape[1])
        wide = _widest_profile(N, d, profiles[0][0], profiles[0][-1], cap)
        dims = [max(dims[i], wide[i]) for i in range(N + 1)]
    ranks = [np.array([p[i] for p in profiles], dtype=np.int64) for i in range(N + 1)]
    B = _Batch(be, states, mpos, dims, ranks, fixed=fixed)
    k = B.k
    L0 = [be.ones((1, 1, 1))] * k if L0s[0] is None else [be.array(x) for x in L0s]
    RN = [be.ones((1, 1, 1))] * k if RNs[0] is None else [be.array(x) for x in RNs]
    B.L[0] = _pad_env(be, L0, dims[0])
    B.R[N] = _pad_env(be, RN, dims[N])
    for i in range(N - 1, 0, -1):
        B.R[i] = update_right_env_b(B.R[i + 1], B.M[i], B.W[i], be)
    return B


def _run(B, dt, n_steps, m, o, leg_ids, m_max):
    """Run one batch group for `n_steps`, with the per-leg runtime Krylov residual guard.

    `m` is the starting Krylov dimension and MAY GROW as the run proceeds: every trajectory
    step collects the Saad (1992) a-posteriori residual estimate from every Lanczos solve
    that step made (kept on-device by `lanczos_expm_fixed_batched` -- see its docstring),
    reduces it to one number per leg with a SINGLE host transfer per step (not one per
    solve), and compares the worst leg against `o.krylov_tol`. A leg over tolerance grows
    `m` for every subsequent step, bounded by `m_max`; a leg still over tolerance with `m`
    already at `m_max` raises `BatchKrylovError` naming it -- the step that tripped the
    guard is not redone (growing helps every later step; it cannot retroactively fix the one
    just taken), which is the same trade `evolve_batch`'s global `krylov_m < KRYLOV_FLOOR`
    check makes: loud and forward-looking, not a silent retry.

    `leg_ids` are the caller's ORIGINAL leg indices for this group (only used for error
    messages and to key `stats["krylov_residual_max"]` -- `_Batch` itself is indexed 0..k-1
    within the group).

    Returns `(records, fill, disc_leg, leg_max_resid, m)`: `disc_leg` is the cumulative
    discarded weight PER LEG -- a health clause is written per arm, comparing one leg's drift
    against the weight THAT leg threw away, so a batch reporting only the mean would make a
    batched arm unscoreable against a serial one -- `leg_max_resid` is the per-leg
    (group-local order) max residual observed over the whole run, and the returned `m` is
    whatever it grew to by the end.
    """
    records = []
    k = B.k
    disc_leg = np.zeros(k, dtype=np.float64)
    tol = float(o.krylov_tol)
    leg_max_resid = np.zeros(k, dtype=np.float64)
    if o.measure is not None:
        records.append((0.0, o.measure(B.states(host=not o.keep_device))))
    for step in range(n_steps):
        resid_log: List[Any] = []
        if o.evolver == "1tdvp":
            for i in range(B.N):
                _site_step_right(B, i, dt, m, o.expm_where, resid_log)
            for i in range(B.N - 1, -1, -1):
                _site_step_left(B, i, dt, m, o.expm_where, resid_log)
        else:
            tight = o.ragged == "tight"
            for i in range(0, B.N - 1):
                disc_leg += _two_site_step(B, i, dt, m, o.expm_where, o.chi_max,
                                           o.svd_tol, o.renorm, "right", tight, resid_log)
            for i in range(B.N - 2, -1, -1):
                disc_leg += _two_site_step(B, i, dt, m, o.expm_where, o.chi_max,
                                           o.svd_tol, o.renorm, "left", tight, resid_log)
        invariants.sweep([A[0] for A in B.M], [W[0] for W in B.W], B.L[0][0], B.R[B.N][0])
        if resid_log:
            # The ONE sync this step pays, however many Lanczos solves fed it.
            stacked = B.be.stack_axis(resid_log, axis=0)
            step_max = np.max(np.abs(np.asarray(B.be.to_numpy(stacked))), axis=0)   # (k,)
            leg_max_resid = np.maximum(leg_max_resid, step_max)
            worst = int(np.argmax(step_max))
            if step_max[worst] > tol:
                if m < m_max:
                    m = min(m_max, m + KRYLOV_GROWTH_STEP)
                else:
                    raise BatchKrylovError(
                        f"leg {leg_ids[worst]}'s Lanczos residual estimate "
                        f"{step_max[worst]:.3e} exceeds krylov_tol={tol:.3e} at trajectory "
                        f"step {step} with krylov_m={m} already at the configured ceiling "
                        f"krylov_m_max={m_max}. Raise krylov_m_max, pass a larger explicit "
                        f"krylov_m, or accept a looser krylov_tol.")
        if o.measure is not None and (step + 1) % o.meas_every == 0:
            records.append(((step + 1) * dt, o.measure(B.states(host=not o.keep_device))))
    return records, B.fill(), disc_leg, leg_max_resid, m


__all__ = ["evolve_batch", "BatchTDVPOptions", "BatchGeometryError", "BatchKrylovError",
           "lanczos_expm_fixed_batched", "apply_one_site_b", "apply_zero_site_b",
           "apply_two_site_b", "update_left_env_b", "update_right_env_b"]
