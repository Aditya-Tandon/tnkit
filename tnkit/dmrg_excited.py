"""Excited-state (tower) DMRG by orthogonality penalty.

Two-site DMRG that targets the lowest eigenstate of

    H + w * sum_i |psi_i><psi_i|

for a list of previously converged states ``psi_i``.  Running it repeatedly, feeding
back each converged state, walks up the low-lying spectrum -- which is what a
finite-volume (Luscher) analysis needs and what ``tnkit.dmrg`` cannot do (it only ever
returns the ground state, and ``QuasiparticleSolver.bands_at`` is an infinite-volume
tangent-space object).

The penalty is applied *locally*, never as a full MPO: for each previous state the
overlap environments ``EL[i]`` (sites 0..i-1) and ``ER[i]`` (sites i..N-1) are carried
alongside the Hamiltonian environments, and the two-site block of ``|psi_i>`` in the
current left/right isometry basis is the rank-one vector

    V[a,s1,s2,b] = EL[i][a,x] P[i][x,s1,z] P[i+1][z,s2,y] ER[i+2][b,y] .

Because two-site DMRG keeps everything left of the block left-canonical and everything
right of it right-canonical, ``V`` is exactly the projection of ``|psi_i>`` into the
block, so ``w V V^dagger`` is exactly the block representation of ``w |psi_i><psi_i|``.
No MPO of bond dimension chi^2 is ever built.

Choosing ``w``: it must exceed the spread of the levels being climbed, or the penalised
copy of a low state can sit below the state one is looking for.  It must not be so large
that the local eigenproblem is badly conditioned.  ``w = 5 * (E_top - E_0)`` of the
window of interest is a sane default and the returned ``info["penalty_overlaps"]``
reports the residual overlaps so the choice can be checked rather than assumed.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional, Sequence

import numpy as np

from .backend import get_backend
from . import _options
from . import invariants
from . import linalg as tl
from .eigensolver import (block_lowk, check_reorth_periodic, lanczos_ground,
                          lanczos_ground_restart)
from .mps import MPS
from .observables import _backend_for
from ._dmrg import (
    _truncate,
    apply_one_site,
    apply_two_site,
    random_mps,
    right_canonicalize,
    update_left_env,
    update_right_env,
)

__all__ = ["mpo_expectation", "mpo_sq_expectation", "mpo_variance", "two_site_variance",
           "mps_inner", "dmrg_excited", "DMRGExcitedOptions", "dmrg_tower", "dmrg_block",
           "degenerate_clusters"]

_NP = get_backend("numpy")


def mps_inner(A, B, be=None):
    """<A|B> for two MPS in (left, phys, right) order, arbitrary gauge."""
    be = be or _NP
    E = be.ones((1, 1))
    for a, b in zip(A, B):
        E = be.einsum("ax,asb->xsb", E, be.conj(a))
        E = be.einsum("xsb,xsy->by", E, b)
    return be.to_scalar(E.reshape(-1)[0])


def mpo_expectation(mps, mpo, be=None):
    """<psi|W|psi> / <psi|psi> for an MPS and an MPO (bra/ket the same state).

    ``be`` defaults to the backend the MPS lives on."""
    be = be or _backend_for(mps)
    F = be.ones((1, 1, 1))
    for A, W in zip(mps, mpo):
        F = update_left_env(F, A, be.array(W), be)
    num = be.to_scalar(F.reshape(-1)[0])
    return num / mps_inner(mps, mps, be)


def mpo_sq_expectation(mps, mpo, be=None):
    """<psi|W W|psi> / <psi|psi>, contracted layer by layer without forming W^2.

    The environment carries (bra, upper MPO, lower MPO, ket) bonds, so memory is
    chi^2 D^2 and the cost is O(N (chi^3 D^2 d + chi^2 D^3 d^2)).
    ``be`` defaults to the backend the MPS lives on."""
    be = be or _backend_for(mps)
    F = be.ones((1, 1, 1, 1))
    for A, Wx in zip(mps, mpo):
        W = be.array(Wx)
        t = be.einsum("pvwq,psd->vwqsd", F, be.conj(A))
        t = be.einsum("vwqsd,vrst->wqdrt", t, W)
        t = be.einsum("wqdrt,wmty->qdrmy", t, W)
        F = be.einsum("qdrmy,qye->drme", t, A)
    num = be.to_scalar(F.reshape(-1)[0])
    return num / mps_inner(mps, mps, be)


def mpo_variance(mps, mpo, be=None):
    """Energy variance <W^2> - <W>^2, clipped at zero. Returns ``(var, floor)``.

    By the Weinstein bound some eigenvalue of W lies within sqrt(var) of <W>, so this
    certifies convergence without a reference spectrum. It cannot detect a state that
    converged onto an eigenvalue already found. ``floor`` = eps * max(|<W^2>|, 1) is the
    round-off level of the difference: a ``var`` at or below it is unresolved."""
    be = be or _backend_for(mps)
    e2 = float(np.real(mpo_sq_expectation(mps, mpo, be)))
    e1 = float(np.real(mpo_expectation(mps, mpo, be)))
    floor = float(np.finfo(np.float64).eps * max(abs(e2), 1.0))
    return max(e2 - e1 * e1, 0.0), floor


def two_site_variance(mps, mpo, be=None):
    """One- and two-site parts of the energy variance of W, ``(var1, var2)``.

    After Hubig, Haegeman and Schollwoeck, PRB 97, 045125 (2018), Sec. IV. For a normalised
    |psi>, the Hilbert space splits orthogonally into span(|psi>) and spaces of variations of
    n neighbouring sites, each orthogonal to the ones before; with P_n the projector onto the
    n-site space,

        var_n = <psi| W P_n W |psi>,      <W^2> - <W>^2 = var1 + var2 + var3 + ...

    In mixed gauge (left-canonical A, right-canonical B, one-site centre M_i, two-site centre
    Theta_i = M_i B_{i+1}), with F_i spanning the complement of A_i and G_i that of B_i,

        var1 = sum_{i=0}^{N-1} || F_i^dag (H_i M_i) ||^2
        var2 = sum_{i=0}^{N-2} || F_i^dag (H_{i,i+1} Theta_i) G_{i+1}^dag ||^2

    where H_i and H_{i,i+1} are the one- and two-site effective Hamiltonians. F and G are never
    formed; the code applies 1 - A A^dag and 1 - B^dag B instead.

    The terms are squared norms of parts of (W - <W>)|psi>, rates with the units of energy
    squared. var1 is the part inside the one-site tangent space: the squared velocity of
    one-site TDVP, which is not an error. var2 is the next part, outside that space and
    inside the two-site spaces. One-site TDVP evolves psi by projecting W|psi> onto the
    tangent space, so ||(1 - P_T) W|psi>||^2 = var2 + var3 + ... >= var2 is the squared norm
    of what it neglects (Paeckel et al., Ann. Phys. 411, 167998 (2019), Sec. 6.2.2, use the
    two-site variance of Hubig et al. as a diagnostic of this; Hubig et al. treat ground states
    only and do not discuss dynamics). So var2 is a lower bound on that rate, and equal to it when W
    couples only nearest neighbours, where var_n = 0 for n >= 3 (then two-site TDVP neglects
    nothing either). For a longer-range W (e.g. the Schwinger electric term) var_{n>=3} is
    nonzero in general: var2 can undershoot by an unknown amount, and it says nothing about
    two-site TDVP, whose leading neglected term is var3 (not computed here). Converting a rate
    into an accumulated state error needs a time integral of the full residual; neither paper
    does that.

    The tangent spaces are those of the bond dimensions the MPS carries: at a bond with fewer
    Schmidt values than its dimension they include directions the state does not define (the
    completions that the QR returns), which is also what one-site TDVP evolves in.

    Cost: one-site and two-site H_eff applications, one QR per site, and one right
    canonicalisation, O(N chi^3 D d^2): about 8x `mpo_expectation` (numpy, N = 60, chi = 32-128,
    D = 5). ``be`` defaults to the backend the MPS lives on."""
    be = be or _backend_for(mps)
    W = [be.array(w) for w in mpo]
    M = right_canonicalize([be.array(t) for t in mps], be)
    N = len(M)
    R = [None] * (N + 1)
    R[N] = be.ones((1, 1, 1))
    for i in range(N - 1, 0, -1):
        R[i] = update_right_env(R[i + 1], M[i], W[i], be)
    L = be.ones((1, 1, 1))
    var1 = var2 = 0.0
    for i in range(N):
        Dl, d, Dr = M[i].shape
        Q, C = be.qr(M[i].reshape(Dl * d, Dr))
        Qh = be.conj(Q).T

        def off_left(y):
            y = y.reshape(Dl * d, -1)
            return y - Q @ (Qh @ y)

        var1 += float(be.norm(off_left(apply_one_site(M[i], L, W[i], R[i + 1], be)))) ** 2
        if i < N - 1:
            B = M[i + 1].reshape(M[i + 1].shape[0], -1)
            y = apply_two_site(be.einsum("asb,btc->astc", M[i], M[i + 1]), L, W[i], W[i + 1],
                               R[i + 2], be)
            y = off_left(y).reshape(-1, B.shape[1])
            var2 += float(be.norm(y - (y @ be.conj(B).T) @ B)) ** 2
            L = update_left_env(L, Q.reshape(Dl, d, -1), W[i], be)
            M[i + 1] = be.einsum("ab,bsc->asc", C, M[i + 1])
    return var1, var2


def _overlap_envs(M, P, be):
    """Left/right overlap environments between the current MPS ``M`` and a fixed ``P``.

    ``EL[i][a,x]`` contracts sites 0..i-1 (conjugated on the M side);
    ``ER[i][a,x]`` contracts sites i..N-1.
    """
    N = len(M)
    EL = [None] * (N + 1)
    ER = [None] * (N + 1)
    EL[0] = be.ones((1, 1))
    ER[N] = be.ones((1, 1))
    for i in range(N - 1, -1, -1):
        t = be.einsum("asb,by->asy", be.conj(M[i]), ER[i + 1])
        ER[i] = be.einsum("asy,xsy->ax", t, P[i])
    return EL, ER


def _grow_EL(EL_i, Mi, Pi, be):
    t = be.einsum("ax,asb->xsb", EL_i, be.conj(Mi))
    return be.einsum("xsb,xsy->by", t, Pi)


def _grow_ER(ER_i1, Mi, Pi, be):
    t = be.einsum("asb,by->asy", be.conj(Mi), ER_i1)
    return be.einsum("asy,xsy->ax", t, Pi)


def _block_vector(EL_i, Pi, Pi1, ER_i2, be):
    """Two-site block representation V[a,s1,s2,b] of the state P."""
    t = be.einsum("ax,xsz->asz", EL_i, Pi)
    t = be.einsum("asz,zty->asty", t, Pi1)
    return be.einsum("asty,by->astb", t, ER_i2)


_PROJ_RTOL = 1e-5     # singular-value cutoff of the projector basis, relative to the largest
_PROJ_KEEP = 0.5      # least norm a projected solution must keep in the range of Q


def _nrm(x, be):
    return float(np.real(be.to_scalar(be.norm(x))))


def _projector(Vs, be):
    """``(Q, rank)``: ``Q = x -> (1 - B^dag B) x`` with ``B`` an orthonormal basis of span(Vs),
    from an SVD of the stacked vectors (not the Gram matrix, which squares their condition
    number). Directions with singular value at most ``_PROJ_RTOL`` times the largest are not
    deflated, so a state may keep an overlap of up to about that fraction of the largest
    block vector's norm with them."""
    if not Vs:
        return (lambda x: x), 0
    _, s, Vh = be.svd(be.stack(Vs))
    s = np.asarray(be.to_numpy(s))
    r = int(np.sum(s > _PROJ_RTOL * s[0])) if s.size and s[0] > 0 else 0
    B = Vh[:r]
    Bc = be.conj(B)
    return (lambda x: x - be.vm(be.mv(Bc, x), B)), r


def _lifted(h, proj, starts, margin, be):
    """``Q H Q + sigma (1 - Q)``, equal to ``Q H Q`` on the range of ``Q``.

    Plain ``Q H Q`` puts the deflated span at eigenvalue 0, and the eigensolver amplifies
    round-off into it whenever the wanted eigenvalues lie above 0. ``sigma`` is ``margin``
    above the largest Rayleigh quotient of ``starts`` (nonzero vectors in the range of ``Q``),
    so above the lowest eigenvalue on that range. A block solve wants more than the lowest;
    for those it relies on ``margin`` and on its caller's check of the solution."""
    def hq(x):
        return proj(h(proj(x)))

    sigma = margin + max(float(np.real(be.to_scalar(be.vdot(v, hq(v))))) / _nrm(v, be) ** 2
                         for v in starts)

    def matvec(x):
        px = proj(x)
        return proj(h(px)) + sigma * (x - px)
    return matvec


@dataclasses.dataclass
class DMRGExcitedOptions:
    """Settings of `dmrg_excited`, with the defaults of its old flat signature."""
    prev_states: Sequence[Any] = ()
    penalty: float = 1.0
    n_sweeps: int = 12
    tol: float = 1e-11
    eig_tol: float = 1e-9
    trunc: str = "svd"
    seed: int = 0
    init_mps: Optional[Any] = None
    backend: Any = "auto"
    verbose: bool = False
    eig_max_iter: int = 100
    reorth: str = "full"
    eig: str = "lanczos-restart"
    restart_m: int = 40
    restart_k: int = 10
    reorth_alg: str = "cgs2"
    deflate: str = "penalty"


def _dense_only(mpo, name):
    if getattr(next(iter(mpo), None), "is_block", False):
        raise ValueError(f"{name} takes a dense MPO; block tensors run in dmrg only")


def dmrg_excited(mpo, chi, options=None, **legacy_kwargs):
    """Lowest eigenstate of ``H`` deflated against ``prev_states``.  Returns (E, M, info),
    with ``M`` an `MPS` whose `center` is 0.

    Settings go in a `DMRGExcitedOptions` or as flat keywords, as in `dmrg`; a flat keyword
    overrides the field of an `options` object given alongside it.

    ``E`` is the *physical* energy <M|H|M>, with the penalty contribution removed, so it
    is directly comparable across states.

    deflate : {"penalty", "project"}
        "penalty" (default) solves ``H + penalty * sum |psi_i><psi_i|``. "project" solves
        ``Q H Q`` in each two-site block, ``Q`` the orthogonal projector off the block
        representations of ``prev_states``: deflation is exact within the block (no
        leakage of order 1/penalty). ``penalty`` then lifts the deflated span above the
        start's energy, and is the penalty of a fallback: a block with no room left outside
        the deflated span, or whose solution keeps less than half its norm there, is solved
        by the penalty method instead, counted in ``info["project_fallbacks"]``.

    eig : {"lanczos", "lanczos-restart"}
        Local eigensolver for each two-site block, mirroring ``tnkit.dmrg``'s ``eig=``.
        Default: "lanczos-restart" -- thick restart at identical
        accuracy and 25-31% faster wall time. "lanczos" is the historical, unbounded-basis
        solver; pass it to get the old behaviour exactly (bit-for-bit -- this function had
        no solver choice before this flip, so "lanczos" reproduces every prior call site).
        ``reorth`` applies to "lanczos" only; thick restart is always fully reorthogonalised
        within its ``restart_m``-vector window.
    restart_m, restart_k : int
        Krylov window and Ritz vectors kept per restart, used only by eig="lanczos-restart".
        Names and defaults mirror ``tnkit.dmrg.DMRGOptions``.
    reorth_alg : {"cgs2", "cgs2v2", "mgs"}
        Reorthogonalisation kernel inside the restart window; ignored by "lanczos".
    backend : as in ``tnkit.dmrg`` -- 'numpy' | 'torch-*' | a '-real' variant | a Backend
        object | 'auto'. Default 'auto': real dtype when `mpo` (and
        `init_mps`, if given) are real, complex otherwise. Pass backend='numpy' for the
        old complex-always default.
    """
    _dense_only(mpo, "dmrg_excited")
    o = _options.resolve(DMRGExcitedOptions, options, legacy_kwargs, "dmrg_excited")
    tl.warn_if_deprecated(o.trunc)
    return _dmrg_excited(mpo, chi, **{f.name: getattr(o, f.name)
                                      for f in dataclasses.fields(o)})


def _dmrg_excited(mpo, chi, *, prev_states, penalty, n_sweeps, tol, eig_tol, trunc, seed,
                  init_mps, backend, verbose, eig_max_iter, reorth, eig, restart_m,
                  restart_k, reorth_alg, deflate):
    """The solver; `dmrg_excited` documents it."""
    if eig not in ("lanczos", "lanczos-restart"):
        raise ValueError(f"eig must be 'lanczos' or 'lanczos-restart', got {eig!r}")
    if deflate not in ("penalty", "project"):
        raise ValueError(f"deflate must be 'penalty' or 'project', got {deflate!r}")
    # reorth='periodic' is refuted (see `check_reorth_periodic` in eigensolver.py). eig="lanczos"
    # already gets this check for free every block, inside `lanczos_ground` itself (stop=
    # "residual", its default -- this function has no eig_stop kwarg); "lanczos-restart" never
    # even looks at `reorth`, so an explicit reorth='periodic' from the caller must still be
    # refused up front here, once, for parity with the "lanczos" path.
    if eig == "lanczos-restart":
        check_reorth_periodic(reorth, "residual")
    prev_states = list(prev_states)
    invariants.entry(init_mps, mpo)
    for P in prev_states:
        invariants.entry(P, mpo)
    be = get_backend(backend, mpo=mpo, init_mps=init_mps)
    N = len(mpo)
    d = mpo[0].shape[2]
    W = [be.array(x) for x in mpo]
    prev = [[be.array(x) for x in P] for P in prev_states]

    if init_mps is not None:
        M = right_canonicalize([be.array(m) for m in init_mps], be)
    else:
        M = right_canonicalize(random_mps(N, chi, d, np.random.default_rng(seed), be), be)

    L = [None] * (N + 1)
    R = [None] * (N + 1)
    L[0] = be.ones((1, 1, 1))
    R[N] = be.ones((1, 1, 1))
    for i in range(N - 1, 0, -1):
        R[i] = update_right_env(R[i + 1], M[i], W[i], be)

    ELs, ERs = [], []
    for P in prev:
        el, er = _overlap_envs(M, P, be)
        ELs.append(el)
        ERs.append(er)

    rng = np.random.default_rng(seed + 1)
    energies = []
    total_disc = 0.0
    t_eig = 0.0
    fallbacks = 0

    def solve_block(i, direction):
        nonlocal total_disc, t_eig, fallbacks
        W1, W2 = W[i], W[i + 1]
        Li, Ri = L[i], R[i + 2]
        a, s1 = M[i].shape[0], M[i].shape[1]
        s2, b = M[i + 1].shape[1], M[i + 1].shape[2]
        theta0 = be.einsum("asc,ctb->astb", M[i], M[i + 1])
        shape = tuple(int(x) for x in theta0.shape)
        dim = int(np.prod(shape))

        Vs = [
            _block_vector(ELs[j][i], P[i], P[i + 1], ERs[j][i + 2], be).reshape(-1)
            for j, P in enumerate(prev)
        ]

        def h(x):
            return apply_two_site(x.reshape(shape), Li, W1, W2, Ri, be).reshape(-1)

        def penalised(x):
            y = h(x)
            for v in Vs:
                c = be.to_scalar(be.vdot(v, x))
                if not be.is_complex:
                    # to_scalar returns a Python complex; multiplying it in would promote
                    # y to complex128 on the real backend and force a silent cast back to
                    # real inside the eigensolver at every Lanczos step. v and x are real
                    # here, so the imaginary part is exactly zero.
                    c = c.real
                y = y + (penalty * c) * v
            return y

        # The iteration cap is identical across solvers -- eig= changes how the basis is
        # held and reorthogonalised, not how much work is allowed. Two changes to the local
        # eigenproblem at once would make an eig="lanczos" vs "lanczos-restart" comparison
        # unattributable.
        def eigsolve(matvec, v):
            if eig == "lanczos-restart":
                e, x, _ = lanczos_ground_restart(
                    matvec, v, be, tol=eig_tol, max_iter=min(dim, eig_max_iter),
                    m=restart_m, k=restart_k, reorth_alg=reorth_alg)
            else:
                e, x, _ = lanczos_ground(
                    matvec, v, be, tol=eig_tol,
                    max_iter=min(dim, eig_max_iter), reorth=reorth)
            return float(np.real(e)), x

        v0 = theta0.reshape(-1)
        t0 = time.perf_counter()
        x = None
        if deflate == "project":
            proj, rank = _projector(Vs, be)
            if rank < dim:
                v = proj(v0)
                if _nrm(v, be) <= 1e-6 * max(_nrm(v0, be), 1e-300):
                    v = proj(be.array(rng.standard_normal(dim)))   # start lay in the span
                e, x = eigsolve(_lifted(h, proj, [v], penalty, be), v)
                x = proj(x)                     # remove round-off drift into the span
                nx = _nrm(x, be)
                x = x / nx if nx > _PROJ_KEEP else None
            fallbacks += x is None
        if x is None:
            e, x = eigsolve(penalised, v0)
        t_eig += time.perf_counter() - t0

        U, S, Vh, disc = _truncate(x.reshape(a * s1, s2 * b), chi, trunc, be, rng,
                                   "left" if direction == "right" else "right")
        S = S / be.norm(S)
        total_disc += disc
        k = int(S.shape[0])
        if direction == "right":
            M[i] = U.reshape(a, s1, k)
            M[i + 1] = (S[:, None] * Vh).reshape(k, s2, b)
            L[i + 1] = update_left_env(L[i], M[i], W1, be)
            for j, P in enumerate(prev):
                ELs[j][i + 1] = _grow_EL(ELs[j][i], M[i], P[i], be)
        else:
            M[i] = (U * S).reshape(a, s1, k)
            M[i + 1] = Vh.reshape(k, s2, b)
            R[i + 1] = update_right_env(R[i + 2], M[i + 1], W2, be)
            for j, P in enumerate(prev):
                ERs[j][i + 1] = _grow_ER(ERs[j][i + 2], M[i + 1], P[i + 1], be)
        return e

    prev_E = None
    sweep = 0
    for sweep in range(n_sweeps):
        for i in range(0, N - 1):
            E = solve_block(i, "right")
        for i in range(N - 2, -1, -1):
            E = solve_block(i, "left")
        invariants.sweep(M, W)
        energies.append(E)
        if verbose:
            print(f"    sweep {sweep+1:2d}: E_pen = {E:.12f}", flush=True)
        if prev_E is not None and abs(E - prev_E) < tol:
            break
        prev_E = E

    M = [be.array(m) for m in M]
    ov = [abs(mps_inner(M, [be.array(x) for x in P], be)) for P in prev_states]
    E_phys = float(np.real(mpo_expectation(M, mpo, be)))
    info = {"sweeps": sweep + 1, "energies": energies, "E_penalised": E,
            "penalty_overlaps": ov, "discarded_weight": total_disc,
            "eig_time": t_eig, "chi": chi, "eig": eig, "backend": be.name, "deflate": deflate,
            "project_fallbacks": fallbacks,
            "restart_m": restart_m if eig == "lanczos-restart" else None,
            "restart_k": restart_k if eig == "lanczos-restart" else None}
    return E_phys, MPS(M, center=0), info        # the last sweep ends at site 0


def _shared_isometry(Thetas, chi, direction, be):
    """Isometry for a state-averaged two-site block (equal weights) and the discarded weight.

    The SVD of the stacked blocks yields the eigenvectors of the averaged reduced density
    matrix without forming it. "right" keeps U over (a s1); "left" keeps Vh over (s2 b)."""
    St = be.stack(Thetas) / np.sqrt(len(Thetas))           # (k, a*s1, s2*b)
    k, nl, nr = (int(x) for x in St.shape)
    if direction == "right":
        X = be.einsum("kij->ikj", St).reshape(nl, k * nr)
    else:
        X = St.reshape(k * nl, nr)
    U, S, Vh = be.svd(X)
    keep = min(chi, int(S.shape[0]))
    disc = float(np.sum(np.asarray(be.to_numpy(S))[keep:] ** 2))
    return (U[:, :keep] if direction == "right" else Vh[:keep]), disc


def dmrg_block(mpo, chi, nstates, prev_states=(), penalty=1.0, n_sweeps=12, tol=1e-11,
               eig_tol=1e-9, seed=0, init_mps=None, backend="auto", eig_max_iter=100,
               verbose=False, deflate="penalty"):
    """The ``nstates`` lowest eigenstates of ``H`` deflated against ``prev_states``, together.

    State-averaged two-site DMRG: the targets share one set of isometries, and each two-site
    block is solved for its ``nstates`` lowest eigenpairs at once (`block_lowk`). Inside a
    near-degenerate subspace this is well posed where `dmrg_excited`'s one-state-at-a-time
    solve is not. A shared basis is worse for each member than its own: it can need up to
    ``nstates`` times a single state's bond dimension to be exact, so at fixed ``chi`` the
    energies carry a truncation price (``info["discarded_weight"]``);
    `dmrg_tower(pair_solve=True)` polishes it away.

    ``deflate`` is as in `dmrg_excited`, fallbacks included.

    Returns ``(energies, states, info)`` with ``energies`` ascending (physical, penalty
    removed) and ``states`` backend-native.
    """
    _dense_only(mpo, "dmrg_block")
    if deflate not in ("penalty", "project"):
        raise ValueError(f"deflate must be 'penalty' or 'project', got {deflate!r}")
    prev_states = list(prev_states)
    invariants.entry(None, mpo)
    for P in prev_states:
        invariants.entry(P, mpo)
    be = get_backend(backend, mpo=mpo, init_mps=init_mps)
    N, d, k = len(mpo), mpo[0].shape[2], int(nstates)
    W = [be.array(x) for x in mpo]
    prev = [[be.array(x) for x in P] for P in prev_states]
    if init_mps is not None:
        M = right_canonicalize([be.array(m) for m in init_mps], be)
    else:
        M = right_canonicalize(random_mps(N, chi, d, np.random.default_rng(seed), be), be)

    L, R = [None] * (N + 1), [None] * (N + 1)
    L[0], R[N] = be.ones((1, 1, 1)), be.ones((1, 1, 1))
    for i in range(N - 1, 0, -1):
        R[i] = update_right_env(R[i + 1], M[i], W[i], be)
    ELs, ERs = [], []
    for P in prev:
        el, er = _overlap_envs(M, P, be)
        ELs.append(el)
        ERs.append(er)

    # C[j]: state j's own centre tensor, at site C_site; every other tensor is shared
    st = {"C": None, "C_site": None, "disc": 0.0, "unconverged": 0, "fallbacks": 0}

    def solve_block(i, direction):
        W1, W2, Li, Ri = W[i], W[i + 1], L[i], R[i + 2]
        a, s1 = M[i].shape[0], M[i].shape[1]
        s2, b = M[i + 1].shape[1], M[i + 1].shape[2]
        C, C_site = st["C"], st["C_site"]
        if C is not None and C_site == i:
            starts = [be.einsum("asc,ctb->astb", Cj, M[i + 1]) for Cj in C]
        elif C is not None and C_site == i + 1:
            starts = [be.einsum("asc,ctb->astb", M[i], Cj) for Cj in C]
        else:
            starts = [be.einsum("asc,ctb->astb", M[i], M[i + 1])]
        shape = (int(a), int(s1), int(s2), int(b))
        Vs = [_block_vector(ELs[j][i], P[i], P[i + 1], ERs[j][i + 2], be).reshape(-1)
              for j, P in enumerate(prev)]

        dim = int(np.prod(shape))

        def h(x):
            return apply_two_site(x.reshape(shape), Li, W1, W2, Ri, be).reshape(-1)

        def penalised(x):
            y = h(x)
            for v in Vs:
                c = be.to_scalar(be.vdot(v, x))
                y = y + (penalty * (c if be.is_complex else c.real)) * v
            return y

        V0 = [x.reshape(-1) for x in starts]
        solve = dict(nev=k, tol=eig_tol, max_iter=min(dim, eig_max_iter), seed=seed + 7 * i)
        res = None
        if deflate == "project":
            proj, rank = _projector(Vs, be)
            if rank + k <= dim:
                P0 = [proj(v) for v in V0]
                live = [v for v in P0 if _nrm(v, be) > 1e-6] or [
                    proj(be.array(np.random.default_rng(seed + 7 * i).standard_normal(dim)))]
                res = block_lowk(_lifted(h, proj, live, penalty, be), P0, be, prep=proj,
                                 **solve)
                if min(_nrm(proj(x), be) for x in res[1]) <= _PROJ_KEEP:
                    res = None
            st["fallbacks"] += res is None
        evs, xs, minfo = res or block_lowk(penalised, V0, be, **solve)
        st["unconverged"] += not minfo["converged"]
        Thetas = [x.reshape(a * s1, s2 * b) for x in xs]
        iso, disc = _shared_isometry(Thetas, chi, direction, be)
        st["disc"] += disc
        if direction == "right":
            m = int(iso.shape[1])
            M[i] = iso.reshape(a, s1, m)
            st["C"] = [be.einsum("ix,ij->xj", be.conj(iso), T).reshape(m, s2, b)
                       for T in Thetas]
            st["C_site"] = i + 1
            M[i + 1] = st["C"][0]
            L[i + 1] = update_left_env(L[i], M[i], W1, be)
            for j, P in enumerate(prev):
                ELs[j][i + 1] = _grow_EL(ELs[j][i], M[i], P[i], be)
        else:
            m = int(iso.shape[0])
            M[i + 1] = iso.reshape(m, s2, b)
            st["C"] = [be.einsum("ij,xj->ix", T, be.conj(iso)).reshape(a, s1, m)
                       for T in Thetas]
            st["C_site"] = i
            M[i] = st["C"][0]
            R[i + 1] = update_right_env(R[i + 2], M[i + 1], W2, be)
            for j, P in enumerate(prev):
                ERs[j][i + 1] = _grow_ER(ERs[j][i + 2], M[i + 1], P[i + 1], be)
        return evs

    history, prev_E, sweep = [], None, 0
    for sweep in range(n_sweeps):
        for i in range(0, N - 1):
            Ev = solve_block(i, "right")
        for i in range(N - 2, -1, -1):
            Ev = solve_block(i, "left")
        history.append(list(Ev))
        if verbose:
            print("    sweep %2d: E = %s" % (sweep + 1, ", ".join("%.10f" % e for e in Ev)),
                  flush=True)
        if prev_E is not None and max(abs(x - y) for x, y in zip(Ev, prev_E)) < tol:
            break
        prev_E = Ev

    # the last pass ended at site 0: sites 1..N-1 are shared and right-canonical
    states = []
    for Cj in st["C"]:
        Mj = [Cj] + list(M[1:])
        Mj[0] = Mj[0] / float(np.sqrt(abs(mps_inner(Mj, Mj, be))))
        states.append(Mj)
    E = [float(np.real(mpo_expectation(Mj, mpo, be))) for Mj in states]
    order = np.argsort(E)
    E, states = [E[t] for t in order], [states[t] for t in order]
    info = {"sweeps": sweep + 1, "energies": history,
            "penalty_overlaps": [max((abs(mps_inner(Mj, P, be)) for P in prev), default=0.0)
                                 for Mj in states],
            "block_overlaps": [max(abs(mps_inner(states[p], states[q], be))
                                   for q in range(k) if q != p) if k > 1 else 0.0
                               for p in range(k)],
            "discarded_weight": st["disc"], "eig_unconverged": st["unconverged"],
            "backend": be.name, "deflate": deflate, "project_fallbacks": st["fallbacks"]}
    return E, states, info


def degenerate_clusters(energies, tau, m1_ref=None, max_k=4):
    """Runs of consecutive levels whose spacings are all below ``tau * m1``.

    ``m1`` defaults to the tower's own first gap, so no reference spectrum is needed. A
    negative spacing (levels out of order, a sign the tower missed one) counts as close. A run
    longer than ``max_k`` is cut, recursively, at its widest spacing that leaves at least two
    levels on each side, so no level of a run is left to a one-state solve (a 3-level run
    stays whole even when ``max_k=2``). Returns ``(lo, hi)`` inclusive index ranges.
    """
    E = np.asarray(energies, dtype=float)
    if E.size < 2:
        return []
    m1 = float(m1_ref) if m1_ref else float(E[1] - E[0])
    if not np.isfinite(m1) or m1 <= 0:
        return []
    gaps = np.diff(E)
    close = gaps / m1 < tau
    runs, lo = [], None
    for i, c in enumerate(close):
        if c and lo is None:
            lo = i
        elif not c and lo is not None:
            runs.append((lo, i))
            lo = None
    if lo is not None:
        runs.append((lo, len(close)))

    def split(lo, hi):
        if hi - lo < max(max_k, 3):
            return [(lo, hi)]
        c = lo + 1 + int(np.argmax(gaps[lo + 1:hi - 1]))     # cut between levels c and c+1
        return split(lo, c) + split(c + 1, hi)

    return [(a, b) for lo, hi in runs for a, b in split(lo, hi)]


def _host(M):
    """Host numpy copies of an MPS, keeping each tensor's layout (np.asarray fails on GPU)."""
    be = _backend_for(M)
    return [be.to_numpy(x) for x in M]


def _tower_penalty(energies, penalty):
    """The tower's adaptive penalty: 3x the spread of ``energies`` plus ``penalty``."""
    if not energies:
        return penalty
    return max(penalty, 3.0 * (max(energies) - min(energies)) + penalty)


def dmrg_tower(mpo, chi, nlevels, penalty=1.0, seed=0, verbose=False,
               ov_tol=1e-5, max_retries=2, warm_chain=True,
               deflate="penalty", pair_solve=False, pair_tau=0.05, pair_max_k=4,
               pair_m1=None, **kw):
    """Climb the lowest ``nlevels`` eigenstates by repeated penalised DMRG.

    The penalty is **adaptive**.  A fixed ``w`` fails in a way that is easy to miss: once
    the tower spans more than ``w``, the penalised copy of a low state, sitting at
    ``E_0 + w``, drops below the level being searched for and DMRG returns the ground
    state again.  Observed exactly this failure mode with ``w = 4`` on a tower of
    spread 6.1.  So ``w`` is raised to ``3 x`` the spread found so far, and any level that
    still comes back with a large overlap on a previous state is recomputed with ``w``
    quadrupled and a different random seed.

    The local eigensolver and backend are `dmrg_excited`'s and reach here through ``**kw``:
    thick-restart (`eig="lanczos-restart"`) and auto real dtype (`backend="auto"`) are the
    defaults.

    ``deflate="project"`` searches each level in the exact orthogonal complement of the levels
    below (see `dmrg_excited`; the penalty is used only to lift the deflated span and by
    fallback blocks). It starts each level from a random state: ``warm_chain`` applies to
    the penalty method only, because the level below lies almost wholly in the deflated span
    and its small remainder can miss an exactly degenerate partner.

    ``pair_solve=True`` adds a repair pass for near-degenerate levels, which a one-state-at-a-
    time solve can return as arbitrary mixtures. After the sequential tower (unchanged), each
    run of levels found by ``degenerate_clusters(energies, pair_tau, pair_m1, pair_max_k)`` is
    re-solved together by `dmrg_block`, deflated against the levels below it; each member is
    then polished alone by `dmrg_excited` from its block solution, deflated against the levels
    below it. Every other level from the first cluster up is re-solved from its own state in
    the same way, so the whole tower stays deflated against itself. ``pair_m1`` is the energy
    unit for ``pair_tau`` (default: the tower's first gap). ``infos[0]`` gains
    ``pair_clusters`` and ``energies_pre`` (the sequential tower); each repaired level's info
    gains ``repaired``, ``cluster`` and ``E_block``.

    Returns ``(energies, states, infos)``; ``states`` are host numpy arrays on every backend.
    """
    _dense_only(mpo, "dmrg_tower")
    if "options" in kw:
        raise TypeError("dmrg_tower takes dmrg_excited settings as flat keywords, not options=")
    states, energies, infos = [], [], []
    for n in range(nlevels):
        w = _tower_penalty(energies, penalty)
        for attempt in range(max_retries):
            warm = warm_chain and deflate == "penalty" and states and attempt == 0
            init = states[-1] if warm else None
            E, M, info = dmrg_excited(mpo, chi, prev_states=states, penalty=w,
                                      seed=seed + 17 * n + 101 * attempt,
                                      init_mps=init, verbose=False, deflate=deflate, **kw)
            ov = max(info["penalty_overlaps"], default=0.0)
            info["penalty"] = w
            info["attempt"] = attempt
            if ov < ov_tol:
                break
            w *= 4.0
            if verbose:
                print(f"  level {n}: overlap {ov:.2e} > {ov_tol:g}; "
                      f"retry with penalty {w:g}", flush=True)
        if verbose:
            print(f"  level {n}: E = {E:.12f}   sweeps={info['sweeps']:2d} "
                  f"max|<prev|psi>|={ov:.2e}  w={w:g}  disc={info['discarded_weight']:.2e}",
                  flush=True)
        states.append(_host(M))
        energies.append(E)
        infos.append(info)
    if not pair_solve:
        return np.array(energies), states, infos

    energies_pre = list(energies)
    clusters = degenerate_clusters(energies, pair_tau, m1_ref=pair_m1, max_k=pair_max_k)
    blk_kw = {key: kw[key] for key in ("n_sweeps", "tol", "eig_tol", "eig_max_iter", "backend")
              if key in kw}
    span = dict(clusters)
    n = clusters[0][0] if clusters else nlevels
    while n < nlevels:
        # a cluster is re-solved as a block; every other level from the first cluster up is
        # re-solved from its own state, so the whole tower stays deflated against the new one
        hi = span.get(n, n)
        w0 = max(infos[t]["penalty"] for t in range(n, hi + 1))
        if hi > n:
            for attempt in range(max_retries):
                E_blk, inits, binfo = dmrg_block(mpo, chi, hi - n + 1, prev_states=states[:n],
                                                 penalty=w0, deflate=deflate,
                                                 seed=seed + 17 * n + 9901 + attempt, **blk_kw)
                if max(binfo["penalty_overlaps"], default=0.0) < ov_tol:
                    break
                w0 *= 4.0
            inits = [_host(Mj) for Mj in inits]
        else:
            inits = [states[n]]
        for t in range(hi - n + 1):
            w, init = max(w0, _tower_penalty(energies[:n + t], penalty)), inits[t]
            for attempt in range(max_retries):
                E, M, info = dmrg_excited(mpo, chi, prev_states=states[:n + t], penalty=w,
                                          seed=seed + 17 * (n + t) + 9902 + 101 * attempt,
                                          init_mps=init, verbose=False, deflate=deflate, **kw)
                if max(info["penalty_overlaps"], default=0.0) < ov_tol:
                    break
                w, init = w * 4.0, None
            info["penalty"] = w
            if hi > n:
                info.update(repaired=True, cluster=(n, hi), E_block=E_blk[t])
            energies[n + t], states[n + t], infos[n + t] = E, _host(M), info
        if verbose and hi > n:
            print(f"  [pair] levels {n}-{hi}: {energies_pre[n:hi + 1]} -> "
                  f"{energies[n:hi + 1]}", flush=True)
        n = hi + 1

    order = np.argsort(energies, kind="stable")
    energies = [energies[t] for t in order]
    states = [states[t] for t in order]
    infos = [infos[t] for t in order]
    infos[0]["pair_clusters"] = clusters
    infos[0]["energies_pre"] = energies_pre
    return np.array(energies), states, infos
