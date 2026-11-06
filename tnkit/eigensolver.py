"""Matrix-free eigensolvers for DMRG, fully backend-agnostic.

The whole point: the only operation these need is the *action* of the effective
Hamiltonian on a vector (a tensor contraction). They never form the dense
operator and never call eigh/svd, so they run natively on ANY backend that does
matmul — including Apple's MPS and (in production) NVIDIA, where they keep the
entire eigensolve on-device with no host<->device transfer.

Two solvers:
  - lanczos_ground:  Lanczos with selective full reorthogonalisation, returns the
                     lowest eigenpair. Robust default for DMRG.
  - lobpcg_ground:   LOBPCG (block size 1) with optional preconditioner; often
                     fewer matvecs than Lanczos when warm-started from the
                     previous sweep's site tensor (the DMRG use-case).

`matvec` is any callable x -> H x operating on a flat 1-D array of the chosen
backend. `ops` is a tiny linear-algebra adaptor (see backend.py) so the same
code runs on numpy or torch tensors.
"""
from __future__ import annotations
import os
import warnings
import time
import numpy as np

from . import block_lanczos

try:
    from scipy.linalg import eigvalsh_tridiagonal, eigh_tridiagonal
    _HAVE_SCIPY_TRIDIAG = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY_TRIDIAG = False


def check_reorth_periodic(reorth, stop):
    """Guard against the REFUTED reorth='periodic' setting.

    Shared by every statics solver that accepts a `reorth` choice, not just
    `lanczos_ground` -- `reorth='periodic'` is refuted regardless of which local
    eigensolver a caller asked for (`lanczos`, `lanczos-restart`, ...), and an
    explicit request for it must not be silently ignored just because the chosen
    solver (e.g. thick restart) does not itself consult `reorth`.

    'periodic' re-orthogonalises only every `reorth_period` steps, which is
    identical to full for the ~2-iteration warm-started solves that dominate, but
    at chi >= 64 near criticality some solves need 100+ iterations, the basis
    loses orthogonality, a ghost eigenpair forms and DMRG stalls ABOVE the true
    energy (measured dE ~ 1.3-2.0 -- no ground state at all).

    The energy-delta stop MASKED this at moderate chi by quitting before the long
    solves ran far enough to matter. The residual stop does not, so the
    combination fails loudly. Neither should fail silently.
    """
    if reorth != "periodic":
        return
    if stop == "residual":
        raise ValueError(
            "reorth='periodic' with stop='residual' is refuted: periodic reorthogonalisation "
            "loses basis orthogonality on the long (100+ iteration) local solves that occur at "
            "chi >= 64 near criticality, and the residual rule then iterates against a "
            "tridiagonal that no longer describes the basis. Measured dE ~ 0.8-2.0 (no ground "
            "state). Use reorth='full' -- which is also FASTER here, since correct solves need "
            "a median of 2 iterations.")
    warnings.warn(
        "reorth='periodic' is refuted at chi >= 64 near criticality (ghost eigenpair from lost "
        "orthogonality). It only looks correct "
        "because short warm-started solves make it equivalent to 'full'. Use reorth='full'.",
        RuntimeWarning, stacklevel=2)


def _tridiag_ground_value(alphas, betas, k):
    """Smallest eigenvalue of the k x k symmetric tridiagonal (diag=alphas,
    off-diag=betas[:k-1]). Eigenvalues only — no eigenvectors — so this is cheap."""
    d = np.asarray(alphas[:k], dtype=np.float64)
    if k == 1:
        return float(d[0])
    e = np.asarray(betas[: k - 1], dtype=np.float64)
    if _HAVE_SCIPY_TRIDIAG:
        try:
            return float(eigvalsh_tridiagonal(d, e, select="i", select_range=(0, 0))[0])
        except Exception:                            # pragma: no cover
            pass
    T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    return float(np.linalg.eigvalsh(T)[0])


def _tridiag_ground_vector(alphas, betas, k):
    """Eigenvector of the smallest eigenpair of the k x k tridiagonal. Called once
    at the end, so the full eigenvector solve happens a single time per Lanczos."""
    d = np.asarray(alphas[:k], dtype=np.float64)
    if k == 1:
        return np.array([1.0])
    e = np.asarray(betas[: k - 1], dtype=np.float64)
    if _HAVE_SCIPY_TRIDIAG:
        try:
            _, vecs = eigh_tridiagonal(d, e, select="i", select_range=(0, 0))
            return vecs[:, 0]
        except Exception:                            # pragma: no cover
            pass
    T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    _, vecs = np.linalg.eigh(T)
    return vecs[:, 0]


def lanczos_expm_fixed(matvec, v, tau, ops, m, info=None, reorth="cgs2", expm_where="host",
                       squarings=8, taylor_order=18):
    """GPU FAST PATH for exp(tau H) v: a FIXED-length Krylov schedule with no host branch, one
    synchronisation per solve instead of ~2m, and batched re-orthogonalisation. Same mathematics
    as `lanczos_expm`, different control flow.

    Why a second implementation rather than a flag inside the first. The adaptive solver's cost
    on GPU is not arithmetic — measurement showed the 1TDVP
    per-step cost scaling as chi^0.14 against the FLOP-bound chi^3, i.e. the card is idle waiting for
    Python. The three things keeping it idle are all *control flow*, and each is impossible to
    remove without changing the loop's shape:

      1. **91 device->host synchronisations per site-step.** Every `to_scalar(vdot(...))` and
         `to_scalar(norm(...))` is an `.item()`: the host blocks until the queue drains, then
         the queue refills from empty. Here alphas and betas stay 0-d DEVICE tensors from the
         first iteration to the last, and the projected tridiagonal is assembled from them with
         `ops.stack` + `ops.tridiag`. Nothing is read back, so nothing stalls.
      2. **44 host `scipy.expm` calls per site-step.** The adaptive stop needs `expm(tau T)` at
         *every* Krylov step to evaluate the Saad first-neglected-term estimate, and computes it
         on the host from host floats. A fixed schedule needs the exponential exactly ONCE, at
         the end. Note what is NOT done here: moving that one exponential on-device
         (`torch.matrix_exp`) was MEASURED SLOWER than leaving it on the host, because
         scaling-and-squaring a 10x10 is tens of tiny kernel launches to do microseconds of
         arithmetic. The win is calling it once rather than m-1 times; where it runs is a
         second-order question and the measurement answered it the other way (`expm_where`).
      3. **403 re-orthogonalisation `vdot` calls per site-step.** The sequential
         modified-Gram-Schmidt loop `for u in V: w -= vdot(u,w)*u` issues two kernel launches per
         basis vector per iteration, i.e. O(m^2) launches. Two batched classical-Gram-Schmidt
         passes (CGS2) against a PREALLOCATED buffer give the same orthogonality in 4 launches
         per iteration, transported from the DMRG
         eigensolver to the dynamics one. Twice is what makes CGS as stable as MGS
         (Giraud-Langou-Rozloznik), and the crossover-with-basis-size trap is avoided by
         never restacking the basis per iteration.

    BREAKDOWN WITHOUT A BRANCH. A fixed schedule cannot test `beta < tol` and stop, because the
    test is a sync. It does not need to: the Lanczos coefficient that carries basis vector j into
    the answer is `expm(tau T)[j, 0]`, which is bounded by the product of the off-diagonals
    beta_1...beta_j over (j-1)!. If the space breaks down at step p then beta_p is at roundoff, so
    every j > p contributes at roundoff regardless of what direction the (meaningless) vector
    V_{j} points in. The only thing that must be prevented is 0/0, and `ops.guard_denom` does that
    arithmetically. Breakdown therefore costs a few wasted matvecs and nothing else.

    `m` is the FIXED Krylov dimension, capped at the vector length (a shape, not a value — free).
    Choosing it is the caller's job; `tnkit.tdvp` calibrates it from a warmup sweep against the
    adaptive solver's own measured requirement.

    `reorth` selects the re-orthogonalisation kernel: 'cgs2' (default, batched, plain matmuls),
    'cgs2e' (identical but expressed as `einsum`, to price the equation-string parsing), or 'mgs'
    (the sequential loop). 'mgs' exists so the per-lever benchmark can price the batched reorth
    SEPARATELY from the sync/host-expm removal, against the same fixed schedule — otherwise the
    two are confounded and the waterfall is a guess. It is also the honest fallback above the
    chi ~ 256 crossover where measurement showed CGS2 losing (0.9623x at
    chi=512), well above the chi-cap 48 a two-packet collision run uses.
    """
    n = 1
    for s in v.shape:
        n *= int(s)
    m = max(1, min(int(m), n))
    shape = v.shape

    # TWO preallocated buffers: the basis and its conjugate. Carrying the conjugate costs one
    # extra (m, n) allocation -- ~1 MB at chi-cap 48 -- and removes the
    # (k, n) `conj(Vlive)` copy that CGS2 would otherwise pay on EVERY iteration. Both operands
    # handed to einsum are then contiguous materialised buffers, which is what keeps the call on
    # the fast cuBLAS path; the alternative of conjugating `w` instead is the REFUTED `cgs2v2`
    # (23x slower -- a lazy conj view falls off that path).
    cgs = reorth in ("cgs2", "cgs2e")
    use_mm = (reorth == "cgs2")                      # 'cgs2e' keeps the einsum form, for the A/B
    Vb = ops.zeros((m, n))
    Vc = ops.zeros((m, n)) if cgs else None
    beta0 = ops.norm(v)                              # 0-d, DEVICE, never read
    v0 = v.reshape(-1) / ops.guard_denom(beta0)
    Vb[0] = v0
    if cgs:
        Vc[0] = ops.conj(v0)
    alphas, betas = [], []
    for k in range(m):
        w = matvec(Vb[k])
        if k > 0:
            w = w - betas[k - 1] * Vb[k - 1]
        # CGS2 against the live basis -- VIEWS into the preallocated buffers, never a restack
        # (`torch.stack(V)` per iteration is copy-bound in the
        # vector dimension and OOM'd at chi=512, and capping m does not cap that copy).
        # The FIRST projection coefficient against V_k IS the Lanczos alpha, so it is read out of
        # the pass that was going to compute it anyway rather than by a separate `vdot` + `axpy`
        # pair -- two fewer kernel launches per iteration, and one fewer rounding step.
        Vlk = Vb[:k + 1]
        if cgs:
            Vck = Vc[:k + 1]
            # `mv`/`vm` are plain matmuls. `torch.einsum` re-parses its equation STRING on every
            # call, and at m=10 this loop issues 40 of them per solve and ~160 per site-step, for
            # contractions whose shapes never change -- the torch-side twin of the recomputed
            # `einsum_path` the cost model flagged on the numpy arm. Priced as its own arm.
            mv_, vm_ = ((ops.mv, ops.vm) if use_mm else
                        (lambda A, x: ops.einsum("jn,n->j", A, x),
                         lambda cc, A: ops.einsum("j,jn->n", cc, A)))
            c = mv_(Vck, w)
            alphas.append(c[k].real)                 # 0-d, DEVICE
            w = w - vm_(c, Vlk)
            c = mv_(Vck, w)
            w = w - vm_(c, Vlk)
        else:                                        # 'mgs' -- the sequential loop, for the A/B
            a = ops.vdot(Vb[k], w).real
            alphas.append(a)
            w = w - a * Vb[k]
            for u in Vlk:
                w = w - ops.vdot(u, w) * u
        b = ops.norm(w)                              # 0-d, DEVICE
        if k < m - 1:
            betas.append(b)
            vn = w / ops.guard_denom(b)
            Vb[k + 1] = vn
            if cgs:
                Vc[k + 1] = ops.conj(vn)
    da, eb = ops.stack(alphas), (ops.stack(betas) if betas else None)
    if expm_where in ("eigh", "taylor"):
        # CUDA-GRAPH ROUTES. `torch.matrix_exp` is not capturable -- it reads ||A||_1
        # back to the host to choose its Pade degree -- so `expm_where='device'` cannot go
        # inside a graph even though every other operation in this function already can.
        # Both routes below are host-branch-free; see `backend.expm_sym_tridiag_eigh`.
        if expm_where == "eigh":
            E = ops.expm_sym_tridiag_eigh(da, eb, tau)
        else:
            E = ops.expm_taylor(tau * ops.tridiag(da, eb), squarings, taylor_order)
        c = beta0 * E[:, 0]
        out = ops.vm(c, Vb).reshape(shape)
        if info is not None:
            info["krylov_dim"] = m
            for _k in ("err_est", "beta1", "est_ref"):
                info[_k] = float("nan")
            info["est_ref_j"] = 0
        return out
    T = ops.tridiag(da, eb)
    if expm_where == "host":
        # ONE synchronisation per solve, on an m x m matrix (~1.6 kB at m=10), against the 2m the
        # adaptive path pays and the m-1 host `scipy.expm` calls it makes. `torch.matrix_exp` runs
        # scaling-and-squaring as a chain of tiny GEMMs plus a linear solve -- on a 10x10 that is
        # tens of kernel launches to do microseconds of arithmetic, and it MEASURED slower than
        # this single transfer (see the per-lever benchmark). The device path is kept behind
        # `expm_where='device'` because the balance must invert at larger m.
        E = ops.expm_host(tau * T)
        c = beta0 * ops.array(E[:, 0])
    else:
        c = beta0 * ops.expm(tau * T)[:, 0]
    out = ops.vm(c, Vb).reshape(shape)
    if info is not None:
        # Krylov dimension is a SHAPE and free to report; every other diagnostic in the contract
        # is a device scalar, and reading one back would reintroduce exactly the synchronisation
        # this path exists to delete. They are reported as NaN, which `_KrylovMeter` already
        # skips NaN-safely. The adaptive step-size controller consumes `est_ref` and therefore CANNOT
        # run on this path -- `tdvp*_evolve` refuses the combination rather than silently
        # feeding the controller a zero.
        info["krylov_dim"] = m
        info["err_est"] = float("nan")
        info["beta1"] = float("nan")
        info["est_ref"] = float("nan")
        info["est_ref_j"] = 0
    return out


def _tridiag(alphas, betas):
    """The Lanczos tridiagonal of `alphas` and the first ``len(alphas) - 1`` `betas`, in float64."""
    j = len(alphas)
    T = np.diag(np.asarray(alphas, dtype=np.float64))
    if j > 1:
        off = np.asarray(betas[: j - 1], dtype=np.float64)
        T += np.diag(off, 1) + np.diag(off, -1)
    return T


class _ExpmMonitor:
    """The host-side part of `lanczos_expm`, shared by its block twin: the breakdown test, the
    error estimates, the adaptive stop, the coefficients and `info`."""

    def __init__(self, tau, beta0, tol, adaptive, info, est_ref_j):
        if adaptive is None:                         # env override so the A/B needs no plumbing
            adaptive = os.environ.get("TNKIT_EXPM_ADAPTIVE", "1") != "0"
        self.tau, self.beta0, self.tol, self.adaptive = tau, beta0, tol, adaptive
        self.info, self.est_ref_j = info, est_ref_j
        self.est = float("inf")
        self.est_ref, self.est_ref_at, self.beta1 = float("nan"), 0, float("nan")

    def check(self, k, alphas, betas, beta):
        """After step `k` (alphas up to it, betas before it, its new norm `beta`): "breakdown"
        (stop; `beta` is not used), "converged" (stop after this vector) or None."""
        from scipy.linalg import expm as _expm
        if k == 0:
            self.beta1 = beta                        # = sqrt(<H^2>-<H>^2), free
        if beta < self.tol:
            if self.info is not None and self.est_ref_at == 0:
                self.est_ref, self.est_ref_at = 0.0, k + 1   # exact breakdown: no neglected term
            return "breakdown"
        # ADAPTIVE STOP. The fixed m=30 had no error estimate at all -- it neither checked that 30
        # was enough nor noticed when it was far too many. Measured:
        # at the default dt=0.05 machine precision is reached at m=8, so m=30 spent ~3.75x the
        # matvecs it needed in the dominant TDVP kernel. This is the standard a-posteriori Krylov
        # estimator (Saad 1992) -- the first neglected term, |beta_j| |tau| |exp(tau T)[j-1,0]| ||v||.
        # It is a conservative over-estimate (measured est/true in [4, 16]), so stopping on it is safe.
        jj = len(alphas)
        # `est_ref` freezes the SAME estimator at a fixed j so it stays dt-sensitive; the
        # adaptive stop drives `est` to `tol` and so cannot be used to choose dt.
        want_ref = self.info is not None and self.est_ref_at == 0 and jj >= self.est_ref_j
        if (self.adaptive and jj >= 2) or want_ref:
            Et = _expm(self.tau * _tridiag(alphas, betas))
            self.est = abs(beta) * abs(self.tau) * abs(Et[jj - 1, 0]) * self.beta0
            if want_ref:
                self.est_ref, self.est_ref_at = self.est, jj
            if self.adaptive and jj >= 2 and self.est < self.tol:
                return "converged"
        return None

    def finish(self, alphas, betas, capped):
        """The coefficients exp(tau T) (beta0 e_1); fills `info`."""
        from scipy.linalg import expm as _expm
        c = self.beta0 * _expm(self.tau * _tridiag(alphas, betas))[:, 0]
        if self.info is not None:
            self.info["krylov_dim"] = len(alphas)
            self.info["err_est"] = self.est
            self.info["beta1"] = self.beta1
            self.info["est_ref"] = self.est_ref
            self.info["est_ref_j"] = self.est_ref_at
            self.info["capped"] = capped
        return c


def lanczos_expm(matvec, v, tau, ops, m=30, tol=1e-12, adaptive=None, info=None,
                 est_ref_j=2):
    """Krylov (Lanczos) approximation of exp(tau * H) @ v for Hermitian H, using
    only the matrix-free action `matvec` (a contraction). tau = -1j*dt for
    real-time evolution. Returns the evolved vector.

    Builds the Krylov subspace K_m(H, v), forms the small (m x m) tridiagonal T,
    and returns ||v|| * V @ expm(tau*T) @ e_1. Full reorthogonalisation (the Krylov
    space is short; orthogonality is essential for a good exponential). This is the
    time-evolution twin of `lanczos_ground` — the same matrix-free machinery serves
    ground states (eigenpair) and dynamics (exponential).

    `info` (optional dict) collects the a-posteriori diagnostics:

      krylov_dim : m actually used
      err_est    : the Saad estimate at the dimension where the loop STOPPED. With
                   `adaptive=True` this is pinned just under `tol` by construction and
                   therefore carries NO information about dt -- do not control on it.
      beta1      : the first Lanczos off-diagonal = sqrt(<H^2>-<H>^2) on the normalised
                   input vector, i.e. the local energy spread of this block. Free.
      est_ref    : the Saad first-neglected-term estimate frozen at the FIXED reference
                   dimension `est_ref_j` (default 2). Unlike `err_est` this is a genuine
                   function of |tau|: est_ref ~ |tau|^j * beta_1...beta_j / (j-1)!, so it
                   scales as dt^j and can be inverted for a step size. This is the signal
                   the adaptive dt controller uses.
      est_ref_j  : the j at which est_ref was frozen (may be < est_ref_j if the Krylov
                   space broke down early).
      capped     : True when the adaptive solve used all `m` vectors without meeting `tol`
                   (no breakdown, no adaptive stop), i.e. the cap decided the dimension.

    For a block tensor `v`, `block_lanczos.lanczos_expm` runs instead unless it is switched off: the
    same result, bit for bit, for less host work.
    """
    if getattr(v, "is_block", False) and block_lanczos.enabled():
        return block_lanczos.lanczos_expm(matvec, v, tau, ops, m=m, tol=tol, adaptive=adaptive,
                                          info=info, est_ref_j=est_ref_j)
    beta0 = ops.to_scalar(ops.norm(v)).real
    if not np.isfinite(beta0) or beta0 < 1e-300:
        return v
    mon = _ExpmMonitor(tau, beta0, tol, adaptive, info, est_ref_j)
    V = [v / beta0]
    alphas, betas = [], []
    beta = 0.0
    v_prev = ops.zeros_like(v)
    capped = False
    for k in range(m):
        w = matvec(V[-1]) - beta * v_prev
        alpha = ops.to_scalar(ops.vdot(V[-1], w)).real
        alphas.append(alpha)
        w = w - alpha * V[-1]
        for u in V:                                  # full reorthogonalisation
            w = w - ops.vdot(u, w) * u
        beta = ops.to_scalar(ops.norm(w)).real
        stop = mon.check(k, alphas, betas, beta)
        if stop == "breakdown":
            break
        betas.append(beta)
        v_prev = V[-1]
        V.append(w / beta)
        if stop:
            break
    else:
        capped = bool(mon.adaptive)
    c = mon.finish(alphas, betas, capped)
    out = ops.zeros_like(V[0])
    for coeff, vj in zip(c, V):
        out = out + complex(coeff) * vj
    return out


def _tridiag_ground_pair(alphas, betas, k):
    """(smallest eigenvalue, its eigenvector) of the k x k tridiagonal, in one solve.

    Used by the residual stopping rule, which needs the LAST COMPONENT of the ground Ritz
    vector every step. Cost is O(k^2)-O(k^3) host work on a k <= max_iter tridiagonal —
    negligible beside one device matvec (3.2 GFLOP at chi=512), and the same trick
    ``lanczos_ground_restart`` already uses every step.
    """
    d = np.asarray(alphas[:k], dtype=np.float64)
    if k == 1:
        return float(d[0]), np.array([1.0])
    e = np.asarray(betas[: k - 1], dtype=np.float64)
    if _HAVE_SCIPY_TRIDIAG:
        try:
            vals, vecs = eigh_tridiagonal(d, e, select="i", select_range=(0, 0))
            return float(vals[0]), vecs[:, 0]
        except Exception:                            # pragma: no cover
            pass
    T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    vals, vecs = np.linalg.eigh(T)
    return float(vals[0]), vecs[:, 0]


def lanczos_ground(matvec, v0, ops, tol=1e-8, max_iter=200, reorth="full",
                   reorth_period=5, timers=None, stop="residual"):
    """Lowest eigenpair of a Hermitian operator via matrix-free Lanczos.

    Parameters
    ----------
    matvec : callable
        x -> H @ x, with x a flat backend array.
    v0 : backend array
        Starting vector (e.g. the current site tensor flattened — a warm start
        dramatically cuts iterations in DMRG).
    ops : Backend
        Linear-algebra adaptor (vdot, norm, zeros_like, to_scalar ...).
    tol : float
        Convergence threshold, read against `stop`.
    stop : {"residual", "energy"}
        WHAT `tol` MEASURES, and the single most consequential argument here.

          residual (default) — the true Ritz residual ||H x - e x|| = |beta|*|y[-1]|,
              i.e. convergence of the EIGENVECTOR. This is the quantity DMRG
              consumes: x goes straight into the SVD truncation and into the MPS.
          energy (legacy)    — |e - e_prev| < tol, i.e. convergence of the Rayleigh
              quotient, which is what most textbook implementations stop on.

        These are NOT the same standard, and the difference is not small. The
        Rayleigh quotient is QUADRATICALLY accurate in the eigenvector error, so
        the energy rule returns a vector only ~sqrt(tol) converged: measured at
        tol = 1e-10, "energy" leaves a residual of ~1e-5 where "residual" leaves
        ~1e-11, and at g = 0.9 that cost five orders of DMRG accuracy.

        The failure is silent — a run under "energy" reports converged=True and an
        energy that looks settled. This distinction has invalidated real comparisons
        before: a head-to-head against the thesis implementation reported a 6.54x
        speed-up while actually stopping 16,000x short of the comparison run's
        convergence; an external baseline built on cuTensorNet showed a flat accuracy
        floor of 2.2e-5 at every chi, which is sqrt(1e-9) — the stopping rule, not the
        backend; a thick-restart pre-adoption run reported converged=True at a true
        residual of 7.7e-6; and a finite-size-scaling exponent revalidation had the
        energy rule masking a separately refuted reorthogonalisation setting.
        Measured, "residual" gives 172x / 1,272x / 21,794x better dE at chi = 128 /
        256 / 512, for 8-25% end-to-end wall.

        Keep the default. Set "energy" only to reproduce an old number, and record
        that you did. `dmrg` exposes it as `eig_stop`.
    max_iter : int
        Hard cap on Krylov dimension.
    reorth : {"full", "periodic", "none"} (or bool for back-compat)
        Reorthogonalisation strategy against the stored basis:
          full     — every step (numerically safest; O(k^2) vector ops)
          periodic — every `reorth_period` steps (cheaper; good enough for the
                     short Krylov spaces of warm-started DMRG)
          none     — rely on the 3-term recurrence only (fastest; may lose
                     orthogonality and need more iters / lose a little accuracy)
    reorth_period : int
        Full-reorth cadence when reorth == "periodic".
    timers : dict | None
        If given, accumulates wall-time by phase: 'matvec', 'reorth', 'tridiag'.

    Returns
    -------
    energy : float
    eigvec : backend array   (flat, normalised)
    info : dict              (iters, converged)
    """
    if reorth is True:
        reorth = "full"
    elif reorth is False:
        reorth = "none"

    # reorth="periodic" is REFUTED -- see
    # `check_reorth_periodic` above, which is the single source of truth for this check so
    # that every statics solver enforces it identically, not just this one.
    check_reorth_periodic(reorth, stop)
    _t = (lambda: time.perf_counter()) if timers is not None else None

    n0 = ops.to_scalar(ops.norm(v0)).real
    if not np.isfinite(n0) or n0 < 1e-300:       # degenerate start -> nothing to do
        return 0.0, v0, {"iters": 0, "converged": False}
    v = v0 / n0
    V = [v]
    alphas, betas = [], []          # betas holds only *confirmed* off-diagonals
    beta_prev = 0.0
    v_prev = ops.zeros_like(v)
    prev_e = None
    converged = False
    for it in range(max_iter):
        # three-term recurrence (standard Lanczos)
        if _t:
            t0 = _t()
        w = matvec(v) - beta_prev * v_prev
        alpha = ops.to_scalar(ops.vdot(v, w)).real
        alphas.append(alpha)
        w = w - alpha * v
        if _t:
            timers["matvec"] += _t() - t0
            t0 = _t()
        do_reorth = (reorth == "full") or (reorth == "periodic"
                                           and (it % reorth_period == 0))
        if do_reorth:
            for u in V:
                w = w - ops.vdot(u, w) * u
        if _t:
            timers["reorth"] += _t() - t0
            t0 = _t()

        # tridiagonal ground energy from confirmed quantities only (size k).
        k = len(alphas)
        if stop == "residual":
            e, y_min = _tridiag_ground_pair(alphas, betas, k)
        else:
            # legacy: eigenvalue only (cheaper per step, but see the note below)
            e = float(_tridiag_ground_value(alphas, betas, k))
            y_min = None
        if _t:
            timers["tridiag"] += _t() - t0

        beta = ops.to_scalar(ops.norm(w)).real
        # breakdown (invariant subspace) — relative threshold, checked BEFORE
        # any division by beta, so we never form NaNs. Non-finite beta/alpha also
        # ends the iteration (defensive: a malformed block can't poison eigh).
        breakdown = (not np.isfinite(beta) or not np.isfinite(alpha)
                     or beta < 1e-12 * max(1.0, abs(alpha)))
        # STOPPING RULE.
        #   "residual" (default): the TRUE Ritz residual ||H x - e x|| = |beta| * |y_min[-1]|,
        #     i.e. convergence of the EIGENVECTOR — the quantity DMRG actually consumes, since
        #     x goes straight into the SVD truncation and into the MPS.
        #   "energy" (legacy): |e - e_prev| < tol, i.e. convergence of the Rayleigh quotient.
        # These are not the same standard. The Rayleigh quotient is quadratically accurate in
        # the eigenvector error, so an energy-delta stop leaves the vector only ~sqrt(tol)
        # converged: measured at tol=1e-10, "energy" returns residual ~1e-5 while "residual"
        # returns ~1e-11, and at g=0.9 that cost 5 orders of DMRG accuracy.
        if stop == "residual":
            hit = abs(beta) * abs(float(y_min[-1])) < tol
        else:
            hit = prev_e is not None and abs(e - prev_e) < tol
        if breakdown or hit:
            converged = True
            it += 1
            break
        prev_e = e
        betas.append(beta)
        v_prev = v
        beta_prev = beta
        v = w / beta
        V.append(v)
    else:
        it = max_iter

    # reconstruct the ground eigenvector in the original space (one eigh, at the
    # end, on the final tridiagonal): x = sum_j evecs[j, 0] * V[j]
    ground = _tridiag_ground_vector(alphas, betas, len(alphas))
    x = ops.zeros_like(V[0])
    for coeff, vj in zip(ground[: len(V)], V):
        x = x + float(coeff) * vj
    x = x / ops.norm(x)
    return e, x, {"iters": it, "converged": converged}


def lanczos_ground_restart(matvec, v0, ops, tol=1e-8, max_iter=200, m=40, k=10,
                           timers=None, reorth_alg="cgs2"):
    """Lowest eigenpair via THICK-RESTART Lanczos (Wu & Simon, SIAM J. Matrix Anal.
    2000) — memory-capped Krylov with full reorthogonalisation *within the window*.

    The `reorth="full"` correctness fix
    stores the WHOLE Krylov basis; near criticality local solves run 100+ iterations, so at
    χ≥1024 the basis alone is tens of GB and the O(k²) reorth cost explodes. Thick-restart
    caps the basis at ``m`` vectors: run Lanczos to dimension ``m``, keep the ``k`` lowest
    Ritz vectors (which carry the useful spectral information) plus the residual direction,
    and continue. This is a *different* answer to the same orthogonality-loss problem that
    caused the ghost-eigenpair stall in periodic reorth — here orthogonality is maintained
    exactly against the ≤m stored vectors, so no ghost eigenpair can form.

    The projected matrix is tridiagonal within a cycle and picks up an **arrowhead** border
    at each restart (the kept Ritz values on the diagonal, coupled to the first post-restart
    vector by ``beta_last * (last component of each kept Ritz vector)``). It stays real
    symmetric (Hermitian H ⇒ real tridiagonal/Ritz values), size ≤ m, so the projected
    eigensolve is a cheap dense ``eigh`` on the host — identical trick to ``lanczos_ground``.

    Convergence is checked EVERY step on the true Ritz residual ``|beta| * |y_min[-1]|`` (=
    ‖H x - θ x‖), so warm-started easy DMRG solves still finish in ~2 iterations and only
    the long near-critical solves ever restart. Memory: ≤ m+1 device vectors, independent of
    how many iterations convergence takes.

    Parameters mirror ``lanczos_ground`` plus:
      m : int   Krylov window (max stored basis vectors). Full reorth within it (cheap ≤40).
      k : int   Ritz vectors kept on restart (k < m; typ. m//4). Must keep the target plus a
                few nearby to avoid re-discovering the subspace each cycle.
    """
    _t = (lambda: time.perf_counter()) if timers is not None else None
    if k >= m:
        k = max(1, m // 2)

    n0 = ops.to_scalar(ops.norm(v0)).real
    if not np.isfinite(n0) or n0 < 1e-300:
        return 0.0, v0, {"iters": 0, "converged": False, "restarts": 0}
    # PREALLOCATED basis buffer: one (m+1, n) allocation for the whole solve, rows reused across
    # restarts. Replaces the Python list of vectors -- see the re-basis note at the restart below.
    nvec = int(np.prod(v0.shape))
    Vb = ops.zeros((m + 1, nvec))
    Vb[0] = v0.reshape(-1) / n0
    nV = 1                                       # number of live rows in Vb
    # Dense projected matrix M (real symmetric), grown in place; H holds host copy.
    M = np.zeros((m + 1, m + 1), dtype=np.float64)
    n_kept = 0            # number of Ritz vectors carried from the previous cycle (border block)
    border = None        # length-n_kept array of arrowhead couplings for the current first step
    beta_prev = 0.0
    v_prev = ops.zeros_like(Vb[0])
    theta = 0.0
    y_min = np.array([1.0])
    converged = False
    restarts = 0
    total_iters = 0

    j = n_kept                                   # index of the vector we are about to expand
    while total_iters < max_iter:
        v = Vb[j]
        if _t:
            t0 = _t()
        w = matvec(v)
        # subtract the three-term / arrowhead couplings BEFORE reorth
        if n_kept and j == n_kept:               # first step after a restart: arrowhead border
            w = w - ops.einsum("r,rn->n", ops.array(border), Vb[:n_kept])
        elif j > 0:
            w = w - beta_prev * v_prev
        alpha = ops.to_scalar(ops.vdot(v, w)).real
        w = w - alpha * v
        if _t:
            timers["matvec"] += _t() - t0
            t0 = _t()
        # Full reorth within the window (≤ m+1 vectors).
        #   "cgs2": two BATCHED classical-Gram-Schmidt passes. Each pass is two GEMVs against
        #     the whole live buffer instead of nV sequential vdot+axpy pairs, so the traffic is
        #     one streaming read of Vb per pass rather than nV separate kernel launches. Twice
        #     is what makes CGS as stable as MGS (Giraud-Langou-Rozloznik).
        #   "mgs": the original sequential modified-Gram-Schmidt loop, kept for A/B.
        # Reorth is ~34% of the eigensolver, and it is the bandwidth-bound term that caps every
        # GEMM-side speed-up (the Amdahl bound on any scheme that only accelerates the matmuls).
        if reorth_alg in ("cgs2", "cgs2v2"):
            Vlive = Vb[:nV]
            if reorth_alg == "cgs2v2":
                # Avoid materialising conj(Vlive): that is a full (nV, n) copy EVERY step
                # (~650 MB at chi=512 with nV=41), which is what erased cgs2's advantage at
                # large chi. Use <u,w> = conj(u).w = conj( u . conj(w) ) so only the length-n
                # vector w and the length-k coefficients are conjugated, never the buffer.
                wc = ops.conj(w)
                for _ in range(2):
                    c = ops.conj(ops.einsum("kn,n->k", Vlive, wc))
                    w = w - ops.einsum("k,kn->n", c, Vlive)
                    wc = ops.conj(w)
            else:
                Vc = ops.conj(Vlive)
                for _ in range(2):
                    c = ops.einsum("kn,n->k", Vc, w)
                    w = w - ops.einsum("k,kn->n", c, Vlive)
        else:
            for u in Vb[:nV]:
                w = w - ops.vdot(u, w) * u
        if _t:
            timers["reorth"] += _t() - t0
            t0 = _t()

        M[j, j] = alpha
        beta = ops.to_scalar(ops.norm(w)).real

        # projected eigenproblem on the current (j+1)x(j+1) block
        cur = j + 1
        if cur == 1:
            theta_new, y_min = alpha, np.array([1.0])
        else:
            evals, evecs = np.linalg.eigh(M[:cur, :cur])
            theta_new, y_min = float(evals[0]), evecs[:, 0]
        if _t:
            timers["tridiag"] += _t() - t0

        total_iters += 1
        # Ritz residual = ||H x - theta x|| = |beta| * |last component of Ritz vector|
        res = abs(beta) * abs(float(y_min[-1])) if cur > 0 else abs(beta)
        breakdown = (not np.isfinite(beta) or not np.isfinite(alpha)
                     or beta < 1e-12 * max(1.0, abs(alpha)))
        theta = theta_new
        if breakdown or res < tol:
            converged = True
            break

        if cur >= m:                             # window full -> thick restart
            evals, evecs = np.linalg.eigh(M[:cur, :cur])
            kk = min(k, cur - 1)
            Yk = evecs[:, :kk]                    # kk lowest Ritz vectors (columns)
            # New basis: kk Ritz vectors U = Yk^T @ Vb[:cur], then the carried residual v_new.
            # BATCHED: one (kk x cur)(cur x n) GEMM. The previous form was a kk*cur loop of
            # full-length AXPYs, each allocating -- ~400 device vector ops per restart at the
            # default m=40,k=10. Free at the chi=32/64 the restart was validated at; at chi=512 with
            # max_iter=4*chi it is up to ~50 restarts ~ 20k vector ops, the same copy-bound
            # pattern that made naive CGS2 slower at chi=256 and OOM at chi=512.
            # Writes into a temp then back into the buffer, so
            # peak stays (m+1+kk) rows rather than doubling.
            vnew = w / beta                       # next Lanczos direction (residual)
            U = ops.einsum("rc,rn->cn", ops.array(Yk), Vb[:cur])   # (kk, n)
            Vb[:kk] = U
            Vb[kk] = vnew
            nV = kk + 1
            del U
            # rebuild projected matrix: diag = kept Ritz values; arrowhead border to v_new
            M[:] = 0.0
            border = beta * np.asarray(Yk[cur - 1, :], dtype=np.float64)   # beta_m * eta_i
            for i in range(kk):
                M[i, i] = float(evals[i])
                M[i, kk] = border[i]                 # arrowhead coupling Ritz_i <-> v_new ...
                M[kk, i] = border[i]                 # ... written into the projected matrix
            n_kept = kk
            j = kk
            beta_prev = 0.0
            v_prev = ops.zeros_like(Vb[0])
            # Re-express the current best Ritz vector in the NEW basis. The kept columns are the
            # kk lowest Ritz vectors in ascending order, so the target is exactly new-basis
            # vector 0. Without this, an exit on the max_iter bound immediately after a restart
            # would reconstruct x with the PRE-restart y_min (length cur) against the new
            # length-(kk+1) basis -- zip() truncates and returns a non-eigenvector while theta
            # still looks converged.
            y_min = np.zeros(kk + 1, dtype=np.float64)
            y_min[0] = 1.0
            restarts += 1
            continue

        # ordinary tridiagonal extension
        M[j, j + 1] = beta
        M[j + 1, j] = beta
        v_prev = v
        beta_prev = beta
        Vb[j + 1] = w / beta
        nV = j + 2
        j += 1

    # reconstruct the ground eigenvector: x = sum_r y_min[r] Vb[r]  (batched)
    ny = min(len(y_min), nV)
    x = ops.einsum("r,rn->n", ops.array(y_min[:ny]), Vb[:ny])
    x = x / ops.norm(x)
    return theta, x, {"iters": total_iters, "converged": converged, "restarts": restarts}


def lanczos_ground_mixed(matvec_lo, matvec_hi, v0, ops, tol=1e-8, max_iter=200,
                         m=40, k=10, tol_lo=1e-4, timers=None, counts=None):
    """Mixed-precision thick-restart Lanczos with an fp64 guardrail.

    The inexact-Krylov bet: Lanczos builds its subspace *iteratively*, so a NOISY (fp32/TF32)
    matvec can still steer the Krylov space to the right invariant subspace — unlike a
    one-shot randomized sketch, whose error is frozen in. We therefore spend the cheap
    ~25-48x fp32/TF32 matvec on the bulk of the iterations, then pay a tiny fp64 tail to
    certify the answer:

      Stage 1 (cheap): thick-restart Lanczos with ``matvec_lo`` (fp32/TF32 GEMMs) but ALL
        projection/accumulation in fp64 (``ops`` is the fp64 backend: vdot/norm, the
        tridiagonal, and the Rayleigh quotient are fp64 over fp32-rounded vectors). Converge
        the Ritz residual to ``tol_lo`` (~1e-4) — cheap because the operator error, not the
        subspace, dominates there.
      Guardrail (fp64): the Rayleigh quotient of an approximate eigenvector has error
        O(residual^2) (Kato/Temple), so a SINGLE fp64 matvec turns a 1e-4 vector into a
        ~1e-8 *energy*. We then run a short fp64 thick-restart refinement warm-started from
        the Stage-1 vector to drive the residual itself to ``tol``. Warm-started that close,
        it needs only a handful of fp64 matvecs — the eigenpair analogue of the truncation
        SVD's fp32-sketch / fp64-core split.

    ``matvec_lo`` / ``matvec_hi`` : x -> H x in low / full precision. Both take and return
        fp64 backend vectors (the low one casts to fp32 internally); only the GEMM precision
        differs. ``counts`` : optional dict, accumulates 'lo' and 'hi' matvec counts so the
        precision split can be reported (most matvecs should be 'lo').
    """
    # Stage 1 — fp32/TF32 matvecs, fp64 accumulation, converge to tol_lo.
    tol_lo = max(tol_lo, tol)
    c1 = {"n": 0}
    def mv1(x):
        c1["n"] += 1
        return matvec_lo(x)
    e_lo, x_lo, info_lo = lanczos_ground_restart(mv1, v0, ops, tol=tol_lo,
                                                 max_iter=max_iter, m=m, k=k, timers=timers)
    # Stage 2 — fp64 guardrail/refinement, warm-started from the fp32 eigenvector.
    c2 = {"n": 0}
    def mv2(x):
        c2["n"] += 1
        return matvec_hi(x)
    e_hi, x_hi, info_hi = lanczos_ground_restart(mv2, x_lo, ops, tol=tol,
                                                 max_iter=max_iter, m=m, k=k, timers=timers)
    if counts is not None:
        counts["lo"] = counts.get("lo", 0) + c1["n"]
        counts["hi"] = counts.get("hi", 0) + c2["n"]
    info = {"iters": info_lo["iters"] + info_hi["iters"],
            "converged": info_hi["converged"],
            "restarts": info_lo["restarts"] + info_hi["restarts"],
            "n_lo": c1["n"], "n_hi": c2["n"], "e_lo": e_lo}
    return e_hi, x_hi, info


def scipy_host_ground(matvec, v0, ops, tol=1e-8, max_iter=200, transfers=None):
    """REFERENCE BASELINE ONLY -- emulates the prior thesis's eigensolver architecture.
    Not a production solver: it exists so the thesis comparison can be run on ONE card.

    The thesis could not run a matrix-free eigensolver inside PyTorch, so it drove
    `scipy.sparse.linalg.eigsh` (Lanczos loop on the CPU, in host memory) against an
    effective-Hamiltonian *action* evaluated on the GPU. On a discrete GPU that pays a
    host->device copy of the input vector and a device->host copy of the result on EVERY
    matvec -- the thesis's measured #1 bottleneck.

    This function reproduces exactly that: scipy's Lanczos loop on the host, wrapping the
    caller's device `matvec` in a LinearOperator that transfers both ways per call. Passing
    `eig="scipy-host"` to dmrg() therefore changes ONE thing versus `eig="lanczos"` -- where
    the Krylov loop lives and whether vectors cross PCIe -- while the sweep, environments,
    truncation, precision and device stay identical. That isolates the architectural
    contribution from the hardware, which a cross-card wall-time comparison cannot.

    `transfers` : optional dict; if given, accumulates 'n' (matvec count) and 'bytes'
    (total host<->device traffic) so the transfer volume can be reported.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    v0_np = ops.to_numpy(v0).ravel()
    n = int(v0_np.size)
    dtype = v0_np.dtype

    def host_matvec(x_host):
        x_host = np.asarray(x_host, dtype=dtype).ravel()
        x_dev = ops.array(x_host)                 # host -> device (the thesis's per-matvec cost)
        y_dev = matvec(x_dev)                     # the GPU-side effective-Hamiltonian action
        y_host = ops.to_numpy(y_dev).ravel()      # device -> host
        if transfers is not None:
            transfers["n"] = transfers.get("n", 0) + 1
            transfers["bytes"] = transfers.get("bytes", 0) + x_host.nbytes + y_host.nbytes
        return y_host

    A = LinearOperator((n, n), matvec=host_matvec, dtype=dtype)
    if n <= 2:                                    # eigsh needs k < n-1; trivial block
        return lanczos_ground(matvec, v0, ops, tol=tol, max_iter=max_iter)
    vals, vecs = eigsh(A, k=1, which="SA", v0=v0_np, tol=tol,
                       maxiter=max_iter * 10, ncv=min(n - 1, max(20, min(100, n - 1))))
    e = float(vals[0])
    x = ops.array(np.ascontiguousarray(vecs[:, 0], dtype=dtype))
    x = x / ops.norm(x)
    return e, x, {"iters": -1, "converged": True}


def lobpcg_ground(matvec, v0, ops, tol=1e-8, max_iter=200, precond=None):
    """Lowest eigenpair via single-vector LOBPCG (locally optimal CG).

    Often converges in fewer matvecs than Lanczos when `v0` is a good guess
    (the DMRG warm-start), and accepts a diagonal preconditioner. Implemented
    with a 3-vector Rayleigh-Ritz on the host.
    """
    x = v0 / ops.norm(v0)
    p = None
    prev_e = None
    Ax = matvec(x)
    for it in range(max_iter):
        mu = ops.to_scalar(ops.vdot(x, Ax)).real
        r = Ax - mu * x                              # residual
        rnorm = ops.to_scalar(ops.norm(r)).real
        if prev_e is not None and abs(mu - prev_e) < tol and rnorm < 1e-6:
            return mu, x, {"iters": it, "converged": True}
        prev_e = mu
        w = precond(r) if precond is not None else r
        w = w - ops.vdot(x, w) * x                   # orthogonalise to x
        wn = ops.to_scalar(ops.norm(w)).real
        if wn < 1e-14:
            return mu, x, {"iters": it, "converged": True}
        w = w / wn

        basis = [x, w] if p is None else [x, w, p]
        Abasis = [Ax, matvec(w)] + ([matvec(p)] if p is not None else [])
        m = len(basis)
        G = np.zeros((m, m), dtype=np.complex128)    # Rayleigh quotient matrix
        S = np.zeros((m, m), dtype=np.complex128)    # overlap (Gram) matrix
        for i in range(m):
            for j in range(m):
                G[i, j] = ops.to_scalar(ops.vdot(basis[i], Abasis[j]))
                S[i, j] = ops.to_scalar(ops.vdot(basis[i], basis[j]))
        G = 0.5 * (G + G.conj().T)
        S = 0.5 * (S + S.conj().T)
        evals, evecs = _eigh_general(G, S)
        c = evecs[:, 0]

        # new x and new search direction p (exclude the x-component)
        x_new = _combine(basis, c, ops)
        Ax_new = _combine(Abasis, c, ops)
        c_p = c.copy(); c_p[0] = 0.0
        p = _combine(basis, c_p, ops)
        nx = ops.norm(x_new)
        x, Ax = x_new / nx, Ax_new / nx
        p = p / ops.norm(p) if ops.to_scalar(ops.norm(p)).real > 1e-14 else None
    return mu, x, {"iters": max_iter, "converged": False}


def _combine(vecs, coeffs, ops):
    out = ops.zeros_like(vecs[0])
    for coeff, v in zip(coeffs, vecs):
        out = out + complex(coeff) * v
    return out


def _eigh_general(G, S):
    """Solve the small generalised eigenproblem G c = lambda S c (S SPD)."""
    # Cholesky of S, transform to standard eigenproblem
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    M = Linv @ G @ Linv.conj().T
    M = 0.5 * (M + M.conj().T)
    w, y = np.linalg.eigh(M)
    c = Linv.conj().T @ y
    return w, c


def block_lowk(matvec, V0, ops, nev=2, tol=1e-8, max_iter=200, seed=0, prep=None):
    """The ``nev`` lowest eigenpairs of a Hermitian operator, solved together.

    Use it where the wanted eigenvalues are (near-)degenerate: a one-vector solver returns
    an arbitrary mixture inside such a subspace, while the subspace's Rayleigh-Ritz pairs are
    well defined. Block Krylov in residual-expansion form: the basis starts from ``V0``
    (topped up with random vectors) and grows each cycle by the wanted pairs' residuals,
    orthogonalised against the whole basis by two classical Gram-Schmidt passes. Stops when
    the largest true residual ``||A x - theta x||`` is below ``tol`` or the basis holds
    ``max_iter`` vectors (the same matvec budget as a one-vector solve with that cap).

    ``prep``, if given, is applied to the random top-up vectors (e.g. a projector the
    operator is restricted by).

    Returns ``(evals, evecs, info)``: ascending eigenvalues, flat orthonormal vectors, and
    ``{"iters", "converged", "resid"}``.
    """
    dim = int(np.prod(V0[0].shape))
    nev = int(max(1, nev))
    rng = np.random.default_rng(seed)
    dtype = np.complex128 if ops.is_complex else np.float64
    Q, AQ = [], []

    def _append(v):
        if Q:
            Qm = ops.stack(Q)
            for _ in range(2):
                v = v - ops.vm(ops.mv(ops.conj(Qm), v), Qm)
        n = float(np.real(ops.to_scalar(ops.norm(v))))
        if n < 1e-10:
            return False
        Q.append(v / n)
        AQ.append(matvec(Q[-1]))
        return True

    for v in list(V0)[:nev]:
        _append(v)
    for _ in range(4 * nev):
        if len(Q) >= min(nev, dim):
            break
        r = ops.array(rng.standard_normal(dim))
        _append(prep(r) if prep is not None else r)
    if not Q:
        raise RuntimeError("block_lowk: could not build a starting block")

    G = np.zeros((0, 0), dtype=dtype)
    while True:
        n, n_old = len(Q), G.shape[0]
        if n > n_old:                       # extend the projected matrix by the new rows
            blk = ops.to_numpy(ops.einsum("ij,kj->ik", ops.conj(ops.stack(Q[n_old:])),
                                          ops.stack(AQ)))
            Gn = np.zeros((n, n), dtype=dtype)
            Gn[:n_old, :n_old] = G
            Gn[n_old:, :] = blk
            Gn[:, n_old:] = blk.conj().T
            G = 0.5 * (Gn + Gn.conj().T)
        evals, y = np.linalg.eigh(G)
        k = min(nev, n)
        Qm, AQm = ops.stack(Q), ops.stack(AQ)
        xs, rs = [], []
        for m in range(k):
            ym = ops.array(np.ascontiguousarray(y[:, m]))
            xs.append(ops.vm(ym, Qm))
            rs.append(ops.vm(ym, AQm) - float(evals[m]) * xs[-1])
        resid = max(float(np.real(ops.to_scalar(ops.norm(r)))) for r in rs)
        if resid < tol or n >= min(max_iter, dim):
            converged = resid < tol
            break
        if not sum(_append(r) for r in rs):
            converged = True                # residuals lie in the basis: invariant subspace
            break
    return ([float(e) for e in evals[:k]], xs,
            {"iters": len(Q), "converged": bool(converged), "resid": resid})
