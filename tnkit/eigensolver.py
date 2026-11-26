"""Matrix-free eigensolvers and Krylov exponentials for DMRG and TDVP, on any backend.

These need only the action of the effective Hamiltonian on a vector (a tensor contraction).
They never form the dense operator and never call eigh or svd on it, so they run on any backend
that does matmul (numpy, torch on CPU, CUDA or Apple MPS) and keep the whole solve on the device.

Ground states:
  - lanczos_ground: Lanczos with full reorthogonalisation; the lowest eigenpair.
  - lanczos_ground_restart: thick-restart Lanczos, the same answer with a bounded basis.
  - lanczos_ground_mixed: thick restart with low-precision matvecs and a double-precision
    refinement.
  - lobpcg_ground: LOBPCG (block size 1).
  - scipy_host_ground: scipy's Lanczos on the host around a device matvec.

`matvec` is any callable x -> H x on a flat 1-D array of the chosen backend. `ops` is the
backend (see backend.py), so the same code runs on numpy or torch tensors.
"""
from __future__ import annotations
import os
import warnings
import time
import numpy as np

from . import block_lanczos
from .blocks import BlockTensor

try:
    from scipy.linalg import eigvalsh_tridiagonal, eigh_tridiagonal
    _HAVE_SCIPY_TRIDIAG = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY_TRIDIAG = False


def check_reorth_periodic(reorth, stop):
    """Refuse or warn on ``reorth='periodic'``, for every statics solver that takes `reorth`.

    'periodic' reorthogonalises only every `reorth_period` steps. That equals 'full' on short
    warm-started solves, but the long local solves (100 or more iterations) that occur at large
    bond dimension near a critical point lose orthogonality: a spurious copy of an eigenpair
    forms and DMRG stalls above the ground-state energy. With ``stop='residual'`` this raises;
    with ``stop='energy'``, whose early exits can hide the problem, it warns. The check is made
    even for solvers that ignore `reorth` (thick restart), so an explicit request is never
    silently dropped.
    """
    if reorth != "periodic":
        return
    if stop == "residual":
        raise ValueError(
            "reorth='periodic' with stop='residual' is not supported: periodic "
            "reorthogonalisation loses basis orthogonality on long (100+ iteration) local "
            "solves, which occur at large bond dimension near a critical point, and the "
            "residual rule then iterates against a tridiagonal that no longer describes the "
            "basis. Use reorth='full', which costs little since most solves take a few "
            "iterations.")
    warnings.warn(
        "reorth='periodic' loses orthogonality on long local solves (large bond dimension "
        "near a critical point) and can stall DMRG above the ground state. Use reorth='full'.",
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
    """exp(tau H) v from a FIXED number `m` of Krylov steps, with no host branch. The same
    mathematics as `lanczos_expm`, in a control flow suited to GPUs.

    The adaptive solver reads scalars back to the host at every step (to test for breakdown
    and to evaluate its error estimate with a host `scipy.linalg.expm`), and each read waits
    for the device queue to drain. Here:

      1. The Lanczos coefficients stay 0-d device tensors throughout, and the tridiagonal is
         assembled from them on the device (`ops.stack`, `ops.tridiag`); nothing is read back
         during the loop.
      2. The small exponential is taken once, at the end, on the host by default
         (`expm_where='host'`: one transfer of an m x m matrix). 'device' uses
         `torch.matrix_exp`; 'eigh' and 'taylor' are free of host branches, so they can run
         inside a CUDA graph.
      3. Reorthogonalisation is two batched classical Gram-Schmidt passes (CGS2) against a
         preallocated basis, a few kernel launches per step instead of two per basis vector.
         Two passes make CGS as stable as modified Gram-Schmidt (Giraud, Langou and
         Rozloznik).

    Breakdown needs no branch. The coefficient of basis vector j in the result is
    ``expm(tau T)[j, 0]``, bounded by beta_1 ... beta_j / (j-1)!. If the space breaks down at
    step p, beta_p is at rounding level and every later vector contributes at rounding level,
    whatever direction it points in; `ops.guard_denom` only prevents 0/0.

    `m` is capped at the vector length. `tnkit.tdvp` chooses it from the Krylov dimensions the
    adaptive solver needs in a first sweep.

    `reorth`: 'cgs2' (default, batched matmuls), 'cgs2e' (the same written with `einsum`) or
    'mgs' (the sequential modified Gram-Schmidt loop, for comparison and as a fallback for
    large vectors).
    """
    n = 1
    for s in v.shape:
        n *= int(s)
    m = max(1, min(int(m), n))
    shape = v.shape

    # Two preallocated buffers: the basis and its conjugate. The conjugate costs one more
    # (m, n) buffer and avoids a (k, n) conj copy at every step; both einsum operands are then
    # contiguous, which keeps the call on the fast matmul path.
    cgs = reorth in ("cgs2", "cgs2e")
    use_mm = (reorth == "cgs2")                      # 'cgs2e' uses einsum
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
        # CGS2 against the live basis, as views into the buffers (restacking the basis at every
        # step would copy it). The first projection coefficient against V_k is the Lanczos
        # alpha, so it is read from that pass rather than from a separate vdot.
        Vlk = Vb[:k + 1]
        if cgs:
            Vck = Vc[:k + 1]
            # `mv`/`vm` are plain matmuls; `torch.einsum` parses its equation string on every
            # call.
            mv_, vm_ = ((ops.mv, ops.vm) if use_mm else
                        (lambda A, x: ops.einsum("jn,n->j", A, x),
                         lambda cc, A: ops.einsum("j,jn->n", cc, A)))
            c = mv_(Vck, w)
            alphas.append(c[k].real)                 # 0-d, DEVICE
            w = w - vm_(c, Vlk)
            c = mv_(Vck, w)
            w = w - vm_(c, Vlk)
        else:                                        # 'mgs': the sequential loop
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
        # Routes for CUDA graphs: `torch.matrix_exp` reads a norm back to the host to choose its
        # Pade degree, so it cannot be captured. Both routes below have no host branch; see
        # `backend.expm_sym_tridiag_eigh`.
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
        # One synchronisation per solve, on an m x m matrix. On a small matrix
        # `torch.matrix_exp` is a chain of tiny kernels, slower than this transfer; it is
        # available as `expm_where='device'` for larger m.
        E = ops.expm_host(tau * T)
        c = beta0 * ops.array(E[:, 0])
    else:
        c = beta0 * ops.expm(tau * T)[:, 0]
    out = ops.vm(c, Vb).reshape(shape)
    if info is not None:
        # The Krylov dimension is a shape and free to report. The other diagnostics are device
        # scalars, and reading one back would add a synchronisation, so they are NaN
        # (`_KrylovMeter` skips NaN). The adaptive step-size controller needs `est_ref`, so
        # `tdvp*_evolve` refuses that combination with this path.
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
        if adaptive is None:                         # TNKIT_EXPM_ADAPTIVE=0 turns it off
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
        # Adaptive stop on the a-posteriori Krylov estimate of Saad (1992), the first neglected
        # term: |beta_j| |tau| |exp(tau T)[j-1, 0]| ||v||. It over-estimates the error, so
        # stopping on it is safe; at small time steps it stops well below the cap `m`.
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
    if isinstance(v, BlockTensor) and block_lanczos.enabled():
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

    Used by the residual stopping rule, which needs the last component of the ground Ritz
    vector at every step. The host work on a k <= max_iter tridiagonal is small beside one
    matvec.
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
        What `tol` measures.

          residual (default): the Ritz residual ``norm(H x - e x) = abs(beta * y[-1])``,
              the convergence of the eigenvector. DMRG uses the vector: it goes into the
              truncation and into the MPS.
          energy: ``abs(e - e_prev) < tol``, the convergence of the Rayleigh quotient.

        The Rayleigh quotient is quadratically accurate in the eigenvector error, so the
        energy rule returns a vector converged only to about ``sqrt(tol)``: at tol = 1e-10
        the residual is about 1e-5 under "energy" and about 1e-11 under "residual". Nothing in
        the output shows the difference, since the energy looks settled either way. `dmrg`
        exposes this as `eig_stop`.
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

    # reorth="periodic": see `check_reorth_periodic`, shared by every statics solver.
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
            # stop="energy": the eigenvalue only (cheaper per step; see `stop`)
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
        # Stopping rule (see `stop`): the Ritz residual abs(beta * y_min[-1]), or the change of
        # the Rayleigh quotient.
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

    Full reorthogonalisation stores the whole Krylov basis. Near a critical point a local
    solve can take 100 or more iterations, so at large bond dimension the basis and the
    O(k²) reorthogonalisation grow large. Thick restart caps the basis at ``m`` vectors: run
    Lanczos to dimension ``m``, keep the ``k`` lowest Ritz vectors plus the residual
    direction, and continue. Orthogonality is kept exactly against the ≤ m stored vectors.

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
        # Full reorthogonalisation within the window (≤ m+1 vectors).
        #   "cgs2": two batched classical Gram-Schmidt passes, each two GEMVs against the live
        #     buffer instead of nV sequential vdot + axpy pairs. Two passes make CGS as stable
        #     as MGS (Giraud, Langou and Rozloznik).
        #   "cgs2v2": the same, conjugating w instead of the buffer.
        #   "mgs": the sequential modified Gram-Schmidt loop.
        if reorth_alg in ("cgs2", "cgs2v2"):
            Vlive = Vb[:nV]
            if reorth_alg == "cgs2v2":
                # <u, w> = conj(u . conj(w)): only w and the coefficients are conjugated, not
                # the (nV, n) buffer.
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
            # New basis: kk Ritz vectors U = Yk^T @ Vb[:cur], then the carried residual v_new,
            # in one (kk x cur)(cur x n) GEMM. It goes through a temporary and back into the
            # buffer, so the peak is m + 1 + kk rows.
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

    Lanczos builds its subspace iteratively, so a matvec with low-precision (fp32/TF32) GEMMs
    can still steer the Krylov space to the right invariant subspace, unlike a one-shot
    randomized sketch, whose error stays. Most iterations therefore use the cheap matvec, and a
    short double-precision tail certifies the answer:

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
    """Lowest eigenpair from `scipy.sparse.linalg.eigsh` on the host, around the caller's
    device `matvec`: each matvec copies its input to the device and its result back.

    `eig="scipy-host"` in `dmrg` differs from `eig="lanczos"` only in where the Krylov loop
    runs and in those copies; the sweep, environments, truncation, precision and device are
    the same, so it isolates the cost of a host eigensolver. It is a reference, not a
    solver to run by default.

    `transfers`: optional dict; accumulates 'n' (matvec count) and 'bytes' (host-device
    traffic).
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    v0_np = ops.to_numpy(v0).ravel()
    n = int(v0_np.size)
    dtype = v0_np.dtype

    def host_matvec(x_host):
        x_host = np.asarray(x_host, dtype=dtype).ravel()
        x_dev = ops.array(x_host)                 # host -> device
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
