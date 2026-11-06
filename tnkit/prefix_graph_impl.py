"""CUDA-Graph capture with IDENTICAL ARITHMETIC: overshoot-and-truncate.

WHY THIS EXISTS, AND WHY `cudagraph.py` IS NOT ENOUGH.
The earlier fixed-schedule capture in `cudagraph.py` captured the 1TDVP site-step and measured
4.83x. A follow-up production run on that path was then REJECTED: not
because replay is unfaithful (it is bit-identical to its own eager twin, 0.000e+00) but
because capture, as implemented there, REQUIRED a different solver. `lanczos_expm_fixed`
is not `lanczos_expm` with the branch removed -- it is a different algorithm:

    quantity            eager `lanczos_expm`            captured `lanczos_expm_fixed`
    ----------------    ---------------------------     -----------------------------
    reorthogonalisation sequential MGS over V            two batched CGS2 passes
    alpha               vdot(V[k], w) then axpy          read out of the CGS2 projection
    schedule            adaptive, stop on Saad estimate  fixed m
    exponential         scipy Pade on the host           Taylor scaling-and-squaring, device
    final combination   sequential  out += c_j V_j       one batched  c @ V

Every one of those five rows is a different sequence of floating-point operations. Their
combined effect is a ~1e-11 perturbation per step -- and the TDVP flow's measured
Lyapunov exponent is lambda ~ 0.29/time, so 1e-11 becomes 1e-3 by t=204. The separation
crosses the 1e-6 physics requirement at t* ~ 45. THE PERTURBATION, NOT THE GRAPH, IS THE
DEFECT: only a BIT-IDENTICAL instrument can be substituted into a production run.

THE DESIGN THAT SURVIVES: OVERSHOOT AND TRUNCATE.
Lanczos is a PREFIX algorithm -- iteration j reads only iterations < j. Full-reorthogonalised
MGS preserves this (the reorth pass at step k runs over V[0..k], which is already fixed).
So: capture a graph that always runs `m_max` iterations of EAGER'S OWN LOOP, then apply
eager's stopping rule AFTER replay, on the host, from the residual scalars the graph already
computed, and form the answer from the leading `m_eff` block with EAGER'S OWN CODE. The
first m_eff Krylov vectors are then not "close to" eager's -- they are the same doubles,
because they are the output of the same kernels in the same order on the same inputs.

WHAT MAKES THE TRANSCRIPTION LEGITIMATE. `_lanczos_prefix_device` below is
`eigensolver.lanczos_expm` with exactly two edits, and no others:

  1. every `ops.to_scalar(...)` is DELETED. `to_scalar` is `.item()`, a device->host read,
     which is the one thing a capture cannot contain. What it returns is the same IEEE double
     the 0-d device tensor already holds, so `w - alpha_host * V[k]` and
     `w - alpha_device * V[k]` are the same multiply-add on the same operands.
  2. the two divisions by a Lanczos norm are wrapped in `ops.guard_denom` (= clamp(x, 1e-200)),
     because eager avoids 0/0 with `if beta < tol: break` -- a host branch. For any beta above
     1e-200 the clamp is the identity and the division is unchanged; below it, eager would have
     stopped and the vector is discarded by the truncation anyway.

Everything else -- the operand order, the `- beta*v_prev` three-term recurrence, the explicit
alpha subtraction FOLLOWED BY a full MGS pass over the whole basis (which therefore projects
against V[k] twice, as eager does), the normalisation -- is copied verbatim.

WHAT IS DELIBERATELY *NOT* MOVED ON-DEVICE. The small-matrix exponential stays on the host,
in `scipy.linalg.expm`, called on the same m_eff x m_eff float64 tridiagonal eager builds.
`torch.matrix_exp` is not capturable anyway, but that is not the reason: a device exponential
ROUNDS DIFFERENTLY, and measurement showed that a differently-rounded anything is a seed for
e^{lambda t}. The launch saving being bought here is in the Krylov loop, which is where the
~602 dispatches per site-step actually are; the exponential is one 8x8 call.

THE OVERSHOOT IS NOW ALMOST FREE, AND THAT IS THE WHOLE TRADE. Eagerly, an over-served
Krylov iteration costs a DISPATCH (~7 us of gap), which is why an earlier attempt at a frozen
schedule was rejected for measuring 0.92x (a slowdown). Inside a graph the dispatch is paid once
for the whole recorded sequence, so an over-served iteration costs only its execution time.
`m_max` is chosen from the MEASURED demand distribution (mean 6.87 / max 8 at sigma=10;
mean 6.07 / max 7 at sigma=30) plus margin, and the cost of that margin is reported as
`overshoot` in the stats rather than assumed small.

THE ESCAPE HATCH IS EXACT, NOT APPROXIMATE. If the host stopping rule concludes that eager
would have needed MORE than m_max iterations, the graph's prefix is insufficient and the
site-step is re-run entirely on the eager path. Nothing is truncated early and nothing is
approximated; the step simply costs eager's price. `stats['short']` counts it, and a nonzero
count is a signal to raise m_max, never something to tolerate silently.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm as _host_expm

from ._dmrg import apply_one_site, apply_zero_site, update_left_env, update_right_env
from .cudagraph import GraphContextLost, GraphFaults, _capture  # noqa: F401


# --------------------------------------------------------------------------- the device loop
def _lanczos_prefix_device(matvec, v, ops, m_max):
    """`eigensolver.lanczos_expm`'s Krylov loop, run for a FIXED m_max iterations with no
    host read and no branch. See the module docstring for the two permitted edits.

    Returns (V, scal) where V is a python list of m_max device basis vectors and `scal` is a
    single real device vector of length 2*m_max + 1 laid out as

        [ alpha_0 .. alpha_{m_max-1} | beta_0 .. beta_{m_max-1} | beta0 ]

    so the host needs exactly ONE device->host transfer per solve (against the ~2m the
    adaptive path pays). `beta_k` is the norm computed at iteration k, i.e. eager's `beta`
    at the moment it decides whether to continue -- eager stores it as `betas[k]` only if it
    does continue, and the host-side selector reproduces that bookkeeping.
    """
    beta0 = ops.norm(v)                              # 0-d, DEVICE. eager: to_scalar(...).real
    V = [v / ops.guard_denom(beta0)]
    alphas, betas = [], []
    beta = 0.0                                       # eager's initialiser, a python float
    v_prev = ops.zeros_like(v)
    for k in range(m_max):
        w = matvec(V[-1]) - beta * v_prev
        alpha = ops.vdot(V[-1], w).real              # eager: to_scalar(...).real
        alphas.append(alpha)
        w = w - alpha * V[-1]
        for u in V:                                  # full reorthogonalisation, eager's order
            w = w - ops.vdot(u, w) * u
        beta = ops.norm(w)                           # eager: to_scalar(...).real
        betas.append(beta)
        v_prev = V[-1]
        if k < m_max - 1:
            V.append(w / ops.guard_denom(beta))
    scal = ops.stack(alphas + betas + [beta0])
    return V, scal


# --------------------------------------------------------------------------- the device loop, CGS2
def _lanczos_prefix_device_cgs2(matvec, v, ops, m_max, Vb):
    """`eigensolver.lanczos_expm_fixed`'s CGS2 branch, run for a FIXED m_max iterations with
    no host read, transcribed to report the SAME `scal` layout `_lanczos_prefix_device` does
    (so `select_prefix` needs no changes) -- but writing the basis into the CALLER-OWNED
    buffer `Vb` (m_max, n) instead of allocating it locally, so the basis survives past this
    capture and a later graph can read it directly.

    NOT BIT-IDENTICAL TO EAGER. Where `_lanczos_prefix_device` reproduces `lanczos_expm`'s
    sequential MGS reorthogonalisation operation-for-operation, this reorthogonalises the
    live basis in two BATCHED matmul passes (CGS2) -- the same algorithm
    `cudagraph.py`/`lanczos_expm_fixed` already use, and the reduction order therefore differs
    from eager's at the level measured for the fixed-schedule graph (median 1.2e-14, max
    4.0e-14 on the exponential alone; ~1e-11 relative on a full site-step). What this class
    keeps from the prefix graph that the fixed-schedule graph did not have is the ADAPTIVE
    part: the Krylov dimension is still eager's own, replayed from the same scalars this loop
    reports -- only the reorthogonalisation ARITHMETIC changes; the stopping rule itself
    (`select_prefix`) is unchanged CODE, but it is fed THIS loop's own, differently-rounded
    alphas/betas, so its decision is not provably identical to eager's right at a `tol`
    decision boundary -- see `select_prefix`'s docstring for what is and isn't guaranteed
    there. Measured on an H100 MIG slice: this
    reorth option cuts the graphed site-step from 6.86 to 4.45 ms at chi=48 (m_max 12 -> 10),
    and cgs2's alphas/betas deviate from mgs's by ~3e-15 relative, worst case, over 80
    real-TDVP-block comparisons.
    """
    n = 1
    for s in v.shape:
        n *= int(s)
    v = v.reshape(-1)
    beta0 = ops.norm(v)
    v0 = v / ops.guard_denom(beta0)
    Vb[0] = v0
    Vc = ops.zeros((m_max, n))
    Vc[0] = ops.conj(v0)
    alphas, betas = [], []
    for k in range(m_max):
        w = matvec(Vb[k])
        if k > 0:
            w = w - betas[k - 1] * Vb[k - 1]
        Vlk, Vck = Vb[:k + 1], Vc[:k + 1]
        c = ops.mv(Vck, w)
        alphas.append(c[k].real)
        w = w - ops.vm(c, Vlk)
        c = ops.mv(Vck, w)
        w = w - ops.vm(c, Vlk)
        b = ops.norm(w)
        betas.append(b)
        if k < m_max - 1:
            vn = w / ops.guard_denom(b)
            Vb[k + 1] = vn
            Vc[k + 1] = ops.conj(vn)
    return ops.stack(alphas + betas + [beta0])


# --------------------------------------------------------------------------- the host selector
def select_prefix(scal_host, tau, m_max, tol=1e-12, adaptive=True, est_ref_j=2,
                  info=None, m_cap=30):
    """Replay `eigensolver.lanczos_expm`'s STOPPING RULE on the host, from the scalars the
    graph already computed, and return eager's coefficient vector.

    Returns (m_eff, c) with `c` a length-m_eff host complex128 vector, or (None, None) if
    eager would have run past m_max -- in which case the caller MUST fall back to the eager
    site-step, because the captured prefix does not contain the answer.

    This is a transcription, not a re-derivation. Line for line it mirrors the adaptive
    solver: the same Saad first-neglected-term estimator, at the same j, against the same
    `tol`, built from the same float64 alphas/betas, exponentiated by the same
    `scipy.linalg.expm`. The only thing it does NOT do is generate the basis -- the graph did.

    THE GUARANTEE IS ON THE CODE PATH, NOT ON EVERY OUTPUT. For `reorth='mgs'` the
    alphas/betas fed in are bit-identical to eager's, so this function reproduces eager's
    exact `m_eff` decision, always. For `reorth='cgs2'` the alphas/betas come from a
    different reduction order (`_lanczos_prefix_device_cgs2`) and differ from eager's true
    values at the ~1e-15 relative level everywhere -- so on most inputs the `est < tol`
    comparison lands nowhere near `tol` and the decision is unaffected, but for an input
    whose true `est` sits within that same ~1e-15 relative band of `tol`, `cgs2` can pick an
    `m_eff` one Krylov dimension off from eager's true answer (see the boundary-case test in
    `tests/test_prefix_cgs2_identity.py`, which constructs one such input and confirms the
    effect is bounded to exactly that one dimension, and does not occur away from a
    boundary). This is the same knob-name-collision as
    `eigensolver.lanczos_expm_fixed`'s unrelated `reorth="cgs2"` default (a different,
    `fast_expm`-only mechanism) -- unrelated code, same keyword, opposite default; that
    default is untouched by this module.
    """
    alphas_all = scal_host[:m_max]
    betas_all = scal_host[m_max:2 * m_max]
    beta0 = float(scal_host[2 * m_max])

    alphas, betas = [], []
    est = float("inf")
    est_ref, est_ref_at, beta1 = float("nan"), 0, float("nan")
    stopped = False
    for k in range(m_max):
        alphas.append(float(alphas_all[k]))
        beta = float(betas_all[k])
        if k == 0:
            beta1 = beta
        if beta < tol:                               # eager: exact breakdown, break BEFORE append
            if info is not None and est_ref_at == 0:
                est_ref, est_ref_at = 0.0, k + 1
            stopped = True
            break
        jj = len(alphas)
        want_ref = info is not None and est_ref_at == 0 and jj >= est_ref_j
        if (adaptive and jj >= 2) or want_ref:
            Tt = np.diag(np.asarray(alphas, dtype=np.float64))
            if jj > 1:
                offt = np.asarray(betas[: jj - 1], dtype=np.float64)
                Tt += np.diag(offt, 1) + np.diag(offt, -1)
            Et = _host_expm(tau * Tt)
            est = abs(beta) * abs(tau) * abs(Et[jj - 1, 0]) * beta0
            if want_ref:
                est_ref, est_ref_at = est, jj
            if adaptive and jj >= 2 and est < tol:
                betas.append(beta)
                stopped = True
                break
        betas.append(beta)
        if len(alphas) >= m_cap:                     # eager's own loop bound, reached
            stopped = True
            break
    if not stopped:
        # eager's loop would have continued past m_max (its own cap is m_cap=30). The captured
        # prefix does not contain enough vectors; the caller re-runs the step eagerly.
        return None, None

    j = len(alphas)
    T = np.diag(np.asarray(alphas, dtype=np.float64))
    if j > 1:
        off = np.asarray(betas[: j - 1], dtype=np.float64)
        T += np.diag(off, 1) + np.diag(off, -1)
    c = beta0 * _host_expm(tau * T)[:, 0]
    if info is not None:
        info["krylov_dim"] = j
        info["err_est"] = est
        info["beta1"] = beta1
        info["est_ref"] = est_ref
        info["est_ref_j"] = est_ref_at
    return j, c


def _combine_eager(V, c, ops, shape):
    """`out = sum_j c_j V_j` in EAGER'S OWN ORDER: a sequential accumulation over python
    complex coefficients, not a batched `c @ V`.

    This is the fifth of the five rows in the module docstring's table and it is not
    cosmetic: a batched matmul reduces over j in whatever order cuBLAS chooses, and
    measurement showed that merely rewriting a contraction as `einsum` instead of `matmul`
    seeds the same e^{lambda t} amplification as a change of algorithm.
    """
    out = ops.zeros_like(V[0])
    for coeff, vj in zip(c, V):
        out = out + complex(coeff) * vj
    return out.reshape(shape)


def lanczos_expm_prefix(matvec, v, tau, ops, m_max, tol=1e-12, adaptive=True,
                        est_ref_j=2, info=None, stats=None):
    """Overshoot-and-truncate WITHOUT a graph: run the fixed-length device loop, select the
    prefix on the host, combine eagerly.

    This is the arithmetic contract on its own, separated from capture on purpose. It runs on
    ANY backend including numpy and torch-cpu, so the claim "the retained arithmetic is
    eager's arithmetic" is testable with no GPU at all (tests/test_prefix_identity.py), and
    the GPU-side test then only has to establish the thing already established twice on the
    earlier capture paths -- that replaying a captured graph reproduces its eager twin
    bit-for-bit.

    Returns exactly what `lanczos_expm` returns. If the prefix is too short it FALLS BACK to
    `lanczos_expm` itself rather than returning an approximation.
    """
    from .eigensolver import lanczos_expm
    shape = v.shape
    V, scal = _lanczos_prefix_device(matvec, v, ops, m_max)
    scal_host = np.asarray(ops.to_numpy(scal), dtype=np.float64)
    beta0 = float(scal_host[2 * m_max])
    if not np.isfinite(beta0) or beta0 < 1e-300:     # eager's degenerate-input early return
        return v
    m_eff, c = select_prefix(scal_host, tau, m_max, tol, adaptive, est_ref_j, info)
    if m_eff is None:
        if stats is not None:
            stats["short"] = stats.get("short", 0) + 1
        return lanczos_expm(matvec, v, tau, ops, tol=tol, adaptive=adaptive, info=info,
                            est_ref_j=est_ref_j)
    if stats is not None:
        stats["m_eff_hist"][m_eff] = stats["m_eff_hist"].get(m_eff, 0) + 1
        stats["m_eff_sum"] = stats.get("m_eff_sum", 0) + m_eff
        stats["solves"] = stats.get("solves", 0) + 1
    return _combine_eager(V, c, ops, shape)


def lanczos_expm_prefix_cgs2(matvec, v, tau, ops, m_max, tol=1e-12, adaptive=True,
                             est_ref_j=2, info=None, stats=None):
    """`lanczos_expm_prefix`'s CGS2 twin: same overshoot-and-truncate contract (eager's own
    stopping rule, replayed from the device loop's scalars), but the device loop is
    `_lanczos_prefix_device_cgs2` (batched reorthogonalisation) and the combination is the
    batched `c @ V` instead of `_combine_eager`'s sequential sum.

    Runs on any backend, so the SIZE of the deviation from eager -- not just its existence --
    is testable with no GPU (`tests/test_prefix_cgs2_identity.py`). This is the arithmetic
    `PrefixSiteStepGraph(reorth='cgs2')` captures; see its docstring for what changes and why.
    """
    from .eigensolver import lanczos_expm
    shape = v.shape
    n = 1
    for s in shape:
        n *= int(s)
    Vb = ops.zeros((m_max, n))
    scal = _lanczos_prefix_device_cgs2(matvec, v.reshape(-1), ops, m_max, Vb)
    scal_host = np.asarray(ops.to_numpy(scal), dtype=np.float64)
    beta0 = float(scal_host[2 * m_max])
    if not np.isfinite(beta0) or beta0 < 1e-300:     # eager's degenerate-input early return
        return v
    m_eff, c = select_prefix(scal_host, tau, m_max, tol, adaptive, est_ref_j, info)
    if m_eff is None:
        if stats is not None:
            stats["short"] = stats.get("short", 0) + 1
        return lanczos_expm(matvec, v, tau, ops, tol=tol, adaptive=adaptive, info=info,
                            est_ref_j=est_ref_j)
    if stats is not None:
        stats["m_eff_hist"][m_eff] = stats["m_eff_hist"].get(m_eff, 0) + 1
        stats["m_eff_sum"] = stats.get("m_eff_sum", 0) + m_eff
        stats["solves"] = stats.get("solves", 0) + 1
    padded = np.zeros(m_max, dtype=np.complex128)
    padded[:m_eff] = c
    c_dev = ops.array(padded)
    return ops.vm(c_dev, Vb).reshape(shape)


# --------------------------------------------------------------------------- capture
class PrefixSiteStepGraph:
    """A 1TDVP site-step captured as FOUR graphs with two host stopping decisions.

    G1   one-site Krylov loop, m_max iterations                          (captured)
    GC1  combine: Mev = sum_j c1_j V1_j                                   (captured)
    G2   QR (centre move) + env update + zero-site Krylov loop, m_max it (captured)
    GC2  combine: C0  = sum_j c0_j V0_j                                   (captured)
    then one eager einsum to absorb the bond into the neighbour.

    THE HOST DECISIONS ARE FORCED, THE COMBINE IS NOT. The truncation length m_eff is
    decided by eager's stopping rule on the host, from scalars the device loop produced --
    that decision cannot live inside a capture, so the step is cut at the two points where
    it happens. What crosses that cut is only the COEFFICIENT vector: G1/G2 write their
    m_max Krylov vectors into a static (m_max, n) buffer, the host computes eager's
    coefficients for the first m_eff of them, pads the rest with EXACT zeros, and GC1/GC2
    replay the reduction over the FULL m_max terms every time -- moving what used to be up
    to 2*m_max python-dispatched tensor ops (profiling done for both graph designs attributed a flat
    ~22us to EVERY dispatch, captured or not) into one graph replay.

    `reorth` selects the Krylov loop AND the combine reduction together, as a pair:

      'mgs'   `_lanczos_prefix_device` (eager's own sequential modified-Gram-Schmidt) plus
              a SEQUENTIAL zero-padded accumulation in GC1/GC2. BIT-IDENTICAL to eager: 0.0
              times any finite Krylov vector is exact zero and out+0 is the identity, so the
              padding terms are silent and the real ones are added in eager's own order.
              O(m_max^2) reorthogonalisation kernels, all inside the graph -- the floor that
              profiling of the fixed-schedule graph identified: capture removes their
              LAUNCH cost, not their count.
      'cgs2'  `_lanczos_prefix_device_cgs2` (two batched matmul passes, transported from
              `eigensolver.lanczos_expm_fixed`/`cudagraph.py`) plus a BATCHED matmul combine
              `c @ V`. O(m_max) kernels instead of O(m_max^2) -- this is the dispatch-cost
              fix -- but NOT bit-identical: CGS2's reduction order differs from MGS's at the
              level already measured for the fixed-schedule graph (median 1.2e-14 on the
              exponential, ~1e-11 relative on a full site-step). Measured on this design
              specifically, on an H100 MIG slice: worst case 3.0e-15 relative over 80
              real-TDVP-block comparisons, alongside a graphed site-step cost of 6.86 -> 4.45 ms
              at chi=48 (m_max 12 -> 10) -- see `tests/test_prefix_cgs2_identity.py` for the
              CPU-side bound. Also note: `select_prefix`'s stopping decision runs on THESE
              alphas/betas, not eager's -- see its docstring for the (narrower) guarantee that
              implies near a `tol` boundary.

    STATIC BUFFERS, as in `cudagraph.SiteStepGraph`: capture fixes addresses as well as
    shapes, so each call copies its inputs into the buffers the graphs were recorded
    against and clones the outputs back out. The five per-call input copies are issued as
    one `torch._foreach_copy_` batch rather than five separate `.copy_()` dispatches when
    available -- exact byte-for-byte copies either way, so this changes launch count, not
    arithmetic.
    """

    def __init__(self, be, torch, sig, dt, m_max, direction="right", pool=None,
                reorth="mgs"):
        if reorth not in ("mgs", "cgs2"):
            raise ValueError(f"reorth must be 'mgs' or 'cgs2', got {reorth!r}")
        self.be, self.torch, self.sig, self.direction = be, torch, sig, direction
        self.m_max, self.reorth = m_max, reorth
        a, s, b, c, Dl, Dr = sig
        self.a, self.s, self.b, self.c = a, s, b, c
        self.tau1 = -1j * dt / 2.0
        self.tau0 = +1j * dt / 2.0
        self._foreach_copy = getattr(torch, "_foreach_copy_", None)
        z = be.zeros
        self.in_M = z((a, s, b))
        self.in_Mn = z((b, s, c)) if direction == "right" else z((c, s, a))
        self.in_L = z((a, Dl, a))
        self.in_R = z((b, Dr, b))
        self.in_W = z((Dl, Dr, s, s))
        self.in_Mev = z((a, s, b))                   # the combined one-site result
        c0_shape = (b, b) if direction == "right" else (a, a)
        self.buf_C0 = z(c0_shape)                     # the combined zero-site result

        n1 = a * s * b
        n0 = b * b if direction == "right" else a * a
        cdtype = be.dtype
        self.buf_V1 = z((m_max, n1))
        self.buf_V0 = z((m_max, n0))
        self.buf_scal1 = torch.zeros((2 * m_max + 1,), dtype=be.rdtype, device=be.device)
        self.buf_scal0 = torch.zeros((2 * m_max + 1,), dtype=be.rdtype, device=be.device)
        self.in_c1 = torch.zeros((m_max,), dtype=cdtype, device=be.device)
        self.in_c0 = torch.zeros((m_max,), dtype=cdtype, device=be.device)

        def _krylov(mv, x0, Vb):
            if reorth == "cgs2":
                return _lanczos_prefix_device_cgs2(mv, x0, be, m_max, Vb)
            V, scal = _lanczos_prefix_device(mv, x0, be, m_max)
            for j in range(m_max):
                Vb[j].copy_(V[j])
            return scal

        def _combine(c_buf, Vb, out_buf, shape):
            if reorth == "cgs2":
                out_buf.copy_(be.vm(c_buf, Vb).reshape(shape))
                return
            out = be.zeros_like(Vb[0])
            for j in range(m_max):
                out = out + c_buf[j] * Vb[j]
            out_buf.copy_(out.reshape(shape))

        def body1():
            mv1 = lambda x: apply_one_site(x.reshape(a, s, b), self.in_L, self.in_W,
                                           self.in_R, be).reshape(-1)
            scal = _krylov(mv1, self.in_M.reshape(-1), self.buf_V1)
            self.buf_scal1.copy_(scal)

        self.g1, _ = _capture(torch, body1, pool=pool)
        self.pool = self.g1.pool()

        def bodyc1():
            _combine(self.in_c1, self.buf_V1, self.in_Mev, (a, s, b))

        self.gc1, _ = _capture(torch, bodyc1, pool=self.pool)

        if direction == "right":
            def body2():
                Q, C = be.qr(self.in_Mev.reshape(a * s, b))
                Mo = Q.reshape(a, s, b)
                Lo = update_left_env(self.in_L, Mo, self.in_W, be)
                mv0 = lambda x: apply_zero_site(x.reshape(b, b), Lo, self.in_R,
                                                be).reshape(-1)
                sc0 = _krylov(mv0, C.reshape(-1), self.buf_V0)
                self.buf_scal0.copy_(sc0)
                return Mo, Lo
        else:
            def body2():
                Qt, Ct = be.qr(be.einsum("asb->sba", self.in_Mev).reshape(s * b, a))
                Mo = be.einsum("sbr->rsb", Qt.reshape(s, b, a))
                Ro = update_right_env(self.in_R, Mo, self.in_W, be)
                C = be.einsum("ra->ar", Ct)
                mv0 = lambda x: apply_zero_site(x.reshape(a, a), self.in_L, Ro,
                                                be).reshape(-1)
                sc0 = _krylov(mv0, C.reshape(-1), self.buf_V0)
                self.buf_scal0.copy_(sc0)
                return Mo, Ro

        self.g2, self.out2 = _capture(torch, body2, pool=self.pool)

        def bodyc2():
            _combine(self.in_c0, self.buf_V0, self.buf_C0, c0_shape)

        self.gc2, _ = _capture(torch, bodyc2, pool=self.pool)

    # -- one transfer, then eager's own decision ------------------------------------------
    def _select(self, scal, tau, tol, adaptive, stats):
        scal_host = self.scal_to_host(scal)
        beta0 = float(scal_host[2 * self.m_max])
        if not np.isfinite(beta0) or beta0 < 1e-300:
            return None, None
        return select_prefix(scal_host, tau, self.m_max, tol=tol, adaptive=adaptive)

    def scal_to_host(self, scal):
        return np.asarray(self.be.to_numpy(scal), dtype=np.float64)

    def _write_c(self, buf, c, m_eff):
        """Zero-padded host->device write of eager's coefficients into a static buffer.

        Padding beyond m_eff with EXACT zeros is what makes replaying the fixed m_max-term
        accumulation in GC1/GC2 reproduce eager's variable-length sum bit-for-bit -- see
        the class docstring.
        """
        padded = np.zeros(self.m_max, dtype=np.complex128)
        padded[:m_eff] = c
        buf.copy_(self.torch.as_tensor(padded, dtype=buf.dtype, device=buf.device))

    def run(self, M, Mn, L, R, W, tol=1e-12, adaptive=True, stats=None):
        """Copy in, replay, decide on the host, replay the combine.

        Returns (M_new, env_new, Mnext_new) or None if eager's rule wanted a longer Krylov
        space than the capture provides -- in which case the caller runs the eager step and
        this call is charged as a miss, never as an approximation.
        """
        be = self.be
        dsts = [self.in_M, self.in_Mn, self.in_L, self.in_R, self.in_W]
        srcs = [M, Mn, L, R, W]
        if self._foreach_copy is not None:
            self._foreach_copy(dsts, srcs)
        else:
            for d, s in zip(dsts, srcs):
                d.copy_(s)
        self.g1.replay()
        m1, c1 = self._select(self.buf_scal1, self.tau1, tol, adaptive, stats)
        if m1 is None:
            return None
        self._write_c(self.in_c1, c1, m1)
        self.gc1.replay()
        self.g2.replay()
        Mo, env = self.out2
        m0, c0 = self._select(self.buf_scal0, self.tau0, tol, adaptive, stats)
        if m0 is None:
            return None
        self._write_c(self.in_c0, c0, m0)
        self.gc2.replay()
        if self.direction == "right":
            Mno = be.einsum("rb,bsc->rsc", self.buf_C0, self.in_Mn)
        else:
            Mno = be.einsum("csa,ar->csr", self.in_Mn, self.buf_C0)
        if stats is not None:
            for mm in (m1, m0):
                stats["m_eff_hist"][mm] = stats["m_eff_hist"].get(mm, 0) + 1
                stats["m_eff_sum"] += mm
                stats["solves"] += 1
        return Mo.clone(), env.clone(), Mno


class PrefixSiteStepGraphCache(GraphFaults):
    """Lazily captures one `PrefixSiteStepGraph` per (signature, direction) and tallies use.

    Mirrors `cudagraph.SiteStepGraphCache` -- one shared memory pool, a capacity cap, and a
    replayed/eager tally that is REPORTED with every timing number in this direction. The cap
    default is 1024, set by hand after discovering that a production run has 296 signatures
    against the 100 measured earlier and the 256 defaulted to at the time.

    `stats` additionally carries the quantity that is specific to overshoot-and-truncate:
    `m_eff_hist` (how deep eager's rule actually went, per solve) and `short` (how often it
    wanted deeper than m_max and the step had to be re-run eagerly). Their ratio to m_max IS
    the overshoot cost, and it is measured rather than assumed.

    `reorth`: 'mgs' (default, unchanged from before this fix) reproduces eager bit-for-bit at
    the O(m_max^2)-kernel cost first measured for this graph; every existing caller that does not
    pass `reorth` keeps that guarantee. 'cgs2' (opt-in) is the dispatch-cost fix -- batched
    reorthogonalisation and combine, NOT bit-identical to eager (see `PrefixSiteStepGraph`)
    -- pass it explicitly to opt in.

    Capture and replay failures fall back to eager; see `cudagraph.GraphFaults`.
    """

    def __init__(self, be, torch, dt, m_max, max_graphs=1024, tol=1e-12, adaptive=True,
                reorth="mgs", fault_budget=3):
        self.be, self.torch, self.dt, self.m_max = be, torch, dt, m_max
        self.max_graphs, self.tol, self.adaptive, self.reorth = max_graphs, tol, adaptive, reorth
        self.graphs, self.pool = {}, None
        self.stats = {"replayed": 0, "eager": 0, "captures": 0, "short": 0,
                      "solves": 0, "m_eff_sum": 0, "m_eff_hist": {}, "signatures": {}}
        self._init_faults(fault_budget)

    def _make(self, sig, direction):
        return PrefixSiteStepGraph(self.be, self.torch, sig, self.dt, self.m_max,
                                   direction, pool=self.pool, reorth=self.reorth)

    def get(self, sig, direction):
        key = (sig, direction)
        if key in self.graphs:
            return self.graphs[key]
        if key in self.refused:
            return None
        if len(self.graphs) >= self.max_graphs:
            return None                              # stay eager rather than thrash
        g = self._capture_or_none(key)
        if g is None:
            return None
        if self.pool is None:
            self.pool = g.pool
        self.graphs[key] = g
        self.stats["captures"] += 1
        self.stats["signatures"][f"{sig}|{direction}"] = 0
        return g

    def note(self, sig=None, direction=None, replayed=True):
        self.stats["replayed" if replayed else "eager"] += 1
        if replayed and sig is not None:
            k = f"{sig}|{direction}"
            self.stats["signatures"][k] = self.stats["signatures"].get(k, 0) + 1

    def summary(self):
        st = self.stats
        tot = st["replayed"] + st["eager"]
        return {
            "replayed": st["replayed"], "eager": st["eager"], "captures": st["captures"],
            "replayed_frac": (st["replayed"] / tot) if tot else 0.0,
            "short": st["short"], "solves": st["solves"],
            "m_eff_mean": (st["m_eff_sum"] / st["solves"]) if st["solves"] else 0.0,
            "m_eff_hist": dict(sorted(st["m_eff_hist"].items())),
            "m_max": self.m_max,
            "overshoot": (self.m_max * st["solves"] / st["m_eff_sum"])
                         if st["m_eff_sum"] else float("nan"),
            "capture_failed": st["capture_failed"], "replay_failed": st["replay_failed"],
            "refused": len(self.refused), "first_fault": st["first_fault"],
            "breaker_tripped": st["breaker_tripped"],
        }
