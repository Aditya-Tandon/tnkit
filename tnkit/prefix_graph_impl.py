"""CUDA-graph capture of the one-site TDVP step with the eager arithmetic: overshoot and
truncate.

The fixed-schedule capture of `cudagraph.py` needs a different solver: `lanczos_expm_fixed`
is not `lanczos_expm` without its branch, but a different sequence of floating-point
operations:

    quantity            eager `lanczos_expm`            captured `lanczos_expm_fixed`
    ----------------    ---------------------------     -----------------------------
    reorthogonalisation sequential MGS over V            two batched CGS2 passes
    alpha               vdot(V[k], w) then axpy          read out of the CGS2 projection
    schedule            adaptive, stop on Saad estimate  fixed m
    exponential         scipy Pade on the host           Taylor scaling-and-squaring, device
    final combination   sequential  out += c_j V_j       one batched  c @ V

Together these change each step at about the 1e-11 level, and TDVP dynamics can amplify a
perturbation exponentially in time (chaotic flows), so the two runs drift apart. This module
captures the eager arithmetic itself, so a graphed run reproduces the eager run bit for bit.

Overshoot and truncate. Lanczos is a prefix algorithm: iteration j reads only iterations < j,
and full reorthogonalisation by MGS keeps this (the pass at step k runs over V[0..k], already
fixed). So a graph always runs `m_max` iterations of the eager loop; the eager stopping rule
is then applied on the host to the residual scalars the graph computed, and the answer is
formed from the leading `m_eff` block with the eager code. The first m_eff Krylov vectors are
the same doubles as eager's: the same kernels in the same order on the same inputs.

`_lanczos_prefix_device` is `eigensolver.lanczos_expm` with exactly two edits:

  1. every `ops.to_scalar(...)` is removed. `to_scalar` is `.item()`, a device-to-host read,
     which a capture cannot contain. It returns the same IEEE double the 0-d device tensor
     holds, so `w - alpha_host * V[k]` and `w - alpha_device * V[k]` are the same multiply-add.
  2. the two divisions by a Lanczos norm are wrapped in `ops.guard_denom` (clamp at 1e-200),
     because eager avoids 0/0 with `if beta < tol: break`, a host branch. Above 1e-200 the
     clamp is the identity; below it, eager would have stopped and the vector is discarded by
     the truncation anyway.

Everything else (operand order, the `- beta*v_prev` recurrence, the alpha subtraction followed
by a full MGS pass over the whole basis, which projects against V[k] twice as eager does, the
normalisation) is the eager code.

The small exponential stays on the host, in `scipy.linalg.expm`, on the same m_eff x m_eff
float64 tridiagonal eager builds: a device exponential rounds differently. The launches saved
are in the Krylov loop; the exponential is one small call.

Inside a graph an extra Krylov iteration costs only its execution time, not a dispatch, so
running m_max iterations is cheap. `m_max` should sit a little above the Krylov dimensions the
adaptive solver needs; the cost of the margin is reported as `overshoot` in the stats.

If the host stopping rule finds that eager would have needed more than m_max iterations, the
site step is re-run entirely on the eager path: nothing is truncated early or approximated.
`stats['short']` counts it; a nonzero count means m_max should be raised.
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

    Not bit-identical to eager. Where `_lanczos_prefix_device` reproduces the sequential MGS
    of `lanczos_expm` operation for operation, this reorthogonalises the live basis in two
    batched matmul passes (CGS2), as `lanczos_expm_fixed` does, so the reduction order and the
    rounding differ from eager's (alphas and betas agree to about 1e-15 relative). The Krylov
    dimension is still chosen by the eager stopping rule, replayed from these scalars; since
    they are rounded differently, the decision can differ from eager's right at a `tol`
    boundary (see `select_prefix`). It needs O(m_max) kernels instead of O(m_max^2).
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

    The guarantee is on the code path. For `reorth='mgs'` the alphas and betas are eager's
    bit for bit, so this reproduces eager's `m_eff` exactly. For `reorth='cgs2'` they differ
    from eager's at about 1e-15 relative, so on an input whose estimate lies within that band
    of `tol` the chosen `m_eff` can be one Krylov dimension off eager's (and only then). This
    `reorth` is unrelated to the `reorth="cgs2"` default of `eigensolver.lanczos_expm_fixed`.
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

    This is the last row of the table in the module docstring: a batched matmul reduces over
    j in an order cuBLAS chooses, which rounds differently.
    """
    out = ops.zeros_like(V[0])
    for coeff, vj in zip(c, V):
        out = out + complex(coeff) * vj
    return out.reshape(shape)


def lanczos_expm_prefix(matvec, v, tau, ops, m_max, tol=1e-12, adaptive=True,
                        est_ref_j=2, info=None, stats=None):
    """Overshoot-and-truncate WITHOUT a graph: run the fixed-length device loop, select the
    prefix on the host, combine eagerly.

    The arithmetic of the graphed step without the capture. It runs on any backend, numpy and
    torch-cpu included, so that it matches eager can be tested without a GPU
    (tests/test_prefix_identity.py); on a GPU it remains to show that replaying a captured
    graph reproduces its eager run bit for bit.

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

    Runs on any backend, so the size of the deviation from eager can be tested without a GPU
    (`tests/test_prefix_cgs2_identity.py`). It is the arithmetic
    `PrefixSiteStepGraph(reorth='cgs2')` captures.
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
    replay the reduction over all m_max terms every time, which turns up to 2 m_max Python
    dispatches into one graph replay.

    `reorth` selects the Krylov loop AND the combine reduction together, as a pair:

      'mgs'   `_lanczos_prefix_device` (eager's sequential modified Gram-Schmidt) plus a
              sequential zero-padded accumulation in GC1/GC2. Bit-identical to eager: 0.0 times
              a finite Krylov vector is an exact zero and out + 0 is the identity, so the
              padding terms are silent and the real ones are added in eager's order. O(m_max^2)
              reorthogonalisation kernels, all inside the graph: capture removes their launch
              cost, not their number.
      'cgs2'  `_lanczos_prefix_device_cgs2` (two batched matmul passes, as in
              `eigensolver.lanczos_expm_fixed`) plus a batched combine `c @ V`. O(m_max)
              kernels, but not bit-identical: the reduction order differs from MGS's, at about
              1e-15 relative in the Lanczos scalars (`tests/test_prefix_cgs2_identity.py`).
              `select_prefix` decides on these scalars; see its docstring for the guarantee
              near a `tol` boundary.

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
    replayed/eager tally to report with any timing. The cap default, 1024 graphs, is several
    times the number of distinct shapes a long run meets.

    `stats` additionally carries the quantity that is specific to overshoot-and-truncate:
    `m_eff_hist` (how deep eager's rule actually went, per solve) and `short` (how often it
    wanted deeper than m_max and the step was re-run eagerly). Their ratio to m_max is the
    overshoot cost.

    `reorth`: 'mgs' (default) reproduces eager bit for bit at O(m_max^2) kernels per solve.
    'cgs2' (opt-in) needs fewer kernels: batched reorthogonalisation and combine, not
    bit-identical
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
