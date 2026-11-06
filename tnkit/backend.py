"""Backend adaptor so the same MPS/DMRG code runs on numpy or torch (CPU, CUDA,
or Apple MPS). The eigensolver needs only vdot/norm/zeros_like; the DMRG kernel and
truncations additionally need einsum/svd/qr/eigh/diag. Each backend implements the
same small interface, so `dmrg(..., backend="torch-cuda")` runs unchanged on GPU.

Design: production target is NVIDIA, where torch gives native cuSOLVER svd/qr/eigh
in double precision. Apple MPS is single precision and lacks native svd/qr/eigh
(they fall back to CPU); CUDA has all of them. Numpy stays the default (full
back-compat for the CPU experiments).
"""
from __future__ import annotations
import os
import numpy as np


# Compiled einsum plans for the numpy backend. An MPS sweep issues the same few contractions
# at the same few shapes many thousands of times, and at small bond dimension most of
# `np.einsum(..., optimize=True)`'s time is per-call Python: parsing, path search and
# array-function dispatch. A two-operand contraction with no trace and no batch index is one
# transpose/reshape/GEMM sequence fixed by (subscripts, shapes, dtypes), so it is worked out
# once per key and replayed with method calls.
#
# A plan is installed only if it reproduces `np.einsum` element for element on the caller's
# operands and on two random samples of the same shapes and dtypes; otherwise the key falls
# back to `np.einsum` for good. Each contraction is therefore bit-identical to `np.einsum`.
# Downstream results can still differ at round-off (a plan returns a C-contiguous buffer, and
# QR/SVD results depend on the buffer they are handed), so evolved tensors are not guaranteed
# bit-identical to a run without plans. Installing costs several `np.einsum` calls, so
# operands above `_EINSUM_PLAN_MAX_SIZE` elements (where the GEMM dwarfs the per-call
# overhead and a plan gains nothing) are never planned. TNKIT_EINSUM_PLAN=0 disables plans.

_EINSUM_PLANS = {}
_EINSUM_PLAN_CAP = 4096
_EINSUM_PLAN_MAX_SIZE = 2 ** 15
_EINSUM_FALLBACK = object()
_EINSUM_PLAN_ENABLED = os.environ.get("TNKIT_EINSUM_PLAN", "1") not in ("0", "false", "off")


def _prod(xs):
    n = 1
    for x in xs:
        n *= int(x)
    return n


def _build_einsum_plan(sub, shapes, swap=False):
    """A replayable plan for `sub` at `shapes`, or None when it is not a single permute or
    a single GEMM (implicit output, ellipsis, trace, partial sum, batch index, outer product,
    three or more operands). `swap` puts the second operand on the left of the GEMM."""
    if "->" not in sub or "." in sub:
        return None
    lhs, out = sub.split("->")
    terms = lhs.split(",")
    if len(terms) != len(shapes) or len(set(out)) != len(out):
        return None
    if any(len(set(t)) != len(t) for t in terms):
        return None
    if len(terms) == 1:
        t = terms[0]
        if set(t) != set(out):
            return None
        perm = tuple(t.index(c) for c in out)
        return ("copy",) if perm == tuple(range(len(perm))) else ("permute", perm)
    if len(terms) != 2:
        return None
    ta, tb = (terms[1], terms[0]) if swap else (terms[0], terms[1])
    sa, sb = (shapes[1], shapes[0]) if swap else (shapes[0], shapes[1])
    shared = set(ta) & set(tb)
    summed = [c for c in ta if c in shared and c not in out]
    if not summed or shared - set(summed):
        return None
    free_a = [c for c in ta if c not in summed]
    free_b = [c for c in tb if c not in summed]
    if sorted(free_a + free_b) != sorted(out):
        return None
    ax_a, ax_b = [ta.index(c) for c in summed], [tb.index(c) for c in summed]
    keep_a, keep_b = [ta.index(c) for c in free_a], [tb.index(c) for c in free_b]
    m, k, n = (_prod(sa[i] for i in keep_a), _prod(sa[i] for i in ax_a),
               _prod(sb[i] for i in keep_b))
    mid_shape = tuple(int(sa[i]) for i in keep_a) + tuple(int(sb[i]) for i in keep_b)
    mid = free_a + free_b
    perm_out = tuple(mid.index(c) for c in out)
    if perm_out == tuple(range(len(perm_out))):
        perm_out = None
    return ("gemm", tuple(keep_a + ax_a), (m, k), tuple(ax_b + keep_b), (k, n), mid_shape,
            perm_out, swap)


def _run_einsum_plan(plan, ops):
    if plan[0] == "copy":
        return ops[0].copy()
    if plan[0] == "permute":
        return ops[0].transpose(plan[1]).copy()
    _, perm_a, sh_a, perm_b, sh_b, mid_shape, perm_out, swap = plan
    a, b = (ops[1], ops[0]) if swap else (ops[0], ops[1])
    res = a.transpose(perm_a).reshape(sh_a).dot(b.transpose(perm_b).reshape(sh_b))
    res = res.reshape(mid_shape)
    return res.transpose(perm_out).copy() if perm_out is not None else res


def _plan_agrees(plan, sub, ops, ref):
    """True iff `plan` equals `np.einsum` exactly on `ops` and on two random samples."""
    try:
        got = _run_einsum_plan(plan, ops)
        if not (got.dtype == ref.dtype and got.shape == ref.shape
                and np.array_equal(got, ref)):
            return False
        rng = np.random.default_rng(0)
        for _ in range(2):
            probe = []
            for o in ops:
                x = rng.standard_normal(o.shape)
                if np.iscomplexobj(o):
                    x = x + 1j * rng.standard_normal(o.shape)
                probe.append(np.ascontiguousarray(x, dtype=o.dtype))
            if not np.array_equal(_run_einsum_plan(plan, probe),
                                  np.einsum(sub, *probe, optimize=True)):
                return False
    except (ValueError, TypeError):
        return False
    return True


def einsum_planned(sub, ops):
    """`np.einsum(sub, *ops, optimize=True)`, replayed from a compiled plan where one exists."""
    if not _EINSUM_PLAN_ENABLED:
        return np.einsum(sub, *ops, optimize=True)
    try:
        shapes = tuple(o.shape for o in ops)
        key = (sub, shapes, tuple(o.dtype for o in ops))
    except AttributeError:                      # a python scalar or list operand
        return np.einsum(sub, *ops, optimize=True)
    plan = _EINSUM_PLANS.get(key)
    if plan is None:
        ref = np.einsum(sub, *ops, optimize=True)
        out = ref
        if max(o.size for o in ops) <= _EINSUM_PLAN_MAX_SIZE:
            # numpy's two-operand path puts the second operand on the left of its GEMM, and the
            # two orders can round differently, so that order is tried first
            for swap in (True, False):
                cand = _build_einsum_plan(sub, shapes, swap)
                if cand is None:
                    continue
                if _plan_agrees(cand, sub, ops, ref):
                    plan = cand
                    out = _run_einsum_plan(cand, ops)   # same buffer layout as later warm calls
                    break
                if cand[0] != "gemm":
                    break
        if len(_EINSUM_PLANS) >= _EINSUM_PLAN_CAP:
            _EINSUM_PLANS.clear()
        _EINSUM_PLANS[key] = plan if plan is not None else _EINSUM_FALLBACK
        return out
    if plan is _EINSUM_FALLBACK:
        return np.einsum(sub, *ops, optimize=True)
    return _run_einsum_plan(plan, ops)


def _negligible_imag(a, tol=1e-12):
    """(is_negligible, max|imag|) for array-like `a`, at the tolerance the real backend's
    own runtime guard (`NumpyBackend.array` / `TorchBackend._check_real`) uses to accept or
    reject data: |imag| <= tol * max(1, |real|). Real (non-complex) input is trivially
    negligible. One function so the auto-dtype default (`get_backend(..., mpo=...)`) and the
    runtime guard can never disagree about what counts as real."""
    a = np.asarray(a)
    if not np.iscomplexobj(a):
        return True, 0.0
    m = float(np.max(np.abs(a.imag))) if a.size else 0.0
    r = float(np.max(np.abs(a.real))) if a.size else 1.0
    return m <= tol * max(1.0, r), m


def is_real_problem(*tensor_groups, tol=1e-12):
    """True when every tensor in the given groups (an MPO, an initial MPS, ...) is real to
    `_negligible_imag`'s tolerance -- the SAME rule the real backend's runtime guard enforces
    on every array it is handed. `None` groups are skipped, so callers can pass an optional
    ``init_mps=None`` straight through.

    This is what resolves ``backend='auto'`` in the statics front doors (``dmrg``,
    ``dmrg_excited``): pick the real backend when the problem is real, the complex backend
    otherwise -- so the production default never silently drops physics for a genuinely
    complex model, and never hard-fails a real one. If this check is ever WRONG about a
    tensor supplied later (e.g. a complex `init_mps` passed after this call inspected only
    the MPO), the same runtime guard that this function reuses will catch it, loudly, the
    moment that tensor reaches ``be.array()`` -- never silently.
    """
    for group in tensor_groups:
        if group is None:
            continue
        for t in group:
            if not _negligible_imag(t, tol)[0]:
                return False
    return True


def get_backend(spec="numpy", *, mpo=None, init_mps=None, real_tol=1e-12):
    """spec in {'numpy', 'torch-cpu', 'torch-cuda', 'torch-mps'}, a Backend, or 'auto'.

    A ``-real`` suffix ('numpy-real', 'torch-cuda-real') selects the REAL dtype (float64).
    The DMRG statics path is real-safe -- the TFIM MPO is real, the ground state of a
    real-symmetric H can be chosen real, and every complex literal in dmrg/linalg is an
    arbitrary seed rather than physics. Real is 2x the memory and 2-4x the dense flops.
    Dynamics (tdvp/tebd) and the non-Hermitian transfer spectrum (krylov/vumps) must keep a
    complex backend; a real ground state promotes to complex with one astype at the start of
    time evolution.

    ``spec='auto'`` -- the production statics default -- resolves to
    ``'numpy-real'`` when `is_real_problem(mpo, init_mps)` holds and to ``'numpy'`` otherwise,
    so it engages the cheaper real dtype exactly where it is valid and falls back to complex
    for a genuinely complex MPO/initial state (e.g. one built with SY) rather than either
    silently truncating it or hard-failing a call that used to work. Requires `mpo=`; raises
    if omitted, since there is nothing to inspect. Torch specs are unaffected -- 'auto' only
    ever resolves to a numpy spec, matching the CPU-default front doors it backs.
    """
    if not isinstance(spec, str):
        return spec                                   # already a backend object
    if spec == "auto":
        if mpo is None:
            raise ValueError("get_backend(spec='auto') needs mpo=... to inspect")
        spec = "numpy-real" if is_real_problem(mpo, init_mps, tol=real_tol) else "numpy"
    real = spec.endswith("-real")
    if real:
        spec = spec[: -len("-real")]
    if spec == "numpy":
        return NumpyBackend(dtype=np.float64 if real else np.complex128)
    if spec.startswith("torch-"):
        import torch
        dev = spec.split("-", 1)[1]
        dt = None
        if real:
            dt = torch.float32 if dev == "mps" else torch.float64
        return TorchBackend(device=dev, dtype=dt)
    raise ValueError(f"unknown backend {spec!r}")


class NumpyBackend:
    name = "numpy"
    is_torch = False

    def __init__(self, dtype=np.complex128):
        self.dtype = dtype
        self.device = "cpu"
        self.is_complex = bool(np.issubdtype(np.dtype(dtype), np.complexfloating))
        self.name = "numpy" if self.is_complex else "numpy-real"

    def randn(self, rng, shape):
        """Standard normal in the backend dtype (complex only when the backend is complex)."""
        r = rng.standard_normal(shape)
        if self.is_complex:
            r = r + 1j * rng.standard_normal(shape)
        return self.array(r)

    # --- array creation / movement ---
    def array(self, x):
        if not self.is_complex:
            a = np.asarray(x)
            if np.iscomplexobj(a):
                # The statics path is real-safe, but dropping a NON-negligible imaginary part
                # would be a silent physics error. Check (the same rule `is_real_problem`
                # uses to pick this backend in the first place) rather than let numpy
                # warn-and-truncate.
                ok, m = _negligible_imag(a)
                if not ok:
                    raise TypeError(
                        f"real backend received an array with a significant imaginary part "
                        f"(max|imag|={m:.3e}). Use a complex backend for this quantity.")
                a = a.real
            return np.asarray(a, dtype=self.dtype)
        return np.asarray(x, dtype=self.dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def astype(self, x, which):
        """Cast to reduced ('low', complex64) or full ('high', complex128) precision.

        The hook for mixed-precision truncation: the randomized range finder runs
        the sketch GEMMs in 'low' and the projected core in 'high'. On numpy this
        just exercises the fp32 *numerics* (no speed-up) so the accuracy check is
        CPU-testable; the throughput win is on the GPU backend.
        """
        return x.astype(np.complex64 if which == "low" else np.complex128)

    def ones(self, shape):
        return np.ones(shape, dtype=self.dtype)

    def zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype)

    def conj(self, a):
        return np.conj(a)

    def transpose(self, x, perm):
        """Permute the axes of `x` (a view, as `np.transpose`)."""
        return np.transpose(x, perm)

    def diag(self, v):
        return np.diag(v)

    # --- reductions / scalars ---
    def vdot(self, a, b):
        return np.vdot(a, b)

    def norm(self, a):
        return np.linalg.norm(a)

    def zeros_like(self, a):
        return np.zeros_like(a)

    def to_scalar(self, x):
        return complex(x)

    # --- linear algebra ---
    def einsum(self, sub, *ops):
        return einsum_planned(sub, ops)

    # --- small dense helpers for the on-device Krylov exponential (tdvp-batch) ---
    def stack(self, xs):
        """Stack a list of 0-d/1-d backend scalars into one backend vector, no host round-trip."""
        return np.stack([np.asarray(x) for x in xs])

    def tridiag(self, d, e):
        """Symmetric tridiagonal from diagonal `d` (len j) and off-diagonal `e` (len j-1),
        in the BACKEND dtype and on the BACKEND device -- so it can be exponentiated without
        ever touching the host."""
        T = np.diag(np.asarray(d, dtype=self.dtype))
        if len(e):
            off = np.asarray(e, dtype=self.dtype)
            T = T + np.diag(off, 1) + np.diag(off, -1)
        return T

    def expm(self, A):
        """Dense matrix exponential of a small (Krylov-sized) matrix, on the backend device."""
        from scipy.linalg import expm as _e
        return _e(A)

    def expm_host(self, A):
        """Matrix exponential computed on the HOST, returned as a numpy array. One transfer for
        the whole (m x m) matrix instead of 2m scalar reads."""
        from scipy.linalg import expm as _e
        return _e(np.asarray(A))

    # --- numpy mirrors of the CUDA-graph-capturable exponentials ------------------
    # These have no performance role on numpy -- capture is a CUDA concept. They exist so
    # the SAME `expm_where` code path is exercised by the CPU test suite, which is where
    # the correctness of the Taylor route is actually checked; a GPU-only formulation would
    # be tested only on the machine that is hardest to test on.

    def expm_sym_tridiag_eigh(self, d, e, tau):
        """exp(tau*T) for real-symmetric tridiagonal T, via its eigendecomposition."""
        dd = np.asarray(d, dtype=self.rdtype if hasattr(self, "rdtype") else np.float64).real
        T = np.diag(dd)
        if e is not None and len(np.atleast_1d(e)):
            off = np.asarray(e).real.astype(dd.dtype)
            T = T + np.diag(off, 1) + np.diag(off, -1)
        lam, V = np.linalg.eigh(T)
        return (V * np.exp(tau * lam)) @ V.T

    def expm_taylor(self, A, squarings, order=18):
        """exp(A) by scaling-and-squaring with a truncated Taylor series -- GEMMs only."""
        A = np.asarray(A)
        As = A / (2.0 ** int(squarings))
        E = np.eye(A.shape[-1], dtype=A.dtype)
        term = E
        for k in range(1, int(order) + 1):
            term = (term @ As) / k
            E = E + term
        for _ in range(int(squarings)):
            E = E @ E
        return E

    def mv(self, A, x):
        """A @ x for a (j, n) matrix and length-n vector -- the same contraction as
        einsum('jn,n->j', A, x), reached without an equation string to parse."""
        return A @ x

    def vm(self, c, A):
        """c @ A for a length-j vector and (j, n) matrix -- einsum('j,jn->n', c, A)."""
        return c @ A

    def guard_denom(self, x, tiny=1e-200):
        """`x` where it exceeds `tiny`, else 1 -- a DIVISION GUARD that needs no host branch.

        The fixed-schedule Krylov loop must not read any scalar back to the host, so it cannot
        test for Lanczos breakdown. Instead it divides by this guarded value: when beta is a
        genuine breakdown the basis vector is meaningless, but its coupling into the projected
        tridiagonal is `beta` itself, so `expm(tau T)[j, 0]` suppresses it to the same order.

        A clamp, not a `where(x > tiny, x, 1)`: the latter is three kernels (compare, ones_like,
        select) and is called m+1 times per solve, which is ~30 launches per solve in a routine
        whose whole problem is launch count. `tiny = 1e-200` is chosen so that w/tiny cannot
        overflow for any w that produced a norm that small -- an exactly-zero w divides to zero
        rather than to NaN, which is the only outcome that has to be prevented.
        """
        return np.maximum(x, tiny)

    # --- leading-axis batched helpers (one call for k independent problems) --------
    #
    # numpy's linalg already broadcasts over leading axes, so `qr`/`svd`/`eigh` need no
    # separate entry point here; the three below have no stacked spelling in the base API
    # and are what the batched evolver calls instead of a Python loop over legs.

    def norm_axis(self, x, axis=-1):
        """Euclidean norm along one axis, keeping every other axis. Real dtype out."""
        return np.linalg.norm(x, axis=axis)

    def stack_axis(self, xs, axis=-1):
        """Stack a list of arrays along a NEW axis (the batched twin of `stack`)."""
        return np.stack([np.asarray(x) for x in xs], axis=axis)

    def expm_sym_tridiag_eigh_batched(self, d, e, tau):
        """exp(tau*T_n) for a STACK of real-symmetric tridiagonals, one per leg.

        `d` is (k, m) and `e` is (k, m-1) or None; the result is (k, m, m) in the backend
        dtype. Same identity as the unbatched `expm_sym_tridiag_eigh` -- T = V diag(lam) V^T
        with lam real, so exp(tau T) = V diag(exp(tau lam)) V^T -- applied to the whole stack
        in one eigensolve, which is the point: the k tridiagonals are independent, and the
        unbatched route would issue k eigensolves of a 10x10 to do nanoseconds of arithmetic.
        """
        dd = np.asarray(d).real.astype(np.float64)
        k, m = dd.shape
        T = np.zeros((k, m, m), dtype=np.float64)
        idx = np.arange(m)
        T[:, idx, idx] = dd
        if e is not None and np.asarray(e).size:
            off = np.asarray(e).real.astype(np.float64)
            j = np.arange(m - 1)
            T[:, j, j + 1] = off
            T[:, j + 1, j] = off
        lam, V = np.linalg.eigh(T)
        ph = np.exp(tau * lam).astype(self.dtype)
        return (V.astype(self.dtype) * ph[:, None, :]) @ np.swapaxes(V, -1, -2).astype(self.dtype)

    def svd(self, A):
        """Economy SVD, robust to LAPACK gesdd non-convergence (gesvd fallback)."""
        try:
            return np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            from scipy.linalg import svd as _ssvd
            return _ssvd(A, full_matrices=False, lapack_driver="gesvd")

    def svd_batched(self, A):
        """Economy SVD of a STACK of matrices. numpy's `svd` already broadcasts over leading
        axes, so this is `svd` -- the separate name exists because the torch backend needs a
        different CUDA driver when the input is stacked."""
        return self.svd(A)

    def qr(self, A):
        return np.linalg.qr(A, mode="reduced")

    def eigh(self, A):
        return np.linalg.eigh(A)

    def sync(self):
        pass


class PlainEinsumNumpyBackend(NumpyBackend):
    """`NumpyBackend` with plain `np.einsum(optimize=True)` instead of compiled plans.

    Each planned contraction has the same values as `np.einsum` but not always the same
    memory layout, and later contractions can round differently on a different layout.
    Code whose results must not depend on the plans uses this backend.
    """

    def einsum(self, sub, *ops):
        return np.einsum(sub, *ops, optimize=True)


class TorchBackend:
    name = "torch"
    is_torch = True

    def __init__(self, device="cpu", dtype=None):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if dtype is None:
            # MPS is single precision only; CPU/CUDA get double
            dtype = torch.complex64 if self.device.type == "mps" else torch.complex128
        self.dtype = dtype
        self.is_complex = bool(torch.is_complex(torch.zeros((), dtype=dtype)))
        if self.is_complex:
            self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        else:
            self.rdtype = dtype
        self.name = f"torch-{self.device.type}" + ("" if self.is_complex else "-real")

    def randn(self, rng, shape):
        """Standard normal in the backend dtype (complex only when the backend is complex)."""
        r = rng.standard_normal(shape)
        if self.is_complex:
            r = r + 1j * rng.standard_normal(shape)
        return self.array(r)

    # --- array creation / movement ---
    def array(self, x):
        if isinstance(x, self.torch.Tensor):
            if not self.is_complex and self.torch.is_complex(x):
                self._check_real(x.imag, x.real)
                x = x.real
            return x.to(self.device, self.dtype)
        a = np.asarray(x)
        if not self.is_complex and np.iscomplexobj(a):
            self._check_real(a.imag, a.real)
            a = a.real
        return self.torch.as_tensor(a, dtype=self.dtype, device=self.device)

    @staticmethod
    def _check_real(im, re):
        """Dropping a NON-negligible imaginary part would be a silent physics error. Same
        rule as `_negligible_imag` / `is_real_problem`, applied to an already-split
        real/imag pair (the torch tensor may not support `.imag` cheaply pre-split)."""
        m = float(abs(im).max()) if getattr(im, "size", 1) else 0.0
        r = float(abs(re).max()) if getattr(re, "size", 1) else 1.0
        if m > 1e-12 * max(1.0, r):
            raise TypeError(f"real backend received an array with a significant imaginary part "
                            f"(max|imag|={m:.3e}). Use a complex backend for this quantity.")

    def to_numpy(self, x):
        # torch's conj()/imag-view results are LAZY: the conj/neg bit rides the tensor and
        # .numpy() refuses to cross the bridge ("Can't call numpy() on Tensor that has
        # conjugate bit set"). Any state that ever stored a be.conj(...) result then poisons
        # every downstream transfer. Resolve materializes the same values; cost is one copy
        # of a tensor already being copied to host.
        x = x.detach()
        if x.is_conj():
            x = x.resolve_conj()
        if x.is_neg():
            x = x.resolve_neg()
        return x.cpu().numpy()

    def astype(self, x, which):
        """Cast to reduced ('low', complex64) or full ('high', complex128) precision.

        Hook for mixed-precision truncation: the range finder's sketch GEMMs run in
        'low' (fp32/TF32 — where the L40S is ~30-100x its fp64 rate), the
        projected core in 'high' (fp64) for accuracy. On an fp32-only device (MPS)
        'high' would upcast, so the mixed path targets CPU/CUDA.
        """
        dt = self.torch.complex64 if which == "low" else self.torch.complex128
        return x.to(dt)

    def ones(self, shape):
        return self.torch.ones(shape, dtype=self.dtype, device=self.device)

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def conj(self, a):
        return self.torch.conj(a)

    def transpose(self, x, perm):
        """Permute the axes of `x` (a strided view, as `torch.permute`)."""
        return x.permute(*perm)

    def diag(self, v):
        return self.torch.diag(v.to(self.dtype) if v.dtype != self.dtype else v)

    # --- reductions / scalars ---
    def vdot(self, a, b):
        return self.torch.vdot(a.reshape(-1), b.reshape(-1))

    def norm(self, a):
        return self.torch.linalg.norm(a)

    def zeros_like(self, a):
        return self.torch.zeros_like(a)

    def to_scalar(self, x):
        return complex(x.item()) if hasattr(x, "item") else complex(x)

    # --- linear algebra (torch.einsum optimises internally; no `optimize` kwarg) ---
    def einsum(self, sub, *ops):
        return self.torch.einsum(sub, *ops)

    # --- small dense helpers for the on-device Krylov exponential (tdvp-batch) ---
    def stack(self, xs):
        """Stack a list of 0-d device scalars into one device vector. The whole point is that
        the Lanczos alphas/betas NEVER become Python floats: `.item()` on each of them is a
        device->host synchronisation, and the profile measured 91 of those per site-step."""
        return self.torch.stack([self.torch.as_tensor(x, device=self.device) if not
                                 isinstance(x, self.torch.Tensor) else x for x in xs])

    def tridiag(self, d, e):
        """Symmetric tridiagonal from device vectors `d` (len j) and `e` (len j-1), in the
        backend dtype, built with two `diag_embed`-free `torch.diag` calls -- three kernel
        launches total, against the 44 host `scipy.expm` round-trips per site-step it replaces."""
        t = self.torch
        d = d.to(self.dtype)
        T = t.diag(d)
        if e is not None and int(e.shape[0]) > 0:
            e = e.to(self.dtype)
            T = T + t.diag(e, 1) + t.diag(e, -1)
        return T

    def expm(self, A):
        """Dense matrix exponential of a small (Krylov-sized) matrix, ON DEVICE.

        `torch.matrix_exp` runs the scaling-and-squaring Pade approximant as CUDA kernels, so
        the Krylov tridiagonal is exponentiated where it was built. The replaced path formed the
        same matrix from host floats and called `scipy.linalg.expm`, which forces a sync per
        Lanczos iteration by construction."""
        return self.torch.matrix_exp(A)

    def expm_host(self, A):
        """Matrix exponential on the HOST, returned as numpy. Costs exactly ONE device->host
        synchronisation for the whole (m x m) Krylov tridiagonal -- against the 2m scalar `.item()`
        reads the adaptive solver pays per solve, and the m-1 host `expm` calls it makes."""
        from scipy.linalg import expm as _e
        return _e(A.detach().cpu().numpy())

    # --- CUDA-GRAPH-CAPTURABLE exponentials of the Krylov tridiagonal -------------
    #
    # `torch.matrix_exp` is NOT capturable, and that is MEASURED rather than assumed: it picks its Pade degree and squaring
    # count from ||A||_1 read back to the HOST, so recording it raises "operation not
    # permitted when stream is capturing". Both routes below remove that host branch, and
    # both exploit the same structural fact -- the Krylov tridiagonal T is REAL SYMMETRIC
    # (the alphas are Rayleigh quotients, the betas are norms) and tau is PURE IMAGINARY,
    # so tau*T is anti-Hermitian and exp(tau*T) is UNITARY.

    def expm_sym_tridiag_eigh(self, d, e, tau):
        """exp(tau * T) for a real-symmetric tridiagonal T, via its eigendecomposition.

        EXACT to eigensolver accuracy with NO scaling parameter to calibrate and no host
        branch: T = V diag(lam) V^T with lam real, so exp(tau T) = V diag(exp(tau lam)) V^T.
        Because T is built in the REAL dtype the eigensolver is `syevd` on an (m x m) real
        matrix -- the smallest dense problem in the whole step.

        This is the preferred capturable route when `linalg.eigh` can be captured; the
        Taylor route below is the fallback that needs only GEMMs.
        """
        t = self.torch
        T = t.diag(d.to(self.rdtype))
        if e is not None and int(e.shape[0]) > 0:
            ee = e.to(self.rdtype)
            T = T + t.diag(ee, 1) + t.diag(ee, -1)
        lam, V = t.linalg.eigh(T)
        ph = t.exp(t.as_tensor(tau, dtype=self.dtype, device=self.device) * lam.to(self.dtype))
        return (V.to(self.dtype) * ph) @ V.to(self.dtype).T

    def expm_taylor(self, A, squarings, order=18):
        """exp(A) by scaling-and-squaring with a TRUNCATED TAYLOR series -- GEMMs only.

        Two deliberate departures from Higham's scaling-and-squaring, both forced by capture:

          * the Pade approximant is replaced by a Taylor sum, because Pade needs
            `(V-U)^{-1}(V+U)`, i.e. a pivoted LU whose `info` check is a host read;
          * `squarings` is a FIXED, CALIBRATED integer rather than one derived from ||A||,
            because deriving it is exactly the host branch that makes `torch.matrix_exp`
            uncapturable. It is frozen from a warmup sweep alongside the Krylov dimension.

        Over-scaling is unusually benign here: exp(tau T) is unitary for anti-Hermitian
        tau*T, and repeatedly squaring a unitary neither grows nor amplifies error the way
        squaring a general matrix does. Cost is `order + squarings` tiny GEMMs, which is
        the kind of thing a graph replay makes free.
        """
        t = self.torch
        As = A / (2.0 ** int(squarings))
        E = t.eye(A.shape[-1], dtype=A.dtype, device=self.device)
        term = E
        for k in range(1, int(order) + 1):
            term = (term @ As) / k
            E = E + term
        for _ in range(int(squarings)):
            E = E @ E
        return E

    def mv(self, A, x):
        """A @ x for a (j, n) matrix and length-n vector -- the same contraction as
        einsum('jn,n->j', A, x), reached without an equation string to parse."""
        return A @ x

    def vm(self, c, A):
        """c @ A for a length-j vector and (j, n) matrix -- einsum('j,jn->n', c, A)."""
        return c @ A

    def guard_denom(self, x, tiny=1e-200):
        """`x` where it exceeds `tiny`, else 1 -- a DIVISION GUARD that needs no host branch.

        The fixed-schedule Krylov loop must not read any scalar back to the host, so it cannot
        test for Lanczos breakdown. Instead it divides by this guarded value: when beta is a
        genuine breakdown the basis vector is meaningless, but its coupling into the projected
        tridiagonal is `beta` itself, so `expm(tau T)[j, 0]` suppresses its contribution to the
        same order -- the breakdown handles itself, arithmetically, with no branch.
        """
        return self.torch.clamp(x, min=tiny)

    # --- leading-axis batched helpers (one call for k independent problems) --------
    #
    # `torch.linalg.qr` and `torch.linalg.eigh` already take a leading batch axis, so the
    # batched evolver calls `qr`/`eigh` unchanged. The three below do not exist in the base
    # API in a stacked spelling, and `svd` needs a different CUDA driver when stacked.

    def norm_axis(self, x, axis=-1):
        """Euclidean norm along one axis, keeping every other axis. Real dtype out."""
        return self.torch.linalg.vector_norm(x, dim=axis)

    def stack_axis(self, xs, axis=-1):
        """Stack a list of device tensors along a NEW axis (the batched twin of `stack`).

        As with `stack`, the whole point is that the per-leg Lanczos alphas and betas stay
        DEVICE tensors: `.item()` on any of them is a synchronisation, and the batched loop
        issues k times fewer of everything else precisely so that it can afford none.
        """
        return self.torch.stack(list(xs), dim=axis)

    def expm_sym_tridiag_eigh_batched(self, d, e, tau):
        """exp(tau*T_n) for a STACK of real-symmetric tridiagonals, one per leg.

        `d` is (k, m) and `e` is (k, m-1) or None; the result is (k, m, m) in the backend
        dtype. Same identity as the unbatched `expm_sym_tridiag_eigh` -- tau*T is
        anti-Hermitian for real-symmetric T and pure-imaginary tau, so the eigendecomposition
        route is exact to eigensolver accuracy with no scaling parameter and no host branch.

        Built with `diag_embed` and `diagonal` rather than `diag`, which does not stack. One
        `linalg.eigh` on a (k, m, m) real batch replaces k eigensolves of an (m, m): at the
        m ~ 10 a calibrated TDVP schedule uses, the arithmetic is nanoseconds and the launch
        is microseconds, so the count is the whole cost.
        """
        t = self.torch
        dd = d.real.to(self.rdtype) if t.is_complex(d) else d.to(self.rdtype)
        T = t.diag_embed(dd)
        if e is not None and int(e.shape[-1]) > 0:
            off = e.real.to(self.rdtype) if t.is_complex(e) else e.to(self.rdtype)
            T = T + t.diag_embed(off, offset=1) + t.diag_embed(off, offset=-1)
        lam, V = t.linalg.eigh(T)
        ph = t.exp(t.as_tensor(tau, dtype=self.dtype, device=self.device) * lam.to(self.dtype))
        Vc = V.to(self.dtype)
        return (Vc * ph.unsqueeze(-2)) @ Vc.transpose(-1, -2)

    def svd_batched(self, A):
        """Economy SVD of a STACK of matrices, (k, m, n) -> (k, m, p), (k, p), (k, p, n).

        On CUDA the driver is `gesvd`, as in `svd`. Measured on real truncation blocks (H100
        MIG slice, complex128), `gesvdj` was 1.11-1.23x slower per call on padded (6, 96, 96)
        batched-2TDVP stacks and 1.18-3.06x slower on serial blocks at chi 48-384; a batched
        2TDVP step ran 1.04x faster with `gesvd`, with the same kept ranks and energies to
        6e-14, and `gesvd`'s singular values were ~100x tighter (6e-15 against 7e-13 of
        sigma_max). cuSOLVER batches its Jacobi solver only up to 32x32, so `gesvdj` gains
        nothing from the stack at these sizes. Random matrices of the same shapes reverse the
        ordering, so benchmark on real blocks.

        TNKIT_CUDA_SVD_DRIVER_BATCHED selects another driver ('default' = torch's choice,
        i.e. `gesvdj`). `gesvdj` can stop at its sweep limit and only warn; that warning is
        treated as a failure, the stack is redone with `gesvd`, and `svd_batched_fallbacks`
        counts it.
        """
        if self.device.type != "cuda":
            return self.torch.linalg.svd(A, full_matrices=False)
        _d = os.environ.get("TNKIT_CUDA_SVD_DRIVER_BATCHED", "gesvd")
        if _d == "gesvd":
            return self.torch.linalg.svd(A, full_matrices=False, driver="gesvd")
        import warnings
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")     # otherwise repeats would be suppressed
                out = self.torch.linalg.svd(A, full_matrices=False,
                                            driver=None if _d == "default" else _d)
            if not any("converge" in str(w.message) for w in caught):
                return out
        except Exception:                                    # cuSOLVER hiccup
            pass
        self.svd_batched_fallbacks = getattr(self, "svd_batched_fallbacks", 0) + 1
        return self.torch.linalg.svd(A, full_matrices=False, driver="gesvd")

    def svd(self, A):
        # Torch's default CUDA
        # SVD driver is gesvdj, ~14-20x slower than gesvd at identical fp64 accuracy
        # (measured). Prefer gesvd as the PRIMARY path on CUDA. The env toggle
        # TNKIT_CUDA_SVD_DRIVER=default restores the old default for A/B re-profiling. CPU/MPS
        # keep driver=None (unchanged); the except-fallback below is untouched.
        driver = None
        if self.device.type == "cuda":
            _d = os.environ.get("TNKIT_CUDA_SVD_DRIVER", "gesvd")
            driver = None if _d == "default" else _d
        try:
            return self.torch.linalg.svd(A, full_matrices=False, driver=driver)
        except Exception:                                    # LAPACK/cuSOLVER gesdd hiccup
            t = self.torch
            if self.device.type == "cuda":                   # native robust driver
                return t.linalg.svd(A, full_matrices=False, driver="gesvd")
            # cpu/mps: robust gesvd via scipy, then move back to the device
            from scipy.linalg import svd as _ssvd
            U, S, Vh = _ssvd(A.detach().cpu().numpy(), full_matrices=False,
                             lapack_driver="gesvd")
            return (t.as_tensor(U, dtype=self.dtype, device=self.device),
                    t.as_tensor(S, dtype=self.rdtype, device=self.device),
                    t.as_tensor(Vh, dtype=self.dtype, device=self.device))

    def qr(self, A):
        return self.torch.linalg.qr(A, mode="reduced")

    def eigh(self, A):
        return self.torch.linalg.eigh(A)

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        elif self.device.type == "mps":
            self.torch.mps.synchronize()
