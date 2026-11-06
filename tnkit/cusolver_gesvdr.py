"""Direct ctypes binding for cuSOLVER's randomized SVD, cusolverDnXgesvdr.

WHY THIS EXISTS. gesvdr (the randomized/subspace-iteration SVD in cuSOLVER's 64-bit
Xgesvdr API) computes only the top-k singular triplets, which is exactly what a rank-chi
MPS truncation needs from a ~2chi x 2chi block. It is NOT exposed by torch.linalg.svd
(which offers only gesvd / gesvdj / gesvda) nor by cupy 14.1.1 (which binds gesvda and
gesvdj but not Xgesvdr). The C symbols cusolverDnXgesvdr and cusolverDnXgesvdr_bufferSize
DO exist in libcusolver.so.11, so this module reaches them directly.

STATUS: written against the cuSOLVER 12.x Xgesvdr signature but UNTESTED until the first
GPU run. A benchmark that uses it should validate every gesvdr result against a torch reference
SVD (singular values + reconstruction) before trusting any timing, so a signature/layout
bug surfaces as an explicit mismatch, not as silent garbage.

Reference (cuSOLVER Library, cusolverDnXgesvdr):
  cusolverDnXgesvdr_bufferSize(handle, jobu, jobv, m, n, k, p, niters,
      dataTypeA, A, lda, dataTypeSrand, S, dataTypeUrand, U, ldU,
      dataTypeVrand, V, ldV, computeType, &workDev, &workHost)
  cusolverDnXgesvdr(handle, jobu, jobv, m, n, k, p, niters,
      dataTypeA, A, lda, dataTypeSrand, S, dataTypeUrand, U, ldU,
      dataTypeVrand, V, ldV, computeType, bufDev, workDev, bufHost, workHost, d_info)
Requires m >= n (transpose the block if not). A is OVERWRITTEN. Column-major throughout.
"""
from __future__ import annotations
import ctypes
import ctypes.util

# cudaDataType enum (library_types.h)
CUDA_R_32F = 0
CUDA_R_64F = 1
CUDA_C_32F = 4
CUDA_C_64F = 5

_JOB_S = ctypes.c_char(b"S")          # economy: return k columns of U and V
_lib = None
_handle = None
_params = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    last = None
    for name in ("libcusolver.so.11", "libcusolver.so", ctypes.util.find_library("cusolver")):
        if not name:
            continue
        try:
            _lib = ctypes.CDLL(name)
            break
        except OSError as e:
            last = e
    if _lib is None:
        raise RuntimeError(f"could not load libcusolver (load CUDA module first): {last}")

    # cusolverDnCreate(handle*) / CreateParams(params*) / SetStream / Xgesvdr(_bufferSize)
    _lib.cusolverDnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _lib.cusolverDnCreate.restype = ctypes.c_int
    _lib.cusolverDnCreateParams.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _lib.cusolverDnCreateParams.restype = ctypes.c_int
    _lib.cusolverDnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.cusolverDnSetStream.restype = ctypes.c_int

    i64 = ctypes.c_int64
    vp = ctypes.c_void_p
    dt = ctypes.c_int
    sz = ctypes.c_size_t
    # NB: the 2nd arg is cusolverDnParams_t params (a required opaque handle) -- omitting it
    # was the bug behind the constant CUSOLVER_STATUS_INVALID_VALUE (status 3).
    common = [vp, vp, ctypes.c_char, ctypes.c_char, i64, i64, i64, i64, i64,  # handle,params,jobu,jobv,m,n,k,p,niters
              dt, vp, i64,          # dataTypeA, A, lda
              dt, vp,               # dataTypeS, S
              dt, vp, i64,          # dataTypeU, U, ldU
              dt, vp, i64,          # dataTypeV, V, ldV
              dt]                   # computeType
    _lib.cusolverDnXgesvdr_bufferSize.argtypes = common + [ctypes.POINTER(sz), ctypes.POINTER(sz)]
    _lib.cusolverDnXgesvdr_bufferSize.restype = ctypes.c_int
    _lib.cusolverDnXgesvdr.argtypes = common + [vp, sz, vp, sz, ctypes.POINTER(ctypes.c_int)]
    _lib.cusolverDnXgesvdr.restype = ctypes.c_int
    return _lib


def _get_handle(stream_ptr=None):
    global _handle
    lib = _load()
    if _handle is None:
        h = ctypes.c_void_p()
        st = lib.cusolverDnCreate(ctypes.byref(h))
        if st != 0:
            raise RuntimeError(f"cusolverDnCreate failed: status {st}")
        _handle = h
    if stream_ptr is not None:
        lib.cusolverDnSetStream(_handle, ctypes.c_void_p(stream_ptr))
    return _handle


def _get_params():
    """The cusolverDnParams_t opaque handle required by every Xgesvdr call (created once)."""
    global _params
    lib = _load()
    if _params is None:
        p = ctypes.c_void_p()
        st = lib.cusolverDnCreateParams(ctypes.byref(p))
        if st != 0:
            raise RuntimeError(f"cusolverDnCreateParams failed: status {st}")
        _params = p
    return _params


def gesvdr(theta, k, p=None, niters=2, compute_type=None):
    """Top-k randomized SVD of a 2-D CUDA torch tensor `theta` (complex128 or complex64).

    Returns (U, S, Vh) with U:(m,k), S:(k,) real, Vh:(k,n), so theta ~= U @ diag(S) @ Vh.
    `p` = oversampling (default min(k, min(m,n)-k)); `niters` = subspace iterations.
    `compute_type` overrides the cudaDataType used for the compute path (default = the
    complex data type; some cuSOLVER builds want the REAL type here -- the bench sweeps it).
    A copy of `theta` is made because gesvdr overwrites its input.
    """
    import torch
    assert theta.is_cuda and theta.dim() == 2, "gesvdr expects a 2-D CUDA tensor"
    cdouble = theta.dtype == torch.complex128
    if not cdouble and theta.dtype != torch.complex64:
        raise TypeError(f"gesvdr wrapper handles complex64/complex128, got {theta.dtype}")
    dtA = CUDA_C_64F if cdouble else CUDA_C_32F
    dtS = CUDA_R_64F if cdouble else CUDA_R_32F
    ct = dtA if compute_type is None else int(compute_type)
    rdtype = torch.float64 if cdouble else torch.float32

    m, n = theta.shape
    transposed = m < n                      # Xgesvdr requires m >= n
    A_src = theta.mH.contiguous() if transposed else theta   # if transposed, SVD of A^H
    if transposed:
        m, n = n, m
    k = int(min(k, n))
    if p is None:
        p = max(1, min(k, n - k))
    p = int(min(p, n - k)) if n > k else 0

    lib = _load()
    stream_ptr = torch.cuda.current_stream().cuda_stream
    handle = _get_handle(stream_ptr)
    params = _get_params()

    # column-major buffers. A_src is (m,n) logically (row-major); cuSOLVER wants column-major,
    # and a row-major (n,m) buffer IS the column-major (m,n) layout -> plain transpose-contiguous.
    A_cm = A_src.transpose(0, 1).contiguous()          # (n,m) row-major == (m,n) column-major
    U_cm = torch.empty((k, m), dtype=theta.dtype, device=theta.device)   # (m,k) col-major
    V_cm = torch.empty((k, n), dtype=theta.dtype, device=theta.device)   # (n,k) col-major
    S = torch.empty((k,), dtype=rdtype, device=theta.device)
    info = torch.zeros((1,), dtype=torch.int32, device=theta.device)

    wdev = ctypes.c_size_t(0)
    whost = ctypes.c_size_t(0)
    st = lib.cusolverDnXgesvdr_bufferSize(
        handle, params, _JOB_S, _JOB_S, m, n, k, p, int(niters),
        dtA, ctypes.c_void_p(A_cm.data_ptr()), m,
        dtS, ctypes.c_void_p(S.data_ptr()),
        dtA, ctypes.c_void_p(U_cm.data_ptr()), m,
        dtA, ctypes.c_void_p(V_cm.data_ptr()), n,
        ct, ctypes.byref(wdev), ctypes.byref(whost))
    if st != 0:
        raise RuntimeError(f"cusolverDnXgesvdr_bufferSize failed: status {st}")

    buf_dev = torch.empty((max(1, wdev.value),), dtype=torch.uint8, device=theta.device)
    buf_host = (ctypes.c_char * max(1, whost.value))()
    info_ptr = ctypes.cast(info.data_ptr(), ctypes.POINTER(ctypes.c_int))
    st = lib.cusolverDnXgesvdr(
        handle, params, _JOB_S, _JOB_S, m, n, k, p, int(niters),
        dtA, ctypes.c_void_p(A_cm.data_ptr()), m,
        dtS, ctypes.c_void_p(S.data_ptr()),
        dtA, ctypes.c_void_p(U_cm.data_ptr()), m,
        dtA, ctypes.c_void_p(V_cm.data_ptr()), n,
        ct, ctypes.c_void_p(buf_dev.data_ptr()), wdev,
        ctypes.cast(buf_host, ctypes.c_void_p), whost, info_ptr)
    if st != 0:
        raise RuntimeError(f"cusolverDnXgesvdr failed: status {st}")
    iv = int(info.item())
    if iv != 0:
        raise RuntimeError(f"cusolverDnXgesvdr info={iv} (convergence/argument error)")

    # U_cm is (m,k) column-major, held as (k,m) row-major -> transpose to (m,k).
    U = U_cm.transpose(0, 1)                 # (m,k)
    V = V_cm.transpose(0, 1)                 # (n,k)
    Vh = V.mH                                # (k,n),  theta ~= U diag(S) Vh
    if transposed:                           # we decomposed A^H = V S U^H -> swap back
        U, Vh = Vh.mH, U.mH
    return U.contiguous(), S, Vh.contiguous()
