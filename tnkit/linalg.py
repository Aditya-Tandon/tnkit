"""Truncation / canonicalisation routines for MPS.

Every method takes a matrix A (the reshaped two-site tensor) and returns a rank-
chi factorisation U, S, Vh with U S Vh ~= A, plus the discarded weight
(sum of squared truncated singular values) — the quantity DMRG actually needs
to track truncation error.

Methods:
  truncated_svd      reference: full SVD then cut. Most accurate, cuSOLVER-native
                     on NVIDIA, CPU-fallback on Apple.
  randomized_svd     range-finder (Halko-Martinsson-Tropp). Cost ~ O(mn*chi) with
                     power iterations for spectrum sharpening. Needs QR.
  gram_truncation    eigendecomposition of the smaller Gram matrix (A A^H or A^H A),
                     keep top chi, with one factor an exact isometry. GEMM + eigh
                     instead of an SVD; squares the condition number, so it needs
                     double precision at a 1e-8 discarded-weight budget.
  density_truncation deprecated: its Vh = U^H A / S is not an isometry for small S;
                     use truncated_svd.

All are written in numpy/double for a correct DMRG core; on NVIDIA the same
calls map to cuSOLVER.
"""
from __future__ import annotations
import warnings

import numpy as np


def robust_svd(A):
    """Economy SVD that survives LAPACK gesdd non-convergence.

    numpy's default `gesdd` (divide-and-conquer) is fast but occasionally raises
    'SVD did not converge' on ill-conditioned matrices — which happens for real
    DMRG blocks near criticality. Fall back to the slower, more robust `gesvd`.
    """
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _ssvd
        return _ssvd(A, full_matrices=False, lapack_driver="gesvd")


def truncated_svd(A, chi, tol=0.0):
    """Full SVD, keep up to chi singular values (and drop any below tol)."""
    U, S, Vh = robust_svd(A)
    keep = min(chi, S.shape[0])
    if tol > 0:
        keep = min(keep, int(np.sum(S > tol)))
    keep = max(keep, 1)
    discarded = float(np.sum(S[keep:] ** 2))
    return U[:, :keep], S[:keep], Vh[:keep], discarded


def randomized_svd(A, chi, oversample=8, power_iters=2, rng=None):
    """Randomised range-finder SVD (Halko, Martinsson, Tropp 2011).

    Draws a Gaussian sketch, optionally sharpens with `power_iters` subspace
    iterations (essential when the spectrum decays slowly), orthonormalises,
    then does a small dense SVD on the projected matrix.
    """
    rng = np.random.default_rng() if rng is None else rng
    m, n = A.shape
    ell = min(chi + oversample, n)
    Omega = rng.standard_normal((n, ell))
    if np.iscomplexobj(A):
        Omega = Omega + 1j * rng.standard_normal((n, ell))
    Omega = Omega.astype(A.dtype)
    Y = A @ Omega
    Q, _ = np.linalg.qr(Y)
    for _ in range(power_iters):                 # subspace (power) iteration
        Q, _ = np.linalg.qr(A.conj().T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    B = Q.conj().T @ A                           # small: ell x n
    Ub, S, Vh = robust_svd(B)
    U = Q @ Ub
    keep = min(chi, S.shape[0])
    discarded = float(np.sum(S[keep:] ** 2))
    return U[:, :keep], S[:keep], Vh[:keep], discarded


def randomized_svd_device(A, chi, be, oversample=8, power_iters=2, rng=None,
                          mixed=False):
    """Backend-native randomised range-finder SVD (Halko-Martinsson-Tropp).

    Same algorithm as `randomized_svd`, but expressed entirely through the backend
    `be` (matmul via `@`, `be.qr`, `be.svd`) so it stays on whatever device/dtype
    `A` lives on — **no host round-trip**. This is the GPU-relevant path: the
    expensive work becomes GEMM + QR + one *small* dense SVD (of the ell x n
    projection B), rather than a full dense SVD of A. That plays to GEMM
    throughput instead of the bandwidth-bound dense SVD (which is very slow in
    fp64 on weak-fp64 GPUs like the L40S).

    Only the Gaussian sketch touches the host (seeded numpy for reproducibility,
    then moved on-device); its size is n x ell = O(chi*n), negligible vs the
    O(chi^3) GEMMs.

    mixed : run the **range finder** (sketch GEMM + power iterations + their QRs)
        in reduced precision (complex64) and the **projected core** (B = Q^H A and
        its dense SVD) in full precision (complex128). The range finder only has to
        pin down the retained *subspace* — a task robust to fp32 rounding — so Q's
        subspace error is tiny for the fast-decaying blocks DMRG produces; then the
        singular values/vectors and discarded weight are recovered at fp64 quality
        by projecting the *original* fp64 A onto that subspace. This is the
        mixed-precision truncation: on the L40S the fp32 sketch/power GEMMs run at
        ~30-100x the fp64 rate while dE stays at the fp64 floor. On CPU it
        buys no speed but reproduces the fp32 numerics, so the accuracy check is
        CPU-testable.
    """
    rng = np.random.default_rng() if rng is None else rng
    m, n = int(A.shape[0]), int(A.shape[1])
    if min(m, n) <= chi + oversample:            # no compression possible -> full SVD
        U, S, Vh = be.svd(A)
        keep = min(chi, int(S.shape[0]))
        disc = float(np.sum(be.to_numpy(S)[keep:] ** 2))
        return U[:, :keep], S[:keep], Vh[:keep], disc
    ell = int(min(chi + oversample, n))
    Omega = be.randn(rng, (n, ell))
    if mixed:                                    # fp32 range finder -> fp64 core
        As = be.astype(A, "low")
        Q, _ = be.qr(As @ be.astype(Omega, "low"))
        for _ in range(power_iters):             # subspace iteration (spectrum sharpening)
            Q, _ = be.qr(As.conj().T @ Q)
            Q, _ = be.qr(As @ Q)
        Q = be.astype(Q, "high")                 # lift the basis; project A in fp64
    else:
        Q, _ = be.qr(A @ Omega)
        for _ in range(power_iters):
            Q, _ = be.qr(A.conj().T @ Q)
            Q, _ = be.qr(A @ Q)
    B = Q.conj().T @ A                           # small: ell x n (fp64: A native)
    Ub, S, Vh = be.svd(B)
    U = Q @ Ub
    keep = min(chi, int(S.shape[0]))
    disc = float(np.sum(be.to_numpy(S)[keep:] ** 2))
    return U[:, :keep], S[:keep], Vh[:keep], disc


def _gram_left(A, chi, be):
    """Top-chi left factorisation A ~= Q C from the smaller Gram matrix, Q an exact isometry.

    Returns Q (m, k), C (k, n), w (the Gram eigenvalues, descending, on the host) and k.
    m <= n: eigh(A A^H) -> Q, and C = Q^H A.
    m >  n: eigh(A^H A) -> V, then A V = Q R (QR) and C = R V^H. The first j columns of Q span
            the top-j left singular subspace for every j <= k, so cutting U, S, Vh further
            (as the adaptive-rank path does) stays a best-in-subspace truncation.
    """
    m, n = (int(x) for x in A.shape)
    Ah = A.conj().T
    w, Z = be.eigh(A @ Ah if m <= n else Ah @ A)              # ascending
    r = min(m, n)
    k = min(int(chi), r)
    top = list(range(r - 1, r - k - 1, -1))                     # descending order
    Z = Z[:, top]
    if m <= n:
        return Z, Z.conj().T @ A, be.to_numpy(w)[::-1], k
    Q, R = be.qr(A @ Z)
    return Q, R @ Z.conj().T, be.to_numpy(w)[::-1], k


def gram_truncation(A, chi, be, iso="left"):
    """Rank-chi truncation from an eigendecomposition of the smaller Gram matrix.

    Same contract as an SVD truncation: U (m, k), S (k,), Vh (k, n) with U S Vh ~= A, plus the
    absolute discarded weight. One factor is an EXACT isometry, chosen by `iso`:

    iso='left'  U^H U = 1 (a left-canonical site, sweeping right). The carried factor
                C = U^H A is formed directly; S is the row norms of C and Vh = C / S, so the
                caller's S[:, None] * Vh returns C to rounding. Vh is not an isometry.
    iso='right' Vh Vh^H = 1 (a right-canonical site, sweeping left); the mirror image, with
                U * S returning the carried A Vh^H.

    No step divides by a singular value obtained from an eigenvalue. White's density-matrix
    route sets Vh = U^H A / sqrt(lambda) and then uses Vh as an isometry; for small lambda that
    Vh is far from orthonormal and breaks the canonical form.

    Opt-in, not the default: it failed its speed/accuracy acceptance test and is kept for GPU
    speed. Its rounding differs from an SVD's, so chaotic dynamics run with it do not reproduce
    an SVD run (see `DMRGOptions.trunc` for the measured size and the safe range).

    Precision. The Gram eigenvalues carry an absolute error ~ n eps ||A||_2^2, so directions with
    sigma / sigma_max below the floor, ~ sqrt(eps) ~ 1e-8 (up to sqrt(n eps) ~ 1e-7 for
    n ~ 1000), cannot be ranked against each other; which of them is kept is arbitrary, which
    costs nothing while the cut sits well above the floor (svd_tol >= 1e-12). Measured bounds:
    below ~1e-8 S_max, S holds norms of the kept directions, not singular values, is off by up
    to ~1e-8 S_max and need not be monotone; on CUDA one matrix gave S errors of ~2e-10 S_max.
    The discarded weight is the sum of the dropped eigenvalues and is meaningful only above
    ~1e-16 ||A||_F^2. Rank decisions above the floor are unaffected. There is no floor guard.

    Single precision is refused: with eps ~ 1e-7 the discarded weight is wrong by ~ eps ||A||_F^2,
    far above any practical discarded-weight budget.
    """
    if _finfo(A.dtype).eps > np.finfo(np.float64).eps:
        raise ValueError(f"trunc='gram' needs double precision, got {A.dtype}: the Gram matrix "
                         "squares the condition number, so the discarded weight is wrong by "
                         "~ eps ||A||_F^2 (~1e-7 relative in float32); use trunc='svd'")
    if iso == "right":
        Q, C, w, k = _gram_left(A.conj().T, chi, be)
        Vh, US = _adjoint(Q), C.conj().T                         # A ~= US Vh
        S = be.norm_axis(US, axis=0)
        return US / _nonzero(S, be)[None, :], S, Vh, float(np.sum(np.clip(w[k:], 0.0, None)))
    if iso != "left":
        raise ValueError(f"iso must be 'left' or 'right', got {iso!r}")
    U, C, w, k = _gram_left(A, chi, be)
    S = be.norm_axis(C, axis=1)
    return U, S, C / _nonzero(S, be)[:, None], float(np.sum(np.clip(w[k:], 0.0, None)))


def _finfo(dtype):
    """numpy or torch float info of `dtype` (complex dtypes give their real component)."""
    try:
        return np.finfo(dtype)
    except (TypeError, ValueError):
        import torch
        return torch.finfo(dtype)


def _adjoint(x):
    """x^H as a materialised array: on torch, `.conj()` is a lazy view whose conjugate bit makes
    `.numpy()` raise, so it is resolved here."""
    h = x.conj().T
    return h.resolve_conj() if hasattr(h, "resolve_conj") else h


def _nonzero(S, be):
    """S with exact zeros lifted to the smallest normal number of its dtype, so a zero row
    divides to zero, not NaN."""
    return be.guard_denom(S, float(_finfo(S.dtype).tiny))


def warn_if_deprecated(trunc, stacklevel=3):
    """DeprecationWarning for trunc='density'; call it from the public entry point that
    resolved `trunc`, so the warning is raised once and points at the user's call."""
    if trunc == "density":
        warnings.warn("trunc='density' is deprecated: its Vh is not an isometry for small "
                      "singular values, which breaks the canonical form on right-to-left "
                      "steps; use trunc='svd'", DeprecationWarning, stacklevel=stacklevel)


def density_truncation(A, chi):
    """White's density-matrix truncation: eigendecompose rho = A A^H, keep top chi.

    Deprecated: use trunc='svd'. Returns U (left vectors), S (= sqrt eigenvalues) and
    Vh = U^H A / S. That Vh is not an isometry when S is small (its rows overlap by
    ~ eps S_max^2 / (S_i S_j)), so it must not serve as a right-canonical site; `dmrg` and
    `tdvp2_evolve` do exactly that on every right-to-left step. Kept unchanged so that old
    'density' numbers reproduce.
    WARNING: forming A A^H squares the condition number -> singular values below
    ~sqrt(eps)*S_max are lost. Characterised in the accuracy tests.
    """
    rho = A @ A.conj().T
    rho = 0.5 * (rho + rho.conj().T)
    w, U = np.linalg.eigh(rho)                   # ascending
    idx = np.argsort(w)[::-1]                     # descending
    w = w[idx]
    U = U[:, idx]
    # numerical rank: drop eigenvalues at/below the rho noise floor. Keeping
    # near-zero modes would divide by ~0 below and inject garbage rows into Vh
    # (this is exactly what broke DMRG at small chi).
    wmax = max(float(w[0]), 1e-300)
    rank = int(np.sum(w > wmax * 1e-14))
    keep = max(min(chi, rank), 1)
    discarded = float(np.sum(np.clip(w[keep:], 0.0, None)))
    U = U[:, :keep]
    S = np.sqrt(np.clip(w[:keep], 0.0, None))
    # recover row space: S Vh = U^H A  ->  Vh = (U^H A) / S
    SVh = U.conj().T @ A
    Vh = SVh / S[:, None]
    return U, S, Vh, discarded


TRUNCATIONS = {
    "svd": truncated_svd,
    "randomized": randomized_svd,
    "density": density_truncation,
}
