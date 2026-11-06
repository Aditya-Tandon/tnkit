"""Backend-native matrix-free Krylov solvers: dominant-eigenpair Arnoldi and GMRES.

These replace scipy's ``eigs``/``gmres`` in the VUMPS environment solves so the whole
uniform-MPS optimisation runs on the chosen backend (numpy / torch-cpu / torch-cuda)
with **no host round-trip on the heavy operation**. The operators here are applied
matrix-free: the caller passes an ``apply(x) -> y`` closure that contracts the transfer
(an O(D^3 d) einsum) — never a dense D^2 x D^2 matrix — so cost stays O(D^3) and the
GEMM-heavy transfer runs on the GPU exactly like the DMRG kernel.

Only the small projected problems (the m x m Hessenberg eig / least-squares, m ~ 30-60)
touch host numpy — negligible next to the O(D^3) applies, and identical across devices,
so the torch-cuda path is validated by the torch-cpu == numpy cross-check.

Convention: ``be.vdot(a, b)`` conjugates the first argument (numpy/torch ``vdot``), so
``be.vdot(q, w)`` is the Gram-Schmidt coefficient <q|w>.
"""
from __future__ import annotations
import numpy as np


def _c(be, x):
    """A backend scalar -> python complex (host)."""
    return be.to_scalar(x)


def _arnoldi(apply, q0, be, m):
    """One Arnoldi factorisation A Q_m = Q_{m+1} H (m+1, m).

    Returns (Q, H, mm, breakdown): Q a list of mm+1 device basis vectors (Q[-1] is the
    next vector, present only if no breakdown), H the (m+1, m) upper-Hessenberg (host,
    complex), mm the number of completed steps, breakdown True if an invariant subspace
    was hit. Uses one reorthogonalisation pass ("twice is enough") for numerical
    stability against the notoriously non-normal transfer operator.
    """
    H = np.zeros((m + 1, m), dtype=np.complex128)
    Q = [q0]
    mm = m
    breakdown = False
    for j in range(m):
        w = apply(Q[j])
        for _ in range(2):                       # modified Gram-Schmidt, twice
            for i in range(j + 1):
                h = _c(be, be.vdot(Q[i], w))
                H[i, j] += h
                w = w - h * Q[i]
        hn = float(_c(be, be.norm(w)).real)
        H[j + 1, j] = hn
        if hn < 1e-13:                            # invariant subspace reached
            mm = j + 1
            breakdown = True
            break
        Q.append(w / hn)
    return Q, H, mm, breakdown


def eigs_dominant(apply, v0, be, k=1, want_vectors=False, m=None, tol=1e-9,
                  restarts=60):
    """The ``k`` eigenvalues of largest magnitude of a (generally non-Hermitian) linear
    operator ``apply``, via explicitly-restarted Arnoldi.

    Parameters
    ----------
    apply : callable(x) -> y   matrix-free operator on flat backend vectors.
    v0    : backend array      start vector (flat).
    k     : int                number of dominant eigenpairs wanted.
    want_vectors : bool        also return the k Ritz vectors (device arrays).
    m     : int                Krylov dimension per cycle (default max(2k+20, 32)).
    tol   : float              Ritz-residual convergence on the k-th pair.

    Returns
    -------
    (eigvals, vecs): eigvals a host complex ndarray (len k, magnitude-descending);
    vecs a list of k flat backend vectors if want_vectors else None.
    """
    n = int(v0.reshape(-1).shape[0])
    if m is None:
        m = max(2 * k + 20, 32)
    m = min(m, n)
    q0 = v0.reshape(-1)
    q0 = q0 / be.norm(q0)

    w = V = None
    for _ in range(restarts):
        Q, H, mm, breakdown = _arnoldi(apply, q0, be, m)
        Hs = H[:mm, :mm]
        w, V = np.linalg.eig(Hs)
        order = np.argsort(-np.abs(w))
        w = w[order]
        V = V[:, order]
        kk = min(k, mm)
        # cheap Ritz residual: |h_{mm+1,mm}| * |last component of the Ritz vector|
        beta = 0.0 if breakdown else abs(H[mm, mm - 1])
        res = beta * np.abs(V[mm - 1, :kk])
        if breakdown or np.max(res) < tol:
            break
        # restart from the (normalised) sum of the wanted Ritz vectors
        y = V[:, :kk].sum(axis=1)
        q0 = _combine(Q[:mm], y, be)
        q0 = q0 / be.norm(q0)

    eigvals = w[:k]
    vecs = None
    if want_vectors:
        vecs = [_combine(Q[:mm], V[:, r], be) for r in range(min(k, mm))]
    return eigvals, vecs


def _combine(Q, y, be):
    """Σ_j y[j] Q[j] as a device vector (y host complex coefficients)."""
    out = complex(y[0]) * Q[0]
    for j in range(1, len(y)):
        out = out + complex(y[j]) * Q[j]
    return out


def _givens(a, b):
    """Complex Givens rotation zeroing b: returns (cs, sn) with cs real >= 0 such that

        [[ cs,        sn ],  [[a],     [[r],
         [-conj(sn),  cs ]]   [b]]  =   [0]]

    with r = e^{i·arg(a)}·sqrt(|a|²+|b|²) (LAPACK zlartg convention). Explicitly,
        cs = |a| / sqrt(|a|²+|b|²),   sn = (a/|a|)·conj(b) / sqrt(|a|²+|b|²),
    so cs²+|sn|² = 1 and the rotation is unitary."""
    if b == 0:
        return 1.0 + 0j, 0.0 + 0j
    if a == 0:
        return 0.0 + 0j, (np.conj(b) / abs(b))
    t = float(np.hypot(abs(a), abs(b)))
    cs = abs(a) / t
    sn = (a / abs(a)) * (np.conj(b) / t)
    return cs, sn


def gmres(apply, b, be, x0=None, tol=1e-9, atol=1e-11, restart=48, maxiter=40):
    """Restarted GMRES(restart): solve ``apply(x) = b`` matrix-free on the backend.

    Uses Givens rotations so the residual norm is tracked for free per step and the
    inner cycle exits the instant it converges (no wasted matvecs). Returns the flat
    backend solution vector. Only the small triangular solve touches host numpy — tiny
    next to the O(D^3) applies.
    """
    b = b.reshape(-1)
    bnorm = float(be.to_scalar(be.norm(b)).real)
    if bnorm == 0.0:
        return be.zeros_like(b)
    n = int(b.shape[0])
    m = min(restart, n)
    x = be.zeros_like(b) if x0 is None else x0.reshape(-1)
    thresh = max(tol * bnorm, atol)

    for _ in range(maxiter):
        r = b - apply(x)
        beta = float(be.to_scalar(be.norm(r)).real)
        if beta < thresh:
            break
        Q = [r / beta]
        H = np.zeros((m + 1, m), dtype=np.complex128)
        cs = np.zeros(m, dtype=np.complex128)
        sn = np.zeros(m, dtype=np.complex128)
        gg = np.zeros(m + 1, dtype=np.complex128)
        gg[0] = beta
        jm = m
        for j in range(m):
            w = apply(Q[j])
            for i in range(j + 1):               # modified Gram-Schmidt
                h = be.to_scalar(be.vdot(Q[i], w))
                H[i, j] += h
                w = w - h * Q[i]
            hn = float(be.to_scalar(be.norm(w)).real)
            H[j + 1, j] = hn
            if hn > 1e-13:
                Q.append(w / hn)
            # apply the accumulated Givens rotations to the new column
            for i in range(j):
                tmp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -np.conj(sn[i]) * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = tmp
            cs[j], sn[j] = _givens(H[j, j], H[j + 1, j])
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            gg[j + 1] = -np.conj(sn[j]) * gg[j]
            gg[j] = cs[j] * gg[j]
            if abs(gg[j + 1]) < thresh or hn <= 1e-13:
                jm = j + 1
                break
        # back-substitution on the (upper-triangular) H[:jm,:jm] y = gg[:jm]
        y = np.linalg.solve(np.triu(H[:jm, :jm]), gg[:jm])
        x = x + _combine(Q[:jm], y, be)
    return x
