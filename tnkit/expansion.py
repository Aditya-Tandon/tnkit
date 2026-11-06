"""Subspace expansion: global for TDVP (Yang and White, Phys. Rev. B 102, 094315 (2020)), local
for DMRG (Hubig, McCulloch, Schollwoeck and Wolf, Phys. Rev. B 91, 155115 (2015)).

TDVP moves weight only into directions its bond bases already contain. A term of H that reaches
past the sites being updated (a hop over a neighbouring qubit, say) acts on the bond basis as it
stands, so a direction the basis lacks stays empty. Dense tensors carry round-off in every
direction and so grow into it slowly; block tensors carry none, and a charge sector that a bond
lacks stays empty for good.

`expand_right` adds to the right basis of every bond the part of H|psi> that the basis misses,
with zero weight: the state is unchanged, and the next sweep can move weight there. On block
tensors every added state has a definite charge, so the symmetry stays exact, and a sector that
was empty can be filled.

`enriched_split` is the DMRG step: it splits the optimised centre tensor at the bond the sweep
moves across, choosing the new bond basis from the centre and from H applied to the centre on
the side that keeps the basis. The basis then holds states the Hamiltonian couples to, in their
own charge sectors, and the next local solve can put weight there.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from . import invariants
from .legs import fuse, split


def _is_block(x) -> bool:
    return getattr(x, "is_block", False)


def _top(mat, k: int, be, rel_tol: float):
    """U, S, Vh for the `k` largest singular values of `mat` above `rel_tol` times the largest."""
    if _is_block(mat):
        from .blocks import truncated_svd
        U, S, Vh, _ = truncated_svd(mat, k, zero_tol=rel_tol)
        return U, S, Vh
    U, S, Vh = be.svd(mat)
    s = be.to_numpy(S)
    n = min(k, max(1, int((s > rel_tol * s[0]).sum())))
    return U[:, :n], S[:n], Vh[:n]


def _join(a, b, axis: int):
    if _is_block(a):
        from .blocks import concat
        return concat(a, b, axis)
    if isinstance(a, np.ndarray):
        return np.concatenate([a, b], axis)
    import torch
    return torch.cat([a, b], axis)


def _pad_right(t, leg, be):
    """`t` (a, s, b) with zero columns appended to leg b: `leg` (block) or that many (dense)."""
    if _is_block(t):
        from .blocks import zeros
        return _join(t, zeros(t.legs[:2] + (leg,), t.charge, t.data), 2)
    return _join(t, be.zeros((t.shape[0], t.shape[1], leg)), 2)


def _project_out(Y, B, be):
    """Y minus its part in the row space of the right-orthonormal B (twice, for orthogonality)."""
    for _ in range(2):
        P = be.einsum("csb,asb->ca", Y, be.conj(B))
        Y = Y - be.einsum("ca,asb->csb", P, B)
    return Y


def _check_right_orthonormal(M, be, tol: float = 1e-10) -> None:
    """Raise unless every B = M[j], j >= 1, has B B^dag = 1. For a Gram matrix P = B B^dag,
    ||P - 1||^2 = ||P||^2 - 2 ||B||^2 + a, with a the left bond."""
    for j in range(1, len(M)):
        a = int(M[j].shape[0])
        n1 = float(be.to_numpy(be.norm(M[j]))) ** 2
        n2 = float(be.to_numpy(be.norm(be.einsum("asb,csb->ac", M[j], be.conj(M[j]))))) ** 2
        if n2 - 2 * n1 + a > tol * a:
            raise ValueError(f"expand_right needs M right-canonical from site 1; site {j} is not")


def expand_right(M: List[Any], mpo: Sequence[Any], n_extra: int, chi: int, be, L0, RN,
                 rel_tol: float = 1e-8, check: bool = True) -> List[int]:
    """Add up to `n_extra` states to the right basis of every bond of `M`, in place.

    `M` must be right-canonical with its centre at site 0, and `L0`, `RN` the trivial edge
    environments ((bra, mpo, ket), as the TDVP drivers build them). H|psi> is built by applying
    `mpo` site by site from the left and cutting its bonds to `chi` (a zip-up). Then, from the
    right, each bond receives the right Schmidt states of H|psi> that lie outside its basis,
    largest weight first, skipping any below `rel_tol` of the largest. The new rows of B[j] are
    orthonormal to the old ones; M[j-1] gets zero columns, so the state does not change.
    Neither cut (the zip-up, the choice of states) keeps degenerate multiplets whole, so on a
    degenerate spectrum dense and block tensors can receive different states. Returns the number
    of states added at each bond. `check` verifies right-canonicality unless `TNKIT_CHECK=off`.
    """
    if check and invariants.mode() != "off":
        _check_right_orthonormal(M, be)
    N = len(M)
    A, X = [], L0                               # H|psi>, left-canonical; X carries the rest
    for i in range(N):
        T = be.einsum("cwa,asb->cwsb", X, M[i])
        T = be.einsum("cwsb,wvts->ctvb", T, mpo[i])
        if i == N - 1:
            break
        mat, rec = fuse(T, ((0, 1), (2, 3)), be)
        U, S, Vh = _top(mat, chi, be, 1e-14)
        A.append(split(U, rec, be, legs=(0,)))
        X = split(Vh.scale(S, 0) if _is_block(Vh) else S[:, None] * Vh, rec, be, legs=(1,))
    # Y: the right parts of H|psi> at bond j-1, in the basis s_j x (right basis of bond j)
    Y = be.einsum("ctvb,dvb->ctd", T, RN)
    added = [0] * (N - 1)
    for j in range(N - 1, 0, -1):
        Yp = _project_out(Y, M[j], be)
        if float(be.to_numpy(be.norm(Yp))) > 1e-10 * float(be.to_numpy(be.norm(Y))):
            mat, rec = fuse(Yp, ((0,), (1, 2)), be)
            _, _, Vh = _top(mat, n_extra, be, rel_tol)
            new = split(Vh, rec, be, legs=(1,))
            M[j] = _join(M[j], new, 0)
            M[j - 1] = _pad_right(M[j - 1], new.legs[0].flip() if _is_block(new)
                                  else int(new.shape[0]), be)
            added[j - 1] = int(new.shape[0])
        if j > 1:
            Tj = be.einsum("asb,csb->ac", be.conj(M[j]), Y)
            Y = be.einsum("csd,bd->csb", A[j - 1], Tj)
    return added


def _gram(t, first: bool, be):
    """t t^H summed over every leg but the first two (`first`) or the last two, as a four-leg
    tensor: (kept, kept^*) for the first two legs, (kept^*, kept) for the last two."""
    s = "abcdefg"[:t.ndim]
    if first:
        k, o = s[:2], s[2:]
        return be.einsum(f"{s},{k.upper()}{o}->{k}{k.upper()}", t, be.conj(t))
    o, k = s[:-2], s[-2:]
    return be.einsum(f"{o}{k.upper()},{s}->{k.upper()}{k}", be.conj(t), t)


def enriched_split(x, extra, alpha: float, chi: int, be, right: bool, rel_tol: float = 1e-14):
    """Split the centre `x` into an isometry and the part that moves on, with the bond basis
    taken from x and the expansion term `extra`.

    right=True: the new basis spans the first two legs of `x`; it is the `chi` leading
    eigenvectors of x x^H + alpha^2 P P^H, where P is `extra` (same first two legs, any others)
    less its part inside the span of x on those legs. The states x already holds keep their
    weights, and P competes only for the rest. Returns ``(A, C, disc)`` with A (x0, x1, k) an isometry and C = A^H x
    (k, rest of x), so x ~ A C. right=False: the same on the last two legs; returns
    ``(B, C, disc)`` with B (k, x-2, x-1) and C = x B^H. `disc` is the norm^2 of x lost.
    Singular values of x below `rel_tol` times the largest, and eigenvalues below `rel_tol`^2
    times the largest, are dropped.
    """
    s = "abcdefg"[:x.ndim]
    e = "abcdefg"[:extra.ndim]
    k = (s[:2], e[:2], e[2:]) if right else (s[-2:], e[-2:], e[:-2])
    groups = ((0, 1), tuple(range(2, x.ndim))) if right else (tuple(range(x.ndim - 2)),
                                                             (x.ndim - 2, x.ndim - 1))
    mat, rec = fuse(x, groups, be)
    U, _, Vh = _top(mat, min(mat.shape), be, rel_tol)
    Q = split(U, rec, be, legs=(0,)) if right else split(Vh, rec, be, legs=(1,))
    # the part of `extra` outside the span of x on the kept legs
    if right:
        c = be.einsum(f"{k[1]}K,{e}->K{k[2]}", be.conj(Q), extra)
        extra = extra - be.einsum(f"{k[1]}K,K{k[2]}->{e}", Q, c)
    else:
        c = be.einsum(f"{e},K{k[1]}->{k[2]}K", extra, be.conj(Q))
        extra = extra - be.einsum(f"{k[2]}K,K{k[1]}->{e}", c, Q)
    rho = _gram(x, right, be) + _gram(extra, right, be) * alpha ** 2
    mat, rec = fuse(rho, ((0, 1), (2, 3)), be)
    U, _, Vh = _top(mat, chi, be, rel_tol ** 2)
    if right:
        A = split(U, rec, be, legs=(0,))
        C = be.einsum(f"{s[:2]}K,{s}->K{s[2:]}", be.conj(A), x)
    else:
        A = split(Vh, rec, be, legs=(1,))
        C = be.einsum(f"{s},K{s[-2:]}->{s[:-2]}K", x, be.conj(A))
    nx, nc = (float(be.to_numpy(be.norm(t))) for t in (x, C))
    return A, C, max(nx ** 2 - nc ** 2, 0.0)
