"""Factorisations of symmetric tensors, one coupled sector at a time.

A `SymmetricTensor` t, read as a map from its domain to its codomain, is the direct sum over
coupled sectors c of its reduced blocks t_c, each tensored with the identity on the multiplet of
c. An SVD or QR of t is therefore one SVD or QR of each t_c. The factors meet on a new plain bond
leg that holds sector c with degeneracy k_c, the number of values kept in t_c.

A reduced singular value s of block c stands for dim(c) equal dense ones, so it carries the weight
dim(c) s^2. `truncated_svd` ranks values by that weight; the cap `chi` and `svd_tol` count reduced
values, i.e. whole multiplets, and the discarded weight is the sum of dim(c) s^2 over the dropped
values, the squared dense norm of t - U S Vh. A cut inside a multiplet cannot be expressed. A cap
of `chi` multiplets gives a dense bond of sum dim(c) over the kept ones, which `info` reports.

`degenerate_tol` compares weights as well: it treats multiplets whose dim(c) s^2 agree as tied.
Equal weights are equal dense values only between sectors of the same dimension, so the tolerance
is meaningful only there. Across sectors of different dimension it can split a set of equal dense
values, or keep together values that differ.

Provisional: not exported from `tnkit`; this module's API may change before 1.0.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .blocks import _svd_block
from .linalg import _finfo
from .symmetric_tensor import Space, SymmetricTensor

__all__ = ["truncated_svd", "svd", "qr"]


def _grouped(be, mats: List, f) -> List[tuple]:
    """f of every matrix in `mats`, in order; matrices of equal shape share one stacked call."""
    out: List[Optional[tuple]] = [None] * len(mats)
    groups: Dict[tuple, List[int]] = {}
    for i, M in enumerate(mats):
        groups.setdefault(tuple(M.shape), []).append(i)
    for idx in groups.values():
        res = f(be.stack_axis([mats[i] for i in idx], axis=0))
        for j, i in enumerate(idx):
            out[i] = tuple(x[j] for x in res)
    return out


def _svds(be, X, driver: Optional[str]):
    """Economy SVDs of a stack of matrices; on CUDA `driver` goes to ``torch.linalg.svd``. If the
    stacked call fails (gesdd may not converge), each matrix is redone by the backend's SVD,
    which falls back to gesvd."""
    try:
        if driver is not None and getattr(X, "is_cuda", False):
            return be.torch.linalg.svd(X, full_matrices=False, driver=driver)
        return be.svd_batched(X)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        parts = [be.svd(M) for M in X]
        return tuple(be.stack_axis([p[i] for p in parts], axis=0) for i in range(3))


def _cat(be, xs):
    return be.torch.cat(xs) if be.is_torch else np.concatenate(xs)


def _tensor(be, sym, codomain, domain, mats) -> SymmetricTensor:
    """The tensor whose blocks, in sector order, are `mats`."""
    return SymmetricTensor(codomain, domain, be.array(_cat(be, [m.reshape(-1) for m in mats])),
                           be, sym)


def _bond(sym, counts: Dict) -> Space:
    secs = sorted((c for c, k in counts.items() if k), key=sym.sort_key)
    return Space(sym, secs, tuple(counts[c] for c in secs))


def truncated_svd(t: SymmetricTensor, chi: Optional[int] = None, *, svd_tol: float = 0.0,
                  method: str = "svd", driver: Optional[str] = None, zero_tol: float = 0.0,
                  degenerate_tol: float = 0.0, density: bool = False,
                  info: Optional[dict] = None):
    """``t ~ U S Vh``, keeping at most `chi` multiplets (all with `chi` None) chosen across all
    coupled sectors by their weight dim(c) s^2.

    Returns ``(U, S, Vh, disc)``: U maps the new bond to t's codomain, Vh maps t's domain to the
    bond, S is the diagonal bond matrix of the kept values, and `disc` is the absolute discarded
    weight, sum of dim(c) s^2 over the dropped values. The bond is a plain leg holding each
    sector that keeps a value, with the number kept as its degeneracy. One host copy of the
    singular values per call.

    svd_tol  : > 0 keeps the fewest multiplets (at least one) whose relative discarded weight
               stays within svd_tol, still capped at `chi`.
    zero_tol : drop values s <= zero_tol * max(s) before the cap; the largest s always stays.
    degenerate_tol : > 0 keeps or drops whole each group of multiplets whose weights
               dim(c) s^2 agree within this relative tolerance (compared as sqrt(dim(c)) s, with
               the rule of `tnkit._dmrg._cut`). Meaningful only between sectors of equal
               dimension; see the module docstring. One multiplet is never split in any case.
    method   : 'svd' (the backend's SVD, blocks of equal shape stacked into one call) or 'gram'
               (`tnkit.blocks._svd_block`, block by block, double precision only). The Gram
               route squares the condition number: at condition numbers 1e8 to 1e16 its product
               U S Vh reconstructs t only to ~1e-8 relative (measured 2e-10 to 2e-8).
    density  : `t` is a density matrix (Hermitian, positive semi-definite), whose singular
               values are its eigenvalues lambda: rank and discard by the weight dim(c) lambda
               instead of dim(c) lambda^2.
    driver   : CUDA SVD driver for method 'svd', passed to ``torch.linalg.svd``; None keeps the
               backend's (gesvd). Ignored off CUDA and by 'gram'.
    info     : optional dict, filled with ``pinned`` (as in `tnkit.blocks.truncated_svd`),
               ``kept``, the number of multiplets kept in each sector, and ``dim``, the dense
               dimension of the bond.
    """
    if chi is not None and chi < 1:
        raise ValueError(f"chi must be at least 1, got {chi}")
    if method not in ("svd", "gram"):
        raise ValueError(f"unknown method {method!r}; expected 'svd' or 'gram'")
    if not t.sectors:
        raise ValueError("the tensor has no coupled sectors to factorise")
    be, sym = t.be, t.sym
    if method == "gram" and _finfo(t.data.dtype).eps > np.finfo(np.float64).eps:
        raise ValueError(f"method='gram' needs double precision, got {t.data.dtype}: the Gram "
                         "matrix squares the condition number; use method='svd'")
    from ._dmrg import _cut
    mats = [t.block(c) for c in t.sectors]
    if method == "svd":
        parts = _grouped(be, mats, lambda X: _svds(be, X, driver))
    else:
        parts = [_svd_block(M, "gram", driver) for M in mats]
    sv = np.asarray(be.to_numpy(_cat(be, [S for _, S, _ in parts])), np.float64)
    lens = [int(S.shape[0]) for _, S, _ in parts]
    owner = np.repeat(np.arange(len(parts)), lens)
    w = np.repeat([float(sym.dim(c)) for c in t.sectors], lens) * (sv if density else sv ** 2)
    tot = float(w.sum()) or 1.0
    alive = sv > zero_tol * sv.max() if zero_tol > 0 else np.ones(len(sv), bool)
    alive[np.argmax(sv)] = True
    idx = np.flatnonzero(alive)
    order = idx[np.lexsort((idx, -w[idx]))]                 # by weight, descending, stable
    cap = len(order) if chi is None else int(chi)
    k, pinned, _ = _cut(np.sqrt(w[order]), cap, svd_tol, degenerate_tol,
                        float(w[~alive].sum()) / tot, tot)
    kept = np.bincount(owner[order[:k]], minlength=len(parts))
    drop = np.ones(len(sv), bool)
    drop[order[:k]] = False
    disc = float(w[drop].sum())
    counts = {c: int(n) for c, n in zip(t.sectors, kept)}
    if info is not None:
        info["pinned"] = pinned
        info["kept"] = {c: n for c, n in counts.items() if n}
        info["dim"] = sum(sym.dim(c) * n for c, n in counts.items())
    B = _bond(sym, counts)
    keep = [(U[:, :n], S[:n], Vh[:n]) for (U, S, Vh), n in zip(parts, kept) if n]
    return (_tensor(be, sym, t.codomain, (B,), [u for u, _, _ in keep]),
            _tensor(be, sym, (B,), (B,), [be.diag(s) for _, s, _ in keep]),
            _tensor(be, sym, (B,), t.domain, [v for _, _, v in keep]), disc)


def svd(t: SymmetricTensor, method: str = "svd", driver: Optional[str] = None):
    """``t = U S Vh`` with every value kept: `truncated_svd` without a cap."""
    U, S, Vh, _ = truncated_svd(t, None, method=method, driver=driver)
    return U, S, Vh


def qr(t: SymmetricTensor, right: bool = False) -> Tuple[SymmetricTensor, SymmetricTensor]:
    """``t = Q R`` block by block, Q an isometry from the new bond into t's codomain; with
    `right`, ``t = L Q``, Q a co-isometry from t's domain onto the bond. The bond holds sector c
    with degeneracy min(rows, columns) of t_c."""
    if right:
        Q, R = qr(t.adjoint())
        return R.adjoint(), Q.adjoint()
    if not t.sectors:
        raise ValueError("the tensor has no coupled sectors to factorise")
    be, sym = t.be, t.sym
    parts = _grouped(be, [t.block(c) for c in t.sectors], be.qr)
    B = _bond(sym, {c: int(q.shape[1]) for c, (q, _) in zip(t.sectors, parts)})
    return (_tensor(be, sym, t.codomain, (B,), [q for q, _ in parts]),
            _tensor(be, sym, (B,), t.domain, [r for _, r in parts]))
