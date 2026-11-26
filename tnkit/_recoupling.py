"""Recoupling of symmetric tensors: the elementary moves on fusion-tree pairs, the recoupling
matrix of one sector assignment, and `Recoupler`, which applies a permutation to a buffer.

For one sector on every leg, a permutation maps the old tree pairs onto a fixed combination of
the new ones (old pair i = sum_j M[i, j] new pair j). M is built from F, R, three-leg CG arrays
and cups by exchanging adjacent leaves and bending the last leaf of one side to the other, and is
cached on the host by labels alone (`RECOUPLING_CACHE`). In a product that takes its
coefficients from its factors, factors with trivial coefficients, such as U(1), do not enter the
key: assignments that differ only in their labels share one matrix (for U(1) x SU(2), every
charge shares the matrix of its spins). The rest of a permutation that does not depend on the
degeneracies is cached by sectors and dual flags (`Structure`, `PERMUTE_CACHE`).
"""
from __future__ import annotations

from functools import lru_cache, reduce
from operator import itemgetter
from typing import Any, Dict, Tuple

import numpy as np

from . import _symmetric_layout as sl
from . import invariants
from ._bytecache import Growing, env_cache, nbytes
from ._symmetric_layout import Layout, Target, fusions, lru_info, to_index
from .backend import _negligible_imag
from .fusion_trees import FusionTree, fusion_trees
from .symmetry import (Label, ProductSymmetry, _plain_product, cg_tensor, cup, f_tensor,
                       r_matrix)

# host recoupling matrices, keyed by labels; permute structures, keyed by sectors and flags
RECOUPLING_CACHE = env_cache("TNKIT_SYMMETRIC_RECOUPLING_CACHE_MB", 256)
PERMUTE_CACHE = env_cache("TNKIT_SYMMETRIC_PERMUTE_CACHE_MB", 1024)


# --------------------------------------------------------------------------- elementary moves
@lru_cache(maxsize=1 << 16)
def _swap(sym, t: FusionTree, i: int):
    """Leaves i and i + 1 of a splitting tree exchanged: ((tree, coefficient), ...)."""
    u, fl, v = list(t.uncoupled), list(t.isdual), list(t.vertices)
    lines = (u[0],) + t.innerlines + (t.coupled,)
    u[i], u[i + 1], fl[i], fl[i + 1] = u[i + 1], u[i], fl[i + 1], fl[i]
    out = []
    if i == 0:                                   # R on the first vertex
        R = r_matrix(sym, t.uncoupled[0], t.uncoupled[1], lines[1])
        for nu in range(R.shape[1]):
            out.append((FusionTree(tuple(u), t.coupled, tuple(fl), t.innerlines,
                                   (nu,) + tuple(v[1:])), complex(R[v[0], nu])))
        return tuple(out)
    A, B, C, E, D = lines[i - 1], t.uncoupled[i], t.uncoupled[i + 1], lines[i], lines[i + 1]
    acc: Dict[Label, np.ndarray] = {}
    for F, _ in fusions(sym, B, C):              # F-move, R on (B C)_F, inverse F-move
        if not sym.multiplicity(A, F, D):
            continue
        left = f_tensor(sym, A, B, C, D, E, F)[v[i - 1], v[i]]
        R = r_matrix(sym, B, C, F)
        for E2, _ in fusions(sym, A, C):
            if sym.multiplicity(E2, B, D):
                x = np.einsum("kl,kr,abrl->ab", left, R, f_tensor(sym, A, C, B, D, E2, F).conj())
                acc[E2] = acc[E2] + x if E2 in acc else x
    for E2, x in acc.items():
        inner = t.innerlines[:i - 1] + (E2,) + t.innerlines[i:]
        for (a, b), val in np.ndenumerate(x):
            verts = tuple(v[:i - 1]) + (a, b) + tuple(v[i + 1:])
            out.append((FusionTree(tuple(u), t.coupled, tuple(fl), inner, verts), complex(val)))
    return tuple(out)


@lru_cache(maxsize=1 << 14)
def _bend_coefficients(sym, a, b, c, dual: bool) -> np.ndarray:
    """B[mu, nu]: the leaf b of the vertex (a b -> c)_mu, moved to the other side as dual(b) on
    the vertex (c dual(b) -> a)_nu. An overlap of two CG arrays with the cups of the leg."""
    bb, eye = sym.dual(b), np.eye(sym.dim(b))
    old = cup(sym, bb).conj() if dual else eye
    new = eye if dual else cup(sym, b).conj()
    lhs = np.einsum("xz,azwm->axwm", old, cg_tensor(sym, a, b, c))
    rhs = np.einsum("xz,wzan->axwn", new.conj(), cg_tensor(sym, c, bb, a).conj())
    return np.einsum("axwn,axwm->mn", rhs.conj(), lhs) / sym.dim(a)


@lru_cache(maxsize=1 << 16)
def _bend(sym, s: FusionTree, f: FusionTree):
    """The last leaf of the splitting tree s moved to the end of the fusion tree f:
    (((s', f'), coefficient), ...)."""
    n, m = len(s.uncoupled), len(f.uncoupled)
    if n == 1:
        a, b, c, mu, s2 = sym.unit(), s.uncoupled[0], s.coupled, 0, FusionTree((), sym.unit(), ())
    else:
        a, b, c, mu = s.steps()[-1]
        s2 = FusionTree(s.uncoupled[:-1], a, s.isdual[:-1], s.innerlines[:-1], s.vertices[:-1])
    d, bb = s.isdual[-1], sym.dual(b)
    B = _bend_coefficients(sym, a, b, c, d)
    out = []
    for nu in range(B.shape[1]):
        if m == 0:
            f2 = FusionTree((bb,), a, (not d,))
        else:
            f2 = FusionTree(f.uncoupled + (bb,), a, f.isdual + (not d,),
                            f.innerlines + ((c,) if m >= 2 else ()), f.vertices + (nu,))
        out.append(((s2, f2), complex(B[mu, nu])))
    return tuple(out)


def _transform(sym, pair, n1: int, p1, p2) -> Dict[Tuple[FusionTree, FusionTree], complex]:
    """The tree pair `pair` after `permute(·, p1, p2)`, as {new pair: coefficient}. Legs bound for
    the other side are moved to the end of their side and bent; then each side is ordered."""
    state = {pair: 1.0}
    sides = [list(range(n1)), list(range(n1, n1 + len(pair[1].uncoupled)))]

    def step(moves, side):
        nonlocal state
        new: Dict[Any, complex] = {}
        for (s, f), v in state.items():
            for q, k in moves(s, f):
                new[q] = new.get(q, 0) + v * (k if side == 0 else k.conjugate())
        state = new

    def swap(side, j):
        if side == 0:
            step(lambda s, f: (((x, f), k) for x, k in _swap(sym, s, j)), 0)
        else:
            step(lambda s, f: (((s, x), k) for x, k in _swap(sym, f, j)), 1)
        lst = sides[side]
        lst[j], lst[j + 1] = lst[j + 1], lst[j]

    def bend(side):
        if side == 0:
            step(lambda s, f: _bend(sym, s, f), 0)
        else:
            step(lambda s, f: (((y, x), k) for (x, y), k in _bend(sym, f, s)), 1)
        sides[1 - side].append(sides[side].pop())

    for side, target in ((0, set(p2)), (1, set(p1))):
        for leg in [x for x in sides[side] if x in target]:
            for j in range(sides[side].index(leg), len(sides[side]) - 1):
                swap(side, j)
            bend(side)
    for side, target in ((0, p1), (1, p2)):
        for t, leg in enumerate(target):
            for j in range(sides[side].index(leg) - 1, t - 1, -1):
                swap(side, j)
    return state


def tree_pairs(sym, leaves, flags, n1):
    """The tree pairs of these leaves in buffer order: coupled sector, then row tree, then
    column tree."""
    rows = fusion_trees(sym, leaves[:n1], flags[:n1])
    cols: Dict[Label, list] = {}
    for f in fusion_trees(sym, leaves[n1:], flags[n1:]):
        cols.setdefault(f.coupled, []).append(f)
    return [(s, f) for s in rows for f in cols.get(s.coupled, ())]


def _recoupling_matrix(sym, n1, p1, p2, leaves, flags) -> np.ndarray:
    """M with old pair i = sum_j M[i, j] new pair j, for one sector on every leg."""
    old = tree_pairs(sym, leaves, flags, n1)
    moved = [(leaves[k], flags[k]) if (k < n1) == (j < len(p1)) else (sym.dual(leaves[k]),
                                                                       not flags[k])
             for j, k in enumerate(p1 + p2)]
    new = tree_pairs(sym, tuple(x for x, _ in moved), tuple(y for _, y in moved), len(p1))
    col = {q: j for j, q in enumerate(new)}
    M = np.zeros((len(old), len(new)), complex)
    for i, pair in enumerate(old):
        for q, v in _transform(sym, pair, n1, p1, p2).items():
            M[i, col[q]] += v
    if not M.imag.any():
        M = M.real.copy()
    M.flags.writeable = False
    return M


_ONE = np.ones((1, 1))
_ONE.flags.writeable = False


def _same(a):
    return a


@lru_cache(maxsize=None)
def recoupling_part(sym):
    """(part, project): the symmetry whose labels decide the recoupling matrices of `sym`, and
    the map from a label of `sym` to one of `part`. A plain product (`_plain_product`) drops its
    factors with trivial coefficients (`Symmetry.trivial_coefficients`): they multiply every
    matrix by 1 and keep its tree order, so one matrix serves every charge. `part` is None when
    nothing is left, as for U(1)."""
    if sym.trivial_coefficients:
        return None, _same
    if not _plain_product(sym):
        return sym, _same
    keep = [i for i, f in enumerate(sym.factors) if not f.trivial_coefficients]
    if len(keep) == len(sym.factors):
        return sym, _same
    part = (sym.factors[keep[0]] if len(keep) == 1
            else ProductSymmetry(tuple(sym.factors[i] for i in keep)))
    return part, itemgetter(*keep)


def matrix_key(sym, n1, p1, p2, leaves, flags):
    """The key of a recoupling matrix in `RECOUPLING_CACHE`, on the labels of
    `recoupling_part(sym)` alone; None when the part is None."""
    part, project = recoupling_part(sym)
    return None if part is None else (part, n1, p1, p2, tuple(map(project, leaves)), flags)


def recoupling_matrix(sym, n1, p1, p2, leaves, flags) -> np.ndarray:
    """The cached host recoupling matrix (see `_recoupling_matrix`), keyed by `matrix_key`."""
    key = matrix_key(sym, n1, p1, p2, leaves, flags)
    if key is None:                     # unique fusion: one tree pair if both sides couple alike
        cod, dom = (reduce(sym.fuse, x, sym.unit()) for x in (leaves[:n1], leaves[n1:]))
        return _ONE if cod == dom else np.zeros((0, 0))
    return _cached(key)


def _cached(key) -> np.ndarray:
    return RECOUPLING_CACHE.get(key, lambda: _recoupling_matrix(*key))


# --------------------------------------------------------------------------- recoupler
def _pair_table(lay: Layout, old_leg):
    """Every tree pair of `lay`, sorted by its sector assignment: the sector position on each leg
    (legs in the old order: leg j of `lay` is old leg old_leg[j]), the assignment as one integer,
    and the pair's row tree, column tree and coupled sector (`Layout.pair_trees`)."""
    counts = [len(V.sectors) for V in lay.codomain + lay.domain]
    counts = [counts[j] for j in np.argsort(old_leg)]         # in old leg order
    r, f, sec = lay.pair_trees()
    pos = np.concatenate([lay.row.trees.tup[r], lay.col.trees.tup[f]], axis=1)
    pos = pos[:, np.argsort(old_leg)]
    code = (np.ravel_multi_index(pos.T, counts) if len(counts)
            else np.zeros(len(r), np.int64))
    order = np.argsort(code, kind="stable")                   # buffer order within a group
    return pos[order], code[order], (r[order], f[order], sec[order])


def _suffix_products(x):
    """out[:, k] = prod(x[:, k + 1:]) for an integer array of shape (rows, legs)."""
    out = np.ones_like(x)
    if x.shape[1] > 1:
        out[:, :-1] = np.cumprod(x[:, :0:-1], axis=1)[:, ::-1]
    return out


def _exclusive_cumsum(x):
    return np.cumsum(x) - x


def _offsets_within(group, size):
    """For each entry, the sum of `size` over the earlier entries of its group."""
    order = np.argsort(group, kind="stable")
    g, before = group[order], _exclusive_cumsum(size[order])
    first = np.r_[True, g[1:] != g[:-1]] if len(g) else np.zeros(0, bool)
    out = np.empty_like(size)
    out[order] = before - before[first][np.cumsum(first) - 1]
    return out


class Structure:
    """What `permute(·, p1, p2)` needs from a layout beyond its degeneracies, shared by every
    layout with the same sectors and dual flags (`structure`).

    Tree pairs of the old and new layouts sorted alike by sector assignment (`old`, `new`: row
    tree, column tree, sector position), the assignment of each pair (`owner`) and its place
    in it (`rank`); per assignment, the sector position on each old leg (`pos`), its number of
    tree pairs, and either the single factor of a one-pair assignment (`value`) or the id of
    its recoupling matrix (`key`, -1 for one pair). `mats[p]` stacks M^T of every matrix with
    p tree pairs; matrix k is `mats[kp[k]][klocal[k]]`."""

    def __init__(self, lay: Layout, out: Layout, p1, p2):
        sym, spaces, n1 = lay.sym, lay.codomain + lay.domain, len(lay.codomain)
        self.perm, self.n1, self.n1_new = p1 + p2, n1, len(p1)
        flags = tuple(V.isdual for V in spaces)
        old_pos, old_code, self.old = _pair_table(lay, list(range(len(spaces))))
        _, new_code, self.new = _pair_table(out, list(self.perm))
        if not np.array_equal(old_code, new_code):
            raise AssertionError("tree pairs of the two layouts do not match")
        first = np.r_[True, old_code[1:] != old_code[:-1]] if len(old_code) else np.zeros(0, bool)
        starts = np.flatnonzero(first)
        self.owner = np.cumsum(first) - 1
        self.rank = np.arange(len(old_code)) - starts[self.owner]
        self.pos = old_pos[starts]
        self.npairs = np.diff(np.r_[starts, len(old_code)])
        # assignments that share their labels on the recoupling part share one matrix
        part, project = recoupling_part(sym)
        ids: Dict[Any, int] = {}
        leg_ids = [np.array([ids.setdefault(None if part is None else project(V.leaf(a)),
                                            len(ids)) for a in V.sectors], np.int64)
                   for V in spaces]
        labels = list(ids)
        lid = np.stack([leg_ids[k][self.pos[:, k]] for k in range(len(spaces))], axis=1)
        code = np.zeros(len(lid), np.int64)
        for k in range(lid.shape[1]):                    # rows as integers, mixed radix
            code = code * len(labels) + lid[:, k]
        if len(labels) ** lid.shape[1] >= 2 ** 62:
            code = np.unique(lid, axis=0, return_inverse=True)[1].reshape(-1)
        first, which = np.unique(code, return_index=True, return_inverse=True)[1:]
        rows = lid[first]
        Ms = [_ONE if part is None else
              _cached((part, n1, p1, p2, tuple(labels[i] for i in row), flags))
              for row in rows.tolist()]
        size = np.array([len(M) for M in Ms], np.int64)
        if not np.array_equal(size[which], self.npairs):
            raise AssertionError("recoupling matrices do not match the tree pairs")
        multi = np.flatnonzero(size > 1)
        mid = np.full(len(Ms), -1, np.int64)
        mid[multi] = np.arange(len(multi))
        self.key = mid[which]
        self.value = np.asarray([M[0, 0] if len(M) == 1 else 1.0 for M in Ms] or [1.0])[which]
        self.kp = size[multi]
        self.klocal = np.zeros(len(multi), np.int64)
        self.mats = {}
        for p in np.unique(self.kp).tolist():
            idx = np.flatnonzero(self.kp == p)
            self.klocal[idx] = np.arange(len(idx))
            self.mats[p] = np.stack([Ms[multi[k]].T for k in idx])

    def nbytes(self) -> int:
        arrays = [*self.old, *self.new, self.owner, self.rank, self.pos, self.npairs, self.key,
                  self.value, self.kp, self.klocal, *self.mats.values()]
        return sum(x.nbytes for x in arrays)


def structure(lay: Layout, out: Layout, p1, p2) -> Structure:
    """The cached `Structure` of permute(·, p1, p2) on layouts with the sectors and dual flags
    of `lay`; `out` is the layout it produces."""
    key = (lay.sym, tuple((V.sectors, V.isdual) for V in lay.codomain),
           tuple((V.sectors, V.isdual) for V in lay.domain), p1, p2)
    return PERMUTE_CACHE.get(key, lambda: Structure(lay, out, p1, p2))


@lru_cache(maxsize=None)
def _numpy_dtype(dtype) -> np.dtype:
    """The numpy dtype of a numpy or torch dtype."""
    if isinstance(dtype, (np.dtype, type)):
        return np.dtype(dtype)
    import torch
    return torch.empty(0, dtype=dtype).numpy().dtype


@lru_cache(maxsize=None)
def real_dtype(dtype) -> np.dtype:
    """The numpy real dtype of a numpy or torch dtype."""
    return np.empty(0, _numpy_dtype(dtype)).real.dtype


def _cast(be, M, dtype):
    """(M on the backend's device for data of `dtype`, whether that made a copy), from the host
    array M: in `dtype`, or on numpy, for real M, in its real counterpart. The cast is done on the
    host, so a device without 64-bit types never receives one. A real dtype refuses a significant
    imaginary part, as a real backend does."""
    want = _numpy_dtype(dtype)
    if not be.is_torch and M.dtype.kind == "f":
        want = real_dtype(want)
    if np.iscomplexobj(M) and want.kind != "c":
        if not _negligible_imag(M)[0]:
            raise TypeError("a complex recoupling matrix needs complex data, not a real dtype")
        M = M.real
    h = np.ascontiguousarray(M, want)
    if not be.is_torch:
        return h, h is not M
    x = be.torch.as_tensor(h, device=be.device)
    return x, h is not M or be.device.type != "cpu"


def _positions(start, coef, shape, it, device=None):
    """Rows k of `start` + sum_j digit_j coef[k, :, j] for every pair (rows of `shape`, entries
    >= 1) and every combination of digits 0 <= digit_j < shape[:, j]: pairs in order, digits
    row-major, two separate arrays of dtype `it`. Built one leg at a time: each entry's
    children take its value plus their place times the leg's coefficient, so no intermediate
    exceeds its final value. With `device`, it runs in torch there, by gathers alone (on MPS,
    repeats and index_select of integers go through float32)."""
    legs = [j for j in range(shape.shape[1]) if (shape[:, j] > 1).any()]
    L = len(legs)
    host = np.concatenate([start.T, coef[0][:, legs], coef[1][:, legs], shape[:, legs]],
                          axis=1).astype(it)
    if device is None:
        x, arange = host, np.arange
    else:
        import torch
        x = torch.as_tensor(host, device=device)
        arange = (lambda m, dtype: torch.arange(m, dtype=dtype, device=device))
    pid = arange(len(shape), dtype=x.dtype)
    g, h = x[pid, 0], x[pid, 1]
    size = np.ones(len(shape), np.int64)
    for k, j in enumerate(legs):
        size = size * shape[:, j]
        m = int(size.sum())
        r = x[pid, 2 + 2 * L + k]
        first = r.cumsum(0, dtype=x.dtype) - r
        if device is None:
            parent = np.repeat(np.arange(len(r)), r)
        else:
            mark = x.new_zeros(m)
            mark[first[1:]] = 1
            parent = mark.cumsum(0, dtype=x.dtype)
        pid = pid[parent]
        e = arange(m, dtype=x.dtype) - first[parent]
        g = g[parent] + e * x[pid, 2 + k]
        h = h[parent] + e * x[pid, 2 + L + k]
    return g, h


def _on_device(be):
    """Whether a map's positions are built on the backend's device (torch, not the CPU): there
    only per-pair arrays cross from the host, not one integer per buffer entry. The maps are
    queued on the current stream; work on another stream must wait for it."""
    return be.is_torch and be.device.type != "cpu"


def _is_permutation(v, n: int) -> bool:
    """Whether the integer array v (numpy or torch) holds each of 0, ..., n - 1 once."""
    if len(v) != n:
        return False
    if n == 0:
        return True
    if int(v.min()) < 0:
        return False
    c = np.bincount(v, minlength=n) if isinstance(v, np.ndarray) else v.bincount(minlength=n)
    return len(c) == n and bool((c == 1).all())


class Recoupler(Growing):
    """`permute(·, p1, p2)` on the buffers of one layout, on one device:

        x = old[src];  y[:n_scalar] = scale x[:n_scalar];  y[stack] = M_stack @ x[stack];
        new = y[inv].

    Sector assignments with a single tree pair take one factor each (`scale`). The others, with
    p tree pairs and D degeneracy combinations, are blocks of shape (p, D) in x. Assignments that
    share their matrix (as charges that differ on the same spins do) sit side by side in one
    block of shape (p, D_1 + D_2 + ...), and blocks of one shape are contiguous, so each shape
    takes one batched product with its stacked matrices M^T. On torch, where each product has
    a fixed cost, assignments are merged only if that adds no shapes. `n_groups` counts the
    assignments with several tree pairs.

    The trees and matrices come from the cached `Structure`; only the positions depend on the
    degeneracies, and they are computed in one pass of array operations. The matrices and
    factors are kept on the host in their own dtype, and cast and moved to the device once for
    each dtype of data they meet; the cache that holds the map counts each new cast."""

    def __init__(self, lay: Layout, p1, p2, be):
        spaces, n1 = lay.codomain + lay.domain, len(lay.codomain)
        self.target = Target(lay.sym,
                             tuple(spaces[k] if k < n1 else spaces[k].dual() for k in p1),
                             tuple(spaces[k] if k >= n1 else spaces[k].dual() for k in p2))
        out = self.target.get()
        s = structure(lay, out, p1, p2)
        perm, n1_new = list(s.perm), s.n1_new
        # degeneracies of each assignment on the old legs, and in the new leg order
        d = np.stack([np.asarray(V.degeneracies, np.int64)[s.pos[:, k]]
                      for k, V in enumerate(spaces)], axis=1)
        shape = d[:, perm]
        D = shape.prod(axis=1)
        old_stride = np.concatenate([_suffix_products(d[:, :n1]),
                                     _suffix_products(d[:, n1:])], axis=1)
        # where each assignment's block sits in x, and its row length there
        scalar = s.npairs == 1
        base, width = np.zeros(len(D), np.int64), D.copy()
        base[scalar] = _exclusive_cumsum(D[scalar])
        self.n_scalar = n_scalar = int(D[scalar].sum())
        multi = np.flatnonzero(~scalar)
        self.n_groups = len(multi)
        key = s.key[multi]
        by_key = np.zeros(len(s.kp), np.int64)
        np.add.at(by_key, key, D[multi])
        per_key = len(set(zip(s.kp[key].tolist(), by_key[key].tolist())))
        per_group = len(set(zip(s.npairs[multi].tolist(), D[multi].tolist())))
        if not be.is_torch or per_key <= per_group:          # one block per matrix
            uniq, item = np.unique(key, return_inverse=True)
            col = _offsets_within(item, D[multi])
            item_key, item_p, item_w = uniq, s.kp[uniq], by_key[uniq]
        else:                                                # one block per assignment
            item = np.arange(len(multi))
            col = np.zeros(len(multi), np.int64)
            item_key, item_p, item_w = key, s.npairs[multi], D[multi]
        # blocks of one shape are contiguous: stacks in order of (p, width)
        shapes, stack_of = np.unique(np.stack([item_p, item_w], axis=1).reshape(-1, 2), axis=0,
                                     return_inverse=True)
        stack_of = stack_of.reshape(-1)
        order = np.argsort(stack_of, kind="stable")
        count = np.bincount(stack_of, minlength=len(shapes))
        size = count * shapes[:, 0] * shapes[:, 1]
        stack_start = n_scalar + _exclusive_cumsum(size)
        slot = _offsets_within(stack_of, np.ones(len(item_p), np.int64))
        item_base = stack_start[stack_of] + slot * item_p * item_w
        base[multi] = item_base[item] + col
        width[multi] = item_w[item]
        self.stacks = []
        for j, (p, w) in enumerate(shapes.tolist()):
            ks = item_key[order[_exclusive_cumsum(count)[j]:][:count[j]]]
            mats = s.mats[p][s.klocal[ks]]
            self.stacks.append((int(stack_start[j]), int(size[j]), (int(count[j]), p, w), mats))
        # every entry: old buffer position g and new buffer position h, both affine in its
        # degeneracy combination (row-major in the new leg order); pairs taken in the order of
        # their blocks in x, so that the entries' places in x are 0, 1, ... and src = g
        a, E = s.owner, D[s.owner]
        n = int(E.sum())
        if not n == lay.size == out.size:
            raise AssertionError("recoupling does not preserve the buffer size")
        it = sl.index_dtype(n)
        old_start, old_nc = lay.pair_starts(*s.old)
        new_start, new_nc = out.pair_starts(*s.new)
        xs = base[a] + s.rank * width[a]
        by_x = np.argsort(xs, kind="stable")
        if not np.array_equal(xs[by_x], _exclusive_cumsum(E[by_x])):
            raise AssertionError("recoupling blocks do not tile x")
        cod = np.arange(len(perm)) < n1_new
        new_stride = np.concatenate([_suffix_products(shape[:, :n1_new]),
                                     _suffix_products(shape[:, n1_new:])], axis=1)
        gc = np.where(np.asarray(perm) < n1, old_nc[:, None], 1) * old_stride[a][:, perm]
        hc = np.where(cod, new_nc[:, None], 1) * new_stride[a]
        on_device = _on_device(be)
        g, h = _positions(np.stack([old_start, new_start])[:, by_x],
                          np.stack([gc, hc])[:, by_x], shape[a][by_x], it,
                          be.device if on_device else None)
        scale = None
        if (s.value[scalar] != 1).any():
            scale = np.repeat(s.value[scalar], D[scalar])
        if on_device:
            torch = be.torch
            inv = torch.full((n,), -1, dtype=g.dtype, device=be.device)
            inv[h] = torch.arange(n, dtype=g.dtype, device=be.device)
        else:
            inv = np.full(n, -1, it)
            inv[h] = np.arange(n, dtype=it)
            if (inv < 0).any():
                raise AssertionError("recoupling does not cover the new buffer once")
        if invariants.mode() == "debug" and not (_is_permutation(g, n) and _is_permutation(h, n)):
            raise AssertionError("recoupling positions are not permutations of the buffers")
        if self.stacks:
            self.src, self.inv = to_index(be, g, n), to_index(be, inv, n)
            self.scale = scale
        else:                                     # one gather: src composed with inv
            self.src, self.inv = to_index(be, g[inv], n), None
            if scale is not None:
                scale = scale[inv.cpu().numpy() if on_device else inv]
            self.scale = scale
        self.be, self._cast, self._cast_bytes = be, {}, 0

    def nbytes(self) -> int:
        return self._cast_bytes + sum(nbytes(x) for x in [self.src, self.inv, self.scale]
                                      + [m for *_, m in self.stacks] if x is not None)

    def _cast_to(self, dtype):
        got = self._cast.get(dtype)
        if got is None:
            new = [None if self.scale is None else _cast(self.be, self.scale, dtype)]
            new += [_cast(self.be, M, dtype) for *_, M in self.stacks]
            casts = [None if c is None else c[0] for c in new]
            got = (casts[0], [st[:3] + (M,) for st, M in zip(self.stacks, casts[1:])])
            if self._cast.setdefault(dtype, got) is got:
                self._cast_bytes += sum(nbytes(c[0]) for c in new if c is not None and c[1])
                self.grew()
            got = self._cast[dtype]
        return got

    def apply(self, data):
        scale, stacks = self._cast_to(data.dtype)
        x = data[self.src]
        if self.inv is None:
            return x if scale is None else x * scale
        if self.be.is_torch:                     # products written in place into y
            torch = self.be.torch
            y, mul, mm = torch.empty_like(x), torch.mul, torch.bmm
        else:
            y, mul, mm = np.empty_like(x), np.multiply, np.matmul
        ns = self.n_scalar
        if scale is not None:
            mul(x[:ns], scale, out=y[:ns])
        elif ns:
            y[:ns] = x[:ns]
        for o, size, shape, M in stacks:
            mm(M, x[o:o + size].reshape(shape), out=y[o:o + size].reshape(shape))
        return y[self.inv]


class Gather:
    """new buffer = old buffer[index], onto the layout `target`; the old buffer has n entries."""

    def __init__(self, target: Target, index, n: int, be):
        self.target, self.index = target, to_index(be, index, n)

    def nbytes(self) -> int:
        return nbytes(self.index)

    def apply(self, data):
        return data[self.index]


def cache_info():
    return {"recoupling": RECOUPLING_CACHE.info(), "permutes": PERMUTE_CACHE.info(),
            "swaps": lru_info(_swap),
            "bend_coefficients": lru_info(_bend_coefficients), "bends": lru_info(_bend)}
