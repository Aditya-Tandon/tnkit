"""Fusion trees: how the sectors on several legs couple, pairwise from the left, to one total.

A splitting tree c -> u_1 x ... x u_n records the leaves `uncoupled`, which legs are dual
(`isdual`), the total `coupled`, the inner lines e_1, ..., e_(n-2) and one label per vertex.
Vertex i fuses (e_(i-1), u_(i+1)) -> e_i, with e_0 = u_1 and e_(n-1) = c; its label mu_i < N
picks one copy of e_i in that fusion (always 0 without multiplicity). A rank-1 tree has c = u_1,
a rank-0 tree c = unit. The fusion tree of a map's domain is the adjoint of a splitting tree.

`tree_tensor` gives the dense form, shape (dim u_1, ..., dim u_n, dim c): the chain of CG arrays
cg(e_(i-1), u_(i+1), e_i)[..., mu_i]. A dual leg's leaf is the sector that fuses, u, while the
leg's space is the dual of a = dual(u), in the dual basis; the tree carries conj(cup(a)) on that
leg, which maps the basis of u onto the dual basis of a. Tree tensors are isometries, and the
trees of fixed leaves and duality flags, over every coupled sector, form an orthonormal basis.

Provisional: not exported from `tnkit`; this module's API may change before 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ._bytecache import env_cache
from .symmetry import Label, Symmetry, cg_tensor, cup

TREE_CACHE = env_cache("TNKIT_SYMMETRIC_DENSE_TREE_CACHE_MB", 256)
"""Dense tree tensors, kept for dense conversion and tests only; ``TREE_CACHE.set_limit(n)``
bounds it to n bytes."""


@dataclass(frozen=True)
class FusionTree:
    uncoupled: Tuple[Label, ...]
    coupled: Label
    isdual: Tuple[bool, ...]
    innerlines: Tuple[Label, ...] = ()
    vertices: Tuple[int, ...] = ()

    def steps(self):
        """(left line, leaf, right line, vertex label) for each vertex, left to right."""
        u = self.uncoupled
        lines = (u[0],) + self.innerlines + (self.coupled,) if u else ()
        return tuple(zip(lines[:-1], u[1:], lines[1:], self.vertices))


def check(sym: Symmetry, tree: FusionTree) -> None:
    """Raise ValueError unless `tree` is a valid splitting tree of `sym`."""
    n = len(tree.uncoupled)
    if (len(tree.isdual) != n or len(tree.innerlines) != max(n - 2, 0)
            or len(tree.vertices) != max(n - 1, 0)):
        raise ValueError(f"{tree}: wrong number of flags, inner lines or vertices")
    if n == 0 and tree.coupled != sym.unit() or n == 1 and tree.coupled != tree.uncoupled[0]:
        raise ValueError(f"{tree}: coupled sector does not match its leaves")
    for left, u, right, mu in tree.steps():
        if not 0 <= mu < sym.multiplicity(left, u, right):
            raise ValueError(f"{tree}: no vertex {mu} in {left} x {u} -> {right}")


def fusion_trees(sym: Symmetry, uncoupled: Sequence, isdual: Optional[Sequence[bool]] = None,
                 coupled=None) -> Tuple[FusionTree, ...]:
    """Every splitting tree with these leaves (and this coupled sector, if given). Ordered by
    coupled sector (`sort_key`), then inner lines (`sort_key`), then vertex labels."""
    u = tuple(sym.label(a) for a in uncoupled)
    flags = (False,) * len(u) if isdual is None else tuple(bool(x) for x in isdual)
    if len(flags) != len(u):
        raise ValueError(f"{len(u)} leaves but {len(flags)} duality flags")
    coupled = None if coupled is None else sym.label(coupled)
    if not u:
        paths = [((sym.unit(),), ())]
    else:
        paths = [((u[0],), ())]
        for a in u[1:]:
            paths = [(lines + (c,), mus + (mu,)) for lines, mus in paths
                     for c, n in sym.fusions(lines[-1], a) for mu in range(n)]
    trees = [FusionTree(u, lines[-1], flags, lines[1:-1], mus) for lines, mus in paths
             if coupled is None or lines[-1] == coupled]
    key = sym.sort_key
    return tuple(sorted(trees, key=lambda t: (key(t.coupled), tuple(map(key, t.innerlines)),
                                              t.vertices)))


def tree_tensor(sym: Symmetry, tree: FusionTree) -> np.ndarray:
    """The dense splitting tensor of `tree`, shape (dim u_1, ..., dim u_n, dim c), read-only."""
    return TREE_CACHE.get((sym, tree), lambda: _tree_tensor(sym, tree))


def _tree_tensor(sym, tree):
    check(sym, tree)
    if not tree.uncoupled:
        t = np.ones(1)
    else:
        t = np.eye(sym.dim(tree.uncoupled[0]))
        for left, u, right, mu in tree.steps():
            t = np.tensordot(t, cg_tensor(sym, left, u, right)[..., mu], axes=(-1, 0))
        for i, (u, dual) in enumerate(zip(tree.uncoupled, tree.isdual)):
            if dual:
                w = cup(sym, sym.dual(u)).conj()
                t = np.moveaxis(np.tensordot(w, t, axes=(1, i)), 0, i)
    t.flags.writeable = False
    return t
