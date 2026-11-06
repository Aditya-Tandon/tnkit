"""Fuse groups of tensor legs into single legs, and split them back.

`fuse` permutes the legs so that each group is contiguous, then reshapes each group into one
leg. It returns the fused array and a `FuseRecord` holding everything needed to undo it::

    theta_m, rec = fuse(theta, [(0, 1), (2, 3)], be)   # (a, s1, s2, b) -> (a*s1, s2*b)
    theta2 = split(theta_m, rec, be)                   # back to (a, s1, s2, b)

Arrays in and out are the backend's own (numpy arrays or torch tensors). Reshaping a
permuted view copies whenever the view is not contiguous, so `rec.copied` says whether
`fuse` allocated; with the groups already in leg order and a contiguous input it never does.

After a factorisation some fused legs survive at their positions while others are replaced,
e.g. ``U`` of shape ``(a*s1, k)`` from the SVD of ``theta_m``. ``split(U, rec, be,
legs=(0,))`` restores the surviving group and keeps every other leg as it is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FuseRecord:
    """How `fuse` built its output: the unfused `shape` and the `groups` of original legs
    (fused leg `g` is ``groups[g]``; `perm` is the groups flattened). `copied` reports
    whether `fuse` allocated a new buffer (not whether the result is contiguous); it is
    not part of equality. `tnkit.blocks.BlockFuseRecord` extends it with the sector maps
    that block-sparse legs need."""
    shape: Tuple[int, ...]
    groups: Tuple[Tuple[int, ...], ...]
    copied: bool = field(default=False, compare=False)

    def __post_init__(self):
        flat = [a for g in self.groups for a in g]
        if any(not g for g in self.groups) or sorted(flat) != list(range(len(self.shape))):
            raise ValueError(f"groups {self.groups} do not partition the {len(self.shape)} "
                             f"legs of a tensor of shape {self.shape}")

    @property
    def perm(self) -> Tuple[int, ...]:
        """The leg permutation applied before reshaping."""
        return tuple(a for g in self.groups for a in g)

    def dims(self, g: int) -> Tuple[int, ...]:
        """Unfused dimensions of fused leg `g`, in group order."""
        return tuple(self.shape[a] for a in self.groups[g])


def _shares_memory(a: Any, b: Any) -> bool:
    if hasattr(a, "untyped_storage"):
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    return bool(np.may_share_memory(a, b))


def fuse(x: Any, groups: Sequence[Sequence[int]], be: Any) -> Tuple[Any, FuseRecord]:
    """Fuse each group of legs of `x` into one leg; the fused legs come out in group order.

    `groups` must partition ``range(x.ndim)``; each group is a non-empty sequence of legs,
    fused in the order given. Returns ``(y, record)``. A block-sparse `x` (`tnkit.blocks`)
    fuses by sector and returns a `BlockFuseRecord`.
    """
    if getattr(x, "is_block", False):
        return x.fuse(groups)
    rec = FuseRecord(tuple(int(n) for n in x.shape),
                     tuple(tuple(int(a) for a in g) for g in groups))
    perm = rec.perm
    xt = x if perm == tuple(range(len(perm))) else be.transpose(x, perm)
    y = xt.reshape(tuple(math.prod(rec.dims(g)) for g in range(len(rec.groups))))
    return y, FuseRecord(rec.shape, rec.groups, copied=not _shares_memory(x, y))


def split(y: Any, rec: FuseRecord, be: Any, legs: Optional[Sequence[int]] = None) -> Any:
    """Undo `fuse`, after checking that the leg sizes of `y` match `rec`.

    With `legs` None, `y` must have exactly the fused shape and the result has
    ``rec.shape``. With `legs` given (indices of fused legs), only those are split, each at
    its own position, and every other leg of `y` is kept whatever its size; this needs a
    fuse that did not permute. The check compares sizes only: a record from another tensor
    with the same fused sizes passes.
    """
    if getattr(y, "is_block", False):
        return y.split(rec, legs)
    if getattr(rec, "fused", None) is not None:
        raise TypeError("a BlockFuseRecord splits only block tensors: a densified block tensor "
                        "has its fused legs in sector order, not in the order `fuse` reshapes")
    n = len(rec.groups)
    if y.ndim != n:
        raise ValueError(f"expected {n} fused legs, got shape {tuple(y.shape)}")
    todo = set(range(n)) if legs is None else {int(g) for g in legs}
    if not todo <= set(range(n)):
        raise ValueError(f"legs {sorted(todo)} out of range for {n} fused legs")
    for g in todo:
        if int(y.shape[g]) != math.prod(rec.dims(g)):
            raise ValueError(f"leg {g} has size {int(y.shape[g])}, but fused group {g} is "
                             f"{rec.dims(g)}")
    identity = rec.perm == tuple(range(len(rec.shape)))
    if legs is None:
        xt = y.reshape(tuple(rec.shape[a] for a in rec.perm))
        return xt if identity else be.transpose(xt, tuple(int(i) for i in np.argsort(rec.perm)))
    if not identity:
        raise ValueError(f"a partial split needs an unpermuted fuse; this one used perm "
                         f"{rec.perm}")
    out = []
    for g, size in enumerate(y.shape):
        out.extend(rec.dims(g) if g in todo else (int(size),))
    return y.reshape(tuple(out))
