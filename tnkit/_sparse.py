"""The two kinds of sparse tensor the drivers take, block tensors (`tnkit.blocks`) and symmetric
tensors (`tnkit.symmetric_tensor`): which module factorises one, its symmetry, the number of
states on a bond as a cap counts them."""
from __future__ import annotations

from .symmetric_tensor import SymmetricTensor


def is_symmetric(x) -> bool:
    return isinstance(x, SymmetricTensor)


def linalg_of(x):
    """The module whose `truncated_svd` and `qr` factorise the sparse tensor `x`."""
    if isinstance(x, SymmetricTensor):
        from . import symmetric_linalg
        return symmetric_linalg
    from . import blocks
    return blocks


def symmetry_of(t):
    return t.sym if isinstance(t, SymmetricTensor) else t.struct.sym


def bond_states(t, axis: int) -> int:
    """The size of leg `axis` in the units of a bond cap: multiplets on a symmetric tensor,
    dense states otherwise."""
    if isinstance(t, SymmetricTensor):
        return sum((t.codomain + t.domain)[axis].degeneracies)
    return int(t.shape[axis])


def is_empty(t) -> bool:
    """True for a sparse tensor with no allowed entries."""
    if isinstance(t, SymmetricTensor):
        return t.layout.size == 0
    return not t.struct.blocks

