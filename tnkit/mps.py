"""The MPS type: a list of rank-3 site tensors with a documented index order.

`MPS` subclasses `list`, so indexing, slicing, iteration, `len`, unpacking and in-place
mutation behave as on a list, and every function that takes an MPS also takes a plain list.
The type adds the index convention, the host copy (`observables.to_numpy`) and the
entanglement profile (`observables.all_entropies`); the arithmetic stays in
`tnkit.observables`.
"""
from __future__ import annotations
import functools
import operator
from typing import Any, MutableSequence, Optional, Sequence

import numpy as np

# A site tensor is whatever array type the backend produced -- numpy on the CPU
# path, torch (CPU/CUDA/MPS) on the device paths. tnkit deliberately never imports
# torch to name that type (see `observables._backend_for`), so the alias is `Any`
# and the duck-typed backend detection is what actually decides.
Tensor = Any
#: Anything the library accepts where an MPS is expected: an `MPS` or a plain list.
MPSLike = Sequence[Tensor]
#: An MPS that a routine will MUTATE IN PLACE. Distinct from `MPSLike` on purpose:
#: `Sequence` promises read-only access, and `right_canonicalize` assigns into its
#: argument, which a type checker is right to object to.
MutableMPS = MutableSequence[Tensor]


class MPS(list):
    """A matrix-product state: one rank-3 tensor per site, in tnkit's index order.

    Index convention::

        M[i] : (left_bond, phys, right_bond) = (a, s, b)

    with ``M[i].shape[2] == M[i+1].shape[0]`` at every bond, outer bonds of dimension 1
    (open boundaries), and physical dimension ``d = M[i].shape[1]``. Sites are numbered from
    0. The MPO a solver takes alongside it uses a different order::

        W[i] : (mpo_left, mpo_right, phys_out, phys_in) = (wl, wr, s, s')

    so an MPO tensor is not an MPS tensor with an extra leg.

    Parameters
    ----------
    tensors : sequence of tensors, default ()
        The site tensors, on any backend (numpy, torch; block or symmetric tensors too).
    center : int or None, default None
        The orthogonality centre, ``0 <= center < len(tensors)``, if known: every tensor left
        of it is left-canonical and every tensor right of it right-canonical, to the isometry
        accuracy of the truncation that produced the state.

    Notes
    -----
    The solvers, `right_canonicalize`, `compress` and `tnkit.window.build_window` set
    `center`. Any change to the list's contents (item assignment, append, ...) resets it to
    None, since the state may no longer be in that gauge. Writing into a tensor in place
    (``mps[3][...] *= 2``) is not seen. The observables never rely on it; they canonicalise
    their own copy.

    ``dmrg(..., backend="torch-cuda")`` returns CUDA tensors inside an `MPS`;
    `.entropies()` measures them on the device and `.to_numpy()` copies the state to the
    host. Both refuse block and symmetric tensors: `tnkit.block_backend.to_dense_mps` and
    `tnkit.symmetric_backend.to_dense_mps` convert those first.

    Examples
    --------
    >>> from tnkit import dmrg                         # doctest: +SKIP
    >>> from tnkit.models import tfim_mpo              # doctest: +SKIP
    >>> E, mps, info = dmrg(tfim_mpo(32, 1.0), chi=32) # doctest: +SKIP
    >>> isinstance(mps, list), len(mps), mps[0].shape  # doctest: +SKIP
    (True, 32, (1, 2, 2))
    >>> S = mps.entropies()                            # doctest: +SKIP
    """

    __slots__ = ("_center",)

    def __init__(self, tensors: Sequence[Tensor] = (), center: Optional[int] = None):
        super().__init__(tensors)
        self.center = center

    @property
    def center(self) -> Optional[int]:
        """Site of the orthogonality centre, or None when unknown."""
        return getattr(self, "_center", None)

    @center.setter
    def center(self, site: Optional[int]) -> None:
        if site is not None:
            site = operator.index(site)
            if not 0 <= site < len(self):
                raise ValueError(f"center {site} outside 0..{len(self) - 1}")
        self._center = site

    def __reduce__(self):
        return MPS, (list(self), self.center)

    def to_numpy(self) -> "MPS":
        """Host copy of this state as contiguous numpy arrays, as an `MPS` with the same
        `center`. Never mutates self; one device-to-host transfer per site on torch."""
        from .observables import to_numpy
        return MPS(to_numpy(self), center=self.center)

    def entropies(self) -> np.ndarray:
        """Von Neumann entropy (natural log) of every bond, from one sweep.

        Returns an array of length N - 1; entry i is the bond between sites i and i + 1. One
        sweep costs O(N chi^3), where calling `entanglement_entropy` per bond costs
        O(N^2 chi^3). Delegates to `tnkit.observables.all_entropies`.
        """
        from .observables import all_entropies
        return all_entropies(self)

    def __repr__(self) -> str:
        """A shape summary instead of every tensor; the elements are untouched."""
        if not self:
            return "MPS(empty)"
        try:
            d = int(self[0].shape[1])
            chi = max(int(A.shape[2]) for A in self[:-1]) if len(self) > 1 else 1
            where = type(self[0]).__module__.split(".")[0]
            return f"MPS(N={len(self)}, d={d}, chi_max={chi}, tensors={where})"
        except Exception:                                # pragma: no cover
            return f"MPS(N={len(self)})"


def _resetting(name: str):
    base = getattr(list, name)

    @functools.wraps(base)
    def method(self, *args, **kwargs):
        self._center = None
        return base(self, *args, **kwargs)
    return method


for _name in ("__setitem__", "__delitem__", "__iadd__", "__imul__", "append", "extend",
              "insert", "pop", "remove", "clear", "reverse", "sort"):
    setattr(MPS, _name, _resetting(_name))
del _name


def as_mps(tensors: MPSLike) -> MPS:
    """Wrap a list of site tensors as an `MPS` without copying the tensors.

    Cheap and idempotent: the container is new, the tensors are shared. Use it on a
    state built by hand or returned by a routine that still hands back a bare list.
    """
    return tensors if isinstance(tensors, MPS) else MPS(tensors)


__all__ = ["MPS", "MPSLike", "MutableMPS", "Tensor", "as_mps"]
