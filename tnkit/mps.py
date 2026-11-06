"""The MPS type: a list of rank-3 site tensors that knows what its indices mean.

An MPS in tnkit has always been a bare Python list of rank-3 arrays, and it still
is -- `MPS` subclasses `list`, so `isinstance(mps, list)` is True, indexing,
slicing, iteration, `len`, `zip`, unpacking and in-place mutation all behave
exactly as before, and every script written against the bare-list convention keeps
working with no change. What the type adds is a home for three things that were
scattered: the index convention (previously a sentence in the `tnkit.dmrg` module
docstring), the host round-trip (`observables.to_numpy`), and the entanglement
profile (`observables.all_entropies`).

Nothing here computes anything new. Both methods delegate to `tnkit.observables`,
which remains the place the arithmetic lives.
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

    INDEX CONVENTION (the whole contract, and the reason this type exists)::

        M[i] : (left_bond, phys, right_bond) = (a, s, b)

    with ``M[i].shape[2] == M[i+1].shape[0]`` at every bond, and the outermost
    bonds of dimension 1 for the open-boundary chains tnkit solves. The physical
    dimension is ``d = M[i].shape[1]`` (2 for the TFIM). For reference, the MPO
    that a solver takes alongside it uses a *different* order::

        W[i] : (mpo_left, mpo_right, phys_out, phys_in) = (wl, wr, s, s')

    -- so an MPO tensor is not an MPS tensor with an extra leg, and mixing the two
    orders is the bug class this type is named after. Both conventions were
    previously documented only in module docstrings (`tnkit.dmrg` for the MPS,
    `tnkit.models` for the MPO).

    GAUGE. `center` is the site of the orthogonality centre when it is known, else
    None: every tensor left of it is left-canonical, every tensor right of it is
    right-canonical, to the isometry accuracy of the truncation that produced the
    state (exact for the SVD and Gram paths, approximate for ``trunc='density'``). The solvers,
    `right_canonicalize`, `compress` and `tnkit.window.build_window` set it; any
    change to the list's contents (item assignment, append, ...) resets it to None,
    since the state may no longer be in that gauge. It is a record, not a guarantee:
    writing into a tensor in place (``mps[3][...] *= 2``, ``mps[3].mul_(2)``) is
    invisible to it. The observables never rely on it; they canonicalise their own copy.

    Tensors may live on any backend. `dmrg(..., backend="torch-cuda")` returns CUDA
    tensors inside an `MPS`; `.entropies()` measures them on the device and
    `.to_numpy()` brings the state itself home. Both refuse block tensors (symmetric
    DMRG); `tnkit.block_backend.to_dense_mps` converts those first.

    Examples
    --------
    >>> from tnkit import dmrg                         # doctest: +SKIP
    >>> from tnkit.models import tfim_mpo              # doctest: +SKIP
    >>> E, mps, info = dmrg(tfim_mpo(32, 1.0), chi=32) # doctest: +SKIP
    >>> isinstance(mps, list), len(mps), mps[0].shape  # doctest: +SKIP
    (True, 32, (1, 2, 2))
    >>> S = mps.entropies()                            # every bond, one sweep
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
        """Host copy of this state as contiguous numpy arrays. Never mutates self.

        A no-op cost on the numpy backend (a copy of O(N chi^2) numbers, far less
        than one DMRG sweep moves) and one device-to-host transfer per site on a
        torch backend. Delegates to `tnkit.observables.to_numpy`.
        """
        from .observables import to_numpy
        return MPS(to_numpy(self), center=self.center)

    def entropies(self) -> np.ndarray:
        """Von-Neumann entropy at every bond, from ONE sweep. Returns array[N-1].

        Delegates to `tnkit.observables.all_entropies`: `entanglement_entropy` in a
        loop re-canonicalises and re-sweeps per call and so costs O(N^2 chi^3),
        which is hours at N = 600 where this is 33 s.
        """
        from .observables import all_entropies
        return all_entropies(self)

    def __repr__(self) -> str:
        """Shape summary, not 2 N chi^2 numbers.

        `list.__repr__` on a 600-site state prints every tensor, which is useless
        in a REPL and slow in a log. The elements themselves are untouched: this
        changes only what `repr(mps)`/`print(mps)` show, never what they contain.
        """
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
