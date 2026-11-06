"""Symmetries for block-sparse tensors: sector labels and how they combine.

A `Symmetry` names the sectors (irreducible representations) a tensor leg can carry and says
how they fuse. Block tensors (`tnkit.blocks`) ask it four things: the trivial sector, the dual
of a sector, the fusion of two sectors, and the order in which sectors sit along a leg. Labels
are plain hashable host values (ints, or tuples of ints for products).

Conventions:
  * A leg with ``sign = +1`` enters a block's charge with its label ``a``; a leg with
    ``sign = -1`` with the dual ``dual(a)``. A block is allowed when its signed labels fuse to
    the tensor's total charge. Contracted legs carry the same labels with opposite signs.
  * Complex conjugation flips every sign and dualises the total charge.
  * Sectors along a leg are ordered by `sort_key`; this order fixes the block layout, so two
    symmetries with the same labels but different orders give different (equally valid) layouts.

`U1`, `Zn` and their products are abelian: each pair of sectors fuses to exactly one sector,
every sector has dimension 1, and the recoupling (F) and exchange (R) coefficients are 1 on
allowed fusions. `SU2` is not (``fusion = "simple"``): a x b can hold several sectors, each once,
and its coefficients and their conventions come from `tnkit.su2`. Block tensors accept only
unique fusion so far.

Provisional: this module's API may change before 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Optional, Tuple

import numpy as np

from . import su2

Label = Hashable


class Symmetry:
    """The interface. Subclasses implement `unit`, `dual`, `fuse`, `label` and `sort_key`, and set
    `fusion` ("unique": every fusion a x b has exactly one outcome, i.e. abelian; "simple": a x b
    may have several outcomes, each once)."""
    fusion: Optional[str] = None

    def unit(self) -> Label:
        raise NotImplementedError

    def dual(self, a: Label) -> Label:
        raise NotImplementedError

    def fuse(self, a: Label, b: Label) -> Label:
        """The one sector in a x b (abelian symmetries)."""
        raise NotImplementedError

    def label(self, a: Any) -> Label:
        """`a` as this symmetry's canonical label; raises if it is not a sector."""
        raise NotImplementedError

    def sort_key(self, a: Label):
        return a

    # --- derived ---
    def fusions(self, a: Label, b: Label) -> Tuple[Tuple[Label, int], ...]:
        """The sectors in a x b with their multiplicities."""
        return ((self.fuse(a, b), 1),)

    def total(self, labels: Iterable[Label], signs: Iterable[int]) -> Label:
        """The fusion of `labels`, each dualised where its sign is -1."""
        c = self.unit()
        for a, s in zip(labels, signs):
            c = self.fuse(c, a if s > 0 else self.dual(a))
        return c

    def dim(self, a: Label) -> int:
        return 1

    def F(self, a, b, c, d, e, f) -> float:
        """Recoupling coefficient ((a b)_e c)_d -> (a (b c)_f)_d: 1 if both trees exist, else 0."""
        ok = (e == self.fuse(a, b) and f == self.fuse(b, c) and d == self.fuse(e, c)
              and d == self.fuse(a, f))
        return 1.0 if ok else 0.0

    def R(self, a, b, c) -> float:
        """Exchange coefficient of a x b -> c: 1 if c is in a x b, else 0."""
        return 1.0 if c == self.fuse(a, b) else 0.0

    def cg(self, a, b, c) -> np.ndarray:
        """Clebsch-Gordan coefficients <a; b | c>, shape (dim a, dim b, dim c)."""
        return np.full((1, 1, 1), 1.0 if c == self.fuse(a, b) else 0.0)

    def frobenius_schur(self, a) -> int:
        """+1 or -1: the sign a sector picks up when its leg is bent twice. 1 for every abelian
        sector, self-dual or not."""
        return 1


@dataclass(frozen=True)
class U1(Symmetry):
    """Integer charges that add."""
    fusion = "unique"

    def unit(self):
        return 0

    def dual(self, a):
        return -a

    def fuse(self, a, b):
        return a + b

    def label(self, a):
        if isinstance(a, (bool, np.bool_)) or int(a) != a:
            raise ValueError(f"U(1) labels are integers, got {a!r}")
        return int(a)

    def total(self, labels, signs):
        return sum(s * a for a, s in zip(labels, signs))


@dataclass(frozen=True)
class Zn(Symmetry):
    """Charges 0..n-1 that add modulo n."""
    fusion = "unique"
    n: int

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"Z_n needs n >= 2, got {self.n}")

    def unit(self):
        return 0

    def dual(self, a):
        return (-a) % self.n

    def fuse(self, a, b):
        return (a + b) % self.n

    def label(self, a):
        if isinstance(a, (bool, np.bool_)) or int(a) != a or not 0 <= int(a) < self.n:
            raise ValueError(f"Z_{self.n} labels are 0..{self.n - 1}, got {a!r}")
        return int(a)


@dataclass(frozen=True)
class ProductSymmetry(Symmetry):
    """Several symmetries at once; labels are tuples with one entry per factor. Sectors sort
    lexicographically, the first factor most significant."""
    factors: Tuple[Symmetry, ...]

    def __post_init__(self):
        if len(self.factors) < 2:
            raise ValueError("a product needs at least two factors")
        if any(f.fusion not in ("unique", "simple") for f in self.factors):
            raise NotImplementedError("only multiplicity-free factors are supported")

    @property
    def fusion(self):
        return "unique" if all(f.fusion == "unique" for f in self.factors) else "simple"

    def unit(self):
        return tuple(f.unit() for f in self.factors)

    def dual(self, a):
        return tuple(f.dual(x) for f, x in zip(self.factors, a))

    def fuse(self, a, b):
        return tuple(f.fuse(x, y) for f, x, y in zip(self.factors, a, b))

    def label(self, a):
        a = tuple(a)
        if len(a) != len(self.factors):
            raise ValueError(f"labels of {self} have {len(self.factors)} entries, got {a!r}")
        return tuple(f.label(x) for f, x in zip(self.factors, a))

    def sort_key(self, a):
        return tuple(f.sort_key(x) for f, x in zip(self.factors, a))

    def fusions(self, a, b):
        out = [((), 1)]
        for f, x, y in zip(self.factors, a, b):
            out = [(c + (z,), n * m) for c, n in out for z, m in f.fusions(x, y)]
        return tuple(out)

    def dim(self, a):
        return int(np.prod([f.dim(x) for f, x in zip(self.factors, a)]))

    def F(self, a, b, c, d, e, f):
        return float(np.prod([s.F(*x) for s, *x in zip(self.factors, a, b, c, d, e, f)]))

    def R(self, a, b, c):
        return float(np.prod([s.R(*x) for s, *x in zip(self.factors, a, b, c)]))

    def cg(self, a, b, c):
        """The Kronecker product of the factors' arrays; flat index is factor-major (the first
        factor's index most significant), as in `numpy.kron`."""
        out = np.ones((1, 1, 1))
        for s, x, y, z in zip(self.factors, a, b, c):
            t = s.cg(x, y, z)
            out = np.einsum("abc,xyz->axbycz", out, t).reshape(
                out.shape[0] * t.shape[0], out.shape[1] * t.shape[1], out.shape[2] * t.shape[2])
        return out

    def frobenius_schur(self, a):
        return int(np.prod([f.frobenius_schur(x) for f, x in zip(self.factors, a)]))


@dataclass(frozen=True)
class SU2(Symmetry):
    """SU(2), sectors labelled by twice the spin: 0, 1 (spin 1/2), 2 (spin 1), ... Every sector
    is self-dual, and a x b = |a - b|, |a - b| + 2, ..., a + b, each once. Coefficients and
    their conventions (Condon-Shortley CG, F as 6j-symbols, bosonic R): `tnkit.su2`."""
    fusion = "simple"

    def unit(self):
        return 0

    def dual(self, a):
        return a

    def fuse(self, a, b):
        """The one sector in a x b; raises unless a or b is spin 0 (use `fusions`)."""
        if a == 0 or b == 0:
            return a + b
        raise ValueError(f"SU(2): {a} x {b} has {min(a, b) + 1} outcomes; use fusions()")

    def fusions(self, a, b):
        return tuple((c, 1) for c in range(abs(a - b), a + b + 1, 2))

    def label(self, a):
        if isinstance(a, (bool, np.bool_)) or int(a) != a or int(a) < 0:
            raise ValueError(f"SU(2) labels are twice the spin, integers >= 0; got {a!r}")
        return int(a)

    def dim(self, a):
        return a + 1

    def F(self, a, b, c, d, e, f):
        return su2.fsymbol(a, b, c, d, e, f)

    def R(self, a, b, c):
        return su2.rsymbol(a, b, c)

    def cg(self, a, b, c):
        return su2.cg_array(a, b, c)

    def frobenius_schur(self, a):
        return -1 if a % 2 else 1
