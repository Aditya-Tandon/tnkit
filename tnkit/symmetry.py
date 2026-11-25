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
and its coefficients and their conventions come from `tnkit.su2`. Under ``fusion = "generic"`` a
sector may appear N > 1 times in a x b, and each vertex a x b -> c carries a label mu < N. Block
tensors accept only unique fusion so far.

Coefficients with multiplicity (`cg_tensor`, `f_tensor` and `r_matrix` give arrays for every
fusion style; an axis of length 0 marks a fusion that is not allowed):
  * ``cg(a, b, c)[x, y, z, mu]`` = <a x; b y | c z, mu>, the splitting map c -> a x b. Under
    unique and simple fusion `cg` has no mu axis. The maps intertwine the group action and the
    coupling matrix of a x b, columns (c, mu, z), is unitary.
  * ``F(a, b, c, d, e, f)[mu, nu, kappa, lam]``: the coefficients of
    |((a b)_e^mu c)_d^nu> = sum F |(a (b c)_f^kappa)_d^lam>, axes in vertex order (a b -> e),
    (e c -> d), (b c -> f), (a f -> d). A float without multiplicity.
  * ``R(a, b, c)[mu, nu]``: cg(a, b, c)[x, y, :, mu] = sum_nu R[mu, nu] cg(b, a, c)[y, x, :, nu].
  * Products flatten the multiplicity index factor-major, as states are flattened in `cg`.
  * The cup of sector a is ``Z_a = sqrt(dim a) cg(a, dual(a), unit)``, a unitary matrix.

Provisional: this module's API may change before 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Optional, Tuple

import numpy as np

from . import su2

Label = Hashable
_STYLES = ("unique", "simple", "generic")


class Symmetry:
    """The interface. Subclasses implement `unit`, `dual`, `fuse`, `label` and `sort_key`, and set
    `fusion` ("unique": every fusion a x b has exactly one outcome, i.e. abelian; "simple": a x b
    may have several outcomes, each once; "generic": an outcome may appear several times, and
    `fusions`, `dim`, `cg`, `F` and `R` must be overridden)."""
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

    def multiplicity(self, a: Label, b: Label, c: Label) -> int:
        """How often c appears in a x b."""
        return dict(self.fusions(a, b)).get(c, 0)

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
        """kappa_a = dim a F(a, dual(a), a, a, unit, unit). For a self-dual sector it is fixed by
        the group, +1 or -1: the sign its leg picks up when bent twice. Otherwise it is a phase
        set by the gauge of the cups, conj R(dual(a), a, unit); 1 in the gauge used here."""
        return 1

    @property
    def trivial_coefficients(self) -> bool:
        """Unique fusion with every dimension, F, R and CG coefficient equal to 1 (the defaults
        of this class, as for `U1` and `Zn`). Such a symmetry, or such a factor of a product,
        never changes a recoupling matrix."""
        return self.fusion == "unique" and not _overrides(self, Symmetry)


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
        if any(f.fusion not in _STYLES for f in self.factors):
            raise NotImplementedError("every factor needs a fusion style: " + ", ".join(_STYLES))

    @property
    def fusion(self):
        return max((f.fusion for f in self.factors), key=_STYLES.index)

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
        if self.fusion == "generic":
            return _kron([f_tensor(s, *x) for s, *x in zip(self.factors, a, b, c, d, e, f)])
        return _scalar(np.prod([s.F(*x) for s, *x in zip(self.factors, a, b, c, d, e, f)]))

    def R(self, a, b, c):
        if self.fusion == "generic":
            return _kron([r_matrix(s, *x) for s, *x in zip(self.factors, a, b, c)])
        return _scalar(np.prod([s.R(*x) for s, *x in zip(self.factors, a, b, c)]))

    def cg(self, a, b, c):
        """The Kronecker product of the factors' arrays; flat index is factor-major (the first
        factor's index most significant), as in `numpy.kron`. The multiplicity axis, under generic
        fusion, likewise."""
        if self.fusion == "generic":
            return _kron([cg_tensor(s, *x) for s, *x in zip(self.factors, a, b, c)])
        out = np.ones((1, 1, 1))
        for s, x, y, z in zip(self.factors, a, b, c):
            t = s.cg(x, y, z)
            out = np.einsum("abc,xyz->axbycz", out, t).reshape(
                out.shape[0] * t.shape[0], out.shape[1] * t.shape[1], out.shape[2] * t.shape[2])
        return out

    def frobenius_schur(self, a):
        return _scalar(np.prod([f.frobenius_schur(x) for f, x in zip(self.factors, a)]))

    @property
    def trivial_coefficients(self):
        return _plain_product(self) and all(f.trivial_coefficients for f in self.factors)


_COEFFICIENTS = ("dim", "fusions", "multiplicity", "F", "R", "cg", "frobenius_schur")


def _overrides(sym, base) -> bool:
    """Whether the class of `sym` redefines a fusion rule or coefficient of `base`."""
    return any(getattr(type(sym), m) is not getattr(base, m) for m in _COEFFICIENTS)


def _plain_product(sym) -> bool:
    """A `ProductSymmetry` whose fusion rules and coefficients are those of its factors: no
    subclass redefines them."""
    return isinstance(sym, ProductSymmetry) and not _overrides(sym, ProductSymmetry)


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


# --------------------------------------------------------------------------- arrays, any style
def _scalar(x):
    """A product of coefficients: a float, or a complex number when the factors are complex."""
    return complex(x) if np.iscomplexobj(x) else float(x)


def _kron(arrays):
    """Kronecker product of equal-rank arrays, axis by axis, the first array most significant."""
    out = np.ones((1,) * arrays[0].ndim)
    for t in arrays:
        n = out.ndim
        out = np.tensordot(out, t, axes=0).transpose(
            [i for k in range(n) for i in (k, n + k)]).reshape(
            [x * y for x, y in zip(out.shape, t.shape)])
    return out


def cg_tensor(sym: Symmetry, a, b, c) -> np.ndarray:
    """`sym.cg(a, b, c)` with its multiplicity axis: shape (dim a, dim b, dim c, N)."""
    if sym.fusion == "generic":
        return sym.cg(a, b, c)
    x = sym.cg(a, b, c)
    return x[..., None] if sym.multiplicity(a, b, c) else np.zeros(x.shape + (0,))


def f_tensor(sym: Symmetry, a, b, c, d, e, f) -> np.ndarray:
    """`sym.F(a, b, c, d, e, f)` as an array of shape (N_ab^e, N_ec^d, N_bc^f, N_af^d)."""
    if sym.fusion == "generic":
        return sym.F(a, b, c, d, e, f)
    n = (sym.multiplicity(a, b, e), sym.multiplicity(e, c, d), sym.multiplicity(b, c, f),
         sym.multiplicity(a, f, d))
    return np.full(n, sym.F(a, b, c, d, e, f))


def r_matrix(sym: Symmetry, a, b, c) -> np.ndarray:
    """`sym.R(a, b, c)` as an array of shape (N_ab^c, N_ba^c)."""
    if sym.fusion == "generic":
        return sym.R(a, b, c)
    return np.full((sym.multiplicity(a, b, c), sym.multiplicity(b, a, c)), sym.R(a, b, c))


def cup(sym: Symmetry, a) -> np.ndarray:
    """Z_a = sqrt(dim a) cg(a, dual(a), unit): the invariant pairing of a with its dual, unitary.
    The conjugate of Z_a maps the basis of dual(a) onto the dual basis of a."""
    return np.sqrt(sym.dim(a)) * cg_tensor(sym, a, sym.dual(a), sym.unit())[:, :, 0, 0]


def f_from_cg(sym: Symmetry, a, b, c, d, e, f) -> np.ndarray:
    """F(a, b, c, d, e, f) from overlaps of CG trees at the first state of d; shape as
    `f_tensor`. The source of F for symmetries defined by their CG arrays."""
    left = np.einsum("xyeu,ezv->xyzuv", cg_tensor(sym, a, b, e), cg_tensor(sym, e, c, d)[:, :, 0])
    right = np.einsum("yzfk,xfl->xyzkl", cg_tensor(sym, b, c, f), cg_tensor(sym, a, f, d)[:, :, 0])
    return np.einsum("xyzkl,xyzuv->uvkl", right.conj(), left)


def r_from_cg(sym: Symmetry, a, b, c) -> np.ndarray:
    """R(a, b, c) from overlaps of CG arrays at the first state of c; shape as `r_matrix`."""
    return np.einsum("yxv,xyu->uv", cg_tensor(sym, b, a, c)[:, :, 0].conj(),
                     cg_tensor(sym, a, b, c)[:, :, 0])
