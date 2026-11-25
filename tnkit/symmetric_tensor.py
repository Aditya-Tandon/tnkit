"""Tensors that commute with a symmetry, abelian or not, stored as one matrix per coupled sector.

A `SymmetricTensor` is a map from its domain W_1 x ... x W_m to its codomain V_1 x ... x V_n that
commutes with the group. Its dense array has axes (codomain legs, domain legs) and equals

    sum_c sum_(s, f) t_c[(s, i), (f, j)] (e_i x e_j) x sum_w X_s[..., w] conj(X_f[..., w]),

with X_s and X_f splitting trees (`tnkit.fusion_trees.tree_tensor`) of the codomain and of the
domain, both with coupled sector c. The tensor stores the reduced matrices t_c.

Legs are `Space`s: sectors in `sort_key` order, the degeneracy of each, and a dual flag. A plain
leg's dense space is V = sum_a C^(n_a) x V_a; a dual leg's is the dual space, in the dual basis,
so a plain and a dual leg with the same sectors contract index for index. On a dense leg each
sector takes n_a dim(a) consecutive indices, i dim(a) + m for degeneracy index i and multiplet
state m. The dense array is invariant under D on plain codomain legs and conj(D) on dual ones,
and the reverse on domain legs.

Rows of t_c run over leaf tuples (one sector per leg, in lexicographic order of the sectors'
positions, the first leg most significant), then over the trees of that tuple in `fusion_trees`
order, then over the degeneracy indices row-major. Columns do the same for the domain. All blocks
share one flat buffer of the backend's array type, in `sort_key` order of c. Tree tensors are
isometries, so `vdot(a, b) = sum_c dim(c) Tr(a_c^dag b_c)` is the dense inner product.

Tensors are invariant. A tensor of total charge q is one with an extra plain domain leg holding
only q, once.

Recoupling. `permute` reorders the legs and moves them between codomain and domain (a leg that
changes side is dualised, and its dense index is unchanged). For one sector on every leg, the old
tree pairs are a fixed linear combination of the new ones; that recoupling matrix is built from F,
R, CG arrays of three legs and cups, never from dense trees, and is cached on the host by the
sectors, flags and permutation alone (`RECOUPLING_CACHE`). In a product that takes its
coefficients from its factors, only the factors with nontrivial coefficients enter: U(1) x SU(2)
caches one matrix per assignment of spins, shared by every charge. What a permutation needs beyond
the degeneracies (tree pairs, their grouping and matrices) is cached by sectors and dual flags
(`PERMUTE_CACHE`). Buffer positions depend on the degeneracies too; they are computed in one pass
of NumPy array operations per layout (the space of a tensor) and device, and kept with the matrices
in `INDEX_CACHE` (on the host) or `DEVICE_INDEX_CACHE` (on other devices), until the layout dies or
the cache's byte limit evicts them. A map casts its matrices once to each dtype it meets, so real
and complex tensors share it, and the cache counts each cast. A layout lives while a
tensor uses it, while the map that created it is cached, or while it is among the layouts last
mapped from or made by `contract` (`RECENT_LAYOUTS`, bounded by entries and bytes).
`contract` permutes both tensors so that the contracted legs are the first one's domain and the
second one's codomain, then multiplies the blocks of each coupled sector into the result's
buffer, with one batched product for blocks of one shape (`_Products`). `norm` and `vdot` are one
reduction over the buffer. `fuse` and `split` replace legs by the single leg of their coupled
sectors and back. `cache_info()` reports every cache.

Provisional: not exported from `tnkit`; this module's API may change before 1.0.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import _recoupling, _symmetric_layout
from ._bytecache import Growing, env_cache, nbytes
from ._recoupling import RECOUPLING_CACHE, Gather, Recoupler
from ._symmetric_layout import (DEVICE_INDEX_CACHE, INDEX_CACHE, RECENT_LAYOUTS, STRUCTURE_CACHE,
                                Layout, Side, Target, backend_key, layout, to_index)
from .backend import NumpyBackend
from .legs import FuseRecord
from .fusion_trees import TREE_CACHE, FusionTree, tree_tensor
from .symmetry import Label, Symmetry

# dense tree pairs, for dense conversion only
DENSE_PAIR_CACHE = env_cache("TNKIT_SYMMETRIC_DENSE_PAIR_CACHE_MB", 256)


def cache_info() -> Dict[str, Dict[str, int]]:
    """Entries, bytes, limits, hits, misses and evictions of each cache. ``layouts``: the live
    layouts and the bytes of their tree positions; ``trees``: the `FusionTree` objects built for
    the cached tree structures (no limit beyond those structures'); ``label_tables``: the label
    ids and fusion table of each symmetry, kept for the process; the caches of elementary moves
    and fusions are bounded by their entry counts. ``limit_source`` says where a limit came
    from: ``default``, ``env``, ``set`` or ``default-device``."""
    return {**_recoupling.cache_info(), **_symmetric_layout.cache_info(),
            "dense_trees": TREE_CACHE.info(), "dense_pairs": DENSE_PAIR_CACHE.info()}


@dataclass(frozen=True)
class Space:
    """One leg: the sectors of a space V in `sort_key` order, the degeneracy of each, and whether
    the leg is V or its dual."""
    sym: Symmetry
    sectors: Tuple[Label, ...]
    degeneracies: Tuple[int, ...]
    isdual: bool = False

    def __post_init__(self):
        object.__setattr__(self, "sectors", tuple(self.sectors))
        object.__setattr__(self, "degeneracies", tuple(int(n) for n in self.degeneracies))
        object.__setattr__(self, "isdual", bool(self.isdual))
        if (list(self.sectors) != sorted(set(self.sectors), key=self.sym.sort_key)
                or len(self.degeneracies) != len(self.sectors)
                or any(n < 1 for n in self.degeneracies)
                or any(self.sym.label(a) != a for a in self.sectors)):
            raise ValueError(f"invalid space {self}")

    @cached_property
    def dim(self) -> int:
        """The dense size of the leg."""
        return sum(n * self.sym.dim(a) for a, n in zip(self.sectors, self.degeneracies))

    def dual(self) -> "Space":
        return Space(self.sym, self.sectors, self.degeneracies, not self.isdual)

    def leaf(self, a) -> Label:
        """The sector that fuses in a tree: `a`, or its dual on a dual leg."""
        return self.sym.dual(a) if self.isdual else a

    @cached_property
    def offsets(self) -> Tuple[int, ...]:
        """Where each sector starts on the dense leg."""
        sizes = [n * self.sym.dim(a) for a, n in zip(self.sectors, self.degeneracies)]
        return tuple(np.cumsum([0] + sizes[:-1]).tolist())


@dataclass(frozen=True)
class SymmetricFuseRecord(FuseRecord):
    """`tnkit.legs.FuseRecord` of a symmetric tensor: also the spaces of each group as they sit
    on the codomain (group 0) and the domain (group 1) before fusing."""
    spaces: Tuple[Tuple[Space, ...], ...] = field(default=(), compare=False)


def _stacked(sym, trees: Tuple[FusionTree, ...]) -> np.ndarray:
    """The tree tensors of `trees` (one coupled sector), shape (trees, leaf states, dim c)."""
    return np.stack([tree_tensor(sym, t).reshape(-1, sym.dim(t.coupled)) for t in trees])


def _pairs(sym, strees, ftrees) -> np.ndarray:
    """sum_w X_s[a, w] conj(X_f[b, w]) for every pair, shape (s, a, f, b)."""
    return DENSE_PAIR_CACHE.get((sym, strees, ftrees), lambda: np.einsum(
        "saw,fbw->safb", _stacked(sym, strees), _stacked(sym, ftrees).conj()))


def _interleave(n_legs, n_rows):
    """The axis order taking (row degs, row dims, col degs, col dims) to (deg, dim) per leg."""
    n, m = n_rows, n_legs - n_rows
    return ([a for k in range(n) for a in (k, n + k)]
            + [a for k in range(m) for a in (2 * n + k, 2 * n + m + k)])


def _permute(be, x, perm):
    return x if list(perm) == sorted(perm) else be.transpose(x, tuple(perm))


def _is_scalar(c) -> bool:
    return isinstance(c, numbers.Number) or getattr(c, "ndim", None) == 0


def _sym_of(spaces, sym):
    syms = {V.sym for V in spaces} | ({sym} if sym is not None else set())
    if len(syms) != 1:
        raise ValueError("the spaces need one common symmetry (pass `sym` for a tensor "
                         "without legs)")
    return syms.pop()


# --------------------------------------------------------------------------- the tensor
class SymmetricTensor:
    """An invariant map from `domain` to `codomain`: the reduced blocks in one flat buffer of the
    backend `be`. Operations return new tensors, which may share `data`."""
    __slots__ = ("sym", "codomain", "domain", "data", "be", "layout")
    __array_ufunc__ = None                       # numpy scalars defer to the operators below

    def __init__(self, codomain, domain, data, be, sym=None):
        self.codomain, self.domain = tuple(codomain), tuple(domain)
        self.sym = _sym_of(self.codomain + self.domain, sym)
        self.layout = layout(self.sym, self.codomain, self.domain)
        if tuple(data.shape) != (self.layout.size,):
            raise ValueError(f"buffer of shape {tuple(data.shape)} for {self.layout.size} "
                             f"reduced entries")
        self.data, self.be = data, be

    @classmethod
    def _of(cls, layout, data, be) -> "SymmetricTensor":
        t = cls.__new__(cls)
        t.sym, t.codomain, t.domain, t.layout = layout.sym, layout.codomain, layout.domain, layout
        t.data, t.be = data, be
        return t

    @property
    def shape(self) -> Tuple[int, ...]:
        """Dense sizes of the legs, codomain first."""
        return tuple(V.dim for V in self.codomain + self.domain)

    @property
    def sectors(self) -> Tuple[Label, ...]:
        return self.layout.sectors

    def __repr__(self):
        return (f"SymmetricTensor(shape={self.shape}, codomain legs={len(self.codomain)}, "
                f"sectors={len(self.sectors)}, stored={self.layout.size})")

    def block(self, c):
        """The reduced matrix of coupled sector `c` (a view of `data`)."""
        o, shp = self.layout.offsets[c], self.layout.shapes[c]
        return self.data[o:o + math.prod(shp)].reshape(shp)

    def blocks(self) -> Dict[Label, Any]:
        return {c: self.block(c) for c in self.sectors}

    def to_dense(self):
        be, perm = self.be, _interleave(len(self.shape), len(self.codomain))
        out = be.zeros(self.shape)
        for c in self.sectors:
            blk = self.block(c)
            for _, r in self.layout.rows[c]:
                for _, f in self.layout.cols[c]:
                    P = be.array(_pairs(self.sym, r.trees, f.trees))
                    B = blk[r.start:r.stop, f.start:f.stop].reshape(
                        len(r.trees), math.prod(r.degs), len(f.trees), math.prod(f.degs))
                    T = be.einsum("sifj,safb->iajb", B, P).reshape(
                        r.degs + r.dims + f.degs + f.dims)
                    T = _permute(be, T, perm).reshape(
                        tuple(a * b for a, b in zip(r.degs + f.degs, r.dims + f.dims)))
                    sl = r.slices + f.slices
                    if sl:
                        out[sl] += T
                    else:
                        out = out + T
        return out

    # --- vector algebra on the flat buffer ---
    def _like(self, data) -> "SymmetricTensor":
        return SymmetricTensor._of(self.layout, data, self.be)

    def _check(self, other):
        if not isinstance(other, SymmetricTensor):
            return False
        if (self.codomain, self.domain, self.sym) != (other.codomain, other.domain, other.sym):
            raise ValueError("symmetric tensors on different spaces")
        if self.be.is_torch != other.be.is_torch:
            raise TypeError("cannot combine a numpy and a torch symmetric tensor")
        return True

    def __add__(self, other):
        return self._like(self.data + other.data) if self._check(other) else NotImplemented

    def __sub__(self, other):
        return self._like(self.data - other.data) if self._check(other) else NotImplemented

    def __mul__(self, c):
        return self._like(self.data * c) if _is_scalar(c) else NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, c):
        return self._like(self.data / c) if _is_scalar(c) else NotImplemented

    def __neg__(self):
        return self._like(-self.data)

    def norm(self):
        return norm(self)

    def adjoint(self) -> "SymmetricTensor":
        """The adjoint map, from codomain to domain: each block conjugate-transposed. Its dense
        array is the conjugate of this one with axes (domain, codomain)."""
        be = self.be
        return from_blocks({c: be.transpose(be.conj(b), (1, 0)) for c, b in self.blocks().items()},
                           self.domain, self.codomain, be, self.sym)

    def conj(self) -> "SymmetricTensor":
        """The complex conjugate of the dense array, axes unchanged: every leg becomes its dual.
        It is the adjoint with its two sides swapped back."""
        n2, n = len(self.domain), len(self.shape)
        return permute(self.adjoint(), range(n2, n), range(n2))

    # --- the interface of the two-site drivers (`tnkit.symmetric_backend`) ---
    is_block = True                              # the drivers' sparse-tensor branches apply

    @property
    def ndim(self) -> int:
        return len(self.codomain) + len(self.domain)

    @property
    def dtype(self):
        return self.data.dtype

    def with_data(self, data) -> "SymmetricTensor":
        """A tensor on the same spaces with the flat buffer `data`."""
        return SymmetricTensor(self.codomain, self.domain, data, self.be, self.sym)

    def canonical(self) -> "SymmetricTensor":
        """The same dense array with every leg in the codomain. Its only coupled sector is the
        unit, so the plain inner product of two such buffers is the dense one."""
        return repartition(self, self.ndim)

    def scale(self, s: "SymmetricTensor", axis: int) -> "SymmetricTensor":
        """Leg `axis` multiplied by the diagonal bond matrix `s` (from `truncated_svd`), as a
        dense ``x * s`` along that leg would be. The codomain-domain split is kept."""
        n1, n = len(self.codomain), self.ndim
        if axis < n1:
            out = contract(s, self, [1], [axis])
            order = list(range(1, axis + 1)) + [0] + list(range(axis + 1, n))
        else:
            out = contract(self, s, [axis], [0])
            order = list(range(axis)) + [n - 1] + list(range(axis, n - 1))
        return permute(out, order[:n1], order[n1:])

    def fuse(self, groups) -> Tuple["SymmetricTensor", "SymmetricFuseRecord"]:
        """The matrix of `tnkit.legs.fuse`: the legs of ``groups[0]`` fused into one codomain
        leg and those of ``groups[1]`` into one domain leg (a group of one leg is kept as it
        is), and the record `split` needs."""
        rec = SymmetricFuseRecord(self.shape, tuple(tuple(int(a) for a in g) for g in groups))
        if len(rec.groups) != 2:
            raise ValueError(f"symmetric tensors fuse into two legs, got groups {rec.groups}")
        g0, g1 = rec.groups
        t = permute(self, g0, g1)
        rec = SymmetricFuseRecord(rec.shape, rec.groups, spaces=(t.codomain, t.domain))
        if len(g0) > 1:
            t = fuse(t, range(len(g0)))
        if len(g1) > 1:
            t = fuse(t, range(1, 1 + len(g1)))
        return t, rec

    def split(self, rec: "SymmetricFuseRecord", legs=None) -> "SymmetricTensor":
        """Undo `fuse` on the legs `legs` (default both) of a matrix, and return the result with
        every leg in the codomain (`canonical`). As `tnkit.legs.split`: the split legs must
        have the fused sizes, and a partial split needs a fuse that did not permute."""
        if not isinstance(rec, SymmetricFuseRecord):
            raise TypeError("a symmetric tensor needs a SymmetricFuseRecord")
        if self.ndim != 2:
            raise ValueError(f"expected 2 fused legs, got shape {self.shape}")
        todo = (0, 1) if legs is None else tuple(sorted(int(g) for g in legs))
        if not set(todo) <= {0, 1}:
            raise ValueError(f"legs {todo} out of range for 2 fused legs")
        for g in todo:
            if self.shape[g] != math.prod(rec.dims(g)):
                raise ValueError(f"leg {g} has size {self.shape[g]}, but fused group {g} is "
                                 f"{rec.dims(g)}")
        if legs is not None and rec.perm != tuple(range(len(rec.perm))):
            raise ValueError(f"a partial split needs an unpermuted fuse; this one used perm "
                             f"{rec.perm}")
        t = self if (len(self.codomain), len(self.domain)) == (1, 1) else repartition(self, 1)
        if 1 in todo and len(rec.groups[1]) > 1:
            t = split(t, 1, rec.spaces[1])
        if 0 in todo and len(rec.groups[0]) > 1:
            t = split(t, 0, rec.spaces[0])
        t = t.canonical()
        perm = rec.perm
        if legs is None and perm != tuple(range(len(perm))):
            t = permute(t, [int(i) for i in np.argsort(perm)], ())
        return t


# --------------------------------------------------------------------------- construction
def zeros(codomain: Sequence[Space], domain: Sequence[Space] = (), be=None,
          sym: Optional[Symmetry] = None) -> SymmetricTensor:
    """The zero tensor (`sym` is needed only when there are no legs)."""
    be = be or NumpyBackend()
    codomain, domain = tuple(codomain), tuple(domain)
    lay = layout(_sym_of(codomain + domain, sym), codomain, domain)
    return SymmetricTensor._of(lay, be.zeros((lay.size,)), be)


def random(codomain: Sequence[Space], domain: Sequence[Space] = (), be=None, rng=None,
           sym: Optional[Symmetry] = None) -> SymmetricTensor:
    """Independent standard-normal reduced entries (complex on a complex backend)."""
    t = zeros(codomain, domain, be, sym)
    rng = np.random.default_rng() if rng is None else rng
    return t._like(t.be.randn(rng, (t.layout.size,)))


def from_blocks(blocks: Mapping[Label, Any], codomain: Sequence[Space],
                domain: Sequence[Space] = (), be=None,
                sym: Optional[Symmetry] = None) -> SymmetricTensor:
    """The tensor with these reduced blocks; a coupled sector left out is zero."""
    t = zeros(codomain, domain, be, sym)
    extra = set(blocks) - set(t.sectors)
    if extra:
        raise ValueError(f"no block for coupled sectors {sorted(extra, key=t.sym.sort_key)}")
    for c, b in blocks.items():
        if tuple(b.shape) != t.layout.shapes[c]:
            raise ValueError(f"block {c} has shape {tuple(b.shape)}, expected "
                             f"{t.layout.shapes[c]}")
        t.block(c)[...] = t.be.array(b)
    return t


def from_dense(x, codomain: Sequence[Space], domain: Sequence[Space] = (), be=None,
               sym: Optional[Symmetry] = None) -> Tuple[SymmetricTensor, float]:
    """Project the dense array `x` (axes: codomain legs, then domain legs) onto the invariant
    tensors. Returns the tensor and the squared norm of what the projection dropped."""
    t = zeros(codomain, domain, be, sym)
    be = t.be
    inv = [int(i) for i in np.argsort(_interleave(len(t.shape), len(t.codomain)))]
    x = be.array(x)
    if tuple(x.shape) != t.shape:
        raise ValueError(f"array of shape {tuple(x.shape)} for legs of sizes {t.shape}")
    for c in t.sectors:
        blk = t.block(c)
        for _, r in t.layout.rows[c]:
            for _, f in t.layout.cols[c]:
                sl = r.slices + f.slices
                X = (x[sl] if sl else x).reshape(
                    tuple(v for p in zip(r.degs + f.degs, r.dims + f.dims) for v in p))
                X = _permute(be, X, inv).reshape(
                    math.prod(r.degs), math.prod(r.dims), math.prod(f.degs), math.prod(f.dims))
                P = be.conj(be.array(_pairs(t.sym, r.trees, f.trees)))
                B = be.einsum("iajb,safb->sifj", X, P) / t.sym.dim(c)
                blk[r.start:r.stop, f.start:f.stop] = B.reshape(r.stop - r.start,
                                                                f.stop - f.start)
    return t, float(be.norm(x - t.to_dense())) ** 2


# --------------------------------------------------------------------------- inner product
class _Weights(Growing):
    """dim(c) on every entry of a layout's buffer (or its square root), in the real dtype of
    each dtype asked for; None where every coupled sector has dimension one."""

    def __init__(self, lay: Layout, be):
        dims = [lay.sym.dim(c) for c in lay.sectors]
        self.w, self.be, self._cast, self._cast_bytes = None, be, {}, 0
        if any(d != 1 for d in dims):
            self.w = np.repeat(np.asarray(dims, np.float64),
                               [math.prod(lay.shapes[c]) for c in lay.sectors])

    def nbytes(self) -> int:
        return 0 if self.w is None else self.w.nbytes + self._cast_bytes

    def get(self, dtype, root: bool):
        if self.w is None:
            return None
        key = (_recoupling.real_dtype(dtype), root)
        got = self._cast.get(key)
        if got is None:
            w = np.sqrt(self.w) if root else self.w
            got, copied = _recoupling._cast(self.be, w, key[0])
            if self._cast.setdefault(key, got) is got:
                self._cast_bytes += nbytes(got) if copied or w is not self.w else 0
                self.grew()
            got = self._cast[key]
        return got


def _weights(t: SymmetricTensor, root: bool):
    lay = t.layout
    return lay.map(t.be, ("weights",), lambda: _Weights(lay, t.be)).get(t.data.dtype, root)


CHUNK = 1 << 16     # entries per weighted reduction on numpy, which needs no larger temporary


def _chunks(n: int):
    return (slice(i, i + CHUNK) for i in range(0, n, CHUNK))


def vdot(a: SymmetricTensor, b: SymmetricTensor):
    """The dense inner product <a, b>, conjugating a: sum_c dim(c) Tr(a_c^dag b_c), one
    reduction over the buffers (on numpy, one per `CHUNK` entries)."""
    if not a._check(b):
        raise TypeError("vdot needs two symmetric tensors")
    x, y, w = a.data, b.data, _weights(b, False)
    if w is None:
        return a.be.vdot(x, y)
    if a.be.is_torch or len(w) <= CHUNK:
        return a.be.vdot(x, y * w)
    return sum(np.vdot(x[k], y[k] * w[k]) for k in _chunks(len(w)))


def norm(a: SymmetricTensor):
    """The dense Frobenius norm, sqrt(sum_c dim(c) |a_c|^2), one reduction over the buffer (on
    numpy, one per `CHUNK` entries)."""
    x, w = a.data, _weights(a, True)
    if w is None:
        return a.be.norm(x)
    if a.be.is_torch or len(w) <= CHUNK:
        return a.be.norm(x * w)
    return np.sqrt(sum(np.vdot(z, z).real for z in (x[k] * w[k] for k in _chunks(len(w)))))


# --------------------------------------------------------------------------- recoupling
def permute(t: SymmetricTensor, codomain_axes: Sequence[int],
            domain_axes: Sequence[int]) -> SymmetricTensor:
    """The tensor with dense array ``transpose(t.to_dense(), codomain_axes + domain_axes)``, the
    legs `codomain_axes` in its codomain and `domain_axes` in its domain. Axes count the codomain
    legs first. A leg that changes side is dualised."""
    p1, p2 = tuple(int(k) for k in codomain_axes), tuple(int(k) for k in domain_axes)
    n1, n = len(t.codomain), len(t.codomain) + len(t.domain)
    if sorted(p1 + p2) != list(range(n)):
        raise ValueError(f"{p1}, {p2} is not a permutation of the {n} legs")
    if p1 == tuple(range(n1)) and p2 == tuple(range(n1, n)):
        return t
    lay, be = t.layout, t.be
    op = lay.map(be, ("permute", p1, p2), lambda: Recoupler(lay, p1, p2, be))
    return SymmetricTensor._of(op.target.get(), op.apply(t.data), be)


def repartition(t: SymmetricTensor, n: int) -> SymmetricTensor:
    """The same dense array with its first `n` legs in the codomain and the rest in the domain."""
    return permute(t, range(n), range(n, len(t.shape)))


def transpose(t: SymmetricTensor) -> SymmetricTensor:
    """The transpose map: codomain and domain swapped, each in reverse order of legs."""
    n1, n = len(t.codomain), len(t.shape)
    return permute(t, range(n - 1, n1 - 1, -1), range(n1 - 1, -1, -1))



# --------------------------------------------------------------------------- contraction
def _common_backend(a: SymmetricTensor, b: SymmetricTensor):
    """The backend of a product of a and b: the one of the promoted dtype."""
    ba, bb = a.be, b.be
    if ba is bb or (backend_key(ba) == backend_key(bb) and str(ba.dtype) == str(bb.dtype)):
        return ba
    if ba.is_torch != bb.is_torch:
        raise TypeError("cannot combine a numpy and a torch symmetric tensor")
    if ba.is_torch:
        if ba.device != bb.device:
            raise ValueError(f"symmetric tensors on devices {ba.device} and {bb.device}")
        dtype = ba.torch.promote_types(ba.dtype, bb.dtype)
        same = (lambda be: be.dtype == dtype)
    else:
        dtype = np.result_type(ba.dtype, bb.dtype)
        same = (lambda be: np.dtype(be.dtype) == dtype)
    for be in (ba, bb):
        if same(be):
            return be
    raise TypeError(f"no backend of the common dtype {dtype} of {ba.dtype} and {bb.dtype}")


def _to(t: SymmetricTensor, be) -> SymmetricTensor:
    return t if t.be is be else SymmetricTensor._of(t.layout, be.array(t.data), be)


def contract(a: SymmetricTensor, b: SymmetricTensor, axes_a: Sequence[int],
             axes_b: Sequence[int]) -> SymmetricTensor:
    """``tensordot(a, b, (axes_a, axes_b))``: the free legs of `a` form the codomain and those of
    `b` the domain. A contracted pair is a leg and its dual, as seen from the codomain: a codomain
    leg V meets a domain leg V or a codomain leg dual(V). A real and a complex tensor give a
    complex one; numpy and torch tensors do not mix."""
    axes_a, axes_b = tuple(int(k) for k in axes_a), tuple(int(k) for k in axes_b)
    if (len(axes_a) != len(axes_b) or len(set(axes_a)) < len(axes_a)
            or len(set(axes_b)) < len(axes_b)):
        raise ValueError(f"cannot contract legs {axes_a} with legs {axes_b}")
    if a.sym != b.sym:
        raise ValueError(f"tensors of different symmetries, {a.sym} and {b.sym}")
    be = _common_backend(a, b)
    na, nb = len(a.codomain) + len(a.domain), len(b.codomain) + len(b.domain)
    A = _to(permute(a, [k for k in range(na) if k not in axes_a], axes_a), be)
    B = _to(permute(b, axes_b, [k for k in range(nb) if k not in axes_b]), be)
    for ka, kb, V, W in zip(axes_a, axes_b, A.domain, B.codomain):
        if V == W:
            continue
        legs = f"leg {ka} of the first tensor and leg {kb} of the second"
        if V.sectors != W.sectors:
            raise ValueError(f"{legs} have different sectors: {V.sectors} and {W.sectors}")
        if V.degeneracies != W.degeneracies:
            raise ValueError(f"{legs} have different degeneracies: {V.degeneracies} and "
                             f"{W.degeneracies}")
        raise ValueError(f"{legs} are not dual to each other")
    lay = layout(a.sym, A.codomain, B.domain).keep()
    prod = A.layout.map(be, ("contract", B.layout.uid),
                        lambda: _Products(A.layout, B.layout, lay, be), partner=B.layout)
    return SymmetricTensor._of(lay, prod.apply(A.data, B.data, be), be)


BATCH_GATHER_MIN = 4     # equal-shape blocks apart in the buffers: gathered into one product


class _Products:
    """The block products of `contract` for two layouts: out_c = A_c B_c for every coupled
    sector of the result, written in place into its buffer. Blocks of one shape (m, k, n)
    that sit one after another in all three buffers take one strided batched product; other
    blocks of one shape, if at least `BATCH_GATHER_MIN`, are gathered into one."""

    def __init__(self, la: Layout, lb: Layout, out: Layout, be):
        self.size = out.size
        items = [(la.shapes[c] + (lb.shapes[c][1],), la.offsets[c], lb.offsets[c],
                  out.offsets[c]) for c in out.sectors if c in la.offsets and c in lb.offsets]
        self.complete = len(items) == len(out.sectors)
        by_shape: Dict[Tuple[int, int, int], list] = {}
        for shp, oa, ob, oo in items:
            by_shape.setdefault(shp, []).append((oa, ob, oo))
        self.runs, self.gathers = [], []
        for (m, k, n), xs in by_shape.items():
            if len(xs) >= BATCH_GATHER_MIN and not _consecutive(xs, (m * k, k * n, m * n)):
                ia, ib, io = (np.concatenate([o + np.arange(sz) for o in col])
                              for col, sz in zip(zip(*xs), (m * k, k * n, m * n)))
                self.gathers.append(((len(xs), m, k, n), to_index(be, ia, la.size),
                                     to_index(be, ib, lb.size), to_index(be, io, out.size)))
                continue
            start = 0
            for j in range(1, len(xs) + 1):     # runs of blocks that follow one another
                if j == len(xs) or not _consecutive(xs[j - 1:j + 1], (m * k, k * n, m * n)):
                    self.runs.append(((j - start, m, k, n),) + xs[start])
                    start = j

    def nbytes(self) -> int:
        return 64 * len(self.runs) + sum(nbytes(x) for g in self.gathers for x in g[1:])

    def apply(self, a, b, be):
        mm, bmm = (be.torch.mm, be.torch.bmm) if be.is_torch else (np.matmul, np.matmul)
        if be.is_torch:
            data = (be.torch.empty if self.complete else be.torch.zeros)(
                self.size, dtype=be.dtype, device=be.device)
        else:
            data = (np.empty if self.complete else np.zeros)(self.size, be.dtype)
        for (r, m, k, n), oa, ob, oo in self.runs:
            lead = (r,) if r > 1 else ()
            (bmm if r > 1 else mm)(a[oa:oa + r * m * k].reshape(lead + (m, k)),
                                   b[ob:ob + r * k * n].reshape(lead + (k, n)),
                                   out=data[oo:oo + r * m * n].reshape(lead + (m, n)))
        for (g, m, k, n), ia, ib, io in self.gathers:
            data[io] = bmm(a[ia].reshape(g, m, k), b[ib].reshape(g, k, n)).reshape(-1)
        return data


def _consecutive(xs, sizes) -> bool:
    """True if each block of `xs` (offsets in three buffers) follows the one before in all
    three buffers."""
    return all(y[i] - x[i] == sz for x, y in zip(xs, xs[1:]) for i, sz in enumerate(sizes))


# --------------------------------------------------------------------------- fuse and split
def fuse_spaces(spaces: Sequence[Space]) -> Space:
    """The plain leg of the coupled sectors of `spaces`, each as often as its trees and
    degeneracies reach it."""
    spaces = tuple(spaces)
    if not spaces:
        raise ValueError("nothing to fuse")
    sym = _sym_of(spaces, None)
    rows = {sym.unit(): 1}                  # rows reaching each line, one leg at a time
    for V in spaces:
        new: Dict[Label, int] = {}
        for e, k in rows.items():
            for a, n in zip(V.sectors, V.degeneracies):
                for c, m in _symmetric_layout.fusions(sym, e, V.leaf(a)):
                    new[c] = new.get(c, 0) + k * n * m
        rows = new
    sectors = sorted(rows, key=sym.sort_key)
    return Space(sym, sectors, tuple(rows[c] for c in sectors))


def _fuse_index(lay: Layout, k: int) -> Tuple[Target, np.ndarray]:
    """The layout with the first k codomain legs fused, and the gather onto its buffer. In a
    left-associated tree, those legs couple to inner line k - 1; the rest of the tree takes that
    line as its first leaf, so each row only moves."""
    sym, cod = lay.sym, lay.codomain
    fused = fuse_spaces(cod[:k])
    target = Target(sym, (fused,) + cod[k:], lay.domain)
    out = target.get()
    old, new = lay.row, out.row
    used = frozenset(fused.sectors[i] for i in np.unique(new.trees.tup[:, 0]))     # on line k - 1
    sub = Side(sym, cod[:k], coupled=used)
    to, ts, tn = old.trees, sub.trees, new.trees
    sub_at = {ts.key(i): i for i in range(len(ts.coupled))}
    new_at = {tn.key(i): i for i in range(len(tn.coupled))}
    pos = {e: i for i, e in enumerate(fused.sectors)}
    rows = {c: np.empty(lay.shapes[c][0], np.int64) for c in lay.sectors}
    for t in range(len(to.coupled)):
        c = to.labels[to.coupled[t]]
        if c not in rows:
            continue
        tup, inner, _, verts = to.key(t)
        lines = (to.leaves[0][tup[0]],) + inner + (c,) if len(tup) > 1 else (c,)
        e = lines[k - 1]
        i1 = sub_at[(tup[:k], lines[1:k - 1], e, verts[:k - 1])]
        i2 = new_at[((pos[e],) + tup[k:], lines[k:-1], c, verts[k - 1:])]
        d1, d2 = int(sub.deg[i1]), int(old.deg[t] // sub.deg[i1])
        rows[c][new.start[i2] + (sub.start[i1] + np.arange(d1))[:, None] * d2
                + np.arange(d2)] = old.start[t] + np.arange(d1 * d2).reshape(d1, d2)
    parts = [lay.offsets[c] + rows[c][:, None] * lay.shapes[c][1]
             + np.arange(lay.shapes[c][1]) for c in out.sectors]
    return target, (np.concatenate([p.reshape(-1) for p in parts]) if parts
                    else np.zeros(0, np.int64))


def _fuse_leading(t: SymmetricTensor, k: int) -> SymmetricTensor:
    lay, be = t.layout, t.be
    op = lay.map(be, ("fuse", k), lambda: Gather(*_fuse_index(lay, k), lay.size, be))
    return SymmetricTensor._of(op.target.get(), op.apply(t.data), be)


def _splitter(lay: Layout, spaces: Tuple[Space, ...], be) -> Gather:
    if lay.codomain[0] != fuse_spaces(spaces):
        raise ValueError("the leg is not the fusion of these spaces")
    target = Target(lay.sym, spaces + lay.codomain[1:], lay.domain)
    _, index = _fuse_index(target.get(), len(spaces))
    inv = np.empty(len(index), np.int64)
    inv[index] = np.arange(len(index))
    return Gather(target, inv, lay.size, be)


def _split_leading(t: SymmetricTensor, spaces: Tuple[Space, ...]) -> SymmetricTensor:
    lay, be = t.layout, t.be
    op = lay.map(be, ("split", spaces), lambda: _splitter(lay, spaces, be))
    return SymmetricTensor._of(op.target.get(), op.apply(t.data), be)


def fuse(t: SymmetricTensor, axes: Sequence[int]) -> SymmetricTensor:
    """The legs `axes` (all in the codomain or all in the domain) replaced by one leg,
    `fuse_spaces` of them in this order, at the place of the first. On codomain legs the dense
    array is U applied to them, with U the unitary onto the fused leg whose reduced blocks are
    identities; on domain legs it is the dense array times U^dag."""
    axes, n1, n = tuple(int(k) for k in axes), len(t.codomain), len(t.shape)
    if not axes or len(set(axes)) < len(axes) or not all(0 <= k < n for k in axes):
        raise ValueError(f"cannot fuse legs {axes}")
    if all(k >= n1 for k in axes):
        return fuse(t.adjoint(), [k - n1 for k in axes]).adjoint()
    if not all(k < n1 for k in axes):
        raise ValueError("fused legs must all be in the codomain or all in the domain")
    rest = [k for k in range(n1) if k not in axes]
    f = _fuse_leading(permute(t, list(axes) + rest, range(n1, n)), len(axes))
    keep = [k for k in range(n1) if k not in axes or k == axes[0]]
    m = len(keep)
    return permute(f, [0 if k == axes[0] else 1 + rest.index(k) for k in keep],
                   range(m, m + n - n1))


def split(t: SymmetricTensor, axis: int, spaces: Sequence[Space]) -> SymmetricTensor:
    """The inverse of `fuse`: the leg `axis`, the fusion of `spaces`, replaced by them. The leg
    must be on the side where `fuse` put it, as the plain leg `fuse_spaces(spaces)`; moved to the
    other side it is dual, and does not split."""
    spaces, n1, n = tuple(spaces), len(t.codomain), len(t.shape)
    if not 0 <= axis < n:
        raise ValueError(f"no leg {axis}")
    if axis >= n1:
        return split(t.adjoint(), axis - n1, spaces).adjoint()
    rest = [k for k in range(n1) if k != axis]
    s = _split_leading(permute(t, [axis] + rest, range(n1, n)), spaces)
    k = len(spaces)
    order = [j for p in range(n1) for j in (range(k) if p == axis else [k + rest.index(p)])]
    return permute(s, order, range(len(order), len(order) + n - n1))


def concat(a: SymmetricTensor, b: SymmetricTensor, axis: int) -> SymmetricTensor:
    """`a` and `b` joined along leg `axis` (legs counted codomain first), every other leg and the
    dual flag of that leg equal: the direct sum of the two spaces on that leg, the states of `a`
    first within each sector. The result has every leg in the codomain."""
    a, b = a.canonical(), b.canonical()
    n = a.ndim
    if not 0 <= axis < n or b.ndim != n:
        raise ValueError(f"cannot join tensors of {n} and {b.ndim} legs along leg {axis}")
    rest = [k for k in range(n) if k != axis]
    pa, pb = permute(a, rest, (axis,)), permute(b, rest, (axis,))
    Va, Vb = pa.domain[0], pb.domain[0]
    if pa.codomain != pb.codomain or Va.isdual != Vb.isdual:
        raise ValueError("concat needs equal legs except along `axis`, and equal dual flags")
    na, nb = dict(zip(Va.sectors, Va.degeneracies)), dict(zip(Vb.sectors, Vb.degeneracies))
    secs = sorted(set(na) | set(nb), key=a.sym.sort_key)
    V = Space(a.sym, secs, tuple(na.get(c, 0) + nb.get(c, 0) for c in secs), Va.isdual)
    out = zeros(pa.codomain, (V,), a.be, a.sym)
    for c in out.sectors:
        k = pa.layout.shapes[c][1] if c in pa.layout.shapes else 0
        if k:
            out.block(c)[:, :k] = pa.block(c)
        if c in pb.layout.shapes:
            out.block(c)[:, k:] = pb.block(c)
    return permute(out, list(range(axis)) + [n - 1] + list(range(axis, n - 1)), ())
