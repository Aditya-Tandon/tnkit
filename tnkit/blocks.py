"""Block-sparse tensors for abelian symmetries.

Each leg of a `BlockTensor` is split into sectors, labelled by a `tnkit.symmetry.Symmetry`.
A `Leg` built here without one is U(1), a convenience of this low-level layer; the converters in
`tnkit.block_backend` always take the symmetry from the caller. A block takes one sector per leg
and is allowed when the leg labels, each dualised where the leg's sign is -1, fuse to the
tensor's total charge. Every allowed block is stored, so the legs and the total charge fix which
blocks exist. All blocks live in one flat buffer of the backend's array type (a numpy array, or a
torch tensor on any device). Where each block sits, and with which strides, is host metadata in
a `Struct`: two tensors with the same struct differ only in their numbers.

Contractions run on plans cached per pair of structs. `tensordot` gathers each operand into one
matrix per contracted charge (one indexed copy per operand), multiplies each charge's matrices with
one GEMM, and leaves the products where they are: the result's blocks are strided views of them.
Permuting legs only relabels that layout. A repeated contraction therefore issues the same few
kernels and does no host work beyond looking up its plan. There is one `Struct` object per layout
content, so a plan is found again whenever its layouts recur, however they were built.

A plan has two levels. The symbolic level says which blocks meet in which GEMM; it depends only
on the sector labels and the block order, so it survives any change of sector sizes, and it is
cached on its own. The numeric level (offsets, GEMM shapes and gather indices) is computed from
the symbolic one and the sector sizes with a few vectorised operations, without a loop over
blocks. A gather index is stored as runs (arithmetic progressions) and expanded once per device
it is used on: a small one on the host, then copied through pinned memory without waiting for the
device; a large one on the device itself. An operand's gather depends on that operand alone, so it
is reused when only the other operand changes.

Plans and the other gather indices live in one cache bounded by the bytes they occupy on the host
and on every device (`set_cache_limit`, `cache_info`, ``TNKIT_BLOCK_CACHE_MB``); symbolic plans
live in a second cache (``TNKIT_BLOCK_LABEL_CACHE_MB``, 512 MiB). Both variables are read at
import. Unless ``TNKIT_BLOCK_CACHE_MB`` is set or `set_cache_limit` is called, the plan cache
holds 4 GiB, lowered to 30% of the device's memory if that is less when an index first reaches a
CUDA device. The pinned host buffers that uploads go through come from torch's pinned-memory
pool, which these limits do not bound.
``TNKIT_BLOCK_EXPAND`` (``auto``, ``host`` or ``device``; read at each expansion) sets where
indices are expanded. Run a layout once eagerly before capturing it in a CUDA graph;
`tnkit.block_graph` does so for Krylov matvecs, and `keep_operands` records what such a graph
reads.

`truncated_svd` factorises a two-leg block tensor sector by sector and truncates globally across
sectors. `tnkit.block_backend.BlockBackend` puts these tensors behind the backend adaptor.

Signs: contracted legs must carry the same sectors with opposite signs. For an MPS tensor
``(a, s, b)`` with bonds labelled by the charge to their left, the signs are ``(+1, +1, -1)`` and
the total charge is the trivial sector.

Provisional: this module's API may change before 1.0.
"""
from __future__ import annotations

import contextlib
import itertools
import math
import numbers
import os
import time
import weakref
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .legs import FuseRecord
from .symmetry import Symmetry, U1

__all__ = ["Leg", "BlockTensor", "BlockFuseRecord", "from_dense", "random", "zeros", "concat",
           "tensordot", "truncated_svd", "qr", "set_cache_limit", "cache_info", "clear_cache"]

_META_CACHE = 4096                     # interned structs and records: host metadata only
_U1 = U1()
_HOST_EXPAND_MAX = 1 << 17             # indices up to this many elements expand on the host
_PIN_MAX_BYTES = 8 * _HOST_EXPAND_MAX  # largest pinned upload (bytes): a host-expanded int64 index


@dataclass(frozen=True)
class Leg:
    """The sectors of one leg: distinct sector labels `charges` of the symmetry `sym` (U(1) if
    not given), in its `sort_key` order, their `dims`, and the `sign` (+1 or -1) with which the
    leg enters the conservation rule (-1: the dual of each label)."""
    charges: Tuple[Any, ...]
    dims: Tuple[int, ...]
    sign: int = 1
    sym: Symmetry = _U1

    def __post_init__(self):
        if (list(self.charges) != sorted(set(self.charges), key=self.sym.sort_key)
                or len(self.dims) != len(self.charges)
                or any(d < 1 for d in self.dims) or self.sign not in (1, -1)
                or any(self.sym.label(c) != c for c in self.charges)):
            raise ValueError(f"invalid leg {self}")

    @property
    def size(self) -> int:
        return sum(self.dims)

    def flip(self) -> "Leg":
        return Leg(self.charges, self.dims, -self.sign, self.sym)

    @classmethod
    def from_charges(cls, q, sign: int = 1, sym: Symmetry = _U1) -> Tuple["Leg", np.ndarray]:
        """The leg whose dense index i has sector ``q[i]``, and the dense indices in sector order
        (stable within a sector): the `orders` entry `from_dense` and `to_dense` take."""
        if type(sym) is U1:
            q = np.asarray(q, dtype=np.int64)
            c, n = np.unique(q, return_counts=True)
            return (cls(tuple(map(int, c)), tuple(map(int, n)), sign, sym),
                    np.argsort(q, kind="stable"))
        labels = [sym.label(a) for a in q]
        sectors = sorted(set(labels), key=sym.sort_key)
        rank = {a: i for i, a in enumerate(sectors)}
        r = np.array([rank[a] for a in labels], dtype=np.int64)
        n = np.bincount(r, minlength=len(sectors))
        return (cls(tuple(sectors), tuple(map(int, n)), sign, sym),
                np.argsort(r, kind="stable"))


def _row_major(shape) -> Tuple[int, ...]:
    st, n = [], 1
    for d in reversed(shape):
        st.append(n)
        n *= d
    return tuple(reversed(st))


def _row_major_rows(shp: np.ndarray) -> np.ndarray:
    """Row-major strides of every row of a (blocks, axes) array of shapes."""
    st = np.ones_like(shp)
    for j in range(shp.shape[1] - 2, -1, -1):
        st[:, j] = st[:, j + 1] * shp[:, j + 1]
    return st


def _sym_of(legs: Sequence[Leg]) -> Symmetry:
    if not legs:
        return _U1
    sym = legs[0].sym
    if any(leg.sym != sym for leg in legs):
        raise ValueError("all legs of a tensor need the same symmetry")
    if sym.fusion != "unique":
        raise NotImplementedError(f"{sym} needs unique fusion, which block tensors require")
    return sym


class Struct:
    """Where each block of a tensor sits in its buffer: ``blocks[k]`` (one sector index per leg)
    starts at ``offsets[k]`` with element strides ``strides[k]``. Hashed by identity: build it
    with `_struct`, which returns one object per layout content while that object lives, and
    plans are keyed on these objects. The numbers are held as int64 arrays (`arrays`); the
    tuples are made on demand. `refs` counts the cached plans whose output it is. Pickling and
    copying return the interned object."""

    def __init__(self, legs, charge, blocks, off, st, shp, size, sym=_U1):
        self.legs, self.charge, self.blocks = tuple(legs), charge, blocks
        self.arrays = (off, st, shp)           # offsets, strides, shapes: one row per block
        self.size, self.sym, self.refs = int(size), sym, 0

    def __reduce__(self):
        return _struct, (self.legs, self.charge, self.blocks, *self.arrays, self.size, self.sym)

    @property
    def nbytes(self) -> int:
        """Host bytes of the arrays, and of the copy of offsets and strides that interns it.
        Arrays shared with a related layout (a permutation, the dual) are counted in each."""
        off, st, shp = self.arrays
        return 2 * (off.nbytes + st.nbytes) + shp.nbytes

    @cached_property
    def offsets(self) -> Tuple[int, ...]:
        return tuple(self.arrays[0].tolist())

    @cached_property
    def strides(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(map(tuple, self.arrays[1].tolist()))

    @cached_property
    def index(self) -> Dict[Tuple[int, ...], int]:
        return {b: k for k, b in enumerate(self.blocks)}

    @cached_property
    def labels(self):
        """Everything but the sizes: the key of symbolic plans."""
        return (tuple((leg.charges, leg.sign) for leg in self.legs), self.charge, self.blocks,
                self.sym)

    def shape(self, k: int) -> Tuple[int, ...]:
        return tuple(leg.dims[s] for leg, s in zip(self.legs, self.blocks[k]))

    def grid(self, k: int) -> np.ndarray:
        """Buffer positions of block `k`'s elements, as an array of the block's shape."""
        axes = [np.arange(d, dtype=np.int64) * s for d, s in zip(self.shape(k), self.strides[k])]
        return self.offsets[k] + sum(np.ix_(*axes), np.zeros((), np.int64))


def _layout_key(s: Struct):
    """Everything that defines a layout: labels, sector sizes, buffer size, offsets, strides."""
    off, st, _ = s.arrays
    return (s.labels, tuple(leg.dims for leg in s.legs), s.size, off.tobytes(), st.tobytes())


_LAYOUTS: "weakref.WeakValueDictionary[Any, Struct]" = weakref.WeakValueDictionary()


def _struct(legs, charge, blocks, off, st, shp, size, sym=_U1) -> Struct:
    """The `Struct` with this content: the live one if there is one, else a new one."""
    s = Struct(legs, charge, blocks, off, st, shp, size, sym)
    return _LAYOUTS.setdefault(_layout_key(s), s)


def _block_shapes(legs: Sequence[Leg], sec: np.ndarray) -> np.ndarray:
    shp = np.empty(sec.shape, np.int64)
    for j, leg in enumerate(legs):
        shp[:, j] = np.asarray(leg.dims, np.int64)[sec[:, j]]
    return shp


@lru_cache(maxsize=_META_CACHE)
def _allowed(labels, charge, sym: Symmetry):
    """Every allowed block, in lexicographic sector order, and the same as an int array."""
    blocks = tuple(b for b in itertools.product(*(range(len(c)) for c, _ in labels))
                   if sym.total([c[s] for (c, _), s in zip(labels, b)],
                                [sg for _, sg in labels]) == charge)
    return blocks, np.array(blocks, dtype=np.int64).reshape(len(blocks), len(labels))


@lru_cache(maxsize=_META_CACHE)
def canonical(legs: Tuple[Leg, ...], charge) -> Struct:
    """Every allowed block in lexicographic sector order, each packed row-major."""
    sym = _sym_of(legs)
    charge = sym.label(charge)
    blocks, sec = _allowed(tuple((leg.charges, leg.sign) for leg in legs), charge, sym)
    shp = _block_shapes(legs, sec)
    n = shp.prod(1)
    return _struct(legs, charge, blocks, np.cumsum(n) - n, _row_major_rows(shp), shp, n.sum(),
                   sym)


@lru_cache(maxsize=_META_CACHE)
def _permuted(s: Struct, perm: Tuple[int, ...]) -> Struct:
    off, st, shp = s.arrays
    p = list(perm)
    return _struct(tuple(s.legs[i] for i in perm), s.charge,
                  tuple(tuple(b[i] for i in perm) for b in s.blocks), off, st[:, p], shp[:, p],
                  s.size, s.sym)


@lru_cache(maxsize=_META_CACHE)
def _dual(s: Struct) -> Struct:
    off, st, shp = s.arrays
    return _struct(tuple(leg.flip() for leg in s.legs), s.sym.dual(s.charge), s.blocks, off, st,
                   shp, s.size, s.sym)


# --- gather indices and their caches ---

def _runs(src0, shp, ss, dst0, ds):
    """Split boxes into runs, arithmetic progressions on both sides. Box b covers the
    multi-indices i < shp[b], read at ``src0[b] + i.ss[b]`` and written at ``dst0[b] + i.ds[b]``.
    Each box runs along its trailing axes, merged while both sides stay contiguous, or along its
    longest axis if that is longer. Returns (src start, src step, dst start, dst step, length) of
    every run."""
    B, m = shp.shape
    if m == 0:
        one = np.ones(B, np.int64)
        return src0, one, dst0, one, one
    rows = np.arange(B)
    tail, start = shp[:, -1].copy(), np.full(B, m - 1)
    chain = ds[:, -1] == 1
    for j in range(m - 2, -1, -1):
        chain &= (ss[:, j] == tail * ss[:, -1]) & (ds[:, j] == tail)
        if not chain.any():
            break
        tail = np.where(chain, tail * shp[:, j], tail)
        start = np.where(chain, j, start)
    j = shp.argmax(1)
    longest = shp[rows, j]
    use_tail = tail >= longest
    sstep = np.where(use_tail, ss[:, -1], ss[rows, j])
    dstep = np.where(use_tail, ds[:, -1], ds[rows, j])
    length = np.where(use_tail, tail, longest)
    ax = np.arange(m)
    in_run = np.where(use_tail[:, None], ax >= start[:, None], ax == j[:, None])
    outer = np.where(in_run, 1, shp)
    nrun = outer.prod(1)
    box = np.repeat(rows, nrun)
    local = np.arange(box.size) - (np.cumsum(nrun) - nrun)[box]
    src, dst = src0[box], dst0[box]
    for k in np.flatnonzero((outer > 1).any(0))[::-1].tolist():
        d = outer[:, k][box]
        i = local % d
        local //= d
        src = src + i * ss[:, k][box]
        dst = dst + i * ds[:, k][box]
    return src, sstep[box], dst, dstep[box], length[box]


def _upload(h: np.ndarray, device):
    """The host array `h` on a torch device. To a GPU, one of at most `_PIN_MAX_BYTES` goes
    through pinned memory without waiting for the device (torch keeps the pinned buffer until the
    copy has run); a larger one is copied synchronously, so the pinned pool holds small buffers
    only."""
    import torch
    t = torch.from_numpy(h)
    if device.type == "cpu":
        return t
    if device.type == "cuda":
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("a block layout is new during CUDA graph capture; run it once "
                               "eagerly before capturing")
        if h.nbytes <= _PIN_MAX_BYTES:
            return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


class _Index:
    """A gather index on the host (int32 when it fits), plus its copy on each device it has been
    used on. Built either from explicit position arrays or from runs. Runs are expanded on first
    use: on the host for the CPU and for indices of at most `_HOST_EXPAND_MAX` elements, then
    copied; on the device itself otherwise (``TNKIT_BLOCK_EXPAND=host`` or ``device`` forces one
    way). They are then dropped; later devices copy the expanded index."""
    __slots__ = ("_host", "_dev", "_runs", "n", "_identity", "_small", "refs")

    def __init__(self, parts: List[np.ndarray]):
        h = np.concatenate([p.ravel() for p in parts]) if parts else np.zeros(0, np.int64)
        self._host = h.astype(np.int32) if h.size and h.max() < 2 ** 31 else h.astype(np.int64)
        self._dev, self._runs, self.n, self.refs = {}, None, int(h.size), 0
        self._identity = None

    @classmethod
    def from_runs(cls, runs, top: int, size: int) -> "_Index":
        """An index of runs (from `_runs`) into a buffer of `top` elements; `size` is the length
        of the gathered buffer, for `is_identity`. The runs must cover 0..n-1 once each."""
        src, sstep, dst, dstep, length = runs
        ix = cls.__new__(cls)
        ix._host, ix._dev, ix.refs, ix.n = None, {}, 0, int(length.sum())
        ix._identity = (ix.n == size and bool((src == dst).all())
                        and bool(((sstep == dstep) | (length == 1)).all()))
        ix._small = max(top, ix.n) < 2 ** 31
        if (dstep == 1).all():                 # sorted by destination, they need no scatter
            o = np.argsort(dst, kind="stable")
            src, sstep, length, dst = src[o], sstep[o], length[o], dst[o]
            if not np.array_equal(dst, np.cumsum(length) - length):
                raise AssertionError("the runs do not tile the destination")
            ix._runs = np.stack([src, sstep, length]).astype(np.int64)
        else:
            ix._runs = np.stack([src, sstep, length, dst, dstep]).astype(np.int64)
        return ix

    @property
    def nbytes(self) -> int:
        n = 0 if self._runs is None else self._runs.nbytes
        n += 0 if self._host is None else self._host.nbytes
        return n + sum(d.element_size() * d.numel() for d in self._dev.values())

    def _expand(self, xp, device=None):
        """Every position of the runs, computed with `xp` (numpy, or torch on `device`)."""
        unit = bool((self._runs[1] == 1).all())
        if xp is np:
            r = self._runs
            rep = lambda v: np.repeat(v, r[2])
            ar = np.arange(self.n, dtype=np.int64)
        else:
            r = _upload(self._runs, device)
            rep = lambda v: xp.repeat_interleave(v, r[2], output_size=self.n)
            ar = xp.arange(self.n, dtype=xp.int64, device=device)
        first = xp.cumsum(r[2], 0) - r[2]
        if len(r) == 3:
            idx = rep(r[0] - first) + ar if unit else rep(r[0] - first * r[1]) + ar * rep(r[1])
        else:
            rid = rep(xp.arange(r.shape[1]) if xp is np else xp.arange(r.shape[1], device=device))
            local = ar - first[rid]
            idx = xp.empty_like(ar)
            idx[r[3][rid] + local * r[4][rid]] = r[0][rid] + local * r[1][rid]
        if not self._small:
            return idx
        return idx.astype(np.int32) if xp is np else idx.to(xp.int32)

    def _account(self, nbytes: int, t0: float, c0: float):
        dt, stats = time.perf_counter() - t0, _INDEX_CACHE.stats
        stats["build_s"] += dt                   # expansions run outside builds
        stats["build_cpu_s"] += time.thread_time() - c0
        stats["expand_s"] += dt
        if self._runs is not None:               # the expanded index replaces its runs
            nbytes -= self._runs.nbytes
            self._runs = None
        if self.refs:
            _INDEX_CACHE.bytes += nbytes
            _INDEX_CACHE._evict()

    @property
    def host(self) -> np.ndarray:
        if self._host is None:
            t0, c0 = time.perf_counter(), time.thread_time()
            self._host = (self._expand(np) if self._runs is not None
                          else next(iter(self._dev.values())).cpu().numpy())
            self._account(self._host.nbytes, t0, c0)
        return self._host

    def on(self, like):
        if isinstance(like, np.ndarray):
            return self.host
        d = self._dev.get(like.device)
        if d is None:
            mode = os.environ.get("TNKIT_BLOCK_EXPAND", "auto")
            if mode not in ("auto", "host", "device"):
                raise ValueError(f"TNKIT_BLOCK_EXPAND must be auto, host or device, got {mode!r}")
            if self._host is not None or mode == "host":
                h = self.host                    # times its own expansion
                t0, c0 = time.perf_counter(), time.thread_time()
                d = _upload(h, like.device)
            else:
                t0, c0 = time.perf_counter(), time.thread_time()
                if self._runs is None:
                    d = next(iter(self._dev.values())).to(like.device)
                elif mode == "auto" and (like.device.type == "cpu"
                                         or self.n <= _HOST_EXPAND_MAX):
                    d = _upload(self._expand(np), like.device)
                else:
                    import torch
                    d = self._expand(torch, like.device)
            self._dev[like.device] = d
            _INDEX_CACHE.on_device(like.device)
            self._account(d.element_size() * d.numel(), t0, c0)
        if _KEEP is not None:
            _KEEP.add(self)
        return d

    def take(self, x):
        return x[self.host] if isinstance(x, np.ndarray) else x.index_select(0, self.on(x))

    def is_identity(self, n: int) -> bool:
        if self._identity is not None:
            return self._identity and n == self.n
        h = self._host
        return h.size == n and (n == 0 or (h[0] == 0 and h[-1] == n - 1
                                           and bool((np.diff(h) == 1).all())))


def _indexes(v):
    if isinstance(v, _Index):
        yield v
    elif isinstance(v, tuple):
        for x in v:
            yield from _indexes(x)
    elif isinstance(v, _Plan):
        yield from (x for x in (v.ia, v.ib) if x is not None)
    elif isinstance(v, _Gather) and v.ix is not None:
        yield v.ix


def _shared(v):
    """What entry `v` may share with other entries: its indices and, for a plan, its output
    layout (there is one `Struct` per layout)."""
    yield from _indexes(v)
    if isinstance(v, _Plan):
        yield v.out


class _IndexCache:
    """Least-recently-used cache of plans and layout maps, bounded by the bytes they hold on the
    host and on every device (checked on each insertion and when an index reaches a new device).
    `stats` counts hits, misses, evictions and the seconds spent building entries, wall
    (``build_s``) and this thread's CPU (``build_cpu_s``). The byte total is kept as entries come
    and go and as their indices are expanded on devices; an index or output layout held by several
    entries (a plan and its operand's gather, two plans) or by a CUDA graph (`Kept`) is counted
    once, while any of them holds it (`refs`)."""

    def __init__(self, max_bytes: int, source: str):
        self.max_bytes, self._d, self.stats = int(max_bytes), OrderedDict(), Counter()
        self.source = source                     # env, set, default-host or default-device
        self.bytes = self.depth = 0
        self.kept: set = set()                   # unreleased `Kept` records

    def get(self, key, build):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
            self.stats["hits"] += 1
            return v
        t0, c0 = time.perf_counter(), time.thread_time()
        self.depth += 1
        try:
            v = build()
        finally:
            self.depth -= 1
        self.stats["misses"] += 1
        if key[0] == "_plan":
            self.stats["plan_misses"] += 1
        if not self.depth:                       # a nested build is timed by its caller
            self.stats["build_s"] += time.perf_counter() - t0
            self.stats["build_cpu_s"] += time.thread_time() - c0
        self._d[key] = v
        self._track(v, 1)
        self._evict()
        return v

    def _track(self, v, step: int):
        """Count entry `v` in (`step` 1) or out (-1): its own bytes, and each object it shares
        when the first holder comes in or the last goes out."""
        self.bytes += step * getattr(v, "static_nbytes", 0)
        for x in _shared(v):
            x.refs += step
            if x.refs == (1 if step > 0 else 0):
                self.bytes += step * x.nbytes

    def nbytes(self) -> int:
        return self.bytes

    def clear(self):
        for k in list(self.kept):
            k.release()
        for v in self._d.values():
            self._track(v, -1)
        self._d.clear()
        self.stats.clear()

    def _evict(self):
        while len(self._d) > 1 and self.bytes > self.max_bytes:
            self._track(self._d.popitem(last=False)[1], -1)
            self.stats["evictions"] += 1

    def on_device(self, device):
        """Resolve the default limit on the first CUDA device: min(4 GiB, 30% of its memory)."""
        if self.source == "default-host" and device.type == "cuda":
            mb = min(_DEFAULT_MB, int(_DEVICE_FRACTION * _device_memory(device) / 2 ** 20))
            self.max_bytes, self.source = mb * 2 ** 20, "default-device"


def _device_memory(device) -> int:
    import torch
    return torch.cuda.get_device_properties(device).total_memory


def _env_cache(name: str, default_mb: int) -> _IndexCache:
    v = os.environ.get(name)
    if v is None:
        return _IndexCache(default_mb * 2 ** 20, "default-host")
    return _IndexCache(int(float(v) * 2 ** 20), "env")


_DEFAULT_MB, _DEVICE_FRACTION = 4096, 0.3
_INDEX_CACHE = _env_cache("TNKIT_BLOCK_CACHE_MB", _DEFAULT_MB)
_LABEL_CACHE = _env_cache("TNKIT_BLOCK_LABEL_CACHE_MB", 512)


def set_cache_limit(max_bytes: int, label_max_bytes: Optional[int] = None) -> None:
    """Bound the plan and index cache to `max_bytes` (host plus device copies) and, if given, the
    symbolic-plan cache to `label_max_bytes`, evicting now."""
    for c, n in ((_INDEX_CACHE, max_bytes), (_LABEL_CACHE, label_max_bytes)):
        if n is not None:
            c.max_bytes, c.source = int(n), "set"
            c._evict()


def cache_info() -> Dict[str, float]:
    """Hits, misses (all entries; ``plan_misses``: contractions only), evictions, build seconds,
    entries, bytes held, the byte limit and where it came from (``limit_source``: ``env``,
    ``set``, ``default-host`` or ``default-device``); ``label_*``: the same for the symbolic plans
    alone; ``contractions`` and the GEMMs they issued (``gemms``). A plan miss without a label miss
    is a change of sector sizes only.

    ``build_s`` is the wall time of builds not nested in another build (a plan's build includes
    its symbolic plan and its operands' gathers, each once), plus index expansions, which run
    outside builds; ``build_cpu_s`` is the same in this thread's CPU time. ``expand_s`` is the
    expansion time alone."""
    c, s = _INDEX_CACHE, _LABEL_CACHE
    return {**{k: c.stats[k] for k in ("hits", "misses", "plan_misses", "evictions", "build_s",
                                       "expand_s", "build_cpu_s", "contractions", "gemms")},
            "entries": len(c._d), "bytes": c.nbytes(), "max_bytes": c.max_bytes,
            "limit_source": c.source,
            **{f"label_{k}": s.stats[k] for k in ("hits", "misses", "evictions", "build_s")},
            "label_entries": len(s._d), "label_bytes": s.nbytes(), "label_max_bytes": s.max_bytes}


def clear_cache() -> None:
    """Drop every cached plan and index, release every `Kept`, and reset the counters. The
    caches are module-level and not thread-safe."""
    _INDEX_CACHE.clear()
    _LABEL_CACHE.clear()


def _cached(fn):
    def wrapper(*args):
        return _INDEX_CACHE.get((fn.__name__,) + args, lambda: fn(*args))
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


@_cached
def _to_canonical(s: Struct) -> _Index:
    c = canonical(s.legs, s.charge)
    off, st, _ = s.arrays
    coff, cst, cshp = c.arrays
    k = np.array([s.index[b] for b in c.blocks], dtype=np.int64)
    return _Index.from_runs(_runs(off[k], cshp, st[k], coff, cst), s.size, s.size)


@_cached
def _dense_positions(s: Struct) -> Tuple[_Index, _Index]:
    """(buffer positions, flat positions in the sector-ordered dense array) of every element."""
    starts = [np.cumsum((0,) + leg.dims[:-1]) for leg in s.legs]
    shape = tuple(leg.size for leg in s.legs)
    dense = []
    for k, b in enumerate(s.blocks):
        axes = [starts[i][sec] + np.arange(d) for i, (sec, d) in enumerate(zip(b, s.shape(k)))]
        dense.append(np.ravel_multi_index(np.ix_(*axes), shape) if axes else np.zeros(1, np.int64))
    return _Index([s.grid(k) for k in range(len(s.blocks))]), _Index(dense)


@_cached
def _leg_positions(s: Struct, axis: int) -> _Index:
    """For every buffer element, its sector-ordered index along leg `axis`."""
    off, st, shp = s.arrays
    if not len(off):
        return _Index([])
    start = np.cumsum((0,) + s.legs[axis].dims[:-1])
    sec = np.array(s.blocks, np.int64).reshape(len(off), -1)[:, axis]
    step = np.zeros_like(st)
    step[:, axis] = 1
    return _Index.from_runs(_runs(start[sec], shp, step, off, st), s.legs[axis].size, s.size)


def _lib_zeros(like, n):
    return np.zeros(n, dtype=like.dtype) if isinstance(like, np.ndarray) else like.new_zeros(n)


def _take_axis(x, idx: np.ndarray, axis: int):
    if isinstance(x, np.ndarray):
        return np.take(x, idx, axis=axis)
    import torch
    return x.index_select(axis, torch.as_tensor(idx, device=x.device))


def _is_scalar(c) -> bool:
    return isinstance(c, numbers.Number) or getattr(c, "ndim", None) == 0


class BlockTensor:
    """A block-sparse tensor: a host `struct` and a flat, contiguous `data` buffer (numpy or
    torch). It is immutable: operations return new tensors, which may share `data`."""
    __slots__ = ("struct", "_data")
    __array_ufunc__ = None                       # numpy scalars defer to the operators below
    is_block = True

    def __init__(self, struct: Struct, data: Any):
        if tuple(data.shape) != (struct.size,):
            raise ValueError(f"buffer of shape {tuple(data.shape)} for a struct of size "
                             f"{struct.size}")
        if not (data.flags.c_contiguous if isinstance(data, np.ndarray) else data.is_contiguous()):
            raise ValueError("the buffer must be contiguous")
        self.struct, self._data = struct, data

    @property
    def data(self):
        return self._data

    @property
    def legs(self) -> Tuple[Leg, ...]:
        return self.struct.legs

    @property
    def charge(self):
        return self.struct.charge

    @property
    def ndim(self) -> int:
        return len(self.struct.legs)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Dense sizes of the legs."""
        return tuple(leg.size for leg in self.struct.legs)

    @property
    def dtype(self):
        return self._data.dtype

    def __array__(self, *args, **kwargs):
        raise TypeError("a BlockTensor is not a dense array: use .to_dense(), or "
                        "tnkit.block_backend.to_dense_mps for a block MPS")

    def __repr__(self):
        return (f"BlockTensor(shape={self.shape}, charge={self.charge}, "
                f"blocks={len(self.struct.blocks)}, stored={self.struct.size})")

    # --- layout ---
    def transpose(self, perm: Sequence[int]) -> "BlockTensor":
        """Permute the legs. Moves no data."""
        perm = tuple(int(p) for p in perm)
        if sorted(perm) != list(range(self.ndim)):
            raise ValueError(f"{perm} is not a permutation of {self.ndim} legs")
        if perm == tuple(range(self.ndim)):
            return self
        return BlockTensor(_permuted(self.struct, perm), self._data)

    def canonical(self) -> "BlockTensor":
        """The same tensor in the canonical layout: one gather, or none if already there."""
        c = canonical(self.legs, self.charge)
        if self.struct is c:
            return self
        return BlockTensor(c, _to_canonical(self.struct).take(self._data))

    def to_dense(self, orders: Optional[Sequence[Optional[np.ndarray]]] = None):
        """The dense array with every leg in sector order, or, for leg i, in the dense order
        ``orders[i]`` from `Leg.from_charges` (None keeps sector order)."""
        buf, dense = _dense_positions(self.struct)
        flat = _lib_zeros(self._data, math.prod(self.shape))
        flat[dense.on(flat)] = buf.take(self._data)
        x = flat.reshape(self.shape)
        for i, order in enumerate(orders or ()):
            if order is not None:
                x = _take_axis(x, np.argsort(order), i)
        return x

    # --- vector algebra on the flat buffer ---
    def _aligned(self, other: "BlockTensor"):
        if self.struct is other.struct:
            return self._data, other._data, self.struct
        if self.legs != other.legs or self.charge != other.charge:
            raise ValueError("block tensors with different legs or charge")
        a, b = self.canonical(), other.canonical()
        return a._data, b._data, a.struct

    def __add__(self, other):
        if not isinstance(other, BlockTensor):
            return NotImplemented
        a, b, s = self._aligned(other)
        return BlockTensor(s, a + b)

    def __sub__(self, other):
        if not isinstance(other, BlockTensor):
            return NotImplemented
        a, b, s = self._aligned(other)
        return BlockTensor(s, a - b)

    def __mul__(self, c):
        """Multiplication by a scalar (a number, or a 0-d array or tensor) only."""
        if not _is_scalar(c):
            return NotImplemented
        return BlockTensor(self.struct, self._data * c)

    __rmul__ = __mul__

    def __truediv__(self, c):
        if not _is_scalar(c):
            return NotImplemented
        return BlockTensor(self.struct, self._data / c)

    def __neg__(self):
        return BlockTensor(self.struct, -self._data)

    def conj(self) -> "BlockTensor":
        """Complex conjugate, with every leg's sign and the total charge flipped (the dual), so
        a bra contracts with the ket it came from."""
        return BlockTensor(_dual(self.struct), self._data.conj())

    def scale(self, s, axis: int) -> "BlockTensor":
        """Multiply leg `axis` by the vector `s` (one entry per index, in sector order), as a
        dense ``x * s`` broadcast along that leg would."""
        return BlockTensor(self.struct, self._data * _leg_positions(self.struct, axis).take(s))

    # --- fusion (reached through `tnkit.legs.fuse` and `split`) ---
    def fuse(self, groups: Sequence[Sequence[int]]) -> Tuple["BlockTensor", "BlockFuseRecord"]:
        rec = _fuse_record(self.legs, self.charge, tuple(tuple(int(a) for a in g) for g in groups))
        out, idx = _fuse_index(self.struct, rec)
        return BlockTensor(out, idx.take(self._data)), rec

    def split(self, rec: "BlockFuseRecord", legs: Optional[Sequence[int]] = None) -> "BlockTensor":
        """Undo `fuse`. With `legs`, split only those fused legs, each at its own position, and
        keep every other leg (the record's fuse must not have permuted)."""
        if not isinstance(rec, BlockFuseRecord):
            raise TypeError("a block-sparse tensor needs a BlockFuseRecord")
        todo = tuple(range(len(rec.groups))) if legs is None else tuple(sorted(int(g) for g in legs))
        if legs is not None and rec.perm != tuple(range(len(rec.legs))):
            raise ValueError(f"a partial split needs an unpermuted fuse; this one used {rec.perm}")
        if self.ndim != len(rec.groups) or any(self.legs[g] != rec.fused[g] for g in todo):
            raise ValueError("the record does not describe this tensor's fused legs")
        out, idx = _split_index(self.struct, rec, todo)
        return BlockTensor(out, idx.take(self._data))


def from_dense(x, legs: Sequence[Leg], charge=None,
               orders: Optional[Sequence[Optional[np.ndarray]]] = None
               ) -> Tuple[BlockTensor, float]:
    """Project the dense array `x` onto the allowed blocks of `legs` and `charge` (default: the
    trivial sector). Returns the block tensor and the squared norm of the entries it dropped.
    Leg i of `x` is in the dense order ``orders[i]`` (from `Leg.from_charges`), or in sector
    order when that is None."""
    legs = tuple(legs)
    if tuple(x.shape) != tuple(leg.size for leg in legs):
        raise ValueError(f"array of shape {tuple(x.shape)} for legs of sizes "
                         f"{[leg.size for leg in legs]}")
    for i, order in enumerate(orders or ()):
        if order is not None:
            x = _take_axis(x, np.asarray(order), i)
    s = canonical(legs, _sym_of(legs).unit() if charge is None else charge)
    _, dense = _dense_positions(s)
    flat = x.reshape(-1)
    data = flat[dense.on(flat)]
    rest = flat.copy() if isinstance(flat, np.ndarray) else flat.clone()
    rest[dense.on(flat)] = 0
    return BlockTensor(s, data), float(abs(_vdot(rest, rest)))


def _vdot(a, b):
    return np.vdot(a, b) if isinstance(a, np.ndarray) else (a.conj() * b).sum().item()


def random(legs: Sequence[Leg], charge, be, rng) -> BlockTensor:
    """Standard-normal entries in every allowed block, in `be`'s dtype and on its device."""
    s = canonical(tuple(legs), charge)
    return BlockTensor(s, be.randn(rng, (s.size,)))


def zeros(legs: Sequence[Leg], charge, like) -> BlockTensor:
    """A zero tensor with the dtype and device of the buffer `like`."""
    s = canonical(tuple(legs), charge)
    return BlockTensor(s, _lib_zeros(like, s.size))


def concat(a: BlockTensor, b: BlockTensor, axis: int) -> BlockTensor:
    """`a` and `b` joined along leg `axis`. All other legs, the sign of that leg and the total
    charge must agree. Within each sector the indices of `a` come first. Goes through the dense
    array, so it is meant for tensors of MPS size."""
    la, lb = a.legs[axis], b.legs[axis]
    rest = lambda t: t.legs[:axis] + t.legs[axis + 1:]
    if rest(a) != rest(b) or la.sign != lb.sign or a.charge != b.charge:
        raise ValueError("concat needs equal legs except along `axis`, and equal charges")
    da, db = dict(zip(la.charges, la.dims)), dict(zip(lb.charges, lb.dims))
    qs = sorted(set(da) | set(db), key=la.sym.sort_key)
    leg = Leg(tuple(qs), tuple(da.get(q, 0) + db.get(q, 0) for q in qs), la.sign, la.sym)
    start = dict(zip(qs, np.cumsum((0,) + leg.dims[:-1])))
    pa = [start[q] + np.arange(n) for q, n in zip(la.charges, la.dims)]
    pb = [start[q] + da.get(q, 0) + np.arange(n) for q, n in zip(lb.charges, lb.dims)]
    x, y = a.to_dense(), b.to_dense()
    if isinstance(x, np.ndarray):
        xy = np.concatenate([x, y], axis)
    else:
        import torch
        xy = torch.cat([x, y], axis)
    xy = _take_axis(xy, np.argsort(np.concatenate(pa + pb)), axis)
    legs = list(a.legs)
    legs[axis] = leg
    return from_dense(xy, legs, a.charge)[0]


# --- contraction ---

def _label_groups(labels, axes, sym: Symmetry) -> Dict[Any, List[Tuple[int, ...]]]:
    """Sector tuples over `axes`, grouped by the sector their signed labels fuse to, each group
    in lexicographic order."""
    groups: Dict[Any, list] = {}
    for t in itertools.product(*(range(len(labels[a][0])) for a in axes)):
        q = sym.total([labels[a][0][s] for a, s in zip(axes, t)], [labels[a][1] for a in axes])
        groups.setdefault(q, []).append(t)
    return groups


def _sector_groups(legs: Sequence[Leg], axes: Sequence[int]):
    """Sector tuples over `axes`, grouped by signed charge: {q: ([(tuple, offset, dims)], size)},
    each group in lexicographic order with offsets packed."""
    out = {}
    for q, ts in _label_groups([(leg.charges, leg.sign) for leg in legs], axes,
                               _sym_of(legs)).items():
        items, off = [], 0
        for t in ts:
            dims = tuple(legs[a].dims[s] for a, s in zip(axes, t))
            items.append((t, off, dims))
            off += math.prod(dims)
        out[q] = (items, off)
    return out


def _block(n: int, axes_parts) -> Tuple[int, ...]:
    b = [0] * n
    for axes, t in axes_parts:
        for a, s in zip(axes, t):
            b[a] = s
    return tuple(b)


class _SymPlan:
    """The size-free part of a contraction plan. Per contracted charge (a GEMM group) it lists
    row, inner (k) and column items, each a sector tuple over the free legs of a, the contracted
    legs, or the free legs of b; ``a_box``/``b_box`` say which block of each operand fills which
    (row, k) or (k, column) item pair, and ``out_rc`` which (row, column) pair is which output
    block (``out_blocks``). Items are numbered globally, group by group."""
    TABLES = ("r_sec", "k_sec", "c_sec", "r_grp", "k_grp", "c_grp", "r_beg", "k_beg", "c_beg",
              "r_end", "k_end", "c_end", "a_box", "b_box", "out_rc")
    __slots__ = TABLES + ("out_blocks", "charge", "static_nbytes")


def _sym_plan(la, lb, ax_a: Tuple[int, ...], ax_b: Tuple[int, ...]) -> _SymPlan:
    (legs_a, qa, blocks_a, sym), (legs_b, qb, blocks_b, _) = la, lb
    fa = tuple(i for i in range(len(legs_a)) if i not in ax_a)
    fb = tuple(j for j in range(len(legs_b)) if j not in ax_b)
    rows, ks, cols = (_label_groups(legs_a, fa, sym), _label_groups(legs_a, ax_a, sym),
                      _label_groups(legs_b, fb, sym))
    index_a = {b: k for k, b in enumerate(blocks_a)}
    index_b = {b: k for k, b in enumerate(blocks_b)}
    cs = ({sym.fuse(qa, sym.dual(f)) for f in rows} & {sym.fuse(g, sym.dual(qb)) for g in cols})
    items = {"r": [], "k": [], "c": []}
    grp = {"r": [], "k": [], "c": []}
    a_box, b_box, out_rc, out_blocks = [], [], [], []
    na, nb = len(legs_a), len(legs_b)
    for gi, c in enumerate(sorted(cs, key=sym.sort_key)):
        r_items, c_items = rows[sym.fuse(qa, sym.dual(c))], cols[sym.fuse(qb, c)]
        k_items = ks.get(c, [])
        r0, k0, c0 = len(items["r"]), len(items["k"]), len(items["c"])
        for key, its in (("r", r_items), ("k", k_items), ("c", c_items)):
            items[key] += its
            grp[key] += [gi] * len(its)
        for kj, kt in enumerate(k_items):
            for ri, rt in enumerate(r_items):
                a_box.append((index_a[_block(na, ((fa, rt), (ax_a, kt)))], r0 + ri, k0 + kj))
            for ci, ct in enumerate(c_items):
                b_box.append((index_b[_block(nb, ((ax_b, kt), (fb, ct)))], k0 + kj, c0 + ci))
        for ri, rt in enumerate(r_items):
            for ci, ct in enumerate(c_items):
                out_rc.append((r0 + ri, c0 + ci))
                out_blocks.append(rt + ct)
    p = _SymPlan()
    n_groups = len(cs)
    for key, axes in (("r", fa), ("k", ax_a), ("c", fb)):
        sec = np.array(items[key], dtype=np.int64).reshape(len(items[key]), len(axes))
        g = np.array(grp[key], dtype=np.int64)
        setattr(p, f"{key}_sec", sec)
        setattr(p, f"{key}_grp", g)
        setattr(p, f"{key}_beg", np.searchsorted(g, np.arange(n_groups), side="left"))
        setattr(p, f"{key}_end", np.searchsorted(g, np.arange(n_groups), side="right"))
    p.a_box = np.array(a_box, dtype=np.int64).reshape(len(a_box), 3)
    p.b_box = np.array(b_box, dtype=np.int64).reshape(len(b_box), 3)
    p.out_rc = np.array(out_rc, dtype=np.int64).reshape(len(out_rc), 2)
    p.out_blocks = tuple(out_blocks)
    p.charge = sym.fuse(qa, qb)
    p.static_nbytes = (sum(getattr(p, k).nbytes for k in _SymPlan.TABLES)
                       + 100 * len(out_blocks))          # tables, and roughly the block tuples
    return p


class _Plan:
    """A cached contraction: one gather per operand (None when the buffer is already the stacked
    matrices), then per contracted charge one ``(R, K) @ (K, C)`` GEMM at the listed offsets;
    K = 0 marks an allowed output region with no contributions, which is zero-filled. `nmm` is
    the number of GEMMs with K > 0."""
    __slots__ = ("ia", "ib", "gemms", "out", "macs", "nmm")


class _Gather:
    """A cached operand gather: its index (None when the buffer already is the stacked matrices)
    and its item offsets (`_side`)."""
    __slots__ = ("ix", "offs", "static_nbytes")

    def __init__(self, ix: Optional["_Index"], offs):
        self.ix, self.offs = ix, offs
        self.static_nbytes = sum(a.nbytes for a in offs)


def _item_shapes(sec: np.ndarray, axes, legs) -> np.ndarray:
    return _block_shapes([legs[a] for a in axes], sec)


def _group_offsets(n: np.ndarray, grp: np.ndarray, beg: np.ndarray, end: np.ndarray):
    """For items of sizes `n` listed group by group (group g is items beg[g]:end[g]): each item's
    offset inside its group, and each group's total."""
    cum = np.zeros(len(n) + 1, np.int64)
    np.cumsum(n, out=cum[1:])
    first = cum[beg]
    return cum[:-1] - first[grp], cum[end] - first


def _side(sp: _SymPlan, s: Struct, axes, rows: bool):
    """Item offsets of one operand: for `rows` (operand a) its row and k items, else (operand b)
    its k and column items. Returns (outer shapes, outer offsets, outer totals, k offsets,
    k totals) per item and group."""
    sec, grp, beg, end = ((sp.r_sec, sp.r_grp, sp.r_beg, sp.r_end) if rows else
                          (sp.c_sec, sp.c_grp, sp.c_beg, sp.c_end))
    free = tuple(i for i in range(len(s.legs)) if i not in axes)
    shp = _item_shapes(sec, free, s.legs)
    o, n = _group_offsets(shp.prod(1), grp, beg, end)
    ko, K = _group_offsets(_item_shapes(sp.k_sec, axes, s.legs).prod(1), sp.k_grp, sp.k_beg,
                           sp.k_end)
    return shp, o, n, ko, K


def _operand(sp: _SymPlan, lkey, s: Struct, axes, side: int):
    """For operand `side` (0: a, 1: b) of the plan `sp` (cached under `lkey`), a `_Gather`: the
    index that stacks it into the GEMM matrices, and its item offsets. Depends on this operand
    alone, so it is cached apart from the plan, keyed by `lkey` so that it does not keep `sp`
    alive."""
    def build():
        offs = _side(sp, s, axes, side == 0)
        _, o, n, ko, K = offs
        box = sp.a_box if side == 0 else sp.b_box
        free = tuple(i for i in range(len(s.legs)) if i not in axes)
        perm = list(free + axes) if side == 0 else list(axes + free)
        g = sp.k_grp[box[:, 2 - side]]
        if side == 0:      # (R, K) matrices: rows = free legs, row stride K
            nrow, stride = len(free), K[g]
            dst0 = (np.cumsum(n * K) - n * K)[g] + o[box[:, 1]] * K[g] + ko[box[:, 2]]
        else:              # (K, C) matrices: rows = contracted legs, row stride C
            nrow, stride = len(axes), n[g]
            dst0 = (np.cumsum(K * n) - K * n)[g] + ko[box[:, 1]] * n[g] + o[box[:, 2]]
        off, st, shp = s.arrays
        blk = box[:, 0]
        bs, ss = shp[blk][:, perm], st[blk][:, perm]
        ds = np.concatenate([_row_major_rows(bs[:, :nrow]) * stride[:, None],
                             _row_major_rows(bs[:, nrow:])], 1)
        ix = _Index.from_runs(_runs(off[blk], bs, ss, dst0, ds), s.size, s.size)
        return _Gather(None if ix.is_identity(s.size) else ix, offs)
    return _INDEX_CACHE.get(("_gather", lkey, s, side), build)


@_cached
def _plan(sa: Struct, sb: Struct, ax_a: Tuple[int, ...], ax_b: Tuple[int, ...]) -> _Plan:
    for i, j in zip(ax_a, ax_b):
        la, lb = sa.legs[i], sb.legs[j]
        if (la.charges != lb.charges or la.dims != lb.dims or la.sign != -lb.sign
                or la.sym != lb.sym):
            raise ValueError(f"cannot contract leg {i} {la} with leg {j} {lb}: the sectors must "
                             f"match and the signs be opposite")
    lkey = (sa.labels, sb.labels, ax_a, ax_b)
    sp = _LABEL_CACHE.get(lkey, lambda: _sym_plan(*lkey))
    p = _Plan()
    ga, gb = _operand(sp, lkey, sa, ax_a, 0), _operand(sp, lkey, sb, ax_b, 1)
    p.ia, p.ib = ga.ix, gb.ix
    (r_shp, ro, R, _, K), (c_shp, co, C, _, _) = ga.offs, gb.offs
    excl = lambda n: np.cumsum(n) - n
    pa, pb, po = excl(R * K), excl(K * C), excl(R * C)
    p.gemms = tuple(zip(*(x.tolist() for x in (pa, pb, po, R, K, C))))
    p.macs = int((R * K * C).sum())
    p.nmm = int(np.count_nonzero(K))
    orr, oc = sp.out_rc[:, 0], sp.out_rc[:, 1]
    go = sp.r_grp[orr]
    st = np.concatenate([_row_major_rows(r_shp[orr]) * C[go][:, None],
                         _row_major_rows(c_shp[oc])], 1)
    p.out = _struct(tuple(sa.legs[i] for i in range(len(sa.legs)) if i not in ax_a)
                    + tuple(sb.legs[j] for j in range(len(sb.legs)) if j not in ax_b), sp.charge,
                    sp.out_blocks, po[go] + ro[orr] * C[go] + co[oc], st,
                    np.concatenate([r_shp[orr], c_shp[oc]], 1), (R * C).sum(), sa.sym)
    return p


class Kept:
    """What a CUDA graph reads, recorded by `keep_operands`: every gather index used on a torch
    device and every contraction's operand buffers. Until `release`, the indices stay alive when
    the cache evicts them, and count toward its bytes and limit as held (`refs`), once each however
    many entries and records hold them; the cache never frees them, so while they alone exceed the
    limit it keeps one entry. The buffers are the caller's or the graph pool's and are not counted;
    they are kept so that a matvec which rebinds its operands mid-solve leaves the graph reading
    stale but valid memory. `clear_cache` releases every record; a graph whose record is released
    must not be replayed, and a released record records nothing more."""
    __slots__ = ("held", "released")

    def __init__(self):
        self.held: Dict[int, Any] = {}
        self.released = False

    def add(self, x) -> None:
        if self.released:
            return
        if id(x) not in self.held:
            self.held[id(x)] = x
            if isinstance(x, _Index):
                _INDEX_CACHE._track(x, 1)

    def release(self) -> None:
        if not self.released:
            self.released = True
            _INDEX_CACHE.kept.discard(self)
            for x in self.held.values():
                if isinstance(x, _Index):
                    _INDEX_CACHE._track(x, -1)
            self.held.clear()


_KEEP: Optional[Kept] = None      # while a CUDA graph is captured: what its kernels read


@contextlib.contextmanager
def keep_operands():
    """A new `Kept` that records, within the block, what each contraction and index use reads.
    Release it when the graph captured meanwhile is dropped; it is released if the block raises."""
    global _KEEP
    into = Kept()
    _INDEX_CACHE.kept.add(into)
    prev, _KEEP = _KEEP, into
    try:
        yield into
    except BaseException:
        into.release()
        raise
    finally:
        _KEEP = prev


def tensordot(a: BlockTensor, b: BlockTensor,
              axes: Tuple[Sequence[int], Sequence[int]]) -> BlockTensor:
    """Contract legs ``axes[0]`` of `a` with legs ``axes[1]`` of `b`. The result has the free legs
    of `a`, then those of `b`, in order, and total charge ``a.charge + b.charge``; contracting
    every leg gives a tensor with no legs and one element."""
    p = _plan(a.struct, b.struct, tuple(int(i) for i in axes[0]), tuple(int(j) for j in axes[1]))
    A = a.data if p.ia is None else p.ia.take(a.data)
    B = b.data if p.ib is None else p.ib.take(b.data)
    stats = _INDEX_CACHE.stats
    stats["contractions"] += 1
    stats["gemms"] += p.nmm
    if _KEEP is not None:
        _KEEP.add(a.data)
        _KEEP.add(b.data)
    if isinstance(A, np.ndarray):
        out = np.empty(p.out.size, dtype=np.result_type(A, B))
        for pa, pb, po, R, K, C in p.gemms:
            o = out[po:po + R * C].reshape(R, C)
            if K:
                np.matmul(A[pa:pa + R * K].reshape(R, K), B[pb:pb + K * C].reshape(K, C), out=o)
            else:
                o.fill(0)
        return BlockTensor(p.out, out)
    import torch
    if A.dtype != B.dtype:                               # torch.matmul does not promote
        dt = torch.promote_types(A.dtype, B.dtype)
        A, B = A.to(dt), B.to(dt)
    out = A.new_empty(p.out.size)
    a0, b0, o0 = A.storage_offset(), B.storage_offset(), out.storage_offset()
    for pa, pb, po, R, K, C in p.gemms:                 # one strided view per operand
        o = out.as_strided((R, C), (C, 1), o0 + po)
        if K:
            torch.matmul(A.as_strided((R, K), (K, 1), a0 + pa),
                         B.as_strided((K, C), (C, 1), b0 + pb), out=o)
        else:
            o.zero_()
    return BlockTensor(p.out, out)


# --- fusion records ---

@dataclass(frozen=True)
class BlockFuseRecord(FuseRecord):
    """A `FuseRecord` for block-sparse legs. `legs` and `charge` are the unfused tensor's. Fused
    leg g is ``fused[g]`` (sign +1); its sector k stacks the sub-blocks ``sectors[g][k]``, each
    ``(sector tuple over groups[g], row offset, size)``, in lexicographic order. A densified
    fused tensor therefore has its fused legs in this sector order, not in the row-major order
    a dense `fuse` produces, and only a block tensor can be split with this record.

    Because fused legs have sign +1, fuse first and conjugate after: ``fuse(t)`` contracts with
    its conjugate ``fuse(t).conj()``, but not with ``fuse(t.conj())``."""
    legs: Tuple[Leg, ...] = ()
    charge: Any = 0
    fused: Tuple[Leg, ...] = ()
    sectors: Tuple[Tuple[Tuple[Tuple[Tuple[int, ...], int, int], ...], ...], ...] = ()


@lru_cache(maxsize=_META_CACHE)
def _fuse_record(legs: Tuple[Leg, ...], charge, groups) -> BlockFuseRecord:
    fused, sectors = [], []
    for g in groups:
        by_q = _sector_groups(legs, g)
        sym = _sym_of(legs)
        qs = sorted(by_q, key=sym.sort_key)
        sectors.append(tuple(tuple((t, off, math.prod(d)) for t, off, d in by_q[q][0]) for q in qs))
        fused.append(Leg(tuple(qs), tuple(by_q[q][1] for q in qs), 1, sym))
    return BlockFuseRecord(tuple(leg.size for leg in legs), groups, copied=True, legs=legs,
                           charge=charge, fused=tuple(fused), sectors=tuple(sectors))


@_cached
def _fuse_index(s: Struct, rec: BlockFuseRecord) -> Tuple[Struct, _Index]:
    out = canonical(rec.fused, rec.charge)
    perm, n = list(rec.perm), len(rec.legs)
    boxes = []                        # (src block, dst offset, dst strides in perm order)
    for k, fb in enumerate(out.blocks):
        ost = out.strides[k]
        for subs in itertools.product(*(rec.sectors[g][fb[g]] for g in range(len(rec.groups)))):
            kb = s.index[_block(n, ((rec.groups[g], t) for g, (t, _, _) in enumerate(subs)))]
            dims = s.shape(kb)
            ds = [r * ost[g] for g, grp in enumerate(rec.groups)
                  for r in _row_major([dims[a] for a in grp])]
            boxes.append((kb, out.offsets[k] + sum(off * ost[g] for g, (_, off, _)
                                                   in enumerate(subs)), ds))
    return out, _box_index(s, boxes, perm, out.size)


@_cached
def _split_index(s: Struct, rec: BlockFuseRecord, todo: Tuple[int, ...]) -> Tuple[Struct, _Index]:
    """Split the fused legs `todo` of a tensor laid out as `s`; other legs are kept as they are.
    A full split also undoes the fuse's leg permutation."""
    where = {g: {t: (k, off, n) for k, sec in enumerate(rec.sectors[g]) for t, off, n in sec}
             for g in todo}
    legs, origin = [], []                  # in group order; origin = (leg of s, place in group)
    for g in range(len(s.legs)):
        if g in where:
            legs += [rec.legs[a] for a in rec.groups[g]]
            origin += [(g, p) for p in range(len(rec.groups[g]))]
        else:
            legs.append(s.legs[g])
            origin.append((g, None))
    full = len(todo) == len(rec.groups)
    perm = tuple(int(i) for i in np.argsort(rec.perm)) if full else tuple(range(len(legs)))
    out = canonical(tuple(legs[i] for i in perm), s.charge)
    rows = []                         # (src block, src offset, src strides in out order)
    for b in out.blocks:
        bg = [0] * len(legs)
        for j, i in enumerate(perm):
            bg[i] = b[j]
        fb, sub = [0] * len(s.legs), {}
        for i, (g, p) in enumerate(origin):
            if p is None:
                fb[g] = bg[i]
            else:
                sub.setdefault(g, []).append(bg[i])
        start = {}
        for g, t in sub.items():
            fb[g], start[g], _ = where[g][tuple(t)]
        kb = s.index[tuple(fb)]
        sst = s.strides[kb]
        inner = {g: _row_major([legs[i].dims[bg[i]] for i, (h, p) in enumerate(origin)
                                if h == g and p is not None]) for g in sub}
        st = [sst[g] if p is None else sst[g] * inner[g][p] for g, p in origin]
        rows.append((kb, s.offsets[kb] + sum(start[g] * sst[g] for g in sub),
                     [st[i] for i in perm]))
    off, st, shp = out.arrays
    if not rows:
        return out, _Index([])
    kb, src0, ss = (np.array(x, dtype=np.int64) for x in zip(*rows))
    return out, _Index.from_runs(_runs(src0, shp, ss.reshape(shp.shape), off, st), s.size,
                                 out.size)


def _box_index(s: Struct, boxes, perm, size: int) -> _Index:
    """The index that copies, for each box (block of `s`, destination offset, destination
    strides), that block with its legs in order `perm`."""
    if not boxes:
        return _Index([])
    kb, dst0, ds = zip(*boxes)
    kb = np.array(kb, dtype=np.int64)
    off, st, shp = s.arrays
    return _Index.from_runs(_runs(off[kb], shp[kb][:, perm], st[kb][:, perm],
                                  np.array(dst0, dtype=np.int64),
                                  np.array(ds, dtype=np.int64).reshape(len(kb), len(perm))),
                            s.size, size)


# --- truncated SVD ---

def _out_label(leg: Leg, r: int):
    """The label of row sector `r` as seen from the far side of a new bond."""
    return leg.charges[r] if leg.sign > 0 else leg.sym.dual(leg.charges[r])


def _matrix_blocks(m: BlockTensor):
    """(row sector, col sector, R x C matrix view) of a canonical two-leg block tensor."""
    s = m.struct
    for k, (r, c) in enumerate(s.blocks):
        R, C = s.shape(k)
        yield r, c, m.data[s.offsets[k]:s.offsets[k] + R * C].reshape(R, C)


def _svd_lapack(M, driver: Optional[str]):
    """Economy SVD. On CUDA `driver` defaults to gesvd: torch's default, gesvdj, is much slower
    at the same accuracy. If the first attempt fails to converge, retry once with gesvd."""
    if isinstance(M, np.ndarray):
        try:
            return np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            import scipy.linalg
            return scipy.linalg.svd(M, full_matrices=False, lapack_driver="gesvd")
    import torch
    if M.is_cuda:
        d = driver or "gesvd"
        try:
            return torch.linalg.svd(M, full_matrices=False, driver=d)
        except torch.linalg.LinAlgError:
            if d == "gesvd":
                raise
            return torch.linalg.svd(M, full_matrices=False, driver="gesvd")
    try:
        return torch.linalg.svd(M, full_matrices=False)
    except torch.linalg.LinAlgError:
        import scipy.linalg
        x = M.detach().cpu().numpy()
        out = scipy.linalg.svd(x, full_matrices=False, lapack_driver="gesvd")
        return tuple(torch.as_tensor(y, device=M.device) for y in out)


def _svd_block(M, method: str, driver: Optional[str]):
    """U, S (descending), Vh of one block, by `_svd_lapack` or through the smaller Gram
    matrix (`'gram'`: Hermitian eigendecomposition, then a QR that re-orthonormalises the
    computed singular vectors)."""
    torch_ = not isinstance(M, np.ndarray)
    if torch_:
        import torch
        la = torch.linalg
    else:
        la = np.linalg
    if method == "svd":
        return _svd_lapack(M, driver)
    if method != "gram":
        raise ValueError(f"unknown method {method!r}; expected 'svd' or 'gram'")
    R, C = M.shape
    H = (lambda x: x.mH) if torch_ else (lambda x: x.conj().T)
    flip = (lambda x: torch.flip(x, (-1,))) if torch_ else (lambda x: x[..., ::-1])
    if R >= C:
        w, V = la.eigh(H(M) @ M)
        V = flip(V)
        Q, Rr = la.qr(M @ V)
    else:
        w, U = la.eigh(M @ H(M))
        U = flip(U)
        Q, Rr = la.qr(H(M) @ U)                  # columns of V, scaled by S
    d = Rr.diagonal() if torch_ else np.diagonal(Rr)
    S = abs(d)
    Q = Q * (d / (S + (S == 0)))                 # unit phases; 1 where S == 0
    o = torch.argsort(S, descending=True) if torch_ else np.argsort(-S, kind="stable")
    S, Q = S[o], Q[:, o]
    if R >= C:
        return Q, S, H(V[:, o])
    return U[:, o], S, H(Q)


def truncated_svd(m: BlockTensor, chi: int, *, svd_tol: float = 0.0, method: str = "svd",
                  driver: Optional[str] = None, bucket: int = 1, zero_tol: float = 0.0,
                  degenerate_tol: float = 0.0, info: Optional[dict] = None):
    """``m ~ U diag(S) Vh`` for a two-leg block tensor, keeping at most `chi` singular values
    chosen across all sectors together, as a dense truncated SVD of the same matrix would. With
    `bucket` > 1 the new leg can be larger than `chi` (see `bucket`).

    Returns ``(U, S, Vh, disc)``: `U` has legs ``(m.legs[0], k)``, `Vh` has ``(k, m.legs[1])``,
    `S` is a backend vector over the new leg in sector order, and `disc` is the absolute
    discarded weight sum(sigma_dropped^2). The new leg ``k`` (sign -1 on `U`, +1 on `Vh`) is
    labelled by the charge `U` carries out. One host copy of the singular values per call.

    svd_tol  : > 0 keeps the smallest rank whose relative dropped weight stays under svd_tol,
               still capped at chi (the two-site TDVP rule).
    zero_tol : drop singular values at or below zero_tol * max(S) before the cap.
    bucket   : round each kept sector size up to a multiple of `bucket` with zero singular
               values and zero vectors, so that repeated truncations repeat the same layout
               and reuse their plans. The padding changes no physical quantity, but adds up to
               ``bucket - 1`` entries per kept sector, so the new leg can exceed `chi`.
    method, driver : per-block factorisation, see `_svd_block`; `driver` is passed to
               ``torch.linalg.svd`` on CUDA (default gesvd) and ignored elsewhere.
    degenerate_tol : > 0 keeps or drops a group of singular values within degenerate_tol of
               each other whole, across sectors as within one, by the dense rule
               `tnkit._dmrg._cut`.
    info     : optional dict, filled with ``pinned``: True when degenerate_tol moved the cap cut
               below the cap and svd_tol did not cut lower.
    """
    if m.ndim != 2:
        raise ValueError(f"truncated_svd needs a two-leg tensor, got {m.ndim} legs")
    if chi < 1:
        raise ValueError(f"chi must be at least 1, got {chi}")
    if bucket < 1:
        raise ValueError(f"bucket must be at least 1, got {bucket}")
    m = m.canonical()
    lr, lc = m.legs
    parts = [(r, c) + tuple(_svd_block(M, method, driver)) for r, c, M in _matrix_blocks(m)]
    if not parts:
        raise ValueError("the matrix has no allowed blocks")
    torch_ = not isinstance(m.data, np.ndarray)
    if torch_:
        import torch
        cat = torch.cat
        s_host = torch.cat([p[3] for p in parts]).cpu().numpy().astype(np.float64)
    else:
        cat = np.concatenate
        s_host = np.concatenate([p[3] for p in parts]).astype(np.float64)
    owner = np.concatenate([np.full(len(p[3]), j) for j, p in enumerate(parts)])
    order = np.lexsort((np.arange(s_host.size), -s_host))      # descending, stable
    sq = s_host[order] ** 2
    tot = sq.sum() or 1.0
    live = s_host.size
    if zero_tol > 0:
        live = max(1, int((s_host > zero_tol * s_host.max()).sum()))
    from ._dmrg import _cut
    # values below zero_tol leave the spectrum before the cap, so they never pin it
    k, pinned, _ = _cut(s_host[order][:live], int(chi), svd_tol, degenerate_tol,
                        float(sq[live:].sum()) / tot, tot)
    if info is not None:
        info["pinned"] = pinned
    kept = np.bincount(owner[order[:k]], minlength=len(parts))
    disc = float(sq[k:].sum())
    # the new leg has one sector per block with kept values, labelled by the charge U carries out
    sym = m.struct.sym
    new = sorted(((_out_label(lr, r), r, U[:, :n], S[:n], Vh[:n], -n % bucket)
                  for (r, c, U, S, Vh), n in zip(parts, map(int, kept)) if n),
                 key=lambda x: (sym.sort_key(x[0]), x[1]))
    bond = Leg(tuple(q for q, *_ in new), tuple(len(S) + pad for _, _, _, S, _, pad in new), 1,
               sym)

    def pad(x, n, axis):
        if not n:
            return x
        shape = list(x.shape)
        shape[axis] = n
        z = x.new_zeros(shape) if torch_ else np.zeros(shape, x.dtype)
        return cat([x, z], axis)

    us = [(r, pad(U, n, 1)) for _, r, U, _, _, n in new]
    U = cat([u.reshape(-1) for _, u in sorted(us, key=lambda x: x[0])])     # canonical: by row sector
    S = cat([pad(S, n, 0) for _, _, _, S, _, n in new])
    Vh = cat([pad(Vh, n, 0).reshape(-1) for _, _, _, _, Vh, n in new])     # canonical: by new sector
    return (BlockTensor(canonical((lr, bond.flip()), sym.unit()), U), S,
            BlockTensor(canonical((bond, lc), m.charge), Vh), disc)


def qr(m: BlockTensor, right: bool = False) -> Tuple[BlockTensor, BlockTensor]:
    """``m = Q R`` block by block for a two-leg block tensor, Q with orthonormal columns; with
    `right`, ``m = L Q`` with Q having orthonormal rows. The new leg is labelled as in
    `truncated_svd` and has one sector per block, of size min(rows, cols) of that block."""
    if m.ndim != 2:
        raise ValueError(f"qr needs a two-leg tensor, got {m.ndim} legs")
    m = m.canonical()
    lr, lc = m.legs
    torch_ = not isinstance(m.data, np.ndarray)
    if torch_:
        import torch
        la, cat, H = torch.linalg, torch.cat, (lambda x: x.mH)
    else:
        la, cat, H = np.linalg, np.concatenate, (lambda x: x.conj().T)
    new = []
    for r, c, M in _matrix_blocks(m):
        if right:
            q, t = la.qr(H(M))
            a, b = H(t), H(q)
        else:
            a, b = la.qr(M)
        new.append((_out_label(lr, r), r, a, b))
    if not new:
        raise ValueError("the matrix has no allowed blocks")
    sym = m.struct.sym
    new.sort(key=lambda x: sym.sort_key(x[0]))
    bond = Leg(tuple(q for q, *_ in new), tuple(int(a.shape[1]) for _, _, a, _ in new), 1, sym)
    A = cat([a.reshape(-1) for _, _, a, _ in sorted(new, key=lambda x: x[1])])
    B = cat([b.reshape(-1) for _, _, _, b in new])
    return (BlockTensor(canonical((lr, bond.flip()), sym.unit()), A),
            BlockTensor(canonical((bond, lc), m.charge), B))
