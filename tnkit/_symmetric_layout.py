"""Tree structure and buffer layout of symmetric tensors, and the caches that keep them.

`Trees` holds every splitting tree of one side for given sectors and dual flags; `Side` adds the
degeneracies; `Layout` is the space of maps from a domain to a codomain: its coupled sectors
and the shape and offset of each block in the flat buffer. Layouts are interned weakly, one live
object per space (`layout`). A layout builds the side with fewer leaf tuples first and enumerates
the other only for the coupled sectors of the first.

Trees store labels as integer ids shared by the symmetry (`label_ids`), whose table of fusions
lets a side grow one leg at a time in array operations.

The maps that move a tensor's buffer to another layout (recoupling, fusing, splitting, products)
live in `INDEX_CACHE` on the host (numpy, torch on the CPU) and in `DEVICE_INDEX_CACHE` on other
devices, keyed by the source layout, the backend type and device, and the operation. They go
when the source layout (or a partner layout) dies or the byte limit evicts them. A map holds the
layout it produces only if it created it (`Target`), so references between layouts never form a
cycle. A layout lives while a tensor uses it, while the map that created it is cached, or while
it is one of the layouts most recently mapped from or produced by `contract` (`RECENT_LAYOUTS`,
bounded by entries and bytes).
That last rule keeps the maps of short-lived intermediates, such as the operands of the products
in a matrix-vector step, from one step to the next.
"""
from __future__ import annotations

import itertools
import math
import threading
import weakref
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Dict, Tuple

import numpy as np

from ._bytecache import ByteCache, env_cache, env_number
from .fusion_trees import FusionTree

# Default limits, from the caches' use in two-site TDVP on two-flavour chains (U(1) x SU(2)):
# one site update at a cap of 384 multiplets serves up to 141 MB of maps, 0.4 MB of trees and
# 34 layouts. The structural caches hold a sweep of 160 sites (318 updates), the maps several
# updates; on a CUDA device the maps take min(2 GB, a tenth of its memory). Each limit can be
# set in MiB by an environment variable, read at import.
STRUCTURE_CACHE = env_cache("TNKIT_SYMMETRIC_STRUCTURE_CACHE_MB", 512)    # trees of one side
INDEX_CACHE = env_cache("TNKIT_SYMMETRIC_INDEX_CACHE_MB", 1024)           # maps on the host
INDEX_DEVICE_MB, INDEX_CUDA_FRACTION = 2048, 0.1
DEVICE_INDEX_CACHE = env_cache("TNKIT_SYMMETRIC_DEVICE_INDEX_CACHE_MB", INDEX_DEVICE_MB)
_RECENT = env_number("TNKIT_SYMMETRIC_RECENT_LAYOUTS", 32768, integer=True)
RECENT_LAYOUTS = env_cache("TNKIT_SYMMETRIC_RECENT_LAYOUTS_MB", 256,      # kept alive by uid
                           32768 if _RECENT is None else _RECENT)


def backend_key(be):
    """Maps hold index arrays on a device, and cast their matrices to the dtype of the data
    they meet, so one map serves real and complex tensors alike."""
    return (type(be).__name__, str(be.device))


def index_cache(be) -> ByteCache:
    """The cache of maps on the backend's device. Unless its limit was set, the device cache
    takes min(`INDEX_DEVICE_MB`, `INDEX_CUDA_FRACTION` of the memory) of the first CUDA device
    it meets."""
    if not be.is_torch or be.device.type == "cpu":
        return INDEX_CACHE
    c = DEVICE_INDEX_CACHE
    if c.source == "default" and be.device.type == "cuda":
        total = be.torch.cuda.get_device_properties(be.device).total_memory
        c.set_limit(min(INDEX_DEVICE_MB * 2 ** 20, int(INDEX_CUDA_FRACTION * total)),
                    source="default-device")
    return c


INDEX32_LIMIT = 2 ** 31


def index_dtype(n: int):
    """The dtype of positions in a buffer of n entries: int32 below `INDEX32_LIMIT`, else
    int64."""
    return np.int32 if n < INDEX32_LIMIT else np.int64


def to_index(be, a, n: int):
    """An index array into a buffer of n entries (`index_dtype`), on the backend's device."""
    if be.is_torch and isinstance(a, be.torch.Tensor):
        dt = be.torch.int32 if index_dtype(n) == np.int32 else be.torch.int64
        return a.to(device=be.device, dtype=dt).contiguous()
    a = np.ascontiguousarray(a, index_dtype(n))
    return be.torch.as_tensor(a, device=be.device) if be.is_torch else a


# --------------------------------------------------------------------------- trees of one side
@lru_cache(maxsize=1 << 16)
def fusions(sym, a, b):
    return sym.fusions(a, b)


def _sort_rows(keys):
    """The permutation that sorts rows by `keys` (non-negative integer arrays, the first most
    significant): one argsort of a mixed-radix code when it fits in int64, else a lexsort."""
    sizes = [int(k.max()) + 1 if len(k) else 1 for k in keys]
    if math.prod(sizes) >= 2 ** 62:
        return np.lexsort(keys[::-1])
    code = np.zeros(len(keys[0]), np.int64)
    for k, n in zip(keys, sizes):
        code = code * n + k
    return np.argsort(code, kind="stable")


class _Fusions:
    """Integer ids of the labels of one symmetry, shared by all its trees, and a table of the
    fusion of every pair of ids met so far: for ids (a, b), the outcomes c, each as often as
    its multiplicity, with their vertex labels, at ``out[start[a, b]:][:count[a, b]]``."""

    def __init__(self, sym):
        self.sym, self.labels, self.ids = sym, [], {}
        self.start = np.full((0, 0), -1, np.int64)
        self.count = np.zeros((0, 0), np.int64)
        self.out_c, self.out_mu = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.lock = threading.RLock()

    def id(self, a) -> int:
        i = self.ids.get(a)
        if i is None:
            with self.lock:
                i = self.ids.get(a)
                if i is None:
                    i = self.ids[a] = len(self.labels)
                    self.labels.append(a)
        return i

    def table(self, a, b):
        """(start, count) of the pairs a[:, None], b[None, :] (arrays of ids), and the
        outcome arrays they index."""
        with self.lock:
            if len(a) and len(b):
                n = max(int(a.max()), int(b.max())) + 1
                if n > len(self.start) or (self.start[np.ix_(a, b)] < 0).any():
                    self._fill(a, b)
            return self.start[np.ix_(a, b)], self.count[np.ix_(a, b)], self.out_c, self.out_mu

    def _fill(self, a, b):
        new = []
        for x in a.tolist():
            for y in b.tolist():
                if x >= len(self.start) or y >= len(self.start) or self.start[x, y] < 0:
                    out = [(self.id(c), m) for c, m in fusions(self.sym, self.labels[x],
                                                                self.labels[y])]
                    new.append((x, y, [c for c, m in out for _ in range(m)],
                                [mu for _, m in out for mu in range(m)]))
        n = len(self.labels)
        if n > len(self.start):
            size = max(n, 2 * len(self.start))
            start = np.full((size, size), -1, np.int64)
            count = np.zeros((size, size), np.int64)
            k = len(self.start)
            start[:k, :k], count[:k, :k] = self.start, self.count
            self.start, self.count = start, count
        base = len(self.out_c)
        for x, y, cs, _ in new:
            self.start[x, y], self.count[x, y] = base, len(cs)
            base += len(cs)
        self.out_c = np.concatenate([self.out_c] + [np.asarray(c, np.int64)
                                                    for _, _, c, _ in new])
        self.out_mu = np.concatenate([self.out_mu] + [np.asarray(m, np.int64)
                                                      for _, _, _, m in new])

    def nbytes(self) -> int:
        return sum(x.nbytes for x in (self.start, self.count, self.out_c, self.out_mu))


_LABEL_IDS: Dict[Any, _Fusions] = {}


def label_ids(sym) -> _Fusions:
    """The label ids and fusion table of `sym`, one per symmetry, kept for the process."""
    fz = _LABEL_IDS.get(sym)
    return fz if fz is not None else _LABEL_IDS.setdefault(sym, _Fusions(sym))


class Trees:
    """Every splitting tree of one side, for every leaf tuple, as arrays sorted by (coupled
    sector, leaf tuple, inner lines, vertex labels): `tup` the sector position on each leg, and
    label ids (into `labels`, shared by the symmetry's trees) of the coupled sector and inner
    lines. Independent of degeneracies. `tree(k)` builds the `FusionTree` of row k once.

    `coupled`, if given, keeps only the trees that end in one of these sectors."""

    def __init__(self, sym, sectors, flags, coupled=None):
        self.sym, self.flags = sym, flags
        self.leaves = tuple(tuple(sym.dual(a) if d else a for a in s)
                            for s, d in zip(sectors, flags))
        fz = label_ids(sym)
        leaf = [np.array([fz.id(a) for a in s], np.int64) for s in self.leaves]
        n = len(sectors)
        if n == 0:
            tup, line = np.zeros((1, 0), np.int64), np.array([fz.id(sym.unit())], np.int64)
            if coupled is not None and sym.unit() not in coupled:
                tup, line = tup[:0], line[:0]
        else:
            keep = np.array([n > 1 or coupled is None or a in coupled for a in self.leaves[0]],
                            bool).reshape(-1)
            tup, line = np.flatnonzero(keep)[:, None], leaf[0][keep]
        last = None if coupled is None else np.array([fz.id(c) for c in coupled], np.int64)
        lines = verts = np.zeros((len(line), 0), np.int64)
        for i in range(1, n):                       # grow the trees one leg at a time
            uniq, pos = np.unique(line, return_inverse=True)
            nk = len(leaf[i])
            start, count, all_c, all_mu = fz.table(uniq, leaf[i])
            start, count = start.reshape(-1), count.reshape(-1)
            at = np.repeat(start - (np.cumsum(count) - count), count) + np.arange(count.sum())
            out_c, out_mu = all_c[at], all_mu[at]
            if i == n - 1 and last is not None:
                ok = np.isin(out_c, last)
                count = np.bincount(np.repeat(np.arange(len(count)), count)[ok],
                                    minlength=len(count))
                out_c, out_mu = out_c[ok], out_mu[ok]
            first = (np.cumsum(count) - count).reshape(len(uniq), nk)
            count = count.reshape(len(uniq), nk)
            pair = (pos.reshape(-1)[:, None] * nk + np.arange(nk)).reshape(-1)
            m = count.reshape(-1)[pair]
            src = np.repeat(np.arange(len(pair)), m)
            entry = (np.repeat(first.reshape(-1)[pair] - (np.cumsum(m) - m), m)
                     + np.arange(int(m.sum())))
            s = src // nk
            tup = np.column_stack([tup[s], src % nk])
            lines = np.column_stack([lines[s], line[s]]) if i >= 2 else lines[s]
            verts = np.column_stack([verts[s], out_mu[entry]])
            line = out_c[entry]
        labels = fz.labels
        used = np.unique(np.concatenate([line, lines.reshape(-1)]))
        rank = np.zeros(len(labels), np.int64)
        rank[used[sorted(range(len(used)), key=lambda j: sym.sort_key(labels[used[j]]))]] = (
            np.arange(len(used)))
        keys = [rank[line], *tup.T, *rank[lines].T, *verts.T]
        perm = _sort_rows(keys)
        self.labels = labels
        self.tup, self.coupled = tup[perm], line[perm]
        self.lines, self.verts = lines[perm], verts[perm]
        cr = rank[self.coupled]
        new = np.r_[True, cr[1:] != cr[:-1]] if len(cr) else np.zeros(0, bool)
        starts = np.flatnonzero(new)
        self.group_start = starts
        self.group = np.cumsum(new) - 1
        self.sectors = tuple(labels[self.coupled[k]] for k in starts)
        stops = np.r_[starts[1:], len(cr)]
        self.bounds = {c: (int(a), int(b)) for c, a, b in zip(self.sectors, starts, stops)}
        self.built: Dict[int, FusionTree] = {}

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in (self.tup, self.coupled, self.lines, self.verts, self.group))

    def tree(self, k: int) -> FusionTree:
        t = self.built.get(k)
        if t is None:
            lab = self.labels
            t = self.built[k] = FusionTree(
                tuple(self.leaves[j][i] for j, i in enumerate(self.tup[k])), lab[self.coupled[k]],
                self.flags, tuple(lab[x] for x in self.lines[k]),
                tuple(int(v) for v in self.verts[k]))
        return t

    def key(self, k: int):
        """(leaf tuple, inner lines, coupled sector, vertex labels) of tree k, as labels."""
        lab = self.labels
        return (tuple(self.tup[k].tolist()), tuple(lab[x] for x in self.lines[k]),
                lab[self.coupled[k]], tuple(self.verts[k].tolist()))


def trees(sym, spaces, coupled=None) -> Trees:
    secs, flags = tuple(V.sectors for V in spaces), tuple(V.isdual for V in spaces)
    key = (sym, secs, flags, coupled)
    return STRUCTURE_CACHE.get(key, lambda: Trees(*key))


class Side:
    """The trees of one side with the degeneracies of its spaces: each tree's degeneracy count
    `deg`, its first row `start` in the block of its coupled sector, and each block's rows.
    `coupled` restricts the coupled sectors as in `Trees`."""

    def __init__(self, sym, spaces, coupled=None):
        self.spaces, self.trees = spaces, (tr := trees(sym, spaces, coupled))
        deg = np.ones(len(tr.coupled), np.int64)
        for k, V in enumerate(spaces):
            deg *= np.asarray(V.degeneracies, np.int64)[tr.tup[:, k]]
        before = np.cumsum(deg) - deg
        self.deg, self.start = deg, before - before[tr.group_start][tr.group]
        rows = np.add.reduceat(deg, tr.group_start).tolist() if len(deg) else []
        self.rows = dict(zip(tr.sectors, rows))

    @property
    def nbytes(self) -> int:
        return self.deg.nbytes + self.start.nbytes


# --------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class Run:
    """The rows (or columns) of one leaf tuple in the block of one coupled sector."""
    trees: Tuple[FusionTree, ...]
    start: int
    slices: Tuple[slice, ...]          # the tuple's ranges on the dense legs
    degs: Tuple[int, ...]
    dims: Tuple[int, ...]

    @property
    def stop(self) -> int:
        return self.start + len(self.trees) * math.prod(self.degs)


_UIDS = itertools.count()
_MAP_KEYS: Dict[int, Dict[tuple, Any]] = {}       # map_keys of each live layout, by uid


def _forget(uid, keys):
    """Drop the maps of a dead layout, and their keys from its partners' `map_keys`."""
    _MAP_KEYS.pop(uid, None)
    for k, other in list(keys.items()):
        INDEX_CACHE.pop(k)
        DEVICE_INDEX_CACHE.pop(k)
        if other is not None and other != uid:
            _MAP_KEYS.get(other, {}).pop(k, None)


class Layout:
    """The space of maps from `domain` to `codomain`: the coupled sectors present and the shape
    and offset of each block. `map_keys` names its maps in the index caches, each with the uid
    of its partner layout (or None); they are dropped when either layout dies."""

    def __init__(self, sym, codomain, domain):
        self.sym, self.codomain, self.domain = sym, codomain, domain
        self.key, self.uid = (sym, codomain, domain), next(_UIDS)
        self.row, self.col = _sides(sym, codomain, domain)
        common = set(self.row.rows) & set(self.col.rows)
        self.sectors = tuple(c for c in self.row.trees.sectors if c in common)
        self.shapes = {c: (self.row.rows[c], self.col.rows[c]) for c in self.sectors}
        self.offsets, n = {}, 0
        for c in self.sectors:
            self.offsets[c] = n
            n += math.prod(self.shapes[c])
        self.size = n
        self.map_keys: Dict[tuple, Any] = {}
        _MAP_KEYS[self.uid] = self.map_keys
        weakref.finalize(self, _forget, self.uid, self.map_keys).atexit = False

    def nbytes(self) -> int:
        return self.row.nbytes + self.col.nbytes

    @cached_property
    def rows(self):
        return _runs(self.sym, self.codomain, self.row, self.sectors)

    @cached_property
    def cols(self):
        return _runs(self.sym, self.domain, self.col, self.sectors)

    @cached_property
    def sector_arrays(self):
        """The offset and the column count of each coupled sector's block, in sector order."""
        return (np.array([self.offsets[c] for c in self.sectors], np.int64),
                np.array([self.shapes[c][1] for c in self.sectors], np.int64))

    def pair_trees(self):
        """Every tree pair in buffer order: row tree, column tree and the position of its
        coupled sector in `sectors`. Independent of the degeneracies."""
        tr, tc = self.row.trees, self.col.trees
        a, b = np.array([tr.bounds[c] for c in self.sectors], np.int64).reshape(-1, 2).T
        x, y = np.array([tc.bounds[c] for c in self.sectors], np.int64).reshape(-1, 2).T
        nr, nc = b - a, y - x
        sec = np.repeat(np.arange(len(self.sectors)), nr * nc)
        k = np.arange(len(sec)) - np.repeat(np.cumsum(nr * nc) - nr * nc, nr * nc)
        return a[sec] + k // nc[sec], x[sec] + k % nc[sec], sec

    def pair_starts(self, r, f, sec):
        """For tree pairs (row tree, column tree, sector position): the start of each pair's
        sub-block in the buffer and the row length of its block."""
        off, nc = self.sector_arrays
        return off[sec] + self.row.start[r] * nc[sec] + self.col.start[f], nc[sec]

    def keep(self) -> "Layout":
        """Mark this layout recently used (`RECENT_LAYOUTS`); returns it."""
        return RECENT_LAYOUTS.get(self.uid, lambda: self)

    def map(self, be, key, build, partner: "Layout" = None):
        """The map `key` of this layout on backend `be`, from `index_cache(be)` or `build()`;
        with a `partner` layout, the map also goes when the partner dies."""
        self.keep()
        k = (self.uid, backend_key(be), *key)
        if k not in self.map_keys:
            self.map_keys[k] = None if partner is None else partner.uid
            if partner is not None:
                partner.map_keys[k] = self.uid
        return index_cache(be).get(k, build)


def _sides(sym, codomain, domain):
    """The two sides of a layout. The side with fewer leaf tuples is built first, and the other
    only for its coupled sectors; equal sides share their trees."""
    cost = [math.prod(len(V.sectors) for V in side) for side in (codomain, domain)]
    k = 0 if cost[0] <= cost[1] else 1
    a, b = (codomain, domain)[k], (codomain, domain)[1 - k]
    first = Side(sym, a)
    same = [(V.sectors, V.isdual) for V in a] == [(V.sectors, V.isdual) for V in b]
    second = Side(sym, b) if same else Side(sym, b, coupled=frozenset(first.trees.sectors))
    return (first, second) if k == 0 else (second, first)


LAYOUTS: "weakref.WeakValueDictionary[tuple, Layout]" = weakref.WeakValueDictionary()


def layout(sym, codomain, domain) -> Layout:
    key = (sym, codomain, domain)
    lay = LAYOUTS.get(key)
    if lay is None:
        lay = LAYOUTS[key] = Layout(sym, codomain, domain)
    return lay


class Target:
    """The layout a map produces. Held by the map if the map created it; otherwise referred to
    weakly and fetched again by its key, or rebuilt, when needed."""
    __slots__ = ("key", "_held", "_ref")

    def __init__(self, sym, codomain, domain):
        self.key, self._held, self._ref = (sym, codomain, domain), None, None
        self.get()

    def get(self) -> Layout:
        lay = self._held
        if lay is None:
            lay = self._ref() if self._ref is not None else None
            if lay is None:
                lay = LAYOUTS.get(self.key)
                if lay is None:
                    lay = self._held = layout(*self.key)
                self._ref = weakref.ref(lay)
        return lay


def _runs(sym, spaces, side, sectors):
    """For each coupled sector: the (leaf tuple position, run) pairs of one side, in row order."""
    tr, out = side.trees, {}
    for c in sectors:
        a, b = tr.bounds[c]
        runs, k = [], a
        while k < b:
            j = k + 1
            while j < b and (tr.tup[j] == tr.tup[k]).all():
                j += 1
            idx = tuple(int(i) for i in tr.tup[k])
            degs = tuple(V.degeneracies[i] for V, i in zip(spaces, idx))
            dims = tuple(sym.dim(V.sectors[i]) for V, i in zip(spaces, idx))
            sl = tuple(slice(V.offsets[i], V.offsets[i] + n * d)
                       for V, i, n, d in zip(spaces, idx, degs, dims))
            runs.append((idx, Run(tuple(tr.tree(t) for t in range(k, j)), int(side.start[k]),
                                  sl, degs, dims)))
            k = j
        out[c] = tuple(runs)
    return out


def cache_info():
    """Sizes of the caches of this module and of the live layouts."""
    live = list(LAYOUTS.values())
    tables = list(_LABEL_IDS.values())
    info = {"structure": STRUCTURE_CACHE.info(), "indices": INDEX_CACHE.info(),
            "device_indices": DEVICE_INDEX_CACHE.info(),
            "recent_layouts": RECENT_LAYOUTS.info(),
            "layouts": {"entries": len(live), "bytes": sum(x.nbytes() for x in live)},
            "trees": {"entries": sum(len(t.built) for t in STRUCTURE_CACHE.values())},
            "fusions": lru_info(fusions),
            "label_tables": {"entries": len(tables), "bytes": sum(t.nbytes() for t in tables)}}
    return info


def lru_info(f):
    i = f.cache_info()
    return {"entries": i.currsize, "max_entries": i.maxsize, "hits": i.hits, "misses": i.misses}
