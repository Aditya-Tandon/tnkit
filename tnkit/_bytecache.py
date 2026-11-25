"""A least-recently-used map bounded by the bytes of its values."""
from __future__ import annotations

import os
import threading
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Hashable, Optional


def nbytes(x) -> int:
    """Bytes held by an array (numpy or torch), a tuple of them, or an object with an `nbytes()`
    method."""
    if isinstance(x, tuple):
        return sum(nbytes(v) for v in x)
    n = getattr(x, "nbytes", None)
    if callable(n):
        return int(n())
    if n is not None:
        return int(n)
    if hasattr(x, "element_size"):
        return int(x.element_size() * x.numel())
    return 0


class ByteCache:
    """`get(key, build)` returns the cached value or builds, stores and returns it. The least
    recently used entries go once the total exceeds `max_bytes` (or the count `max_entries`);
    one entry larger than the cap is still kept until the next insertion. A value's size is
    taken when it is stored, and again when a stored `Growing` value calls `grew()`. Safe to
    share between threads: two threads that miss the same key both build, and the first stored
    value wins. The lock is re-entrant, so a value's finalizer may remove entries; dropped values
    are released outside it. `source` says where the limits came from: ``default``, ``env`` or
    ``set`` (by `set_limit`)."""

    def __init__(self, max_bytes: int, max_entries: Optional[int] = None,
                 source: str = "default"):
        self.max_bytes = int(max_bytes)
        self.max_entries = max_entries
        self.source = source
        self._d: "OrderedDict[Hashable, tuple]" = OrderedDict()
        self._lock = threading.RLock()
        self.bytes = 0
        self.stats: Counter = Counter()

    def _hit(self, key):
        hit = self._d.get(key)
        if hit is not None:
            self._d.move_to_end(key)
            self.stats["hits"] += 1
            return hit[0]
        return None

    def get(self, key: Hashable, build: Callable[[], Any]):
        with self._lock:
            v = self._hit(key)
        if v is not None:
            return v
        v = build()
        n = nbytes(v)
        with self._lock:
            self.stats["misses"] += 1
            old = self._d.get(key)
            if old is not None:
                return old[0]
            self._d[key] = (v, n)
            self.bytes += n
            if isinstance(v, Growing):
                v._held = (self, key)
            dropped = self._evict()
        del dropped                                # freed outside the lock
        return v

    def _resize(self, key, v):
        with self._lock:
            hit = self._d.get(key)
            if hit is None or hit[0] is not v:
                return
            n = nbytes(v)
            self._d[key] = (v, n)
            self.bytes += n - hit[1]
            dropped = self._evict()
        del dropped

    def pop(self, key: Hashable):
        """Remove `key`; returns its value or None."""
        with self._lock:
            hit = self._d.pop(key, None)
            if hit is not None:
                self.bytes -= hit[1]
        return None if hit is None else hit[0]

    def _evict(self) -> list:
        dropped = []
        while len(self._d) > 1 and (self.bytes > self.max_bytes or (
                self.max_entries is not None and len(self._d) > self.max_entries)):
            _, (v, n) = self._d.popitem(last=False)
            self.bytes -= n
            self.stats["evictions"] += 1
            dropped.append(v)
        return dropped

    def set_limit(self, max_bytes: int, max_entries: Optional[int] = None,
                  source: str = "set"):
        """New limits; `max_entries=None` keeps the current count limit."""
        with self._lock:
            self.max_bytes = int(max_bytes)
            if max_entries is not None:
                self.max_entries = int(max_entries)
            self.source = source
            dropped = self._evict()
        del dropped

    def clear(self):
        with self._lock:
            old, self._d = self._d, OrderedDict()
            self.bytes = 0
            self.stats.clear()
        del old

    def values(self) -> list:
        with self._lock:
            return [v for v, _ in self._d.values()]

    def keys(self) -> list:
        with self._lock:
            return list(self._d)

    def __len__(self):
        return len(self._d)

    def info(self) -> Dict[str, Any]:
        return {"entries": len(self._d), "bytes": self.bytes, "max_bytes": self.max_bytes,
                **({} if self.max_entries is None else {"max_entries": self.max_entries}),
                "limit_source": self.source,
                **{k: self.stats[k] for k in ("hits", "misses", "evictions")}}


class Growing:
    """A cached value whose size can grow after it is stored: it calls `grew()`, and the cache
    that holds it counts its new size and evicts as needed."""
    _held = None

    def grew(self):
        held = self._held
        if held is not None:
            held[0]._resize(held[1], self)


def env_cache(name: str, default_mb: float, max_entries: Optional[int] = None) -> ByteCache:
    """A `ByteCache` of `default_mb` MiB, or of the MiB in the environment variable `name`."""
    v = os.environ.get(name)
    if v is None:
        return ByteCache(int(default_mb * 2 ** 20), max_entries)
    return ByteCache(int(float(v) * 2 ** 20), max_entries, "env")
