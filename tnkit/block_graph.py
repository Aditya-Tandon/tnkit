"""CUDA-graph replay of the block-sparse matvecs of one Krylov solve.

A Krylov solve applies one effective Hamiltonian to a sequence of vectors which, after the first
one or two, share one block layout. Each application issues the same kernels: per contraction a
gather per operand and one small GEMM per contracted charge, a few hundred in all. Issued one by
one from Python, these launches, not the arithmetic, set the time of a block TDVP step on a GPU.

`wrap(fn, x)` returns `fn` itself unless graphs are on (`set_enabled`, or ``TNKIT_BLOCK_GRAPH=1``)
and `x` lives on a CUDA device. Otherwise it returns a `MatvecGraph`. For each input layout, the
first call runs eagerly, which builds and uploads every plan the layout needs. The second call is
captured into a CUDA graph. That call and every later one copy the input into the graph's buffer,
replay the graph and copy the output out. A replay runs the kernels the eager call runs, on the
same operands, so the results are bitwise the same. A capture that fails, for instance because a
plan was evicted and would have to be uploaded again, leaves that layout eager for the rest of
the solve.

`fn` must be one fixed linear map for the life of the wrapper: a graph reads its operands where
they were at capture time. Each graph keeps the gather indices and buffers it reads alive
(`blocks.Kept`), so the plan cache may evict them meanwhile; the indices stay counted toward the
cache's limit until the wrapper is closed. `solve` closes it when the solve ends, `clear` and
`set_enabled(False)` close every live one. `blocks.clear_cache` releases what the graphs keep; a
layout whose graph it released starts again with an eager call. All graphs on a device share one
memory pool, which the cache limit does not bound; this is safe because a replay writes every pool
buffer it reads before reading it, and its output is copied out at once.
"""
from __future__ import annotations

import contextlib
import os
import time
import weakref
from collections import Counter
from typing import Any, Callable, Dict

from . import blocks
from .blocks import BlockTensor

__all__ = ["set_enabled", "enabled", "clear", "info", "clear_stats"]


def _from_env() -> bool:
    v = os.environ.get("TNKIT_BLOCK_GRAPH", "")
    if v not in ("", "0", "1"):
        raise ValueError(f"TNKIT_BLOCK_GRAPH must be 0 or 1, got {v!r}")
    return v == "1"


_ON = _from_env()
_STATS: Counter = Counter()
_DEVICES: Dict[Any, tuple] = {}            # device -> (capture stream, pool, anchor graph)
_LIVE: "weakref.WeakSet[MatvecGraph]" = weakref.WeakSet()
_WARM, _EAGER = object(), object()


def set_enabled(on: bool) -> None:
    """Turn graph replay of block matvecs on or off (default: ``TNKIT_BLOCK_GRAPH=1`` or off).
    Turning it off closes every live wrapper."""
    global _ON
    _ON = bool(on)
    if not _ON:
        clear()


def clear() -> None:
    """Close every live `MatvecGraph`."""
    for m in list(_LIVE):
        m.close()


def enabled() -> bool:
    return _ON


def info() -> Dict[str, float]:
    """Counters since the last `clear_stats`: wrapped solves, matvec calls run ``eager``,
    ``captures`` (with their seconds, ``capture_s``), ``replays``, ``failed`` captures, and the
    contractions and GEMMs recorded in captures (``captured_*``, also counted by
    `blocks.cache_info`, which sees a capture as an eager call) and executed by replays
    (``replayed_*``, which it does not see)."""
    keys = ("solves", "eager", "captures", "replays", "failed", "capture_s", "captured_contractions",
            "captured_gemms", "replayed_contractions", "replayed_gemms")
    return {k: _STATS[k] for k in keys}


def clear_stats() -> None:
    _STATS.clear()


def wrap(fn: Callable[[BlockTensor], BlockTensor], x: BlockTensor):
    """`fn`, or a `MatvecGraph` around it when graphs are on and `x` is on a CUDA device."""
    if not _ON or not getattr(x.data, "is_cuda", False):
        return fn
    return MatvecGraph(fn)


@contextlib.contextmanager
def solve(fn: Callable[[BlockTensor], BlockTensor], x: BlockTensor):
    """`wrap(fn, x)` for one Krylov solve, closed when the block exits."""
    f = wrap(fn, x)
    try:
        yield f
    finally:
        if isinstance(f, MatvecGraph):
            f.close()


class _Graph:
    """A captured matvec: the graph, its input buffer, the output it writes, and what it reads
    (`keep`, a `blocks.Kept`, set by `MatvecGraph`)."""
    __slots__ = ("graph", "x", "out", "keep", "contractions", "gemms")

    def __init__(self, graph, x, out: BlockTensor):
        self.graph, self.x, self.out = graph, x, out

    def run(self, x: BlockTensor) -> BlockTensor:
        self.x.copy_(x.data)
        self.graph.replay()
        _STATS["replays"] += 1
        _STATS["replayed_contractions"] += self.contractions
        _STATS["replayed_gemms"] += self.gemms
        return BlockTensor(self.out.struct, self.out.data.clone())


class MatvecGraph:
    """`fn` with its calls replayed from CUDA graphs, one per input layout and dtype (see the
    module docstring). `capture(fn, x)` returns a `_Graph`; the default captures on CUDA. Close it
    when the solve ends: until then, or `blocks.clear_cache`, what its graphs read stays alive and
    counted, even if the wrapper is dropped. After `close` it runs `fn` eagerly."""

    def __init__(self, fn: Callable[[BlockTensor], BlockTensor], capture=None):
        self.fn, self._capture = fn, capture or _cuda_capture
        self._graphs: Dict[Any, Any] = {}
        self.closed = False
        _LIVE.add(self)
        _STATS["solves"] += 1

    def __call__(self, x: BlockTensor) -> BlockTensor:
        key = (x.struct, x.data.dtype)
        g = self._graphs.get(key)
        if isinstance(g, _Graph) and g.keep.released:    # its indices may be gone
            g = None
        if g is None:
            if not self.closed:
                self._graphs[key] = _WARM
        elif g is _WARM:
            self._graphs[key] = _EAGER                  # stays so if the capture is interrupted
            g = self._graphs[key] = self._make(x)
        if isinstance(g, _Graph):
            return g.run(x)
        _STATS["eager"] += 1
        return self.fn(x)

    def close(self) -> None:
        """Drop every graph and release what it keeps."""
        self.closed = True
        _LIVE.discard(self)
        for g in self._graphs.values():
            if isinstance(g, _Graph):
                g.keep.release()
        self._graphs.clear()

    def _make(self, x: BlockTensor):
        stats = blocks._INDEX_CACHE.stats
        c0, g0, t0 = stats["contractions"], stats["gemms"], time.perf_counter()
        try:
            with blocks.keep_operands() as keep:
                g = self._capture(self.fn, x)
        except Exception:  # noqa: BLE001 -- the eager call is always available
            _STATS["failed"] += 1
            return _EAGER
        if self.closed:                                 # by `fn`, during its own capture
            keep.release()
            return _EAGER
        g.keep = keep
        g.contractions, g.gemms = stats["contractions"] - c0, stats["gemms"] - g0
        _STATS["captures"] += 1
        _STATS["capture_s"] += time.perf_counter() - t0
        _STATS["captured_contractions"] += g.contractions
        _STATS["captured_gemms"] += g.gemms
        return g


def _device_state(like):
    """The capture stream and memory pool of `like`'s device, made on first use. The stream's
    cuBLAS handle and workspace are created outside any capture, and an anchor graph keeps the
    pool alive between solves."""
    import torch
    st = _DEVICES.get(like.device)
    if st is None:
        stream = torch.cuda.Stream(like.device)
        with torch.cuda.stream(stream):
            a = torch.ones((2, 2), dtype=like.dtype, device=like.device)
            torch.matmul(a, a)
        stream.synchronize()
        pool, anchor = torch.cuda.graph_pool_handle(), torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            anchor.capture_begin(pool=pool)
            keep = torch.zeros(1, device=like.device)
            anchor.capture_end()
        st = _DEVICES[like.device] = (stream, pool, (anchor, keep))
    return st[0], st[1]


def _cuda_capture(fn, x: BlockTensor) -> _Graph:
    import torch
    from .cudagraph import _release_failed_capture
    stream, pool = _device_state(x.data)
    xs, g = x.data.clone(), torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            g.capture_begin(pool=pool)
            try:
                out = fn(BlockTensor(x.struct, xs))
            finally:
                g.capture_end()
    except BaseException:
        with torch.cuda.device(x.data.device):
            _release_failed_capture(torch, pool)
        raise
    return _Graph(g, xs, out)
