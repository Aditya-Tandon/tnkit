"""A graph capture or replay that raises falls back to the eager step instead of crashing.

The CPU tests inject the failure by overriding the cache's `_make` (capture) or the graph's
`run` (replay), so the fallback path runs on the numpy backend and is compared bit for bit
against the all-eager evolution. The `cuda` tests make a real capture fail on a device.
"""
from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.cudagraph import GraphContextLost, SiteStepGraphCache
from tnkit.models import tfim_mpo
from tnkit.prefix_graph_impl import PrefixSiteStepGraphCache
from tnkit.tdvp import _replay, tdvp1_evolve, tdvp2_evolve

DT = 0.05
FIXED_M = 8          # frozen Krylov depth for the `cuda_graph` path
CUSOLVER_MSG = "cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnXgeqrf`"


def _problem(n=10, chi=6, seed=2026):
    rng = np.random.default_rng(seed)
    dims = [1] + [min(chi, 2 ** min(i + 1, n - i - 1)) for i in range(n - 1)] + [1]
    mps = [(rng.normal(size=(dims[i], 2, dims[i + 1]))
            + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))) / 10.0 for i in range(n)]
    return mps, tfim_mpo(n, g=1.0, h=0.10)


def _same_bits(a, b):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes(), f"site {i} differs"


class _Stub:
    """A recorded graph whose replay raises, or returns None (prefix too short)."""

    pool = None

    def __init__(self, outcome):
        self.outcome = outcome

    def run(self, *a, **kw):
        if self.outcome == "raise":
            raise RuntimeError(CUSOLVER_MSG + " on replay")
        return None


def _failing(base, mode):
    """A cache class whose captures raise (mode='capture') or whose replays raise."""

    class Cache(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.attempts = []

        def _make(self, sig, direction):
            self.attempts.append((sig, direction))
            if mode == "capture":
                raise RuntimeError(CUSOLVER_MSG)
            return _Stub(mode)

    return Cache


def _prefix_cache(mode, **kw):
    return _failing(PrefixSiteStepGraphCache, mode)(
        get_backend("numpy"), types.SimpleNamespace(), DT, 12, **kw)


def _cg1_cache(mode, **kw):
    return _failing(SiteStepGraphCache, mode)(
        get_backend("numpy"), types.SimpleNamespace(), DT, FIXED_M, **kw)


def _evolve(mps, mpo, n_steps, **kw):
    out, _ = tdvp1_evolve([m.copy() for m in mps], mpo, DT, n_steps, backend="numpy", **kw)
    return out


# ------------------------------------------------------------------ prefix_graph
def test_prefix_capture_failure_falls_back_bit_identically(capsys):
    mps, mpo = _problem()
    ref = _evolve(mps, mpo, 4, graph="off")
    gc = _prefix_cache("capture", fault_budget=0)
    _same_bits(_evolve(mps, mpo, 4, prefix_graph=gc), ref)
    st = gc.summary()
    assert st["capture_failed"] == len(gc.refused) > 0
    assert st["captures"] == st["replayed"] == 0 and st["eager"] > 0
    assert "CUSOLVER" in st["first_fault"]
    assert len(gc.attempts) == len(set(gc.attempts))          # each signature tried once
    out = capsys.readouterr().out
    assert "TNKIT_GRAPH_FALLBACK phase=capture" in out
    assert out.count("TNKIT_GRAPH_FALLBACK") <= gc._FAULT_LINES


def test_prefix_replay_failure_evicts_and_falls_back_bit_identically(capsys):
    mps, mpo = _problem()
    ref = _evolve(mps, mpo, 4, graph="off")
    gc = _prefix_cache("raise", fault_budget=0)
    _same_bits(_evolve(mps, mpo, 4, prefix_graph=gc), ref)
    st = gc.summary()
    assert st["replay_failed"] > 0 and st["short"] == 0
    assert gc.graphs == {}
    assert "TNKIT_GRAPH_FALLBACK phase=replay" in capsys.readouterr().out


def test_short_prefix_is_not_a_replay_failure():
    mps, mpo = _problem()
    gc = _prefix_cache("short")
    _evolve(mps, mpo, 2, prefix_graph=gc)
    st = gc.summary()
    assert st["short"] > 0 and st["replay_failed"] == 0 and gc.graphs


def test_breaker_stops_new_captures(capsys):
    mps, mpo = _problem()
    gc = _prefix_cache("capture", fault_budget=2)
    _evolve(mps, mpo, 4, prefix_graph=gc)
    assert len(gc.attempts) == gc.summary()["capture_failed"] == 2
    assert gc.summary()["breaker_tripped"] == 2
    assert "TNKIT_GRAPH_BREAKER" in capsys.readouterr().out


def test_breaker_off_tries_every_signature():
    mps, mpo = _problem()
    gc = _prefix_cache("capture", fault_budget=0)
    _evolve(mps, mpo, 4, prefix_graph=gc)
    assert gc.summary()["breaker_tripped"] == 0 and gc.summary()["capture_failed"] > 2


def test_one_site_two_site_one_site_with_failing_captures():
    """A two-site interval changes bond dimensions, so the second one-site leg asks the
    cache for signatures it has not seen; every one of them fails to capture."""
    mps, mpo = _problem(n=12, chi=8)

    def leg(gc):
        kw = {"prefix_graph": gc} if gc is not None else {"graph": "off"}
        st, _ = tdvp1_evolve([m.copy() for m in mps], mpo, DT, 3, backend="numpy", **kw)
        st, _ = tdvp2_evolve(list(st), mpo, 16, DT, 1, backend="numpy")
        st, _ = tdvp1_evolve(list(st), mpo, DT, 3, backend="numpy", **kw)
        return st

    gc = _prefix_cache("capture", fault_budget=0)
    _same_bits(leg(gc), leg(None))
    assert len({s for s, _ in gc.attempts}) > 1


def test_lost_device_is_raised():
    gc = _prefix_cache("capture")
    gc._device_ok = lambda: False
    with pytest.raises(GraphContextLost, match="could not do arithmetic"):
        gc.get((4, 2, 4, 4, 3, 3), "right")


def test_replay_failure_without_handler_propagates():
    with pytest.raises(RuntimeError, match="INTERNAL_ERROR"):
        _replay(object(), _Stub("raise"), (1, 2, 1, 1, 3, 3), "right")


def test_stream_is_restored_after_failed_capture():
    """`capture_end` raising skips torch's own stream restore; the cache puts it back."""
    current = {"s": "caller"}
    cuda = types.SimpleNamespace(
        is_available=lambda: True, synchronize=lambda: None,
        current_stream=lambda: current["s"],
        set_stream=lambda s: current.__setitem__("s", s))

    class Cache(PrefixSiteStepGraphCache):
        def _make(self, sig, direction):
            current["s"] = "capture"
            raise RuntimeError("operation failed due to a previous error during capture")

    gc = Cache(get_backend("numpy"), types.SimpleNamespace(cuda=cuda), DT, 12)
    assert gc.get((4, 2, 4, 4, 3, 3), "right") is None
    assert current["s"] == "caller"


def test_last_graph_evicted_drops_the_pool():
    """A pool whose last graph is gone cannot be shared with a new capture."""
    gc = _prefix_cache("short")
    gc.graphs[("s", "right")] = gc.graphs[("t", "left")] = _Stub("short")
    gc.pool = "pool"
    gc.replay_failed("s", "right", RuntimeError("x"))
    assert gc.pool == "pool"
    gc.replay_failed("t", "left", RuntimeError("x"))
    assert gc.pool is None


class _FakeTorch:
    """Just enough of torch.cuda to run `cudagraph._capture`; `end_raises` makes the allocator
    entry point refuse (the capture had ended cleanly), `capture_exit` is what leaving the
    capture context raises. `CUDAGraph` records the raw capture that resets the generator."""

    def __init__(self, capture_exit=None, end_raises=False, with_hooks=True):
        self.calls = []
        calls = self.calls

        class Stream:
            def wait_stream(self, other):
                pass

        class Graph:
            def __init__(self, g, pool=None):
                calls.append(("graph", pool))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                if capture_exit is not None:
                    raise capture_exit

        class CUDAGraph:
            def capture_begin(self, pool=None):
                calls.append(("capture_begin", pool))

            def capture_end(self):
                calls.append(("capture_end",))

        def end(dev, pool):
            calls.append(("end", dev, pool))
            if end_raises:
                raise RuntimeError("endAllocatePool: not currently recording to mempool_id")

        cuda = types.SimpleNamespace(
            Stream=Stream, current_stream=Stream, stream=lambda s: contextlib.nullcontext(),
            synchronize=lambda: None, CUDAGraph=CUDAGraph, graph=Graph,
            graph_pool_handle=lambda: "fresh", current_device=lambda: 0,
            empty_cache=lambda: calls.append(("empty_cache",)))
        c = types.SimpleNamespace()
        if with_hooks:
            c._cuda_endAllocateCurrentStreamToPool = end
            c._cuda_releasePool = lambda dev, pool: calls.append(("release", dev, pool))
        self.cuda, self._C = cuda, c
        t = types.SimpleNamespace(zero_=lambda: calls.append(("zero_",)))
        self.empty = lambda *shape, device=None: t


RNG_RESET = [("capture_begin", None), ("zero_",), ("capture_end",)]


def _run_capture(fake, pool=None, fn=lambda: "out"):
    from tnkit.cudagraph import _capture
    return _capture(fake, fn, warmup=1, pool=pool)


def test_failed_capture_is_released_from_the_allocator():
    """`capture_end` raising leaves the capture underway in torch's allocator, which then fails
    every later capture; `_capture` ends it and releases the pool use, on the pool's own id."""
    fake = _FakeTorch(capture_exit=RuntimeError("operation failed due to a previous error"))
    with pytest.raises(RuntimeError, match="previous error"):
        _run_capture(fake, pool="shared")
    assert fake.calls == [("graph", "shared"), ("end", 0, "shared"), ("release", 0, "shared"),
                          ("empty_cache",)] + RNG_RESET


def test_first_capture_gets_a_pool_handle_so_a_failure_can_be_ended():
    fake = _FakeTorch(capture_exit=RuntimeError("invalidated"))
    with pytest.raises(RuntimeError):
        _run_capture(fake)
    assert ("graph", "fresh") in fake.calls and ("end", 0, "fresh") in fake.calls


def test_capture_that_ended_cleanly_is_left_alone():
    """The body raised but `capture_end` succeeded: nothing is underway, nothing is released.
    The generator is reset all the same: instantiation can fail after the allocator let go."""
    fake, calls = _FakeTorch(end_raises=True), []

    def body():
        calls.append(1)
        if len(calls) > 1:                       # the warm-up call succeeds, the capture fails
            raise ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        _run_capture(fake, pool="p", fn=body)
    assert ("end", 0, "p") in fake.calls and fake.calls[-3:] == RNG_RESET
    assert not any(c[0] in ("release", "empty_cache") for c in fake.calls)


def test_a_successful_capture_touches_nothing():
    fake = _FakeTorch()
    g, out = _run_capture(fake, pool="p")
    assert out == "out" and [c[0] for c in fake.calls] == ["graph"]


def test_old_torch_without_the_allocator_hooks_behaves_as_before():
    fake = _FakeTorch(capture_exit=RuntimeError("invalidated"), with_hooks=False)
    with pytest.raises(RuntimeError, match="invalidated"):
        _run_capture(fake)
    assert fake.calls == [("graph", "fresh")] + RNG_RESET


def test_both_caches_use_one_capture():
    import tnkit.cudagraph as cg1
    import tnkit.prefix_graph_impl as cg3
    assert cg3._capture is cg1._capture


# ------------------------------------------------------------------ cuda_graph
@pytest.mark.parametrize("mode", ["capture", "raise"])
def test_cg1_failure_falls_back_bit_identically(mode):
    mps, mpo = _problem()
    ref = _evolve(mps, mpo, 4, fast_expm=FIXED_M, graph="off")
    gc = _cg1_cache(mode, fault_budget=0)
    _same_bits(_evolve(mps, mpo, 4, fast_expm=FIXED_M, cuda_graph=gc), ref)
    st = gc.stats
    assert st[f"{'capture' if mode == 'capture' else 'replay'}_failed"] > 0
    assert st["replayed"] == 0 and st["eager"] > 0 and gc.graphs == {}


# ------------------------------------------------------------------ on a device
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


def _host_sync_in_capture(mod, monkeypatch, bad):
    """Route the cache's captures through the real `_capture`, with a host sync inside the
    recorded body while `bad[0]` is set: not permitted while capturing, so the capture is
    invalidated on the device (the warm-up runs the same body outside a capture, where the
    sync is harmless)."""
    real = mod._capture

    def capture(torch_, fn, warmup=3, pool=None):
        def body():
            out = fn()
            if bad[0]:
                torch_.zeros(1, device="cuda").item()
            return out
        return real(torch_, body, warmup, pool)

    monkeypatch.setattr(mod, "_capture", capture)


def _device_run(which, be, mps, mpo, steps, ref=False):
    kw = {"fast_expm": FIXED_M} if which == "cg1" else {}
    if ref:
        return tdvp1_evolve([m.copy() for m in mps], mpo, DT, steps, backend="torch-cuda",
                            graph="off", **kw)[0]
    gc = (PrefixSiteStepGraphCache(be, torch, DT, 12) if which == "prefix"
          else SiteStepGraphCache(be, torch, DT, FIXED_M))
    key = "prefix_graph" if which == "prefix" else "cuda_graph"
    got = tdvp1_evolve([m.copy() for m in mps], mpo, DT, steps, backend="torch-cuda",
                       **{key: gc}, **kw)[0]
    return gc, got


@pytest.mark.cuda
@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device on this runner")
@pytest.mark.parametrize("which", ["prefix", "cg1"])
def test_cuda_failed_capture_falls_back_and_later_captures_work(monkeypatch, which):
    """A real capture is invalidated by a host sync inside it. The run finishes on the eager
    path with the eager result and the default stream current; and the failed capture must not
    leave the allocator thinking one is underway, so a later capture in the same process
    succeeds and replays (it used to fail on `captures_underway.empty()`, silently turning
    every later graph off)."""
    import tnkit.cudagraph as cg1
    import tnkit.prefix_graph_impl as cg3
    mod = cg3 if which == "prefix" else cg1
    bad = [True]
    _host_sync_in_capture(mod, monkeypatch, bad)
    be = get_backend("torch-cuda")
    mps, mpo = _problem()
    ref = _device_run(which, be, mps, mpo, 3, ref=True)
    gc, got = _device_run(which, be, mps, mpo, 3)
    assert gc.stats["capture_failed"] > 0 and gc.stats["captures"] == 0
    assert torch.cuda.current_stream() == torch.cuda.default_stream()
    for x, y in zip(got, ref):
        np.testing.assert_array_equal(be.to_numpy(x), be.to_numpy(y))

    bad[0] = False                                      # a fresh cache, in the same process
    gc, got = _device_run(which, be, mps, mpo, 3)
    assert gc.stats["capture_failed"] == 0 and gc.stats["captures"] > 0
    assert gc.stats["replayed"] > 0
    torch.cuda.empty_cache()                            # not underway: this asserts otherwise
