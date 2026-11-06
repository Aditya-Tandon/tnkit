"""CUDA-Graph capture of the one-site TDVP step.

WHY THIS EXISTS. Measurement showed the 1TDVP site-step of a two-packet
collision run at 10.21 ms scaling as chi^0.14 against a FLOP-bound chi^3, and
a follow-up measurement closed the mechanism with an operation counter: the
site-step dispatches ~602 torch operations at a FLAT ~22 us each, against arithmetic of
tens of kFLOP per operation. The card is not computing, it is waiting to be told what to
do 602 times. A CUDA graph replays a recorded kernel sequence with a single submission,
so it attacks the only quantity that was ever binding -- the count.

WHY IT IS NOT SIMPLY `torch.cuda.graph(...)` AROUND THE EXISTING STEP. Three measured
obstacles:

  * `torch.matrix_exp` is NOT capturable. It reads ||A||_1 back to the host to pick
    a Pade degree, and a host read inside a capture is a hard error. So is
    `torch.linalg.eigh`, which would otherwise be the exact route for the
    real-symmetric Krylov tridiagonal. `backend.expm_taylor` -- solve-free
    scaling-and-squaring, GEMMs only -- is capturable and agrees with
    `scipy.linalg.expm` to 4.0e-14 worst case.
  * `torch.linalg.qr` IS capturable. The 1TDVP centre move was the risk everyone
    expected and it is not the problem.
  * Krylov ADAPTIVITY is structurally incompatible with capture: the stop test is a host
    branch on a device scalar. The schedule must be frozen. `lanczos_expm_fixed` already
    is exactly that, and this module reuses it rather than re-deriving it.

WHY A FIXED KRYLOV DEPTH IS CHEAP HERE AND EXPENSIVE EAGERLY. This is the whole reason
the direction is worth re-opening after an earlier attempt at a fixed-depth fast path
was rejected for measuring 0.92x (a slowdown) when run eagerly. On an
evolved leg the adaptive solver needs mean m 6.63-6.90 and max 8, so a frozen schedule
over-serves by 1.45-1.51x in MATVECS. Eagerly, a matvec costs dispatch, so over-service
costs wall time and the fast path loses. Inside a graph, dispatch is paid once for the
whole step, so the marginal cost of an over-served Krylov iteration collapses to its
actual (tiny, chi^0.14) execution time. The graph does not merely speed up the fixed
schedule -- it removes the reason the fixed schedule was a bad trade.

SHAPE CLASSES. A graph is a recording of kernels at FIXED shapes and FIXED addresses, so
one graph serves one (a, s, b, c) signature. 1TDVP is exactly bond-preserving, so in the
saturated-chi bulk -- where a leg spends ~95% of its time -- every interior site shares
one signature and one graph serves them all. The chain edges ramp (a = 1, 2, 4, ...) and
get their own graphs, or fall back to eager; `SiteStepGraphCache` counts both so the
report can state how much of the sweep was actually replayed.
"""
from __future__ import annotations

from ._dmrg import apply_one_site, apply_zero_site, update_left_env, update_right_env
from .eigensolver import lanczos_expm_fixed


def _release_failed_capture(torch, pool):
    """Take a capture that raised out of the allocator's books; True if one was there.

    `CUDAGraph::capture_end` calls `cudaStreamEndCapture` before the caching allocator's
    `endAllocateToPool`, so a capture invalidated on the device (a cuSOLVER error, a host
    sync) raises from the first and never reaches the second. The allocator then keeps the
    capture "underway", and every later `torch.cuda.graph` in the process fails on
    `captures_underway.empty()` (PyTorch 2.5.1, aten/src/ATen/cuda/CUDAGraph.cpp). Ending it
    by hand needs the pool's id, which is why `_capture` always passes one. A capture that had
    ended before the exception is no longer underway: the call raises and nothing is done.
    The pool use that `capture_begin` took is then released too, since the failed graph never
    owns it, and the pool's blocks are freed. Torch versions without these entry points keep
    their old behaviour.

    The device's default CUDA generator is then returned to eager use (`_end_rng_capture`).
    """
    released = False
    end = getattr(torch._C, "_cuda_endAllocateCurrentStreamToPool", None)
    release = getattr(torch._C, "_cuda_releasePool", None)
    if end is not None and release is not None:
        device = torch.cuda.current_device()
        try:
            end(device, pool)
        except RuntimeError:                  # not underway: the capture ended cleanly
            pass
        else:
            release(device, pool)
            torch.cuda.empty_cache()
            released = True
    _end_rng_capture(torch)
    return released


def _end_rng_capture(torch):
    """Mark the current device's default CUDA generator as no longer capturing.

    `CUDAGraph::capture_begin` marks it as capturing, and only a `capture_end` that gets through
    instantiation unmarks it. After a capture that raised before that point, every replay of any
    graph on the device, and every random op on it, raises "Offset increment outside graph
    capture encountered unexpectedly" (PyTorch 2.5.1, CUDAGeneratorState::increase) until some
    later capture succeeds. A complete capture of one fill, never replayed, unmarks it; it leaves
    the generator's seed and offset as they were.
    """
    t = torch.empty(1, device=torch.cuda.current_device())
    s, g = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g.capture_begin()
        t.zero_()
        g.capture_end()


def _capture(torch, fn, warmup=3, pool=None):
    """Warm up on a SIDE stream, then record.

    The side-stream warmup is mandatory, not defensive: cuBLAS/cuSOLVER allocate their
    workspaces and create their handles lazily on first call, and doing that inside the
    capture either records a one-time allocation as if it were part of the step or trips
    the capture outright. Warming up on a non-default stream also keeps those allocations
    out of the default stream's history, which is what torch's own graph examples do.

    A capture that fails is released from the allocator (`_release_failed_capture`) before
    the exception propagates.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    if pool is None:
        pool = torch.cuda.graph_pool_handle()
    try:
        with torch.cuda.graph(g, pool=pool):
            out = fn()
    except Exception:
        _release_failed_capture(torch, pool)
        raise
    return g, out


class GraphContextLost(RuntimeError):
    """A capture or replay failed and the device then failed a trivial computation too,
    so the eager fallback is not available either."""


class GraphFaults:
    """Capture and replay failures for a graph cache: fall back to eager, never crash.

    A captured step is an optimisation of an eager step that is always available, so a
    capture or replay that raises (e.g. a cuSOLVER error during capture) strikes that
    (signature, direction) off for good and the caller runs the step eagerly. Each fault is
    counted in `stats` and printed (the first `_FAULT_LINES` of them). After
    `fault_budget` faults the cache stops capturing new signatures; recorded graphs keep
    replaying. 0 disables that. If the device cannot do arithmetic after a fault,
    `GraphContextLost` is raised.
    """

    _FAULT_LINES = 8

    def _init_faults(self, fault_budget):
        self.fault_budget = fault_budget
        self.refused = {}
        self.stats.update({"capture_failed": 0, "replay_failed": 0, "first_fault": "",
                           "breaker_tripped": 0})

    def _make(self, sig, direction):
        raise NotImplementedError

    def _capture_or_none(self, key):
        """`_make(*key)`, or None after recording the failure."""
        stream = self._current_stream()
        try:
            return self._make(*key)
        except Exception as exc:  # noqa: BLE001 -- any capture failure falls back to eager
            self._restore_stream(stream)
            self._fault("capture", key, exc)
            return None

    def replay_failed(self, sig, direction, exc):
        """Drop a graph whose replay raised. `run` writes only into the graph's own
        buffers, so the caller's tensors are untouched and the eager retry is clean."""
        key = (sig, direction)
        self.graphs.pop(key, None)
        if not self.graphs:
            self.pool = None          # its last graph is gone: the pool cannot be shared any more
        self._fault("replay", key, exc)

    def _cuda(self):
        cuda = getattr(self.torch, "cuda", None)
        return cuda if cuda is not None and cuda.is_available() else None

    def _current_stream(self):
        cuda = self._cuda()
        return cuda.current_stream() if cuda is not None else None

    def _restore_stream(self, stream):
        # A capture that raises in `capture_end` skips torch's stream restore and leaves
        # the capture stream current for the rest of the process.
        try:
            cuda = self._cuda()
            if cuda is not None and stream is not None and cuda.current_stream() != stream:
                cuda.set_stream(stream)
        except Exception:  # noqa: BLE001 -- best effort; the canary below decides
            pass

    def _device_ok(self):
        try:
            y = self.be.zeros((2, 2)) + 1.0
            self.be.to_numpy(y + y)
            cuda = self._cuda()
            if cuda is not None:
                cuda.synchronize()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _fault(self, phase, key, exc):
        sig, direction = key
        self.refused[key] = f"{phase}: {type(exc).__name__}"
        self.stats[f"{phase}_failed"] += 1
        n = self.stats["capture_failed"] + self.stats["replay_failed"]
        msg = " ".join(str(exc).split())[:240]
        if not self.stats["first_fault"]:
            self.stats["first_fault"] = f"{phase} {sig}|{direction} {msg}"
        if n <= self._FAULT_LINES:
            print(f"TNKIT_GRAPH_FALLBACK phase={phase} n={n} sig={sig}|{direction} "
                  f"err={msg}", flush=True)
        if self.fault_budget and n >= self.fault_budget and self.max_graphs > len(self.graphs):
            self.max_graphs = len(self.graphs)
            self.stats["breaker_tripped"] = n
            print(f"TNKIT_GRAPH_BREAKER n={n} kept={len(self.graphs)}: no further captures",
                  flush=True)
        if not self._device_ok():
            raise GraphContextLost(
                f"{phase} failed at {sig}|{direction} and the device could not do "
                f"arithmetic afterwards: {msg}") from exc


class SiteStepGraph:
    """One captured 1TDVP site-step for one shape signature, in one sweep direction.

    The captured region is the whole body of `tdvp._evolve_one_site_right` (or `_left`):
    the one-site Krylov exponential, the QR that moves the orthogonality centre, the
    environment update, the zero-site (bond) Krylov exponential, and the absorption of
    the bond into the neighbouring site. Everything except the Python.

    STATIC BUFFERS. Capture fixes ADDRESSES as well as shapes, so the recorded kernels
    read from wherever their operands happened to live at capture time. The caller's
    tensors move (every site has different `L[i]`, `M[i]`, ...), so each step copies its
    five inputs into the buffers the graph was recorded against, replays, and clones the
    three outputs back out. That is 8 dispatched operations to replace ~602 -- the copies
    are the price of admission and they are counted in every timing arm here.
    """

    def __init__(self, be, torch, sig, dt, m, squarings, taylor_order,
                 direction="right", pool=None):
        self.be, self.torch, self.sig, self.direction = be, torch, sig, direction
        # The MPO's two bond dimensions are carried SEPARATELY: a window's edge tensors
        # are not square in the MPO index, so one `D` would silently mis-shape the
        # environments at exactly the sites where the ramp lives.
        a, s, b, c, Dl, Dr = sig
        self.a, self.s, self.b, self.c = a, s, b, c
        z = be.zeros
        # --- static inputs -------------------------------------------------------
        self.in_M = z((a, s, b))            # the site being evolved
        self.in_Mn = z((b, s, c)) if direction == "right" else z((c, s, a))
        self.in_L = z((a, Dl, a))
        self.in_R = z((b, Dr, b))
        self.in_W = z((Dl, Dr, s, s))
        kw = dict(expm_where="taylor", squarings=squarings, taylor_order=taylor_order)

        def body():
            L, R, W, M = self.in_L, self.in_R, self.in_W, self.in_M
            if direction == "right":
                mv1 = lambda x: apply_one_site(x.reshape(a, s, b), L, W, R, be).reshape(-1)
                Mev = lanczos_expm_fixed(mv1, M.reshape(-1), -1j * dt / 2.0, be, m,
                                         **kw).reshape(a, s, b)
                Q, C = be.qr(Mev.reshape(a * s, b))
                Mo = Q.reshape(a, s, b)
                Lo = update_left_env(L, Mo, W, be)
                mv0 = lambda x: apply_zero_site(x.reshape(b, b), Lo, R, be).reshape(-1)
                Co = lanczos_expm_fixed(mv0, C.reshape(-1), +1j * dt / 2.0, be, m,
                                        **kw).reshape(b, b)
                Mno = be.einsum("rb,bsc->rsc", Co, self.in_Mn)
                return Mo, Lo, Mno
            # ---- leftward mirror: LQ via QR of the transpose -----------------------
            mv1 = lambda x: apply_one_site(x.reshape(a, s, b), L, W, R, be).reshape(-1)
            Mev = lanczos_expm_fixed(mv1, M.reshape(-1), -1j * dt / 2.0, be, m,
                                     **kw).reshape(a, s, b)
            Qt, Ct = be.qr(be.einsum("asb->sba", Mev).reshape(s * b, a))
            Mo = be.einsum("sbr->rsb", Qt.reshape(s, b, a))
            Ro = update_right_env(R, Mo, W, be)
            C = be.einsum("ra->ar", Ct)
            mv0 = lambda x: apply_zero_site(x.reshape(a, a), L, Ro, be).reshape(-1)
            Co = lanczos_expm_fixed(mv0, C.reshape(-1), +1j * dt / 2.0, be, m,
                                    **kw).reshape(a, a)
            Mno = be.einsum("csa,ar->csr", self.in_Mn, Co)
            return Mo, Ro, Mno

        self.graph, self.out = _capture(torch, body, pool=pool)
        self.pool = self.graph.pool()

    def run(self, M, Mn, L, R, W):
        """Copy in, replay, clone out. Returns (M_new, env_new, Mnext_new)."""
        self.in_M.copy_(M); self.in_Mn.copy_(Mn)
        self.in_L.copy_(L); self.in_R.copy_(R); self.in_W.copy_(W)
        self.graph.replay()
        o0, o1, o2 = self.out
        return o0.clone(), o1.clone(), o2.clone()


class SiteStepGraphCache(GraphFaults):
    """Lazily captures one `SiteStepGraph` per (signature, direction) and tallies use.

    All graphs share ONE memory pool. Without that each capture reserves its own private
    pool and a 600-site sweep would reserve hundreds of them; sharing is safe here because
    the graphs are replayed strictly one at a time.

    `stats` is reported alongside every timing number in this direction, for the same
    reason any benchmark here must print the state property
    it depends on: a graph speed-up is meaningless without knowing what FRACTION of the
    sweep was replayed rather than run eagerly. Capture and replay failures fall back to
    eager; see `GraphFaults`.
    """

    def __init__(self, be, torch, dt, m, squarings=8, taylor_order=18, max_graphs=1024,
                 fault_budget=3):
        self.be, self.torch, self.dt, self.m = be, torch, dt, m
        self.squarings, self.taylor_order = squarings, taylor_order
        self.max_graphs = max_graphs
        self.graphs, self.pool = {}, None
        self.stats = {"replayed": 0, "eager": 0, "captures": 0, "signatures": {}}
        self._init_faults(fault_budget)

    def _make(self, sig, direction):
        return SiteStepGraph(self.be, self.torch, sig, self.dt, self.m, self.squarings,
                             self.taylor_order, direction, pool=self.pool)

    def get(self, sig, direction):
        key = (sig, direction)
        if key in self.graphs:
            return self.graphs[key]
        if key in self.refused:
            return None
        if len(self.graphs) >= self.max_graphs:
            # More shape classes than graphs allowed. This cap is NOT a tuning knob -- it
            # was the single biggest lever measured in this whole direction, and setting it
            # too low silently costs most of the speed-up:
            #
            #     cap  16 ->  27.2% replayed -> 1.33x
            #     cap  64 ->  76.2% replayed -> 2.63x
            #     cap 128 ->  99.3% replayed -> 4.83x   (only 100 graphs ever built)
            #
            # A run at wavepacket width sigma=10 has exactly 100 distinct (shape, direction)
            # signatures, so that sweep SATURATES at 128 and 256/512 buy nothing.
            #
            # A LATER RUN THEN MEASURED A DIFFERENT GEOMETRY AND THE OLD DEFAULT OF 256 WAS
            # TOO LOW. A production run at sigma=30 (N_s=467, chi below the cap throughout)
            # has **296** distinct signatures. The default that was set *above* its own
            # measured requirement at the time sat *below* the very next geometry's, and
            # nothing was lost only because that later run happened to use 1024 graphs and
            # checked. The default is now 1024, i.e. 3.5x
            # the largest requirement yet measured rather than 2.6x -- but the real protection
            # is not the number, it is that the replayed fraction is REPORTED in every result,
            # so a cap that binds shows up as a number rather than as a quietly bad ratio.
            #
            # Signature counts are a property of the bond-dimension PROFILE, not of chi alone:
            # 1TDVP is exactly bond-preserving, so once the 2TDVP phase ends the profile (and
            # hence the signature set) is frozen for the rest of the leg. Each graph costs its
            # static buffers (~1 MB at chi=48) plus a share of one shared pool.
            return None                       # stay eager rather than thrash
        g = self._capture_or_none(key)
        if g is None:
            return None
        if self.pool is None:
            self.pool = g.pool
        self.graphs[key] = g
        self.stats["captures"] += 1
        self.stats["signatures"][f"{sig}|{direction}"] = 0
        return g

    def note(self, sig=None, direction=None, replayed=True):
        self.stats["replayed" if replayed else "eager"] += 1
        if replayed and sig is not None:
            self.stats["signatures"][f"{sig}|{direction}"] = \
                self.stats["signatures"].get(f"{sig}|{direction}", 0) + 1
