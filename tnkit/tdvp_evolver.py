"""TDVP that keeps its state between calls.

`tdvp1_evolve` and `tdvp2_evolve` start from scratch on every call: they copy the state to the
backend, right-canonicalise it and rebuild every environment. A driver that measures every few
steps pays that on each call. `TDVPEvolver` holds the backend state, the environments, the
Krylov schedule and the CUDA-graph cache instead, and each `run(n)` continues where the last one
stopped.

Both sweeps end with the orthogonality centre at site 0 and the right environments current, so
nothing needs rebuilding between calls. The arithmetic is the integrators' own (the same site
kernels in the same order): `run` from a fresh evolver gives the same bits as one
`tdvp*_evolve` call, and a run continued across calls gives the same bits as re-entering
`tdvp*_evolve` with `canonicalize=False`. With `canonicalize=True` the re-entry path re-gauges
the state on every call, so the two agree only to round-off.

Per-bond schedule (`method='12site'`): a bond is updated one-site when it is full and two-site
otherwise (below the cap it can grow, above it is truncated to the cap), decided afresh at the start
of every step. A bond is full when its dimension equals the cap, or when its last two-site update
was cut below the cap only to keep a group of near-equal singular values whole (`degenerate_tol`; a
rebuild of the state, after `reset`, `invalidate()` or a replaced tensor, forgets this). With
`two_site_every=k`, every k-th step (counted from `step0`) is two-site on every bond, so bonds at
the cap still exchange weight with their neighbours. A step is the symmetric sweep over blocks: each
two-site bond is a block and every site outside those blocks is a one-site block; consecutive blocks
overlap in a site or a bond, which is back-evolved. All bonds two-site is exactly the 2TDVP step,
all one-site exactly the 1TDVP step. Where a two-site block meets a one-site block the shared site
is back-evolved and then forward-evolved with the same effective Hamiltonian; the pair cancels to
Krylov tolerance.

Subspace expansion (`options.expand`, default off): before every step, each bond's right basis
gains up to that many zero-weight states from H|psi> (`tnkit.expansion`), every method runs on
the per-bond sweep with eager one-site steps, and the returning sweep cuts each one-site bond
back to its size at the start of the step (to the cap, outside '1site', for a bond held below
it by `degenerate_tol`, which the cut back also applies).

Staleness is read from `MPS.center`: `run` leaves `mps.center` at 0, and any change to the
list (replacing a tensor) or `reset(new_state)` sets it to None, so the next `run` rebuilds
from the new tensors (canonicalising them if `options.canonicalize`). Editing a tensor's data
in place is not seen; call `invalidate()`.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ._dmrg import right_canonicalize, update_right_env
from .backend import get_backend
from .mps import MPS, MPSLike
from . import invariants
from . import linalg as tl
from .expansion import expand_right
from .tdvp import (FAST_EXPM_RECAL, TDVPOptions, _check_blocks, _check_degenerate_tol,
                   _edge_envs, _is_block, _evolve_one_site_left, _evolve_one_site_left_cut,
                   _evolve_one_site_right, _evolve_two_site, _fast_step, _graphed_site_step,
                   _graphed_site_step_prefix, _KrylovMeter, _resolve_fast_expm,
                   _warn_gram_tol, resolve_graph_mode)
from . import _options

METHODS = ("1site", "2site", "12site")


class TDVPEvolver:
    """Stateful TDVP on a fixed MPO and step `dt`.

    `chi` is the two-site bond cap (`options.chi_max` overrides it, as in `tdvp2_evolve`); it may
    be None if only one-site steps are run. `options` is a `TDVPOptions` or a dict, with the
    meaning it has for the integrators; `adaptive_dt` and a callable (time-dependent) MPO are
    not supported here.

    `run(n, method)` takes n steps, `method` in {'1site', '2site', '12site'}, and returns the
    records of `options.measure` taken every `options.meas_every` steps of that call. Step dicts
    passed to `step_callback` count steps and time from `step0` (default 0), the number of steps
    already taken when the evolver is created, so a resumed run keeps its step count, time and
    `two_site_every` phase. Graph settings are checked here, not at the first one-site run.
    """

    def __init__(self, mps: MPSLike, mpo: Any, dt: float, chi: Optional[int] = None,
                 options: Optional[Union[TDVPOptions, Dict[str, Any]]] = None,
                 step0: int = 0, **kwargs: Any) -> None:
        o = _options.resolve(TDVPOptions, options, kwargs, "TDVPEvolver")
        tl.warn_if_deprecated(o.trunc)
        _warn_gram_tol(o)
        if o.adaptive_dt is not None:
            raise ValueError("TDVPEvolver does not support adaptive_dt; use tdvp*_evolve")
        if callable(mpo):
            raise ValueError("TDVPEvolver needs a fixed MPO; use tdvp*_evolve for a callable")
        if o.krylov_m < 1:
            raise ValueError(f"krylov_m must be a positive Krylov dimension, got {o.krylov_m!r}")
        _check_degenerate_tol(o)
        if o.cuda_graph is not None and o.prefix_graph is not None:
            raise ValueError("cuda_graph and prefix_graph are alternatives, not a stack.")
        if not isinstance(o.expand, int) or isinstance(o.expand, bool) or o.expand < 0:
            raise ValueError(f"expand must be an int >= 0, got {o.expand!r}")
        if o.expand and (o.L0 is not None or o.RN is not None or o.graph == "on"
                         or o.cuda_graph is not None or o.prefix_graph is not None):
            raise ValueError("expand needs open boundaries and eager one-site steps (no graphs)")
        self.o, self.dt = o, float(dt)
        self.be = be = get_backend(o.backend)
        _check_blocks(mps, mpo, be, o.fast_expm,
                      "off" if (o.cuda_graph, o.prefix_graph) == (None, None) else "on", o.trunc)
        self.mpo = [be.array(W) for W in mpo]
        cap = chi if o.chi_max is None else o.chi_max
        self.cap = None if cap is None else int(cap)
        self.rng = np.random.default_rng(o.seed + 1)
        self.step, self.t = int(step0), int(step0) * self.dt
        self._fm: Dict[str, Tuple[Any, Any]] = {}
        self._graphs: Optional[Tuple[Any, Any]] = None
        self._graph_caches(*self._fast("1site"))
        self.reset(mps)

    # ------------------------------------------------------------------ state
    @property
    def mps(self) -> MPS:
        """The live state, on the backend. Right-canonical with centre at site 0 after `run`.

        The tensors are the evolver's own: on numpy, `observables.to_numpy(ev.mps)` returns the
        same arrays. Never write into them in place without calling `invalidate()`."""
        return self._mps

    @property
    def center(self) -> Optional[int]:
        """`mps.center`: 0 while the state is as the last step left it, else None."""
        return self._mps.center

    def reset(self, mps: MPSLike) -> None:
        """Take a new state; it is moved to the backend and rebuilt at the next `run`. The step
        count, time and Krylov calibration carry on; create a new evolver for a new trajectory."""
        invariants.entry(mps, self.mpo, self.o.L0, self.o.RN)
        _check_blocks(mps, self.mpo, self.be)
        self._mps = MPS(self.be.array(m) for m in mps)
        self._pinned = [False] * max(len(self._mps) - 1, 0)

    def prepare(self) -> None:
        """Canonicalise (if `options.canonicalize`) and build the environments now rather than in
        the next `run`; the arithmetic is the same."""
        if self._mps.center != 0:
            self._rebuild()
            self._mps.center = 0

    def invalidate(self) -> None:
        """Force a rebuild at the next `run` (after editing a tensor's data in place)."""
        self._mps.center = None

    def _rebuild(self) -> None:
        be, o, M, W = self.be, self.o, self._mps, self.mpo
        invariants.entry(M, W, o.L0, o.RN)
        M[:] = [be.array(m) for m in M]
        self._pinned = [False] * max(len(M) - 1, 0)
        if o.canonicalize:
            right_canonicalize(M, be)
        N = len(M)
        _check_blocks(M, W, be, o.fast_expm, "off" if self._graphs == (None, None) else "on")
        self.L, self.R = [None] * (N + 1), [None] * (N + 1)
        self.L[0], self.R[N] = _edge_envs(M, W, be, o.L0, o.RN)
        for i in range(N - 1, 0, -1):
            self.R[i] = update_right_env(self.R[i + 1], M[i], W[i], be)

    # ------------------------------------------------------------------ solvers
    def _fast(self, method: str):
        if method not in self._fm:
            fm, cache = _resolve_fast_expm(self.o.fast_expm, None)
            self._fm[method] = ({"fm": fm}, cache)
        return self._fm[method]

    def _graph_caches(self, spec, cache):
        """(cuda_graph, prefix_graph) for one-site steps, checked against the Krylov schedule of
        the method about to run and built once per evolver."""
        o = self.o
        fm = spec["fm"]
        gc, pgc = o.cuda_graph, o.prefix_graph
        if gc is not None and (fm is None or not fm.get("m")
                               or (cache is not None and cache.get("recal", FAST_EXPM_RECAL))):
            raise ValueError("cuda_graph requires a frozen Krylov schedule (fast_expm=<int>) "
                             "with recal=0.")
        if pgc is not None and ((fm is not None and fm.get("m")) or o.krylov_m != 30):
            raise ValueError("prefix_graph requires the adaptive solver at the default krylov_m.")
        if gc is None and pgc is None and o.graph != "off":
            if self._graphs is not None:
                return self._graphs
            mode = resolve_graph_mode(o.graph, self.be)
            frozen = fm is not None and fm.get("m")
            if mode == "capture" and (frozen or o.krylov_m != 30):
                if o.graph == "on":
                    raise ValueError("graph='on' needs fast_expm=None and the default krylov_m.")
                mode = "eager"
            if mode == "capture":
                import torch
                from .prefix_graph_impl import PrefixSiteStepGraphCache
                pgc = PrefixSiteStepGraphCache(self.be, torch, self.dt, o.graph_m_max,
                                               max_graphs=o.graph_cache_cap)
        self._graphs = (gc, pgc)
        return self._graphs

    # ------------------------------------------------------------------ sweeps
    def _one_site(self, i: int, h: float, direction: str, meter, fm, gc, pgc) -> None:
        M, L, R, W, be = self._mps, self.L, self.R, self.mpo, self.be
        graphed = _graphed_site_step if gc is not None else _graphed_site_step_prefix
        cache = gc if gc is not None else pgc
        if cache is not None:
            if graphed(M, L, R, W, i, be, cache, direction):
                return
            cache.note(replayed=False)
        step = _evolve_one_site_right if direction == "right" else _evolve_one_site_left
        step(M, L, R, W, i, h, be, meter, fm, self.o.krylov_m)

    def _sweep1(self, h: float, meter, fm, eager: bool) -> None:
        gc, pgc = (None, None) if eager else self._graph_caches(*self._fast("1site"))
        N = len(self._mps)
        for i in range(N):
            self._one_site(i, h, "right", meter, fm, gc, pgc)
        for i in range(N - 1, -1, -1):
            self._one_site(i, h, "left", meter, fm, gc, pgc)

    def _sweep2(self, h: float, meter, fm, pstep) -> None:
        o, M, L, R, W, be = self.o, self._mps, self.L, self.R, self.mpo, self.be
        tr = dict(trunc=o.trunc, svd_tol=o.svd_tol, rng=self.rng, renorm=o.renorm, meter=meter,
                  fm=fm, krylov_m=o.krylov_m, degenerate_tol=o.degenerate_tol)
        N, pin = len(M), self._pinned
        for i in range(0, N - 1):
            pin[i] = _evolve_two_site(M, L, R, W, i, self.cap, h, be, "right", pstep, **tr)
        for i in range(N - 2, -1, -1):
            pin[i] = _evolve_two_site(M, L, R, W, i, self.cap, h, be, "left", pstep, **tr)

    def _expand(self, pstep) -> None:
        """Subspace expansion (`options.expand`), then the right environments it changed."""
        M, W, be, N = self._mps, self.mpo, self.be, len(self._mps)
        if pstep is not None:
            be.sync(); t0 = time.perf_counter()
        expand_right(M, W, self.o.expand, self.cap, be, self.L[0], self.R[N], check=False)
        for i in range(N - 1, 0, -1):
            self.R[i] = update_right_env(self.R[i + 1], M[i], W[i], be)
        if pstep is not None:
            be.sync(); pstep["expand"] += time.perf_counter() - t0

    def _sweep12(self, h: float, meter, fm, pstep, two, eager: bool, cut=None) -> None:
        """One step on the per-bond schedule; `two[i]` says whether bond (i, i+1) is two-site.
        With `cut`, the returning sweep cuts each one-site bond i back to `cut[i]` states."""
        o, M, L, R, W, be = self.o, self._mps, self.L, self.R, self.mpo, self.be
        gc, pgc = (None, None) if eager else self._graph_caches(*self._fast("12site"))
        tr = dict(trunc=o.trunc, svd_tol=o.svd_tol, rng=self.rng, renorm=o.renorm, meter=meter,
                  fm=fm, krylov_m=o.krylov_m, degenerate_tol=o.degenerate_tol)
        N, pin = len(M), self._pinned

        def one(i, direction, chi=None):
            t0 = None
            if pstep is not None:
                be.sync(); t0 = time.perf_counter()
            if chi is not None:                      # leftward, cutting bond i-1 to chi
                pin[i - 1] = _evolve_one_site_left_cut(M, L, R, W, i, h, chi, be, meter, fm,
                                                       o.krylov_m, o.renorm, pstep,
                                                       o.degenerate_tol)
            else:
                self._one_site(i, h, direction, meter, fm, gc, pgc)
            if t0 is not None:
                be.sync(); pstep["one"] += time.perf_counter() - t0

        for i in range(N):
            if i < N - 1 and two[i]:
                pin[i] = _evolve_two_site(M, L, R, W, i, self.cap, h, be, "right", pstep, **tr)
            elif not (i == N - 1 and N > 1 and two[N - 2]):    # last site of a two-site block
                one(i, "right")
        for i in range(N - 1, -1, -1):
            if i > 0 and two[i - 1]:
                pin[i - 1] = _evolve_two_site(M, L, R, W, i - 1, self.cap, h, be, "left", pstep,
                                              **tr)
            elif not (i == 0 and N > 1 and two[0]):
                if cut is not None and i > 0 and int(M[i].shape[0]) > cut[i - 1]:
                    one(i, "left", cut[i - 1])
                else:
                    one(i, "left")

    # ------------------------------------------------------------------ driver
    def run(self, n_steps: int, method: str = "2site",
            step_callback: Optional[Callable[..., Any]] = None,
            two_site_every: int = 0) -> List[Tuple[float, Any]]:
        """Take `n_steps` steps of `method`; `step_callback` overrides `options.step_callback`.
        `two_site_every` (method '12site' only): every that many steps is two-site everywhere."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if two_site_every and method != "12site":
            raise ValueError("two_site_every applies to method='12site' only")
        if two_site_every < 0:
            raise ValueError(f"two_site_every must be >= 0, got {two_site_every!r}")
        if (method != "1site" or self.o.expand) and self.cap is None:
            raise ValueError("two-site steps and expand need a bond cap: pass chi or "
                             "options.chi_max")
        o, be = self.o, self.be
        if self._mps.center != 0:
            self._rebuild()
        cb = step_callback if step_callback is not None else o.step_callback
        prof = o.profile
        do_prof = (method != "1site" or o.expand) and (prof is not None or cb is not None)
        if do_prof and prof is None:
            prof = {}
        if do_prof:
            prof.setdefault("steps", [])
        kmeter = _KrylovMeter() if (do_prof or cb is not None) else None
        spec, cache = self._fast(method)
        snapshot = (lambda: self._mps) if o.keep_device or _is_block(self._mps[0]) else \
            (lambda: [be.to_numpy(m) for m in self._mps])
        records = []

        def sweep(h, meter=None):
            eager = meter is not None                # a calibration sweep runs eager
            if meter is None and kmeter is not None:
                kmeter.reset()
                meter = kmeter
            pstep = ({"expm": 0.0, "trunc": 0.0, "env": 0.0, "disc": 0.0, "one": 0.0,
                      "expand": 0.0} if do_prof else None)
            n_two, M = None, self._mps
            if method == "12site":
                every = two_site_every and (self.step + 1) % two_site_every == 0
                two = [bool(every) or (int(M[i].shape[2]) != self.cap and not self._pinned[i])
                       for i in range(len(M) - 1)]
                n_two = sum(two)
            else:
                two = [method == "2site"] * (len(M) - 1)
            if o.expand:                  # pinned two-site bonds go back to the cap
                cut = [int(m.shape[2]) if method == "1site" or not p else self.cap
                       for m, p in zip(M, self._pinned)]
                self._expand(pstep)
                self._sweep12(h, meter, spec["fm"], pstep, two, True, cut)
            elif method == "2site":
                self._sweep2(h, meter, spec["fm"], pstep)
            elif method == "1site":
                self._sweep1(h, meter, spec["fm"], eager)
            else:
                self._sweep12(h, meter, spec["fm"], pstep, two, eager)
            invariants.sweep(self._mps, self.mpo, o.L0, o.RN)
            self.step += 1
            self.t += h
            if not do_prof:
                return meter
            d = {"step": self.step, "t": self.t, "dt": h,
                 "max_chi": max(int(m.shape[2]) for m in self._mps[:-1]),
                 "disc": pstep["disc"], "expm_s": pstep["expm"], "trunc_s": pstep["trunc"],
                 "env_s": pstep["env"], "krylov_max": meter.m_max,
                 "krylov_cap_hits": meter.cap_hits}
            if n_two is not None:
                d.update(one_site_s=pstep["one"], n_two=n_two)
            if o.expand:
                d["expand_s"] = pstep["expand"]
            prof["steps"].append(d)
            return d

        for k in range(n_steps):
            out = _fast_step(sweep, self.dt, spec, cache)
            rec = None
            if o.measure is not None and (k + 1) % o.meas_every == 0:
                rec = (self.t, o.measure(snapshot()))
                records.append(rec)
            if cb is not None:
                d = out if do_prof else {"step": self.step, "t": self.t, "dt": self.dt,
                                          "krylov_max": out.m_max,
                                          "krylov_cap_hits": out.cap_hits}
                cb(d, rec)
        if n_steps > 0:
            self._mps.center = 0                    # every sweep ends at site 0
        return records
