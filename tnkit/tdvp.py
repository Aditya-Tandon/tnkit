"""TDVP (the time-dependent variational principle) for real-time evolution of an MPS.

Where TEBD applies Trotterised nearest-neighbour gates, TDVP exponentiates the effective
Hamiltonian of each block, the matrix-free action `apply_two_site` / `apply_one_site`, with a
Krylov (Lanczos) exponential (`lanczos_expm`). It works from an MPO, so it also handles
long-range Hamiltonians (2D cylinders, lattice gauge theories) where nearest-neighbour gates
do not apply. Time evolution is ``exp(-i H t)`` with hbar = 1.

Two-site TDVP, a symmetric second-order integrator per step dt:
  forward sweep (left to right): at each bond, evolve the two-site block by exp(-i H2 dt/2),
    cut it back to chi by SVD (bonds can grow), then evolve the new one-site centre back by
    exp(+i H1 dt/2);
  backward sweep (right to left): the mirror image.
Every bond is evolved forward by dt in total and every interior site back by dt. One-site TDVP
(`tdvp1_evolve`) does the same on single sites and keeps every bond's size.
"""
from __future__ import annotations
import dataclasses
import numbers
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .backend import get_backend
from .eigensolver import lanczos_expm, lanczos_expm_fixed
from .legs import fuse, split
from . import block_graph, invariants
from . import linalg as tl
from .mps import MPS, MPSLike, Tensor
from . import _options
from ._dmrg import (apply_two_site, apply_one_site, apply_zero_site, update_left_env,
                   update_right_env, right_canonicalize, _truncate, _cut, _edge_envs)
from ._sparse import bond_states, linalg_of


def _is_block(x) -> bool:
    return getattr(x, "is_block", False)


def _krylov(fn, x, tau, be, fm, info, krylov_m):
    """exp(tau H) x for a dense or block tensor `x`, H given by `fn` on tensors of x's shape. For
    block tensors on CUDA, `block_graph` may replay `fn` from CUDA graphs (off by default)."""
    if _is_block(x):
        with block_graph.solve(fn, x) as f:
            return _expm_apply(f, x, tau, be, fm, info, krylov_m)
    shape = tuple(int(n) for n in x.shape)
    return _expm_apply(lambda v: fn(v.reshape(shape)).reshape(-1), x.reshape(-1), tau, be, fm,
                       info, krylov_m).reshape(shape)


def _check_blocks(M, mpo, be, fast_expm=None, graph="off", trunc="svd"):
    """Block tensors need a block MPS, a block MPO and a `BlockBackend` (symmetric tensors: a
    symmetric MPS and MPO and a `SymmetricBackend`), truncate with 'svd', and run the adaptive
    Krylov solver eagerly: no frozen schedule and no step graphs (`block_graph` may replay the
    matvecs of block tensors)."""
    if not (_is_block(M[0]) or _is_block(mpo[0])):
        return
    from .block_backend import BlockBackend
    from .symmetric_backend import SymmetricBackend
    from .symmetric_tensor import SymmetricTensor
    kinds = {isinstance(M[0], SymmetricTensor), isinstance(mpo[0], SymmetricTensor)}
    adaptor = SymmetricBackend if True in kinds else BlockBackend
    if not (_is_block(M[0]) and _is_block(mpo[0]) and len(kinds) == 1
            and isinstance(be, adaptor)):
        raise ValueError("block tensors need a block MPS, a block MPO and "
                         "backend=BlockBackend(dense); symmetric tensors a symmetric MPS and MPO "
                         "and backend=SymmetricBackend(dense)")
    if fast_expm is not None or graph != "off":
        raise ValueError("block tensors need fast_expm=None and no CUDA graphs")
    if trunc != "svd":
        raise ValueError(f"block tensors truncate with trunc='svd' only, got {trunc!r}")


@dataclasses.dataclass
class TDVPOptions:
    """Settings of `tdvp1_evolve`, `tdvp2_evolve` and `TDVPEvolver`, apart from the state, the
    Hamiltonian and the time grid.

    Every field can also be passed to the integrators as a keyword, with the same default.
    A field that belongs to one integrator raises when it is passed as a keyword to the other;
    set inside a `TDVPOptions`, it is ignored by the other integrator.

    Parameters
    ----------
    backend : str or backend, default 'numpy'
        As in `get_backend`. Time evolution needs a complex backend. With block tensors
        (provisional: a block MPS and MPO with ``backend=BlockBackend(dense)``) or symmetric
        tensors (``backend=SymmetricBackend(dense)``), `measure` and the returned MPS get
        tensors of that type.
    measure : callable or None, default None
        ``measure(state)`` is called at t = 0 and every `meas_every` steps, and its return
        value is recorded with the time. `state` is a list of host numpy tensors (dense
        backends, unless `keep_device`), or the backend's own tensors.
    meas_every : int, default 1
        Steps between measurements, >= 1.
    L0, RN : tensor or None, default None
        Boundary environments, for a window on an infinite background (`tnkit.window`); None
        for open boundaries.
    canonicalize : bool, default True
        Bring the state to right-canonical form before the first step. A window on an infinite
        background needs it once if its centre is not at site 0 (a built window has it in the
        middle); this leaves the two boundary bonds that meet `L0` and `RN` unchanged. Later
        calls may pass False.
    keep_device : bool, default False
        Hand `measure` the backend's tensors instead of host copies, which saves a transfer
        per state and measurement when many states are evolved in a loop.
    adaptive_dt : dict or None, default None
        None keeps a fixed step. A dict of controller parameters (``tol``, ``order``,
        ``safety``, ``fmin``, ``fmax``, ``dt_min``, ``dt_max``, ``tol_scale``, ``log``) picks the
        step size per step from the Krylov error estimate (see `_DtController`); `dt` is then
        the initial step and ``dt * n_steps`` the total time. Measurement times are those of
        the fixed-step run. It saves time, not accuracy: in a run whose bonds grow, `svd_tol`
        sets the accuracy.
    fast_expm : None, 'auto', int or dict, default None
        Fixed-length Krylov exponentials without host synchronisation, for GPUs
        (`eigensolver.lanczos_expm_fixed`). None: the adaptive solver. int: that fixed Krylov
        dimension. 'auto': the first sweep runs the adaptive solver, and the dimension is fixed
        2 above the largest it needed, then re-measured every 25 steps. dict: as 'auto', with
        the dimension written back as ``'m'`` (and ``'recal'`` for the re-measurement
        interval, 0 to never re-measure), so a caller that re-enters the integrator reuses it.
        Not with `adaptive_dt`.
    krylov_m : int, default 30
        Largest Krylov dimension of the adaptive exponential. The fixed schedules of
        `fast_expm` and of graph replays (`graph_m_max`) set their own; ``graph='auto'``
        therefore runs eagerly when `krylov_m` is not 30.
    step_callback : callable or None, default None
        ``step_callback(step_dict, obs)`` at the end of every step, where `obs` is
        ``(t, measure(state))`` on measured steps and None otherwise; it can write each row as
        it is produced. ``step_dict['krylov_max']`` is the largest Krylov dimension of the
        step's eager solves and ``step_dict['krylov_cap_hits']`` the number of adaptive solves
        that stopped at the `krylov_m` cap without meeting the solver tolerance (graph-replayed
        sites are not counted, and fixed schedules report 0). The two-site dict also carries
        the fields of `profile`. Not with `adaptive_dt` in `tdvp1_evolve`.
    trunc : str, default 'svd'
        Two-site only. As in `DMRGOptions`. 'gram' is for float64 with ``svd_tol >= 1e-12``
        (lower warns), and for energies, local observables and entropies only: it does not
        reproduce 'svd' bit for bit, and in chaotic dynamics the two drift apart. Single
        precision, block tensors and `degenerate_tol` are refused with 'gram'.
    chi_max : int or None, default None
        Two-site only. Hard cap on every bond; None means `chi`. Give it when the input state
        already has bonds larger than `chi` (a window on an infinite background has 3 chi_bg),
        or the first step truncates the initial state.
    svd_tol : float, default 0.0
        Two-site only. Relative discarded weight allowed per bond. 0 keeps `chi` singular
        values whenever there are that many; > 0 keeps the smallest rank whose dropped weight
        ``sum(s_dropped^2) / sum(s^2)`` stays below `svd_tol`, still capped at `chi`.
    renorm : bool, default True
        Two-site only. Rescale the Schmidt values to unit norm after every cut. False leaves
        the norm lost to truncation visible.
    seed : int, default 0
        Two-site only. Seeds the randomized truncations.
    profile : dict or None, default None
        Two-site only. If given, ``profile['steps']`` gets one dict per step: ``step``, ``t``,
        ``dt``, ``max_chi`` (largest bond after the step; multiplets on symmetric tensors),
        ``disc`` (summed relative discarded weight), ``expm_s``, ``trunc_s``, ``env_s``,
        ``krylov_max`` and ``krylov_cap_hits``. It adds GPU synchronisations.
    degenerate_tol : float, default 0.0
        Two-site only, experimental. 0 cuts exactly where the cap and `svd_tol` say. A value
        in (0, 1), with ``trunc='svd'``, keeps or drops whole a group of singular values equal
        within this relative tolerance of the value next to the cut: at the cap the cut moves
        down to the start of the group, below the cap the `svd_tol` cut moves up to its end. A
        group that starts at the largest value is split, since moving down would keep nothing.
        A bond cut below the cap this way counts as full in the '12site' schedule of
        `TDVPEvolver`. It is not known to improve accuracy.
    expand : int, default 0
        `TDVPEvolver` only. 0 leaves the bond bases as the sweeps make them. n > 0 adds, before
        every step, up to n states to the right basis of each bond: the part of H|psi> the basis
        misses (`tnkit.expansion.expand_right`), with zero weight. The step can then move weight
        into them, and on block tensors into charge sectors a bond lacked. The returning sweep
        cuts every one-site bond back, by SVD, to its size before the expansion, so the cap and
        the '12site' schedule are as without it; that cut applies `degenerate_tol`, and outside
        '1site' a bond it pinned goes back to the cap. The expansion's own cuts ignore
        `degenerate_tol`, so on a degenerate spectrum dense and block runs can keep different
        states. Open boundaries only.
    cuda_graph, prefix_graph : cache or None, default None
        One-site only. Graph caches built by the caller (`tnkit.cudagraph`,
        `tnkit.prefix_graph_impl`); either one bypasses `graph`.
    graph : {'auto', 'on', 'off'}, default 'auto'
        One-site only. Whether to build a prefix-graph cache when neither cache is given; see
        `resolve_graph_mode`. 'auto' captures only on a complex torch-cuda backend with a CUDA
        device, and runs eagerly elsewhere.
    graph_m_max, graph_cache_cap : int, default 12 and 1024
        One-site only. Krylov length and number of graphs of the cache `graph` builds.
    """

    # shared
    backend: Any = "numpy"
    measure: Optional[Callable[[Any], Any]] = None
    meas_every: int = 1
    L0: Optional[Tensor] = None
    RN: Optional[Tensor] = None
    canonicalize: bool = True
    keep_device: bool = False
    adaptive_dt: Optional[Dict[str, Any]] = None
    fast_expm: Union[None, str, int, Dict[str, Any]] = None
    krylov_m: int = 30
    step_callback: Optional[Callable[..., Any]] = None
    # two-site only
    trunc: str = "svd"
    chi_max: Optional[int] = None
    svd_tol: float = 0.0
    renorm: bool = True
    seed: int = 0
    profile: Optional[Dict[str, Any]] = None
    degenerate_tol: float = 0.0
    # TDVPEvolver only
    expand: int = 0
    # one-site only
    cuda_graph: Optional[Any] = None
    prefix_graph: Optional[Any] = None
    graph: str = "auto"
    graph_m_max: int = 12
    graph_cache_cap: int = 1024


_TDVP2_ONLY = ("trunc", "chi_max", "svd_tol", "renorm", "seed", "profile", "degenerate_tol")
_TDVP1_ONLY = ("cuda_graph", "prefix_graph", "graph", "graph_m_max", "graph_cache_cap")
_TDVP_SHARED = ("backend", "measure", "meas_every", "L0", "RN", "canonicalize",
                "keep_device", "adaptive_dt", "fast_expm", "krylov_m", "step_callback")


def _tdvp_opts(which: str, options: Any, legacy: Dict[str, Any]) -> TDVPOptions:
    """Resolve TDVP options for `which` in {'tdvp1_evolve', 'tdvp2_evolve'} and
    refuse settings that belong to the other integrator.

    Refused only when they are passed FLAT at this call site. Writing
    `tdvp1_evolve(..., svd_tol=1e-8)` is a misunderstanding worth a loud error --
    1TDVP is strictly bond-preserving and truncates nothing, so the budget would
    have no effect. Carrying the same field inside a shared `TDVPOptions` is not a
    mistake at all: the recommended protocol is 2TDVP until chi saturates and then
    1TDVP, one settings bundle across both, and the bundle necessarily holds
    fields only one of them reads."""
    o = _options.resolve(TDVPOptions, options, legacy, which)
    if o.expand:
        raise ValueError("expand is a TDVPEvolver option; tdvp1_evolve and tdvp2_evolve do not "
                         "expand")
    if o.krylov_m < 1:
        raise ValueError(f"krylov_m must be a positive Krylov dimension, got {o.krylov_m!r}")
    _check_meas_every(o)
    _check_degenerate_tol(o)
    other = _TDVP1_ONLY if which == "tdvp2_evolve" else _TDVP2_ONLY
    wrong = [k for k in other if k in legacy]
    if wrong:
        raise TypeError(
            f"{which}() does not accept {wrong[0]!r}: it belongs to the "
            f"{'one' if which == 'tdvp2_evolve' else 'two'}-site integrator. "
            f"{which}() accepts {sorted(_TDVP_SHARED + (_TDVP2_ONLY if which == 'tdvp2_evolve' else _TDVP1_ONLY))}.")
    return o


def _warn_gram_tol(o: TDVPOptions) -> None:
    if o.trunc == "gram" and 0 < o.svd_tol < 1e-12:
        warnings.warn(f"trunc='gram' with svd_tol={o.svd_tol:g} < 1e-12: the cut falls at the "
                      "Gram noise floor, so the kept rank follows rounding", UserWarning,
                      stacklevel=3)


def _check_meas_every(o: TDVPOptions) -> None:
    m = o.meas_every
    if isinstance(m, bool) or not isinstance(m, numbers.Integral) or m < 1:
        raise ValueError(f"meas_every must be an integer >= 1, got {m!r}")


def _check_degenerate_tol(o: TDVPOptions) -> None:
    if not 0 <= o.degenerate_tol < 1:
        raise ValueError(f"degenerate_tol must be in [0, 1), got {o.degenerate_tol!r}")
    if o.degenerate_tol and o.trunc != "svd":
        raise ValueError(f"degenerate_tol needs trunc='svd', got {o.trunc!r}")


def _expm_apply(matvec, x, tau, be, spec, info, krylov_m=30):
    """exp(tau H) x through the requested Krylov solver.

    `spec` is None (the adaptive solver, capped at `krylov_m`) or a dict
    {m, mode, reorth, expm_where}:
      mode='fast'   -> `lanczos_expm_fixed`: fixed schedule, tridiagonal on the device, one
                       synchronisation per solve, batched reorthogonalisation;
      mode='legacy' -> the adaptive solver with its adaptive stop off: the same fixed schedule,
                       with the tridiagonal assembled on the host. It separates the cost of the
                       fixed schedule from that of the synchronisations.
    """
    if not spec or not spec.get("m"):
        return lanczos_expm(matvec, x, tau, be, m=krylov_m, info=info)
    if spec.get("mode", "fast") == "legacy":
        return lanczos_expm(matvec, x, tau, be, m=spec["m"], adaptive=False, info=info)
    return lanczos_expm_fixed(matvec, x, tau, be, spec["m"], info=info,
                              reorth=spec.get("reorth", "cgs2"),
                              expm_where=spec.get("expm_where", "host"),
                              squarings=spec.get("squarings", 8),
                              taylor_order=spec.get("taylor_order", 18))


def _resolve_fast_expm(fast_expm, adaptive_dt):
    """Validate the `fast_expm` request and return (spec or None, cache or None).

    Accepted forms:
      None    -- off: the adaptive solver.
      int     -- a Krylov schedule fixed at this dimension.
      'auto'  -- one first sweep on the adaptive solver; the schedule is fixed just above the
                 largest Krylov dimension it needed.
      dict    -- 'auto', with the dimension written back into the dict as 'm' and reused, for a
                 caller that re-enters `tdvp*_evolve` once per measurement interval.

    The dimension is calibrated on the run rather than tabulated: the right m depends on
    ``abs(tau) * norm(H_eff)``, not on dt alone, and the adaptive solver measures it for this
    state, step and coupling.
    """
    if fast_expm is None:
        return None, None
    if adaptive_dt is not None:
        raise ValueError(
            "fast_expm is incompatible with adaptive_dt. The adaptive step-size controller uses "
            "`est_ref`, the Krylov error estimate at a fixed dimension, which needs a device "
            "scalar read back to the host at every Lanczos step; the fast path avoids exactly "
            "those reads. Use one or the other.")
    if isinstance(fast_expm, dict):
        fast_expm.setdefault("m", None)
        return fast_expm, (fast_expm if fast_expm.get("m") is None
                           or fast_expm.get("recal", FAST_EXPM_RECAL) else None)
    if fast_expm == "auto":
        spec = {"m": None}
        return spec, spec
    fm = int(fast_expm)
    if fm < 1:
        raise ValueError(f"fast_expm must be None, 'auto', a dict or a positive Krylov "
                         f"dimension, got {fast_expm!r}")
    return {"m": fm}, None


FAST_EXPM_HEADROOM = 2          # Krylov steps carried above the largest one needed
FAST_EXPM_RECAL = 25            # steps between re-measurements of that dimension


def _calibrate_m(meter, headroom=FAST_EXPM_HEADROOM, floor=4):
    """The fixed Krylov dimension from a calibration sweep: the largest dimension the
    adaptive solver needed over the sweep's block exponentials (the hardest block decides),
    plus `headroom`.

    The margin is additive: the dimensions needed vary by about one step around their mode,
    and the neglected term falls like ``abs(tau)^j beta_1 ... beta_j / (j-1)!``, so each extra
    step adds about an order of magnitude of margin, where a percentage margin would mostly add
    matvecs.
    """
    return max(floor, int(meter.m_max) + int(headroom))


def _fast_step(sweep, dt, state, cache):
    """One sweep, re-measuring the fixed Krylov schedule when it is due.

    The local stiffness ``abs(tau) * norm(H_eff)`` can grow during a run (for instance when
    wave packets meet), so every `recal` steps one sweep runs the adaptive solver with a meter
    attached and the dimension is set again. That costs 1/recal of the sweeps (4% at the
    default 25) and stands in for the per-step error estimate the fast path lacks.

    `cache['under']` counts the re-measurements that found the fixed schedule had been too
    short. It is a diagnostic, not a correction: a non-zero count means some steps of the
    preceding interval ran under-converged.
    """
    if cache is None:
        return sweep(dt)
    spec = state["fm"]
    recal = cache.get("recal", FAST_EXPM_RECAL)
    if spec.get("m") is not None and not (recal and cache.get("since", 0) >= recal):
        cache["since"] = cache.get("since", 0) + 1
        return sweep(dt)
    prev = spec.get("m")
    spec["m"] = None                                 # this one sweep runs adaptive, metered
    cal = _KrylovMeter()
    out = sweep(dt, cal)
    if prev is not None and int(cal.m_max) > prev:
        cache["under"] = cache.get("under", 0) + 1
    spec["m"] = cache["m"] = _calibrate_m(cal)
    cache["since"] = 0
    return out


class _KrylovMeter:
    """Per-step aggregate of the Krylov diagnostics over every block solve in one TDVP step.

    The quantity that matters is `est_max` — the largest frozen-j Saad estimate over the
    ~4N exponentials of a sweep. Max, not mean: the step size must satisfy the HARDEST
    block, and a mean over 4N blocks is dominated by the easy bulk.

    Cost: one dict write per `lanczos_expm` call; on the adaptive solver the estimate at j=2
    is computed anyway by the adaptive stop.
    """
    __slots__ = ("n", "est_max", "est_sum", "beta1_max", "m_max", "m_sum", "cap_hits", "info")

    def __init__(self):
        self.info = {}
        self.reset()

    def reset(self):
        self.n = 0
        self.est_max = 0.0
        self.est_sum = 0.0
        self.beta1_max = 0.0
        self.m_max = 0
        self.m_sum = 0
        self.cap_hits = 0

    def collect(self):
        d = self.info
        e = d.get("est_ref", float("nan"))
        if e == e:                                   # NaN-safe
            if e > self.est_max:
                self.est_max = e
            self.est_sum += e
        b = d.get("beta1", float("nan"))
        if b == b and b > self.beta1_max:
            self.beta1_max = b
        mm = int(d.get("krylov_dim", 0))
        if mm > self.m_max:
            self.m_max = mm
        self.m_sum += mm
        self.cap_hits += bool(d.pop("capped", False))   # popped: the info dict is reused
        self.n += 1

    def as_dict(self):
        n = max(self.n, 1)
        return {"est_max": self.est_max, "est_mean": self.est_sum / n,
                "beta1_max": self.beta1_max, "m_max": self.m_max,
                "m_mean": self.m_sum / n, "m_sum": self.m_sum, "n_solves": self.n,
                "cap_hits": self.cap_hits}


class _DtController:
    """Elementary (integral) step-size controller on the frozen-j Saad estimate.

    Why not the adaptive solver's own `err_est`: that one is driven to the Krylov tolerance
    (1e-12) by construction — it says how well the exponential was computed, not how big a
    step the dynamics can take, and it is essentially constant in dt. `est_ref` freezes the
    same estimator at a fixed Krylov dimension j, where it behaves as

        est_ref  ~  |tau|^j * beta_1 ... beta_j / (j-1)!   =   C(state) * dt^j,

    a genuine local-stiffness meter. Equidistributing it over the trajectory is the classical
    step-size rule

        dt_new = h * safety * (tol / est)^(1/order),        order = j,

    with a growth clamp and hard bounds. No step REJECTION: TDVP mutates the MPS and the
    environments in place, so a rejected step would cost a full state+environment snapshot
    every step. The growth clamp (`fmax`) keeps the controller safe instead.

    `tol=None` calibrates on the first step (tol = est_1 * tol_scale), so an adaptive run
    starts at the step size `dt` asked for and only then adapts.
    """

    def __init__(self, tol=None, order=2.0, safety=0.9, fmin=0.5, fmax=1.5,
                 dt_min=0.0, dt_max=float("inf"), tol_scale=1.0):
        self.tol = tol
        self.order = float(order)
        self.safety = float(safety)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.dt_min, self.dt_max = float(dt_min), float(dt_max)
        self.tol_scale = float(tol_scale)

    def update(self, dt_prop, est, h):
        """New proposal from the step just taken (size `h`, measured estimate `est`)."""
        if self.tol is None:
            # Auto-calibrate on step 1 so the controller's EQUILIBRIUM step is exactly the dt the
            # caller asked for: the fixed point of dt -> dt*safety*(tol/est)^(1/order) is
            # est = tol*safety^order, so the safety factor has to be divided back out here.
            # Without this the step would settle at safety * dt0 instead of dt0.
            self.tol = max(est, 1e-300) / self.safety ** self.order * self.tol_scale
        if not np.isfinite(est) or est <= 0.0:
            raw = h * self.fmax
        else:
            raw = h * self.safety * (self.tol / est) ** (1.0 / self.order)
        raw = min(max(raw, self.fmin * dt_prop), self.fmax * dt_prop)
        return float(min(max(raw, self.dt_min), self.dt_max))


def _meas_targets(dt, n_steps, meas_every):
    """Exact times an adaptive run must land on: the fixed-dt measurement grid, plus t_end.

    The adaptive stepper NEVER overshoots one of these — it subdivides the remaining interval
    evenly — so its `records` carry the identical time grid as the fixed-dt run and the two
    trajectories are directly comparable point by point.
    """
    t_end = dt * n_steps
    out = [dt * meas_every * k for k in range(1, n_steps // max(meas_every, 1) + 1)]
    if not out or abs(out[-1] - t_end) > 1e-12 * max(1.0, t_end):
        out.append(t_end)
    return out


def _run_adaptive(sweep, targets, dt0, params):
    """Drive `sweep(h, meter)` with a controlled step size, landing exactly on `targets`.

    Yields (t, n_steps_so_far) at every target, so the caller measures on the same grid a
    fixed-dt run would. Returns the controller in `params['controller']` if a dict is given.
    """
    p = dict(params or {})
    ctrl = _DtController(tol=p.get("tol"), order=p.get("order", 2.0),
                         safety=p.get("safety", 0.9), fmin=p.get("fmin", 0.5),
                         fmax=p.get("fmax", 1.5),
                         dt_min=p.get("dt_min", dt0 / 16.0),
                         dt_max=p.get("dt_max", dt0 * 16.0),
                         tol_scale=p.get("tol_scale", 1.0))
    log = p.get("log")
    if isinstance(params, dict):
        params["controller"] = ctrl
    meter = _KrylovMeter()
    dt_prop, t, nst = float(dt0), 0.0, 0
    for tgt in targets:
        while t < tgt - 1e-12 * max(1.0, tgt):
            rem = tgt - t
            nsub = max(int(np.ceil(rem / dt_prop - 1e-9)), 1)
            h = rem / nsub                           # even subdivision: exact landing, no runt
            meter.reset()
            sweep(h, meter)
            t += h
            nst += 1
            est = meter.est_max
            dt_prop = ctrl.update(dt_prop, est, h)
            if log is not None:
                rec = {"step": nst, "t": t, "dt": h, "est": est, "dt_prop": dt_prop}
                rec.update(meter.as_dict())
                log.append(rec)
        yield t, nst


def _svd_cut(mat, chi, be, trunc="svd", svd_tol=0.0, degenerate_tol=0.0, rng=None, iso=None):
    """Truncated SVD of a matrix (dense or block) at the cap `chi`, with `svd_tol` and
    `degenerate_tol` as in `_cut`. Returns U, S, Vh, the relative dropped weight and the pin
    flag."""
    tot = float(be.to_numpy(be.norm(mat))) ** 2 or 1.0      # = sum(sigma^2), pre-truncation
    if _is_block(mat):
        if trunc != "svd":
            raise ValueError(f"block tensors truncate with trunc='svd' only, got {trunc!r}")
        info = {}
        U, S, Vh, disc = linalg_of(mat).truncated_svd(mat, chi, svd_tol=svd_tol,
                                                      degenerate_tol=degenerate_tol, info=info)
        return U, S, Vh, disc / tot, info["pinned"]
    whole = degenerate_tol > 0                              # `_cut` needs the whole spectrum
    rank = min(int(x) for x in mat.shape) if whole else chi
    U, S, Vh, disc = _truncate(mat, rank, trunc, be, rng, iso)
    # `_truncate` returns the ABSOLUTE dropped weight for every tnkit path and the already
    # RELATIVE one for 'cutn' (cuTensorNet's own convention) -- normalise here, once.
    disc_rel = disc if trunc == "cutn" else disc / tot
    pinned = False
    if whole or (svd_tol > 0.0 and int(S.shape[0]) > 1):
        # cap, then the svd_tol budget on the TOTAL relative dropped weight (what _truncate
        # already threw away plus what is dropped now), then whole groups
        k, pinned, disc_rel = _cut(be.to_numpy(S), chi, svd_tol, degenerate_tol, disc_rel, tot)
        if k < int(S.shape[0]):
            U, S, Vh = U[:, :k], S[:k], Vh[:k]
    return U, S, Vh, disc_rel, pinned


def _evolve_two_site(M, L, R, mpo, i, chi, dt, be, moving, prof=None,
                     trunc="svd", svd_tol=0.0, rng=None, renorm=True, meter=None, fm=None,
                     krylov_m=30, degenerate_tol=0.0):
    """Forward-evolve the 2-site block at bond (i,i+1) by exp(-i H2 dt/2), truncate,
    and back-evolve the resulting 1-site centre by exp(+i H1 dt/2). `moving` says
    which side keeps the centre ('right' -> centre ends at i+1, 'left' -> at i).

    `chi` is a CAP, not a target: the block has rank up to min(2a, 2b), so the bond GROWS as
    entanglement builds and is cut back only when it would exceed `chi`. That is the whole
    difference from `tdvp1_evolve`, which cannot grow at all.

    trunc   : truncation method, dispatched through `dmrg._truncate` -- the same code the
              statics use, so the two paths cannot drift apart.
    svd_tol : relative discarded weight allowed per bond. 0 (default) always keeps `chi`
              singular values. > 0 keeps the smallest rank whose dropped weight
              sum(sigma_dropped^2)/sum(sigma^2) stays under svd_tol, still capped at `chi`, so
              the bond dimension follows the entanglement.
    renorm  : rescale S to unit norm after truncating (default True).
              Set False when the CALLER measures the norm -- renormalising silently absorbs
              the norm lost to truncation, which is exactly the signal a truncation check wants
              to see. It also matters for a window on an infinite background, whose norm is
              fixed by the boundary environments rather than by the local Schmidt values.

    `prof` (optional dict) accumulates per-phase wall time — 'expm' (both Krylov
    exponentials), 'trunc' (SVD + truncation), 'env' (environment update) — and 'disc'
    (relative discarded weight sum(sigma_dropped^2)/sum(sigma^2)). Timing mirrors the DMRG
    profiler: `be.sync()` on both sides so GPU kernels are actually complete when clocked.
    degenerate_tol : see `TDVPOptions` and `_cut`. Returns True when it moved the cut at the
              cap below the cap and `svd_tol` did not cut lower (the bond is then full for the
              '12site' schedule), else False.
    NOTE: the discarded weight is now computed on EVERY call, profiling or
    not, because `_truncate` returns it -- one small device->host copy of the singular values
    per bond. Truncation error is not optional bookkeeping in the dynamics path; a fixed-chi
    run that silently discards weight is indistinguishable from a converged one."""
    W1, W2 = mpo[i], mpo[i + 1]
    theta = be.einsum("asc,ctb->astb", M[i], M[i + 1])
    block = _is_block(theta)

    _minfo = meter.info if meter is not None else None
    if prof is not None:
        be.sync(); _t = time.perf_counter()
    theta = _krylov(lambda x: apply_two_site(x, L[i], W1, W2, R[i + 2], be), theta,
                    -1j * dt / 2.0, be, fm, _minfo, krylov_m)
    if meter is not None:
        meter.collect()
    if prof is not None:
        be.sync(); prof["expm"] += time.perf_counter() - _t; _t = time.perf_counter()

    mat, rec = fuse(theta, ((0, 1), (2, 3)), be)            # (a s1, s2 b)
    U, S, Vh, disc_rel, pinned = _svd_cut(mat, chi, be, trunc, svd_tol, degenerate_tol, rng,
                                          "left" if moving == "right" else "right")
    if renorm:
        S = S / be.norm(S)
    if prof is not None:
        prof["disc"] += disc_rel
        be.sync(); prof["trunc"] += time.perf_counter() - _t

    def _env(fn, *args):
        if prof is None:
            return fn(*args)
        be.sync(); _te = time.perf_counter()
        out = fn(*args)
        be.sync(); prof["env"] += time.perf_counter() - _te
        return out

    def _expm(fn, *args):
        if prof is None:
            return fn(*args)
        be.sync(); _te = time.perf_counter()
        out = fn(*args)
        be.sync(); prof["expm"] += time.perf_counter() - _te
        return out

    if moving == "right":
        M[i] = split(U, rec, be, legs=(0,))
        L[i + 1] = _env(update_left_env, L[i], M[i], W1, be)
        SV = Vh.scale(S, 0) if block else S[:, None] * Vh
        C = split(SV, rec, be, legs=(1,))               # 1-site centre at i+1
        if i < len(M) - 2:                             # back-evolve (skip boundary)
            C = _expm(_krylov, lambda x: apply_one_site(x, L[i + 1], W2, R[i + 2], be), C,
                      +1j * dt / 2.0, be, fm, _minfo, krylov_m)
            if meter is not None:
                meter.collect()
        M[i + 1] = C
    else:
        M[i + 1] = split(Vh, rec, be, legs=(1,))
        R[i + 1] = _env(update_right_env, R[i + 2], M[i + 1], W2, be)
        US = U.scale(S, 1) if block else U * S
        C = split(US, rec, be, legs=(0,))             # 1-site centre at i
        if i > 0:
            C = _expm(_krylov, lambda x: apply_one_site(x, L[i], W1, R[i + 1], be), C,
                      +1j * dt / 2.0, be, fm, _minfo, krylov_m)
            if meter is not None:
                meter.collect()
        M[i] = C
    return pinned


def tdvp2_evolve(mps: MPSLike, mpo: Any, chi: int, dt: float, n_steps: int,
                 options: Optional[Union[TDVPOptions, Dict[str, Any]]] = None,
                 **legacy_kwargs: Any) -> Tuple[MPS, List[Tuple[float, Any]]]:
    """Two-site TDVP: real-time evolution whose bonds grow with the entanglement.

    Parameters
    ----------
    mps : sequence of tensors
        The initial state, rank-3 tensors ``(left, phys, right)``, sites numbered from 0. Dense
        states are cast to the backend (a real state becomes complex); block and symmetric
        states need the matching `backend`.
    mpo : sequence of tensors, or callable
        The Hamiltonian, axes ``(wl, wr, out, in)``, any range (not only nearest
        neighbours). A callable ``t -> MPO`` gives a time-dependent H, evaluated at the
        midpoint of each step (second order, as the sweep), at the cost of one extra
        environment sweep per step.
    chi : int
        Bond cap, >= 1, unless `chi_max` is given. A bond grows (a two-site block has rank up
        to ``min(2a, 2b)`` for bonds a and b) and is cut back only past the cap or by
        `svd_tol`. On symmetric tensors it counts multiplets.
    dt : float
        Time step, > 0. Each step applies ``exp(-i H dt)`` (hbar = 1), in the time units of
        `mpo`'s energies.
    n_steps : int
        Number of steps, >= 0; the total time is ``dt * n_steps``.
    options : TDVPOptions or dict, optional
        Settings; see `TDVPOptions` for every field, its default and its range.
    **legacy_kwargs
        Any field of `TDVPOptions` that this integrator takes, passed flat. A keyword
        overrides the same field of `options`; a field of the other integrator raises.

    Returns
    -------
    mps : MPS
        The evolved state on the backend used, with its orthogonality centre at site 0
        (``mps.center == 0``) and every other site right-canonical.
    records : list of (float, object)
        ``(t, measure(state))`` at t = 0 and every `meas_every` steps; empty when `measure`
        is None.

    Notes
    -----
    Two-site TDVP on block and symmetric tensors is supported (provisional). `fast_expm`
    speeds up the Krylov exponentials only; in two-site TDVP much of the time is in the SVD.
    """
    o = _tdvp_opts("tdvp2_evolve", options, legacy_kwargs)
    tl.warn_if_deprecated(o.trunc)
    _warn_gram_tol(o)
    return _tdvp2(mps, mpo, chi, dt, n_steps, o)


def _tdvp2(mps: MPSLike, mpo: Any, chi: int, dt: float, n_steps: int,
           o: TDVPOptions) -> Tuple[MPS, List[Tuple[float, Any]]]:
    """The integrator itself. Unchanged from the flat-signature version; the options
    are unpacked into the names the body has always used."""
    backend, measure, meas_every = o.backend, o.measure, o.meas_every
    profile, step_callback = o.profile, o.step_callback
    trunc, chi_max, svd_tol, renorm, seed = o.trunc, o.chi_max, o.svd_tol, o.renorm, o.seed
    L0, RN, canonicalize, keep_device = o.L0, o.RN, o.canonicalize, o.keep_device
    adaptive_dt, fast_expm = o.adaptive_dt, o.fast_expm

    be = get_backend(backend)
    fm, cache = _resolve_fast_expm(fast_expm, adaptive_dt)
    mpo_fn = mpo if callable(mpo) else None
    mpo = mpo_fn(0.0) if mpo_fn is not None else mpo
    _check_blocks(mps, mpo, be, fast_expm, "off", trunc)
    mpo = [be.array(W) for W in mpo]
    invariants.entry(mps, mpo, L0, RN)
    N = len(mpo)
    Mraw = [be.array(m) for m in mps]
    M = right_canonicalize(Mraw, be) if canonicalize else Mraw
    cap = int(chi if chi_max is None else chi_max)
    rng = np.random.default_rng(seed + 1)
    L = [None] * (N + 1); R = [None] * (N + 1)
    L[0], R[N] = _edge_envs(M, mpo, be, L0, RN)

    def _rebuild_right_envs():
        for i in range(N - 1, 0, -1):
            R[i] = update_right_env(R[i + 1], M[i], mpo[i], be)

    _rebuild_right_envs()
    centre = 0 if (canonicalize or n_steps > 0) else None   # every sweep ends at site 0

    def snapshot():
        return M if keep_device or _is_block(M[0]) else [be.to_numpy(m) for m in M]

    do_prof = profile is not None or step_callback is not None
    if profile is None and step_callback is not None:
        profile = {}
    records = []
    if measure is not None:
        records.append((0.0, measure(snapshot())))
    if do_prof:
        profile.setdefault("steps", [])
    state = {"step": 0, "t": 0.0, "fm": fm}
    kmeter = _KrylovMeter() if do_prof else None

    def sweep(h, meter=None):
        """One full symmetric 2TDVP step of size `h` (forward + backward sweep)."""
        if meter is None and kmeter is not None:
            kmeter.reset()
            meter = kmeter
        if mpo_fn is not None:
            # Time-dependent H, frozen at the step MIDPOINT: the exponential midpoint rule is
            # 2nd order in h, matching the order of the Strang sweep, so no accuracy order is
            # lost. The right environments contain W, so they are rebuilt.
            mpo[:] = [be.array(x) for x in mpo_fn(state["t"] + 0.5 * h)]
            _rebuild_right_envs()
        pstep = {"expm": 0.0, "trunc": 0.0, "env": 0.0, "disc": 0.0} if do_prof else None
        tr = dict(trunc=trunc, svd_tol=svd_tol, rng=rng, renorm=renorm, meter=meter,
                  fm=state["fm"], krylov_m=o.krylov_m, degenerate_tol=o.degenerate_tol)
        for i in range(0, N - 1):                      # forward sweep (centre -> N-1)
            _evolve_two_site(M, L, R, mpo, i, cap, h, be, "right", pstep, **tr)
        for i in range(N - 2, -1, -1):                 # backward sweep (centre -> 0)
            _evolve_two_site(M, L, R, mpo, i, cap, h, be, "left", pstep, **tr)
        invariants.sweep(M, mpo, L0, RN)
        state["step"] += 1
        state["t"] += h
        step_dict = None
        if do_prof:
            max_chi = max(bond_states(m, 2) for m in M[:-1])
            step_dict = {"step": state["step"], "t": state["t"], "dt": h, "max_chi": max_chi,
                         "disc": pstep["disc"], "expm_s": pstep["expm"],
                         "trunc_s": pstep["trunc"], "env_s": pstep["env"],
                         "krylov_max": meter.m_max, "krylov_cap_hits": meter.cap_hits}
            profile["steps"].append(step_dict)
        return step_dict

    if adaptive_dt is not None:
        for t_now, _ in _run_adaptive(sweep, _meas_targets(dt, n_steps, meas_every),
                                      dt, adaptive_dt):
            obs_rec = None
            if measure is not None:
                obs_rec = (t_now, measure(snapshot()))
                records.append(obs_rec)
            if step_callback is not None:
                step_callback(profile["steps"][-1] if do_prof else None, obs_rec)
        return MPS(M, center=centre), records

    for step in range(n_steps):
        step_dict = _fast_step(sweep, dt, state, cache)
        obs_rec = None
        if measure is not None and (step + 1) % meas_every == 0:
            obs_rec = ((step + 1) * dt, measure(snapshot()))
            records.append(obs_rec)
        if step_callback is not None:
            step_callback(step_dict, obs_rec)
    return MPS(M, center=centre), records


# --------------------------------------------------------------------------- one-site TDVP
def _evolve_one_site_right(M, L, R, mpo, i, dt, be, meter=None, fm=None, krylov_m=30):
    """Forward-evolve site i by exp(-i H1 dt/2), split off the bond, back-evolve it by
    exp(+i H0 dt/2), and push it into site i+1. Bond dimension is UNCHANGED throughout."""
    if _is_block(M[i]):
        return _one_site_blocks(M, L, R, mpo, i, dt, be, meter, fm, krylov_m, "right")
    _mi = meter.info if meter is not None else None
    a, s, b = (int(x) for x in M[i].shape)
    mv1 = lambda x: apply_one_site(x.reshape(a, s, b), L[i], mpo[i], R[i + 1], be).reshape(-1)
    M[i] = _expm_apply(mv1, M[i].reshape(-1), -1j * dt / 2.0, be, fm, _mi, krylov_m).reshape(a, s, b)
    if meter is not None:
        meter.collect()
    if i == len(M) - 1:
        return
    Q, C = be.qr(M[i].reshape(a * s, b))             # centre moves right
    r = int(C.shape[0])
    M[i] = Q.reshape(a, s, r)
    L[i + 1] = update_left_env(L[i], M[i], mpo[i], be)
    mv0 = lambda x: apply_zero_site(x.reshape(r, b), L[i + 1], R[i + 1], be).reshape(-1)
    C = _expm_apply(mv0, C.reshape(-1), +1j * dt / 2.0, be, fm, _mi, krylov_m).reshape(r, b)
    if meter is not None:
        meter.collect()
    M[i + 1] = be.einsum("rb,bsc->rsc", C, M[i + 1])


def _evolve_one_site_left(M, L, R, mpo, i, dt, be, meter=None, fm=None, krylov_m=30):
    """Mirror of the above, sweeping leftwards."""
    if _is_block(M[i]):
        return _one_site_blocks(M, L, R, mpo, i, dt, be, meter, fm, krylov_m, "left")
    _mi = meter.info if meter is not None else None
    a, s, b = (int(x) for x in M[i].shape)
    mv1 = lambda x: apply_one_site(x.reshape(a, s, b), L[i], mpo[i], R[i + 1], be).reshape(-1)
    M[i] = _expm_apply(mv1, M[i].reshape(-1), -1j * dt / 2.0, be, fm, _mi, krylov_m).reshape(a, s, b)
    if meter is not None:
        meter.collect()
    if i == 0:
        return
    # LQ via QR of the transpose
    Qt, Ct = be.qr(be.einsum("asb->sba", M[i]).reshape(s * b, a))
    r = int(Ct.shape[0])
    M[i] = be.einsum("sbr->rsb", Qt.reshape(s, b, r))
    R[i] = update_right_env(R[i + 1], M[i], mpo[i], be)
    C = be.einsum("ra->ar", Ct)                      # (a, r)
    mv0 = lambda x: apply_zero_site(x.reshape(a, r), L[i], R[i], be).reshape(-1)
    C = _expm_apply(mv0, C.reshape(-1), +1j * dt / 2.0, be, fm, _mi, krylov_m).reshape(a, r)
    if meter is not None:
        meter.collect()
    M[i - 1] = be.einsum("csa,ar->csr", M[i - 1], C)


def _evolve_one_site_left_cut(M, L, R, mpo, i, dt, chi, be, meter=None, fm=None, krylov_m=30,
                              renorm=True, prof=None, degenerate_tol=0.0):
    """`_evolve_one_site_left` with a truncated SVD in place of the LQ: the bond to site i-1 keeps
    its `chi` largest singular values. Cuts a bond back after a subspace expansion. Returns
    whether `degenerate_tol` (see `TDVPOptions`) moved the cut below `chi`."""
    _mi = meter.info if meter is not None else None

    def step(fn, x, tau):
        out = _krylov(fn, x, tau, be, fm, _mi, krylov_m)
        if meter is not None:
            meter.collect()
        return out

    M[i] = step(lambda x: apply_one_site(x, L[i], mpo[i], R[i + 1], be), M[i], -1j * dt / 2.0)
    f, rec = fuse(M[i], ((0,), (1, 2)), be)
    U, S, Vh, disc, pinned = _svd_cut(f, chi, be, degenerate_tol=degenerate_tol)
    if renorm:
        S = S / be.norm(S)
    if prof is not None:
        prof["disc"] += disc
    US = U.scale(S, 1) if _is_block(U) else U * S
    M[i], C = split(Vh, rec, be, legs=(1,)), split(US, rec, be, legs=(0,))
    R[i] = update_right_env(R[i + 1], M[i], mpo[i], be)
    C = step(lambda x: apply_zero_site(x, L[i], R[i], be), C, +1j * dt / 2.0)
    M[i - 1] = be.einsum("csa,ar->csr", M[i - 1], C)
    return pinned


def _one_site_blocks(M, L, R, mpo, i, dt, be, meter, fm, krylov_m, direction):
    """`_evolve_one_site_right`/`_left` on block tensors: QR per block, bonds keep their
    sectors (a sector can only shrink, where a block has fewer rows than columns)."""
    _mi = meter.info if meter is not None else None

    def step(fn, x, tau):
        out = _krylov(fn, x, tau, be, fm, _mi, krylov_m)
        if meter is not None:
            meter.collect()
        return out

    M[i] = step(lambda x: apply_one_site(x, L[i], mpo[i], R[i + 1], be), M[i], -1j * dt / 2.0)
    if direction == "right":
        if i == len(M) - 1:
            return
        f, rec = fuse(M[i], ((0, 1), (2,)), be)
        Q, C = linalg_of(f).qr(f)
        M[i], C = split(Q, rec, be, legs=(0,)), split(C, rec, be, legs=(1,))
        L[i + 1] = update_left_env(L[i], M[i], mpo[i], be)
        C = step(lambda x: apply_zero_site(x, L[i + 1], R[i + 1], be), C, +1j * dt / 2.0)
        M[i + 1] = be.einsum("rb,bsc->rsc", C, M[i + 1])
    else:
        if i == 0:
            return
        f, rec = fuse(M[i], ((0,), (1, 2)), be)
        C, Q = linalg_of(f).qr(f, right=True)
        M[i], C = split(Q, rec, be, legs=(1,)), split(C, rec, be, legs=(0,))
        R[i] = update_right_env(R[i + 1], M[i], mpo[i], be)
        C = step(lambda x: apply_zero_site(x, L[i], R[i], be), C, +1j * dt / 2.0)
        M[i - 1] = be.einsum("csa,ar->csr", M[i - 1], C)


GRAPH_MODES = ("auto", "on", "off")


def _cuda_capture_available() -> bool:
    """True iff torch is importable and reports a visible CUDA device.

    This is the ONLY probe `resolve_graph_mode('auto', ...)` performs, and it exists so the
    default can be decided without importing torch at all on a backend that will never use it
    (numpy, or a torch build with no CUDA support)."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _capture_target(backend: Any) -> bool:
    """True iff `backend` -- a `get_backend` spec or a Backend object -- is a complex torch
    backend on the current CUDA device, the one target prefix-graph capture supports.

    A non-torch spec is decided without importing torch; a torch spec is resolved by
    `get_backend`, and a spec it rejects is not a target. Real dtypes are excluded: the
    capture's Krylov coefficient buffers take the backend dtype, so a real one would drop the
    imaginary part of exp(-i dt H). Block backends are excluded because block TDVP takes no step
    graphs (`_check_blocks`); the `isinstance` check also covers subclasses.

    Capture records on the current device's streams, so the device index must be None or that
    of the current CUDA device (`torch.cuda.current_device()`; 0 when no device is visible).
    `get_backend('torch-cuda')` is on the device current when it was built."""
    if isinstance(backend, str):
        if not backend.startswith("torch-"):
            return False
        try:
            backend = get_backend(backend)
        except (ImportError, RuntimeError, ValueError):
            return False
    from .block_backend import BlockBackend
    if isinstance(backend, BlockBackend):
        return False
    device = getattr(backend, "device", None)
    index = getattr(device, "index", None)
    return (bool(getattr(backend, "is_torch", False)) and getattr(device, "type", None) == "cuda"
            and (index is None or index == _current_cuda_index())
            and bool(getattr(backend, "is_complex", False)))


def _current_cuda_index() -> int:
    """The current CUDA device, or 0 when torch or a CUDA device is missing."""
    try:
        import torch
        return torch.cuda.current_device() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 -- no usable CUDA runtime: the default device
        return 0


def _backend_label(backend: Any) -> Any:
    """A spec or Backend object as its `get_backend` spec, for messages."""
    label = getattr(backend, "name", backend)
    device = getattr(backend, "device", None)
    if isinstance(label, str) and isinstance(getattr(device, "index", None), int):
        label = label.replace(f"-{device.type}", f"-{device}", 1)   # the name omits the index
    from .block_backend import BlockBackend
    return f"BlockBackend({label})" if isinstance(backend, BlockBackend) else label


def resolve_graph_mode(graph: str, backend: Any, cuda_available: Optional[bool] = None) -> str:
    """Resolve a `graph` request ('auto' | 'on' | 'off') against a backend spec or object.

    'off'  always resolves to 'eager' -- bit-identical to every version before prefix-graph capture
           existed, regardless of backend.
    'on'   always resolves to 'capture'. Since capture needs a complex torch backend on the
           current CUDA device (`_capture_target`) with a visible CUDA device, 'on' against
           anything else RAISES rather than falling back -- an explicit request that silently
           ran eager instead would hide a configuration bug.
    'auto' (the default) resolves to 'capture' only on such a backend and with a CUDA device
           visible; every other backend (numpy, torch-cpu, torch-mps, a real dtype, a block
           backend, another CUDA device, or torch-cuda with no device reachable) resolves to
           'eager'. Capture replays the eager arithmetic bit for bit, so 'auto' changes only
           the speed, and only where capture is available.

    The CUDA probe imports torch, so it runs only for a capture target. `cuda_available`
    overrides the probe -- how the CPU test suite exercises both halves of the 'auto' rule
    without a GPU.

    Unlike the fixed-schedule graph (`cudagraph.py`, `lanczos_expm_fixed`), prefix-graph capture
    never reads `expm_where`: its small exponential always runs on the host with
    `scipy.linalg.expm` (see `tnkit.prefix_graph_impl`).
    """
    if graph not in GRAPH_MODES:
        raise ValueError(f"graph must be one of {GRAPH_MODES}, got {graph!r}")
    if graph == "off":
        return "eager"
    target = _capture_target(backend)
    avail = cuda_available
    if avail is None and target:                       # the probe imports torch: only here
        avail = _cuda_capture_available()
    supported = target and bool(avail)
    if graph == "on":
        if not supported:
            seen = f", cuda_available={avail}" if target else ""
            raise ValueError(
                f"graph='on' requires a complex torch backend on the current CUDA device "
                f"(backend='torch-cuda') with a visible CUDA device; got "
                f"backend={_backend_label(backend)!r}{seen}. Use graph='auto' to fall back "
                f"to the eager path automatically on this backend, or graph='off' to disable "
                f"capture explicitly.")
        return "capture"
    return "capture" if supported else "eager"          # graph == "auto"


_REPLAY_FAILED = object()   # distinct from a prefix graph's None ("Krylov space too short")


def _replay(gc, g, sig, direction, *args, **kw):
    """`g.run(*args, **kw)`, or `_REPLAY_FAILED` after the cache has recorded the failure.

    A cache without a `replay_failed` handler gets the exception back unchanged.
    """
    try:
        return g.run(*args, **kw)
    except Exception as exc:  # noqa: BLE001 -- the step is retried eagerly
        handler = getattr(gc, "replay_failed", None)
        if handler is None:
            raise
        handler(sig, direction, exc)
        return _REPLAY_FAILED


def _graphed_site_step(M, L, R, mpo, i, be, gc, direction):
    """Replay a captured site-step if one fits, else report False and let the caller run
    the ordinary eager path.

    The two `return False` guards are shape guards, not optimisations. A captured graph
    records ONE set of shapes, and the rank of the QR that moves the orthogonality centre
    is `min(a*s, b)` rightward and `min(s*b, a)` leftward. In the saturated-chi bulk that
    equals the bond dimension and the step is shape-preserving, which is the whole reason
    1TDVP is capturable at all; on the chain's bond-dimension ramp it is not, and a graph
    recorded there would be recorded at the wrong shape. Sites that fail the guard fall
    back to eager and are COUNTED, so the replayed fraction is reported rather than
    assumed.
    """
    N = len(M)
    a, s, b = (int(x) for x in M[i].shape)
    Dl, Dr = int(mpo[i].shape[0]), int(mpo[i].shape[1])
    if direction == "right":
        if i == N - 1 or a * s < b:
            return False
        c = int(M[i + 1].shape[2])
        sig = (a, s, b, c, Dl, Dr)
        g = gc.get(sig, "right")
        if g is None:
            return False
        out = _replay(gc, g, sig, "right", M[i], M[i + 1], L[i], R[i + 1], mpo[i])
        if out is _REPLAY_FAILED:
            return False
        M[i], L[i + 1], M[i + 1] = out
    else:
        if i == 0 or s * b < a:
            return False
        c = int(M[i - 1].shape[0])
        sig = (a, s, b, c, Dl, Dr)
        g = gc.get(sig, "left")
        if g is None:
            return False
        out = _replay(gc, g, sig, "left", M[i], M[i - 1], L[i], R[i + 1], mpo[i])
        if out is _REPLAY_FAILED:
            return False
        M[i], R[i], M[i - 1] = out
    gc.note(sig, direction, True)
    return True


def _graphed_site_step_prefix(M, L, R, mpo, i, be, gc, direction):
    """Replay an overshoot-and-truncate site-step if one fits, else report False.

    Identical shape guards to `_graphed_site_step` -- a capture records ONE set of shapes and
    the QR rank is `min(a*s, b)` rightward, `min(s*b, a)` leftward, so sites on the
    bond-dimension ramp fall back to eager and are counted.

    ONE EXTRA WAY TO RETURN FALSE, and it is the design's escape hatch rather than a guard:
    `g.run` returns None when eager's own stopping rule would have wanted a longer Krylov
    space than `m_max` provides. That step is then run eagerly, at eager's price and with
    eager's bits. It is counted in `stats['short']` and a nonzero count means m_max is too
    low, not that the result is compromised.
    """
    N = len(M)
    a, s, b = (int(x) for x in M[i].shape)
    Dl, Dr = int(mpo[i].shape[0]), int(mpo[i].shape[1])
    if direction == "right":
        if i == N - 1 or a * s < b:
            return False
        c = int(M[i + 1].shape[2])
        sig = (a, s, b, c, Dl, Dr)
        g = gc.get(sig, "right")
        if g is None:
            return False
        out = _replay(gc, g, sig, "right", M[i], M[i + 1], L[i], R[i + 1], mpo[i],
                      tol=gc.tol, adaptive=gc.adaptive, stats=gc.stats)
        if out is _REPLAY_FAILED:
            return False
        if out is None:
            gc.stats["short"] += 1
            return False
        M[i], L[i + 1], M[i + 1] = out
    else:
        if i == 0 or s * b < a:
            return False
        c = int(M[i - 1].shape[0])
        sig = (a, s, b, c, Dl, Dr)
        g = gc.get(sig, "left")
        if g is None:
            return False
        out = _replay(gc, g, sig, "left", M[i], M[i - 1], L[i], R[i + 1], mpo[i],
                      tol=gc.tol, adaptive=gc.adaptive, stats=gc.stats)
        if out is _REPLAY_FAILED:
            return False
        if out is None:
            gc.stats["short"] += 1
            return False
        M[i], R[i], M[i - 1] = out
    gc.note(sig, direction, True)
    return True


def tdvp1_evolve(mps: MPSLike, mpo: Any, dt: float, n_steps: int,
                 options: Optional[Union[TDVPOptions, Dict[str, Any]]] = None,
                 **legacy_kwargs: Any) -> Tuple[MPS, List[Tuple[float, Any]]]:
    """One-site TDVP: real-time evolution at fixed bond dimensions.

    No SVD and no truncation, so the integrator is symplectic and conserves the energy and the
    norm up to the Krylov tolerance, but it cannot grow entanglement: the state stays on the
    manifold of its initial bond dimensions and pays a projection error instead of a truncation
    error. A common schedule is two-site TDVP until the bonds reach their cap, then one-site
    TDVP (`TDVPEvolver` with '12site' does this per bond).

    Parameters
    ----------
    mps : sequence of tensors
        The initial state, rank-3 tensors ``(left, phys, right)``, sites numbered from 0. Its
        bond dimensions are kept throughout.
    mpo : sequence of tensors, or callable
        The Hamiltonian, axes ``(wl, wr, out, in)``, any range; a callable ``t -> MPO`` is
        evaluated at the midpoint of each step.
    dt : float
        Time step, > 0. Each step applies ``exp(-i H dt)`` (hbar = 1), in the time units of
        `mpo`'s energies.
    n_steps : int
        Number of steps, >= 0; the total time is ``dt * n_steps``.
    options : TDVPOptions or dict, optional
        Settings; see `TDVPOptions` for every field, its default and its range.
    **legacy_kwargs
        Any field of `TDVPOptions` that this integrator takes, passed flat. A keyword
        overrides the same field of `options`; a field of the other integrator raises.

    Returns
    -------
    mps : MPS
        The evolved state on the backend used, with its orthogonality centre at site 0
        (``mps.center == 0``) and every other site right-canonical.
    records : list of (float, object)
        ``(t, measure(state))`` at t = 0 and every `meas_every` steps; empty when `measure`
        is None.

    Notes
    -----
    One-site TDVP runs on block and symmetric tensors (provisional); its bond sectors are those
    of the initial state. `fast_expm`, `cuda_graph`, `prefix_graph` and `graph` are GPU options
    (see `TDVPOptions`).
    """
    return _tdvp1(mps, mpo, dt, n_steps, _tdvp_opts("tdvp1_evolve", options, legacy_kwargs))


def _tdvp1(mps: MPSLike, mpo: Any, dt: float, n_steps: int,
           o: TDVPOptions) -> Tuple[MPS, List[Tuple[float, Any]]]:
    """The integrator itself. Unchanged from the flat-signature version; the options
    are unpacked into the names the body has always used."""
    backend, measure, meas_every = o.backend, o.measure, o.meas_every
    L0, RN, canonicalize, keep_device = o.L0, o.RN, o.canonicalize, o.keep_device
    adaptive_dt, fast_expm = o.adaptive_dt, o.fast_expm
    cuda_graph, prefix_graph = o.cuda_graph, o.prefix_graph
    step_callback = o.step_callback
    if step_callback is not None and adaptive_dt is not None:
        raise ValueError("tdvp1_evolve: step_callback is not supported with adaptive_dt")

    be = get_backend(backend)
    N = len(mps)
    fm, cache = _resolve_fast_expm(fast_expm, adaptive_dt)
    kmeter = _KrylovMeter() if step_callback is not None else None
    krylov_m = o.krylov_m

    # GRAPH DEFAULT (prefix-graph capture). Only when the caller has not already handed in a live
    # cache -- `cuda_graph`/`prefix_graph` given explicitly always wins, `graph` is not even
    # consulted.
    # 'auto' (the default) silently declines to build a cache when it would conflict with an
    # explicit `fast_expm`/`adaptive_dt` choice (those already say what solver to use); an
    # explicit `graph='on'` raises instead, because a silent decline would hide the conflict.
    if cuda_graph is None and prefix_graph is None and o.graph != "off":
        mode = resolve_graph_mode(o.graph, be)
        if mode == "capture" and fm is not None and fm.get("m"):
            if o.graph == "on":
                raise ValueError(
                    "graph='on' conflicts with fast_expm: prefix-graph capture replays the "
                    "ADAPTIVE solver's own stopping rule and requires fast_expm=None. Pass "
                    "graph='off' to use the frozen-schedule fast path instead.")
            mode = "eager"
        if mode == "capture" and krylov_m != 30:
            if o.graph == "on":
                raise ValueError(
                    "graph='on' conflicts with krylov_m: prefix-graph capture replays captured "
                    "solves of `graph_m_max` steps, so a Krylov cap set here would apply only "
                    "to the eager fallbacks. Pass graph='off' to use `krylov_m`.")
            mode = "eager"
        if mode == "capture" and adaptive_dt is not None:
            if o.graph == "on":
                raise ValueError(
                    "graph='on' conflicts with adaptive_dt: prefix-graph capture replays the "
                    "ADAPTIVE solver's stopping rule, but the step-size controller needs a "
                    "per-step error estimate that only a metered eager sweep produces. Pass "
                    "graph='off' to use adaptive_dt instead.")
            mode = "eager"
        if mode == "capture":
            import torch
            from .prefix_graph_impl import PrefixSiteStepGraphCache
            prefix_graph = PrefixSiteStepGraphCache(be, torch, dt, o.graph_m_max,
                                                    max_graphs=o.graph_cache_cap)
            # Built fresh for THIS call. A caller that re-enters `tdvp1_evolve` many times over
            # one run (a measurement loop, say) should instead build ONE cache up front and
            # pass it as `prefix_graph=` explicitly, so capture is paid once, not once per
            # re-entry.

    mpo_fn = mpo if callable(mpo) else None
    mpo = mpo_fn(0.0) if mpo_fn is not None else mpo
    _check_blocks(mps, mpo, be, fast_expm,
                  "off" if (cuda_graph, prefix_graph) == (None, None) else "on")
    mpo = [be.array(W) for W in mpo]
    invariants.entry(mps, mpo, L0, RN)
    Mraw = [be.array(m) for m in mps]
    # A WINDOW on an infinite background (AL...AL AC AR...AR) has its centre in the middle, not at
    # site 0; one canonicalize=True pass fixes that and leaves the boundary bonds that meet the
    # fixed environments L0/RN untouched. See tnkit.window.
    M = right_canonicalize(Mraw, be) if canonicalize else Mraw
    L, R = [None] * (N + 1), [None] * (N + 1)
    L[0], R[N] = _edge_envs(M, mpo, be, L0, RN)

    def _rebuild_right_envs():
        for i in range(N - 1, 0, -1):
            R[i] = update_right_env(R[i + 1], M[i], mpo[i], be)

    _rebuild_right_envs()
    centre = 0 if (canonicalize or n_steps > 0) else None   # every sweep ends at site 0

    def snapshot():
        return M if keep_device or _is_block(M[0]) else [be.to_numpy(m) for m in M]

    records = []
    if measure is not None:
        records.append((0.0, measure(snapshot())))
    state = {"t": 0.0, "fm": fm, "gc": None}
    if cuda_graph is not None:
        # CUDA-GRAPH MODE. `cuda_graph` is a live SiteStepGraphCache, built by the caller
        # and carried ACROSS re-entries of this function -- a driver that re-enters once
        # per measurement interval would otherwise pay the capture
        # cost dozens of times for graphs that are identical.
        if fm is None or not fm.get("m"):
            raise ValueError(
                "cuda_graph requires a FROZEN Krylov schedule (fast_expm=<int> or a "
                "calibrated dict). Capture records a fixed kernel sequence, and the "
                "adaptive solver's stop test is a host branch on a device scalar -- the "
                "one thing a capture cannot contain. See tnkit/cudagraph.py.")
        if cache is not None and cache.get("recal", FAST_EXPM_RECAL):
            raise ValueError(
                "cuda_graph requires recal=0. The re-measurement sweep runs the adaptive "
                "solver, so every recal-th step would have a different kernel sequence. Fix m "
                "explicitly instead.")
        state["gc"] = cuda_graph
    state["pgc"] = None
    if prefix_graph is not None:
        # Overshoot-and-truncate mode. Unlike `cuda_graph`, this path captures the adaptive
        # solver's own loop and applies its stopping rule on the host after replay, so it
        # needs the adaptive solver: a fixed schedule would change the arithmetic.
        if cuda_graph is not None:
            raise ValueError("cuda_graph and prefix_graph are alternatives, not a stack.")
        if fm is not None and fm.get("m"):
            raise ValueError(
                "prefix_graph requires the ADAPTIVE solver (fast_expm=None): a frozen "
                "schedule is itself an arithmetic perturbation relative to the adaptive "
                "path this mode is meant to capture; see tnkit/prefix_graph_impl.py.")
        if adaptive_dt is not None:
            raise ValueError("prefix_graph is incompatible with adaptive_dt (the controller "
                             "needs a metered eager sweep).")
        if krylov_m != 30:
            raise ValueError("prefix_graph replays solves of its own Krylov length and ignores "
                             "krylov_m; leave krylov_m at its default.")
        state["pgc"] = prefix_graph

    def sweep(h, meter=None):
        if mpo_fn is not None:                       # time-dependent H, midpoint-frozen
            mpo[:] = [be.array(x) for x in mpo_fn(state["t"] + 0.5 * h)]
            _rebuild_right_envs()
        state["t"] += h
        gc = state["gc"] if meter is None else None    # a metered sweep must run eager
        pgc = state["pgc"] if meter is None else None
        if meter is None and kmeter is not None:       # callback meter: counts eager solves only
            kmeter.reset()
            meter = kmeter
        for i in range(0, N):
            if gc is not None and _graphed_site_step(M, L, R, mpo, i, be, gc, "right"):
                continue
            if pgc is not None and _graphed_site_step_prefix(M, L, R, mpo, i, be, pgc,
                                                             "right"):
                continue
            if gc is not None:
                gc.note(replayed=False)
            if pgc is not None:
                pgc.note(replayed=False)
            _evolve_one_site_right(M, L, R, mpo, i, h, be, meter, state["fm"], krylov_m)
        for i in range(N - 1, -1, -1):
            if gc is not None and _graphed_site_step(M, L, R, mpo, i, be, gc, "left"):
                continue
            if pgc is not None and _graphed_site_step_prefix(M, L, R, mpo, i, be, pgc,
                                                             "left"):
                continue
            if gc is not None:
                gc.note(replayed=False)
            if pgc is not None:
                pgc.note(replayed=False)
            _evolve_one_site_left(M, L, R, mpo, i, h, be, meter, state["fm"], krylov_m)
        invariants.sweep(M, mpo, L0, RN)
        return meter

    if adaptive_dt is not None:
        for t_now, _ in _run_adaptive(sweep, _meas_targets(dt, n_steps, meas_every),
                                      dt, adaptive_dt):
            if measure is not None:
                records.append((t_now, measure(snapshot())))
        return MPS(M, center=centre), records

    for step in range(n_steps):
        meter = _fast_step(sweep, dt, state, cache)
        obs_rec = None
        if measure is not None and (step + 1) % meas_every == 0:
            obs_rec = ((step + 1) * dt, measure(snapshot()))
            records.append(obs_rec)
        if step_callback is not None:
            step_callback({"step": step + 1, "t": state["t"], "dt": dt,
                           "krylov_max": meter.m_max,
                           "krylov_cap_hits": meter.cap_hits}, obs_rec)
    return MPS(M, center=centre), records
