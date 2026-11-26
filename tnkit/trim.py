"""Adaptive active-window trimming for one-site TDVP of wave packets in a window (vacuum skip).

A window on an infinite background (`tnkit.window`) can hold localised wave packets in an
otherwise stationary vacuum: only the region around each packet, and whatever its light cone
has reached, differs from the background tensors AL and AR. The vacuum is stationary for the
windowed problem (which is what makes the fixed boundary environments FL/FR valid), so a
one-site TDVP step on a site that is still the vacuum tensor changes nothing but costs a full
Krylov solve.

Mechanism. Keep an active half-open interval [lo, hi) of the N_s sites. Only sites inside it
are passed to `tdvp1_evolve`; sites outside are left untouched, so they stay whatever they
were, which is correct provided they started as exact copies of AL (left of lo) or AR (right
of hi), the precondition `tnkit.window.grow_window` also relies on.

Two different environments are needed at the cut:

  1. The MPO (energy) environment that TDVP needs as `L0`/`RN` for the active slice. FL/FR
     themselves are right only when the frozen prefix and suffix carry the vacuum's own bond
     dimension up to the cut: true for a plain `tnkit.window.build_window` state, but not for
     an ansatz that represents the whole window, vacuum sites included, at a larger uniform
     bond dimension (an upper-triangular three-block automaton over the chain); there
     ``L0=FL`` does not even have the right shape. `_propagate_left`/`_propagate_right`
     contract FL/FR through the frozen prefix and suffix with the energy recursion
     (`tnkit._dmrg.update_left_env`, `update_right_env`), which gives the environment the
     untrimmed sweep would have built on its way to the cut.

  2. The norm environment (bra-ket, no MPO leg) that the occupation check needs to read the
     right `<op_j>` near the cut. `tnkit.window.window_expectations` closes a window with an
     identity at the true edges (0, N_s), which is right there because the semi-infinite
     background is canonical. An identity at the active slice's own edges (lo, hi) is not:
     it overweights sites near the cut, and on an ansatz with a larger bond dimension it makes
     `detect_occupied` see spurious deviations just outside the window, which then grows to the
     whole chain. `_prefix_norm_env`/`_suffix_norm_env` propagate an identity from the true
     edges through the frozen prefix and suffix with the norm recursion, and `detect_occupied`
     closes the slice with that.

Both propagations are O(lo) and O(N_s - hi) contractions with no Krylov solve, and
`evolve_trimmed` caches all four (L0/RN energy environments, Lnorm/Rnorm norm environments)
across calls, recomputing a side only when that side of the window moves.

The active slice must be re-canonicalised every time its boundary moves, not only on the first
chunk. A slice cut from a larger mixed-canonical state is not itself canonical relative to
L0/RN (only the whole window, in the gauge `tnkit.window` prepares, is), and sites newly taken
in when the window widens are in the same position. Without the re-canonicalisation the energy
drifts once the window has grown several times.

Margin. A Gaussian envelope of width sigma has weight exp(-x^2/(4 sigma^2)) at distance x from
its centre (the packet convention assumed here): 3e-4 of peak at x = 4 sigma, 1e-8 at
x = 6 sigma. Padding the occupied region by about 6 sigma leaves only an exponentially small
tail outside the active window, and a few correlation lengths on top cover the state's own
equal-time correlations. `default_trim_margin` turns this into a site count.

Trimming is opt-in. A frozen site keeps the tensor it started with, not the one the untrimmed
sweep would have carried it to, so trimming changes the energy by an amount bounded by
`threshold`. The change is reproducible (the same threshold, margin and `every` give the same
result) but there is no correction that separates it from the ordinary drift of the untrimmed
evolution; it can only be bounded in advance.

The active window only grows. A packet that recedes toward vacuum is detected, but the window
is not shrunk: a larger window costs time, never correctness, and a shrink could clip a slow
tail.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from .backend import get_backend
from ._dmrg import update_left_env, update_right_env
from .models import SZ
from .tdvp import tdvp1_evolve

__all__ = ["default_trim_margin", "detect_occupied", "evolve_trimmed"]


# --------------------------------------------------------------------------- energy environment
def _propagate_left(FL: Any, M: List[Any], mpo: List[Any], be: Any, lo: int) -> Any:
    """FL contracted through the frozen prefix M[0:lo] with the ENERGY recursion -- the
    environment `tdvp1_evolve` needs as `L0` for the active slice. O(lo) cheap contractions."""
    L = FL
    for i in range(lo):
        L = update_left_env(L, M[i], mpo[i], be)
    return L


def _propagate_right(FR: Any, M: List[Any], mpo: List[Any], be: Any, hi: int) -> Any:
    """Mirror of `_propagate_left`, seeded at FR and walked backward through M[hi:N]."""
    R = FR
    for i in range(len(M) - 1, hi - 1, -1):
        R = update_right_env(R, M[i], mpo[i], be)
    return R


# --------------------------------------------------------------------------- norm environment
def _prefix_norm_env(M: List[Any], be: Any, lo: int) -> np.ndarray:
    """Identity, seeded at the TRUE left edge (site 0), contracted through M[0:lo] with the
    NORM recursion (bra-ket only, no MPO leg) -- the closure `detect_occupied` needs at `lo`.
    Mirrors `tnkit.window.window_expectations`'s own `Lenv` recursion. O(lo) cheap
    contractions.
    """
    D = int(np.asarray(be.to_numpy(M[0])).shape[0])
    L = np.eye(D, dtype=np.complex128)
    for i in range(lo):
        m = np.asarray(be.to_numpy(M[i]), dtype=np.complex128)
        L = np.einsum("ab,asc,bsd->cd", L, m.conj(), m, optimize=True)
    return L


def _suffix_norm_env(M: List[Any], be: Any, hi: int) -> np.ndarray:
    """Mirror of `_prefix_norm_env`, seeded at the TRUE right edge (site N-1)."""
    N = len(M)
    D = int(np.asarray(be.to_numpy(M[-1])).shape[2])
    R = np.eye(D, dtype=np.complex128)
    for i in range(N - 1, hi - 1, -1):
        m = np.asarray(be.to_numpy(M[i]), dtype=np.complex128)
        R = np.einsum("asc,bsd,cd->ab", m.conj(), m, R, optimize=True)
    return R


def _scoped_expectations(M: List[Any], be: Any, lo: int, hi: int, op: np.ndarray,
                         Lstart: np.ndarray, Rstart: np.ndarray) -> np.ndarray:
    """`<op_j>` for j in [lo, hi), closing the slice with `Lstart`/`Rstart` (already-propagated
    norm environments from the true edges) instead of a fresh identity at lo/hi.

    Line for line `tnkit.window.window_expectations`'s own Lenv/Renv/contraction, generalised
    to accept a non-identity start -- see that function's docstring for why identity is the
    right closure at a TRUE edge, and this module's docstring for why it is the wrong one at an
    interior cut.
    """
    n = hi - lo
    Mh = [np.asarray(be.to_numpy(m), dtype=np.complex128) for m in M[lo:hi]]

    Lenv = [None] * (n + 1)
    Lenv[0] = Lstart
    for i in range(n):
        Lenv[i + 1] = np.einsum("ab,asc,bsd->cd", Lenv[i], Mh[i].conj(), Mh[i], optimize=True)

    Renv = [None] * (n + 1)
    Renv[n] = Rstart
    for i in range(n - 1, -1, -1):
        Renv[i] = np.einsum("asc,bsd,cd->ab", Mh[i].conj(), Mh[i], Renv[i + 1], optimize=True)

    out = np.empty(n)
    for j in range(n):
        num = np.einsum("ab,asc,st,btd,cd->", Lenv[j], Mh[j].conj(), op, Mh[j], Renv[j + 1],
                        optimize=True)
        den = np.einsum("ab,asc,bsd,cd->", Lenv[j], Mh[j].conj(), Mh[j], Renv[j + 1],
                        optimize=True)
        out[j] = float(np.real(num / den))
    return out


def default_trim_margin(sigma: float, corr_len: float = 0.0, n_sigma: float = 6.0,
                        n_xi: float = 4.0) -> int:
    """Site-count margin: `n_sigma` packet widths plus `n_xi` correlation lengths.

    `n_sigma=6.0` is the Gaussian-tail argument: exp(-x^2/(4 sigma^2)) is 3e-4 of peak at
    x=4*sigma and 1e-8 at x=6*sigma, so 6 sigma is the point past which the packet's own
    envelope no longer carries a non-negligible tail. `n_xi=4.0` is a few correlation lengths
    of headroom for the state's equal-time correlations, which extend somewhat past the
    envelope itself; `corr_len` defaults to 0, so a caller without `xi` still gets
    a usable (if less generous) margin from the packet width alone.
    """
    return int(math.ceil(n_sigma * float(sigma) + n_xi * float(corr_len)))


def detect_occupied(M: List[Any], be: Any, lo: int, hi: int, threshold: float,
                    background: float, op: Any = SZ,
                    Lnorm: Optional[np.ndarray] = None,
                    Rnorm: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """Tightest (occ_lo, occ_hi) subrange of [lo, hi) whose local `<op>` deviates from
    `background` by more than `threshold`.

    `op` (default `tnkit.models.SZ`) is evaluated at every site of [lo, hi) via
    `_scoped_expectations`, closed with `Lnorm`/`Rnorm` -- the norm environment propagated in
    from the TRUE edges through whatever is outside [lo, hi) (see `_prefix_norm_env` /
    `_suffix_norm_env`). Omitting them defaults to a fresh identity at lo/hi, which is only
    correct when lo == 0 and hi == len(M) (or the frozen exterior happens to be exactly
    isometric there, as it is for a plain `tnkit.window.build_window` state) -- `evolve_trimmed`
    always supplies the propagated ones for an interior cut. `background` is `<op>` in the pure
    vacuum, supplied by the caller.

    Returns `(c, c)` for some `lo <= c <= hi` (an empty, zero-width occupied range) if no site
    in [lo, hi) deviates -- the caller should then leave the window as it is.
    """
    if hi <= lo:
        return lo, lo
    if Lnorm is None:
        D = int(np.asarray(be.to_numpy(M[lo])).shape[0])
        Lnorm = np.eye(D, dtype=np.complex128)
    if Rnorm is None:
        D = int(np.asarray(be.to_numpy(M[hi - 1])).shape[2])
        Rnorm = np.eye(D, dtype=np.complex128)
    vals = _scoped_expectations(M, be, lo, hi, np.asarray(op, dtype=np.complex128),
                                Lnorm, Rnorm)
    dev = np.abs(vals - float(background))
    occ = np.flatnonzero(dev > threshold)
    if occ.size == 0:
        c = (lo + hi) // 2
        return c, c
    return lo + int(occ.min()), lo + int(occ.max()) + 1


def evolve_trimmed(M: List[Any], mpo: List[Any], FL: Any, FR: Any, dt: float, n_steps: int,
                   background: float, backend: str = "numpy",
                   lo: Optional[int] = None, hi: Optional[int] = None,
                   margin: int = 0, threshold: float = 1e-6, every: int = 1, op: Any = SZ,
                   canonicalize: bool = True, keep_device: bool = False,
                   measure=None, meas_every: int = 1,
                   **tdvp_kwargs: Any) -> Tuple[List[Any], List[Tuple[float, Any]],
                                               List[Tuple[float, int, int]]]:
    """1TDVP evolution of a fixed-boundary window, skipping sites still at their vacuum value.

    `background` is `<op>` in the pure vacuum (see `detect_occupied`) -- measure it once, e.g.
    `tnkit.window.window_expectations(psi0, op)[0]` at a site known to be far from any packet,
    and pass the same number on every call.

    `M` is mutated in place (sites outside the active window are never touched; sites inside
    are overwritten with `tdvp1_evolve`'s result) and also returned, matching `tdvp1_evolve`'s
    own contract. `lo`/`hi` default to the full range (equivalent to no trimming on the very
    first call, since nothing has been detected as vacuum yet); a caller that already knows
    the occupied region (e.g. from `detect_occupied` on the initial state) should pass it in
    directly rather than pay a redundant first scan.

    `every` re-detects the active window every `every` steps rather than every step, trading a
    little extra margin consumption for fewer (cheap) detection scans; it does not change the
    `tdvp1_evolve` calls themselves, which always evolve the CURRENT active slice for the full
    interval between detections. `**tdvp_kwargs` is forwarded to `tdvp1_evolve` verbatim
    (`fast_expm`, `cuda_graph`, `prefix_graph`, `graph`, ... ), so trimming composes with every
    other 1TDVP option including the CUDA-graph default.

    `canonicalize` governs the trivial `lo == 0, hi == len(M)` case only, on this call's FIRST
    chunk, matching `tdvp1_evolve`'s own `first`-only convention -- a caller that re-enters
    `evolve_trimmed` itself, once per measurement interval, should pass `canonicalize=True`
    only on its own first call. For any INTERIOR window (lo > 0 or hi < len(M)) this function
    forces re-canonicalisation regardless of the flag, both on the first chunk it runs against
    that window and again every time the window widens -- see the module docstring's third rule. `keep_device` is forwarded to every
    internal `tdvp1_evolve` call.

    Returns `(M, records, sizes)`: `records` is `tdvp1_evolve`'s per-`meas_every` measurement
    list (only `M` is passed to `measure`, the FULL state, so a caller's observable does not
    need to know about the trim); `sizes` is `[(t, lo, hi), ...]`, the active-window history,
    which is what a saving estimate or the adaptivity test reads.
    """
    be = get_backend(backend)
    N = len(M)
    if lo is None:
        lo = 0
    if hi is None:
        hi = N
    if not (0 <= lo <= hi <= N):
        raise ValueError(f"trim window [{lo}, {hi}) is not within [0, {N})")

    records: List[Tuple[float, Any]] = []
    sizes: List[Tuple[float, int, int]] = [(0.0, lo, hi)]
    if measure is not None:
        records.append((0.0, measure(M)))

    # Both environments the active slice needs at its current boundary -- see the module
    # docstring for why these are two DIFFERENT recursions (energy vs. norm) and why neither
    # may be FL/FR (or a fresh identity) verbatim once lo/hi have moved off the true edges.
    # Built once here and again only when a side of the window actually moves, since the
    # frozen prefix/suffix they are built from is by construction unchanged in between.
    L0_cur = _propagate_left(FL, M, mpo, be, lo)
    RN_cur = _propagate_right(FR, M, mpo, be, hi)
    Lnorm_cur = _prefix_norm_env(M, be, lo)
    Rnorm_cur = _suffix_norm_env(M, be, hi)

    step = 0
    t = 0.0
    # The FIRST chunk this call ever runs against [lo, hi) must be canonicalised whenever that
    # range is not the whole state: a slice cut out of a larger mixed-canonical window is not,
    # in general, itself properly canonical relative to L0_cur/RN_cur (only the WHOLE window,
    # in the gauge `tnkit.window` already prepares it in, is guaranteed that). Only the trivial
    # lo == 0, hi == N case defers to the caller's own `canonicalize`, matching the ordinary
    # `tdvp1_evolve` "first-only" convention a caller re-entering across many calls relies on.
    first = bool(canonicalize) or lo > 0 or hi < N
    while step < n_steps:
        k = min(every, n_steps - step)
        sub, _ = tdvp1_evolve(M[lo:hi], mpo[lo:hi], dt, k, backend=backend, L0=L0_cur,
                              RN=RN_cur, canonicalize=first, keep_device=keep_device,
                              **tdvp_kwargs)
        M[lo:hi] = sub
        first = False
        step += k
        t += k * dt

        occ_lo, occ_hi = detect_occupied(M, be, lo, hi, threshold, background, op=op,
                                         Lnorm=Lnorm_cur, Rnorm=Rnorm_cur)
        if occ_lo < occ_hi:                          # something occupied was found at all
            new_lo = min(lo, max(0, occ_lo - margin))
            new_hi = max(hi, min(N, occ_hi + margin))
            if new_lo != lo:
                L0_cur = _propagate_left(FL, M, mpo, be, new_lo)
                Lnorm_cur = _prefix_norm_env(M, be, new_lo)
                first = True                          # the WIDER slice needs fresh canon. too
            if new_hi != hi:
                RN_cur = _propagate_right(FR, M, mpo, be, new_hi)
                Rnorm_cur = _suffix_norm_env(M, be, new_hi)
                first = True
            lo, hi = new_lo, new_hi
        sizes.append((t, lo, hi))

        if measure is not None and step % meas_every == 0:
            records.append((t, measure(M)))

    return M, records, sizes
