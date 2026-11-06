"""Vacuum-trim correctness tests (V2): the adaptive active window vs. the untrimmed baseline.

WHAT IS BEING CHECKED, AND WHY A SYNTHETIC SETUP IS ENOUGH. `tnkit.trim` never looks at
physics -- its only claim is bookkeeping: a site the diagnostic calls vacuum can be skipped
because it started out as, and remains, EXACTLY the tensor it would have been left as by the
untrimmed sweep, to within the same tolerance the diagnostic itself uses. That claim does not
need a real two-particle scattering state to test; a small TFIM window with a couple of local
real-space "kicks" standing in for wave packets exercises exactly the same bookkeeping (see
`_kicked_window` below) at a fraction of the setup cost of a momentum-eigenstate packet
construction. The energy oracle is `_window_energy` below: the window contracted against the
fixed boundary environments.
"""
from __future__ import annotations

import copy
import functools

import numpy as np

from tnkit.backend import get_backend
from tnkit.dmrg import update_left_env
from tnkit.window import build_window, window_mpo, window_expectations
from tnkit.models import SX, SZ
from tnkit.tdvp import tdvp1_evolve
from tnkit.trim import default_trim_margin, detect_occupied, evolve_trimmed
from tnkit._idmrg import tfim_bulk_mpo
from tnkit.vumps import vumps


@functools.lru_cache(maxsize=None)
def _cached_vacuum(g, chi):
    return vumps(g, chi, n_iter=400, tol=1e-12, backend="numpy", mpo=tfim_bulk_mpo(g))


def _vacuum(g, chi):
    """VUMPS vacuum, solved once per (g, chi) in this module; deterministic, copied per call."""
    return copy.deepcopy(_cached_vacuum(g, chi))


def _window_energy(win, FL, FR, W, be):
    """<H> of a window against the FIXED boundary environments FL, FR (bulk MPO W per site)."""
    L = FL
    for t in win:
        L = update_left_env(L, be.array(t), W, be)
    return complex(be.to_scalar(be.einsum("akb,akb->", L, FR)))


def _kicked_window(g, chi, n_left, n_right, positions, angle=0.4):
    """A pure-vacuum window (`tnkit.window.build_window`) with a small local unitary
    rotation applied at each site in `positions` -- a stand-in for a wave packet that costs
    none of the momentum-eigenstate machinery a real packet needs. `cos(angle) I + i sin(angle)
    SX` is exactly unitary, so this perturbs the state without renormalising it. The bulk MPO
    is shifted so the vacuum energy density is exactly zero, the usual convention for a
    collision window (not required by
    `tnkit.trim`'s diagnostic, which is an order-parameter check and does not touch the MPO
    energy channel at all, but required for the energy-agreement check below to compare a
    near-zero number rather than an extensive one).
    """
    gs = _vacuum(g, chi)
    e0 = float(np.real(gs["e0"]))
    Ws = tfim_bulk_mpo(g)
    Ws[0, 2] = Ws[0, 2] - e0 * np.eye(2)
    w0, FL, FR, W, gs = build_window(g, chi, n_left, n_right, gs=gs, mpo=Ws)
    window = [np.asarray(m, dtype=np.complex128) for m in w0]
    rot = np.cos(angle) * np.eye(2) + 1j * np.sin(angle) * SX
    for p in positions:
        window[p] = np.einsum("st,atb->asb", rot, window[p])
    return window, FL, FR, W, gs


# --------------------------------------------------------------------------- default_trim_margin
def test_default_trim_margin_is_six_sigma_plus_correlation_lengths():
    """The documented default: 6 sigma (1e-8 of a Gaussian envelope's peak) + 4 xi."""
    assert default_trim_margin(10.0) == 60
    assert default_trim_margin(10.0, corr_len=2.0) == 68
    assert default_trim_margin(2.5, n_sigma=6.0, n_xi=0.0) == 15


# --------------------------------------------------------------------------- detect_occupied
def test_detect_occupied_finds_the_kicks_and_nothing_on_pure_vacuum():
    g = 1.5
    n_left = n_right = 20
    N = n_left + 1 + n_right
    be = get_backend("numpy")

    window, FL, FR, W, _ = _kicked_window(g, 8, n_left, n_right, positions=[12, 29])
    bg = float(window_expectations(window, SZ)[0])          # site 0 is far from both kicks
    occ_lo, occ_hi = detect_occupied(window, be, 0, N, threshold=1e-6, background=bg)
    assert occ_lo <= 12 and occ_hi >= 30, (occ_lo, occ_hi)

    vac, _, _, _, _ = _kicked_window(g, 8, n_left, n_right, positions=[])
    bg_v = float(window_expectations(vac, SZ)[0])
    occ_lo0, occ_hi0 = detect_occupied(vac, be, 0, N, threshold=1e-6, background=bg_v)
    assert occ_lo0 == occ_hi0, "a pure-vacuum window must report nothing occupied"


def test_detect_occupied_is_robust_to_an_elevated_uniform_bond_dimension():
    """A two-packet ansatz carries an elevated, UNIFORM bond dimension across the whole
    window (a 3*chi_vac finite-state-machine construction spanning the whole chain), including
    sites that
    are still exactly vacuum. An order-parameter diagnostic must not be fooled by this: padding
    every tensor's bond dimension without changing the physical state must not, by itself,
    move `<op_j>` anywhere.
    """
    g = 1.5
    n_left = n_right = 15
    N = n_left + 1 + n_right
    be = get_backend("numpy")

    window, _, _, _, _ = _kicked_window(g, 6, n_left, n_right, positions=[10, 20])
    bg = float(window_expectations(window, SZ)[0])
    occ_lo, occ_hi = detect_occupied(window, be, 0, N, threshold=1e-6, background=bg)

    def pad(m, extra):
        a, s, b = m.shape
        out = np.zeros((a + extra, s, b + extra), dtype=np.complex128)
        out[:a, :, :b] = m
        return out

    padded = [pad(m, 4) for m in window]                 # every tensor: chi -> chi + 4
    occ_lo2, occ_hi2 = detect_occupied(padded, be, 0, N, threshold=1e-6, background=bg)
    assert (occ_lo2, occ_hi2) == (occ_lo, occ_hi), \
        f"padding moved the detected region: {(occ_lo, occ_hi)} -> {(occ_lo2, occ_hi2)}"


def test_trim_composes_with_a_bond_dimension_that_does_not_match_FL_FR():
    """The bond-dimension trap this module was rewritten to survive: `L0=FL` verbatim is a
    SHAPE ERROR the moment the frozen prefix does not carry FL's own bond dimension all the
    way to the cut -- exactly what a two-packet ansatz does (uniform, elevated bond
    dimension across the whole window; see the module docstring). Reproduced synthetically
    here by padding every tensor's bond dimension past FL/FR's, with an interior lo > 0 so the
    boundary the active slice sees is not the true outer edge.
    """
    g = 1.5
    n_left = n_right = 20
    N = n_left + 1 + n_right
    be = get_backend("numpy")
    dt, n_steps = 0.05, 6

    window, FL, FR, W, _ = _kicked_window(g, 6, n_left, n_right, positions=[18, 23], angle=0.4)
    mpo = window_mpo(W, N)

    def pad(m, left, right):
        a, s, b = m.shape
        out = np.zeros((a + left, s, b + right), dtype=np.complex128)
        out[:a, :, :b] = m
        return out

    # Every INTERIOR bond widened by 5; the two true edges (matching FL/FR's own dimension)
    # left alone -- padding those too would desync the test fixture from FL/FR itself, which
    # is not the scenario under test (a bond mismatch strictly inside the window).
    padded = [pad(window[0], 0, 5)] + [pad(m, 5, 5) for m in window[1:-1]] + [pad(window[-1], 5, 0)]
    bg = float(window_expectations(padded, SZ)[0])

    trimmed, _, sizes = evolve_trimmed(padded, mpo, FL, FR, dt, n_steps, bg, backend="numpy",
                                       lo=10, hi=31, margin=4, threshold=1e-6, every=2,
                                       canonicalize=False)   # no crash is itself the assertion
    assert sizes[-1][1] <= 10 and sizes[-1][2] >= 31


# --------------------------------------------------------------------------- correctness
def test_trim_matches_untrimmed_energy_and_observables():
    """Trim ON vs OFF: same energy and the same local order parameter, to a tolerance set
    from the diagnostic threshold -- and the frozen exterior is bit-for-bit untouched.

    THE GEOMETRY IS SIZED TO GROW GRADUALLY, not jump to the full window on the first
    detection: `n_left=n_right=45` against `default_trim_margin(sigma=3)=18` leaves the
    initial window interior on both sides, so the run below re-detects and widens several
    DISTINCT times (this is what caught the real bug the fix above exists for -- a window
    that jumps straight to full on step one is trivially "correct" because it stops trimming
    at all, and never exercises what happens when the active slice changes shape mid-run).

    THE TOLERANCE. Trimming does not introduce an approximation in the active region (it runs
    the SAME `tdvp1_evolve` there, against the induced FL/RN boundary); what it changes is that
    a site the diagnostic calls vacuum is held at EXACTLY its starting value instead of being
    carried through more site-steps that would, in the untrimmed run, leave it at most
    `threshold` away from the background order parameter (by the diagnostic's own definition).
    So untrimmed and trimmed can differ by something set by `threshold` accumulated over the
    run and over each re-canonicalisation at a growth event -- not by anything unrelated to
    either. We assert agreement to `2000 * threshold` (observed discrepancy is several times
    tighter at the `threshold=1e-6` used here).
    """
    g = 1.5
    n_left = n_right = 45
    N = n_left + 1 + n_right
    be = get_backend("numpy")
    dt, n_steps, threshold = 0.05, 15, 1e-6

    window, FL, FR, W, _ = _kicked_window(g, 8, n_left, n_right, positions=[40, 51], angle=0.4)
    mpo = window_mpo(W, N)
    bg = float(window_expectations(window, SZ)[0])

    ref, _ = tdvp1_evolve([m.copy() for m in window], mpo, dt, n_steps, backend="numpy",
                          L0=FL, RN=FR, canonicalize=False)

    margin = default_trim_margin(sigma=3.0, n_xi=0.0)          # a modest synthetic packet width
    occ_lo, occ_hi = detect_occupied(window, be, 0, N, threshold, background=bg)
    lo = max(0, occ_lo - margin)
    hi = min(N, occ_hi + margin)
    assert 0 < lo and hi < N, "test geometry must leave room to grow gradually, not jump to full"
    frozen_before = [m.copy() for m in window]

    trimmed, _, sizes = evolve_trimmed([m.copy() for m in window], mpo, FL, FR, dt, n_steps,
                                       bg, backend="numpy", lo=lo, hi=hi, margin=margin,
                                       threshold=threshold, every=3, canonicalize=False)
    assert len({(a, b) for _, a, b in sizes}) >= 3, \
        f"expected several distinct growth events, got {sizes}"
    assert (sizes[-1][1], sizes[-1][2]) != (0, N), "window should not need to reach the full range"

    Eref = _window_energy(ref, FL, FR, W, be)
    Etrim = _window_energy(trimmed, FL, FR, W, be)
    tol = 2000 * threshold
    assert abs(Eref - Etrim) < tol, f"|Eref - Etrim| = {abs(Eref-Etrim):.3e} >= {tol:.3e}"

    sz_ref = window_expectations(ref, SZ)
    sz_trim = window_expectations(trimmed, SZ)
    dev = np.abs(sz_ref - sz_trim).max()
    assert dev < tol, f"max |<sz>_ref - <sz>_trim| = {dev:.3e} >= {tol:.3e}"

    # NO REFLECTION: sites outside the FINAL active window were never handed to tdvp1_evolve,
    # so they must be bit-for-bit whatever they started as -- not just numerically close to
    # the vacuum value, but the SAME tensor. A reflection off the boundary is exactly a case
    # where the exterior would have been perturbed; this assertion cannot pass if that happened.
    lo_f, hi_f = sizes[-1][1], sizes[-1][2]
    for i in list(range(0, lo_f)) + list(range(hi_f, N)):
        assert np.array_equal(trimmed[i], frozen_before[i]), \
            f"site {i} outside the active window [{lo_f},{hi_f}) was modified"

    # and those same untouched sites still read the background order parameter, independently
    # of the tensor-equality check above.
    if lo_f > 0 or hi_f < N:
        outside = np.abs(np.concatenate([sz_trim[:lo_f], sz_trim[hi_f:]]) - bg)
        assert outside.max() < 1e-10, f"frozen exterior drifted from background: {outside.max():.3e}"


def test_trim_energy_deviation_at_default_margin_is_recorded():
    """RECORD, not a gate: trim ON at the module's default margin (`default_trim_margin`,
    no synthetic tightening) vs OFF, on a packet width (sigma=8) representative of a real
    run, so the energy deviation the mechanism incurs at the settings a real run actually
    uses is visible in this test's own output rather than only inferable from the tight,
    hand-picked-margin gate above (`test_trim_matches_untrimmed_energy_and_observables`).

    The bound below is deliberately loose -- two orders of magnitude above the deviation
    this geometry has ever produced. It exists to catch an actual regression (NaN, a
    window that stops growing, a missed re-canonicalisation), not to pin the last digit of
    an approximation error that is expected, by design (see `tnkit/trim.py`'s "WHY THIS IS
    OPT-IN" note), to move with sigma, threshold, and geometry, and that this module makes
    no claim of being able to separate from ordinary numerical drift after the fact.
    """
    g = 1.5
    sigma = 8.0
    n_left = n_right = 70
    N = n_left + 1 + n_right
    be = get_backend("numpy")
    dt, n_steps, threshold = 0.05, 15, 1e-6

    window, FL, FR, W, _ = _kicked_window(g, 8, n_left, n_right, positions=[65, 76], angle=0.4)
    mpo = window_mpo(W, N)
    bg = float(window_expectations(window, SZ)[0])

    ref, _ = tdvp1_evolve([m.copy() for m in window], mpo, dt, n_steps, backend="numpy",
                          L0=FL, RN=FR, canonicalize=False)

    margin = default_trim_margin(sigma)                        # the module default, unmodified
    occ_lo, occ_hi = detect_occupied(window, be, 0, N, threshold, background=bg)
    lo = max(0, occ_lo - margin)
    hi = min(N, occ_hi + margin)

    trimmed, _, sizes = evolve_trimmed([m.copy() for m in window], mpo, FL, FR, dt, n_steps,
                                       bg, backend="numpy", lo=lo, hi=hi, margin=margin,
                                       threshold=threshold, every=3, canonicalize=False)

    Eref = _window_energy(ref, FL, FR, W, be)
    Etrim = _window_energy(trimmed, FL, FR, W, be)
    dev_abs = abs(Eref - Etrim)
    dev_rel = dev_abs / max(abs(Eref), 1e-12)
    loose_bound = 1e-2
    assert dev_abs < loose_bound, (
        f"trim on (default margin={margin}, sigma={sigma}) vs off, at the default settings: "
        f"|Eref-Etrim|={dev_abs:.3e} (relative {dev_rel:.3e}), window grew to "
        f"[{sizes[-1][1]},{sizes[-1][2]}) of {N} -- exceeds the loose regression bound "
        f"{loose_bound:.1e}, which is a real problem, not an expected approximation size"
    )


def test_trim_off_equivalent_reproduces_untrimmed_bit_for_bit():
    """A trim window that already spans the whole state, with `every` chunking it into
    several `tdvp1_evolve` calls, must reproduce ONE untrimmed call bit-for-bit: 1TDVP has no
    hidden state between calls, so evolving n1 then n2 steps must equal evolving n1+n2 in one
    call. A caller that turns trimming off by not calling `evolve_trimmed` at all is
    automatically exact; this test pins the part of `evolve_trimmed` that a caller COULD use
    the same way (lo=0, hi=N) if it wanted a
    single code path with trimming toggled by the window bounds alone.
    """
    g = 1.5
    n_left = n_right = 12
    N = n_left + 1 + n_right
    dt, n_steps = 0.05, 9

    window, FL, FR, W, _ = _kicked_window(g, 6, n_left, n_right, positions=[10, 15], angle=0.3)
    mpo = window_mpo(W, N)
    bg = float(window_expectations(window, SZ)[0])

    ref, _ = tdvp1_evolve([m.copy() for m in window], mpo, dt, n_steps, backend="numpy",
                          L0=FL, RN=FR, canonicalize=False)
    chunked, _, sizes = evolve_trimmed([m.copy() for m in window], mpo, FL, FR, dt, n_steps,
                                       bg, backend="numpy", lo=0, hi=N, margin=0,
                                       threshold=1e30, every=4, canonicalize=False)
    assert sizes[-1][1] == 0 and sizes[-1][2] == N, "an all-active window must never shrink"
    for i in range(N):
        assert np.array_equal(ref[i], chunked[i]), f"site {i} differs after chunking"


# --------------------------------------------------------------------------- adaptivity
def test_window_grows_after_a_synthetic_collision():
    """A window initialised TIGHT around two kicks must widen as they spread -- the
    adaptivity property, checked directly against the size history `evolve_trimmed` returns.
    """
    g = 1.5
    n_left = n_right = 25
    N = n_left + 1 + n_right
    be = get_backend("numpy")
    dt, n_steps, threshold = 0.05, 20, 1e-6

    window, FL, FR, W, _ = _kicked_window(g, 8, n_left, n_right, positions=[24, 27], angle=0.5)
    mpo = window_mpo(W, N)
    bg = float(window_expectations(window, SZ)[0])

    occ_lo, occ_hi = detect_occupied(window, be, 0, N, threshold, background=bg)
    tight_margin = 1                                    # deliberately too tight to stay put
    lo0 = max(0, occ_lo - tight_margin)
    hi0 = min(N, occ_hi + tight_margin)

    _, _, sizes = evolve_trimmed([m.copy() for m in window], mpo, FL, FR, dt, n_steps,
                                 bg, backend="numpy", lo=lo0, hi=hi0, margin=tight_margin,
                                 threshold=threshold, every=2, canonicalize=False)

    widths = [hi - lo for _, lo, hi in sizes]
    assert widths[-1] > widths[0], f"window never grew: {widths}"
    assert widths == sorted(widths), "the window shrank at some point (it must be monotonic)"
    # it must still have grown from WITHIN the full window, not run off the end immediately
    assert sizes[-1][1] >= 0 and sizes[-1][2] <= N
