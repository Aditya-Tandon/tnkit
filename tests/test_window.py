"""Regression tests for the absorbing-background window (E3)."""
import copy

import numpy as np
import pytest

from tnkit.free_fermion_dynamics import gamma_ground, sigma_z
from tnkit.models import SX, SZ
from tnkit.window import build_window, window_expectations


@pytest.fixture(scope="module")
def _gs16():
    return build_window(1.5, 16, 0, 0)[4]


@pytest.fixture
def gs16(_gs16):
    """The g = 1.5, chi = 16 background, solved once per module (VUMPS is
    deterministic, so sharing it changes no tested value); each test gets its own copy."""
    return copy.deepcopy(_gs16)


def test_unperturbed_window_is_uniform_and_matches_exact(gs16):
    """The closure bug this pins produced a FAKE 3% boundary feature.

    Measuring a window with the ordinary local_expectations closes it with 1x1 boundary vectors,
    which is wrong when the edge tensors carry a full bond of dimension chi onto the background.
    The symptom was <sz> = 0.5357 at the edge vs 0.5522 in the bulk, in a state that is exactly
    uniform -- and the bulk number was wrong too. The correct closure is a chi x chi identity.
    """
    g = 1.5
    w, _, _, _, _ = build_window(g, 16, 6, 6, gs=gs16)
    sz = window_expectations(w, SZ)
    assert sz.max() - sz.min() < 1e-9, "background is exactly uniform; any spread is an artefact"
    exact = sigma_z(gamma_ground(200, g), 100)
    assert abs(sz[0] - exact) < 1e-7, f"window bulk {sz[0]} vs exact {exact}"


def test_sigma_z_perturbation_is_local_at_t0(gs16):
    """sigma^z at the centre must leave <sz> untouched everywhere (it commutes with itself)."""
    g = 1.5
    w0, _, _, _, gs = build_window(g, 16, 5, 5, gs=gs16)
    w1, _, _, _, _ = build_window(g, 16, 5, 5, op=SZ, gs=gs)
    assert np.abs(window_expectations(w1, SZ) - window_expectations(w0, SZ)).max() < 1e-9


def test_growth_does_not_perturb_the_state(gs16):
    """A growth event must be a NO-OP on every observable (B1).

    Growing the window prepends AL / appends AR and leaves FL, FR untouched -- valid only because
    the background is stationary. If that reasoning is right the state is physically unchanged and
    only its representation is wider, so <sigma^z_j> at the pre-existing sites must be bit-for-bit
    comparable and the new sites must read the background value. If it is wrong, growth silently
    injects a perturbation at the seam, which is exactly the class of error the fake-3%-boundary
    bug already demonstrated is easy to miss.
    """
    from tnkit.window import build_window, grow_window, window_expectations

    g = 1.5
    w, _, _, _, gs = build_window(g, 16, 5, 5, op=SZ, gs=gs16)
    before = window_expectations(w, SZ)

    AL = np.asarray(gs["AL"]); AR = np.asarray(gs["AR"])
    w2 = grow_window(w, AL, AR, n_left=3, n_right=3)
    after = window_expectations(w2, SZ)

    assert len(w2) == len(w) + 6
    # the original sites are unchanged
    assert np.abs(after[3:-3] - before).max() < 1e-10, \
        f"growth perturbed the interior by {np.abs(after[3:-3]-before).max():.2e}"
    # the new sites sit at the background value, which is what the untouched edge already read
    bg = float(before[0])
    assert np.abs(after[:3] - bg).max() < 1e-10
    assert np.abs(after[-3:] - bg).max() < 1e-10


def test_edge_deviation_detects_a_disturbance(gs16):
    """The growth trigger must read ~0 on an unperturbed window and rise once perturbed."""
    from tnkit.window import build_window, edge_deviation, window_expectations

    g = 1.5
    w0, _, _, _, gs = build_window(g, 16, 4, 4, gs=gs16)
    bg = float(window_expectations(w0, SZ)[0])
    assert edge_deviation(w0, SZ, bg) < 1e-10        # uniform background, nothing at the edge

    # a sigma^x at the centre flips <sigma^z> there, but the EDGE should still be quiet at t=0
    w1, _, _, _, _ = build_window(g, 16, 4, 4, op=SX, gs=gs)
    assert edge_deviation(w1, SZ, bg) < 1e-8


# ------------------------------------------------------------------ evolve_growing_window
G, CHI, DT = 1.5, 8, 0.05


@pytest.fixture(scope="module")
def vacuum():
    w0, FL, FR, W, gs = build_window(G, CHI, 3, 3)
    bg = float(window_expectations(w0, SZ)[0])
    return dict(FL=FL, FR=FR, W=W, gs=gs, bg=bg, AL=np.asarray(gs["AL"]), AR=np.asarray(gs["AR"]))


def _grow(vac, window, n_steps, **kw):
    from tnkit.window import evolve_growing_window
    return evolve_growing_window([x.copy() for x in window], vac["FL"], vac["FR"], vac["W"], DT,
                                 n_steps, op=SZ, background=vac["bg"], AL=vac["AL"],
                                 AR=vac["AR"], **kw)


def _fixed(vac, n_left, n_right, n_steps):
    """Reference: one wide window, canonicalised once and evolved without growth."""
    from tnkit.tdvp import tdvp1_evolve
    from tnkit.window import window_mpo
    w, *_ = build_window(G, CHI, n_left, n_right, op=SX, gs=vac["gs"])
    out, _ = tdvp1_evolve(w, window_mpo(vac["W"], len(w)), DT, n_steps, backend="numpy",
                          L0=vac["FL"], RN=vac["FR"], canonicalize=True)
    return window_expectations(out, SZ)


def test_growing_window_without_growth_matches_fixed_window(vacuum):
    """A built window has its centre in the middle; 1TDVP needs it at site 0."""
    w, *_ = build_window(G, CHI, 3, 3, op=SX, gs=vacuum["gs"])
    got, _, sizes = _grow(vacuum, w, 6, grow_thresh=np.inf)
    assert sizes[-1][1] == len(w)
    assert np.abs(window_expectations(got, SZ) - _fixed(vacuum, 3, 3, 6)).max() < 1e-12


def test_growing_window_after_left_growth_matches_fixed_window(vacuum):
    """Growing both sides at every step, versus the final width evolved from the start."""
    w, *_ = build_window(G, CHI, 3, 3, op=SX, gs=vacuum["gs"])
    got, _, sizes = _grow(vacuum, w, 6, grow_thresh=-1.0, grow_by=2)
    assert sizes[-1][1] == len(w) + 24
    assert np.abs(window_expectations(got, SZ) - _fixed(vacuum, 15, 15, 6)).max() < 1e-8


def test_growing_window_grows_only_the_side_that_tripped(vacuum):
    w, *_ = build_window(G, CHI, 1, 6, op=SX, gs=vacuum["gs"])     # disturbance near the left
    got, _, sizes = _grow(vacuum, w, 8, grow_thresh=1e-6, grow_by=2, symmetric=False)
    added = len(got) - len(w)
    assert added > 0
    assert np.abs(window_expectations(got, SZ) - _fixed(vacuum, 1 + added, 6, 8)).max() < 1e-8


def test_symmetric_growth_widens_both_sides_when_one_trips(vacuum):
    w, *_ = build_window(G, CHI, 1, 6, op=SX, gs=vacuum["gs"])
    got, _, _ = _grow(vacuum, w, 8, grow_thresh=1e-6, grow_by=2)
    added = (len(got) - len(w)) // 2
    assert added > 0 and (len(got) - len(w)) % 2 == 0
    assert np.abs(window_expectations(got, SZ) - _fixed(vacuum, 1 + added, 6 + added, 8)).max() < 1e-8


def test_max_sites_counts_the_sites_a_growth_adds(vacuum):
    w, *_ = build_window(G, CHI, 3, 3, op=SX, gs=vacuum["gs"])
    _, _, sizes = _grow(vacuum, w, 6, grow_thresh=-1.0, grow_by=2, max_sites=len(w) + 10)
    assert sizes[-1][1] == len(w) + 8


def test_grow_every_sets_the_trigger_cadence(vacuum):
    w, *_ = build_window(G, CHI, 3, 3, op=SX, gs=vacuum["gs"])
    _, _, sizes = _grow(vacuum, w, 4, grow_thresh=-1.0, grow_by=1, grow_every=2)
    assert [n for _, n in sizes] == [7, 7, 9, 9, 11]
