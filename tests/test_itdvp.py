"""Regression tests for uniform infinite TDVP (E2).

The gate is the exact free-fermion solution evaluated at the middle site of a large finite chain,
where the boundary is causally disconnected over the test window.
"""
import numpy as np

from tnkit.free_fermion_dynamics import gamma_ground, evolve, sigma_z
from tnkit.itdvp import itdvp_quench


def _exact_bulk_sz(N, g0, g1, times):
    gamma0 = gamma_ground(N, g0)
    return np.array([sigma_z(G, N // 2) for _, G in evolve(gamma0, N, g1, times)])


def test_itdvp_matches_free_fermions_short_time():
    """At short time and modest chi the infinite-chain answer should be essentially exact."""
    r = itdvp_quench(2.0, 1.0, 16, 0.5, dt=0.05, meas_every=2)
    ex = _exact_bulk_sz(100, 2.0, 1.0, r["t"])
    assert np.abs(r["sz"] - ex).max() < 1e-5


def test_bond_tensor_evolves_FORWARD_not_backward():
    """Pins the sign of the C step -- the bug this module was born with.

    By analogy with finite 1-site TDVP one expects the bond tensor to be counter-rotated
    (+i dt), and that is what was implemented first. It is wrong here: the gauge extraction
    AL ~ AC C^-1 already carries the subtraction, so counter-rotating applies it twice.

    The observable alone does NOT catch this early -- both signs agree to 4e-7 after one step.
    The decisive quantity is the gauge error eps = ||AC - AL C||, which stays at ~1e-6 with the
    correct sign and diverges to ~0.6 by t=0.3 with the wrong one, while the norm and energy
    still look perfectly plausible. So this test checks eps, not just accuracy.
    """
    r = itdvp_quench(2.0, 1.0, 16, 0.3, dt=0.05, meas_every=1)
    eps = np.asarray(r["eps"][1:], dtype=float)      # entry 0 is nan (pre-evolution record)
    assert np.isfinite(eps).all()
    assert eps.max() < 1e-3, f"gauge error diverged (max eps = {eps.max():.2e})"
    # and it must not be growing systematically
    assert eps[-1] < 10 * eps[0]


def test_entanglement_grows_after_quench():
    """A global quench must grow entanglement monotonically in the bulk."""
    r = itdvp_quench(2.0, 1.0, 24, 2.0, dt=0.05, meas_every=4)
    S = r["S"]
    assert S[-1] > S[0]
    # no large non-monotonic excursions -- an early dip was a symptom of the sign bug
    assert np.min(np.diff(S)) > -1e-3
