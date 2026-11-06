"""`degenerate_tol`: a truncation that keeps or drops whole groups of near-equal singular values."""
import numpy as np
import pytest

from tnkit import tdvp_evolver
from tnkit._dmrg import _cut, _truncate, apply_two_site
from tnkit.backend import get_backend
from tnkit.opsum import OpSum, compile_mpo, qudit
from tnkit.api import dmrg
from tnkit.tdvp import TDVPOptions, _evolve_two_site, tdvp1_evolve, tdvp2_evolve
from tnkit.tdvp_evolver import TDVPEvolver

BE = get_backend("numpy")
TOL = 1e-8


def _planted(S, rows=12, cols=10, seed=0, eps=1e-12):
    """A matrix with singular values S (each perturbed by a relative eps) and random vectors."""
    rng = np.random.default_rng(seed)
    S = np.asarray(S, float) * (1 + eps * rng.uniform(-1, 1, len(S)))
    U = np.linalg.qr(rng.normal(size=(rows, len(S))) + 1j * rng.normal(size=(rows, len(S))))[0]
    V = np.linalg.qr(rng.normal(size=(cols, len(S))) + 1j * rng.normal(size=(cols, len(S))))[0]
    return (U * S) @ V.conj().T


SPEC = [5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0, 1.0]          # sum of squares 72
TAIL = [5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]          # sum of squares 58


@pytest.mark.parametrize("chi, k, pinned", [(3, 2, True), (4, 4, False), (5, 4, True),
                                            (6, 4, True), (7, 7, False), (8, 8, False),
                                            (10, 8, False)])
def test_cut_at_the_cap(chi, k, pinned):
    S = np.array(SPEC)
    assert _cut(S, chi, degenerate_tol=TOL)[:2] == (k, pinned)
    assert _cut(S, chi)[:2] == (min(chi, 8), False)


def test_cut_edge_cases():
    assert _cut(np.ones(4), 2, degenerate_tol=TOL)[:2] == (2, False)   # moving down keeps nothing
    assert _cut(np.array([3.0, 2, 0, 0, 0]), 3, degenerate_tol=TOL)[:2] == (3, False)   # zeros
    # svd_tol alone cuts at 3 (drop 22/72) and at 5 (drop 9/72); the groups end at 4 and 7
    S = np.array(SPEC)
    assert _cut(S, 8, 0.35)[:2] == (3, False) and _cut(S, 8, 0.35, TOL)[:2] == (4, False)
    assert _cut(S, 8, 0.15)[:2] == (5, False)
    k, pinned, disc = _cut(S, 8, 0.15, TOL)
    assert (k, pinned) == (7, False) and disc == pytest.approx(1 / 72)
    # the cap cut moves down to 4, then svd_tol cuts lower and the bond is no longer pinned
    k, pinned, disc = _cut(S, 5, 0.45, TOL)
    assert (k, pinned) == (2, False) and disc == pytest.approx(31 / 72)


def test_cut_group_reaching_the_end_keeps_everything():
    # svd_tol alone cuts at 5 (drop 3/58 < 0.06); the group of 1s runs to the end of the kept
    # spectrum, so nothing is cut
    S = np.array(TAIL)
    assert _cut(S, 8, 0.06)[:2] == (5, False)
    assert _cut(S, 8, 0.06, TOL) == (8, False, 0.0)


def test_cut_does_not_chain_along_a_smooth_spectrum():
    """Neighbouring values differ by 1% throughout, so chained comparisons at a 2% tolerance
    would make one group of the whole tail. Measured from the cut, a group spans at most 2%."""
    S = np.array([10.0, 5.0] + [0.99 ** i for i in range(60)])
    k, pinned, _ = _cut(S, 40, degenerate_tol=2e-2)
    assert 38 <= k < 40 and pinned
    assert S[k - 1] - S[40] > 2e-2 * S[k - 1] and S[k] - S[40] <= 2e-2 * S[k]
    sq = S ** 2
    budget = (sq[39:].sum() + sq[40:].sum()) / 2 / sq.sum()            # svd_tol alone cuts at 40
    assert _cut(S, 62, budget)[0] == 40
    k = _cut(S, 62, budget, 2e-2)[0]
    assert 40 < k <= 42 and S[39] - S[k] > 2e-2 * S[39] >= S[39] - S[k - 1]


def test_cut_budget_counts_disc0_and_tot():
    S = np.array(SPEC)
    assert _cut(S, 8, 0.15)[0] == 5
    assert _cut(S, 8, 0.15, disc0=0.05)[0] == 6
    assert _cut(S, 8, 0.15, tot=90.0)[0] == 4


def test_truncate_default_is_unchanged():
    theta = _planted(SPEC)
    U0, S0, Vh0 = BE.svd(theta)
    U, S, Vh, disc = _truncate(theta, 3, "svd", BE)
    assert U.tobytes() == U0[:, :3].tobytes() and S.tobytes() == S0[:3].tobytes()
    assert Vh.tobytes() == Vh0[:3].tobytes()


def _block(S):
    """A two-site block (a, 2, 2, b) with Schmidt values S at its middle cut, an identity
    MPO and trivial environments, so that a step with dt = 0 only truncates."""
    a = b = 4
    theta = _planted(S, rows=2 * a, cols=2 * b).reshape(a, 2, 2, b)
    U, s, Vh = np.linalg.svd(theta.reshape(2 * a, 2 * b), full_matrices=False)
    M = [(U * s).reshape(a, 2, -1), Vh.reshape(-1, 2, b)]
    W = np.eye(2, dtype=complex)[None, None]
    L = [np.eye(a, dtype=complex)[:, None, :], None, None]
    R = [None, None, np.eye(b, dtype=complex)[:, None, :]]
    assert np.allclose(apply_two_site(theta, L[0], W, W, R[2]), theta)
    return M, L, R, [W, W]


def _step(S, chi, moving="right", **kw):
    M, L, R, W = _block(S)
    pinned = _evolve_two_site(M, L, R, W, 0, chi, 0.0, BE, moving, **kw)
    return int(M[0].shape[2]), pinned


@pytest.mark.parametrize("moving", ["right", "left"])
def test_two_site_step_at_the_cap_is_pinned(moving):
    assert _step(SPEC, 3, moving, degenerate_tol=TOL) == (2, True)
    assert _step(SPEC, 4, moving, degenerate_tol=TOL) == (4, False)


def test_two_site_step_below_the_cap_is_not_pinned():
    assert _step(SPEC, 10, degenerate_tol=TOL) == (8, False)
    assert _step(SPEC, 10, svd_tol=1e-3, degenerate_tol=TOL) == (8, False)


def test_two_site_step_cut_lower_by_svd_tol_is_not_pinned():
    assert _step(SPEC, 5, degenerate_tol=TOL) == (4, True)
    assert _step(SPEC, 5, svd_tol=0.45, degenerate_tol=TOL) == (2, False)


def test_two_site_step_below_the_cap_keeps_the_group():
    # svd_tol alone keeps 5 values (drop 9/72, under 0.13); the group of three 2s ends at 7,
    # still within the budget, so the cut moves up and the bond is not full
    assert _step(SPEC, 8, svd_tol=0.13, degenerate_tol=TOL) == (7, False)
    assert _step(SPEC, 8, svd_tol=0.13) == (5, False)


def test_two_site_step_group_reaching_the_end():
    assert _step(TAIL, 8, svd_tol=0.06) == (5, False)
    assert _step(TAIL, 8, svd_tol=0.06, degenerate_tol=TOL) == (8, False)


def test_two_site_step_budget_includes_the_cap_cut_and_reports_disc():
    prof = {"expm": 0.0, "trunc": 0.0, "env": 0.0, "disc": 0.0}
    M, L, R, W = _block(SPEC)
    _evolve_two_site(M, L, R, W, 0, 6, 0.0, BE, "right", prof, svd_tol=0.08)
    assert int(M[0].shape[2]) == 6 and prof["disc"] == pytest.approx(5 / 72, rel=1e-9)
    prof["disc"] = 0.0
    M, L, R, W = _block(SPEC)
    _evolve_two_site(M, L, R, W, 0, 8, 0.0, BE, "right", prof, svd_tol=0.15)
    assert int(M[0].shape[2]) == 5 and prof["disc"] == pytest.approx(9 / 72, rel=1e-9)


@pytest.mark.parametrize("bad", [dict(degenerate_tol=-1e-3), dict(degenerate_tol=1.0),
                                 dict(degenerate_tol=float("nan")),
                                 dict(degenerate_tol=TOL, trunc="randomized")])
def test_invalid_settings_raise_when_resolved(two_copy, bad):
    psi, W = two_copy
    with pytest.raises(ValueError, match="degenerate_tol"):
        tdvp2_evolve(list(psi), W, 5, 0.05, 1, TDVPOptions(**bad))
    with pytest.raises(ValueError, match="degenerate_tol"):
        TDVPEvolver(psi, W, 0.05, 5, TDVPOptions(**bad))


def _two_copy_tfim(n, g):
    """Two uncoupled transverse-field Ising chains, one on each half of a d = 4 site. The swap
    of the copies acts on each site, so the Schmidt values of a state symmetric under it come
    in equal pairs at every cut, and TDVP keeps them equal."""
    I, X, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
    ops = {"Xu": np.kron(X, I), "Xd": np.kron(I, X), "Zu": np.kron(Z, I), "Zd": np.kron(I, Z)}
    os = OpSum()
    for k in range(n):
        for c in "ud":
            os += (-g, "X" + c, k)
            if k < n - 1:
                os += (-1.0, "Z" + c, k, "Z" + c, k + 1)
    return compile_mpo(os, [qudit(4, ops)] * n)[0]


def _tilted_ising(n, g, h):
    """Ising chain in a tilted field: no symmetry, so its Schmidt values are not degenerate."""
    ops = {"X": np.array([[0, 1], [1, 0]]), "Z": np.diag([1.0, -1.0])}
    os = OpSum()
    for k in range(n):
        os += (-g, "X", k)
        os += (-h, "Z", k)
        if k < n - 1:
            os += (-1.0, "Z", k, "Z", k + 1)
    return compile_mpo(os, [qudit(2, ops)] * n)[0]


@pytest.fixture(scope="module")
def two_copy():
    n = 5
    _, m, _ = dmrg(_two_copy_tfim(n, 1.0), chi=64, n_sweeps=8, tol=1e-12)
    return [np.asarray(t) for t in m], _two_copy_tfim(n, 1.5)


def test_twelve_site_schedule_counts_pinned_bonds_as_full(two_copy, monkeypatch):
    psi, W = two_copy
    cap = 5
    calls = []

    def spy(M, L, R, mpo, i, chi, *args, **kw):
        rank = min(M[i].shape[0] * M[i].shape[1], M[i + 1].shape[1] * M[i + 1].shape[2])
        pinned = _evolve_two_site(M, L, R, mpo, i, chi, *args, **kw)
        calls.append((rank, int(M[i].shape[2]), pinned))
        return pinned

    monkeypatch.setattr(tdvp_evolver, "_evolve_two_site", spy)
    ev = TDVPEvolver(psi, W, 0.05, cap, TDVPOptions(svd_tol=1e-12, degenerate_tol=TOL))
    steps = []
    for _ in range(4):
        before = list(ev._pinned)
        dims = [int(t.shape[2]) for t in ev.mps[:-1]]
        ev.run(1, "12site", step_callback=lambda d, o: steps.append(d["n_two"]))
        assert steps[-1] == sum(d != cap and not p for d, p in zip(dims, before))
        dims = [int(t.shape[2]) for t in ev.mps[:-1]]
        assert all(d < cap for d, p in zip(dims, ev._pinned) if p)
    assert all(rank > cap and k < cap for rank, k, pinned in calls if pinned)
    assert any(pinned for *_, pinned in calls)                             # the rule did act
    assert max(dims) <= cap


def test_reset_forgets_the_pins(two_copy):
    psi, W = two_copy
    ev = TDVPEvolver(psi, W, 0.05, 5, TDVPOptions(svd_tol=1e-12, degenerate_tol=TOL))
    ev.run(2, "2site")
    assert any(ev._pinned)
    ev.reset(psi)
    assert not any(ev._pinned)


@pytest.mark.parametrize("stale", ["invalidate", "replace"])
def test_a_rebuild_forgets_the_pins(two_copy, stale):
    psi, W = two_copy
    ev = TDVPEvolver(psi, W, 0.05, 5, TDVPOptions(svd_tol=1e-12, degenerate_tol=TOL))
    ev.run(2, "2site")
    assert any(ev._pinned)
    if stale == "invalidate":
        ev.invalidate()
    else:
        ev.mps[0] = ev.mps[0].copy()
    ev.run(1, "1site")
    assert not any(ev._pinned)


@pytest.mark.parametrize("method", ["2site", "12site"])
def test_the_last_two_site_update_sets_the_pin(method, monkeypatch):
    real = tdvp_evolver._evolve_two_site

    def spy(*a, **kw):
        real(*a, **kw)
        return a[8] == "left"                        # only the backward sweep pins

    monkeypatch.setattr(tdvp_evolver, "_evolve_two_site", spy)
    n = 4
    _, m, _ = dmrg(_tilted_ising(n, 1.0, 0.3), chi=8, n_sweeps=4, tol=1e-10)
    ev = TDVPEvolver([np.asarray(t) for t in m], _tilted_ising(n, 1.4, 0.2), 0.05, 4)
    ev.run(1, method, **(dict(two_site_every=1) if method == "12site" else {}))
    assert ev._pinned == [True] * (n - 1)


def test_a_tolerance_that_never_acts_changes_no_bit():
    """With no near-equal values at any cut, the option takes the full-spectrum path but must
    cut where the default does, to the bit."""
    n = 6
    _, m, _ = dmrg(_tilted_ising(n, 1.0, 0.3), chi=16, n_sweeps=6, tol=1e-12)
    psi, W = [np.asarray(t) for t in m], _tilted_ising(n, 1.4, 0.2)
    for o in (dict(), dict(svd_tol=1e-6)):
        a = tdvp2_evolve(list(psi), W, 4, 0.05, 3, TDVPOptions(**o))[0]
        b = tdvp2_evolve(list(psi), W, 4, 0.05, 3, TDVPOptions(degenerate_tol=1e-12, **o))[0]
        assert [t.tobytes() for t in a] == [t.tobytes() for t in b]
    ev = TDVPEvolver(psi, W, 0.05, 4, TDVPOptions(degenerate_tol=1e-12))
    ev.run(3, "2site")
    assert not any(ev._pinned)


def test_tdvp1_refuses_the_flat_keyword(two_copy):
    psi, W = two_copy
    with pytest.raises(TypeError, match="degenerate_tol"):
        tdvp1_evolve(list(psi), W, 0.05, 1, degenerate_tol=TOL)
