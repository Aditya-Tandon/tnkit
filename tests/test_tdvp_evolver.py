"""`TDVPEvolver`: same arithmetic as the integrators, kept alive between calls."""
import numpy as np
import pytest

from tnkit.api import dmrg, tfim_mpo
from tnkit.observables import local_expectations, mps_overlap
from tnkit.models import SZ
from tnkit.tdvp import TDVPOptions, tdvp1_evolve, tdvp2_evolve
from tnkit.tdvp_evolver import TDVPEvolver

N, CHI, DT = 10, 8, 0.05


@pytest.fixture(scope="module")
def setup():
    _, mps0, _ = dmrg(tfim_mpo(N, 1.0), chi=4, n_sweeps=6, tol=1e-12)
    return [np.asarray(t) for t in mps0], tfim_mpo(N, 1.5)


def _bytes(mps):
    return [np.asarray(t).tobytes() for t in mps]


def _call(method, psi, mpo, n, backend="numpy", **kw):
    o = TDVPOptions(backend=backend, svd_tol=1e-10, chi_max=CHI, **kw)
    if method == "1site":
        o.svd_tol, o.chi_max = 0.0, None
        return tdvp1_evolve(list(psi), mpo, DT, n, o)[0]
    return tdvp2_evolve(list(psi), mpo, CHI, DT, n, o)[0]


def _evolver(psi, mpo, backend="numpy", **kw):
    return TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(backend=backend, svd_tol=1e-10, **kw))


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
@pytest.mark.parametrize("method", ["1site", "2site"])
def test_one_run_is_one_call_bit_for_bit(setup, method, backend):
    if backend != "numpy":
        pytest.importorskip("torch")
    psi, mpo = setup
    ev = _evolver(psi, mpo, backend)
    ev.run(5, method)
    got = [t.cpu().numpy() if hasattr(t, "cpu") else t for t in ev.mps]
    ref = [t.cpu().numpy() if hasattr(t, "cpu") else t for t in _call(method, psi, mpo, 5, backend)]
    assert _bytes(got) == _bytes(ref)


def _host(mps):
    return [t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t) for t in mps]


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
@pytest.mark.parametrize("method", ["1site", "2site"])
def test_prepare_then_run_is_run_bit_for_bit(setup, method, backend):
    if backend != "numpy":
        pytest.importorskip("torch")
    psi, mpo = setup
    ev = _evolver(psi, mpo, backend)
    ev.prepare()
    assert ev._mps.center == 0
    ev.prepare()                                           # a second call does nothing
    ev.run(3, method)
    ref = _evolver(psi, mpo, backend)
    ref.run(3, method)
    assert _bytes(_host(ev.mps)) == _bytes(_host(ref.mps))


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_continued_runs_equal_reentry_without_canonicalising(setup, backend):
    if backend != "numpy":
        pytest.importorskip("torch")
    psi, mpo = setup
    ev = _evolver(psi, mpo, backend)
    ref = list(psi)
    for k, method in enumerate(["2site", "2site", "1site", "2site", "1site"]):
        ev.run(3, method)
        ref = _call(method, ref, mpo, 3, backend, canonicalize=(k == 0))
    assert _bytes(_host(ev.mps)) == _bytes(_host(ref))
    assert ev.step == 15 and ev.t == pytest.approx(15 * DT)


def test_rebuild_after_a_replaced_tensor_is_the_canonicalising_call(setup):
    psi, mpo = setup
    ev = _evolver(psi, mpo)
    ev.run(3, "2site")
    ev.mps[4] = ev.mps[4] * (1 + 1e-3)               # a new array: the evolver must rebuild
    edited = [t.copy() for t in ev.mps]
    ev.run(2, "2site")
    assert _bytes(ev.mps) == _bytes(_call("2site", edited, mpo, 2))


def test_step0_continues_the_step_count_and_time(setup):
    psi, mpo = setup
    seen = []
    ev = _evolver(psi, mpo, step_callback=lambda d, o: seen.append((d["step"], d["t"])))
    ev2 = TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(backend="numpy", svd_tol=1e-10),
                      step0=40)
    ev2.run(2, "1site", step_callback=lambda d, o: seen.append((d["step"], d["t"])))
    assert seen == [(41, pytest.approx(41 * DT)), (42, pytest.approx(42 * DT))]
    assert ev.step == 0


def test_agrees_with_the_rebuild_path_to_round_off(setup):
    """The re-entry path re-gauges every call: different bits, same state."""
    psi, mpo = setup
    ev = _evolver(psi, mpo)
    ref = list(psi)
    for method in ["2site"] * 4 + ["1site"] * 4:
        ev.run(5, method)
        ref = _call(method, ref, mpo, 5)
    a, b = list(ev.mps), list(ref)
    fid = abs(mps_overlap(a, b)) / np.sqrt(abs(mps_overlap(a, a) * mps_overlap(b, b)))
    assert abs(1 - fid) < 1e-12
    assert np.max(np.abs(local_expectations(a, SZ) - local_expectations(b, SZ))) < 1e-11


def test_state_is_right_canonical_with_centre_at_zero(setup):
    psi, mpo = setup
    ev = _evolver(psi, mpo)
    ev.run(4, "2site")
    assert ev.center == 0
    for B in list(ev.mps)[1:]:
        a = B.shape[0]
        g = np.einsum("asb,csb->ac", B, B.conj())
        assert np.allclose(g, np.eye(a), atol=1e-12)
    assert np.linalg.norm(ev.mps[0]) == pytest.approx(1.0, abs=1e-12)


def test_replaced_tensor_triggers_a_rebuild(setup, monkeypatch):
    import tnkit.tdvp_evolver as E
    psi, mpo = setup
    ev = _evolver(psi, mpo)
    calls = []
    real = E.right_canonicalize
    monkeypatch.setattr(E, "right_canonicalize", lambda M, be: calls.append(1) or real(M, be))
    ev.run(2, "2site")
    ev.run(2, "2site")
    assert calls == [1]                               # built once, then kept
    ev.mps[3] = ev.mps[3].copy()
    assert ev.center is None
    ev.run(1, "2site")
    assert calls == [1, 1]
    ev.invalidate()
    ev.run(1, "1site")
    assert calls == [1, 1, 1]
    ev.reset(psi)
    ev.run(1, "1site")
    assert len(calls) == 4 and ev.center == 0


def test_step_callback_counts_from_creation(setup):
    psi, mpo = setup
    seen = []
    ev = _evolver(psi, mpo, step_callback=lambda d, obs: seen.append(d))
    ev.run(2, "2site")
    ev.run(3, "1site", step_callback=lambda d, obs: seen.append(dict(d, one=True)))
    assert [d["step"] for d in seen] == [1, 2, 3, 4, 5]
    assert "max_chi" in seen[0] and seen[-1]["one"]
    assert seen[-1]["t"] == pytest.approx(5 * DT)


def test_measure_records_carry_the_evolver_time(setup):
    psi, mpo = setup
    ev = _evolver(psi, mpo, measure=lambda m: len(m), meas_every=2)
    ev.run(4, "2site")
    rec = ev.run(2, "1site")
    assert [t for t, _ in rec] == [pytest.approx(6 * DT)] and rec[0][1] == N


def test_refusals(setup):
    psi, mpo = setup
    with pytest.raises(ValueError, match="adaptive_dt"):
        TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(adaptive_dt={}))
    with pytest.raises(ValueError, match="fixed MPO"):
        TDVPEvolver(psi, lambda t: mpo, DT, CHI)
    with pytest.raises(ValueError, match="method"):
        _evolver(psi, mpo).run(1, "3site")
    with pytest.raises(ValueError, match="bond cap"):
        TDVPEvolver(psi, mpo, DT).run(1, "2site")


def test_invariant_checks_at_construction_and_rebuild(setup, monkeypatch):
    """The integrators' entry checks, on the state given and on a replaced tensor."""
    from tnkit.invariants import InvariantError
    psi, mpo = setup
    with pytest.raises(InvariantError, match="sites"):
        TDVPEvolver(psi[:-1], mpo, DT, CHI)
    ev = _evolver(psi, mpo)
    ev.run(1, "2site")
    assert ev.mps.center == 0
    ev.mps[0] = np.ones((2, 2, ev.mps[0].shape[2]))     # wrong outer bond
    with pytest.raises(InvariantError, match="outer"):
        ev.run(1, "2site")
    monkeypatch.setenv("TNKIT_CHECK", "off")
    _evolver(psi, mpo).run(1, "2site")


def test_graph_settings_are_checked_at_construction(setup):
    psi, mpo = setup
    with pytest.raises(ValueError, match="graph='on'"):
        TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(backend="numpy", graph="on"))
    with pytest.raises(ValueError, match="cuda_graph"):
        TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(backend="numpy", cuda_graph=object()))
    with pytest.raises(ValueError, match="prefix_graph"):
        TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(backend="numpy", prefix_graph=object(),
                                                    krylov_m=40))


def test_graph_cache_is_built_once_and_carried_across_runs(setup):
    """A prefix-graph cache whose replays are always too short: every site falls back to the
    eager step, bit for bit, and the second run captures nothing new."""
    import types
    from tnkit.backend import get_backend
    from tnkit.prefix_graph_impl import PrefixSiteStepGraphCache

    class Short:
        pool = None

        def run(self, *a, **kw):
            return None

    class Cache(PrefixSiteStepGraphCache):
        made = 0

        def _make(self, sig, direction):
            Cache.made += 1
            return Short()

    psi, mpo = setup
    gc = Cache(get_backend("numpy"), types.SimpleNamespace(), DT, 12)
    ev = TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(backend="numpy", prefix_graph=gc))
    ref = TDVPEvolver(psi, mpo, DT, CHI, TDVPOptions(backend="numpy", graph="off"))
    ev.run(2, "1site")
    made = Cache.made
    ev.run(2, "1site")
    ref.run(4, "1site")
    assert made > 0 and Cache.made == made and gc.stats["short"] > 0
    assert _bytes(ev.mps) == _bytes(ref.mps)


# ------------------------------------------------------------------ per-bond schedule
def test_per_bond_below_the_cap_is_two_site_bit_for_bit(setup):
    psi, mpo = setup
    a, b = _evolver(psi, mpo), _evolver(psi, mpo)
    a.run(5, "12site")
    b.run(5, "2site")
    assert max(int(t.shape[2]) for t in a.mps) < CHI
    assert _bytes(a.mps) == _bytes(b.mps)


def _uniform(n, chi, seed=3):
    """Random MPS with every inner bond equal to chi (chi <= 2 keeps it within the Hilbert space)."""
    rng = np.random.default_rng(seed)
    d = [1] + [chi] * (n - 1) + [1]
    return [rng.normal(size=(d[i], 2, d[i + 1])) + 1j * rng.normal(size=(d[i], 2, d[i + 1]))
            for i in range(n)]


def test_per_bond_with_every_bond_at_the_cap_is_one_site_bit_for_bit(setup):
    _, mpo = setup
    psi = _uniform(N, 2)
    a = TDVPEvolver(psi, mpo, DT, 2, TDVPOptions(backend="numpy"))
    b = TDVPEvolver(psi, mpo, DT, None, TDVPOptions(backend="numpy"))
    seen = []
    a.run(5, "12site", step_callback=lambda d, o: seen.append(d["n_two"]))
    b.run(5, "1site")
    assert seen == [0] * 5
    assert _bytes(a.mps) == _bytes(b.mps)


def test_bonds_above_the_cap_are_truncated_to_it():
    _, m0, _ = dmrg(tfim_mpo(N, 1.0), chi=16, n_sweeps=6, tol=1e-12)
    m0 = [np.asarray(t) for t in m0]
    over = sum(int(t.shape[2]) > 8 for t in m0[:-1])
    assert over > 0
    ev = TDVPEvolver(m0, tfim_mpo(N, 1.5), DT, 8, TDVPOptions(backend="numpy"))
    seen = []
    ev.run(2, "12site", step_callback=lambda d, o: seen.append(d["n_two"]))
    assert seen[0] >= over and max(int(t.shape[2]) for t in ev.mps) <= 8


def test_a_resumed_run_keeps_the_two_site_every_phase(setup):
    psi, mpo = setup
    opts = TDVPOptions(backend="numpy", svd_tol=1e-10)
    full, seen_full = TDVPEvolver(psi, mpo, DT, 4, opts), []
    full.run(7, "12site", step_callback=lambda d, o: seen_full.append(d["n_two"]),
             two_site_every=3)
    first, seen = TDVPEvolver(psi, mpo, DT, 4, opts), []
    first.run(4, "12site", step_callback=lambda d, o: seen.append(d["n_two"]), two_site_every=3)
    opts2 = TDVPOptions(backend="numpy", svd_tol=1e-10, canonicalize=False)
    resumed = TDVPEvolver(list(first.mps), mpo, DT, 4, opts2, step0=4)
    resumed.run(3, "12site", step_callback=lambda d, o: seen.append(d["n_two"]),
                two_site_every=3)
    assert seen == seen_full
    assert _bytes(resumed.mps) == _bytes(full.mps)


def test_per_bond_on_torch_matches_numpy(setup):
    pytest.importorskip("torch")
    psi, mpo = setup
    out = {}
    for backend in ("numpy", "torch-cpu"):
        ev = TDVPEvolver(psi, mpo, DT, 4, TDVPOptions(backend=backend, svd_tol=1e-10))
        seen = []
        ev.run(8, "12site", step_callback=lambda d, o: seen.append(d["n_two"]), two_site_every=3)
        out[backend] = (_host(ev.mps), seen)
    (a, sa), (b, sb) = out["numpy"], out["torch-cpu"]
    assert sa == sb and min(sa) < N - 1
    assert abs(1 - abs(mps_overlap(a, b))) < 1e-12


def _frozen_rank_problem():
    """N = 8 at bonds (2, 4, 8, 8, 8, 4, 2), cap 8: three inner bonds one-site, the four edge
    bonds two-site but full, so nothing is truncated and the step is a pure splitting."""
    n = 8
    _, m0, _ = dmrg(tfim_mpo(n, 1.0), chi=8, n_sweeps=8, tol=1e-12)
    m0 = [np.asarray(t) for t in m0]
    assert [t.shape[2] for t in m0] == [2, 4, 8, 8, 8, 4, 2, 1]
    return m0, tfim_mpo(n, 1.5)


def _dense(M):
    v = M[0]
    for A in M[1:]:
        v = np.tensordot(v, A, axes=([-1], [0]))
    return v.reshape(-1)


def test_per_bond_step_is_time_reversible():
    m0, W = _frozen_rank_problem()
    fwd = TDVPEvolver(m0, W, 0.1, 8, TDVPOptions(backend="numpy"))
    seen = []
    fwd.run(1, "12site", step_callback=lambda d, o: seen.append(d["n_two"]))
    back = TDVPEvolver(list(fwd.mps), W, -0.1, 8, TDVPOptions(backend="numpy", canonicalize=False))
    back.run(1, "12site")
    assert seen == [4]
    a, b = _dense(m0), _dense(list(back.mps))
    assert abs(1 - abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))) < 1e-12


def test_per_bond_step_is_second_order():
    """Global error at t = 0.4 falls 4x per halving of dt (3.36, 3.76, 3.93 from dt = 0.1, 0.05,
    0.025, identical to 1TDVP's), so dt = 0.025 is close enough to the asymptote."""
    m0, W = _frozen_rank_problem()
    out = []
    for k in (1, 2, 4):
        ev = TDVPEvolver(m0, W, 0.025 / k, 8, TDVPOptions(backend="numpy"))
        ev.run(16 * k, "12site")
        v = _dense(list(ev.mps))
        out.append(v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))])))
    ratio = np.linalg.norm(out[0] - out[1]) / np.linalg.norm(out[1] - out[2])
    assert 3.8 < ratio < 4.2


def test_two_site_every_k_is_a_two_site_step_every_k(setup):
    psi, mpo = setup
    a = TDVPEvolver(psi, mpo, DT, 4, TDVPOptions(backend="numpy", svd_tol=1e-10))
    b = TDVPEvolver(psi, mpo, DT, 4, TDVPOptions(backend="numpy", svd_tol=1e-10))
    seen = []
    a.run(6, "12site", step_callback=lambda d, o: seen.append(d["n_two"]), two_site_every=3)
    for _ in range(2):
        b.run(2, "12site")
        b.run(1, "2site")
    assert _bytes(a.mps) == _bytes(b.mps)
    assert seen[2] == seen[5] == N - 1 and min(seen) < N - 1


def test_per_bond_is_exact_at_full_bond_dimension():
    """N = 6 at bond dimension (2, 4, 8, 4, 2) is the exact state; TDVP is then exact, so the
    mixed schedule (cap 8: the middle bond one-site, the rest two-site) must match the dense
    evolution to round-off."""
    cap = 8
    import scipy.linalg as sl
    n = 6
    _, m0, _ = dmrg(tfim_mpo(n, 1.0), chi=8, n_sweeps=8, tol=1e-14)
    m0, W = [np.asarray(t) for t in m0], tfim_mpo(n, 1.5)
    assert [t.shape[2] for t in m0] == [2, 4, 8, 4, 2, 1]

    def dense(M):
        v = M[0]
        for A in M[1:]:
            v = np.tensordot(v, A, axes=([-1], [0]))
        return v.reshape(-1)

    H = W[0][0]
    for A in W[1:]:
        H = np.einsum("wab,wvcd->vacbd", H, A)
        H = H.reshape(H.shape[0], H.shape[1] * H.shape[2], -1)
    ev = TDVPEvolver(m0, W, DT, cap, TDVPOptions(backend="numpy"))
    seen = []
    ev.run(20, "12site", step_callback=lambda d, o: seen.append(d["n_two"]))
    exact = sl.expm(-1j * H[0] * DT * 20) @ dense(m0)
    assert 0 < seen[0] < n - 1
    assert abs(1 - abs(np.vdot(exact, dense(list(ev.mps))))) < 1e-12


def test_per_bond_refusals(setup):
    psi, mpo = setup
    with pytest.raises(ValueError, match="12site"):
        _evolver(psi, mpo).run(1, "2site", two_site_every=10)
    with pytest.raises(ValueError, match=">= 0"):
        _evolver(psi, mpo).run(1, "12site", two_site_every=-1)


@pytest.mark.parametrize("method", ["1site", "2site"])
def test_step_dicts_match_the_integrators(setup, method):
    """Same keys and same non-timing values as tdvp*_evolve's step_callback."""
    psi, mpo = setup
    a, b = [], []
    _evolver(psi, mpo).run(4, method, step_callback=lambda d, o: a.append(d))
    _call(method, psi, mpo, 4, step_callback=lambda d, o: b.append(d))
    timing = {"expm_s", "trunc_s", "env_s"}
    assert [set(d) for d in a] == [set(d) for d in b]
    for x, y in zip(a, b):
        for k in set(x) - timing:
            assert x[k] == pytest.approx(y[k], rel=1e-15, abs=0), k
