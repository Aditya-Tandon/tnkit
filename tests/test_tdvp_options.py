"""`TDVPOptions.krylov_m` and the per-step callback of both TDVP integrators."""
import numpy as np
import pytest

import tnkit.tdvp as T
from tnkit.api import dmrg, tfim_mpo
from tnkit.tdvp import TDVPOptions, tdvp1_evolve, tdvp2_evolve

N, CHI, DT, STEPS = 10, 8, 0.05, 4


@pytest.fixture(scope="module")
def setup():
    _, mps0, _ = dmrg(tfim_mpo(N, 1.0), chi=CHI, n_sweeps=6, tol=1e-12)
    return [np.asarray(t) for t in mps0], tfim_mpo(N, 1.5)


def _run(which, mps0, mpo, **kw):
    o = TDVPOptions(backend="numpy", **kw)
    if which == 1:
        return tdvp1_evolve(list(mps0), mpo, DT, STEPS, o)
    return tdvp2_evolve(list(mps0), mpo, CHI, DT, STEPS, o)


def _bytes(mps):
    return [np.asarray(t).tobytes() for t in mps]


@pytest.mark.parametrize("which", [1, 2])
@pytest.mark.parametrize("m", [4, 30, 40])
def test_krylov_m_equals_capping_lanczos_expm(setup, which, m, monkeypatch):
    """krylov_m=m gives the same bits as wrapping `lanczos_expm` to force m (the old patch)."""
    mps0, mpo = setup
    got, _ = _run(which, mps0, mpo, krylov_m=m)
    orig = T.lanczos_expm

    def capped(matvec, v, tau, ops, **kw):
        kw["m"] = m
        return orig(matvec, v, tau, ops, **kw)

    monkeypatch.setattr(T, "lanczos_expm", capped)
    ref, _ = _run(which, mps0, mpo)
    assert _bytes(got) == _bytes(ref)


@pytest.mark.parametrize("which", [1, 2])
def test_default_krylov_m_is_unchanged(setup, which):
    mps0, mpo = setup
    a, _ = _run(which, mps0, mpo)
    b, _ = _run(which, mps0, mpo, krylov_m=30)
    assert _bytes(a) == _bytes(b)


@pytest.mark.parametrize("which", [1, 2])
def test_step_callback_reports_every_step_and_changes_nothing(setup, which):
    mps0, mpo = setup
    seen = []
    meas = lambda m: float(np.abs(m[0]).sum())
    got, rec = _run(which, mps0, mpo, krylov_m=5, measure=meas, meas_every=2,
                    step_callback=lambda d, obs: seen.append((d, obs)))
    ref, rec_ref = _run(which, mps0, mpo, krylov_m=5, measure=meas, meas_every=2)
    assert _bytes(got) == _bytes(ref) and rec == rec_ref
    assert [d["step"] for d, _ in seen] == list(range(1, STEPS + 1))
    assert all(1 <= d["krylov_max"] <= 5 and d["dt"] == DT for d, _ in seen)
    assert [obs is not None for _, obs in seen] == [False, True, False, True]
    assert seen[-1][0]["t"] == pytest.approx(STEPS * DT)


def test_tdvp1_step_callback_refuses_adaptive_dt(setup):
    mps0, mpo = setup
    with pytest.raises(ValueError, match="adaptive_dt"):
        _run(1, mps0, mpo, step_callback=lambda d, o: None, adaptive_dt={})


@pytest.mark.parametrize("which", [1, 2])
def test_krylov_m_must_be_positive(setup, which):
    mps0, mpo = setup
    with pytest.raises(ValueError, match="krylov_m"):
        _run(which, mps0, mpo, krylov_m=0)


def test_graph_capture_yields_to_a_krylov_cap(setup, monkeypatch):
    """The prefix graph replays solves of `graph_m_max` steps, so it cannot honour `krylov_m`: 'auto' falls
    back to eager and gives the same bits, 'on' and an explicit cache refuse."""
    mps0, mpo = setup
    ref, _ = _run(1, mps0, mpo, krylov_m=7)
    monkeypatch.setattr(T, "resolve_graph_mode", lambda graph, backend: "capture")
    got, _ = _run(1, mps0, mpo, krylov_m=7)
    assert _bytes(got) == _bytes(ref)
    with pytest.raises(ValueError, match="krylov_m"):
        _run(1, mps0, mpo, krylov_m=7, graph="on")
    with pytest.raises(ValueError, match="krylov_m"):
        _run(1, mps0, mpo, krylov_m=7, prefix_graph=object())
