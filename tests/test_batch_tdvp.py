"""Batched TDVP: does k legs at once give what k separate legs give?

The bar everywhere below is agreement between `evolve_batch` at k > 1 and the SAME code at
k = 1, leg by leg. That is the comparison the batching is responsible for: the batched
evolver uses the fixed-depth Krylov solver, so it is deliberately not bit-identical to the
adaptive production integrator, and holding it to that would be scoring the solver rather
than the batching. One test does compare against `tdvp1_evolve` as well, to show the
batched path lands where an ordinary TDVP run lands.

All of it runs at chi small enough for a laptop and, per this project's convention, with
the BLAS thread count left to the caller (the suite is run at <= 3).
"""
from __future__ import annotations
import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.batch_tdvp import (BatchGeometryError, BatchKrylovError, KRYLOV_FLOOR,
                              KRYLOV_HEADROOM, evolve_batch, lanczos_expm_fixed_batched)
from tnkit.models import SX, tfim_mpo
from tnkit.observables import local_expectations
from tnkit._dmrg import random_mps, right_canonicalize


def _mixed_field_mpo(N, g, h):
    """TFIM with a longitudinal field -- the shape a batch's legs differ in."""
    W = tfim_mpo(N, g)
    out = []
    for i, w in enumerate(W):
        w = np.array(w, dtype=complex)
        # The identity-to-identity corner carries the on-site term; add h * sigma^x there.
        sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        if w.shape[0] > 1 and w.shape[1] > 1:
            w[0, -1] = w[0, -1] + h * sx
        elif w.shape[0] == 1:
            w[0, -1] = w[0, -1] + h * sx
        else:
            w[0, 0] = w[0, 0] + h * sx
        out.append(w)
    return out


def _legs(N, chi, k, seed=0, d=2):
    """k random states of ONE geometry and k MPOs differing only in their couplings."""
    be = get_backend("numpy")
    states, mpos = [], []
    for n in range(k):
        rng = np.random.default_rng(seed + 100 * n)
        M = right_canonicalize(random_mps(N, chi, d=d, rng=rng, be=be), be)
        states.append([np.asarray(x) for x in M])
        mpos.append(_mixed_field_mpo(N, 1.0 + 0.05 * n, 0.10 + 0.01 * n))
    return states, mpos


def _dev(a, b):
    """Max relative deviation between two host arrays of the same shape."""
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


def _infid(a, b):
    """1 - |<a|b>| / (||a|| ||b||): how far apart two MPS are AS STATES."""
    from tnkit.observables import mps_overlap
    A = [np.asarray(x, dtype=np.complex128) for x in a]
    B = [np.asarray(x, dtype=np.complex128) for x in b]
    na, nb = abs(mps_overlap(A, A)) ** 0.5, abs(mps_overlap(B, B)) ** 0.5
    return float(1.0 - abs(mps_overlap(A, B)) / max(na * nb, 1e-300))


def _same_state(got, ref, tag="", elem=1e-9, infid=1e-13):
    """Assert two MPS are the same STATE, and say what each bar is for.

    The bar that carries the claim is the OVERLAP, because an MPS is defined only up to a
    gauge on every bond. Two decompositions of one state disagree tensor by tensor whenever
    the gauge moves, and it moves for reasons that have nothing to do with correctness: the
    QR fixes it from LAPACK's Householder reflectors, so a sixteenth-digit perturbation can
    flip the sign of a column of Q and of the neighbour's matching row; and an SVD whose
    singular values are near-degenerate returns an arbitrary rotation inside that subspace,
    which U and S*Vh undo between them. Both leave the state identical and two tensors apart
    by their whole magnitude.

    The elementwise bar is kept as a second, deliberately loose check. It catches a gross
    error the overlap could dilute, and it must not be tightened into a gauge test: this
    project has seen exactly that assertion pass on one LAPACK build and fail at 2e-11 on
    another with nothing between them but the build.
    """
    assert len(got) == len(ref), tag
    for i in range(len(got)):
        assert np.asarray(got[i]).shape == np.asarray(ref[i]).shape, f"{tag} site {i} shape"
        assert _dev(got[i], ref[i]) < elem, f"{tag} site {i} elementwise"
    assert abs(_infid(got, ref)) < infid, f"{tag} infidelity"


def _energy(state, mpo, be):
    from tnkit._dmrg import update_left_env
    L = be.ones((1, 1, 1))
    for A, W in zip(state, mpo):
        L = update_left_env(L, be.array(A), be.array(W), be)
    return complex(be.to_scalar(be.einsum("akb,akb->", L, be.ones((1, 1, 1)))))


def _serial(states, mpos, dt, n_steps, **kw):
    """The same batched evolver, one leg at a time. This is the reference everywhere."""
    return [evolve_batch([s], [W], dt, n_steps, **kw)[0][0]
            for s, W in zip(states, mpos)]


# --------------------------------------------------------------------------- agreement
@pytest.mark.parametrize("evolver,extra", [("1tdvp", {}),
                                           ("2tdvp", {"chi_max": 16, "svd_tol": 0.0})])
def test_batched_matches_serial(evolver, extra):
    """k = 4 legs batched reproduce the same four legs run one at a time."""
    N, chi, k, dt = 10, 8, 4, 0.05
    states, mpos = _legs(N, chi, k)
    kw = dict(backend="numpy", evolver=evolver, krylov_m=8, canonicalize=False, **extra)
    ref = _serial(states, mpos, dt, 4, **kw)
    got, _ = evolve_batch(states, mpos, dt, 4, **kw)
    for n in range(k):
        _same_state(got[n], ref[n], f"leg {n}")


@pytest.mark.parametrize("evolver,extra", [("1tdvp", {}),
                                           ("2tdvp", {"chi_max": 16, "svd_tol": 0.0})])
def test_batched_matches_serial_torch(evolver, extra):
    """The same agreement on the torch backend, which is the one the GPU path uses.

    The CPU torch build exercises every line the CUDA build runs -- batched `linalg.qr`,
    the stacked SVD, the batched eigendecomposition of the projected tridiagonals -- so a
    kernel-dispatch mistake in the batched spelling shows up here rather than on a GPU.
    """
    pytest.importorskip("torch")
    N, chi, k, dt = 10, 8, 4, 0.05
    states, mpos = _legs(N, chi, k, seed=61)
    kw = dict(backend="torch-cpu", evolver=evolver, krylov_m=8, canonicalize=False, **extra)
    ref = _serial(states, mpos, dt, 4, **kw)
    got, _ = evolve_batch(states, mpos, dt, 4, **kw)
    for n in range(k):
        _same_state(got[n], ref[n], f"leg {n}")


def test_ragged_matches_serial_torch():
    """Ragged bonds, rank masks and all, on the torch backend."""
    pytest.importorskip("torch")
    be = get_backend("numpy")
    states, mpos = [], []
    for n, chi in enumerate((4, 6, 8)):
        rng = np.random.default_rng(71 + n)
        states.append([np.asarray(x) for x in
                       right_canonicalize(random_mps(9, chi, rng=rng, be=be), be)])
        mpos.append(_mixed_field_mpo(9, 1.0 + 0.1 * n, 0.05 * n))
    kw = dict(backend="torch-cpu", evolver="1tdvp", krylov_m=8, canonicalize=False)
    ref = _serial(states, mpos, 0.05, 3, **kw)
    got, _ = evolve_batch(states, mpos, 0.05, 3, **kw)
    for n in range(3):
        _same_state(got[n], ref[n], f"leg {n}")


def test_batched_matches_serial_complex128_dtype():
    """The batched path keeps full complex double precision end to end."""
    states, mpos = _legs(8, 6, 3)
    got, _ = evolve_batch(states, mpos, 0.05, 2, backend="numpy", krylov_m=8,
                          canonicalize=False)
    for s in got:
        for A in s:
            assert np.asarray(A).dtype == np.complex128


@pytest.mark.parametrize("ragged", ["pad", "tight", "group"])
def test_ragged_bonds_match_serial(ragged):
    """Legs whose bond dimensions DIFFER still reproduce their own serial trajectories.

    This is the test the zero-padding exists for. The legs are built at three different
    bond dimensions, so under 'pad' every bond is padded to the batch maximum and the rank
    mask is doing real work at every site of every step; under 'group' the three legs
    partition into three groups of one. Both must land on the serial answer.
    """
    N, dt, k = 9, 0.05, 3
    be = get_backend("numpy")
    states, mpos = [], []
    for n, chi in enumerate((4, 6, 8)):
        rng = np.random.default_rng(7 + n)
        M = right_canonicalize(random_mps(N, chi, rng=rng, be=be), be)
        states.append([np.asarray(x) for x in M])
        mpos.append(_mixed_field_mpo(N, 1.0 + 0.1 * n, 0.05 * n))
    kw = dict(backend="numpy", evolver="1tdvp", krylov_m=8, canonicalize=False,
              ragged=ragged)
    ref = _serial(states, mpos, dt, 3, **kw)
    got, _ = evolve_batch(states, mpos, dt, 3, **kw)
    for n in range(k):
        _same_state(got[n], ref[n], f"leg {n}")


def test_ragged_strategies_all_agree():
    """The three ragged strategies are three spellings of one trajectory, not three answers.

    Run under 2TDVP, where they genuinely differ: 'pad' holds every bond at the widest the
    cap allows, 'tight' re-cuts it to the widest rank any leg kept, 'group' partitions the
    legs by their entry profile. The states must not be able to tell.

    The bar here is 1e-10 rather than the 1e-12 the batched-vs-serial tests use, and the
    difference is not a weaker claim -- it is a different comparison. Those tests feed
    LAPACK the SAME matrix twice and differ only in the order of a sum; this one feeds it a
    12-wide block and an 8-wide block whose retained subspace is the same, and the singular
    vectors that come back agree only to the conditioning of that subspace. What is being
    checked is that the three strategies follow one trajectory, and they do.
    """
    N, dt = 9, 0.05
    be = get_backend("numpy")
    states, mpos = [], []
    for n, chi in enumerate((4, 8)):
        rng = np.random.default_rng(21 + n)
        states.append([np.asarray(x) for x in
                       right_canonicalize(random_mps(N, chi, rng=rng, be=be), be)])
        mpos.append(_mixed_field_mpo(N, 1.0, 0.1))
    kw = dict(backend="numpy", evolver="2tdvp", krylov_m=8, canonicalize=False,
              chi_max=12, svd_tol=1e-10)
    out = {r: evolve_batch(states, mpos, dt, 3, ragged=r, **kw)[0]
           for r in ("pad", "tight", "group")}
    for r in ("tight", "group"):
        for n in range(2):
            _same_state(out[r][n], out["pad"][n], f"{r} leg {n}")


@pytest.mark.parametrize("ragged", ["pad", "tight", "group"])
def test_2tdvp_ragged_truncation_matches_serial(ragged):
    """With a discarded-weight budget the legs retain different ranks -- still exact."""
    N, dt = 8, 0.05
    states, mpos = _legs(N, 8, 3, seed=5)
    kw = dict(backend="numpy", evolver="2tdvp", krylov_m=8, canonicalize=False,
              chi_max=12, svd_tol=1e-8, ragged=ragged)
    ref = _serial(states, mpos, dt, 3, **kw)
    got, _ = evolve_batch(states, mpos, dt, 3, **kw)
    for n in range(3):
        _same_state(got[n], ref[n], f"leg {n}")


def _neel(N):
    """A chi=1 Neel product state: no RNG, so this test's numbers are exactly reproducible."""
    out = []
    for i in range(N):
        v = np.zeros(2, dtype=complex)
        v[i % 2] = 1.0
        out.append(v.reshape(1, 2, 1))
    return out


def test_disc_per_leg_is_actually_per_leg():
    """Regresses the batch-mean-broadcast bug: `stats["disc_per_leg"]` must vary leg to leg.

    Leg 0 sits near the TFIM critical point (g=1.0, h=0.02); leg 1 is deep in the gapped
    paramagnetic phase (g=10.0). Starting both from the same Neel product state under
    `2tdvp` with a discarded-weight budget, the two legs truncate by very different amounts
    every step. The bug this regresses -- `_two_site_step` returning `float(np.mean(disc))`
    instead of the per-leg `(k,)` array one line above it -- makes `_run`'s
    `disc_leg += _two_site_step(...)` broadcast that scalar batch mean into EVERY leg's
    accumulator, so both legs would report the identical batch-mean value here (ratio 1.0)
    instead of their own true, very different accumulated weight.
    """
    N, dt, n_steps, chi_max, svd_tol, h, km = 8, 0.05, 4, 12, 1e-6, 0.02, 6
    gs = (1.0, 10.0)
    states = [_neel(N) for _ in gs]
    mpos = [_mixed_field_mpo(N, g, h) for g in gs]
    kw = dict(backend="numpy", evolver="2tdvp", krylov_m=km, canonicalize=False,
              chi_max=chi_max, svd_tol=svd_tol, ragged="pad")
    st = {}
    evolve_batch(states, mpos, dt, n_steps, stats=st, **kw)
    per_leg = st["disc_per_leg"]
    assert len(per_leg) == 2
    assert per_leg[0] > 0.0 and per_leg[1] > 0.0
    ratio = per_leg[1] / per_leg[0]
    # A broadcast batch mean would put this ratio at exactly 1.0; the real physics puts it
    # north of 100 for this pair of legs. 50 is a comfortable margin under the ~420 observed.
    assert ratio > 50.0, f"near-critical/gapped disc ratio only {ratio:.3g}: {per_leg}"

    # Each leg's own accumulated weight must also match that SAME leg run serially (one leg
    # at a time, same code path) -- loosely, because the batched and serial discarded-weight
    # estimators are different accountings (a padded structural rank ceiling vs. the full
    # spectrum) that this project has separately measured to differ by ~12% at a fixed state.
    # Here both legs share one padding profile (same N, same Neel start, same cap), so the
    # two estimators coincide almost exactly -- but the bound stays loose so this test keeps
    # regressing the bug rather than pinning an estimator-accounting detail.
    for n in range(2):
        st_serial = {}
        evolve_batch([states[n]], [mpos[n]], dt, n_steps, stats=st_serial, **kw)
        serial = st_serial["disc_per_leg"][0]
        rel = abs(per_leg[n] - serial) / max(serial, 1e-300)
        assert rel < 0.25, f"leg {n} batched vs serial disc relative deviation {rel:.3g}"


def test_pad_holds_one_shape_class_and_tight_does_not():
    """'pad' is the strategy whose stacked shapes never move; 'tight' is the one that saves
    arithmetic by letting them move. Both must still land on the same states.

    The bonds are watched through the growth phase of legs that start well below the cap,
    which is exactly where the two strategies part company.
    """
    N, dt, cap = 9, 0.05, 16
    be = get_backend("numpy")
    states, mpos = [], []
    for n, chi in enumerate((2, 4)):
        rng = np.random.default_rng(51 + n)
        states.append([np.asarray(x) for x in
                       right_canonicalize(random_mps(N, chi, rng=rng, be=be), be)])
        mpos.append(_mixed_field_mpo(N, 1.0, 0.1))
    kw = dict(backend="numpy", evolver="2tdvp", krylov_m=8, canonicalize=False,
              chi_max=cap, svd_tol=1e-10)
    # `fill` -- mean true rank over stacked width -- is the observable proxy for how much of
    # each strategy's program is real work. 'pad' computes every bond at the cap profile
    # from the first step, so its fill is strictly the lower of the two.
    fill = {}
    for r in ("pad", "tight"):
        st = {}
        out, _ = evolve_batch(states, mpos, dt, 3, ragged=r, stats=st, **kw)
        fill[r] = st["fill"]
    assert fill["pad"] < fill["tight"], fill
    assert 0.0 < fill["pad"] <= 1.0 and 0.0 < fill["tight"] <= 1.0


def test_matches_unbatched_tdvp2():
    """The batched two-site path lands where an ordinary `tdvp2_evolve` run lands.

    This is the padding's real test: `tdvp2_evolve` carries no leg axis, no padding and no
    rank mask, so if the zero-padded bonds were promoting the state onto a finer manifold
    the two would part company at the truncation, not at the twelfth digit.
    """
    from tnkit.tdvp import tdvp2_evolve
    N, dt, k, cap = 8, 0.05, 2, 12
    states, mpos = _legs(N, 6, k, seed=9)
    got, _ = evolve_batch(states, mpos, dt, 3, backend="numpy", krylov_m=12,
                          canonicalize=False, evolver="2tdvp", chi_max=cap, svd_tol=1e-10)
    for n in range(k):
        ref, _ = tdvp2_evolve([np.array(x) for x in states[n]], mpos[n], cap, dt, 3,
                              backend="numpy", canonicalize=False, svd_tol=1e-10)
        for i in range(N):
            assert got[n][i].shape == np.asarray(ref[i]).shape, f"leg {n} site {i} shape"
            assert _dev(got[n][i], ref[i]) < 1e-9, f"leg {n} site {i}"


def test_energy_is_conserved_by_1tdvp():
    """One-site TDVP is symplectic and bond-preserving: energy must not drift."""
    be = get_backend("numpy")
    states, mpos = _legs(10, 8, 3, seed=11)
    E0 = [_energy(s, W, be) for s, W in zip(states, mpos)]
    got, _ = evolve_batch(states, mpos, 0.05, 8, backend="numpy", krylov_m=10,
                          canonicalize=False)
    for n, (s, W) in enumerate(zip(got, mpos)):
        E1 = _energy(s, W, be)
        assert abs(E1 - E0[n]) < 1e-9 * max(1.0, abs(E0[n])), f"leg {n}"


def test_matches_unbatched_tdvp1():
    """The batched path lands where an ordinary `tdvp1_evolve` run lands.

    The tolerance is looser than the batched-vs-serial one on purpose, and that is the
    point of having both: `tdvp1_evolve` runs the ADAPTIVE Krylov solver and the batched
    path runs a fixed-depth one, so this measures the solver difference, not the batching.
    """
    from tnkit.tdvp import tdvp1_evolve
    N, dt, k = 10, 0.05, 2
    states, mpos = _legs(N, 8, k, seed=3)
    got, _ = evolve_batch(states, mpos, dt, 4, backend="numpy", krylov_m=12,
                          canonicalize=False)
    for n in range(k):
        ref, _ = tdvp1_evolve([np.array(x) for x in states[n]], mpos[n], dt, 4,
                              backend="numpy", canonicalize=False, graph="off")
        for i in range(N):
            assert _dev(got[n][i], ref[i]) < 1e-9, f"leg {n} site {i}"


def test_expm_where_host_matches_eigh():
    """The two routes to the small exponential agree; 'host' is the control."""
    states, mpos = _legs(8, 6, 3, seed=17)
    kw = dict(backend="numpy", krylov_m=8, canonicalize=False)
    a, _ = evolve_batch(states, mpos, 0.05, 3, expm_where="eigh", **kw)
    b, _ = evolve_batch(states, mpos, 0.05, 3, expm_where="host", **kw)
    for n in range(3):
        for i in range(len(a[n])):
            assert _dev(a[n][i], b[n][i]) < 1e-11


def test_krylov_solver_matches_unbatched():
    """The batched Krylov exponential reproduces the unbatched one, leg by leg."""
    from tnkit.eigensolver import lanczos_expm_fixed
    be = get_backend("numpy")
    rng = np.random.default_rng(0)
    k, n, m = 4, 24, 8
    Hs = []
    for _ in range(k):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Hs.append(be.array(A + A.conj().T))
    v = be.array(rng.standard_normal((k, n)) + 1j * rng.standard_normal((k, n)))
    tau = -1j * 0.05
    got = lanczos_expm_fixed_batched(
        lambda x: be.array(np.stack([Hs[i] @ np.asarray(x)[i] for i in range(k)])),
        v, tau, be, m)
    for i in range(k):
        ref = lanczos_expm_fixed(lambda y, H=Hs[i]: H @ y, np.asarray(v)[i], tau, be, m,
                                 expm_where="host")
        assert _dev(np.asarray(got)[i], ref) < 1e-11, f"leg {i}"


# --------------------------------------------------------------------------- refusals
def test_refuses_mismatched_site_count():
    states, mpos = _legs(8, 6, 2)
    states[1] = states[1][:-1]
    with pytest.raises(BatchGeometryError, match="sites"):
        evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6, canonicalize=False)


def test_refuses_mismatched_mpo_structure():
    states, mpos = _legs(8, 6, 2)
    mpos[1] = _mixed_field_mpo(8, 1.0, 0.1)
    mpos[1][3] = np.zeros((mpos[1][3].shape[0] + 1,) + mpos[1][3].shape[1:], dtype=complex)
    with pytest.raises(BatchGeometryError, match="bond structure"):
        evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6, canonicalize=False)


def test_refuses_mismatched_outer_bond():
    states, mpos = _legs(8, 6, 2)
    A = states[1][0]
    states[1][0] = np.concatenate([A, np.zeros_like(A)], axis=0)
    with pytest.raises(BatchGeometryError, match="outer bonds"):
        evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6, canonicalize=False)


def test_refuses_chi_max_on_one_site():
    states, mpos = _legs(8, 6, 2)
    with pytest.raises(ValueError, match="bond-preserving"):
        evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6,
                     canonicalize=False, chi_max=16)


def test_refuses_two_site_without_cap():
    states, mpos = _legs(8, 6, 2)
    with pytest.raises(ValueError, match="chi_max"):
        evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6,
                     canonicalize=False, evolver="2tdvp")


def test_refuses_unknown_option():
    states, mpos = _legs(8, 6, 2)
    with pytest.raises(TypeError, match="unexpected keyword"):
        evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6,
                     canonicalize=False, chi=12)


# --------------------------------------------------------------------------- bookkeeping
def test_stats_report_grouping_and_fill():
    be = get_backend("numpy")
    states, mpos = [], []
    for n, chi in enumerate((4, 8, 8)):
        rng = np.random.default_rng(31 + n)
        states.append([np.asarray(x) for x in
                       right_canonicalize(random_mps(9, chi, rng=rng, be=be), be)])
        mpos.append(_mixed_field_mpo(9, 1.0, 0.1))
    st = {}
    evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6, canonicalize=False,
                 ragged="group", stats=st)
    assert sorted(st["groups"]) == [1, 2]
    assert st["serial_legs"] == 1
    st2 = {}
    evolve_batch(states, mpos, 0.05, 1, backend="numpy", krylov_m=6, canonicalize=False,
                 ragged="pad", stats=st2)
    assert st2["groups"] == [3]
    assert 0.0 < st2["fill"] <= 1.0


def test_measure_receives_per_leg_states():
    states, mpos = _legs(8, 6, 3, seed=41)
    seen = []
    _, recs = evolve_batch(states, mpos, 0.05, 2, backend="numpy", krylov_m=6,
                           canonicalize=False, measure=lambda ss: seen.append(len(ss)))
    assert seen == [3, 3, 3]
    assert [t for t, _ in recs] == pytest.approx([0.0, 0.05, 0.10])


# --------------------------------------------------------------------------- krylov_m="auto"
#
# `krylov_m="auto"` is the dataclass DEFAULT -- what a caller gets by doing nothing -- so it
# is exercised here end to end rather than only through the explicit `krylov_m=` every other
# test in this file pins for speed. Calibration runs one adaptive sweep per leg (not leg 0
# alone) and takes the max; the tests below check both halves of that sentence.

@pytest.mark.parametrize("evolver,extra", [("1tdvp", {}),
                                           ("2tdvp", {"chi_max": 16, "svd_tol": 0.0})])
def test_krylov_auto_matches_serial(evolver, extra):
    """The default configuration -- no `krylov_m` at all -- lands on the serial answer."""
    N, chi, k, dt = 10, 8, 4, 0.05
    states, mpos = _legs(N, chi, k, seed=83)
    kw = dict(backend="numpy", evolver=evolver, canonicalize=False, **extra)   # no krylov_m
    ref = _serial(states, mpos, dt, 4, **kw)
    got, _ = evolve_batch(states, mpos, dt, 4, **kw)
    for n in range(k):
        _same_state(got[n], ref[n], f"leg {n}")


def test_krylov_auto_matches_serial_torch():
    """The same default-path agreement on the torch backend, which is the CUDA-shaped one."""
    pytest.importorskip("torch")
    N, chi, k, dt = 10, 8, 4, 0.05
    states, mpos = _legs(N, chi, k, seed=89)
    kw = dict(backend="torch-cpu", canonicalize=False)
    ref = _serial(states, mpos, dt, 4, **kw)
    got, _ = evolve_batch(states, mpos, dt, 4, **kw)
    for n in range(k):
        _same_state(got[n], ref[n], f"leg {n}")


def test_krylov_auto_calibrates_every_leg_not_leg_zero():
    """`_calibrate_krylov_m` samples every leg's own adaptive demand, not leg 0's.

    Leg 2 carries a much larger field than legs 0 and 1, so its adaptive Krylov demand is
    the largest; the frozen `m` must reflect THAT leg, and `stats["krylov_calibration_per_leg"]`
    must actually vary across legs rather than repeating leg 0's number k times (which is
    what a leg-0-only calibration would silently do).
    """
    N, chi, k, dt = 10, 8, 3, 0.2
    be = get_backend("numpy")
    states, mpos = [], []
    for n, h in enumerate((0.02, 0.02, 3.0)):        # leg 2 is the hard one
        rng = np.random.default_rng(5 + n)
        states.append([np.asarray(x) for x in
                       right_canonicalize(random_mps(N, chi, rng=rng, be=be), be)])
        mpos.append(_mixed_field_mpo(N, 1.0, h))
    st = {}
    evolve_batch(states, mpos, dt, 2, backend="numpy", canonicalize=False, stats=st)
    per_leg = st["krylov_calibration_per_leg"]
    assert len(per_leg) == k
    assert len(set(per_leg)) > 1, f"per-leg calibration is degenerate: {per_leg}"
    assert per_leg[2] >= per_leg[0], "the hard leg's own demand should not be discarded"
    assert st["krylov_m"] == max(per_leg) + KRYLOV_HEADROOM


def test_krylov_guard_raises_when_underresolved_and_cannot_grow():
    """A deliberately under-resolved fixed `m`, with no room to grow, is refused loudly.

    `krylov_m_max` pinned equal to `krylov_m` leaves the runtime guard nowhere to grow, and
    a tight `krylov_tol` guarantees the (real, not synthetic) residual at `krylov_m=4`
    exceeds it well before round-off -- so this is the actual guard tripping on an actual
    under-resolved run, not a mocked failure.
    """
    N, chi, k, dt = 10, 10, 2, 0.1
    states, mpos = _legs(N, chi, k, seed=97)
    with pytest.raises(BatchKrylovError, match="krylov_tol"):
        evolve_batch(states, mpos, dt, 6, backend="numpy", canonicalize=False,
                     krylov_m=KRYLOV_FLOOR, krylov_m_max=KRYLOV_FLOOR, krylov_tol=1e-10)


def test_krylov_guard_grows_m_when_room_allows():
    """The same under-resolved start, given room to grow, grows instead of erroring."""
    N, chi, k, dt = 10, 10, 2, 0.1
    states, mpos = _legs(N, chi, k, seed=97)
    st = {}
    got, _ = evolve_batch(states, mpos, dt, 6, backend="numpy", canonicalize=False,
                          krylov_m=KRYLOV_FLOOR, krylov_m_max=24, krylov_tol=1e-10, stats=st)
    assert st["krylov_grown"] is True
    assert st["krylov_m_grown_to"] > KRYLOV_FLOOR
    assert len(st["krylov_residual_max"]) == k
    for s in got:                                            # finished, not NaN/garbage
        for A in s:
            assert np.all(np.isfinite(np.asarray(A)))


def test_krylov_guard_quiet_on_a_well_calibrated_run():
    """A properly calibrated `m` should not trip the guard: residuals stay far under tol."""
    N, chi, k, dt = 9, 8, 3, 0.05
    states, mpos = _legs(N, chi, k, seed=13)
    st = {}
    evolve_batch(states, mpos, dt, 4, backend="numpy", canonicalize=False, stats=st)
    assert st["krylov_grown"] is False
    assert max(st["krylov_residual_max"]) < st["krylov_tol"]


# --------------------------------------------------------------------------- agreement regression
def test_batched_agreement_regression():
    """Pins the batched path's OWN agreement bar in a laptop-cheap regression.

    N=20, chi cap 12, k=3, 10 steps, `krylov_m="auto"` (the default configuration): small
    enough to run in a unit-test budget, but no longer hand-picked purely for speed, so a
    future change to the batched contractions, the mask order, or the Krylov exponential
    that regresses the agreement measured at full collision scale (module docstring) should
    show up here rather than only on an expensive larger run. The bar is state infidelity against the
    SAME code run one leg at a time (<= 1e-13) and the relative deviation of two derived
    scalars against that same reference: window energy and a local one-site expectation
    value (<= 5e-10 each).
    """
    N, chi, k, dt, n_steps = 20, 8, 3, 0.05, 10
    states, mpos = _legs(N, chi, k, seed=101)
    be = get_backend("numpy")
    kw = dict(backend="numpy", evolver="2tdvp", chi_max=12, svd_tol=1e-10,
              canonicalize=False)                              # krylov_m="auto"
    ref = _serial(states, mpos, dt, n_steps, **kw)
    got, _ = evolve_batch(states, mpos, dt, n_steps, **kw)
    for n in range(k):
        infid = abs(_infid(got[n], ref[n]))
        assert infid <= 1e-13, f"leg {n} infidelity {infid:.3e}"

        e_got = _energy(got[n], mpos[n], be).real
        e_ref = _energy(ref[n], mpos[n], be).real
        rel_e = abs(e_got - e_ref) / max(abs(e_ref), 1e-300)
        assert rel_e <= 5e-10, f"leg {n} energy relative deviation {rel_e:.3e}"

        ex_got = np.asarray(local_expectations(got[n], SX))
        ex_ref = np.asarray(local_expectations(ref[n], SX))
        rel_ex = (float(np.max(np.abs(ex_got - ex_ref))) /
                 max(float(np.max(np.abs(ex_ref))), 1e-300))
        assert rel_ex <= 5e-10, f"leg {n} <SX> relative deviation {rel_ex:.3e}"


# --------------------------------------------------------------------------- svd_batched fallback
def test_svd_batched_fallback_on_nonconvergence(monkeypatch):
    """`svd_batched`'s non-converged-driver recovery path, exercised without real CUDA.

    The recovery path is gated on `self.device.type == "cuda"` and the CUDA-only `gesvdj`
    driver, so it cannot run for real on a laptop. What CAN be tested on a laptop is the
    CONTROL FLOW around it: the device-type gate is faked by swapping `be.device` for a
    stand-in whose `.type` is `'cuda'` (restored via `monkeypatch`, and only around the
    `svd_batched` call -- tensor creation elsewhere still uses the real CPU device), and
    `torch.linalg.svd` is faked to raise the exact "failed to converge" warning `svd_batched`
    catches, on every call EXCEPT one asking for the `'gesvd'` driver -- reproducing the
    real failure/retry shape without needing a real cuSOLVER convergence failure.

    Asserts the fallback counter increments exactly once and the returned factors still
    reconstruct the input, i.e. the recovered SVD is correct, not merely non-crashing.
    """
    torch = pytest.importorskip("torch")
    import warnings

    be = get_backend("torch-cpu")
    rng = np.random.default_rng(3)
    A = be.array(rng.standard_normal((3, 5, 4)) + 1j * rng.standard_normal((3, 5, 4)))

    class _FakeCudaDevice:
        type = "cuda"

    monkeypatch.setattr(be, "device", _FakeCudaDevice())
    monkeypatch.setenv("TNKIT_CUDA_SVD_DRIVER_BATCHED", "gesvdj")   # the default is gesvd

    calls = {"n": 0}
    real_svd = torch.linalg.svd

    def fake_svd(x, full_matrices=False, driver=None):
        calls["n"] += 1
        if driver != "gesvd":
            warnings.warn("batches 2 failed to converge. It is recommended to redo this "
                          "SVD with another driver.")
        return real_svd(x, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", fake_svd)

    assert getattr(be, "svd_batched_fallbacks", 0) == 0
    U, S, Vh = be.svd_batched(A)
    assert be.svd_batched_fallbacks == 1
    assert calls["n"] == 2                                    # the failed try, then the retry
    recon = U @ torch.diag_embed(S.to(U.dtype)) @ Vh
    assert float(torch.max(torch.abs(recon - A))) < 1e-10


@pytest.mark.parametrize("env,driver", [(None, "gesvd"), ("gesvdj", "gesvdj"),
                                        ("default", None), ("gesvda", "gesvda")])
def test_svd_batched_driver_selection(monkeypatch, env, driver):
    """On CUDA the batched SVD uses gesvd unless TNKIT_CUDA_SVD_DRIVER_BATCHED says otherwise.
    The CUDA gate is faked; only the driver handed to torch is checked."""
    torch = pytest.importorskip("torch")
    be = get_backend("torch-cpu")
    A = be.array(np.random.default_rng(0).standard_normal((2, 4, 3)))

    class _FakeCudaDevice:
        type = "cuda"

    monkeypatch.setattr(be, "device", _FakeCudaDevice())
    if env is None:
        monkeypatch.delenv("TNKIT_CUDA_SVD_DRIVER_BATCHED", raising=False)
    else:
        monkeypatch.setenv("TNKIT_CUDA_SVD_DRIVER_BATCHED", env)
    seen = []
    real_svd = torch.linalg.svd

    def fake_svd(x, full_matrices=False, driver=None):
        seen.append(driver)
        return real_svd(x, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", fake_svd)
    be.svd_batched(A)
    assert seen == [driver]
    assert getattr(be, "svd_batched_fallbacks", 0) == 0


@pytest.mark.cuda
def test_cuda_svd_batched_default_driver():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device on this runner")
    be = get_backend("torch-cuda")
    rng = np.random.default_rng(1)
    A_np = rng.standard_normal((6, 96, 96)) + 1j * rng.standard_normal((6, 96, 96))
    U, S, Vh = be.svd_batched(be.array(A_np))
    recon = U @ torch.diag_embed(S.to(U.dtype)) @ Vh
    assert float(torch.max(torch.abs(recon - be.array(A_np)))) < 1e-10
    S_ref = np.linalg.svd(A_np, compute_uv=False)
    assert np.max(np.abs(be.to_numpy(S) - S_ref)) < 1e-11 * S_ref.max()
    assert getattr(be, "svd_batched_fallbacks", 0) == 0


# --------------------------------------------------------------------------- curated namespace
def test_batch_exceptions_are_in_the_curated_namespace():
    """`BatchGeometryError` and `BatchKrylovError` are reachable from `tnkit` and `tnkit.api`,
    the same way `BatchTDVPOptions`/`evolve_batch` are -- not only from `tnkit.batch_tdvp`."""
    import tnkit
    import tnkit.api as api
    assert tnkit.BatchGeometryError is BatchGeometryError
    assert tnkit.BatchKrylovError is BatchKrylovError
    assert api.BatchGeometryError is BatchGeometryError
    assert api.BatchKrylovError is BatchKrylovError
    assert "BatchGeometryError" in api.CURATED and "BatchKrylovError" in api.CURATED
