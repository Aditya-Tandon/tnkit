"""`tnkit.invariants`: shape rules for MPS/MPO lists, `TNKIT_CHECK` modes, front-door wiring."""
from __future__ import annotations

import numpy as np
import pytest

from tnkit import invariants as inv
from tnkit import (apply_mpo, compress, dmrg, dmrg_excited, evolve_batch, tdvp1_evolve,
                   tdvp2_evolve)
from tnkit._dmrg import random_mps
from tnkit.invariants import InvariantError, check_mpo, check_mps, check_pair
from tnkit.models import (identity_mpo, longrange_mpo, magnetization_mpo, mpo_sum,
                          tfim2d_cylinder, tfim_mpo, tfim_ring_mpo, translation_mpo)
from tnkit.models_lgt import charge_conjugation_mpo, schwinger_mpo


def _mps(N=6, chi=4, d=2, seed=0):
    return [np.asarray(x) for x in random_mps(N, chi, d, np.random.default_rng(seed))]


@pytest.fixture(autouse=True)
def _entry_mode(monkeypatch):
    monkeypatch.delenv("TNKIT_CHECK", raising=False)


@pytest.mark.parametrize("build", [
    lambda: tfim_mpo(8, 1.0, 0.1), lambda: tfim_ring_mpo(8, 1.0), lambda: magnetization_mpo(8),
    lambda: translation_mpo(6), lambda: identity_mpo(5),
    lambda: mpo_sum([tfim_mpo(6, 1.0), magnetization_mpo(6)]),
    lambda: longrange_mpo(6, [(0, 3), (1, 5)], 1.0), lambda: tfim2d_cylinder(3, 2, 1.0)[0],
    lambda: schwinger_mpo(6, 1.0, 0.1, charge_penalty=2.0), lambda: charge_conjugation_mpo(6),
])
def test_every_builder_satisfies_the_mpo_rules(build):
    check_mpo(build())


def test_a_valid_pair_passes_and_shapes_are_returned():
    M, W = _mps(), tfim_mpo(6, 1.0)
    check_pair(M, W)
    assert check_mps(M)[2] == M[2].shape
    assert check_mpo(W, d=2)[0] == W[0].shape


def test_windows_pass_against_their_environments():
    D, w, d = 4, 3, 2
    M = [np.zeros((D, d, D)) for _ in range(5)]
    W = [np.zeros((w, w, d, d)) for _ in range(5)]
    check_pair(M, W, np.zeros((D, w, D)), np.zeros((D, w, D)))
    with pytest.raises(InvariantError, match="left outer bonds"):
        check_pair(M, W)
    with pytest.raises(InvariantError, match="L0 has shape"):
        check_pair(M, W, np.zeros((D, w + 1, D)), np.zeros((D, w, D)))
    with pytest.raises(InvariantError, match="RN has shape"):
        check_pair(M, W, np.zeros((D, w, D)), np.zeros((D + 1, w, D + 1)))


@pytest.mark.parametrize("mutate, msg", [
    (lambda M: M.__setitem__(2, M[2][:, :, :, None]), r"M\[2\] has shape .* expected rank 3"),
    (lambda M: M.__setitem__(3, M[3][:1]), r"M bond 2-3 does not chain"),
    (lambda M: M.__setitem__(0, np.ones((2, 2, M[0].shape[2]))), r"outer MPS bonds are \(2, 1\)"),
    (lambda M: M.clear(), "M is empty"),
])
def test_mps_violations_name_the_site(mutate, msg):
    M = _mps()
    mutate(M)
    with pytest.raises(InvariantError, match=msg):
        check_mps(M)


@pytest.mark.parametrize("mutate, msg", [
    (lambda W: W.__setitem__(1, W[1][:, :, :, :1]), r"W\[1\] has shape .* physical legs"),
    (lambda W: W.__setitem__(2, W[2][:2]), r"W bond 1-2 does not chain"),
    (lambda W: W.__setitem__(0, W[1]), r"outer MPO bonds are \(3, 1\)"),
    (lambda W: W.__setitem__(4, W[4][0]), r"W\[4\] has shape .* expected rank 4"),
])
def test_mpo_violations_name_the_site(mutate, msg):
    W = tfim_mpo(6, 1.0)
    mutate(W)
    with pytest.raises(InvariantError, match=msg):
        check_mpo(W)


def test_mpo_physical_dimension():
    with pytest.raises(InvariantError, match="of size 3"):
        check_mpo(tfim_mpo(4, 1.0), d=3)


def test_pair_violations():
    M, W = _mps(), tfim_mpo(6, 1.0)
    with pytest.raises(InvariantError, match="MPS has 6 sites, MPO has 5"):
        check_pair(M, W[:5])
    M3 = _mps(d=3)
    with pytest.raises(InvariantError, match="site 0: MPS physical leg 3"):
        check_pair(M3, W)


def test_uniform_is_opt_in():
    M = _mps()
    M[2] = M[2].real                                   # mixed real/complex: legal by default
    check_mps(M)
    with pytest.raises(InvariantError, match=r"M\[2\] is numpy real on cpu; M\[0\] is numpy complex"):
        check_mps(M, uniform=True)
    torch = pytest.importorskip("torch")
    M = _mps()
    M[1] = torch.as_tensor(M[1])                       # mixed libraries: legal by default
    check_mps(M)
    with pytest.raises(InvariantError, match=r"M\[1\]"):
        check_mps(M, uniform=True)


def test_debug_accepts_mixed_inputs_that_the_solver_casts(monkeypatch):
    monkeypatch.setenv("TNKIT_CHECK", "debug")
    W = tfim_mpo(6, 1.0)
    M = _mps()
    M[2] = M[2].real
    dmrg(W, chi=4, n_sweeps=1, init_mps=M)
    tdvp2_evolve(M, W, 4, 0.05, 1)


def test_prev_states_may_be_an_iterator():
    W = tfim_mpo(6, 1.2)
    _, gs, _ = dmrg(W, chi=4, n_sweeps=3)
    a = dmrg_excited(W, 4, prev_states=[gs], n_sweeps=2)
    b = dmrg_excited(W, 4, prev_states=iter([gs]), n_sweeps=2)
    assert a[0] == b[0]


def test_modes(monkeypatch):
    assert inv.mode() == "entry"
    monkeypatch.setenv("TNKIT_CHECK", "Debug")
    assert inv.mode() == "debug"
    monkeypatch.setenv("TNKIT_CHECK", "of")
    with pytest.raises(ValueError, match="TNKIT_CHECK"):
        inv.mode()


def test_off_skips_the_check(monkeypatch):
    """What the check prevents: einsum broadcasts size-1 legs and zip truncates, so a bad
    input is not an error on its own. apply_mpo on a short MPO silently drops sites."""
    M, W = _mps(), tfim_mpo(6, 1.0)
    with pytest.raises(InvariantError):
        apply_mpo(M, W[:5])
    monkeypatch.setenv("TNKIT_CHECK", "off")
    assert len(apply_mpo(M, W[:5])) == 5


_SOLVERS = {
    "dmrg": lambda W, M: dmrg(W, chi=4, n_sweeps=2, tol=0.0, init_mps=M),
    "dmrg_excited": lambda W, M: dmrg_excited(W, 4, prev_states=[M], n_sweeps=2, tol=0.0,
                                              backend="numpy"),
    "tdvp1": lambda W, M: tdvp1_evolve(M, W, 0.05, 2, graph="off"),
    "tdvp2": lambda W, M: tdvp2_evolve(M, W, 4, 0.05, 2),
    "batch": lambda W, M: evolve_batch([M, M], W, 0.05, 2, evolver="1tdvp", krylov_m=6),
}


@pytest.mark.parametrize("name", sorted(_SOLVERS))
def test_sweep_checks_run_only_in_debug(monkeypatch, name):
    calls = []
    real = inv.check_pair
    monkeypatch.setattr(inv, "check_pair", lambda *a, **k: (calls.append(k), real(*a, **k)))
    W, M = tfim_mpo(6, 1.0), _mps()
    _SOLVERS[name](W, M)
    n_entry = len(calls)
    assert n_entry >= 1 and not any(k.get("uniform") for k in calls)
    calls.clear()
    monkeypatch.setenv("TNKIT_CHECK", "debug")
    _SOLVERS[name](W, M)
    assert len(calls) == n_entry + 2 and sum(bool(k.get("uniform")) for k in calls) == 2
    calls.clear()
    monkeypatch.setenv("TNKIT_CHECK", "off")
    _SOLVERS[name](W, M)
    assert calls == []


def test_front_doors_refuse_bad_input():
    W = tfim_mpo(6, 1.0)
    M = _mps()
    bulk = [W[1]] * 6                                  # bulk tensors at the boundary
    with pytest.raises(InvariantError, match="outer MPO bonds"):
        dmrg(bulk, chi=4, n_sweeps=1)
    with pytest.raises(InvariantError, match="MPS has 5 sites"):
        dmrg_excited(W, 4, prev_states=[M[:5]], n_sweeps=1)
    wide = _mps()
    wide[0] = np.ones((3, 2, wide[0].shape[2]))
    for call in (lambda: tdvp2_evolve(wide, W, 4, 0.05, 1),
                 lambda: tdvp1_evolve(wide, W, 0.05, 1, graph="off"),
                 lambda: evolve_batch([wide, wide], W, 0.05, 1, evolver="1tdvp", krylov_m=6)):
        with pytest.raises(InvariantError, match="left outer bonds"):
            call()
    with pytest.raises(InvariantError, match="expected rank 3"):
        compress([m[..., None] for m in M], 4)


def test_open_ended_front_doors_accept_windows():
    """apply_mpo and compress take a window state (outer bonds above 1)."""
    D, w, d = 3, 3, 2
    rng = np.random.default_rng(1)
    M = [rng.standard_normal((D, d, D)) for _ in range(4)]
    W = [rng.standard_normal((w, w, d, d)) for _ in range(4)]
    assert apply_mpo(M, W)[0].shape == (D * w, d, D * w)
    compress(M, 2)
