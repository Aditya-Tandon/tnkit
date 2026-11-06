"""Front-door API: typed vumps result, idmrg/dmrg_excited options, tebd bond_h, and
excitations on a custom MPO. Each old call style keeps its exact numbers."""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from tnkit import UniformMPS, vumps


def test_vumps_returns_a_dict_with_attribute_access():
    gs = vumps(1.5, 6, n_iter=30, tol=1e-10)
    assert isinstance(gs, UniformMPS) and isinstance(gs, dict)
    assert gs.AL is gs["AL"] and gs.e0 == gs["e0"] and gs.get("trace") is None
    assert "trace" not in gs and list(gs)[:3] == ["e0", "S", "xi"]
    with pytest.raises(AttributeError):
        gs.trace
    back = pickle.loads(pickle.dumps(gs))
    assert isinstance(back, UniformMPS) and np.array_equal(back.AL, gs.AL)
    assert copy.deepcopy(gs).e0 == gs.e0
    assert "trace" in vumps(1.5, 6, n_iter=5, tol=1e-10, trace=True)


def _same(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ("e0", "S", "svals", "A", "B"))


def test_idmrg_options_mpo_and_deprecated_be():
    from tnkit import IDMRGOptions, idmrg
    from tnkit.backend import get_backend
    from tnkit.idmrg import mixed_field_bulk_mpo, tfim_bulk_mpo
    ref = idmrg(1.5, 8, n_iter=60)
    assert _same(ref, idmrg(1.5, chi=8, options=IDMRGOptions(n_iter=60)))
    assert _same(ref, idmrg(tfim_bulk_mpo(1.5), 8, n_iter=60))
    with pytest.warns(DeprecationWarning, match="backend"):
        assert _same(ref, idmrg(1.5, 8, n_iter=60, be=get_backend("numpy")))
    with pytest.raises(TypeError, match="both"):
        idmrg(1.5, 8, be="numpy", backend="numpy")
    # a custom bulk MPO reaches the right fixed point: the mixed-field chain against VUMPS
    from tnkit.idmrg import mixed_field_bond_h
    e_i = idmrg(mixed_field_bulk_mpo(1.2, 0.2), 12, n_iter=300, tol=1e-10)["e0"]
    e_v = vumps(1.2, 12, n_iter=200, tol=1e-10, mpo=mixed_field_bulk_mpo(1.2, 0.2),
                bond_h=mixed_field_bond_h(1.2, 0.2))["e0"]
    assert abs(e_i - e_v) < 1e-4
    # D = 5 automaton (three-site term): channel D-1 must end every term
    from tnkit.idmrg import improved_ising_bulk_mpo, improved_triple_h
    e_i = idmrg(improved_ising_bulk_mpo(1.2, 0.2, 0.1), 12, n_iter=300, tol=1e-12)["e0"]
    e_v = vumps(1.2, 12, n_iter=300, tol=1e-12, mpo=improved_ising_bulk_mpo(1.2, 0.2, 0.1),
                triple_h=improved_triple_h(1.2, 0.2, 0.1))["e0"]
    assert abs(e_i - e_v) < 1e-6
    with pytest.raises(TypeError, match="number"):
        idmrg(np.eye(2), 4)
    with pytest.raises(TypeError, match="dmrg"):
        idmrg(tfim_bulk_mpo(1.5)[None], 4)


def _tfim_bond_h(N, g):
    from tnkit.models import I2, SX, SZ
    out = []
    for i in range(N - 1):
        wl = 1.0 if i == 0 else 0.5
        wr = 1.0 if i == N - 2 else 0.5
        out.append(-np.kron(SX, SX) - g * (wl * np.kron(SZ, I2) + wr * np.kron(I2, SZ)))
    return out


@pytest.mark.parametrize("imaginary", [False, True])
def test_tebd_bond_h_reproduces_the_tfim(imaginary):
    from tnkit import tebd_evolve
    from tnkit._dmrg import random_mps
    N = 8
    psi = [np.asarray(x) for x in random_mps(N, 4, 2, np.random.default_rng(2))]
    a, _ = tebd_evolve(psi, N, 1.3, 8, 0.05, 6, imaginary=imaginary)
    b, _ = tebd_evolve(psi, N, None, 8, 0.05, 6, imaginary=imaginary,
                       bond_h=_tfim_bond_h(N, 1.3))
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-12)


def test_tebd_bond_h_d3_matches_exact_evolution():
    """A spin-1 chain (d = 3) against exp(-iHt) on the full state."""
    from scipy.linalg import expm
    from tnkit import tebd_evolve
    from tnkit._dmrg import random_mps
    N, d, dt, n = 4, 3, 0.01, 20
    Sz = np.diag([1.0, 0.0, -1.0])
    Sp = np.sqrt(2) * np.diag([1.0, 1.0], 1)
    hh = np.kron(Sz, Sz) + 0.5 * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp))
    psi = [np.asarray(x) for x in random_mps(N, 9, d, np.random.default_rng(4))]
    out, _ = tebd_evolve(psi, N, None, 27, dt, n, bond_h=[hh] * (N - 1))

    def dense(M):
        v = M[0]
        for A in M[1:]:
            v = np.einsum("...a,asb->...sb", v, A)
        return v.reshape(-1)

    H = sum(np.kron(np.kron(np.eye(d ** i), hh), np.eye(d ** (N - 2 - i))) for i in range(N - 1))
    v0 = dense(psi)
    v0 = v0 / np.linalg.norm(v0)
    ref = expm(-1j * H * dt * n) @ v0
    got = dense(out)
    assert abs(abs(np.vdot(ref, got)) - 1) < 1e-5          # 2nd-order Trotter at dt = 0.01


def test_tebd_bond_h_takes_the_vumps_layout():
    from tnkit import tebd_evolve
    from tnkit._dmrg import random_mps
    from tnkit.idmrg import mixed_field_bond_h
    psi = [np.asarray(x) for x in random_mps(6, 4, 2, np.random.default_rng(1))]
    h4 = mixed_field_bond_h(1.1, 0.3)
    a, _ = tebd_evolve(psi, 6, None, 8, 0.05, 3, bond_h=[h4] * 5)
    b, _ = tebd_evolve(psi, 6, None, 8, 0.05, 3, bond_h=[h4.reshape(4, 4)] * 5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_tebd_bond_h_argument_rules():
    from tnkit import tebd_evolve
    from tnkit._dmrg import random_mps
    psi = [np.asarray(x) for x in random_mps(4, 2, 2, np.random.default_rng(0))]
    with pytest.raises(ValueError, match="g=None"):
        tebd_evolve(psi, 4, 1.0, 4, 0.05, 1, bond_h=_tfim_bond_h(4, 1.0))
    with pytest.raises(ValueError, match="needs g"):
        tebd_evolve(psi, 4, None, 4, 0.05, 1)
    with pytest.raises(ValueError, match="bond_h has 2 bonds"):
        tebd_evolve(psi, 4, None, 4, 0.05, 1, bond_h=_tfim_bond_h(3, 1.0))


def test_dmrg_excited_options_match_flat_bit_for_bit():
    from tnkit import DMRGExcitedOptions, dmrg, dmrg_excited, tfim_mpo
    W = tfim_mpo(8, 1.2)
    _, gs, _ = dmrg(W, chi=8, n_sweeps=4)
    flat = dmrg_excited(W, 8, prev_states=[gs], n_sweeps=3, seed=4)
    grouped = dmrg_excited(W, 8, DMRGExcitedOptions(prev_states=[gs], n_sweeps=3, seed=4))
    mixed = dmrg_excited(W, chi=8, options={"prev_states": [gs], "n_sweeps": 3}, seed=4)
    for other in (grouped, mixed):
        assert other[0] == flat[0]
        assert all(np.array_equal(a, b) for a, b in zip(flat[1], other[1]))
    with pytest.raises(TypeError, match="options must be"):
        dmrg_excited(W, 8, [gs])                          # old positional prev_states
    with pytest.raises(TypeError, match="unexpected keyword"):
        dmrg_excited(W, 8, n_sweep=3)
    from tnkit.dmrg_excited import dmrg_tower
    with pytest.raises(TypeError, match="flat keywords"):
        dmrg_tower(W, 8, 2, options=DMRGExcitedOptions())


def test_excitations_on_a_custom_mpo_match_the_model_front_door():
    from tnkit.excitations import mixed_field_excitations, tfim_excitations
    from tnkit.idmrg import mixed_field_bond_h, mixed_field_bulk_mpo
    g, h, ps = 1.4, 0.1, [0.0, 0.7]
    ref, _ = mixed_field_excitations(g, h, 6, ps, vumps_iter=80, verbose=False)
    got, _ = tfim_excitations(g, 6, ps, vumps_iter=80, verbose=False,
                              mpo=mixed_field_bulk_mpo(g, h), bond_h=mixed_field_bond_h(g, h))
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("which", ["tdvp1", "tdvp2"])
def test_step_dict_counts_krylov_cap_hits(which):
    from tnkit import dmrg, tdvp1_evolve, tdvp2_evolve, tfim_mpo
    W = tfim_mpo(8, 1.0)
    _, gs, _ = dmrg(W, chi=6, n_sweeps=3)
    Wq = tfim_mpo(8, 0.5)
    seen = {}
    for m in (3, 30):
        rows = []
        cb = lambda d, _: rows.append(d)                           # noqa: E731
        if which == "tdvp1":
            tdvp1_evolve(list(gs), Wq, 0.05, 2, graph="off", krylov_m=m, step_callback=cb)
        else:
            tdvp2_evolve(list(gs), Wq, 6, 0.05, 2, krylov_m=m, step_callback=cb)
        seen[m] = [d["krylov_cap_hits"] for d in rows]
    assert all(n > 0 for n in seen[3]) and seen[30] == [0, 0]
