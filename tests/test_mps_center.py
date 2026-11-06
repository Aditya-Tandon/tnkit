"""`MPS.center`: set by the canonicalising routines, true when set, reset on any change."""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from tnkit import (MPS, as_mps, compress, dmrg, dmrg_excited, evolve_batch, right_canonicalize,
                   tdvp1_evolve, tdvp2_evolve, tfim_mpo)
from tnkit._dmrg import random_mps


def _gauge_ok(M, c, tol=1e-10):
    """Left-canonical left of `c`, right-canonical right of it."""
    for i, A in enumerate(M):
        A = np.asarray(A)
        a, s, b = A.shape
        if i < c:
            G = A.reshape(a * s, b).conj().T @ A.reshape(a * s, b)
            ok = np.allclose(G, np.eye(b), rtol=0, atol=tol)
        elif i > c:
            G = A.reshape(a, s * b) @ A.reshape(a, s * b).conj().T
            ok = np.allclose(G, np.eye(a), rtol=0, atol=tol)
        else:
            ok = True
        if not ok:
            return False
    return True


@pytest.fixture(scope="module")
def ground():
    W = tfim_mpo(8, 1.1)
    _, M, _ = dmrg(W, chi=8, n_sweeps=3)
    return W, M


def test_solvers_record_a_true_centre(ground):
    W, M = ground
    assert M.center == 0 and _gauge_ok(M, 0)
    assert not _gauge_ok(M, len(M) - 1)                # the check can fail
    Wq = tfim_mpo(8, 0.7)
    for out in (tdvp2_evolve(list(M), Wq, 8, 0.05, 2)[0],
                tdvp1_evolve(list(M), Wq, 0.05, 2, graph="off")[0]):
        assert out.center == 0 and _gauge_ok(out, 0)
    _, X, _ = dmrg_excited(W, 8, prev_states=[M], n_sweeps=2)
    assert isinstance(X, MPS) and X.center == 0 and _gauge_ok(X, 0)
    C, _ = compress(M, 4)
    assert C.center == len(C) - 1 and _gauge_ok(C, len(C) - 1)


@pytest.mark.parametrize("evolver, ragged", [("1tdvp", "pad"), ("2tdvp", "pad"),
                                             ("2tdvp", "tight"), ("2tdvp", "group")])
def test_batch_centre_on_ragged_legs(ground, evolver, ragged):
    _, M = ground
    legs = [list(M), list(compress(M, 4)[0]), list(compress(M, 2)[0])]
    kw = dict(chi_max=8, svd_tol=1e-10) if evolver == "2tdvp" else {}
    out, _ = evolve_batch(legs, tfim_mpo(8, 0.7), 0.05, 2, evolver=evolver, ragged=ragged,
                          krylov_m=8, **kw)
    for leg in out:
        assert leg.center == 0 and _gauge_ok(leg, 0)


def test_unknown_when_nothing_canonicalised(ground):
    _, M = ground
    Wq = tfim_mpo(8, 0.7)
    for out in (tdvp2_evolve(list(M), Wq, 8, 0.05, 0, canonicalize=False)[0],
                tdvp1_evolve(list(M), Wq, 0.05, 0, canonicalize=False, graph="off")[0],
                evolve_batch([list(M)], Wq, 0.05, 0, evolver="1tdvp", krylov_m=8,
                             canonicalize=False)[0][0]):
        assert out.center is None
    assert MPS(list(M)).center is None and as_mps(list(M)).center is None


def test_right_canonicalize_sets_centre_on_an_mps():
    M = MPS(random_mps(6, 4, 2, np.random.default_rng(0)))
    assert right_canonicalize(M) is M and M.center == 0 and _gauge_ok(M, 0)


@pytest.mark.parametrize("change", [
    lambda M: M.__setitem__(1, M[1]), lambda M: M.append(M[0]), lambda M: M.extend([]),
    lambda M: M.insert(0, M[0]), lambda M: M.pop(), lambda M: M.reverse(),
    lambda M: M.__delitem__(0), lambda M: M.__iadd__([]), lambda M: M.clear(),
    lambda M: M.remove(M[0]), lambda M: M.sort(key=id), lambda M: M.__imul__(1),
    lambda M: M.__setitem__(slice(None), list(M)),
])
def test_any_change_resets_the_centre(ground, change):
    M = MPS(list(ground[1]), center=0)
    change(M)
    assert M.center is None


def test_centre_is_validated_and_defaults_to_unknown(ground):
    M = MPS(list(ground[1]))
    with pytest.raises(ValueError):
        M.center = len(M)
    with pytest.raises(TypeError):
        M.center = 1.5
    assert MPS.__new__(MPS).center is None                # how an old pickle is rebuilt


def test_windows_carry_their_centre():
    from tnkit import window as win
    from tnkit.models import SZ
    w, *_ = win.build_window(1.5, 6, 3, 2, op=SZ)
    assert w.center == 3 and win.grow_window(w, w[0], w[-1], 2, 1).center == 5


def test_copies_keep_the_centre(ground):
    M = ground[1]
    for c in (pickle.loads(pickle.dumps(M)), copy.copy(M), copy.deepcopy(M), M.to_numpy()):
        assert isinstance(c, MPS) and c.center == 0 and len(c) == len(M)
        assert all(np.array_equal(a, b) for a, b in zip(c, M))
    assert type(M[1:]) is list and isinstance(M, list)
