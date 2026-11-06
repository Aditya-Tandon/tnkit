"""The block eigensolver, state-averaged block DMRG, and `dmrg_tower(pair_solve=True)`."""
import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.dmrg_excited import degenerate_clusters, dmrg_block, dmrg_tower
from tnkit.eigensolver import block_lowk
from tnkit.models import mpo_to_dense, tfim_ring_mpo
from tnkit.models_lgt import schwinger_dense, schwinger_mpo


@pytest.fixture(scope="module")
def ring8():
    W = tfim_ring_mpo(8, 1.0, 0.10)
    return W, np.linalg.eigvalsh(mpo_to_dense(W))


@pytest.mark.parametrize("spec", ["numpy", "numpy-real"])
def test_block_lowk_resolves_a_near_degenerate_pair(spec):
    be = get_backend(spec)
    rng = np.random.default_rng(3)
    n = 200
    A = rng.standard_normal((n, n))
    w, V = np.linalg.eigh(A + A.T)
    w[1] = w[0] + 1e-6
    A = (V * w) @ V.T
    A = (A + A.T) / 2
    exact = np.linalg.eigvalsh(A)[:3]
    V0 = [be.array(rng.standard_normal(n)) for _ in range(3)]
    ev, vecs, info = block_lowk(lambda x: be.array(A @ x), V0, be, nev=3, tol=1e-11,
                                max_iter=250)
    assert info["converged"]
    assert np.max(np.abs(np.array(ev) - exact)) < 1e-10
    X = np.stack(vecs)
    assert np.max(np.abs(X.conj() @ X.T - np.eye(3))) < 1e-12


def test_dmrg_block_returns_the_next_pair(ring8):
    W, exact = ring8
    _, states, _ = dmrg_tower(W, 16, 2, penalty=5.0, n_sweeps=12, tol=1e-13, seed=1)
    E, _, info = dmrg_block(W, 16, 2, prev_states=states, penalty=5.0, n_sweeps=12,
                            tol=1e-13, seed=1)
    assert E[0] <= E[1]
    assert np.max(np.abs(np.array(E) - exact[2:4])) < 1e-9
    assert max(info["block_overlaps"]) < 1e-10


@pytest.mark.parametrize("E,kw,want", [
    ([0.0, 1.0, 1.01, 2.01, 2.03, 2.04], {}, [(1, 2), (3, 5)]),
    ([0.0, 1.0, 1.01, 2.01, 2.03, 2.04], {"tau": 0.005}, []),
    ([0.0, 1.0, 1.01, 2.01, 2.03, 2.04], {"max_k": 2}, [(1, 2), (3, 5)]),
    ([0.0, 2.0, 2.02, 3.0], {}, [(1, 2)]),
    ([0.0, 2.0, 2.02, 3.0], {"m1_ref": 0.2}, []),
    ([0.0, 1.0, 0.98, 2.0], {}, [(1, 2)]),                 # out of order counts as close
    # a 5-level run over max_k = 4 is cut at its widest spacing that leaves no singleton
    ([0.0, 1.0, 1.0375, 1.0437, 1.0508, 1.0520, 3.0], {}, [(1, 3), (4, 5)]),
    ([0.0, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 2.0], {"max_k": 2},
     [(1, 2), (3, 4), (5, 6)]),
])
def test_degenerate_clusters(E, kw, want):
    kw = {"tau": 0.05, **kw}
    got = degenerate_clusters(E, **kw)
    assert got == want
    assert all(hi > lo for lo, hi in got)


def test_pair_solve_off_is_the_default(ring8):
    W, _ = ring8
    kw = dict(penalty=5.0, n_sweeps=8, tol=1e-13, seed=1)
    E0, S0, I0 = dmrg_tower(W, 16, 4, **kw)
    E1, S1, I1 = dmrg_tower(W, 16, 4, pair_solve=False, **kw)
    assert np.array_equal(E0, E1) and "pair_clusters" not in I1[0]
    assert all(np.array_equal(a, b) for P, Q in zip(S0, S1) for a, b in zip(P, Q))


def test_repairing_a_well_separated_cluster_changes_nothing(ring8):
    """With tau large enough to flag well-separated levels, the repair runs in full on a
    cluster with nothing to fix and must leave the energies where they were."""
    W, exact = ring8
    kw = dict(penalty=5.0, n_sweeps=12, tol=1e-13, seed=1)
    E0, _, _ = dmrg_tower(W, 16, 5, **kw)
    E1, _, I1 = dmrg_tower(W, 16, 5, pair_solve=True, pair_tau=10.0, **kw)
    assert I1[0]["pair_clusters"] and I1[0]["energies_pre"] == list(E0)
    assert any(info.get("repaired") for info in I1)
    assert np.max(np.abs(E1 - E0)) < 1e-9
    assert np.max(np.abs(E1 - exact[:5])) < 1e-9


@pytest.mark.parametrize("deflate", ["penalty", "project"])
@pytest.mark.parametrize("chi,bar", [(32, 1e-10), (16, 1e-6)])
def test_schwinger_near_degenerate_meson_pair(chi, bar, deflate):
    """N = 10 Schwinger, x = 2, mu = 0.2: levels 4 and 5 are 0.0074 gaps apart."""
    N, x, mu, lam = 10, 2.0, 0.2, 10.0
    exact = np.linalg.eigvalsh(schwinger_dense(N, x, mu, 0.0, lam))[:7]
    gap = exact[1] - exact[0]
    E, _, I = dmrg_tower(schwinger_mpo(N, x, mu, 0.0, lam), chi, 7, penalty=2.0,
                         n_sweeps=10, tol=1e-12, seed=0, pair_solve=True, deflate=deflate)
    assert I[0]["pair_clusters"] == [(4, 5)]
    assert I[4]["repaired"] and I[5]["repaired"]
    assert np.max(np.abs(E - exact)) / gap < bar


def _dense(mps):
    v = np.ones((1, 1))
    for A in mps:
        v = np.einsum("xa,asb->xsb", v, A).reshape(-1, A.shape[2])
    return v[:, 0] / np.linalg.norm(v[:, 0])


def _max_overlap(states):
    V = np.array([_dense(P) for P in states])
    return np.max(np.abs(V.conj() @ V.T) - np.eye(len(states)))


def test_projection_tower_is_orthogonal_to_round_off(ring8):
    W, exact = ring8
    E, S, I = dmrg_tower(W, 16, 5, n_sweeps=12, tol=1e-13, seed=1, deflate="project")
    assert all(info["deflate"] == "project" for info in I)
    assert np.max(np.abs(E - exact[:5])) < 1e-9
    assert _max_overlap(S) < 1e-13


def _block_eigvalsh(H, N):
    """The full spectrum of a Hamiltonian that conserves the number of up bits, one
    particle-number block at a time (exactly the spectrum of H, at a fraction of the cost)."""
    n = np.array([bin(s).count("1") for s in range(2 ** N)])
    blocks = [np.flatnonzero(n == k) for k in range(N + 1)]
    assert not any(H[np.ix_(b, np.flatnonzero(n != k))].any() for k, b in enumerate(blocks))
    return np.sort(np.concatenate([np.linalg.eigvalsh(H[np.ix_(b, b)]) for b in blocks]))


def test_projected_repair_of_a_tight_meson_cluster():
    """Schwinger N = 12, x = 1, mu = 0.5: levels 7-11 sit within 0.04 gaps of each other
    (down to 0.0012), a run longer than ``pair_max_k``. The repair must not make any level
    worse than the sequential tower, and the whole tower must stay orthogonal to round-off."""
    N, x, mu, lam = 12, 1.0, 0.5, 10.0
    exact = _block_eigvalsh(schwinger_dense(N, x, mu, 0.0, lam), N)[:12]
    gap = exact[1] - exact[0]
    E, S, I = dmrg_tower(schwinger_mpo(N, x, mu, 0.0, lam), 24, 12, penalty=2.0, n_sweeps=10,
                         tol=1e-12, seed=0, deflate="project", pair_solve=True)
    assert I[0]["pair_clusters"] == [(1, 2), (7, 9), (10, 11)]
    assert all(I[n].get("repaired") for n in range(7, 12))
    before = np.abs(np.array(I[0]["energies_pre"]) - exact) / gap
    after = np.abs(E - exact) / gap
    assert np.all(after <= before + 1e-11)
    assert _max_overlap(S) < 1e-13


def test_unknown_deflation_is_rejected(ring8):
    W, _ = ring8
    with pytest.raises(ValueError, match="deflate"):
        dmrg_tower(W, 8, 2, deflate="gap")


@pytest.fixture(scope="module")
def schwinger8():
    W = schwinger_mpo(8, 1.0, 0.5, 0.0, 10.0)
    return W, np.linalg.eigvalsh(schwinger_dense(8, 1.0, 0.5, 0.0, 10.0))


def _backend(name):
    if name.startswith("torch"):
        pytest.importorskip("torch")
    return name


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("backend", ["auto", "numpy", "torch-cpu"])
def test_projection_tower_with_levels_above_zero(schwinger8, backend, seed):
    """Levels 8-9 lie above 0, where ``Q H Q`` puts the deflated span; unlifted, the local
    solve drifted into it and the tower came back wrong with overlaps of 1e-16."""
    W, exact = schwinger8
    E, _, I = dmrg_tower(W, 16, 10, penalty=2.0, n_sweeps=10, tol=1e-12, seed=seed,
                         deflate="project", backend=_backend(backend))
    assert np.max(np.abs(E - exact[:10])) < 1e-10
    assert sum(info["project_fallbacks"] for info in I) == 0


def test_projection_parity_torch_numpy(schwinger8):
    W, exact = schwinger8
    kw = dict(penalty=2.0, n_sweeps=10, tol=1e-12, seed=0, deflate="project", pair_solve=True)
    E_np, _, _ = dmrg_tower(W, 16, 10, backend="numpy", **kw)
    E_t, _, _ = dmrg_tower(W, 16, 10, backend=_backend("torch-cpu"), **kw)
    assert np.max(np.abs(E_np - E_t)) < 1e-10
    assert np.max(np.abs(E_np - exact[:10])) < 1e-10


def test_projection_falls_back_where_the_block_has_no_room(schwinger8):
    """From level 16 on, the edge blocks (dimension 16) are spanned by the levels below."""
    W, exact = schwinger8
    E, _, I = dmrg_tower(W, 16, 18, penalty=2.0, n_sweeps=10, tol=1e-12, seed=0,
                         deflate="project")
    assert all(I[n]["project_fallbacks"] == 0 for n in range(16))
    assert all(I[n]["project_fallbacks"] > 0 for n in (16, 17))
    assert np.max(np.abs(E - exact[:18])) < 1e-10


def test_projection_finds_an_exactly_degenerate_partner():
    """TFIM ring N = 10: levels 2 and 3 are exactly degenerate. A warm start from level 2
    lies almost wholly in the deflated span, and from it level 3 came back as level 4."""
    W = tfim_ring_mpo(10, 1.0, 0.10)
    exact = np.linalg.eigvalsh(mpo_to_dense(W))[:5]
    E, _, _ = dmrg_tower(W, 32, 5, penalty=2.0, n_sweeps=10, tol=1e-12, seed=0,
                         deflate="project")
    assert np.max(np.abs(E - exact)) < 1e-10


def test_the_polish_retries_with_a_larger_penalty_on_a_leaking_solve(ring8, monkeypatch):
    import importlib
    dx = importlib.import_module("tnkit.dmrg_excited")     # `tnkit.dmrg_excited` is also a function
    real, calls = dx.dmrg_excited, []

    def leaky_first_polish(*a, **kw):
        E, M, info = real(*a, **kw)
        polish = kw["seed"] >= 9902                       # the polish solves' seed offset
        if polish and not any(c["polish"] for c in calls):
            info = dict(info, penalty_overlaps=[1.0])
        calls.append({"polish": polish, "penalty": kw["penalty"], "cold": kw["init_mps"] is None})
        return E, M, info

    W, exact = ring8
    monkeypatch.setattr(dx, "dmrg_excited", leaky_first_polish)
    E, _, _ = dmrg_tower(W, 16, 5, pair_solve=True, pair_tau=10.0, penalty=5.0, n_sweeps=12,
                         tol=1e-13, seed=1)
    first, retry = [c for c in calls if c["polish"]][:2]
    assert retry["penalty"] == 4 * first["penalty"] and retry["cold"] and not first["cold"]
    assert np.max(np.abs(E - exact[:5])) < 1e-9


_HAS_CUDA = False
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

# numpy always; the device too where there is one (the same assertions, so a CPU run covers them)
BACKENDS = ["numpy", pytest.param("torch-cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")])]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("pair", [False, True])
def test_projection_tower_on_a_backend(schwinger8, backend, pair):
    """The lift, the fallback guard and the block solve: levels above 0, 10 levels, host numpy
    states, energies equal to exact diagonalisation."""
    W, exact = schwinger8
    E, S, I = dmrg_tower(W, 16, 10, penalty=2.0, n_sweeps=10, tol=1e-12, seed=0,
                         deflate="project", pair_solve=pair, backend=backend)
    assert np.max(np.abs(E - exact[:10])) < 1e-10
    assert all(isinstance(A, np.ndarray) for P in S for A in P)
    assert sum(info["project_fallbacks"] for info in I) == 0
    assert _max_overlap(S) < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_dmrg_block_in_project_mode_returns_the_next_levels(schwinger8, backend):
    """Three states share one set of isometries, so they need a bond dimension of up to three
    times a single state's (16 is exact for one state at N = 8, and not for three)."""
    W, exact = schwinger8
    _, S, _ = dmrg_tower(W, 16, 6, penalty=2.0, n_sweeps=10, tol=1e-12, seed=0,
                         deflate="project", backend=backend)
    E, _, info = dmrg_block(W, 32, 3, prev_states=S, penalty=2.0, n_sweeps=12, tol=1e-12,
                            deflate="project", backend=backend)
    assert np.max(np.abs(np.array(E) - exact[6:9])) < 1e-10
    assert max(info["penalty_overlaps"]) < 1e-12 and info["discarded_weight"] < 1e-12


def test_dmrg_block_pays_for_a_shared_basis_that_is_too_small(schwinger8):
    """At chi = 16 the three levels' shared basis truncates: the energies sit above the exact
    ones by less than the discarded weight times the energy range, which the polish of
    `dmrg_tower(pair_solve=True)` removes."""
    W, exact = schwinger8
    _, S, _ = dmrg_tower(W, 16, 6, penalty=2.0, n_sweeps=10, tol=1e-12, seed=0,
                         deflate="project")
    E, _, info = dmrg_block(W, 16, 3, prev_states=S, penalty=2.0, n_sweeps=12, tol=1e-12,
                            deflate="project")
    err = np.array(E) - exact[6:9]
    assert info["discarded_weight"] > 1e-5 and np.all(err > 0)
    assert np.max(err) < (exact[8] - exact[0]) * info["discarded_weight"]
