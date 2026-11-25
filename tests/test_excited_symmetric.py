"""`dmrg_excited` on block and symmetric MPOs, against exact diagonalisation.

* dense runs of `dmrg_excited` and `dmrg_tower` reproduce stored golden values and
  matrix-vector counts;
* at full rank, the first excited level of an SU(2) chain, of two-flavour Schwinger chains in
  two U(1) x SU(2) sectors and of a U(1) chain (block and symmetric tensors) equals exact
  diagonalisation, with penalty and with projected deflation;
* the sector rules and their errors are those of `dmrg`; a time budget cuts a run; bond sizes
  are reported in multiplets and in states; `start_noise` reaches a level the start misses;
  in a multiplet sector the penalty is that of one state;
* torch on the CPU agrees with numpy, also when an implicit host copy would fail.

Run as a script to rewrite the golden file: ``python tests/test_excited_symmetric.py --write``.
"""
import hashlib
import importlib
import json
import os
import platform
import sys
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from tnkit.backend import get_backend
from tnkit.block_backend import to_block_mpo
from tnkit.block_backend import to_dense_mps as block_to_dense_mps
from tnkit.dmrg import dmrg
from tnkit.dmrg_excited import dmrg_excited, dmrg_tower
from tnkit.models import tfim_mpo
from tnkit.models_lgt2f import charge_sparse_diag, isospin_sparse, schwinger2f_sparse
from tnkit.symmetric_backend import (SiteSpace, random_symmetric_mps, spin_half_site,
                                     symmetric_mpo)
from tnkit.symmetric_tensor import Space
from tnkit.symmetry import U1

from test_block_dmrg import SZ2, vector, xxz_opsum_mpo, xxz_sparse

de = importlib.import_module("tnkit.dmrg_excited")      # the module, not the function
GOLDEN = os.path.join(os.path.dirname(__file__), "data", "excited_tower_golden.json")
TIGHT = dict(n_sweeps=10, tol=1e-13, eig_tol=1e-12)
FULL = 400                                       # above every exact bond of these chains


# --------------------------------------------------------------------------- golden values
def _hex(x):
    return float(np.real(x)).hex()


def _sha(M):
    h = hashlib.sha256()
    for t in M:
        h.update(np.ascontiguousarray(np.asarray(t)).tobytes())
    return h.hexdigest()


class _Matvecs:
    """Counts the local matrix-vector products of `tnkit.dmrg_excited`, per sweep."""

    def __enter__(self):
        self.per_sweep, self._n = [], 0
        self._apply, self._sweep = de.apply_two_site, de.invariants.sweep

        def apply(*args):
            self._n += 1
            return self._apply(*args)

        def sweep(*args):
            self.per_sweep.append(self._n)
            self._n = 0
            return self._sweep(*args)
        de.apply_two_site, de.invariants.sweep = apply, sweep
        return self

    def __exit__(self, *exc):
        de.apply_two_site, de.invariants.sweep = self._apply, self._sweep


def _record(E, M, info):
    return {"E": _hex(E), "sweeps": info["sweeps"], "energies": [_hex(e) for e in info["energies"]],
            "E_penalised": _hex(info["E_penalised"]),
            "penalty_overlaps": [_hex(v) for v in info["penalty_overlaps"]],
            "discarded_weight": _hex(info["discarded_weight"]),
            "project_fallbacks": info["project_fallbacks"], "state": _sha(M)}


def golden_runs(**extra):
    """Dense `dmrg_excited` runs (truncated) and `dmrg_tower` runs (full rank), with every number
    in float.hex and every state as a hash of its bytes. `extra` goes to every call. Every retry
    decision is far from its threshold, so the runs take the same path on any machine."""
    out = {}
    W = tfim_mpo(8, 1.2)
    _, gs, _ = dmrg(W, chi=8, n_sweeps=4)
    runs = {"penalty": {}, "complex": {"backend": "numpy"}, "project": {"deflate": "project"},
            "lanczos": {"eig": "lanczos"}, "init": {"init_mps": gs, "penalty": 2.0}}
    for name, kw in runs.items():
        with _Matvecs() as n:
            run = dmrg_excited(W, 8, prev_states=[gs], n_sweeps=6, seed=3, **kw, **extra)
        out["excited-" + name] = {**_record(*run), "matvecs": n.per_sweep}
    for name, kw in {"penalty": {}, "project": {"deflate": "project"},
                     "pair_solve": {"pair_solve": True}}.items():     # levels 3, 4: a cluster
        with _Matvecs() as n:
            E, states, infos = dmrg_tower(W, 16, 5, n_sweeps=8, **kw, **extra)
        out["tower-" + name] = {
            "energies": [_hex(e) for e in E], "states": [_sha(s) for s in states],
            "levels": [{"penalty": _hex(i["penalty"]), "attempt": i.get("attempt"),
                        "sweeps": i["sweeps"],
                        "penalty_overlaps": [_hex(v) for v in i["penalty_overlaps"]]}
                       for i in infos],
            "matvecs": n.per_sweep}
    return out


def _blas():
    try:
        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except TypeError:                                          # numpy < 1.26
        info = np.__config__.get_info("blas_ilp64_opt") or np.__config__.get_info("blas_opt")
        return ",".join(info.get("libraries", []))


def machine():
    """What the golden values' last bits depend on."""
    threads = os.environ.get("OPENBLAS_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
    return {"platform": sys.platform, "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, "blas": _blas(),
            "blas_threads": threads}


def _differences(got, ref, exact, path=""):
    """Paths where `got` and `ref` differ: at all when `exact`, else by more than 1e-12
    (relative for energies, absolute for overlaps and weights), ignoring state hashes and the
    matrix-vector counts (a Lanczos convergence test near its threshold can change them).
    Sweeps and retries are always exact."""
    if not exact and path.endswith("/matvecs"):
        return []
    if isinstance(ref, dict):
        return [p for k in ref for p in _differences(got[k], ref[k], exact, f"{path}/{k}")]
    if isinstance(ref, list):
        if len(got) != len(ref):
            return [path]
        return [p for k, (g, r) in enumerate(zip(got, ref))
                for p in _differences(g, r, exact, f"{path}/{k}")]
    if exact or not isinstance(ref, str):
        return [] if got == ref else [path]
    if not ref.lstrip("-").startswith("0x"):
        return []                                # a state hash: bits only
    g, r = float.fromhex(got), float.fromhex(ref)
    absolute = "overlaps" in path or "discarded" in path
    return [] if abs(g - r) <= 1e-12 * (1.0 if absolute else max(1.0, abs(r))) else [path]


def test_dense_excited_is_unchanged():
    """Bit for bit on the machine that recorded the values; within round-off elsewhere."""
    with open(GOLDEN) as f:
        gold = json.load(f)
    exact = gold["machine"] == machine()
    runs = golden_runs()
    bad = _differences(runs, gold["runs"], exact)
    assert not bad, f"exact={exact}, {machine()}: {bad}"
    counts = {k: (v["matvecs"], gold["runs"][k]["matvecs"]) for k, v in runs.items()
              if v["matvecs"] != gold["runs"][k]["matvecs"]}
    if counts:
        warnings.warn(f"matrix-vector counts (here, recorded) differ on {machine()}: {counts}")


def test_start_noise_changes_the_dense_path():
    W = tfim_mpo(8, 1.2)
    _, gs, _ = dmrg(W, chi=8, n_sweeps=4)
    kw = dict(prev_states=[gs], n_sweeps=6, seed=3)
    noisy = _record(*dmrg_excited(W, 8, start_noise=1e-3, **kw))
    assert noisy != _record(*dmrg_excited(W, 8, **kw))


# --------------------------------------------------------------------------- chains and ED
def _spin_total(N):
    Sp = sp.csr_matrix([[0.0, 1.0], [0.0, 0.0]])
    Z = sp.diags([0.5, -0.5])

    def op(o, i):
        return sp.kron(sp.kron(sp.identity(2 ** i), o), sp.identity(2 ** (N - i - 1)), "csr")
    return sum(op(Sp, i) for i in range(N)), sum(op(Z, i) for i in range(N))


def _levels(H, idx, lift=None, k=6):
    """The lowest `k` eigenpairs of H on the basis states `idx`, with `lift` (an operator that
    is 0 on the wanted states and >= 40 on the others) added."""
    A = (H + lift if lift is not None else H)[idx][:, idx].toarray()
    e, U = np.linalg.eigh(A)
    return e[:k], U[:, :k]


class Chain:
    """One test chain: its MPO, sector, ED levels on the basis states `idx` and the map of an
    MPS to its vector there."""

    def __init__(self, name):
        from test_symmetric_backend import (X, MU, heisenberg_mpo, heisenberg_sparse,
                                            isospin_squared, state_vector, two_flavour)
        self.name = name
        if name == "SU2":
            N = 8
            self.mpo = symmetric_mpo(heisenberg_mpo(N), [spin_half_site()] * N)[0]
            self.q, sites = 0, [spin_half_site()] * N
            Sp, Sz = _spin_total(N)
            S2 = Sz @ Sz + 0.5 * (Sp @ Sp.T + Sp.T @ Sp)
            self.idx = np.flatnonzero(np.abs(Sz.diagonal()) < 1e-9)
            self.E, self.U = _levels(heisenberg_sparse(N), self.idx, 40 * S2)
        elif name.startswith("U1xSU2"):
            N, two_i = 6, int(name[-1])
            self.mpo = two_flavour(N)[1][0]
            self.q, j = (0, two_i), two_i / 2
            from tnkit.models_lgt2f import site_spaces_2f
            sites = site_spaces_2f(N)
            H, I2 = schwinger2f_sparse(N, X, MU), isospin_squared(N)
            iz = isospin_sparse(N)[0].diagonal()
            self.idx = np.flatnonzero((charge_sparse_diag(N) == 0) & (np.abs(iz - j) < 1e-9))
            lift = 40.0 * (I2 - j * (j + 1) * sp.identity(H.shape[0]))
            self.E, self.U = _levels(H, self.idx, lift)
        else:
            N = 10
            W = xxz_opsum_mpo(N, 1.0)
            self.q = 0
            if name == "U1-block":
                self.mpo = to_block_mpo(W, [SZ2] * N, get_backend("numpy-real"), sym=U1())[0]
                self._vec = lambda M: vector(block_to_dense_mps(M, [SZ2] * N))[:, None]
            else:
                sites = [SiteSpace(Space(U1(), (-1, 1), (1, 1)), np.eye(2)[::-1])] * N
                self.mpo = symmetric_mpo(W, sites)[0]
            sz = _spin_total(N)[1].diagonal()
            self.idx = np.flatnonzero(np.abs(sz) < 1e-9)
            self.E, self.U = _levels(xxz_sparse(N, 1.0), self.idx)
        if name != "U1-block":
            self._vec = lambda M: state_vector(M, sites)
        self.penalty = 2.0 * (self.E[1] - self.E[0]) + 1.0

    def weight(self, M, n):
        """The weight of the MPS `M` on the ED eigenspace of level n (levels within 1e-6)."""
        v = np.asarray(self._vec(M))[self.idx]
        v = v[:, np.argmax(np.linalg.norm(v, axis=0))]     # the multiplet state with I_z = I
        U = self.U[:, np.abs(self.E - self.E[n]) < 1e-6]
        return float(np.linalg.norm(U.T @ v) ** 2 / np.linalg.norm(v) ** 2)

    def ground(self, backend="numpy-real"):
        return dmrg(self.mpo, FULL, charge=self.q, backend=backend, **TIGHT)


CHAINS = {}


def chain(name):
    if name not in CHAINS:
        CHAINS[name] = Chain(name)
    return CHAINS[name]


def _check_level(c, E, M, info, n):
    assert abs(E - c.E[n]) < 1e-9
    assert c.weight(M, n) > 1 - 1e-8
    assert max(info["penalty_overlaps"]) < 1e-8
    assert info["converged"] and not info["cut"]


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("name", ["SU2", "U1xSU2-00", "U1xSU2-02", "U1-block", "U1-sym"])
def test_first_excited_level_equals_ed(name):
    c = chain(name)
    E0, gs, _ = c.ground()
    assert abs(E0 - c.E[0]) < 1e-9
    E, M, info = dmrg_excited(c.mpo, FULL, prev_states=[gs], charge=c.q, penalty=c.penalty,
                              **TIGHT)
    _check_level(c, E, M, info, 1)
    assert info["charge"] == c.q and info["last_bond"]["degeneracies"] == [1]
    assert info["expand"] == 0


def test_penalty_in_a_multiplet_sector_is_that_of_one_state():
    """In the isospin-1 sector (three states per multiplet), a penalty below the gap leaves the
    ground state lowest, raised by exactly the penalty."""
    c = chain("U1xSU2-02")
    E0, gs, _ = c.ground()
    w = 0.3 * (c.E[1] - c.E[0])
    E, M, info = dmrg_excited(c.mpo, FULL, prev_states=[gs], charge=c.q, penalty=w, **TIGHT)
    assert abs(info["E_penalised"] - E0 - w) < 1e-12
    assert abs(info["penalty_overlaps"][0] - 1.0) < 1e-12 and abs(E - E0) < 1e-12


@pytest.mark.parametrize("name", ["SU2", "U1xSU2-00", "U1xSU2-02"])
def test_project_deflation_on_symmetric_tensors(name):
    c = chain(name)
    _, gs, _ = c.ground()
    E, M, info = dmrg_excited(c.mpo, FULL, prev_states=[gs], charge=c.q, penalty=c.penalty,
                              deflate="project", **TIGHT)
    _check_level(c, E, M, info, 1)
    assert info["project_fallbacks"] == 0


def test_charge_and_init_rules():
    dense = xxz_opsum_mpo(6, 1.0)
    Wb = to_block_mpo(dense, [SZ2] * 6, get_backend("numpy-real"), sym=U1())[0]
    Ws = symmetric_mpo(dense, [spin_half_site()] * 6)[0]
    for W in (Wb, Ws):
        init = dmrg(W, 8, charge=0, n_sweeps=2)[1]
        with pytest.raises(ValueError, match="exactly one of charge"):
            dmrg_excited(W, 8)
        with pytest.raises(ValueError, match="exactly one of charge"):
            dmrg_excited(W, 8, charge=0, init_mps=init)
        with pytest.raises(ValueError, match="not a sector label"):
            dmrg_excited(W, 8, charge=(0, 0))
        with pytest.raises(ValueError, match="trunc='svd'"), pytest.warns(DeprecationWarning):
            dmrg_excited(W, 8, charge=0, trunc="density")
        with pytest.raises(ValueError, match="block init_mps"):
            dmrg_excited(W, 8, init_mps=dmrg(dense, 8, n_sweeps=1)[1])
        E, M, info = dmrg_excited(W, 8, init_mps=init, n_sweeps=2)
        assert info["charge"] == 0 and len(M) == 6
    with pytest.raises(ValueError, match="charge= needs a block MPO"):
        dmrg_excited(dense, 8, charge=0)
    with pytest.raises(ValueError, match="start_noise"):
        dmrg_excited(dense, 8, start_noise=-1.0)
    with pytest.raises(ValueError, match="time_budget_s"):
        dmrg_excited(dense, 8, time_budget_s=float("nan"))
    with pytest.raises(ValueError, match="no time_budget_s"):
        dmrg_tower(dense, 8, 2, time_budget_s=10.0)


def test_prev_state_of_another_sector_raises():
    c = chain("SU2")
    triplet = random_symmetric_mps(c.mpo, 4, get_backend("numpy-real"), 2)
    with pytest.raises(ValueError, match="not on the target sector"):
        dmrg_excited(c.mpo, 8, prev_states=[triplet], charge=0, n_sweeps=1)
    dense = dmrg(xxz_opsum_mpo(8, 1.0), 8, n_sweeps=2)[1]
    with pytest.raises(TypeError, match="same tensor type"):
        dmrg_excited(c.mpo, 8, prev_states=[dense], charge=0, n_sweeps=1)


def test_budget_stops_after_one_sweep():
    c = chain("SU2")
    _, gs, _ = c.ground()
    for budget in (0.0, -1.0):
        E, M, info = dmrg_excited(c.mpo, 16, prev_states=[gs], charge=0, penalty=c.penalty,
                                  n_sweeps=5, time_budget_s=budget)
        assert info["cut"] and info["sweeps"] == 1 and not info["converged"]
        assert len(info["sweep_s"]) == 1 and info["wall_s"] >= info["sweep_s"][0]
    _, _, info = dmrg_excited(c.mpo, 16, prev_states=[gs], charge=0, penalty=c.penalty,
                              n_sweeps=5, time_budget_s=1e6)
    assert not info["cut"] and info["sweeps"] > 1


def test_budget_counts_the_longest_sweep(monkeypatch):
    """On a fake clock: sweeps of 10 s and 2 s, budget 21 s. The third sweep would end at 22 s
    by the longest sweep (it would at 14 s by the last one), so it does not start."""
    clock, steps, sweep = [0.0], iter([10.0, 2.0, 2.0, 2.0]), de.invariants.sweep

    def timed(*args):
        clock[0] += next(steps)
        return sweep(*args)
    monkeypatch.setattr(de, "_now", lambda: clock[0])
    monkeypatch.setattr(de.invariants, "sweep", timed)
    _, _, info = dmrg_excited(tfim_mpo(8, 1.2), 8, n_sweeps=4, tol=0.0, time_budget_s=21.0)
    assert info["sweep_s"] == [10.0, 2.0] and info["cut"] and not info["converged"]
    assert info["wall_s"] == 12.0


def test_tower_pair_solve_takes_default_krylov_cap():
    E, _, infos = dmrg_tower(tfim_mpo(6, 1.2), 8, 2, pair_solve=True, pair_tau=10.0,
                             eig_max_iter=None, n_sweeps=3)
    assert infos[0]["pair_clusters"] == [(0, 1)] and np.all(np.diff(E) >= 0)


def test_bond_sizes_are_reported():
    for name, chi in (("U1-sym", 6), ("U1-block", 6), ("SU2", 3)):
        c = chain(name)
        _, M, info = dmrg_excited(c.mpo, chi, charge=c.q, n_sweeps=2)
        if name == "SU2":
            bonds = [t.codomain[2] for t in M[:-1]]
            assert info["max_bond"] == max(sum(V.degeneracies) for V in bonds) == chi
            assert info["max_bond_dense"] == max(V.dim for V in bonds) > chi
        else:
            assert info["max_bond"] == info["max_bond_dense"] == chi
    _, M, info = dmrg_excited(tfim_mpo(8, 1.2), 4, n_sweeps=2)
    assert info["max_bond"] == info["max_bond_dense"] == 4 and "last_bond" not in info


def test_start_noise_reaches_a_level_the_start_misses():
    """Level 2 of the SU(2) chain has the other reflection parity than level 1, so as a start it
    has no weight on level 1, and the solve deflated against the ground state stays on level 2.
    A little noise in the first sweep gives the start weight on level 1, which is then found."""
    c = chain("SU2")
    _, gs, _ = c.ground()
    kw = dict(charge=0, penalty=c.penalty, **TIGHT)
    _, l1, _ = dmrg_excited(c.mpo, FULL, prev_states=[gs], **kw)
    _, l2, _ = dmrg_excited(c.mpo, FULL, prev_states=[gs, l1],
                            **{**kw, "penalty": 2 * c.penalty})
    assert c.weight(l2, 2) > 1 - 1e-8
    kw.pop("charge")
    E, M, _ = dmrg_excited(c.mpo, FULL, prev_states=[gs], init_mps=l2, **kw)
    assert abs(E - c.E[2]) < 1e-8
    E, M, info = dmrg_excited(c.mpo, FULL, prev_states=[gs], init_mps=l2, start_noise=1e-3,
                              **kw)
    _check_level(c, E, M, info, 1)


def test_tower_on_block_and_symmetric_mpos():
    for name in ("U1-block", "SU2"):
        c = chain(name)
        E, states, infos = dmrg_tower(c.mpo, FULL, 3, penalty=c.penalty, charge=c.q, **TIGHT)
        assert np.abs(E - c.E[:3]).max() < 1e-9
        assert all(isinstance(t.data, np.ndarray) for M in states for t in M)


@pytest.mark.parametrize("name", ["SU2", "U1xSU2-02"])
def test_torch_cpu_matches_numpy(name):
    pytest.importorskip("torch")
    c = chain(name)
    _, gs, _ = c.ground()
    kw = dict(prev_states=[gs], charge=c.q, penalty=c.penalty, **TIGHT)
    E, _, _ = dmrg_excited(c.mpo, FULL, backend="numpy-real", **kw)
    Et, Mt, info = dmrg_excited(c.mpo, FULL, backend="torch-cpu-real", **kw)
    assert info["backend"].startswith("symmetric-torch")
    assert abs(Et - E) < 1e-12 * max(1.0, abs(E))


def test_hostcheck_torch_cpu(no_implicit_host_copy):
    import torch
    with pytest.raises(TypeError, match="implicit numpy conversion"):
        np.asarray(torch.zeros(2))
    c = chain("U1xSU2-00")
    _, gs, _ = c.ground()
    E, M, info = dmrg_excited(c.mpo, 32, prev_states=[gs], charge=c.q, penalty=c.penalty,
                              backend="torch-cpu-real", n_sweeps=3)
    assert np.isfinite(E) and info["backend"].startswith("symmetric-torch")
    for deflate, kw in (("project", {}), ("penalty", {"start_noise": 1e-3})):
        dmrg_excited(c.mpo, 16, prev_states=[gs], charge=c.q, penalty=c.penalty,
                     backend="torch-cpu-real", n_sweeps=1, deflate=deflate, **kw)
    dmrg_tower(c.mpo, 16, 2, charge=c.q, backend="torch-cpu-real", n_sweeps=2)


@pytest.mark.cuda
def test_cuda_matches_numpy():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    c = chain("U1xSU2-00")
    _, gs, _ = c.ground()
    kw = dict(prev_states=[gs], charge=c.q, penalty=c.penalty, **TIGHT)
    E, _, _ = dmrg_excited(c.mpo, 64, backend="numpy-real", **kw)
    Ec, Mc, _ = dmrg_excited(c.mpo, 64, backend="torch-cuda-real", **kw)
    assert abs(Ec - E) < 1e-10 * max(1.0, abs(E))
    assert all(t.data.is_cuda for t in Mc)


if __name__ == "__main__":
    if "--write" not in sys.argv:
        sys.exit("usage: python tests/test_excited_symmetric.py --write")
    with open(GOLDEN, "w") as f:
        json.dump({"machine": machine(), "runs": golden_runs()}, f, indent=1)
        f.write("\n")
    print("wrote", GOLDEN)
