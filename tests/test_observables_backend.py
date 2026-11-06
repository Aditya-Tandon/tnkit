"""Observables on a GPU-produced state, and the two O(N^2) traps that were in them.

The headline defect was that `dmrg(backend="torch-cuda")` returns
CUDA tensors while every observable was numpy-only, so solve-then-measure -- the
first program anyone writes -- threw. These tests call every public observable on a
torch-produced state, which is the SAME code path CUDA takes; the one thing a GPU
adds is the device, and that is covered by the `cuda` marks at the bottom, skipped
when no GPU is present. Those really do carry `@pytest.mark.cuda`, so `pytest -m cuda`
selects them.

The numerical half of the file is the instrument pin: the promoted `all_entropies`
must be bit-identical to per-bond `entanglement_entropy`, and the `canon=`
arguments must be exact no-ops, or the speedups have silently changed physics.
"""
from __future__ import annotations

import numpy as np
import pytest

import tnkit.models as M
import tnkit.observables as ob
from tnkit.dmrg import dmrg, right_canonicalize
from tnkit.linalg import robust_svd
from tnkit.mps import MPS
from tnkit.vumps import VUMPSOptions, vumps

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)

torch = pytest.importorskip("torch", reason="torch extra not installed")

HAS_CUDA = torch.cuda.is_available()
# Composed, not just skipif: `pytest -m cuda` is the natural selector for "re-run the
# CUDA tests on a GPU before the tag", and pyproject.toml declares the marker. Once no test
# carried it, so that selector matched zero tests and exited green -- a pass from a run
# that executed nothing, on the one check CI cannot perform.
_needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device on this runner")


def cuda_only(fn):
    """Skip without a device AND carry the declared `cuda` marker, so both work."""
    return pytest.mark.cuda(_needs_cuda(fn))


# --------------------------------------------------------------------------- #
# Fixtures: the same ground state, solved on numpy and on torch.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def state_np():
    _, mps, _ = dmrg(M.tfim_mpo(20, 1.0), chi=24, n_sweeps=40, tol=1e-13,
                     reorth="full", eig_stop="residual")
    return mps


@pytest.fixture(scope="module")
def state_torch():
    _, mps, _ = dmrg(M.tfim_mpo(20, 1.0), chi=24, n_sweeps=40, tol=1e-13,
                     reorth="full", eig_stop="residual", backend="torch-cpu")
    return mps


# --------------------------------------------------------------------------- #
# 1. The promoted single-sweep entropy profile.
# --------------------------------------------------------------------------- #
def _reference_all_entropies(mps):
    """The original `all_entropies`, verbatim -- the oracle.

    The implementation that was promoted into the library, reproduced here so
    the promotion is checked against what was actually verified and run, not
    against a paraphrase of it.
    """
    Mc = right_canonicalize([m.copy() for m in mps])
    d = Mc[0].shape[1]
    out = np.empty(len(Mc) - 1)
    carry = None
    for i in range(len(Mc) - 1):
        A = Mc[i]
        if carry is not None:
            A = np.einsum("ab,bsc->asc", carry, A, optimize=True)
        Dl, _, Dr = A.shape
        U, s, Vh = robust_svd(A.reshape(Dl * d, Dr))
        sn = s / np.linalg.norm(s)
        p = sn ** 2
        p = p[p > 1e-16]
        out[i] = float(-np.sum(p * np.log(p)))
        carry = np.diag(s) @ Vh
    return out


def test_all_entropies_bit_identical_to_per_bond(state_np):
    """An N=20 check, in the suite.

    Not a tolerance: `all_entropies` and `entanglement_entropy` share a sweep
    prefix, so equal arithmetic must give equal bits. A tolerance here would let
    a reordering through, and a reordering is exactly what would move a published
    entropy profile.
    """
    prof = ob.all_entropies(state_np)
    per_bond = np.array([ob.entanglement_entropy(state_np, b)
                         for b in range(1, len(state_np))])
    assert np.array_equal(prof, per_bond)


def test_all_entropies_matches_the_reference_implementation(state_np):
    """Promotion is bit-exact against the original source."""
    assert np.array_equal(ob.all_entropies(state_np), _reference_all_entropies(state_np))


def test_all_entropies_does_one_sweep_not_N(state_np, monkeypatch):
    """The O(N^2) trap, measured: one canonicalisation and N-1 SVDs, not N of each.

    `entanglement_entropy` called per bond canonicalises N-1 times and performs
    O(N^2) SVDs; that is what cost hours at N=600. Counting is the only way to
    assert a complexity claim in a test.
    """
    N = len(state_np)
    calls = {"canon": 0, "svd": 0}
    real_canon = ob._canon

    def counting_canon(mps, be=None):
        calls["canon"] += 1
        return real_canon(mps, be)
    monkeypatch.setattr(ob, "_canon", counting_canon)

    from tnkit.backend import NumpyBackend
    real_backend_svd = NumpyBackend.svd

    def counting_svd(self, A):
        calls["svd"] += 1
        return real_backend_svd(self, A)
    monkeypatch.setattr(NumpyBackend, "svd", counting_svd)

    calls["svd"] = 0
    ob.all_entropies(state_np)
    assert calls["canon"] == 1
    assert calls["svd"] == (N - 1) + (N - 1)          # canonicalisation + one per bond

    calls.update(canon=0, svd=0)
    for b in range(1, N):
        ob.entanglement_entropy(state_np, b)
    assert calls["canon"] == N - 1                    # the trap, still there per call
    assert calls["svd"] > 3 * (N - 1)                 # and quadratic in N


def test_entanglement_entropy_canon_argument_is_a_bit_identical_no_op(state_np):
    canon = ob._canon(state_np)
    for b in (1, 7, 13, 19):
        assert ob.entanglement_entropy(state_np, b) == \
            ob.entanglement_entropy(state_np, b, canon=canon)


def test_local_expectations_canon_argument_is_a_bit_identical_no_op(state_np):
    canon = ob._canon(state_np)
    assert np.array_equal(ob.local_expectations(state_np, SZ),
                          ob.local_expectations(state_np, SZ, canon=canon))


def test_structure_factor_canonicalises_once(state_np, monkeypatch):
    """Second trap: the diagonal pass used to canonicalise a second time.

    `structure_factor` measured <op_i^2> through `local_expectations`, which ran
    its own canonicalisation sweep. One state, one sweep.
    """
    calls = []
    real_canon = ob._canon
    monkeypatch.setattr(ob, "_canon",
                        lambda mps, be=None: (calls.append(1), real_canon(mps, be))[1])
    ob.structure_factor(state_np, SX)
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# 2. Every public observable on a torch-produced state.
# --------------------------------------------------------------------------- #
def test_dmrg_torch_state_is_not_numpy(state_torch):
    """Guards the premise: if this stops being a torch tensor the rest is vacuous."""
    assert type(state_torch[0]).__module__.split(".")[0] == "torch"


def test_solve_then_measure_on_a_torch_state(state_torch):
    """The program that was broken: solve on a device, then measure.

    Previously `'Tensor' object has no attribute 'copy'` on the entropy path and
    `can't convert cuda:0 device type tensor to numpy` on the moments path.
    """
    S = ob.entanglement_entropy(state_torch, len(state_torch) // 2)
    m2, m4, U = ob.magnetization_moments(state_torch, SX)
    assert np.isfinite(S) and S > 0
    assert np.isfinite(m2) and np.isfinite(m4) and np.isfinite(U)


@pytest.mark.parametrize("name", [
    "local_expectations", "neighbor_correlations", "two_point", "mps_overlap",
    "magnetization_moments", "structure_factor", "entanglement_entropy",
    "all_entropies", "apply_mpo", "to_numpy",
])
def test_public_observable_agrees_across_backends(name, state_np, state_torch):
    """Every public observable: torch result matches numpy to double-precision noise.

    torch and numpy dispatch to different BLAS and different SVD drivers, so this
    is a tolerance, not equality -- but it is a tight one, and it covers the
    identical code path CUDA takes.
    """
    N = len(state_np)

    def run(mps):
        if name == "local_expectations":
            return ob.local_expectations(mps, SZ)
        if name == "neighbor_correlations":
            return ob.neighbor_correlations(mps, SX, SX)
        if name == "two_point":
            return ob.two_point(mps, SX, SX, 2, list(range(3, N)))
        if name == "mps_overlap":
            return np.array([abs(ob.mps_overlap(mps, mps))])
        if name == "magnetization_moments":
            return np.array(ob.magnetization_moments(mps, SX))
        if name == "structure_factor":
            return np.array(ob.structure_factor(mps, SX))
        if name == "entanglement_entropy":
            return np.array([ob.entanglement_entropy(mps, N // 2)])
        if name == "all_entropies":
            return ob.all_entropies(mps)
        if name == "apply_mpo":
            out = ob.apply_mpo(mps, M.magnetization_mpo(N, SX))
            return np.array([abs(ob.mps_overlap(out, out))])
        if name == "to_numpy":
            hosts = ob.to_numpy(mps)
            assert all(isinstance(A, np.ndarray) for A in hosts)
            return np.array([float(np.linalg.norm(A)) for A in hosts])
        raise AssertionError(name)

    a, b = run(state_np), run(state_torch)
    assert np.allclose(a, b, rtol=1e-10, atol=1e-11), f"{name}: {a} vs {b}"


def test_to_numpy_round_trips_a_device_state(state_torch):
    """The documented escape hatch: hand a device state to numpy-only code."""
    host = ob.to_numpy(state_torch)
    assert np.allclose(ob.all_entropies(host), ob.all_entropies(state_torch),
                       rtol=1e-10, atol=1e-12)


def test_observables_do_not_mutate_the_state(state_torch):
    """`_canon` copies before canonicalising; a measurement must not gauge the caller's state."""
    before = [A.clone() for A in state_torch]
    ob.structure_factor(state_torch, SX)
    ob.all_entropies(state_torch)
    assert all(torch.equal(a, b) for a, b in zip(before, state_torch))


# --------------------------------------------------------------------------- #
# 3. The real device. Skipped everywhere except a GPU runner.
# --------------------------------------------------------------------------- #
@cuda_only
def test_cuda_solve_then_measure_smoke():
    """One short CUDA solve through every public observable.

    The CPU torch tests above take the same code path, so this only adds the
    device itself: transfers, cuSOLVER SVD, and the fact that nothing calls
    `.numpy()` on a CUDA tensor.
    """
    N = 24
    _, mps, _ = dmrg(M.tfim_mpo(N, 1.0), chi=32, n_sweeps=20, tol=1e-12,
                     reorth="full", backend="torch-cuda")
    assert mps[0].device.type == "cuda"
    # The MPS type is backend-agnostic by construction, and the CPU torch path covers
    # it -- but `dmrg` gained that return type later and the CUDA path has not been
    # exercised on a device since, so the claim is checked where it is made.
    assert isinstance(mps, MPS) and isinstance(mps, list)
    assert len(mps.entropies()) == N - 1
    S = ob.all_entropies(mps)
    m2, Sq = ob.structure_factor(mps, SX)
    mz = ob.local_expectations(mps, SZ)
    nn = ob.neighbor_correlations(mps, SX, SX)
    tp = ob.two_point(mps, SX, SX, 2, list(range(3, N)))
    _, _, U = ob.magnetization_moments(mps, SX)
    for arr in (S, mz, nn, tp, np.array([m2, Sq, U])):
        assert np.all(np.isfinite(arr))
    assert len(S) == N - 1
    assert abs(ob.mps_overlap(mps, mps)) == pytest.approx(1.0, abs=1e-8)


@cuda_only
def test_cuda_matches_cpu():
    """A CUDA-solved state and a numpy-solved state give the same observables."""
    N = 20
    kw = dict(chi=24, n_sweeps=40, tol=1e-13, reorth="full", eig_stop="residual")
    _, cpu, _ = dmrg(M.tfim_mpo(N, 1.0), **kw)
    _, gpu, _ = dmrg(M.tfim_mpo(N, 1.0), backend="torch-cuda", **kw)
    assert np.allclose(ob.all_entropies(cpu), ob.all_entropies(gpu),
                       rtol=1e-9, atol=1e-10)
    assert ob.structure_factor(cpu, SX)[0] == \
        pytest.approx(ob.structure_factor(gpu, SX)[0], rel=1e-9)


@cuda_only
def test_cuda_vumps_with_options():
    """VUMPS on the device, driven by a `VUMPSOptions` bundle.

    Two things at once, both new since the last time a GPU saw this code. The options
    object is the front door and the whole grouped-call path had only ever been
    exercised on numpy; and `vumps` is the entry point whose environments run the
    backend-native Arnoldi and GMRES, so it exercises `tnkit/krylov.py` on the device in
    a way `dmrg` does not.

    The reference is the numpy answer at the same seed. VUMPS is iterative and the two
    backends reassociate their reductions differently, so this is a tolerance, not a
    bit-identity -- 1e-9 is the bar `test_vumps_torch_cpu_matches_numpy` already uses.
    """
    opts = VUMPSOptions(n_iter=200, tol=1e-11, seed=1, backend="torch-cuda")
    gpu = vumps(1.5, 12, opts)
    cpu = vumps(1.5, chi=12, n_iter=200, tol=1e-11, seed=1)          # flat, numpy

    assert gpu["AL"].device.type == "cuda", "the state did not stay on the device"
    assert abs(gpu["e0"] - cpu["e0"]) < 1e-9, f"{gpu['e0']!r} vs {cpu['e0']!r}"
    assert abs(gpu["S"] - cpu["S"]) < 1e-8
    assert gpu["eps"] < 1e-6, f"VUMPS did not converge on the device: eps={gpu['eps']}"
    # and the two call styles agree ON THE DEVICE too, which is what the options object
    # promises and what no CPU test can say about the CUDA path
    flat = vumps(1.5, chi=12, n_iter=200, tol=1e-11, seed=1, backend="torch-cuda")
    assert flat["e0"] == gpu["e0"]


# --------------------------------------------------------------------------- #
# Two guards on the guards themselves.
# --------------------------------------------------------------------------- #
def test_every_cuda_test_carries_the_cuda_marker():
    """`pytest -m cuda` must select every CUDA test, not merely some of them.

    The release procedure says the CUDA tests are re-run on a GPU before any tag, and
    that is the one check CI provably cannot perform. A test added later without the marker
    would be silently dropped from that selection, and the run would still exit green.
    """
    import sys as _sys
    mod = _sys.modules[__name__]
    unmarked = []
    for name in dir(mod):
        if not name.startswith("test_cuda"):
            continue
        fn = getattr(mod, name)
        marks = {m.name for m in getattr(fn, "pytestmark", [])}
        if "cuda" not in marks:
            unmarked.append(name)
    assert not unmarked, ("CUDA tests without @pytest.mark.cuda, so `pytest -m cuda` "
                          "would skip them: " + ", ".join(unmarked))


def test_profile_returns_a_contiguous_array(state_np):
    """`_profile` must not hand back a stride-2 view into a complex buffer.

    `structure_factor` sums this array into a published number and numpy's pairwise
    summation is free to branch on stride; a caller reaching for `.tobytes()` or passing
    the array to C also gets a surprise from a view. Measured, not assumed: 2000 trials
    found no `np.sum` difference on numpy 1.24, which is exactly why nothing pinned it.
    """
    prof = ob.local_expectations(state_np, SZ)
    assert prof.flags["C_CONTIGUOUS"], "local_expectations returned a non-contiguous view"
    assert prof.base is None or prof.flags["OWNDATA"], "still a view into a complex buffer"


def test_to_numpy_resolves_lazy_conj_and_neg_views():
    """torch.conj()/.imag return LAZY views whose conj/neg bit makes .numpy() raise
    ("Can't call numpy() on Tensor that has conjugate bit set"). A state that ever
    stored a be.conj(...) result then poisons to_numpy and every observable transfer
    downstream -- the cascade a review hit. The backend must resolve both bits."""
    torch = pytest.importorskip("torch")
    from tnkit.backend import get_backend
    from tnkit.observables import to_numpy as mps_to_numpy

    be = get_backend("torch-cpu")
    rng = np.random.default_rng(7)
    a = rng.standard_normal((2, 2, 3)) + 1j * rng.standard_normal((2, 2, 3))
    A = be.array(a)

    C = be.conj(A)
    assert C.is_conj(), "torch stopped returning lazy conj views; test is vacuous"
    out = be.to_numpy(C)
    np.testing.assert_array_equal(out, a.conj())

    N = be.conj(A).imag          # carries view bits on this path too
    _ = be.to_numpy(N)           # must not raise

    # an MPS containing a conj-view tensor must survive the module-level transfer
    mps = [be.array(a), be.conj(be.array(a))]
    host = mps_to_numpy(mps)
    np.testing.assert_array_equal(host[1], a.conj())


# --------------------------------------------------------------------------- #
# two_site_entropy, compress, mpo_expectation's default backend.
# --------------------------------------------------------------------------- #
def _rand_mps(N, chi, seed):
    rng = np.random.default_rng(seed)
    dims = [1] + [min(chi, 2 ** min(i, N - i)) for i in range(1, N)] + [1]
    return [rng.normal(size=(dims[i], 2, dims[i + 1])) + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))
            for i in range(N)]


def _dense(mps):
    v = np.ones((1, 1))
    for A in mps:
        v = np.einsum("xa,asb->xsb", v, A).reshape(-1, A.shape[2])
    return v[:, 0] / np.linalg.norm(v[:, 0])


def test_two_site_entropy_against_dense_partial_trace():
    N, psi = 8, _rand_mps(8, 6, 11)
    v = _dense(psi)
    ref = []
    for n in range(N - 1):
        t = v.reshape(2 ** n, 4, 2 ** (N - n - 2))
        p = np.linalg.eigvalsh(np.einsum("asb,atb->st", t, t.conj()))
        p = p[p > 1e-16]
        ref.append(-np.sum(p * np.log(p)))
    S = ob.two_site_entropy(psi)
    assert np.abs(S - ref).max() <= 1e-12
    assert np.abs(ob.two_site_entropy([torch.as_tensor(A) for A in psi]) - S).max() <= 1e-12
    prod = [np.array([1.0, 0.0], dtype=complex).reshape(1, 2, 1)] * 5
    assert np.abs(ob.two_site_entropy(prod)).max() == 0
    with pytest.raises(ValueError, match="open chain"):
        ob.two_site_entropy([np.ones((2, 2, 1))] + prod[1:])


@pytest.mark.parametrize("chi_max", [64, 4])
def test_compress_fidelity_and_bonds(chi_max):
    psi = _rand_mps(12, 12, 13)
    before = [A.copy() for A in psi]
    out, disc = ob.compress(psi, chi_max)
    assert all(np.array_equal(a, b) for a, b in zip(psi, before))
    assert max(A.shape[2] for A in out) <= chi_max and disc.shape == (11,)
    infid = 1 - abs(np.vdot(_dense(psi), _dense(out))) ** 2
    if chi_max == 64:
        assert infid <= 1e-12 and disc.max() <= 1e-24
    else:
        assert 0 < infid <= 2 * disc.sum() + 1e-12
    tout, tdisc = ob.compress([torch.as_tensor(A) for A in psi], chi_max)
    assert np.abs(_dense([A.numpy() for A in tout]) - _dense(out)).max() <= 1e-12
    assert np.abs(tdisc - disc).max() <= 1e-12


def test_compress_svd_tol_keeps_every_bond_under_budget():
    psi = _rand_mps(12, 12, 14)
    out, disc = ob.compress(psi, 64, svd_tol=1e-2)
    assert disc.max() <= 1e-2 and max(A.shape[2] for A in out) < 12


def test_mpo_expectation_follows_the_state_backend():
    from tnkit.backend import get_backend
    from tnkit.dmrg_excited import mpo_expectation
    psi, W = _rand_mps(8, 6, 15), M.tfim_mpo(8, 0.7, 0.2)
    e = mpo_expectation(psi, W)
    assert e == mpo_expectation(psi, W, be=get_backend("numpy"))
    assert abs(mpo_expectation([torch.as_tensor(A) for A in psi], W) - e) <= 1e-12 * abs(e)


def test_dmrg_tower_converts_states_through_the_backend(monkeypatch):
    """CPU stand-in for a CUDA tensor: a lazily conjugated torch tensor, which
    `np.asarray` refuses just as it refuses a device tensor."""
    import importlib
    dx = importlib.import_module("tnkit.dmrg_excited")   # `tnkit.dmrg_excited` is also a function
    real = dx.dmrg_excited

    def lazy_conj(*a, **kw):
        E, Ms, info = real(*a, **kw)
        return E, [A.conj().resolve_conj().conj() for A in Ms], info

    W = M.tfim_mpo(6, 1.2, 0.1)
    kw = dict(chi=8, nlevels=2, n_sweeps=6, backend="torch-cpu")
    E_ref, S_ref, _ = dx.dmrg_tower(W, **kw)
    monkeypatch.setattr(dx, "dmrg_excited", lazy_conj)
    E, S, _ = dx.dmrg_tower(W, **kw)
    assert np.array_equal(E, E_ref)
    for P, Q in zip(S, S_ref):
        assert all(isinstance(A, np.ndarray) and np.array_equal(A, B) for A, B in zip(P, Q))


@cuda_only
def test_cuda_dmrg_tower():
    """`dmrg_tower` on the device returns host states and the numpy tower's energies."""
    from tnkit.dmrg_excited import dmrg_tower
    W = M.tfim_mpo(8, 1.2, 0.1)
    kw = dict(chi=12, nlevels=3, n_sweeps=10, tol=1e-12)
    E_cpu, _, _ = dmrg_tower(W, backend="numpy", **kw)
    E_gpu, S_gpu, _ = dmrg_tower(W, backend="torch-cuda", **kw)
    assert all(isinstance(A, np.ndarray) for P in S_gpu for A in P)
    assert np.allclose(E_gpu, E_cpu, rtol=0, atol=1e-8)
