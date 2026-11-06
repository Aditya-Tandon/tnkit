"""Regression tests for the tnkit MPS/DMRG kernel — the core correctness
invariants, consolidated from earlier validation runs so they
are checked in one place and guard against regressions from future optimisation.

Run:  python tests/test_tnkit.py      (no pytest dependency; prints PASS/FAIL)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tnkit import models as M
from tnkit.models import SX, SZ, I2
from tnkit.dmrg import (dmrg, apply_two_site, random_mps, right_canonicalize,
                        _cutn_truncate)
from tnkit.eigensolver import lanczos_ground, lanczos_ground_restart, lobpcg_ground
from tnkit.backend import NumpyBackend, get_backend
from tnkit.linalg import TRUNCATIONS
from tnkit.free_fermion import tfim_free_fermion_energy
from tnkit.observables import (local_expectations, neighbor_correlations,
                               entanglement_entropy, two_point, mps_overlap,
                               structure_factor, magnetization_moments)


def _op_at(op, site, N):
    m = np.array([[1.0 + 0j]])
    for k in range(N):
        m = np.kron(m, op if k == site else I2)
    return m


def test_mpo_equals_hamiltonian():
    for N, g in [(4, 1.0), (6, 0.5), (8, 1.5)]:
        diff = np.linalg.norm(M.tfim_dense(N, g) - M.mpo_to_dense(M.tfim_mpo(N, g)))
        assert diff < 1e-10, f"MPO != H at N={N} g={g}: {diff}"


def test_eigensolvers_vs_dense():
    rng = np.random.default_rng(0); n = 200
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (A + A.conj().T)
    true = float(np.linalg.eigvalsh(H)[0])
    ops = NumpyBackend(); v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    el, _, _ = lanczos_ground(lambda x: H @ x, ops.array(v0), ops, tol=1e-10)
    eb, _, _ = lobpcg_ground(lambda x: H @ x, ops.array(v0), ops, tol=1e-10)
    assert abs(el - true) < 1e-8, f"lanczos {el} vs {true}"
    assert abs(eb - true) < 1e-8, f"lobpcg {eb} vs {true}"


def test_lanczos_breakdown_safe():
    # warm-starting AT an eigenvector must not crash (beta -> 0)
    rng = np.random.default_rng(1); n = 60
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (A + A.conj().T)
    w, V = np.linalg.eigh(H); ops = NumpyBackend()
    e, _, info = lanczos_ground(lambda x: H @ x, ops.array(V[:, 0]), ops, tol=1e-10)
    assert abs(e - w[0]) < 1e-8 and info["iters"] <= 3


def test_restart_returns_eigenvector_at_restart_boundary():
    # Regression: exiting on the max_iter bound immediately after a thick restart used to
    # reconstruct x from the PRE-restart Ritz coefficients against the new (shorter) basis --
    # zip() truncated silently and returned a non-eigenvector while theta still looked
    # converged, so no energy check could catch it. The returned x must satisfy the same
    # residual bound as full-reorth Lanczos at the same iteration count.
    rng = np.random.default_rng(0); n = 300
    A = rng.standard_normal((n, n)); H = 0.5 * (A + A.T)
    ops = NumpyBackend()
    v0 = rng.standard_normal(n); v0 /= np.linalg.norm(v0)
    m = 40
    for max_iter in (m, m + 1, m + 5):          # m is exactly the first restart boundary
        theta, x, info = lanczos_ground_restart(lambda y: H @ y, v0.copy(), ops,
                                                tol=1e-14, max_iter=max_iter, m=m, k=10)
        x = np.asarray(x); x = x / np.linalg.norm(x)
        res = np.linalg.norm(H @ x - theta * x)
        assert info["restarts"] >= 1, f"expected a restart at max_iter={max_iter}"
        assert res < 1e-2, f"restart max_iter={max_iter}: ||Hx-theta x||={res:.3e} (not an eigenvector)"


def test_free_fermion_dynamics_vs_ed():
    # Exact free-fermion quench dynamics must reproduce dense ED. The Jordan-Wigner sign
    # bookkeeping is the whole risk here, and it is the ground truth every dynamics benchmark
    # will be measured against, so it is checked on three different quenches.
    from tnkit.free_fermion_dynamics import validate_against_ed
    for N, g0, g1 in ((8, 2.0, 1.0), (8, 0.5, 1.5), (8, 1.0, 0.3)):
        assert validate_against_ed(N=N, g0=g0, g1=g1, times=(0.0, 0.3, 1.5),
                                   tol=1e-9, verbose=False), f"free-fermion != ED for {N} {g0}->{g1}"


def test_free_fermion_dynamics_bounded_at_large_N():
    # Regression: a per-timestep dense expm blew up at ISOLATED t for large N (|<sigma^z>| ~ 1e14
    # at N=200, t=0.503, while neighbouring t were fine), which the ED check at N=8-10 could never
    # have caught. Observables are bounded by 1 by construction, so assert that directly at the
    # large N the ground-truth instrument is actually used at.
    import numpy as np
    from tnkit.free_fermion_dynamics import quench_observables
    for N in (64, 200):
        rec = quench_observables(N, 2.0, 1.0, np.linspace(0, 8, 40))
        for r in rec:
            assert abs(r["sz"]) <= 1.0 + 1e-9, f"N={N} t={r['t']}: |sz|={abs(r['sz']):.3e} > 1"
            assert abs(r["sxx"]) <= 1.0 + 1e-9, f"N={N} t={r['t']}: |sxx|={abs(r['sxx']):.3e} > 1"


def test_imaginary_time_tebd_is_real_and_finds_ground_state():
    # Imaginary-time TEBD uses exp(-h tau/2) on a real symmetric bond Hamiltonian, so the gates
    # are REAL -- no complex arithmetic anywhere. Check both that the gates are real dtype and
    # that the evolution converges to the ground state on the real backend.
    import numpy as np
    from tnkit.tebd import tebd_evolve, tfim_bond_gates
    from tnkit.dmrg import random_mps, right_canonicalize
    from tnkit.observables import apply_mpo, mps_overlap
    from tnkit.free_fermion import tfim_free_fermion_energy
    from tnkit.backend import get_backend
    assert not np.iscomplexobj(tfim_bond_gates(6, 1.0, 0.05, imaginary=True)[0])
    N, g, chi = 12, 1.0, 24
    exact = tfim_free_fermion_energy(N, g)
    be = get_backend("numpy-real")
    mps = [be.to_numpy(t) for t in right_canonicalize(
        random_mps(N, chi, 2, np.random.default_rng(0), be), be)]
    # tau = 600*0.05 = 30 is past the imaginary-time transient; the residual 1e-6 is the
    # second-order Trotter floor at dt=0.05, not a dtype effect (it is identical on the complex
    # backend and does not improve past tau ~ 60).
    out, _ = tebd_evolve(mps, N, g, chi, 0.05, 600, backend="numpy-real", imaginary=True)
    m = [be.to_numpy(t) for t in out]
    E = float(np.real(mps_overlap(m, apply_mpo(m, M.tfim_mpo(N, g))) / mps_overlap(m, m)))
    assert abs(E - exact) < 1e-5, f"imag-time TEBD E={E} vs exact {exact}"


def test_dmrg_vs_exact_diag():
    from scipy.sparse.linalg import eigsh
    for N, g in [(8, 1.0), (10, 0.5), (10, 1.5), (12, 1.0)]:
        E, _, _ = dmrg(M.tfim_mpo(N, g), chi=24, n_sweeps=16, tol=1e-12, reorth="full")
        E_ed = eigsh(M.tfim_dense(N, g), k=1, which="SA", tol=1e-14)[0][0]   # Lanczos ED
        assert abs(E - E_ed) < 1e-9, f"DMRG N={N} g={g}"


def test_truncation_methods_reconstruct():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((32, 16)) + 1j * rng.standard_normal((32, 16))
    for name, fn in TRUNCATIONS.items():
        U, S, Vh, _ = fn(A, 16)                      # full rank -> exact
        err = np.linalg.norm(A - (U * S) @ Vh) / np.linalg.norm(A)
        assert err < 1e-8, f"{name} reconstruction err {err}"


def test_randomized_svd_device_matches_svd():
    # backend-native randomised truncation must match full SVD (and exact) for the
    # fast-decaying two-site blocks DMRG actually produces, on device (no host
    # round-trip), for both numpy and torch-cpu.
    N, g, chi = 20, 1.0, 16
    Eex = tfim_free_fermion_energy(N, g)
    E_svd, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, n_sweeps=18, tol=1e-12,
                       reorth="full", trunc="svd", backend="numpy")
    for be in ["numpy", "torch-cpu"]:
        # The numpy leg above and this loop's numpy pass have already run when the
        # skip fires, so a torch-free environment loses only the torch coverage --
        # and the suite stops ERRORING on an unguarded import, which is what let it
        # claim the core needs no torch without ever demonstrating it.
        if be.startswith("torch"):
            import pytest as _pt
            _pt.importorskip("torch")
        E_r, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, n_sweeps=18, tol=1e-12,
                         reorth="full", trunc="randomized", backend=be)
        assert abs(E_r - Eex) < 1e-9, f"randomized dE ({be}) {abs(E_r - Eex)}"
        assert abs(E_r - E_svd) < 1e-9, f"randomized vs svd ({be}) {abs(E_r - E_svd)}"


def test_cutn_truncate_host_array_raises():
    # trunc='cutn' (cuTensorNet hybrid) is a GPU-only path: it exists to keep the two-site SVD on the
    # NVIDIA device. Handing it a host numpy array is a configuration error and must fail loudly
    # with an actionable message, not silently round-trip or throw a cryptic cuTensorNet error.
    # (On a CPU-only env without cuquantum the import guard fires first — also a RuntimeError
    # naming 'cutn' — so this contract holds everywhere.)
    A = np.random.default_rng(0).standard_normal((16, 8))
    try:
        _cutn_truncate(A, 8)
    except RuntimeError as e:
        assert "cutn" in str(e), f"unhelpful cutn error: {e}"
    else:
        raise AssertionError("trunc='cutn' on a host numpy array must raise RuntimeError")


def test_dmrg_cutn_matches_svd_gpu():
    # GPU-only: the cuTensorNet hybrid (trunc='cutn', tnkit eig+env + cuTensorNet SVD) must reproduce the
    # full-SVD ground energy / free-fermion exact. Skips cleanly where cuquantum or a CUDA device
    # is absent (CPU CI) so the suite stays green off-GPU; the real check runs on a GPU.
    try:
        import torch
        from cuquantum.tensornet.tensor import decompose  # noqa: F401
        if not torch.cuda.is_available():
            raise ImportError("no CUDA device")
    except Exception as e:                                # noqa
        print(f"  [skip] test_dmrg_cutn_matches_svd_gpu: {e}")
        return
    N, g, chi = 20, 1.0, 16
    Eex = tfim_free_fermion_energy(N, g)
    E_svd, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, n_sweeps=18, tol=1e-12,
                       reorth="full", trunc="svd", backend="numpy")
    E_c, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, n_sweeps=18, tol=1e-12,
                     reorth="full", trunc="cutn", backend="torch-cuda")
    assert abs(E_c - Eex) < 1e-9, f"cutn dE {abs(E_c - Eex)}"
    assert abs(E_c - E_svd) < 1e-9, f"cutn vs svd {abs(E_c - E_svd)}"


def test_right_canonicalize_large_N_no_overflow():
    # right_canonicalize pushes a U*S factor (scale ~sqrt(d*chi)) left at every site,
    # so the running global scale grew geometrically (~1e235 by N=200) and its
    # sum-of-squares norm overflowed to inf -> state collapsed to 0 -> NaN, crashing
    # DMRG at N>~200. Per-step renormalisation caps the scale; verify the canonical
    # form is still exact and everything stays finite well past the old failure point.
    be = NumpyBackend()
    N, chi = 240, 64
    Mps = right_canonicalize(random_mps(N, chi, 2, np.random.default_rng(0), be), be)
    assert all(np.all(np.isfinite(t)) for t in Mps), "non-finite tensor after canonicalise"
    B = Mps[N // 2]                                   # bulk tensor: sum_s B B^H = I
    G = np.einsum("asb,csb->ac", B, B.conj())
    assert np.max(np.abs(G - np.eye(B.shape[0]))) < 1e-10, "not right-canonical"


def test_randomized_mixed_matches_svd():
    # Mixed-precision truncation: fp32 range finder + fp64 core. The complex64
    # sketch must not spoil DMRG accuracy — the retained subspace stays fp64-good, so
    # the energy tracks full SVD / free-fermion exact. On CPU this reproduces the exact
    # fp32 numerics a GPU would use (the accuracy check; speed is measured on GPU).
    N, g, chi = 20, 1.0, 16
    Eex = tfim_free_fermion_energy(N, g)
    E_svd, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, n_sweeps=18, tol=1e-12,
                       reorth="full", trunc="svd", backend="numpy")
    for be in ["numpy", "torch-cpu"]:
        if be.startswith("torch"):            # skip the torch leg, do not error on it
            import pytest as _pt
            _pt.importorskip("torch")
        E_m, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, n_sweeps=18, tol=1e-12,
                         reorth="full", trunc="randomized-mixed", backend=be)
        assert abs(E_m - Eex) < 1e-7, f"mixed dE ({be}) {abs(E_m - Eex)}"
        assert abs(E_m - E_svd) < 1e-7, f"mixed vs svd ({be}) {abs(E_m - E_svd)}"


def test_tebd_quench_vs_exact_diag():
    from scipy.linalg import expm
    from tnkit.tebd import tebd_evolve
    N, g0, g1 = 8, 0.5, 1.5
    _, mps, _ = dmrg(M.tfim_mpo(N, g0), chi=16, n_sweeps=20, tol=1e-12, reorth="full")
    mid = N // 2
    _, rec = tebd_evolve(mps, N, g1, chi=32, dt=0.02, n_steps=50,
                         measure=lambda m: local_expectations(m, SZ)[mid], meas_every=10)
    _, V = np.linalg.eigh(M.tfim_dense(N, g0)); psi0 = V[:, 0]
    Hd1 = M.tfim_dense(N, g1)
    for t, sz_tebd in rec:
        psit = expm(-1j * Hd1 * t) @ psi0
        sze = (psit.conj() @ _op_at(SZ, mid, N) @ psit).real
        assert abs(sz_tebd - sze) < 1e-3, f"TEBD vs exact at t={t}: {abs(sz_tebd-sze)}"


def test_longrange_mpo_and_2d_cylinder():
    # NN long-range builder == tfim_mpo; 2D cylinder MPO == dense 2D H
    nn = M.longrange_mpo(6, [(i, i + 1) for i in range(5)], 1.0)
    assert np.linalg.norm(M.mpo_to_dense(nn) - M.mpo_to_dense(M.tfim_mpo(6, 1.0))) < 1e-10
    for Lx, Ly in [(2, 3), (3, 3)]:
        mpo, _, _ = M.tfim2d_cylinder(Lx, Ly, 1.0)
        assert np.linalg.norm(M.mpo_to_dense(mpo) - M.tfim2d_cylinder_dense(Lx, Ly, 1.0)) < 1e-10


def test_dmrg_2d_cylinder_vs_exact_diag():
    Lx, Ly, g = 3, 3, 2.0
    mpo, _, _ = M.tfim2d_cylinder(Lx, Ly, g)
    E, _, _ = dmrg(mpo, chi=48, n_sweeps=30, tol=1e-11, reorth="full")
    E0 = float(np.linalg.eigvalsh(M.tfim2d_cylinder_dense(Lx, Ly, g))[0])
    assert abs(E - E0) < 1e-8


def test_parity_sector_targeting():
    # parity MPO == prod sz; penalised DMRG gives a definite-parity ground state whose
    # physical energy matches the exact +parity-sector ground energy.
    from tnkit.observables import apply_mpo, mps_overlap
    N, g, lam = 10, 0.6, 0.2
    Pd = np.array([[1.0 + 0j]])
    for _ in range(N):
        Pd = np.kron(Pd, SZ)
    assert np.linalg.norm(M.mpo_to_dense(M.parity_mpo(N)) - Pd) < 1e-10
    w, V = np.linalg.eigh(M.tfim_dense(N, g))
    par = np.array([np.real(V[:, j].conj() @ Pd @ V[:, j]) for j in range(len(w))])
    Eplus = float(w[np.argmin(np.where(par > 0.5, w, 1e9))])
    _, mp, _ = dmrg(M.tfim_parity_mpo(N, g, lam), chi=32, n_sweeps=30, tol=1e-11, reorth="full")
    assert abs(mps_overlap(mp, apply_mpo(mp, M.parity_mpo(N))).real - 1.0) < 1e-6
    EH = mps_overlap(mp, apply_mpo(mp, M.tfim_mpo(N, g))).real
    assert abs(EH - Eplus) < 1e-8


def test_idmrg_energy_density_gapped():
    # iDMRG thermodynamic-limit energy density vs exact free-fermion (gapped phase,
    # where pure-growth infinite-system DMRG is accurate).
    from tnkit.idmrg import idmrg
    from tnkit.free_fermion import tfim_free_fermion_energy
    for g in [0.5, 1.5]:
        e0 = idmrg(g, chi=24, n_iter=400, tol=1e-12)["e0"]
        ex = tfim_free_fermion_energy(4000, g) / 4000
        assert abs(e0 - ex) < 1e-3, f"iDMRG e0 at g={g}: {abs(e0-ex)}"


def test_vumps_energy_density_vs_exact():
    # VUMPS is variationally optimal on the uniform manifold, so its thermodynamic-limit
    # energy density must hit the exact free-fermion value to ~machine precision at the
    # gapped points (where a modest chi captures the small correlation length). This is
    # the acid test for the left/right MPO environments (a wrong right transfer matrix
    # gives e0R != e0L and the energy drifts) — see tnkit/vumps.py.
    from tnkit.vumps import vumps
    from scipy.integrate import quad
    # exact thermodynamic-limit (infinite, translation-invariant) energy density:
    # e0 = -(1/2pi) int_-pi^pi sqrt(1+g^2-2g cos k) dk  (gives -4/pi at g=1). This is the
    # right reference for VUMPS; a finite open-chain free-fermion energy carries an
    # O(1/N) boundary correction that VUMPS (bulk, infinite) correctly does not have.
    def e0_exact(g):
        return -1 / (2 * np.pi) * quad(lambda k: np.sqrt(1 + g**2 - 2*g*np.cos(k)), -np.pi, np.pi)[0]
    for g in [0.5, 1.5]:
        r = vumps(g, chi=16, n_iter=200, tol=1e-11, seed=1)
        assert abs(r["e0"] - e0_exact(g)) < 1e-8, f"VUMPS e0 at g={g}: {abs(r['e0']-e0_exact(g))}"
        assert r["eps"] < 1e-6, f"VUMPS not converged at g={g}: eps={r['eps']}"


def test_vumps_ramping_matches_random():
    # chi-ramping (seed a larger-chi run from the converged smaller-chi state via init=...)
    # must reach the same ground state as a random start. Validated by the energy density;
    # ramping is what tames the near-critical fixed-point stiffness so large chi converges.
    from tnkit.vumps import vumps
    r8 = vumps(1.5, chi=8, n_iter=150, tol=1e-11, seed=1)
    ramp = vumps(1.5, chi=16, n_iter=150, tol=1e-11, seed=1,
                 init={"AL": r8["AL"], "AR": r8["AR"], "C": r8["C"]})
    rand = vumps(1.5, chi=16, n_iter=150, tol=1e-11, seed=1)
    assert abs(ramp["e0"] - rand["e0"]) < 1e-9, f"ramped {ramp['e0']} vs random {rand['e0']}"


def test_vumps_critical_entanglement_grows():
    # at criticality (g=1) the finite-chi entanglement and correlation length must grow
    # with chi (finite-entanglement scaling); the energy stays close to -4/pi. This
    # guards the FES observables S (from the bond matrix C) and xi (transfer-matrix gap).
    from tnkit.vumps import vumps
    r8 = vumps(1.0, chi=8, n_iter=400, tol=1e-10, seed=1)
    r16 = vumps(1.0, chi=16, n_iter=400, tol=1e-10, seed=1)
    assert r16["S"] > r8["S"] > 0, f"S must grow with chi: {r8['S']} -> {r16['S']}"
    assert r16["xi"] > r8["xi"] > 0, f"xi must grow with chi: {r8['xi']} -> {r16['xi']}"
    assert abs(r16["e0"] + 4 / np.pi) < 1e-6, f"critical e0 off: {r16['e0']} vs {-4/np.pi}"


def test_vumps_torch_cpu_matches_numpy():
    # the VUMPS GPU port: every O(D^3) contraction (transfer einsums, environment
    # recursions, both Lanczos eigensolves, polar SVDs) plus the backend-native Arnoldi
    # fixed-point and GMRES energy channel (tnkit/krylov.py) must run on the torch backend
    # and reproduce the numpy result to ~1e-9. torch-cpu == numpy (both double) then
    # certifies the torch-cuda path by construction (identical code, different device).
    try:
        import torch  # noqa
    except ImportError:
        return
    from tnkit.vumps import vumps
    for g in [0.5, 1.5]:                     # gapped: converges fast at modest chi
        rn = vumps(g, chi=12, n_iter=200, tol=1e-11, seed=1, backend="numpy")
        rt = vumps(g, chi=12, n_iter=200, tol=1e-11, seed=1, backend="torch-cpu")
        assert abs(rn["e0"] - rt["e0"]) < 1e-9, f"vumps e0 numpy vs torch g={g}: {abs(rn['e0']-rt['e0'])}"
        assert abs(rn["S"] - rt["S"]) < 1e-8, f"vumps S numpy vs torch g={g}: {abs(rn['S']-rt['S'])}"


def test_lanczos_expm_vs_exact():
    from scipy.linalg import expm as _expm
    from tnkit.eigensolver import lanczos_expm
    rng = np.random.default_rng(3); n = 120
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); H = 0.5 * (A + A.conj().T)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n); op = NumpyBackend()
    approx = lanczos_expm(lambda x: H @ x, op.array(v), -1j * 0.3, op, m=40)
    exact = _expm(-1j * 0.3 * H) @ v
    assert np.linalg.norm(approx - exact) / np.linalg.norm(exact) < 1e-10


def test_tdvp_quench_vs_exact_diag():
    from scipy.linalg import expm as _expm
    from tnkit.tdvp import tdvp2_evolve
    N, g0, g1 = 8, 0.5, 1.5
    _, mps, _ = dmrg(M.tfim_mpo(N, g0), chi=16, n_sweeps=20, tol=1e-12, reorth="full")
    mid = N // 2
    _, rec = tdvp2_evolve(mps, M.tfim_mpo(N, g1), chi=32, dt=0.05, n_steps=30,
                          measure=lambda m: local_expectations(m, SZ)[mid], meas_every=10)
    _, V = np.linalg.eigh(M.tfim_dense(N, g0)); psi0 = V[:, 0]; Hd1 = M.tfim_dense(N, g1)
    for t, sz in rec:
        psit = _expm(-1j * Hd1 * t) @ psi0
        assert abs(sz - (psit.conj() @ _op_at(SZ, mid, N) @ psit).real) < 1e-3


def test_adaptive_dt_is_opt_in_and_lands_on_the_grid():
    """adaptive_dt=None must reproduce the fixed-dt path, and an adaptive run must land on
    the SAME measurement times. Both are load-bearing for every A/B in the dynamics direction --
    a controller that shifts the measurement grid makes its own trajectory incomparable."""
    from tnkit.tdvp import tdvp1_evolve, tdvp2_evolve
    N, chi = 10, 12
    _, mps, _ = dmrg(M.tfim_mpo(N, 2.0), chi=chi, n_sweeps=12, tol=1e-12, reorth="full")
    mpo = M.tfim_mpo(N, 1.0)
    meas = lambda m: np.real(np.asarray(local_expectations(m, SZ)))
    for name, run in (("1TDVP", lambda **k: tdvp1_evolve(mps, mpo, 0.05, 12, measure=meas,
                                                         meas_every=4, **k)),
                      ("2TDVP", lambda **k: tdvp2_evolve(mps, mpo, chi, 0.05, 12, measure=meas,
                                                         meas_every=4, **k))):
        _, ref = run()
        # controller pinned (fmin=fmax=1) -> same arithmetic, meter on
        _, pinned = run(adaptive_dt={"fmin": 1.0, "fmax": 1.0, "tol": 1.0})
        assert len(ref) == len(pinned), name
        for (t0, a), (t1, b) in zip(ref, pinned):
            assert t0 == t1, f"{name} time grid moved"
            assert np.max(np.abs(a - b)) < 1e-12, f"{name} adaptive_dt is not opt-in"
        # controller running free -> different step sizes, IDENTICAL measurement times
        log = []
        _, free = run(adaptive_dt={"tol": None, "tol_scale": 9.0, "log": log})
        assert len(free) == len(ref), name
        assert all(t0 == t1 for (t0, _), (t1, _) in zip(ref, free)), f"{name} overshot a target"
        assert len({round(r["dt"], 10) for r in log}) > 1, f"{name} controller never moved dt"


def test_free_fermion_ramp_reference_is_orthogonal_and_converged():
    """The time-dependent reference must reduce to the spectral form at constant g, and be
    converged in its own step size -- otherwise 'exact' is just another integrator."""
    from tnkit.free_fermion_dynamics import (gamma_ground, evolve, evolve_td,
                                             ramp_reference_convergence)
    N = 12
    g0 = gamma_ground(N, 2.0)
    ts = [0.0, 0.7, 2.1]
    a = [G for _, G in evolve(g0, N, 1.0, ts)]
    b = [G for _, G in evolve_td(g0, N, lambda t: 1.0, ts, dt_ref=0.01)]
    assert max(np.max(np.abs(x - y)) for x, y in zip(a, b)) < 1e-12
    ramp = lambda t: 2.0 - 1.0 * min(t / 2.0, 1.0)
    assert ramp_reference_convergence(N, ramp, 3.0, dt_ref=0.005) < 1e-9


def test_backend_torch_cpu_matches_numpy():
    # the torch-cpu backend must reproduce the numpy DMRG energy (both double)
    try:
        import torch  # noqa
    except ImportError:
        return
    for N, g in [(10, 1.0), (10, 1.5)]:
        En, _, _ = dmrg(M.tfim_mpo(N, g), chi=20, n_sweeps=16, tol=1e-12,
                        reorth="full", backend="numpy")
        Et, _, _ = dmrg(M.tfim_mpo(N, g), chi=20, n_sweeps=16, tol=1e-12,
                        reorth="full", backend="torch-cpu")
        assert abs(En - Et) < 1e-9, f"numpy vs torch-cpu {abs(En-Et)}"


def test_dmrg_reorth_modes_agree():
    ref = M.tfim_ground_energy(10, 1.0)
    for mode in ["full", "none"]:
        E, _, _ = dmrg(M.tfim_mpo(10, 1.0), chi=16, n_sweeps=14, tol=1e-12, reorth=mode)
        assert abs(E - ref) < 1e-9, f"reorth={mode}"
    # reorth="periodic" is REFUTED: it loses basis
    # orthogonality on the long local solves that appear at chi >= 64 near criticality, giving a
    # ghost eigenpair and dE ~ 1-2 (no ground state). It is NOT tested for agreement any more --
    # it is tested for being REJECTED with the residual stopping rule, and for warning otherwise.
    import warnings as _w
    import pytest as _pt
    with _pt.raises(ValueError, match="refuted"):
        dmrg(M.tfim_mpo(10, 1.0), chi=16, n_sweeps=2, reorth="periodic", eig_stop="residual")
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        dmrg(M.tfim_mpo(10, 1.0), chi=16, n_sweeps=2, reorth="periodic", eig_stop="energy")
        assert any("refuted" in str(x.message) for x in rec), "periodic should warn"


def test_free_fermion_vs_exact_diag():
    for N, g in [(6, 0.5), (8, 1.0), (10, 1.5)]:
        assert abs(tfim_free_fermion_energy(N, g) - M.tfim_ground_energy(N, g)) < 1e-9


def test_observables_vs_exact_diag():
    for N, g in [(10, 0.5), (10, 1.5)]:
        H = M.tfim_dense(N, g); _, V = np.linalg.eigh(H); gs = V[:, 0]
        z_ex = np.array([(gs.conj() @ _op_at(SZ, i, N) @ gs).real for i in range(N)])
        xx_ex = np.array([(gs.conj() @ (_op_at(SX, i, N) @ (_op_at(SX, i + 1, N) @ gs))).real
                          for i in range(N - 1)])
        _, mps, _ = dmrg(M.tfim_mpo(N, g), chi=24, n_sweeps=16, tol=1e-12, reorth="full")
        z = local_expectations(mps, SZ); xx = neighbor_correlations(mps, SX, SX)
        assert np.max(np.abs(z - z_ex)) < 1e-7, f"<sz> N={N} g={g}"
        assert np.max(np.abs(xx - xx_ex)) < 1e-7, f"<sxsx> N={N} g={g}"
        S = entanglement_entropy(mps, N // 2)
        assert 0.0 <= S < np.log(2 ** (N // 2)) + 1e-9


def test_structure_factor_vs_exact_diag():
    N, g = 10, 1.0
    H = M.tfim_dense(N, g); _, V = np.linalg.eigh(H); gs = V[:, 0]
    xs = [_op_at(SX, i, N) @ gs for i in range(N)]       # <X_i X_j> = (X_i gs)^dag (X_j gs)
    M2ex = sum((xs[i].conj() @ xs[j]).real for i in range(N) for j in range(N)) / N**2
    _, mps, _ = dmrg(M.tfim_mpo(N, g), chi=24, n_sweeps=16, tol=1e-12, reorth="full")
    m2, S = structure_factor(mps, SX)
    assert abs(m2 - M2ex) < 1e-7 and abs(S - N**2 * m2) < 1e-9


def test_magnetization_moments_vs_exact_diag():
    N, g = 8, 1.0
    H = M.tfim_dense(N, g); _, V = np.linalg.eigh(H); gs = V[:, 0]
    Mt = sum(_op_at(SX, i, N) for i in range(N))
    m2ex = (gs.conj() @ (Mt @ Mt) @ gs).real
    m4ex = (gs.conj() @ (Mt @ Mt @ Mt @ Mt) @ gs).real
    _, mps, _ = dmrg(M.tfim_mpo(N, g), chi=24, n_sweeps=16, tol=1e-12, reorth="full")
    m2, m4, U = magnetization_moments(mps, SX)
    _, S = structure_factor(mps, SX)
    assert abs(m2 - m2ex) < 1e-6 and abs(m4 - m4ex) < 1e-5
    assert abs(m2 - S) < 1e-9                      # <M^2> == structure factor
    assert abs(U - (1 - m4ex / (3 * m2ex**2))) < 1e-6


def test_lanczos_expm_fixed_matches_dense_and_survives_breakdown():
    """The fixed-schedule Krylov exponential handles Lanczos breakdown WITHOUT a host branch.

    A fixed schedule cannot test `beta < tol` and stop, because the test is a device->host
    synchronisation and deleting those is the point of the path. It does not need to: basis vector
    j enters the answer through `expm(tau T)[j, 0]`, which is bounded by beta_1...beta_j / (j-1)!,
    so a breakdown at step p suppresses every later (meaningless) vector to roundoff on its own.
    The only outcome that must be prevented is 0/0, which `guard_denom` handles arithmetically.

    The adversarial input is an EXACT EIGENVECTOR: the Krylov space is one-dimensional and
    beta_1 = 0 at the first step, so every remaining iteration runs on a degenerate basis. The
    result must still be exact and finite. Also covers a zero input, a schedule longer than the
    vector (m > n, which happens at a chain boundary), and the 2-dimensional block that is the
    smallest zero-site solve.
    """
    from scipy.linalg import expm as dexpm
    from tnkit.eigensolver import lanczos_expm_fixed
    for backend in ("numpy", "torch-cpu"):
        try:
            be = get_backend(backend)
        except Exception:                              # torch not installed
            continue
        rng = np.random.default_rng(0)
        n = 24
        Hh = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Hh = Hh + Hh.conj().T
        _, V = np.linalg.eigh(Hh)
        H = be.array(Hh)
        mv = lambda x: be.einsum("ij,j->i", H, x)
        tau = -1j * 0.05
        U = dexpm(tau * Hh)

        # (a) immediate breakdown, at three schedule lengths
        v = be.array(V[:, 3].astype(complex))
        ref = U @ V[:, 3]
        for m in (4, 10, 30):
            out = be.to_numpy(lanczos_expm_fixed(mv, v, tau, be, m))
            assert np.all(np.isfinite(out)), f"{backend} m={m}: non-finite after breakdown"
            assert np.abs(out - ref).max() < 1e-12, f"{backend} m={m}: {np.abs(out-ref).max():.2e}"

        # (b) zero input must give zero, not NaN
        out = be.to_numpy(lanczos_expm_fixed(mv, be.array(np.zeros(n, dtype=complex)), tau, be, 10))
        assert np.all(np.isfinite(out)) and np.abs(out).max() == 0.0

        # (c) schedule longer than the vector
        g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(complex)
        out = be.to_numpy(lanczos_expm_fixed(mv, be.array(g), tau, be, 60))
        assert np.abs(out - U @ g).max() < 1e-12, f"{backend} m>n"

        # (d) the smallest block a zero-site solve can produce
        H2 = np.array([[1.0, 0.3], [0.3, -0.7]], dtype=complex)
        H2d = be.array(H2)
        out = be.to_numpy(lanczos_expm_fixed(lambda x: be.einsum("ij,j->i", H2d, x),
                                             be.array(np.array([1.0, 0.0], dtype=complex)),
                                             tau, be, 10))
        assert np.abs(out - dexpm(tau * H2) @ np.array([1.0, 0.0])).max() < 1e-12


def test_capturable_expm_routes_match_scipy_and_each_other():
    """The two CUDA-graph-capturable exponentials must agree with `scipy.linalg.expm`.

    `torch.matrix_exp` cannot go inside a CUDA graph: it selects its Pade degree from
    ||A||_1 read back to the HOST, and a host read during capture is a hard error
    (measured). `torch.linalg.eigh` fails for the same
    reason. So `lanczos_expm_fixed` grew two host-branch-free routes, and this test
    is where their correctness is actually gated -- on CPU, because the GPU is the
    machine it is hardest to test on.

    Both exploit the structure the fixed schedule already produces: the Krylov tridiagonal
    T is real symmetric and tau is pure imaginary, so tau*T is anti-Hermitian and
    exp(tau*T) is unitary -- which is why the Taylor route can over-scale safely.
    """
    from scipy.linalg import expm as dexpm
    from tnkit.eigensolver import lanczos_expm_fixed
    for backend in ("numpy", "torch-cpu"):
        try:
            be = get_backend(backend)
        except Exception:
            continue
        rng = np.random.default_rng(11)
        # (a) the Taylor exponential itself, over a range of norms wide enough to cover
        #     any |tau| ||H_eff|| this pipeline produces. The eigh route is exercised
        #     end-to-end in (b), where its real-dtype tridiagonal is built by the solver.
        for scale in (1.0, 5.0, 30.0):
            d = rng.normal(scale=scale, size=9)
            e = np.abs(rng.normal(scale=scale, size=8))
            T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
            tau = -1j * 0.025
            ref = dexpm(tau * T)
            E2 = np.asarray(be.to_numpy(be.expm_taylor(be.array(tau * T), 8, 18)))
            assert np.linalg.norm(E2 - ref) / np.linalg.norm(ref) < 1e-11, \
                f"{backend} taylor @ scale {scale}"

        # (b) end to end through the Krylov solver: every route must land on the same
        #     answer, since they differ only in HOW the projected exponential is formed
        n = 32
        Hh = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Hh = (Hh + Hh.conj().T) * 1.5
        H = be.array(Hh)
        mv = lambda x: be.einsum("ij,j->i", H, x)
        g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(complex)
        v = be.array(g)
        tau = -1j * 0.025
        base = be.to_numpy(lanczos_expm_fixed(mv, v, tau, be, 14, expm_where="host"))
        for where in ("taylor", "eigh"):
            out = be.to_numpy(lanczos_expm_fixed(mv, v, tau, be, 14, expm_where=where))
            assert np.abs(out - base).max() < 1e-12, \
                f"{backend} {where}: {np.abs(out - base).max():.2e}"


def test_cuda_graph_mode_refuses_an_unfrozen_or_recalibrating_schedule():
    """Graph capture must refuse the two configurations it cannot honestly serve.

    A graph is a recording of ONE kernel sequence. The adaptive solver's stop test is a
    host branch on a device scalar, so it has no fixed sequence to record; and the
    `recal=25` re-measurement sweep deliberately runs the ADAPTIVE solver, so every 25th
    step would be a structurally different (and differently priced) step inside what is
    supposed to be a uniform timing arm. Both raise rather than silently doing something
    defensible-looking, which is the failure mode that once produced a withdrawn number.
    """
    from tnkit.tdvp import tdvp1_evolve
    rng = np.random.default_rng(0)
    N, chi, d = 6, 4, 2
    mps = [rng.standard_normal((1 if i == 0 else chi, d, 1 if i == N - 1 else chi))
           .astype(complex) for i in range(N)]
    mpo = M.tfim_mpo(N, 1.0, 0.0)
    sentinel = object()
    for kw in ({"fast_expm": None}, {"fast_expm": "auto"},
               {"fast_expm": {"m": 6, "recal": 25}}):
        try:
            tdvp1_evolve(mps, mpo, 0.01, 1, backend="numpy", cuda_graph=sentinel, **kw)
        except ValueError:
            continue
        raise AssertionError(f"cuda_graph should have refused {kw}")


def test_fast_expm_is_opt_in_and_refuses_adaptive_dt():
    """`fast_expm=None` must leave the integrator bit-for-bit unchanged, and the one combination
    that would silently mis-step must raise rather than run.

    The adaptive-step controller drives on `est_ref`, the Saad estimate frozen at a fixed Krylov dimension,
    which only the adaptive solver produces. On the fast path it is NaN by construction (reading it
    back is the synchronisation the path exists to delete), and `_KrylovMeter` skips NaN, so the
    controller would see est = 0 and grow the step by `fmax` every time.
    """
    from tnkit.tdvp import tdvp1_evolve
    N, chi = 10, 8
    W = M.tfim_mpo(N, 1.5)
    _, psi, _ = dmrg(W, chi, n_sweeps=4, seed=1)
    psi = [np.asarray(m).astype(complex) for m in psi]
    Wq = M.tfim_mpo(N, 0.8)

    a, _ = tdvp1_evolve(psi, Wq, 0.05, 3)
    b, _ = tdvp1_evolve(psi, Wq, 0.05, 3, fast_expm=None)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y)), "fast_expm=None changed the default"

    c, _ = tdvp1_evolve(psi, Wq, 0.05, 3, fast_expm="auto")
    da = max(np.abs(np.asarray(x) - np.asarray(y)).max() for x, y in zip(a, c))
    assert da < 1e-6, f"fast path diverged from the adaptive one: {da:.2e}"

    try:
        tdvp1_evolve(psi, Wq, 0.05, 2, fast_expm="auto", adaptive_dt={})
    except ValueError:
        pass
    else:
        raise AssertionError("fast_expm + adaptive_dt must raise, not run")


def test_mps_overlap():
    # self-overlap = 1; overlap with a different-g state is real, in [0,1]
    _, p1, _ = dmrg(M.tfim_mpo(12, 1.0), chi=24, n_sweeps=16, tol=1e-12, reorth="full")
    assert abs(abs(mps_overlap(p1, p1)) - 1.0) < 1e-9
    _, p2, _ = dmrg(M.tfim_mpo(12, 1.2), chi=24, n_sweeps=16, tol=1e-12, reorth="full")
    f = abs(mps_overlap(p1, p2))
    assert 0.0 < f < 1.0


def test_two_point_vs_exact_diag():
    N, g = 10, 1.5
    H = M.tfim_dense(N, g); _, V = np.linalg.eigh(H); gs = V[:, 0]
    i0 = 2; js = list(range(i0 + 1, N))
    ex = np.array([(gs.conj() @ (_op_at(SX, i0, N) @ (_op_at(SX, j, N) @ gs))).real for j in js])
    _, mps, _ = dmrg(M.tfim_mpo(N, g), chi=24, n_sweeps=16, tol=1e-12, reorth="full")
    mp = two_point(mps, SX, SX, i0, js)
    assert np.max(np.abs(mp - ex)) < 1e-7, f"two_point max err {np.max(np.abs(mp - ex))}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:                        # noqa
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
