"""S2xS5 defaults: production statics (dmrg, dmrg_excited/dmrg_tower) now default to
thick-restart Lanczos (S2, `eig="lanczos-restart"`) and an auto-selected real dtype
backend (S5, `backend="auto"`) instead of the historical `eig="lanczos"` /
`backend="numpy"` (always complex). Both are ~3x combined on statics and the
difference between a chi that fits on one GPU and one that OOMs.

This file pins:
  * the new defaults themselves (DMRGOptions and DMRGExcitedOptions fields),
  * the auto-dtype rule -- real when the MPO (and any init_mps) are real to the same
    tolerance the real backend's own runtime guard uses, complex otherwise, so a
    genuinely complex MPO (built with SY here) still gets a working complex backend
    by default rather than a spurious TypeError,
  * that the explicit opt-out (`eig="lanczos"`, `backend="numpy"`) still works,
  * accuracy parity between the new and old defaults at solver-tolerance level,
  * that dynamics (tdvp/tebd) and the non-Hermitian paths (vumps/idmrg) are untouched.

Runs on CPU, small N/chi, no GPU needed.
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

import tnkit.models as M
from tnkit._dmrg import DMRGOptions
from tnkit.dmrg import dmrg
from tnkit.dmrg_excited import dmrg_excited, dmrg_tower
from tnkit.backend import get_backend, is_real_problem
from tnkit.free_fermion import tfim_free_fermion_energy


# --------------------------------------------------------------------------- #
# The defaults themselves
# --------------------------------------------------------------------------- #
def test_dmrg_options_default_eig_is_thick_restart():
    assert DMRGOptions().eig == "lanczos-restart"


def test_dmrg_options_default_backend_is_auto():
    assert DMRGOptions().backend == "auto"


def test_dmrg_excited_default_eig_is_thick_restart_and_backend_auto():
    from tnkit.dmrg_excited import DMRGExcitedOptions
    assert DMRGExcitedOptions().eig == "lanczos-restart"
    assert DMRGExcitedOptions().backend == "auto"


# --------------------------------------------------------------------------- #
# The auto-dtype rule
# --------------------------------------------------------------------------- #
def test_is_real_problem_true_for_tfim_mpo():
    mpo = M.tfim_mpo(10, 1.0)
    assert is_real_problem(mpo)


def test_is_real_problem_false_for_sy_mpo():
    mpo = M.magnetization_mpo(6, op=M.SY)
    assert not is_real_problem(mpo)


def test_get_backend_auto_needs_mpo():
    with pytest.raises(ValueError, match="mpo"):
        get_backend("auto")


def test_dmrg_default_engages_real_backend_on_real_mpo():
    N, g = 10, 1.0
    E, mps, info = dmrg(M.tfim_mpo(N, g), chi=16, n_sweeps=10, tol=1e-12)
    assert info["eig"] == "lanczos-restart"
    assert info["backend"] == "numpy-real"
    assert not np.iscomplexobj(np.asarray(mps[0]))
    assert abs(E - tfim_free_fermion_energy(N, g)) < 1e-8


def test_dmrg_default_falls_back_to_complex_on_complex_mpo():
    """H = sum_i sigma^y_i is genuinely complex (SY has non-negligible imaginary
    entries): the auto default must pick the complex backend, not truncate or raise.
    Ground energy is exactly -N (each site independently minimises <sigma_y> to -1)."""
    N = 6
    mpo = M.magnetization_mpo(N, op=M.SY)
    E, mps, info = dmrg(mpo, chi=2, n_sweeps=6, tol=1e-12)
    assert info["backend"] == "numpy"           # complex, NOT numpy-real
    assert np.iscomplexobj(np.asarray(mps[0]))
    assert abs(E - (-N)) < 1e-8


def test_dmrg_excited_auto_backend_matches_dmrg():
    N, g = 8, 1.0
    _, _, info = dmrg_excited(M.tfim_mpo(N, g), chi=8, n_sweeps=6, tol=1e-11)
    assert info["eig"] == "lanczos-restart"
    assert info["backend"] == "numpy-real"


# --------------------------------------------------------------------------- #
# Explicit opt-out keeps working
# --------------------------------------------------------------------------- #
def test_dmrg_explicit_opt_out_overrides_auto():
    N, g = 10, 1.0
    E, mps, info = dmrg(M.tfim_mpo(N, g), chi=16, n_sweeps=10, tol=1e-12,
                        eig="lanczos", backend="numpy")
    assert info["eig"] == "lanczos"
    assert info["backend"] == "numpy"            # complex, even though the MPO is real
    assert np.iscomplexobj(np.asarray(mps[0]))
    assert abs(E - tfim_free_fermion_energy(N, g)) < 1e-8


def test_dmrg_excited_explicit_opt_out_overrides_auto():
    N, g = 8, 1.0
    _, _, info = dmrg_excited(M.tfim_mpo(N, g), chi=8, n_sweeps=6, tol=1e-11,
                              eig="lanczos", backend="numpy")
    assert info["eig"] == "lanczos"
    assert info["backend"] == "numpy"


def test_dmrg_excited_rejects_unknown_eig():
    with pytest.raises(ValueError, match="eig must be"):
        dmrg_excited(M.tfim_mpo(6, 1.0), chi=4, eig="bogus")


# --------------------------------------------------------------------------- #
# Accuracy parity: new default vs old default, at solver-tolerance level
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N,g,chi", [(12, 1.0, 16), (12, 0.5, 16), (14, 1.5, 20)])
def test_dmrg_new_default_matches_old_default_ground_energy(N, g, chi):
    kw = dict(n_sweeps=20, tol=1e-13, eig_tol=1e-11, reorth="full", seed=0)
    E_new, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, **kw)                       # new default
    E_old, _, _ = dmrg(M.tfim_mpo(N, g), chi=chi, eig="lanczos", backend="numpy", **kw)
    assert abs(E_new - E_old) < 1e-9, f"N={N} g={g}: dE={E_new - E_old:.3e}"


def test_dmrg_excited_new_default_matches_old_hardcoded_solver():
    """dmrg_excited had no `eig=` before this flip (always `lanczos_ground`, complex
    numpy). `eig='lanczos', backend='numpy'` reproduces that historical path exactly;
    the new default (`lanczos-restart` + auto real) must agree at solver tolerance."""
    N, g, chi = 10, 1.0, 12
    kw = dict(n_sweeps=14, tol=1e-11, eig_tol=1e-10, seed=0)
    E_new, _, _ = dmrg_excited(M.tfim_mpo(N, g), chi, **kw)                    # new default
    E_old, _, _ = dmrg_excited(M.tfim_mpo(N, g), chi, eig="lanczos", backend="numpy", **kw)
    assert abs(E_new - E_old) < 1e-8, f"dE={E_new - E_old:.3e}"


def test_dmrg_tower_uses_thick_restart_and_auto_backend_by_default():
    N, g = 8, 1.0
    energies, states, infos = dmrg_tower(M.tfim_mpo(N, g), chi=8, nlevels=2,
                                         n_sweeps=6, tol=1e-10)
    for info in infos:
        assert info["eig"] == "lanczos-restart"
        assert info["backend"] == "numpy-real"
    assert energies[1] >= energies[0] - 1e-9


# --------------------------------------------------------------------------- #
# Dynamics and the non-Hermitian paths are UNTOUCHED
# --------------------------------------------------------------------------- #
def test_tdvp_options_backend_default_unchanged():
    from tnkit.tdvp import TDVPOptions
    assert TDVPOptions().backend == "numpy"


def test_tebd_evolve_backend_default_unchanged():
    from tnkit.tebd import tebd_evolve
    sig = inspect.signature(tebd_evolve)
    assert sig.parameters["backend"].default == "numpy"


def test_vumps_options_backend_default_unchanged():
    from tnkit.vumps import VUMPSOptions
    assert VUMPSOptions().backend == "numpy"


def test_tower_real_path_raises_no_complex_warning():
    """The penalised matvec must stay in the real dtype on the real backend. Before the
    fix, `to_scalar` (which returns a Python complex) promoted the penalty accumulate to
    complex128, and the eigensolver cast it back to real at every Lanczos step -- a silent
    truncation (numpy ComplexWarning) that both defeated S5 on the excited path and would
    mask a genuinely complex vector reaching the real solver."""
    import warnings
    complex_warning = getattr(np, "exceptions", np).ComplexWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=complex_warning)
        dmrg_tower(M.tfim_mpo(8, 1.0), chi=8, nlevels=2, n_sweeps=6, tol=1e-10)


def test_idmrg_backend_default_unchanged():
    """idmrg's `be` default is not real-dtype-safe (its random seed adds an
    unconditional `1j*` imaginary part regardless of backend), so it is deliberately
    excluded from the S2xS5 flip; its default stays the historical complex NumpyBackend."""
    from tnkit._idmrg import IDMRGOptions
    default_be = get_backend(IDMRGOptions().backend)
    assert default_be.is_complex
    assert not default_be.name.endswith("-real")
