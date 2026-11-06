"""The benchmark harness: exact references, the tnkit runner on tiny problems, the run order."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.join(ROOT, "bench", "suite")
sys.path[:0] = [SUITE, os.path.join(SUITE, "runners")]

import reference  # noqa: E402
import schedule  # noqa: E402
from common import load_spec  # noqa: E402


def test_tfim_reference_matches_tnkit():
    from tnkit.free_fermion import tfim_free_fermion_energy
    for N, g in ((8, 1.0), (21, 0.7)):
        assert reference.tfim_energy(N, g) == pytest.approx(tfim_free_fermion_energy(N, g),
                                                            abs=1e-12)


def test_xx_reference_limits():
    sz0 = reference.xx_neel_sz(10, 1.0, 0.0)
    np.testing.assert_allclose(sz0, [0.5, -0.5] * 5, atol=1e-14)
    assert abs(reference.xx_neel_sz(10, 1.0, 3.0).sum()) < 1e-12      # S^z_total conserved
    assert reference.xx_neel_entropy(10, 1.0, 0.0, 5) < 1e-10


def _xx(N=8, T=0.5):
    return {"kind": "tdvp", "model": "xx", "N": N, "J": 1.0, "Jz": 0.0, "dt": 0.05, "T": T,
            "svd_tol": 1e-12}


def test_tnkit_tdvp_matches_exact():
    import tnkit_runner as tk
    w = _xx()
    out = tk.tdvp(w, 16, "numpy", "dense")
    ex = reference.xx_neel_sz(w["N"], w["J"], w["T"])
    assert np.max(np.abs(np.array(out["sz"]) - ex)) < 1e-6
    assert len(out["step_s"]) == 10


def test_tnkit_dmrg_matches_exact_ising():
    import tnkit_runner as tk
    w = {"kind": "dmrg", "model": "tfim", "N": 10, "g": 1.0, "max_sweeps": 10, "e_tol": 1e-12}
    out = tk.dmrg(w, 32, backend="numpy", sym="dense")
    assert out["energy"] == pytest.approx(reference.tfim_energy(10, 1.0), abs=1e-9)
    with pytest.raises(tk.Unsupported):
        tk.check(w, "u1")


def test_tnkit_u1_dmrg_matches_dense():
    import tnkit_runner as tk
    w = {"kind": "dmrg", "model": "xxz", "N": 12, "J": 1.0, "Jz": 1.0, "max_sweeps": 10,
         "e_tol": 1e-12}
    try:
        tk.check(w, "u1")
    except tk.Unsupported:
        pytest.skip("this tnkit has no block-tensor DMRG")
    u1, dense = tk.dmrg(w, 32, backend="numpy", sym="u1"), tk.dmrg(w, 32, backend="numpy", sym="dense")
    assert u1["energy"] == pytest.approx(dense["energy"], abs=1e-10)
    assert u1["settings"]["blocks"] and not dense["settings"]["blocks"]


def test_xxz_mpo_prune_keeps_the_hamiltonian():
    import tnkit_runner as tk
    from tnkit.models import mpo_to_dense
    for Jz in (0.0, 1.0):
        full = tk.xxz_mpo(4, 1.0, Jz)
        assert all(w.shape[1] == n.shape[0] for w, n in zip(full, full[1:]))
        H = np.real(mpo_to_dense(full))
        assert np.allclose(H, H.T) and H.shape == (16, 16)
    assert tk.xxz_mpo(4, 1.0, 0.0)[1].shape[0] < tk.xxz_mpo(4, 1.0, 1.0)[1].shape[0]


def test_plan_interleaves_and_rotates():
    spec = load_spec(os.path.join(SUITE, "suite.json"))
    order = schedule.plan(spec, "cpu-tdvp", 2)
    arms = [a for a in spec["pods"]["cpu-tdvp"]["arms"]
            if "xx_quench" in a.get("workloads", ["xx_quench"])]
    first = [schedule.arm_key(a) for r, w, c, a in order if r == 1 and c == 32]
    second = [schedule.arm_key(a) for r, w, c, a in order if r == 2 and c == 32]
    assert first == [schedule.arm_key(a) for a in arms]
    assert second == first[1:] + first[:1]
    assert len(order) == 2 * len(arms) * len(spec["workloads"]["xx_quench"]["chis"])


def test_specs_agree_in_shape():
    full = load_spec(os.path.join(SUITE, "suite.json"))
    smoke = load_spec(os.path.join(SUITE, "suite_smoke.json"))
    assert set(full["pods"]) == set(smoke["pods"])
    assert set(full["workloads"]) == set(smoke["workloads"])


def test_suite_2u_reuses_suite_1_workloads():
    full = load_spec(os.path.join(SUITE, "suite.json"))
    s2u = load_spec(os.path.join(SUITE, "suite_2u.json"))
    smoke = load_spec(os.path.join(SUITE, "suite_2u_smoke.json"))
    assert set(s2u["pods"]) == set(smoke["pods"]) and set(s2u["workloads"]) == set(smoke["workloads"])
    for name, w in s2u["workloads"].items():
        if not w.get("perf"):
            assert w == full["workloads"][name], name
    assert s2u["repeats"] == full["repeats"]
    for p in s2u["pods"].values():
        assert all(a["sym"] == "u1" or a["lib"] == "tnkit" for a in p["arms"])


def test_block_gate_passes_on_numpy(tmp_path):
    import block_smoke
    import tnkit_runner as tk
    try:
        tk.check({"kind": "dmrg", "model": "xxz"}, "u1")
    except tk.Unsupported:
        pytest.skip("this tnkit has no block-tensor DMRG")
    assert block_smoke.main(["--backend", "numpy", "--out", str(tmp_path / "g.json")]) == 0


def test_xxz_reference_matches_dense_dmrg():
    import tnkit_runner as tk
    w = {"kind": "dmrg", "model": "xxz", "N": 10, "J": 1.0, "Jz": 0.5, "max_sweeps": 10,
         "e_tol": 1e-12}
    out = tk.dmrg(w, 32, backend="numpy", sym="dense")
    assert out["energy"] == pytest.approx(reference.xxz_energy(10, 1.0, 0.5), abs=1e-10)
