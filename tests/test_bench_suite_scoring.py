"""The benchmark's scoring and ladder rules on synthetic records (no solver runs), and the YASTN
collision converter's charges (tnkit only)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.join(ROOT, "bench", "suite")
sys.path[:0] = [SUITE, os.path.join(SUITE, "runners")]

import analyse  # noqa: E402
import analyse2  # noqa: E402
import schedule  # noqa: E402

E0 = -28.0
SPEC = {"name": "t", "repeats": 3, "workloads": {
    "chain": {"kind": "dmrg", "model": "xxz", "N": 4, "chis": [8, 16, 32],
              "accuracy": {"eps": [1e-6, 1e-8]}, "reference": {"chi": 64, "agree_rel": 1e-10}},
    "tf_collision": {"kind": "collision", "N": 2, "chis": [8], "exclude_first": 1,
                     "accuracy": {"eps": [1e-4]}}}}


def rec(lib, chi, rep, status="ok", d=1e-12, wall=1.0, sym="dense", pod="p", cpu="cpu A",
        workload="chain", **kw):
    r = {"workload": workload, "kind": SPEC["workloads"][workload]["kind"], "lib": lib,
         "backend": "numpy", "sym": sym, "chi": chi, "rep": rep, "status": status, "pod": pod}
    if status == "ok":
        r.update(energy=E0 + d * abs(E0), wall_s=wall, env={"cpu_model": cpu}, sweeps=2)
    return {**r, **kw}


def ref_runs(*libs_d):
    return [rec(lib, 64, 1, d=d, pod="ref") for lib, d in libs_d]


def row_of(rows, lib="tnkit", chi=8, **match):
    return next(r for r in rows if r["lib"] == lib and r["chi"] == chi and r["workload"] == "chain"
                and all(r[k] == v for k, v in match.items()))


def tta_of(spec, rows, lib="tnkit", eps=1e-8):
    return next(t for t in analyse.time_to_accuracy(spec, rows)
                if t["lib"] == lib and t["eps"] == eps and t["workload"] == "chain")


# ----------------------------------------------------------------------------- references

def test_reference_is_the_lowest_energy_of_two_agreeing_libraries():
    refs = analyse.references(SPEC, ref_runs(("tnkit", 0.0), ("yastn", -5e-13), ("tenpy", -3e-5)))
    r = refs["chain"]
    assert r["confirmed"] and r["energy"] == E0 - 5e-13 * abs(E0)      # min of the group, not max
    assert r["agreeing"] == ["tnkit/dense", "yastn/dense"]              # the low outlier is alone


def test_reference_needs_two_libraries_within_1e_12():
    one = analyse.references(SPEC, ref_runs(("tnkit", 0.0), ("tnkit", 1e-14)))["chain"]
    far = analyse.references(SPEC, ref_runs(("tnkit", 0.0), ("yastn", 5e-12)))["chain"]
    assert not one["confirmed"] and one["energy"] is None
    assert not far["confirmed"] and far["energy"] is None


def test_unconfirmed_reference_scores_nothing():
    recs = ref_runs(("tnkit", 0.0)) + [rec("tnkit", 8, k) for k in (1, 2, 3)]
    refs, rows = analyse.score(SPEC, recs)
    assert row_of(rows)["err"] is None
    t = tta_of(SPEC, rows)
    assert t["status"] == "unconfirmed reference" and not t["reached"]
    assert "reference unconfirmed" in analyse.table(SPEC, analyse.time_to_accuracy(SPEC, rows),
                                                    rows, refs)


# ----------------------------------------------------------------------- cells and errors

def good_ref():
    return ref_runs(("tnkit", 0.0), ("yastn", 0.0))


def test_error_is_the_worst_repeat_and_time_the_median():
    recs = good_ref() + [rec("tnkit", 8, 1, d=1e-9, wall=1.0), rec("tnkit", 8, 2, d=1e-7, wall=2.0),
                       rec("tnkit", 8, 3, d=1e-9, wall=10.0)]
    _, rows = analyse.score(SPEC, recs)
    r = row_of(rows)
    assert r["err"] == pytest.approx(1e-7, rel=1e-6) and r["wall_med"] == 2.0
    assert (r["wall_min"], r["wall_max"]) == (1.0, 10.0) and r["complete"]


@pytest.mark.parametrize("d, void", [(-1e-9, True), (-1e-11, False), (1e-9, False)])
def test_void_only_below_the_reference_by_more_than_the_tolerance(d, void):
    recs = good_ref() + [rec("tnkit", 8, k, d=d) for k in (1, 2, 3)]
    assert row_of(analyse.score(SPEC, recs)[1])["void"] is void


def test_cells_split_by_pod_and_cpu_model():
    recs = good_ref() + [rec("tnkit", 8, k, pod="a") for k in (1, 2, 3)] + \
        [rec("tnkit", 8, k, pod="b", wall=9.0) for k in (1, 2, 3)] + \
        [rec("tnkit", 8, k, pod="c", cpu="cpu A" if k < 3 else "cpu B") for k in (1, 2, 3)]
    _, rows = analyse.score(SPEC, recs)
    cells = {(r["pod"], r["device"]): r for r in rows if r["workload"] == "chain" and r["chi"] == 8}
    assert set(cells) == {("a", "cpu A"), ("b", "cpu A"), ("c", "cpu A"), ("c", "cpu B")}
    assert cells[("b", "cpu A")]["wall_med"] == 9.0 and cells[("a", "cpu A")]["wall_med"] == 1.0
    assert not cells[("c", "cpu B")]["complete"]          # one repeat on another CPU model


def test_load_takes_the_pod_from_the_directory(tmp_path):
    (tmp_path / "gpu-dmrg").mkdir()
    p = tmp_path / "gpu-dmrg" / "runs.jsonl"
    p.write_text(json.dumps({"rep": 1}) + "\n" + json.dumps({"rep": 2, "pod": "x"}) + "\n")
    assert [r["pod"] for r in analyse.load([str(p)])] == ["gpu-dmrg", "x"]


# ------------------------------------------------------------------------- time to accuracy

def _row(chi, err, complete=True, statuses=("ok",), void=False, n_ok=3, wall=1.0):
    return {"workload": "chain", "hardware": "cpu", "pod": "p", "device": "cpu A", "lib": "tnkit",
            "backend": "numpy", "sym": "dense", "env": "main", "chi": chi, "err": err,
            "complete": complete, "statuses": list(statuses), "void": void, "n_ok": n_ok,
            "n_runs": 3, "wall_med": wall, "wall_min": wall, "wall_max": wall}


def test_smallest_chi_meeting_eps_inclusive():
    rows = [_row(8, 1e-7, wall=1.0), _row(16, 1e-8, wall=2.0), _row(32, 1e-11, wall=3.0)]
    t = tta_of(SPEC, rows, eps=1e-8)
    assert t["reached"] and t["chi"] == 16 and t["wall_med"] == 2.0      # err == eps counts


def test_void_cell_is_skipped_not_reached():
    rows = [_row(8, 1e-9, void=True, wall=1.0), _row(16, 1e-9, wall=2.0)]
    assert tta_of(SPEC, rows)["chi"] == 16


def test_partial_cell_after_timeout_is_not_reached():
    recs = good_ref() + [rec("tnkit", 8, 1, d=1e-9, wall=5.0), rec("tnkit", 8, 2, status="timeout"),
                       rec("tnkit", 8, 3, status="skipped_after_timeout")] + \
        [rec("tnkit", c, k, status="skipped_after_timeout") for c in (16, 32) for k in (1, 2, 3)]
    refs, rows = analyse.score(SPEC, recs)
    r = row_of(rows)
    assert (r["n_ok"], r["n_runs"], r["complete"]) == (1, 3, False)
    tta = analyse.time_to_accuracy(SPEC, rows)
    t = tta_of(SPEC, rows)
    assert (t["status"], t["chi"], t["reached"], t["wall_med"]) == ("timeout", 8, False, None)
    assert "timeout at χ 8 (1 of 3 ok)" in analyse.table(SPEC, tta, rows, refs)


def test_failed_or_missing_repeat_is_incomplete():
    error = good_ref() + [rec("tnkit", 8, 1, d=1e-9), rec("tnkit", 8, 2, status="error"),
                        rec("tnkit", 8, 3, d=1e-9)]
    t = tta_of(SPEC, analyse.score(SPEC, error)[1])
    assert t["status"] == "incomplete" and not t["reached"]
    missing = good_ref() + [rec("tnkit", 16, k, d=1e-9) for k in (1, 2, 3)] + \
        [rec("tnkit", 8, k, d=1e-9) for k in (1, 3)]
    t = tta_of(SPEC, analyse.score(SPEC, missing)[1])
    assert (t["status"], t["chi"]) == ("incomplete", 8)


def test_spread_reruns_count_and_budget_skips_do_not_void():
    recs = good_ref() + [rec("tnkit", 8, k, d=1e-9, wall=w) for k, w in
                       ((1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0))]
    assert row_of(analyse.score(SPEC, recs)[1])["wall_med"] == 3.0       # median of 5
    recs = recs[:-2] + [rec("tnkit", 8, k, status="skipped_budget") for k in (4, 5)]
    r = row_of(analyse.score(SPEC, recs)[1])
    assert r["complete"] and r["wall_med"] == 2.0
    recs = recs[:-2] + [rec("tnkit", 8, 4, status="timeout")]
    assert not row_of(analyse.score(SPEC, recs)[1])["complete"]


def test_analyse2_applies_the_same_rules():
    spec = {**SPEC, "workloads": {"xxz_dmrg": {**SPEC["workloads"]["chain"]}}}
    recs = [dict(r, workload="xxz_dmrg", threads=4) for r in              # its reference: chi 512
            [dict(r, chi=512) for r in good_ref()] +
            [rec("tnkit", 8, 1, d=1e-9, wall=5.0), rec("tnkit", 8, 2, status="timeout"),
             rec("tnkit", 8, 3, status="skipped_after_timeout")]]
    for r in recs:
        if r["status"] == "ok":
            r["env"]["cgroup_cpus"] = 4.0
    refs = analyse2.references(spec, recs)
    assert refs["xxz_dmrg"]["confirmed"]
    _, rows = analyse2.score(spec, recs, {}, refs)
    cell = next(r for r in rows if r["chi"] == 8)
    assert cell["hardware"] == "cpu4" and cell["n_runs"] == 3 and not cell["complete"]
    t = next(t for t in analyse.time_to_accuracy(spec, rows, analyse2.KEYS, ladder_only=False)
             if t["eps"] == 1e-8 and t["chi"] == 8)
    assert t["status"] == "timeout" and not t["reached"]
    lone = analyse2.chain_reference(spec, [r for r in recs if r["lib"] == "tnkit"])
    assert not lone["confirmed"] and lone["energy"] is None


# ------------------------------------------------------------------------- collision tests

def _coll(lib, sym, step, complete=True, void=False):
    return {"workload": "tf_collision", "lib": lib, "sym": sym, "step_med": step,
            "complete": complete, "void": void}


def test_t1_uses_the_faster_tnkit_arm_and_t2_boundary_is_inclusive():
    t = analyse.collision_tests([_coll("tnkit", "dense", 60.0), _coll("tnkit", "u1", 40.0),
                                 _coll("yastn", "u1", 150.0)])
    assert t["T1"]["r_lib"] == pytest.approx(40 / 150) and t["T1"]["tnkit_best"] == "u1"
    assert t["T1"]["verdict"] == "TNKIT-AHEAD"
    assert t["T2"]["r_sym"] == 1.5 and t["T2"]["verdict"] == "BLOCKS-PAY"


def test_incomplete_or_void_collision_arm_voids_the_test():
    t = analyse.collision_tests([_coll("tnkit", "dense", 60.0), _coll("tnkit", "u1", 40.0),
                                 _coll("yastn", "u1", 150.0, complete=False)])
    assert t["T1"]["verdict"] == "VOID" and t["T2"]["verdict"] == "BLOCKS-PAY"
    t = analyse.collision_tests([_coll("tnkit", "dense", 60.0, void=True),
                                 _coll("tnkit", "u1", 40.0), _coll("yastn", "u1", 150.0)])
    assert t["T2"]["verdict"] == "VOID"


# ------------------------------------------------------------------------- the scheduler

ARMS = [{"lib": lib, "backend": "numpy", "sym": "dense", "env": "main"}
        for lib in ("tnkit", "yastn", "tenpy")]


def _run_pod(tmp_path, monkeypatch, behave, spread=None, repeats=2):
    spec = {"name": "t", "repeats": repeats, "envs": {"main": {"site": ""}},
            "workloads": {"w": {"kind": "dmrg", "chis": [4, 8, 16], "timeout_s": {"cpu": 10}}},
            "pods": {"p": {"hardware": "cpu", "threads": 1, "budget_h": 1, "workloads": ["w"],
                           "arms": ARMS}}}
    if spread:
        spec["spread_rerun"] = spread
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    sp = schedule.load_spec(str(path))

    def fake(spec_, pod, rep, wl, chi, arm, out, timeout, *a, **k):
        status, wall = behave(rep, chi, arm["lib"])
        schedule.append_record(out, {**schedule.stub(sp, pod, rep, wl, chi, arm, None, status),
                                     "wall_s": wall, "env": {"env_name": arm["env"]}})
        return status

    monkeypatch.setattr(schedule, "run_one", fake)
    schedule.main(["--spec", str(path), "--pod", "p", "--out", str(tmp_path / "out")])
    with open(tmp_path / "out" / "runs.jsonl") as f:
        runs = [json.loads(line) for line in f]
    return {(r["rep"], r["chi"], r["lib"]): r["status"] for r in runs}, runs


def test_timeout_stops_the_arm_from_that_chi_up_and_unsupported_drops_it(tmp_path, monkeypatch):
    def behave(rep, chi, lib):
        if lib == "tnkit" and (rep, chi) == (1, 8):
            return "timeout", 10.0
        if lib == "yastn":
            return "unsupported", 0.0
        return "ok", 1.0

    st, runs = _run_pod(tmp_path, monkeypatch, behave)
    assert len(runs) == 2 * 3 * 3 and all(r["pod"] == "p" and r["threads"] == 1 for r in runs)
    assert st[(1, 16, "tnkit")] == st[(2, 8, "tnkit")] == "skipped_after_timeout"
    assert st[(2, 4, "tnkit")] == "ok"
    assert {st[(r, c, "yastn")] for r in (1, 2) for c in (4, 8, 16)} - {"unsupported"} == \
        {"skipped_unsupported"} and st[(1, 4, "yastn")] == "unsupported"
    assert {st[(r, c, "tenpy")] for r in (1, 2) for c in (4, 8, 16)} == {"ok"}


def test_spread_reruns_every_arm_of_the_cell_interleaved(tmp_path, monkeypatch):
    def behave(rep, chi, lib):
        return "ok", 2.0 if (lib, chi, rep) == ("tnkit", 4, 2) else 1.0

    st, runs = _run_pod(tmp_path, monkeypatch, behave, spread={"ratio": 1.5, "extra": 2},
                        repeats=3)
    extra = [(r["rep"], r["chi"], r["lib"]) for r in runs if r["rep"] > 3]
    assert extra == [(4, 4, "tnkit"), (4, 4, "yastn"), (4, 4, "tenpy"),
                     (5, 4, "yastn"), (5, 4, "tenpy"), (5, 4, "tnkit")]


def test_spread_trigger_uses_the_step_time_for_collisions():
    w = {"kind": "collision", "exclude_first": 1}
    assert schedule.ranked_time({"wall_s": 99.0, "step_s": [50.0, 2.0, 4.0]}, w) == 3.0
    assert schedule.ranked_time({"wall_s": 99.0}, {"kind": "dmrg"}) == 99.0


# ------------------------------------------------------------- YASTN collision converter

def test_yastn_u1_collision_inputs_conserve_charge_block_by_block():
    import yastn_runner
    from tnkit import dmrg
    from tnkit.models_lgt2f import schwinger2f_mpo
    W = [np.asarray(t) for t in schwinger2f_mpo(2, 1.0, 0.5)]          # 2 sites, 4 qubits
    _, psi, _ = dmrg(W, chi=8, n_sweeps=6, backend="numpy")
    state, mpo, setup = yastn_runner.u1_inputs([np.asarray(t) for t in psi], W, 2, 8)
    assert setup["fidelity"] > 1 - 1e-10 and len(state) == len(mpo) == 4
    for arr, charges, signature in state + mpo:
        assert arr.ndim == len(charges) == len(signature)
        for idx in np.argwhere(np.abs(arr) > 1e-14):           # `_tensor` would drop none of it
            assert sum(s * q[i] for s, q, i in zip(signature, charges, idx)) == 0
    n = sum(np.linalg.norm(a) ** 2 for a, *_ in mpo)
    assert n == pytest.approx(sum(np.linalg.norm(t) ** 2 for t in W))


def test_unsupported_and_gated_arms_are_named_not_incomplete():
    rows = [_row(8, None, complete=False, statuses=("unsupported", "skipped_unsupported"), n_ok=0)]
    assert tta_of(SPEC, rows)["status"] == "unsupported"
    rows = [_row(8, None, complete=False, statuses=("skipped_gate",), n_ok=0)]
    assert tta_of(SPEC, rows)["status"] == "not run: gate"


# ------------------------------------------------------------ repeats, reruns, labels

def _two_arms(t5, y5):
    """tnkit and YASTN at chi 8, repeats 1-3 ok; repeat 4 ok; repeat 5 with the given status."""
    recs = good_ref()
    for lib, s5 in (("tnkit", t5), ("yastn", y5)):
        recs += [rec(lib, 8, k, d=1e-9, wall=float(k)) for k in (1, 2, 3, 4)]
        recs.append(rec(lib, 8, 5, status=s5, d=1e-9, wall=5.0))
    return recs


def test_reruns_count_only_if_every_arm_finished_them():
    rows = analyse.score(SPEC, _two_arms("ok", "ok"))[1]
    assert [row_of(rows, lib)["n_ok"] for lib in ("tnkit", "yastn")] == [5, 5]
    rows = analyse.score(SPEC, _two_arms("ok", "skipped_budget"))[1]
    for lib in ("tnkit", "yastn"):
        r = row_of(rows, lib)
        assert (r["n_ok"], r["wall_med"], r["complete"]) == (3, 2.0, True), lib


def test_planned_repeats_come_from_the_plan(tmp_path):
    d = tmp_path / "cpu-a"
    d.mkdir()
    lines = [rec(lib, 8, k, d=1e-9) for lib in ("tnkit", "yastn") for k in (1, 2)]
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in good_ref() + lines))
    recs = analyse.load([str(d / "runs.jsonl")])
    assert row_of(analyse.score(SPEC, recs)[1])["complete"]         # no plan: 2 of 2 seen
    order = [[k, "chain", 8, f"{lib}/numpy/dense/main"] for k in (1, 2, 3) for lib in ("tnkit",
                                                                                        "yastn")]
    (d / "plan.json").write_text(json.dumps({"reps": 3, "order": order + [[101, "chain", 8, "x"]]}))
    recs = analyse.load([str(d / "runs.jsonl")])
    assert {r.get("planned_reps") for r in recs if r["chi"] == 8} == {3}
    assert not row_of(analyse.score(SPEC, recs)[1])["complete"]     # repeat 3 never ran


def test_an_arm_whose_runs_all_failed_reads_error():
    rows = [_row(8, None, complete=False, statuses=("error",), n_ok=0)]
    t = tta_of(SPEC, rows)
    assert t["status"] == "error" and analyse.tta_cell(t, str) == "error at χ 8 (all 3 runs)"


def test_spread_trigger_uses_seconds_per_sweep_for_throughput():
    w = {"kind": "dmrg", "fixed_sweeps": 3}
    assert schedule.ranked_time({"wall_s": 99.0, "sweep_s": [50.0, 2.0, 4.0]}, w) == 3.0


def _dmrg_rec(wall, rtol, cpus, chi=32):
    return {"workload": "xxz_dmrg", "kind": "dmrg", "lib": "tnkit", "backend": "numpy",
            "sym": "dense", "chi": chi, "rep": 1, "status": "ok", "wall_s": wall,
            "settings": {"rtol": rtol}, "env": {"cgroup_cpus": cpus}}


def test_control_offset_uses_runs_with_the_same_stop():
    here = [_dmrg_rec(2.0, 1e-10, 4.0), _dmrg_rec(2.0, 1e-10, 4.0, chi=64)]
    there = [_dmrg_rec(1.0, 1e-10, None), _dmrg_rec(4.0, 0.0, None)]
    (o,) = analyse2.control_offset(here, there)
    assert (o["rtol"], o["chis"], o["offset"], o["within_0.8_1.25"]) == (1e-10, [32], 2.0, False)


def test_analyse2_labels_multiplets_and_tenpy_on_excited_states():
    assert analyse2._name("block2", "su2", "dmrg") == ("block2", "su2 (χ: multiplets)")
    assert analyse2._name("tenpy", "dense", "excited") == ("TeNPy†", "dense")
    assert analyse2._name("tenpy", "u1", "dmrg") == ("TeNPy", "u1")
    assert analyse2._name("qtea", "dense", "dmrg")[0] == "Quantum TEA*"
