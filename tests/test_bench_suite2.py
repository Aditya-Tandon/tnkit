"""Suite 2 of the library benchmark: the cylinder, references, run plans and the scorer's rules."""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.join(ROOT, "bench", "suite")
sys.path[:0] = [SUITE, os.path.join(SUITE, "runners")]

import reference  # noqa: E402
import schedule  # noqa: E402
from common import bonds, load_spec, schedule_of, settings_for  # noqa: E402

CYL = {"kind": "dmrg", "model": "heis_cyl", "Lx": 4, "Ly": 3, "N": 12, "J": 1.0,
       "ramp": [[0.125, 2], [0.25, 2], [0.5, 2]], "max_sweeps": 10, "e_tol_rel": 1e-10,
       "energy_scale": 7.0}


def test_cylinder_bonds():
    b = bonds({"model": "heis_cyl", "Lx": 12, "Ly": 6, "N": 72})
    assert len(b) == 12 * 6 + 11 * 6 and len(set(b)) == len(b)
    assert max(j - i for i, j in b) == 6 and all(i < j for i, j in b)
    with pytest.raises(ValueError):
        bonds({"model": "heis_cyl", "Lx": 4, "Ly": 2, "N": 8})


def test_cylinder_mpo_is_the_hamiltonian():
    from scipy.sparse.linalg import eigsh

    import tnkit_runner as tk
    from tnkit.opsum import to_dense
    H = to_dense(tk.mpo_for(CYL))
    e = eigsh(H, k=1, which="SA", tol=1e-14)[0][0]           # the reference's own solver
    assert e == pytest.approx(reference.xxz_energy(12, 1.0, 1.0, pairs=bonds(CYL)), abs=1e-10)
    assert max(t.shape[1] for t in tk.mpo_for({**CYL, "Lx": 12, "Ly": 6, "N": 72})) <= 20


def test_schedule_of():
    assert schedule_of(CYL, 64) == [(8, 2), (16, 2), (32, 2), (64, 10)]
    assert schedule_of({"fixed_sweeps": 3}, 256) == [(256, 3)]
    assert schedule_of({"max_sweeps": 20}, 32) == [(32, 20)]


def test_tfim_levels_match_exact_diagonalisation():
    N, g = 8, 1.0
    X, Z = np.array([[0, 1], [1, 0.]]), np.diag([1., -1])
    op = lambda o, i: np.kron(np.kron(np.eye(2 ** i), o), np.eye(2 ** (N - i - 1)))  # noqa: E731
    H = sum(-op(X, i) @ op(X, i + 1) for i in range(N - 1)) + sum(-g * op(Z, i) for i in range(N))
    np.testing.assert_allclose(reference.tfim_levels(N, g, 6), np.linalg.eigvalsh(H)[:6],
                               atol=1e-12)


def test_extrapolate_recovers_a_line():
    e, s = reference.extrapolate([3e-5, 1e-5, 2e-6], [-1 + 2 * x for x in (3e-5, 1e-5, 2e-6)])
    assert e == pytest.approx(-1.0, abs=1e-14) and s < 1e-14


@pytest.mark.parametrize("sym", ["dense", "u1"])
def test_tnkit_cylinder_with_ramp_is_exact(sym):
    import tnkit_runner as tk
    try:
        tk.check(CYL, sym)
    except tk.Unsupported:
        pytest.skip("this tnkit has no block DMRG")
    out = tk.dmrg(CYL, 64, backend="numpy", sym=sym)
    assert out["energy"] == pytest.approx(reference.xxz_energy(12, 1.0, 1.0, pairs=bonds(CYL)),
                                          abs=1e-9)
    assert out["settings"]["stages"] == [(8, 2), (16, 2), (32, 2), (64, 10)]
    assert out["sweep_s"] is not None and len(out["sweep_s"]) == out["sweeps"]


def test_tnkit_fixed_sweeps_run_exactly_that_many():
    import tnkit_runner as tk
    w = {"kind": "dmrg", "model": "xxz", "N": 10, "J": 1.0, "Jz": 1.0, "fixed_sweeps": 3,
         "max_sweeps": 3, "e_tol": 1e-10}
    assert tk.dmrg(w, 32, backend="numpy", sym="dense")["sweeps"] == 3


def test_tnkit_excited_levels():
    import tnkit_runner as tk
    w = {"kind": "excited", "model": "tfim", "N": 10, "g": 1.0, "nstates": 3, "max_sweeps": 20,
         "e_tol": 1e-12}                         # E_2 - E_0 = 0.89 < the default penalty 1
    out = tk.excited(w, 32, "numpy", "dense")
    np.testing.assert_allclose(out["energies_k"], reference.tfim_levels(10, 1.0, 3), atol=1e-8)


def test_spec_plans():
    s = load_spec(os.path.join(SUITE, "suite_2.json"))
    p = s["pods"]["g3-cyl"]
    order = schedule.plan(s, "g3-cyl", 3)
    dense = {c for r, w, c, a in order if a["lib"] == "tnkit" and a["sym"] == "dense" and r < 100}
    assert dense == {256, 512}
    assert {c for r, w, c, a in order if a["sym"] == "u1"} == {256, 512, 1024}
    assert sum(r >= 100 for r, *_ in order) == len(p["arms"])              # sampler-off checks
    o = schedule.plan(s, "cpu-s2b", 3)
    assert sum(w == "heis_cyl6_ref" for _, w, _, _ in o) == 3               # one repeat
    assert o[0][1] == "heis_cyl6_ref"
    assert {schedule.arm_key(a) for *_, a in o if a.get("threads") == 4}
    for name, pod in s["pods"].items():                                    # every arm's ladder
        for a in pod["arms"]:                                              # is in the workload's
            for wl in a.get("workloads", pod["workloads"]):
                assert set(schedule.ladder(s, pod, wl, a)) <= set(s["workloads"][wl]["chis"]) \
                    or a["lib"] == "block2", (name, wl, a)


def test_block2_runs_get_its_library_path(monkeypatch):
    s = load_spec(os.path.join(SUITE, "suite_2.json"))
    arms = {a["lib"]: a for a in s["pods"]["cpu-s2a"]["arms"]}
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/x")
    monkeypatch.setenv("TNBENCH_BLOCK2_LIB", "/tmp/env_mkl/lib")
    env = schedule.child_env(s, "cpu-s2a", arms["block2"])
    assert env["LD_LIBRARY_PATH"] == "/tmp/env_mkl/lib:/usr/x" and env["MKL_NUM_THREADS"] == "4"
    assert schedule.child_env(s, "cpu-s2a", arms["tnkit"])["LD_LIBRARY_PATH"] == "/usr/x"


def test_suite1_plans_unchanged():
    s = load_spec(os.path.join(SUITE, "suite.json"))
    n = {p: len(schedule.plan(s, p, s["pods"][p].get("reps", s["repeats"]))) for p in s["pods"]}
    assert n == {"cpu-dmrg": 144, "cpu-tdvp": 84, "cpu-ref": 3, "gpu-dmrg": 96, "gpu-tdvp": 60,
                 "gpu-collision": 12}


def test_settings_merge_and_reject():
    st = json.load(open(os.path.join(SUITE, "settings_suite2.json")))
    assert settings_for(st, "block2", "dmrg", "heis_cyl6")["stack_mem_gb"] == 40
    assert settings_for(st, "block2", "dmrg", "xxz_dmrg")["stack_mem_gb"] == 4
    import yastn_runner
    with pytest.raises(ValueError):
        yastn_runner.options({"opts": {"opts_eig": {}}}, yastn_runner.OPTS)


def test_itensor_job_and_toml():
    import itensor_runner as it
    job = it._job("dmrg", {**CYL, "opts": {}}, 64, "julia-cuda", "u1")
    assert job["maxdims"] == [8, 8, 16, 16, 32, 32] + [64] * 10 and job["minsweeps"] == 8
    assert job["bonds"][0] == [0, 1] and job["qn"] and job["device"] == "cuda"
    assert job["tol"] == pytest.approx(7e-10)
    tomllib = pytest.importorskip("tomllib")
    back = tomllib.loads(it.toml(job))
    assert back["maxdims"] == job["maxdims"] and back["opts"]["noise"] == it.DEFAULTS["noise"]



RUN_JL_OUT = os.path.join(ROOT, "tests", "data", "itensor_run_jl")   # records as run.jl writes them


def _run_jl_record(kind):
    with open(os.path.join(RUN_JL_OUT, f"{kind}.json")) as f:
        return json.load(f)


def test_run_jl_writes_the_keys_the_runner_reads():
    import itensor_runner as it
    src = open(it.RUN_JL).read()
    for kind, keys in it.OUTPUTS.items():
        body = src.split(f"function run_{kind}(p)")[1].split("\nend\n")[0]
        assert set(keys) <= set(re.findall(r'"(\w+)" =>', body)), kind
    for name in ("run.jl", "calib.jl"):          # a top-level try assigns to new locals
        text = open(os.path.join(SUITE, "jl", "bench", name)).read()
        assert not re.search(r"^try\b", text, re.M), name


@pytest.mark.parametrize("rec, match", [
    ({}, "no 'status'"),
    ({"status": "error", "error": "BoundsError at run_dmrg"}, "BoundsError"),
    ({k: v for k, v in _run_jl_record("dmrg").items() if k != "energy"}, r"lacks \['energy'\]"),
])
def test_itensor_parse_fails_clearly(rec, match):
    import itensor_runner as it
    with pytest.raises(RuntimeError, match=match):
        it.parse("dmrg", rec)


def test_itensor_records_pass_the_smoke_check_and_the_scorer():
    import analyse2
    import itensor_runner as it
    import smoke2_check
    spec = load_spec(os.path.join(SUITE, "suite_2_smoke.json"))
    recs = []
    for kind, wl, chi, sym in (("dmrg", "xxz_dmrg", 64, "u1"), ("excited", "tfim_excited", 64, "dense"),
                               ("tdvp", "xx_quench", 16, "dense")):
        r = {"workload": wl, "kind": kind, "lib": "itensor", "backend": "julia-cpu", "sym": sym,
             "chi": chi, "rep": 1, "threads": 4, "env": {"cgroup_cpus": 4.0, "env_name": "julia"}}
        r.update(it.parse(kind, _run_jl_record(kind)))       # as run.py merges a runner's output
        r["status"] = "ok"
        assert smoke2_check.check(r, spec["workloads"][wl]) is None, wl
        recs.append(r)
    refs = analyse2.references(spec, recs)
    refs["xxz_dmrg"] = {"energy": reference.xxz_energy(12, 1.0, 1.0)}
    _, rows = analyse2.score(spec, recs, {}, refs)
    assert {r["workload"] for r in rows} == {"xxz_dmrg", "tfim_excited", "xx_quench"}
    for row in rows:
        assert row["n_ok"] == 1 and row["hardware"] == "cpu4" and row["env"] == "julia"
        assert row["err"] < 1e-9 and not row["void"] and row["peak_rss_mb"] == 1187.4
    dm = next(r for r in rows if r["workload"] == "xxz_dmrg")
    assert dm["s_per_sweep_med"] == pytest.approx((0.0811 + 0.0759) / 2)
    assert not dm["s_per_sweep_whole_run"] and dm["sweeps"] == [5]


def test_scaling_fit_rule():
    import analyse2
    reps = [[1.0, 1.1], [9.0, 9.5], [70.0, 72.0]]
    f = analyse2.scaling_fit([256, 512, 1024], reps)
    assert f["fitted"] and 2.9 < f["p"] < 3.1 and f["sigma_p"] < 0.1
    assert not analyse2.scaling_fit([256, 512, 1024], [[1.0], [2.0], [9.0]])["fitted"]


SPEC2 = load_spec(os.path.join(SUITE, "suite_2.json"))
SET2 = json.load(open(os.path.join(SUITE, "settings_suite2.json")))
E_CHAIN = -28.17542485974319                     # the chain's confirmed chi-512 reference


def _small(name, settings=True, lib="tnkit"):
    """A suite-2 workload on a few sites, with the library's suite-2 settings (or none)."""
    w = dict(SPEC2["workloads"][name])
    w.update(Lx=4, Ly=3, N=12) if w.get("model") == "heis_cyl" else w.update(N=8)
    if settings:
        w["opts"] = settings_for(SET2, lib, w["kind"], name)
    return w


def _capture_dmrg(monkeypatch):
    import tnkit.api
    calls = []

    def fake(mpo, **kw):
        calls.append(kw)
        n = kw["n_sweeps"]
        return -1.0, [np.zeros((1, 2, kw["chi"]))], {"energies": [-1.0] * n, "sweeps": n}

    monkeypatch.setattr(tnkit.api, "dmrg", fake)
    return calls


CYL_STAGES = [(8, 2, 0.0, 0.0), (16, 2, 0.0, 0.0), (32, 2, 0.0, 0.0), (64, 10, 46e-8, 0.0)]


@pytest.mark.parametrize("name, settings, chi, want", [
    ("xxz_dmrg", False, 16, [(16, 20, 1e-10, 0.0)]),
    ("xxz_dmrg", True, 16, [(16, 20, 1e-10, 1e-10)]),
    ("xxz_ref", True, 16, [(16, 30, 1e-12, 0.0)]),
    ("heis_cyl6", True, 64, CYL_STAGES),
    ("heis_cyl6", False, 64, CYL_STAGES),
    ("heis_throughput", True, 16, [(16, 3, 0.0, 0.0)]),
])
def test_tnkit_dmrg_passes_its_stop_and_expansion(monkeypatch, name, settings, chi, want):
    """tnkit.api.dmrg gets the stop and expansion of the workload and settings, never its own
    defaults: (chi, sweeps, tol, rtol) per stage, expand 0."""
    import tnkit_runner as tk
    calls = _capture_dmrg(monkeypatch)
    out = tk.dmrg(_small(name, settings), chi, backend="numpy", sym="dense")
    got = [(c["chi"], c["n_sweeps"], c["tol"], c["rtol"]) for c in calls]
    assert got == [pytest.approx(w, rel=1e-12) for w in want]
    assert [c["expand"] for c in calls] == [0] * len(want)
    assert (out["settings"]["tol"], out["settings"]["rtol"]) == pytest.approx(want[-1][2:])


def test_tnkit_dmrg_expansion_opens_the_run_only(monkeypatch):
    import tnkit_runner as tk
    calls = _capture_dmrg(monkeypatch)
    w = {**_small("heis_cyl6"), "opts": {"expand": 3, "rtol": 0.0}}
    tk.dmrg(w, 64, backend="numpy", sym="dense")
    assert [c["expand"] for c in calls] == [3, 0, 0, 0]


@pytest.mark.parametrize("opts", [None, {}, {"krylov_m": 12}])
def test_tnkit_tdvp_passes_the_workload_and_settings(monkeypatch, opts):
    import tnkit.tdvp
    import tnkit.tdvp_evolver
    import tnkit_runner as tk
    seen = {"runs": []}

    class Evolver:
        def __init__(self, M, W, dt, chi, options):
            seen.update(dt=dt, chi=chi, options=options)
            self.mps = M

        def run(self, n, mode):
            seen["runs"].append((n, mode))

    monkeypatch.setattr(tnkit.tdvp, "TDVPOptions", lambda **kw: kw)
    monkeypatch.setattr(tnkit.tdvp_evolver, "TDVPEvolver", Evolver)
    w = _small("xx_quench", settings=False)
    if opts is not None:
        w["opts"] = opts
    tk.tdvp(w, 16, "numpy", "dense")
    assert seen["options"] == {"svd_tol": w["svd_tol"], "chi_max": 16, "backend": "numpy",
                               **(opts or {})}
    assert (seen["dt"], seen["chi"]) == (w["dt"], 16)
    assert seen["runs"] == [(1, "12site")] * int(round(w["T"] / w["dt"]))
    assert settings_for(SET2, "tnkit", "tdvp", "xx_quench") == {}


def test_every_library_stops_the_chain_on_one_relative_rule():
    """xxz_dmrg: |dE| < 1e-10 |E_ref| for each library; xxz_ref and the cylinder keep their
    fixed absolute stops."""
    import block2_runner as b2
    import itensor_runner as it
    from common import stop_tol
    for name, want in (("xxz_dmrg", 1e-10 * abs(E_CHAIN)), ("xxz_ref", 1e-12),
                       ("heis_cyl6", 1e-8 * 46)):
        w = SPEC2["workloads"][name]
        o = {lib: settings_for(SET2, lib, "dmrg", name) for lib in SET2 if lib != "_about"}
        rel = {lib: max(stop_tol(w), o[lib].get("rtol", 0.0) * abs(E_CHAIN))
               for lib in ("tnkit", "yastn")}
        job = it._job("dmrg", {**w, "opts": o["itensor"]}, 64, "julia-cpu", "u1")
        rel["itensor"] = max(job["tol"], job["rtol"] * abs(E_CHAIN))
        rel["block2"] = b2.stop_of(w, {**b2.DEFAULTS, **o["block2"]})
        if name == "xxz_dmrg":                   # TeNPy: max_E_err relative to max(|E|, 1)
            rel["tenpy"] = stop_tol(w) / w.get("energy_scale", 1.0) * abs(E_CHAIN)
        assert rel == pytest.approx({k: want for k in rel}, rel=1e-12), name
    assert SET2["block2"]["xxz_dmrg"]["e_ref"] == E_CHAIN


def test_block2_stop():
    import block2_runner as b2
    w = SPEC2["workloads"]["xxz_dmrg"]
    assert b2.stop_of(w, {}) == 1e-10
    assert b2.stop_of(w, {"rtol": 1e-10, "e_ref": -100.0}) == pytest.approx(1e-8)
    assert b2.stop_of({**w, "fixed_sweeps": 3}, {"rtol": 1e-10, "e_ref": -100.0}) == 0.0
    with pytest.raises(ValueError, match="e_ref"):
        b2.stop_of(w, {"rtol": 1e-10})


def test_itensor_job_carries_rtol():
    import itensor_runner as it
    for name, kind, tol, rtol in (("xxz_dmrg", "dmrg", 1e-10, 1e-10),
                                  ("xxz_ref", "dmrg", 1e-12, 0.0),
                                  ("heis_cyl6", "dmrg", 46e-8, 0.0),
                                  ("heis_throughput", "dmrg", -1.0, 0.0),
                                  ("tfim_excited", "excited", 1e-10, 0.0)):
        job = it._job(kind, _small(name, lib="itensor"), 16, "julia-cpu", "dense")
        assert (job["tol"], job["rtol"]) == pytest.approx((tol, rtol), rel=1e-12), name
        assert isinstance(job["rtol"], float) and "rtol = " in it.toml(job)
    assert "max(o.tol, o.rtol * abs(o.e[end]))" in open(it.RUN_JL).read()
