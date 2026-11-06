"""ITensor (ITensors.jl + ITensorMPS.jl, Julia): dense or U(1) (QN) tensors, CPU or CUDA.

Each run is one Julia process running `jl/bench/run.jl`, which reads the run from a TOML file,
warms up on a tiny problem of the same kind, times the real one and writes raw outputs as JSON;
this module writes the TOML, runs Julia and returns its outputs. Package loading and compilation
happen before the warm-up and are never timed. Needs TNBENCH_JULIA (the julia binary) and
TNBENCH_JULIA_PROJECT (the environment, with its depot on JULIA_DEPOT_PATH).

Settings follow the ITensor documentation: DMRG with the noise schedule of its examples
(1e-6, 1e-7, 1e-8, then 0) and `eigsolve_krylovdim` 3; two-site TDVP with `exponentiate`. The
settings file (w["opts"]) may set `noise`, `cutoff`, `eigsolve_krylovdim`, `eigsolve_maxiter`,
`weight` (excited states), `threading` ("auto": dense uses BLAS threads, QN on the CPU uses
ITensor's threaded block-sparse contractions with one BLAS thread) and `rtol`: the energy stop
is |dE| < max(tol, rtol |E|), tol being the workload's absolute stop.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from common import Unsupported, bonds, couplings, neel, options, schedule_of, stop_tol

NAME = "itensor"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_JL = os.path.join(os.path.dirname(HERE), "jl", "bench", "run.jl")
OPTS = ("noise", "cutoff", "eigsolve_krylovdim", "eigsolve_maxiter", "weight", "threading",
        "rtol")
DEFAULTS = {"noise": [1e-6, 1e-7, 1e-8, 0.0], "cutoff": 1e-14, "eigsolve_krylovdim": 3,
            "eigsolve_maxiter": 1, "weight": 20.0, "threading": "auto", "rtol": 0.0}
# The keys of run.jl's record on success, by kind (beside status, settings and the environment).
OUTPUTS = {"dmrg": ("wall_s", "energy", "energies", "sweeps", "sweep_s", "max_bond", "dtype"),
           "excited": ("wall_s", "energies_k", "sweeps_k", "dtype"),
           "tdvp": ("wall_s", "steps", "step_s", "sz", "bonds", "dtype")}


def version():
    return _manifest_versions().get("ITensorMPS")


def check(w, sym):
    if sym == "su2":
        raise Unsupported("ITensor has no SU(2) tensors")
    if w["kind"] == "collision":
        raise Unsupported("the two-flavour collision is not ported to ITensor")
    if w.get("model") == "tfim" and sym != "dense":
        raise Unsupported("the Ising chain has no U(1) symmetry")
    if not (os.environ.get("TNBENCH_JULIA") and os.environ.get("TNBENCH_JULIA_PROJECT")):
        raise RuntimeError("TNBENCH_JULIA and TNBENCH_JULIA_PROJECT must be set")


def _manifest_versions():
    path = os.path.join(os.environ.get("TNBENCH_JULIA_PROJECT", ""), "Manifest.toml")
    try:
        import tomllib
        with open(path, "rb") as f:
            deps = tomllib.load(f)["deps"]
    except (ImportError, OSError, KeyError, ValueError):
        return {}
    return {k: v[0].get("version") for k, v in deps.items()}


def toml(d, prefix=""):
    """A flat TOML document from a dict of scalars, lists and dicts (dicts become tables)."""
    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(float(v)) if isinstance(v, float) else str(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        raise TypeError(type(v))
    lines = [f"{k} = {val(v)}" for k, v in d.items() if not isinstance(v, dict)]
    for k, v in d.items():
        if isinstance(v, dict):
            lines += ["", f"[{prefix}{k}]", toml(v, f"{prefix}{k}.")]
    return "\n".join(lines)


def _job(kind, w, chi, backend, sym):
    """Everything run.jl needs, with the model reduced to explicit couplings."""
    opts = {**DEFAULTS, **options(w, OPTS)}
    job = {"kind": kind, "chi": chi, "device": "cuda" if "cuda" in backend else "cpu",
           "qn": sym == "u1", "threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
           "model": w["model"], "N": w["N"], "opts": opts}
    if w["model"] == "tfim":
        job["g"] = w["g"]
    else:
        J, Jz = couplings(w)
        job.update(J=J, Jz=Jz, bonds=[list(b) for b in bonds(w)],
                   states=["Up" if s == 0 else "Dn" for s in neel(w["N"])])
    if kind == "dmrg":
        stages = schedule_of(w, chi)
        fixed = bool(w.get("fixed_sweeps"))
        job.update(maxdims=[b for b, n in stages for _ in range(n)],
                   tol=-1.0 if fixed else stop_tol(w), rtol=0.0 if fixed else float(opts["rtol"]),
                   minsweeps=sum(n for _, n in stages[:-1]) + 2)
    elif kind == "excited":
        job.update(nstates=w["nstates"], nsweeps=w["max_sweeps"], tol=w["e_tol"],
                   rtol=float(opts["rtol"]))
    elif kind == "tdvp":
        job.update(dt=w["dt"], steps=int(round(w["T"] / w["dt"])), cutoff_tdvp=w["svd_tol"])
    return job


def _run(kind, w, chi, backend, sym):
    job = _job(kind, w, chi, backend, sym)
    with tempfile.TemporaryDirectory(prefix="itensor_") as d:
        src, out = os.path.join(d, "run.toml"), os.path.join(d, "out.json")
        with open(src, "w") as f:
            f.write(toml(job) + "\n")
        threads = os.environ.get("OMP_NUM_THREADS", "1")
        cmd = [os.environ["TNBENCH_JULIA"], f"--project={os.environ['TNBENCH_JULIA_PROJECT']}",
               "--startup-file=no", f"--threads={threads}", RUN_JL, src, out]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out):
            raise RuntimeError(f"julia exited {p.returncode}: {p.stderr[-3000:]}")
        with open(out) as f:
            rec = parse(kind, json.load(f), p.stderr)
    rec["settings"]["toml"] = job["opts"]
    if kind in ("dmrg", "excited"):
        rule = "fixed sweeps, no energy stop" if job["tol"] < 0 else \
            "|dE| < max(tol, rtol |E|) after " + ("minsweeps" if kind == "dmrg" else "2 sweeps")
        rec["settings"]["stop"] = {"tol": job["tol"], "rtol": job["rtol"],
                                   "minsweeps": job.get("minsweeps", 2), "rule": rule}
    rec["julia_packages"] = {k: v for k, v in _manifest_versions().items()
                             if k in ("ITensors", "ITensorMPS", "NDTensors", "KrylovKit",
                                      "CUDA", "cuTENSOR")}
    return rec


def parse(kind, rec, stderr=""):
    """run.jl's record without its status; raises if the run failed or a key is missing."""
    if not isinstance(rec, dict) or "status" not in rec:
        raise RuntimeError(f"run.jl record has no 'status': {json.dumps(rec)[:1000]}; "
                           f"julia stderr: {stderr[-2000:]}")
    rec = dict(rec)
    if rec.pop("status") != "ok":
        raise RuntimeError(rec.get("error", "julia run failed"))
    missing = [k for k in OUTPUTS[kind] + ("settings", "julia_env", "peak_rss_mb_julia")
               if k not in rec]
    if missing:
        raise RuntimeError(f"run.jl {kind} record lacks {missing}: {json.dumps(rec)[:1000]}")
    return rec


def dmrg(w, chi, backend, sym):
    return _run("dmrg", w, chi, backend, sym)


def excited(w, chi, backend, sym):
    return _run("excited", w, chi, backend, sym)


def tdvp(w, chi, backend, sym):
    return _run("tdvp", w, chi, backend, sym)


def warmup(kind, backend, sym):
    """Nothing here: each Julia process warms itself up after loading its packages."""
