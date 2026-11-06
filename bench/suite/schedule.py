"""Run one pod's share of the suite: every arm, bond dimension and repeat, interleaved.

    python bench/suite/schedule.py --spec bench/suite/suite.json --pod cpu-dmrg --out DIR

Each run is its own process (`run.py`) under a timeout, so a library that hangs or leaks costs
one run, and each process's peak memory is its own. Order: repeat, then workload, then bond
dimension, then arm (rotated by one place each repeat), so that the arms of one comparison run
close together in time on the same hardware. An arm stops climbing a workload's ladder after a
timeout there; a run that reports `unsupported` drops that arm from the workload. A pod-level
budget stops new runs once it is spent. Every run, skipped or not, leaves one JSON line in
DIR/runs.jsonl. Refuses to start if that file exists.

Optional spec fields (suite 2): an arm may carry its own `threads`, `chis` per workload and a
`gate` (an environment variable that must be "1", else its runs are recorded as skipped_gate);
a pod may carry `reps_by_workload`, `memwatch` (peak memory of each run's process tree and of the
GPU, sampled from here into DIR/memory.jsonl) and `sampler_check` (in the first repeat, a second
run of one cell with the sampler off, at rep 101, to measure the sampler's cost). A spec's
`spread_rerun` reruns, after the plan, every arm of any (workload, chi) where one arm's slowest
repeat exceeds `ratio` times its fastest: `extra` more repeats, interleaved and rotated as above.
block2 runs get TNBENCH_BLOCK2_LIB (the directory of the MKL libraries it links) on LD_LIBRARY_PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from common import SCHEMA, append_record, load_spec, sweep_time, utc  # noqa: E402

SAMPLER_OFF = 100                                # rep offset of the sampler-off check runs


def arm_key(arm):
    key = f"{arm['lib']}/{arm['backend']}/{arm['sym']}/{arm['env']}"
    return key + (f"/t{arm['threads']}" if "threads" in arm else "")


def ladder(spec, p, wl, arm):
    """The bond dimensions this arm runs on this workload: the arm's own, else the pod's, else
    the workload's."""
    return arm.get("chis", {}).get(wl) or p.get("chis", {}).get(wl) or spec["workloads"][wl]["chis"]


def plan(spec, pod, reps, first_rep=1, cells=None):
    """The ordered list of (rep, workload, chi, arm): repeat, workload, chi, then the arms
    rotated by one place each repeat. `cells` restricts it to those (workload, chi)."""
    p = spec["pods"][pod]
    check = p.get("sampler_check")
    out = []
    for rep in range(first_rep, reps + 1):
        for wl in p["workloads"]:
            if rep > p.get("reps_by_workload", {}).get(wl, reps):
                continue
            arms = [a for a in p["arms"] if wl in a.get("workloads", p["workloads"])]
            k = (rep - 1) % max(len(arms), 1)
            arms = arms[k:] + arms[:k]
            chis = sorted({c for a in arms for c in ladder(spec, p, wl, a)})
            for chi in chis:
                for arm in arms:
                    if chi not in ladder(spec, p, wl, arm):
                        continue
                    if cells is not None and (wl, chi) not in cells:
                        continue
                    out.append((rep, wl, chi, arm))
                    if check and cells is None and rep == 1 and \
                            (wl, chi) == (check["workload"], check["chi"]):
                        out.append((rep + SAMPLER_OFF, wl, chi, arm))
    return out


def child_env(spec, pod, arm, threads=None):
    p = spec["pods"][pod]
    env = dict(os.environ)
    site = os.path.expandvars(spec["envs"][arm["env"]]["site"])
    n = str(threads or arm.get("threads") or p["threads"])
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n,
               MKL_NUM_THREADS=n, NUMEXPR_NUM_THREADS=n, TNBENCH_ENV=arm["env"], TNBENCH_POD=pod,
               PYTHONPATH=os.pathsep.join(x for x in (site, REPO) if x))
    if arm["lib"] == "block2" and os.environ.get("TNBENCH_BLOCK2_LIB"):     # its MKL libraries
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            x for x in (os.environ["TNBENCH_BLOCK2_LIB"], env.get("LD_LIBRARY_PATH")) if x)
    return env


def stub(spec, pod, rep, wl, chi, arm, threads, status, **extra):
    """The record of a run that wrote none itself (skipped, timed out or crashed)."""
    return {"schema": SCHEMA, "suite": spec["name"], "spec_sha256": spec["_sha256"],
            "pod": pod, "workload": wl, "kind": spec["workloads"][wl]["kind"], "lib": arm["lib"],
            "backend": arm["backend"], "sym": arm["sym"], "env_name": arm["env"], "chi": chi,
            "rep": rep, "threads": threads or arm.get("threads") or spec["pods"][pod]["threads"],
            "status": status, **extra, "finished_utc": utc()}


def run_one(spec, pod, rep, wl, chi, arm, out, timeout, spec_path, python, threads=None,
            dev=False):
    """One run in its own process group. `dev`: an `memwatch.Nvml` (or None) to sample memory
    into memory.jsonl beside `out`; False: no sampling."""
    cmd = [python, os.path.join(HERE, "run.py"), "--spec", spec_path, "--workload", wl,
           "--lib", arm["lib"], "--backend", arm["backend"], "--sym", arm["sym"],
           "--chi", str(chi), "--rep", str(rep), "--out", out]
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=child_env(spec, pod, arm, threads), start_new_session=True)
    watch = None
    if dev is not False and rep < SAMPLER_OFF:
        import memwatch
        watch = memwatch.Watch(proc.pid, dev)
        watch.start()
    try:
        try:
            rc = proc.wait(timeout=timeout)
        finally:
            if watch is not None:
                append_record(os.path.join(os.path.dirname(out), "memory.jsonl"),
                              {"workload": wl, "chi": chi, "rep": rep, "arm": arm_key(arm),
                               "lib": arm["lib"], "backend": arm["backend"], "sym": arm["sym"],
                               "env_name": arm["env"], "threads": threads or
                               arm.get("threads") or spec["pods"][pod]["threads"],
                               **watch.stop(), "finished_utc": utc()})
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        append_record(out, stub(spec, pod, rep, wl, chi, arm, threads, "timeout",
                                wall_s=timeout))
        return "timeout"
    if rc != 0:                                  # the child died before writing its record
        append_record(out, stub(spec, pod, rep, wl, chi, arm, threads, "crash", returncode=rc,
                                elapsed_s=time.time() - t0))
        return "crash"
    with open(out) as f:
        last = f.readlines()[-1]
    return json.loads(last)["status"]


def ranked_time(r, w):
    """The time a run is ranked by: the mean step after `exclude_first` (collision), seconds per
    sweep (fixed-sweep throughput), else wall."""
    if w["kind"] == "collision":
        return sum(r["step_s"][w["exclude_first"]:]) / len(r["step_s"][w["exclude_first"]:])
    if w.get("fixed_sweeps"):
        return sweep_time(r)[0]
    return r["wall_s"]


def spread_cells(spec, pod, out, ratio):
    """(workload, chi) where some arm's slowest ok repeat exceeds `ratio` times its fastest
    (sampler-off runs excluded)."""
    times = {}
    with open(out) as f:
        for line in f:
            r = json.loads(line)
            if r["status"] != "ok" or r["rep"] >= SAMPLER_OFF:
                continue
            arm = (r["lib"], r["backend"], r["sym"],
                   r.get("env_name") or (r.get("env") or {}).get("env_name"), r.get("threads"))
            times.setdefault((r["workload"], r["chi"], arm), []).append(
                ranked_time(r, spec["workloads"][r["workload"]]))
    return {k[:2] for k, v in times.items() if len(v) > 1 and max(v) > ratio * min(v)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--pod", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--threads", type=int, default=None,
                    help="override the pod's thread count (laptop smoke tests only)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    p = spec["pods"][a.pod]
    reps = a.reps or p.get("reps", spec["repeats"])
    order = plan(spec, a.pod, reps)
    if a.dry_run:
        for rep, wl, chi, arm in order:
            print(rep, wl, chi, arm_key(arm))
        print(len(order), "runs")
        return
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "runs.jsonl")
    if os.path.exists(out):
        sys.exit(f"{out} exists: refusing to overwrite")
    with open(os.path.join(a.out, "plan.json"), "w") as f:
        json.dump({"spec": spec["_path"], "spec_sha256": spec["_sha256"], "pod": a.pod,
                   "reps": reps, "threads": a.threads or p["threads"], "started_utc": utc(),
                   "order": [[r, w, c, arm_key(m)] for r, w, c, m in order]}, f, indent=1)
    dev = False
    if p.get("memwatch"):
        import memwatch
        dev = memwatch.gpu() if p["hardware"] == "gpu" else None
    t_start, dead, ceiling = time.time(), set(), {}
    log = open(os.path.join(a.out, "progress.log"), "a")

    def execute(order):
        for rep, wl, chi, arm in order:
            key = (arm_key(arm), wl)
            skip = None
            if arm.get("gate") and os.environ.get(arm["gate"]) != "1":
                skip = "skipped_gate"
            elif key in dead:
                skip = "skipped_unsupported"
            elif chi >= ceiling.get(key, float("inf")):
                skip = "skipped_after_timeout"
            elif time.time() - t_start > 3600 * p["budget_h"]:
                skip = "skipped_budget"
            if skip:
                append_record(out, stub(spec, a.pod, rep, wl, chi, arm, a.threads, skip))
                status = skip
            else:
                timeout = spec["workloads"][wl]["timeout_s"][p["hardware"]]
                status = run_one(spec, a.pod, rep, wl, chi, arm, out, timeout, spec["_path"],
                                 a.python, a.threads, dev)
                if status == "unsupported":
                    dead.add(key)
                elif status == "timeout":
                    ceiling[key] = min(chi, ceiling.get(key, chi))
            log.write(f"{utc()} rep {rep} {wl} chi {chi} {arm_key(arm)} {status} "
                      f"elapsed {time.time() - t_start:.0f} s\n")
            log.flush()

    execute(order)
    rerun = spec.get("spread_rerun")
    if rerun:
        cells = spread_cells(spec, a.pod, out, rerun["ratio"])
        extra = plan(spec, a.pod, reps + rerun["extra"], first_rep=reps + 1, cells=cells)
        log.write(f"{utc()} spread rerun: {len(cells)} cells, {len(extra)} runs\n")
        with open(os.path.join(a.out, "spread_rerun.json"), "w") as f:
            json.dump({"ratio": rerun["ratio"], "cells": sorted(cells),
                       "order": [[r, w, c, arm_key(m)] for r, w, c, m in extra]}, f, indent=1)
        execute(extra)
    log.write(f"{utc()} done\n")
    log.close()


if __name__ == "__main__":
    main()
