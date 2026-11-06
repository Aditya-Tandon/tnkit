"""One benchmark run: warm up, time one workload at one bond dimension, append one JSON line.

    python bench/suite/run.py --spec bench/suite/suite.json --workload xxz_dmrg --lib yastn \
        --backend numpy --sym u1 --chi 64 --rep 1 --out results.jsonl

The library comes from whatever environment runs this file; `schedule.py` sets PYTHONPATH and
thread variables per arm. Exit status 0 means a record was written, whatever its status.

The `tf_collision` workload starts from a saved two-flavour Schwinger state and its MPO
(`state_gt0020.000.npz`, `mpo.npz`), which are not distributed. Set `TNBENCH_DATA` to the
directory that holds them; without it the workload is recorded as unsupported.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "runners"), os.path.dirname(os.path.dirname(HERE))]

from common import SCHEMA, Clock, Unsupported, append_record, environment, load_settings, \
    load_spec, peaks, require_data, reset_gpu_peak, set_torch_threads, settings_for, \
    utc  # noqa: E402



def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--lib", required=True,
                    choices=["tnkit", "yastn", "qtea", "tenpy", "block2", "itensor"])
    ap.add_argument("--backend", required=True,
                    choices=["numpy", "torch-cuda", "torch-cpu", "cpu", "julia-cpu", "julia-cuda"])
    ap.add_argument("--sym", required=True, choices=["dense", "u1", "su2"])
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args(argv)

    spec = load_spec(a.spec)
    w = dict(spec["workloads"][a.workload])
    settings, settings_sha, settings_path = load_settings(spec)
    if settings_sha:
        w["opts"] = settings_for(settings, a.lib, w["kind"], a.workload)
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    set_torch_threads(threads)
    rec = {"schema": SCHEMA, "suite": spec["name"], "spec_sha256": spec["_sha256"],
           "pod": os.environ.get("TNBENCH_POD"), "workload": a.workload, "kind": w["kind"],
           "lib": a.lib, "backend": a.backend, "sym": a.sym, "chi": a.chi, "rep": a.rep,
           "threads": threads, "started_utc": utc()}
    if settings_sha:
        rec.update(settings_sha256=settings_sha, settings_file=settings_path, opts=w["opts"])
    runner = importlib.import_module(f"{a.lib}_runner")
    try:
        runner.check(w, a.sym)
        require_data(w)
        rec["lib_version"] = runner.version()
        if not a.no_warmup:
            with Clock(a.backend) as c:
                runner.warmup(w["kind"], a.backend, a.sym)
            rec["warmup_s"] = c.s
        reset_gpu_peak(a.backend)
        out = getattr(runner, w["kind"])(w, a.chi, a.backend, a.sym)
        rec.update(out)
        rec["status"] = "ok"
    except Unsupported as e:
        rec.update(status="unsupported", error=str(e))
    except Exception as e:  # noqa: BLE001 -- a library failure is a result, recorded as such
        rec.update(status="error", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-4000:])
    rec.update(peaks(a.backend))
    rec["env"] = environment(a.backend)
    rec["finished_utc"] = utc()
    append_record(a.out, rec)
    print(a.lib, a.backend, a.sym, a.workload, a.chi, a.rep, rec["status"],
          round(rec.get("wall_s", float("nan")), 3), flush=True)


if __name__ == "__main__":
    main()
