"""Time-step error of the large quench: tnkit dense TDVP at an exact bond dimension (no
truncation), N 16, T 12, at dt 0.1 (the workload's) and 0.05, against free fermions.

    python3 bench/suite/dt_check.py --backend torch-cuda --out dt_check.json

Exit status 1 if the dt 0.1 error exceeds 2e-5 (the workload would then move to dt 0.05).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "runners"), os.path.dirname(os.path.dirname(HERE))]

import tnkit_runner as tk  # noqa: E402
from reference import xx_neel_sz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="numpy")
ap.add_argument("--out", required=True)
ap.add_argument("--N", type=int, default=16)
a = ap.parse_args()
rec = {"N": a.N, "T": 12.0, "chi": 2 ** (a.N // 2), "backend": a.backend}
for dt in (0.1, 0.05):
    w = {"N": a.N, "J": 1.0, "Jz": 0.0, "dt": dt, "T": 12.0, "svd_tol": 1e-12}
    out = tk.tdvp(w, rec["chi"], a.backend, "dense")
    rec[f"err_dt{dt}"] = float(np.max(np.abs(np.array(out["sz"]) - xx_neel_sz(a.N, 1.0, 12.0))))
    rec[f"wall_dt{dt}"] = out["wall_s"]
    print(rec, flush=True)
rec["pass"] = rec["err_dt0.1"] <= 2e-5
with open(a.out, "w") as f:
    json.dump(rec, f, indent=1)
sys.exit(0 if rec["pass"] else 1)
