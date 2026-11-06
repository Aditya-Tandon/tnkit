"""Check for a benchmark machine: block tensors against dense tensors and exact answers on one backend.

    python bench/suite/block_smoke.py --backend torch-cuda --out gate.json

Runs the benchmark's own tnkit runner on tiny problems at exact bond dimension:
- DMRG, Heisenberg chain N = 12, chi 64: block (U(1), 2 S^z = 0) and dense against exact
  diagonalisation, |E - E_exact| <= 1e-10 each;
- TDVP, XX quench from the Neel state, N = 8, cap 16, T = 0.5: block and dense against each other
  (<= 1e-10) and against free fermions (<= 1e-6).
On a CUDA backend it also requires a GPU and nonzero GPU memory use. Exit status 0 iff all pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "runners"), os.path.dirname(os.path.dirname(HERE))]

import tnkit_runner as tk  # noqa: E402
from common import cuda_backend, environment  # noqa: E402
from reference import xx_neel_sz, xxz_energy  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["numpy", "torch-cpu", "torch-cuda"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    checks = {}
    if cuda_backend(a.backend):
        import torch
        checks["cuda_available"] = bool(torch.cuda.is_available())
        if not checks["cuda_available"]:
            return report(a, checks, {})
    w = {"kind": "dmrg", "model": "xxz", "N": 12, "J": 1.0, "Jz": 1.0, "max_sweeps": 10,
         "e_tol": 1e-12}
    e0 = xxz_energy(12, 1.0, 1.0)
    dm = {s: tk.dmrg(w, 64, backend=a.backend, sym=s) for s in ("u1", "dense")}
    q = {"kind": "tdvp", "model": "xx", "N": 8, "J": 1.0, "Jz": 0.0, "dt": 0.05, "T": 0.5,
         "svd_tol": 1e-12}
    td = {s: np.array(tk.tdvp(q, 16, a.backend, s)["sz"]) for s in ("u1", "dense")}
    ex = xx_neel_sz(8, 1.0, 0.5)
    vals = {"E_exact": e0,
            "dmrg_block_minus_exact": dm["u1"]["energy"] - e0,
            "dmrg_dense_minus_exact": dm["dense"]["energy"] - e0,
            "tdvp_block_minus_dense": float(np.max(np.abs(td["u1"] - td["dense"]))),
            "tdvp_block_minus_exact": float(np.max(np.abs(td["u1"] - ex)))}
    checks.update(dmrg_block=abs(vals["dmrg_block_minus_exact"]) <= 1e-10,
                  dmrg_dense=abs(vals["dmrg_dense_minus_exact"]) <= 1e-10,
                  dmrg_blocks_used=bool(dm["u1"]["settings"]["blocks"]),
                  tdvp_block_dense=vals["tdvp_block_minus_dense"] <= 1e-10,
                  tdvp_exact=vals["tdvp_block_minus_exact"] <= 1e-6)
    if cuda_backend(a.backend):
        import torch
        vals["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        checks["gpu_used"] = vals["peak_gpu_mb"] > 0
    return report(a, checks, vals)


def report(a, checks, vals):
    ok = bool(checks) and all(checks.values())
    rec = {"backend": a.backend, "pass": ok, "checks": checks, "values": vals,
           "env": environment(a.backend)}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=float)
    print("BLOCK GATE", "PASS" if ok else "FAIL", a.backend, json.dumps(checks),
          json.dumps(vals, default=float))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
