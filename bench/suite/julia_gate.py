"""Gate of the Julia arms on one machine: ITensor's smoke energies (jl/probe/smoke.jl, Heisenberg
N 16, chi 64) against tnkit's (julia_probe.py ref), within 1e-10.

    python3 bench/suite/julia_gate.py DIR      # DIR holds ref.json, smoke_cpu.json[, smoke_cuda.json]

Prints TNBENCH_JULIA_OK=0|1 (dense and QN on the CPU pass, and dense on CUDA where it ran) and
TNBENCH_JULIA_QN_CUDA=0|1 (also QN on CUDA), for the job script to export.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import julia_probe as jp  # noqa: E402

d = sys.argv[1]
rf = jp.load(d, "ref.json") or {}
e = {s: (rf.get(s) or {}).get("energy") for s in ("dense", "u1")}
cpu = jp.arms(d, "smoke_cpu.json", e)
ok = cpu["dense"]["pass"] and cpu["qn"]["pass"]
qn_cuda = False
if os.path.exists(os.path.join(d, "smoke_cuda.json")):
    cuda = jp.arms(d, "smoke_cuda.json", e, cuda=True)
    ok = ok and cuda["dense"]["pass"]
    qn_cuda = cuda["qn"]["pass"]
    print("# cuda", json.dumps(cuda)[:800], file=sys.stderr)
print("# cpu", json.dumps(cpu)[:800], file=sys.stderr)
print(f"TNBENCH_JULIA_OK={int(ok)}")
print(f"TNBENCH_JULIA_QN_CUDA={int(ok and qn_cuda)}")
