"""Exit 1 unless block2 loads and runs here: U(1) and SU(2) DMRG of the Heisenberg chain N 8, which
must agree. Prints the MKL libraries the process mapped. Run it in block2's run environment
(PYTHONPATH and LD_LIBRARY_PATH as `schedule.py` sets them).

    python bench/suite/block2_check.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "runners")]

W = {"kind": "dmrg", "model": "xxz", "N": 8, "J": 1.0, "Jz": 1.0, "max_sweeps": 10, "e_tol": 1e-12}

try:
    import block2_runner as b
    e = {s: b.dmrg(W, 16, backend="numpy", sym=s)["energy"] for s in ("u1", "su2")}
except Exception as x:  # noqa: BLE001
    print(f"block2 check: {type(x).__name__}: {x}")
    sys.exit(1)
with open("/proc/self/maps") as f:
    mkl = sorted({os.path.basename(ln.split()[-1]) for ln in f if "libmkl" in ln})
print(f"block2 {b.version()} E(u1) {e['u1']:.12f} E(su2) {e['su2']:.12f} mkl {' '.join(mkl)}")
sys.exit(0 if abs(e["u1"] - e["su2"]) < 1e-8 else 1)
