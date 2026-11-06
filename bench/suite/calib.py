"""BLAS and GPU-library calibration of the environment this process runs in, so that a gap
between libraries can be split into library code and the linear algebra underneath it.

    python3 bench/suite/calib.py --backend numpy|torch-cuda --out calib.json

numpy: dgemm (n 4096) and the real and complex SVD (n 1024) at OMP_NUM_THREADS threads.
torch-cuda: the same on the GPU (float64, complex128).
"""
from __future__ import annotations

import argparse
import json
import os
import time


def rate(f, flops, sync=lambda: None, reps=3):
    f()
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        sync()
        ts.append(time.perf_counter() - t)
    m = sorted(ts)[(reps - 1) // 2]
    return {"median_s": m, "tflops": flops / m / 1e12}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "torch-cuda"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    n, m = 4096, 1024
    rec = {"backend": a.backend, "threads": os.environ.get("OMP_NUM_THREADS"),
           "env_name": os.environ.get("TNBENCH_ENV")}
    if a.backend == "numpy":
        import numpy as np
        rng = np.random.default_rng(0)
        A, B, S = rng.random((n, n)), rng.random((n, n)), rng.random((m, m))
        Z = rng.random((m, m)) + 1j * rng.random((m, m))
        rec["numpy"] = np.__version__
        rec["dgemm_4096"] = rate(lambda: A @ B, 2.0 * n ** 3)
        rec["dsvd_1024"] = rate(lambda: np.linalg.svd(S), 22.0 * m ** 3)
        rec["zsvd_1024"] = rate(lambda: np.linalg.svd(Z), 88.0 * m ** 3)
    else:
        import torch
        dev, sync = torch.device("cuda"), torch.cuda.synchronize
        A, B = (torch.rand(n, n, dtype=torch.float64, device=dev) for _ in range(2))
        S = torch.rand(m, m, dtype=torch.float64, device=dev)
        Z = torch.rand(m, m, dtype=torch.complex128, device=dev)
        rec.update(torch=torch.__version__, cuda=torch.version.cuda,
                   device=torch.cuda.get_device_name(0))
        rec["dgemm_4096"] = rate(lambda: A @ B, 2.0 * n ** 3, sync)
        rec["dsvd_1024"] = rate(lambda: torch.linalg.svd(S), 22.0 * m ** 3, sync)
        rec["zsvd_1024"] = rate(lambda: torch.linalg.svd(Z), 88.0 * m ** 3, sync)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("calib", json.dumps(rec))


if __name__ == "__main__":
    main()
