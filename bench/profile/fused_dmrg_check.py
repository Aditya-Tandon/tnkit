"""Dense two-site DMRG on one GPU, run against whichever tnkit is importable: the energy check for
the fused, transpose-free two-site matvec (PR #2, merged as b933cd4) against the unfused one it
replaced (main at ca14103, before PR #2).

usage: python fused_dmrg_check.py LABEL OUT.json
Run it once per tnkit version, with PYTHONPATH pointing at that version's checkout and the same
script both times. Cases (identical for both versions: same MPO, seed, chi, sweeps, eig_tol):
  cyl_Ly6  Heisenberg on a 6 x 8 cylinder (N 48, column-major snake, MPO width 20)
  xxz_N64  Heisenberg chain, N 64 (MPO width 5)
each at chi 128 and 256, backend torch-cuda-real (fp64), n_sweeps 8 with tol = rtol = 0 so every
sweep runs and the two versions see the same schedule. The MPO bulks are the ones in
gpu_matvec_prof.py; the ends follow tnkit's convention (row 0 on the first site, the last column on
the last site). Records per case: the per-sweep energies, the per-sweep wall times, the final energy
per site, the number of matvec calls and the discarded weight, plus which matvec the library
used (apply_two_site_fused present or not). Standalone: numpy, torch and tnkit only (no pytest).
"""
import json, sys, time

import numpy as np
import torch

import tnkit
from tnkit import _dmrg

LX, LY, N_CHAIN = 8, 6, 64
CHIS = (128, 256)
N_SWEEPS = 8


def cyl_bulk(Ly):
    SP = np.array([[0, 1], [0, 0]], float); SM = SP.T.copy(); SZ = np.diag([0.5, -0.5]); I2 = np.eye(2)
    ops = [(SP, 0.5 * SM), (SM, 0.5 * SP), (SZ, SZ)]
    D = 2 + 3 * Ly
    ch = lambda o, k: 1 + o * Ly + (k - 1)
    W = np.zeros((D, D, 2, 2))
    W[0, 0] = W[-1, -1] = I2
    for o, (a, b) in enumerate(ops):
        W[0, ch(o, 1)] = a
        for k in range(1, Ly):
            W[ch(o, k), ch(o, k + 1)] = I2
        W[ch(o, 1), D - 1] = b
        W[ch(o, Ly), D - 1] = b
        if Ly > 1:
            W[ch(o, Ly - 1), D - 1] = b
    return W


def xxz_bulk():
    SP = np.array([[0, 1], [0, 0]], float); SM = SP.T.copy(); SZ = np.diag([0.5, -0.5]); I2 = np.eye(2)
    W = np.zeros((5, 5, 2, 2))
    W[0, 0], W[0, 1], W[0, 2], W[0, 3] = I2, SP, SM, SZ
    W[1, 4], W[2, 4], W[3, 4], W[4, 4] = 0.5 * SM, 0.5 * SP, SZ, I2
    return W


def open_mpo(Wb, N):
    """Open-boundary MPO from a bulk tensor: row 0 on site 0, the last column on site N-1."""
    D = Wb.shape[0]
    return [Wb[0:1] if i == 0 else Wb[:, D - 1:D] if i == N - 1 else Wb for i in range(N)]


def main():
    label, out = sys.argv[1], sys.argv[2]
    fused = hasattr(_dmrg, "apply_two_site_fused")
    rec = dict(label=label, tnkit_file=tnkit.__file__, fused_matvec_in_library=fused,
               torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0),
               n_sweeps=N_SWEEPS, cases=[])
    cases = [("cyl_Ly6", open_mpo(cyl_bulk(LY), LX * LY), LX * LY),
             ("xxz_N64", open_mpo(xxz_bulk(), N_CHAIN), N_CHAIN)]
    for name, mpo, N in cases:
        for chi in CHIS:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            E, mps, info = _dmrg.dmrg(mpo, chi=chi, n_sweeps=N_SWEEPS, tol=0.0, rtol=0.0, seed=0,
                                      backend="torch-cuda-real")
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            row = dict(case=name, N=N, w=int(mpo[1].shape[0]), chi=chi, E=float(E), E_per_site=float(E) / N,
                       energies=[float(e) for e in info["energies"]], sweep_s=[float(s) for s in info["sweep_s"]],
                       sweeps=int(info["sweeps"]), matvec_calls=int(info["matvec_calls"]),
                       discarded_weight=float(info["discarded_weight"]), wall_s=wall, backend=info["backend"])
            rec["cases"].append(row)
            print(json.dumps({k: row[k] for k in ("case", "chi", "E", "sweeps", "wall_s")}), flush=True)
    json.dump(rec, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
