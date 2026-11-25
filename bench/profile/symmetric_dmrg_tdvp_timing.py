"""Two-site DMRG sweeps and two-site TDVP steps on the two-flavour Schwinger chain (merged d = 4
sites): U(1) x SU(2) symmetric tensors against U(1) x U(1) block tensors and dense tensors at the
same dense bond. Standalone: needs numpy and scipy, and imports tnkit from this checkout.

usage: python symmetric_dmrg_tdvp_timing.py N CHI1,CHI2,... [OUT.json]

For each CHI (multiplets), U(1) x SU(2) DMRG (5 sweeps, no expansion, no early stop) finds the
ground state at x = 0.6, mu = 0.3; its largest dense bond D sets the cap of the U(1) x U(1) and
dense runs. Printed: energies, the median wall time of sweeps 2-5, and the median of steps 2-4 of
two-site TDVP after a quench to mu = -0.4 (dt = 0.05; the time inside the Krylov exponentials,
truncations and environment updates), from the U(1) x SU(2) ground state (the block run from
its own).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tnkit.backend import get_backend
from tnkit.block_backend import BlockBackend, to_block_mpo
from tnkit.dmrg import dmrg
from tnkit.models_lgt2f import merge_pairs, schwinger2f_mpo, site_spaces_2f
from tnkit.symmetric_backend import SymmetricBackend, symmetric_mpo, to_dense_mps
from tnkit.symmetry import U1, ProductSymmetry
from tnkit.tdvp import tdvp2_evolve

N, chis = int(sys.argv[1]), [int(c) for c in sys.argv[2].split(",")]
out_path = sys.argv[3] if len(sys.argv) > 3 else None
x, mu0, mu1 = 0.6, 0.3, -0.4
sites = site_spaces_2f(N)
W0, W1 = merge_pairs(schwinger2f_mpo(N, x, mu0)), merge_pairs(schwinger2f_mpo(N, x, mu1))
r0, r1 = symmetric_mpo(W0, sites)[0], symmetric_mpo(W1, sites)[0]
u1 = ProductSymmetry((U1(), U1()))
labels = [[(q, 0), (q + 1, -1), (q + 1, 1), (q + 2, 0)] for q in [-2 * (n % 2) for n in range(N)]]
real, cplx = get_backend("numpy-real"), get_backend("numpy")
b0 = to_block_mpo(W0, labels, real, sym=u1)[0]
b1 = to_block_mpo(W1, labels, cplx, sym=u1)[0]
SW = 5
kw = dict(n_sweeps=SW, tol=0.0, rtol=0.0, eig_tol=1e-10, expand=0)
out = []


def med(xs):
    return float(np.median(xs))


for chi_m in chis:
    row = {"chi_multiplets": chi_m}
    E, psi, info = dmrg(r0, chi_m, charge=(0, 0), backend="numpy-real", **kw)
    dense_chi = max(m.codomain[2].dim for m in psi)
    row.update(E_su2=E, dense_bond=dense_chi, su2_sweep_s=med(info["sweep_s"][1:]))
    E, psi_b, info = dmrg(b0, dense_chi, charge=(0, 0), backend="numpy-real", **kw)
    row.update(E_u1u1=E, u1u1_sweep_s=med(info["sweep_s"][1:]))
    E, _, info = dmrg(W0, dense_chi, backend="numpy-real", **kw)
    row.update(E_dense=E, dense_sweep_s=med(info["sweep_s"][1:]))
    dmps = [np.asarray(t) for t in to_dense_mps(psi, sites)]
    prof = {}
    tdvp2_evolve(list(psi), r1, chi_m, 0.05, 4, backend=SymmetricBackend(cplx), profile=prof)
    st = [s["expm_s"] + s["trunc_s"] + s["env_s"] for s in prof["steps"]]
    row["su2_step_s"] = med(st[1:])
    prof = {}
    tdvp2_evolve([d.astype(complex) for d in dmps], [w.astype(complex) for w in W1], dense_chi,
                 0.05, 4, backend="numpy", profile=prof)
    row["dense_step_s"] = med([s["expm_s"] + s["trunc_s"] + s["env_s"] for s in prof["steps"]][1:])
    prof = {}
    tdvp2_evolve(list(psi_b), b1, dense_chi, 0.05, 4, backend=BlockBackend(cplx), profile=prof)
    row["u1u1_step_s"] = med([s["expm_s"] + s["trunc_s"] + s["env_s"] for s in prof["steps"]][1:])
    print(json.dumps(row), flush=True)
    out.append(row)
if out_path:
    json.dump(out, open(out_path, "w"), indent=1)
