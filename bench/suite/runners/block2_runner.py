"""block2 (pyblock2, C++ with OpenMP; CPU only): U(1) (`SGB`, conserving 2 S^z) or SU(2) DMRG.

Settings follow block2's Heisenberg tutorial: two-site DMRG from a random MPS, noise 1e-5 for
four sweeps, 1e-6 for two, then 0, Davidson threshold 1e-10. The settings file (w["opts"]) may
set `noises`, `thrds`, `dav_max_iter`, `cutoff`, `stack_mem_gb`, and `rtol` with `e_ref`: block2's
energy stop is absolute only, so a relative stop rtol is applied as the absolute stop
rtol |e_ref|, e_ref being the workload's reference energy. For SU(2) the bond dimension counts
multiplets, each standing for 2S+1 states. block2 reports no per-sweep times, so a run
records its whole wall time and its sweep count only.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from common import Clock, Unsupported, bonds, couplings, options, schedule_of, stop_tol

NAME = "block2"
OPTS = ("noises", "thrds", "dav_max_iter", "cutoff", "stack_mem_gb", "rtol", "e_ref")
DEFAULTS = {"noises": [1e-5] * 4 + [1e-6] * 2 + [0.0], "thrds": [1e-10]}


def version():
    from importlib import metadata
    return metadata.version("block2")


def check(w, sym):
    if sym not in ("u1", "su2"):
        raise Unsupported("block2 runs U(1) or SU(2) here")
    if w["kind"] != "dmrg" or w.get("model") not in ("xxz", "heis_cyl"):
        raise Unsupported("block2 runs Heisenberg ground states only in this suite")
    J, Jz = couplings(w)
    if sym == "su2" and J != Jz:
        raise Unsupported("SU(2) needs J = Jz")


def _per_sweep(values, n):
    """A block2 schedule list (the last entry repeats) written out for n sweeps."""
    return [values[min(k, len(values) - 1)] for k in range(n)]


def stop_of(w, opts):
    """The absolute energy stop of the run: none for fixed sweeps; else the workload's, or
    rtol |e_ref| if the settings give a relative stop (the larger of the two)."""
    if w.get("fixed_sweeps"):
        return 0.0
    if not opts.get("rtol"):
        return stop_tol(w)
    if "e_ref" not in opts:
        raise ValueError("a relative stop (rtol) needs the reference energy e_ref")
    return max(stop_tol(w), opts["rtol"] * abs(opts["e_ref"]))


def dmrg(w, chi, backend, sym):
    from pyblock2.driver.core import DMRGDriver, MPOAlgorithmTypes, SymmetryTypes
    opts = {**DEFAULTS, **options(w, OPTS)}
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    scratch = tempfile.mkdtemp(prefix="block2_")
    su2 = sym == "su2"
    try:
        d = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SU2 if su2 else SymmetryTypes.SGB,
                       n_threads=threads, stack_mem=int(opts.get("stack_mem_gb", 4) * 2 ** 30))
        if su2:
            d.initialize_system(n_sites=w["N"], heis_twos=1, heis_twosz=0, spin=0)
        else:
            d.initialize_system(n_sites=w["N"], heis_twos=1, heis_twosz=0)
        J, Jz = couplings(w)
        b = d.expr_builder()
        for i, j in bonds(w):
            if su2:                              # S_i . S_j = -sqrt(3)/2 (T_i T_j)^[0]
                b.add_term("(T+T)0", [i, j], -np.sqrt(3) / 2 * J)
            else:
                b.add_term("PM", [i, j], 0.5 * J)
                b.add_term("MP", [i, j], 0.5 * J)
                b.add_term("ZZ", [i, j], Jz)
        mpo = d.get_mpo(b.finalize(adjust_order=not su2),
                        algo_type=MPOAlgorithmTypes.FastBipartite, iprint=0)
        stages = schedule_of(w, chi)
        n = sum(k for _, k in stages)
        bond_dims = [bd for bd, k in stages for _ in range(k)]
        tol = stop_of(w, opts)
        kw = {k: opts[k] for k in ("dav_max_iter", "cutoff") if k in opts}
        with Clock(backend) as c:
            mps = d.get_random_mps(tag="KET", bond_dim=stages[0][0], nroots=1)
            E = d.dmrg(mpo, mps, n_sweeps=n, tol=tol, bond_dims=bond_dims,
                       noises=_per_sweep(opts["noises"], n), thrds=_per_sweep(opts["thrds"], n),
                       iprint=0, **kw)
        bds, dws, es = d.get_dmrg_results()
        energies = [float(np.ravel(e)[0]) for e in es]
        d.finalize()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"wall_s": c.s, "energy": float(E), "energies": energies, "sweeps": len(energies),
            "sweep_s": None, "discarded_weights": [float(x) for x in dws],
            "bond_dims": [int(x) for x in bds], "max_bond": int(max(bds)), "dtype": "float64",
            "chi_unit": "multiplets" if su2 else "states",
            "settings": {"symmetry": "SU2" if su2 else "SGB (U(1), 2 S^z = 0)",
                         "bond_dims": bond_dims, "tol": tol, "rtol": opts.get("rtol", 0.0),
                         "e_ref": opts.get("e_ref"),
                         "stop": "fixed sweeps, no energy stop" if w.get("fixed_sweeps") else
                         "|dE| < tol (absolute; tol = max(workload stop, rtol |e_ref|))",
                         "noises": opts["noises"],
                         "thrds": opts["thrds"], "n_threads": threads,
                         "stack_mem_gb": opts.get("stack_mem_gb", 4),
                         "mpo": "expr_builder, FastBipartite", "init": "random MPS",
                         **kw}}


def warmup(kind, backend, sym):
    dmrg({"kind": "dmrg", "model": "xxz", "N": 8, "J": 1.0, "Jz": 1.0, "max_sweeps": 2,
          "e_tol": 1e-8}, 8, backend=backend, sym=sym)
