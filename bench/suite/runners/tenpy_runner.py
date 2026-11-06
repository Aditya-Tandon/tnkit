"""TeNPy (physics-tenpy, CPU only): dense (conserve=None) or U(1) (conserve='Sz').

Settings follow the TeNPy documentation: two-site DMRG from a product state with the mixer on,
`max_E_err` as the stop rule; two-site TDVP throughout. TeNPy has no per-bond one-site/two-site
schedule; a global switch to one-site TDVP once the largest bond reaches the cap tripled the
error of the dense quench at cap 32 (0.34 against 0.11), so the suite does not use it.
Truncation uses `trunc_cut`, a discarded-weight cut with the same meaning as tnkit's `svd_tol`.
The settings file (w["opts"]) is merged into the engine's parameters. A ramped workload uses
TeNPy's own `chi_list` schedule; excited states use `orthogonal_to`.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np

from common import Clock, Unsupported, bonds, couplings, laps, neel, schedule_of, stop_tol

NAME = "tenpy"
STOP = "max_E_err, relative: |dE| < max_E_err max(|E|, 1)"
warnings.filterwarnings("ignore", message="unit_cell_width")


def version():
    import tenpy
    return tenpy.__version__


def check(w, sym):
    if sym == "su2":
        raise Unsupported("TeNPy has no SU(2) tensors")
    if w["kind"] == "collision":
        raise Unsupported("the two-flavour collision is not ported to TeNPy")
    if w.get("model") == "tfim" and sym == "u1":
        raise Unsupported("the Ising chain has no U(1) symmetry")


def _model(w, sym):
    if w["model"] in ("xxz", "xx"):
        from tenpy.models.xxz_chain import XXZChain
        return XXZChain({"L": w["N"], "Jxx": w["J"], "Jz": w.get("Jz", 0.0), "hz": 0.0,
                         "bc_MPS": "finite", "conserve": "Sz" if sym == "u1" else None})
    if w["model"] == "tfim":
        from tenpy.models.tf_ising import TFIChain
        return TFIChain({"L": w["N"], "J": 1.0, "g": w["g"], "bc_MPS": "finite",
                         "conserve": None})
    if w["model"] == "heis_cyl":
        return _pair_model(w, sym)
    raise ValueError(w["model"])


def _pair_model(w, sym):
    """Spin 1/2 on a chain of N sites with the workload's couplings between any pairs (the
    cylinder on its MPS path)."""
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite
    J, Jz = couplings(w)

    class PairModel(CouplingMPOModel):
        def init_lattice(self, p):
            site = SpinHalfSite(conserve="Sz" if sym == "u1" else None)
            return Chain(p["L"], site, bc="open", bc_MPS="finite")

        def init_terms(self, p):
            for i, j in bonds(w):
                self.add_coupling_term(0.5 * J, i, j, "Sp", "Sm")
                self.add_coupling_term(0.5 * J, i, j, "Sm", "Sp")
                self.add_coupling_term(Jz, i, j, "Sz", "Sz")

    return PairModel({"L": w["N"]})


def _product(M, w, dtype):
    from tenpy.networks.mps import MPS
    states = ["up" if s == 0 else "down" for s in neel(w["N"])] if w["model"] != "tfim" \
        else ["up"] * w["N"]
    psi = MPS.from_product_state(M.lat.mps_sites(), states, bc="finite", dtype=dtype)
    return psi


def dmrg(w, chi, backend, sym):
    from tenpy.algorithms import dmrg as tdmrg
    M = _model(w, sym)
    stages = schedule_of(w, chi)
    starts = np.cumsum([0] + [n for _, n in stages[:-1]])
    params = {"mixer": True, "max_E_err": stop_tol(w) / w.get("energy_scale", 1.0),
              "max_sweeps": int(sum(n for _, n in stages)),
              "trunc_params": {"chi_max": chi, "svd_min": w.get("sv_tol", 1e-14)},
              "combine": True, "max_trunc_err": 1.0}
    if len(stages) > 1:
        params["chi_list"] = {int(k): b for k, (b, _) in zip(starts, stages)}
        params["min_sweeps"] = int(starts[-1]) + 1
    if w.get("fixed_sweeps"):
        params.update(min_sweeps=w["fixed_sweeps"], max_E_err=0.0)
    params.update(copy.deepcopy(w.get("opts", {})))
    settings = copy.deepcopy(params)
    with Clock(backend) as c:
        psi = _product(M, w, np.float64)
        eng = tdmrg.TwoSiteDMRGEngine(psi, M, params)
        E, psi = eng.run()
    stats = eng.sweep_stats
    return {"wall_s": c.s, "energy": float(E), "energies": [float(e) for e in stats["E"]],
            "sweeps": int(stats["sweep"][-1]), "sweep_s": list(np.diff([0.0] + stats["time"])),
            "max_bond": int(max(psi.chi)), "dtype": "float64",
            "settings": {**settings, "engine": "TwoSiteDMRGEngine", "init": "product state "
                         "(Neel; all up for Ising)",
                         "stop": "fixed sweeps, no energy stop" if w.get("fixed_sweeps") else STOP}}


def excited(w, chi, backend, sym):
    """The lowest `nstates` levels, each by DMRG orthogonal to the states found before."""
    from tenpy.algorithms import dmrg as tdmrg
    M = _model(w, sym)
    states, energies, sweeps = [], [], []
    base = {"mixer": True, "max_E_err": w["e_tol"], "max_sweeps": w["max_sweeps"],
            "trunc_params": {"chi_max": chi, "svd_min": w.get("sv_tol", 1e-14)},
            "combine": True, "max_trunc_err": 1.0, **copy.deepcopy(w.get("opts", {}))}
    with Clock(backend) as c:
        for k in range(w["nstates"]):
            psi = _product(M, w, np.float64)
            if k:                                 # a start not orthogonal to the states before
                psi.perturb({"N_steps": 4, "trunc_params": {"chi_max": 8}}, close_1=False)
            eng = tdmrg.TwoSiteDMRGEngine(psi, M, copy.deepcopy(base), orthogonal_to=list(states))
            E, psi = eng.run()
            states.append(psi)
            energies.append(float(E))
            sweeps.append(int(eng.sweep_stats["sweep"][-1]))
    return {"wall_s": c.s, "energies_k": energies, "sweeps_k": sweeps, "dtype": "float64",
            "settings": {**{k: v for k, v in base.items()}, "engine": "TwoSiteDMRGEngine",
                         "stop": STOP + " per level: looser than an absolute e_tol by max(|E|, 1)",
                         "excited": "orthogonal_to the states before; start: all up, "
                         "perturbed by 4 random two-site unitary layers (chi 8)"}}


def tdvp(w, chi, backend, sym):
    from tenpy.algorithms import tdvp as ttdvp
    M = _model(w, sym)
    psi = _product(M, w, np.complex128)
    steps = int(round(w["T"] / w["dt"]))
    params = {"dt": w["dt"], "N_steps": 1,
              "trunc_params": {"chi_max": chi, "trunc_cut": w["svd_tol"]}, "max_trunc_err": 1.0,
              **copy.deepcopy(w.get("opts", {}))}
    settings = copy.deepcopy(params)
    with Clock(backend) as c:
        eng = ttdvp.TwoSiteTDVPEngine(psi, M, params)
        step_s = [dt for _, dt in laps((eng.run() for _ in range(steps)), backend)]
    sz = psi.expectation_value("Sz")
    return {"wall_s": c.s, "step_s": step_s, "steps": steps, "sz": [float(x) for x in sz],
            "bonds": [int(x) for x in psi.chi], "dtype": "complex128",
            "settings": {**settings, "engine": "TwoSiteTDVPEngine throughout"}}


def warmup(kind, backend, sym):
    if kind == "excited":
        excited({"model": "tfim", "N": 8, "g": 1.0, "nstates": 2, "max_sweeps": 2,
                 "e_tol": 1e-8}, 4, backend, sym)
        return
    w = {"model": "xx", "N": 8, "J": 1.0, "Jz": 0.0, "dt": 0.05, "T": 0.1, "svd_tol": 1e-10}
    tdvp(w, 4, backend, sym)
    if kind == "dmrg":
        dmrg({"model": "xxz", "N": 8, "J": 1.0, "Jz": 1.0, "max_sweeps": 2, "e_tol": 1e-8}, 4,
             backend=backend, sym=sym)
