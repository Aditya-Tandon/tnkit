"""Quantum TEA (qtealeaves + qredtea), dense MPS, numpy (CPU) or torch (CPU or CUDA).

Runs in its own environment (qredtea caps numpy below 2.1.4). The Hamiltonian is an iTPO built
from a qtealeaves model, as `QuantumGreenTeaSimulation` builds it; the solver calls are the
library's own (`optimize` for the ground state, `MPS.timestep` for dynamics), driven directly so
that no simulation folder or file I/O sits inside the timed region. Settings are the library
defaults except bond dimension, tolerances and sweeps. DMRG runs real (data type 'D'); TDVP runs
complex128 with mode 3 (two-tensor TDVP, second order). The settings file (w["opts"]) adds
`TNConvergenceParameters` keywords. The timed region holds the initial state and the effective
operators (environments) of the solver, not the iTPO, as in every runner. DMRG records qtealeaves'
warnings by message (e.g. its Lanczos stopping at max_iter), counted per run.
"""
from __future__ import annotations

import logging
import tempfile
from collections import Counter

import numpy as np

from common import Clock, Unsupported, cuda_backend, laps, neel

NAME = "qtea"


def version():
    from importlib import metadata
    return f"qtealeaves {metadata.version('qtealeaves')}, qredtea {metadata.version('qredtea')}"


def check(w, sym):
    if sym != "dense":
        raise Unsupported("symmetric Quantum TEA tensors are not set up in this suite")
    if w["kind"] == "collision":
        raise Unsupported("the two-flavour collision is not ported to Quantum TEA")
    if w["kind"] == "excited" or w.get("model") == "heis_cyl" or w.get("ramp"):
        raise Unsupported("excited states and the cylinder are not ported to Quantum TEA")


def _device(backend):
    return "gpu" if cuda_backend(backend) else "cpu"


def _tensor_backend(backend, real):
    from qtealeaves.tensors.tensor_backend import TensorBackend
    if backend == "numpy":
        from qtealeaves.tensors.tensor import DataMoverNumpyCupy, QteaTensor
        return TensorBackend(tensor_cls=QteaTensor, base_tensor_cls=QteaTensor, device="cpu",
                             dtype=np.float64 if real else np.complex128,
                             datamover=DataMoverNumpyCupy())
    import torch
    from qredtea.torchapi.qteatorchtensor import DataMoverPytorch, QteaTorchTensor
    return TensorBackend(tensor_cls=QteaTorchTensor, base_tensor_cls=QteaTorchTensor,
                         device=_device(backend), dtype=torch.float64 if real else torch.complex128,
                         datamover=DataMoverPytorch())


def _model(w):
    from qtealeaves import modeling
    if w["model"] == "tfim":                      # H = -J sum X X - g sum Z
        from qtealeaves.models import get_quantum_ising_1d
        model, ops = get_quantum_ising_1d()
        return model, ops, {"L": w["N"], "J": 1.0, "g": w["g"]}
    from qtealeaves.operators import TNSpin12Operators
    ops = TNSpin12Operators()
    model = modeling.QuantumModel(1, "L", name="XXZ")
    model += modeling.TwoBodyTerm1D(["splus", "sminus"], shift=1, strength="J", prefactor=0.5)
    model += modeling.TwoBodyTerm1D(["sminus", "splus"], shift=1, strength="J", prefactor=0.5)
    if w.get("Jz", 0.0):                          # Pauli sz is twice S^z
        model += modeling.TwoBodyTerm1D(["sz", "sz"], shift=1, strength="Jz", prefactor=0.25)
    return model, ops, {"L": w["N"], "J": w["J"], "Jz": w.get("Jz", 0.0)}


def _itpo(w, tb):
    from qtealeaves.mpos import ITPO, DenseMPOList
    model, ops, params = _model(w)
    tb.set_seed([11, 13, 17, 19])                 # the qtealeaves default seed form
    ops = tb.base_tensor_cls.convert_operator_dict(
        ops, params=params, symmetries=[], generators=[], base_tensor_cls=tb.base_tensor_cls,
        dtype=tb.dtype, device=tb.memory_device)
    dense = DenseMPOList.from_model(model, params, tensor_backend=tb)
    dense.initialize(ops, params)
    mpo = ITPO(w["N"])
    mpo.add_dense_mpo_list(dense)
    return mpo, params


class _Warnings(logging.Handler):
    """Counts qtealeaves' warnings by message template."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.counts = Counter()

    def emit(self, record):
        self.counts[str(record.msg)] += 1


def dmrg(w, chi, backend, sym):
    from qtealeaves.convergence_parameters import TNConvergenceParameters
    from qtealeaves.emulator import MPS
    from qtealeaves.simulation.tn_simulation import optimize
    tb = _tensor_backend(backend, real=True)
    n, tol = (w["fixed_sweeps"], 0.0) if w.get("fixed_sweeps") else (w["max_sweeps"], w["e_tol"])
    conv = TNConvergenceParameters(**{"max_iter": n, "max_bond_dimension": chi,
                                      "abs_deviation": tol, "rel_deviation": tol,
                                      "data_type": "D", "device": _device(backend),
                                      **w.get("opts", {})})
    mpo, params = _itpo(w, tb)
    counter, log = _Warnings(), logging.getLogger("qtealeaves")
    log.addHandler(counter)
    try:
        with tempfile.TemporaryDirectory() as folder, Clock(backend) as c:
            state = MPS(w["N"], conv, local_dim=2, initialize="random", tensor_backend=tb)
            mpo.setup_as_eff_ops(state)
            state, et, _ = optimize(state, params, folder, False, 0, lambda *a, **k: None)
    finally:
        log.removeHandler(counter)
    et = np.asarray(et)
    return {"wall_s": c.s, "energy": float(et[-1, 1]), "energies": et[:, 1].tolist(),
            "sweep_s": et[:, 0].tolist(), "sweeps": int(et.shape[0]),
            "max_bond": int(state.current_max_bond_dim), "dtype": "float64",
            "tensor_dtype": str(state[0].dtype), "tensor_device": str(state[0].device),
            "warnings": dict(counter.counts),
            "settings": {"max_iter": n, "max_bond_dimension": chi, "opts": w.get("opts", {}),
                         "abs_deviation": tol, "rel_deviation": tol,
                         "stop": "fixed sweeps, no energy stop" if w.get("fixed_sweeps") else
                         "abs_deviation and rel_deviation (qtealeaves' own rule)",
                         "statics_method": conv.sim_params["statics_method"],
                         "arnoldi_maxiter": conv.arnoldi_maxiter,
                         "cut_ratio": "1e-9 (default)", "data_type": "D", "mpo": "iTPO",
                         "init": "random (default)"}}


def tdvp(w, chi, backend, sym):
    from qtealeaves.convergence_parameters import TNConvergenceParameters
    from qtealeaves.emulator import MPS
    from qtealeaves.solvers import KrylovSolverH
    tb = _tensor_backend(backend, real=False)
    conv = TNConvergenceParameters(**{"max_bond_dimension": chi, "cut_ratio": w["sv_tol"],
                                      "data_type": "Z", "device": _device(backend),
                                      **w.get("opts", {})})
    mpo, _ = _itpo(w, tb)
    up, down = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    mat = np.array([up if s == 0 else down for s in neel(w["N"])])
    state = MPS.product_state_from_local_states(mat, convergence_parameters=conv,
                                                tensor_backend=tb)
    state.solver = KrylovSolverH                  # Hermitian H, as the simulation sets it
    steps = int(round(w["T"] / w["dt"]))
    with Clock(backend) as c:
        mpo.setup_as_eff_ops(state)
        step_s = [dt for _, dt in laps((state.timestep(w["dt"], 3) for _ in range(steps)),
                                       backend)]
    sz = 0.5 * _pauli_z(state)
    return {"wall_s": c.s, "step_s": step_s, "steps": steps, "sz": [float(x) for x in sz],
            "bonds": [int(x) for x in _bonds(state)], "dtype": "complex128",
            "settings": {"mode": "3 (two-tensor TDVP, second order)", "max_bond_dimension": chi,
                         "opts": w.get("opts", {}),
                         "cut_ratio": w["sv_tol"], "krylov_maxiter": "32 (default)",
                         "krylov_tol": "1e-7 (default)", "mpo": "iTPO"}}


def _pauli_z(state):
    t = state[0]
    op = type(t).from_elem_array(np.diag([1.0, -1.0]).astype(complex), dtype=t.dtype,
                                 device=t.device)
    return np.asarray([float(np.real(x)) for x in state.meas_local(op)])


def _bonds(state):
    return [int(state[i].shape[2]) for i in range(state.num_sites - 1)]


def warmup(kind, backend, sym):
    tdvp({"model": "xx", "N": 8, "J": 1.0, "Jz": 0.0, "dt": 0.05, "T": 0.1, "sv_tol": 1e-6}, 4,
         backend, sym)
    if kind == "dmrg":
        dmrg({"model": "xxz", "N": 8, "J": 1.0, "Jz": 1.0, "max_sweeps": 2, "e_tol": 1e-8}, 4,
             backend=backend, sym=sym)
