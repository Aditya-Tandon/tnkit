"""tnkit: matrix-product states and operators, with DMRG and TDVP, on numpy and torch backends.

The curated public names are re-exported here from `tnkit.api`. `dmrg`, `vumps` and `idmrg`
are the solver functions; the modules of the same name remain importable as shims.
"""
from __future__ import annotations

import os as _os
import re as _re

# Load the shim modules before binding the functions of the same names below, so that a
# later `import tnkit.dmrg` finds the submodule loaded and does not rebind the attribute.
from . import _dmrg, _vumps  # noqa: F401
from . import dmrg as _dmrg_shim, idmrg as _idmrg_shim, vumps as _vumps_shim  # noqa: F401
from . import eigensolver, linalg, models, backend, observables, free_fermion  # noqa: F401
from . import tebd, tdvp, batch_tdvp  # noqa: F401
from . import api, mps  # noqa: F401


def _version_from_pyproject() -> str:
    """The `version = ...` line of the checkout's `pyproject.toml`, or "" if unreadable."""
    try:
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        with open(_os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
            for line in fh:
                m = _re.match(r"""\s*version\s*=\s*["']([^"']+)["']""", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def _detect_version() -> str:
    """The checkout's `pyproject.toml` first, then installed metadata, then "0+unknown".

    The checkout wins so that a stale installed copy cannot mislabel the code that runs.
    """
    v = _version_from_pyproject()
    if v:
        return v
    try:
        from importlib.metadata import version as _version
        return _version("tnkit")
    except Exception:
        pass
    return "0+unknown"


__version__ = _detect_version()

from .api import (  # noqa: E402,F401
    dmrg, idmrg, vumps,
    DMRGOptions, TDVPOptions, VUMPSOptions, UniformMPS, IDMRGOptions,
    DMRGExcitedOptions, MPS, as_mps,
    dmrg_excited, tebd_evolve, tdvp1_evolve, tdvp2_evolve,
    evolve_batch, BatchTDVPOptions, BatchGeometryError, BatchKrylovError,
    tfim_mpo, tfim_ring_mpo, magnetization_mpo, tfim_ground_energy, schwinger_mpo,
    SX, SY, SZ,
    all_entropies, entanglement_entropy, local_expectations, neighbor_correlations,
    two_point, structure_factor, magnetization_moments, mps_overlap, apply_mpo,
    to_numpy, two_site_entropy, compress, mpo_expectation, mpo_variance, two_site_variance,
    random_mps, right_canonicalize, get_backend,
    Symmetry, U1, Zn, ProductSymmetry,
    to_block_mpo, to_block_mps, project_to_block_mps, random_block_mps, to_dense_mps,
    BlockBackend,
)

# `import tnkit.dmrg as D` binds the function, so the function carries the implementation's
# mutable registries (the same dict objects, not copies).
dmrg._EIGSOLVERS = _dmrg._EIGSOLVERS
vumps._HOOKS = _vumps._HOOKS

__all__ = [
    "__version__",
    "dmrg", "idmrg", "vumps",
    "api", "backend", "batch_tdvp", "eigensolver", "free_fermion", "linalg",
    "models", "mps", "observables", "tdvp", "tebd",
    "DMRGOptions", "TDVPOptions", "VUMPSOptions", "UniformMPS", "IDMRGOptions",
    "DMRGExcitedOptions",
    "BatchTDVPOptions",
    "MPS", "as_mps",
    "BatchGeometryError", "BatchKrylovError",
    "dmrg_excited", "tebd_evolve", "tdvp1_evolve", "tdvp2_evolve", "evolve_batch",
    "tfim_mpo", "tfim_ring_mpo", "magnetization_mpo", "tfim_ground_energy",
    "schwinger_mpo", "SX", "SY", "SZ",
    "all_entropies", "entanglement_entropy", "local_expectations",
    "neighbor_correlations", "two_point", "structure_factor",
    "magnetization_moments", "mps_overlap", "apply_mpo", "to_numpy",
    "two_site_entropy", "compress", "mpo_expectation", "mpo_variance", "two_site_variance",
    "random_mps", "right_canonicalize", "get_backend",
    "Symmetry", "U1", "Zn", "ProductSymmetry",
    "to_block_mpo", "to_block_mps", "project_to_block_mps", "random_block_mps",
    "to_dense_mps", "BlockBackend",
]
