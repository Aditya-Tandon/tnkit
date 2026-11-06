"""The curated tnkit surface, in one flat namespace.

Everything a user needs to solve a model and measure it::

    from tnkit.api import dmrg, tfim_mpo, all_entropies, structure_factor

Every name here is re-exported from `tnkit` itself. That was not always true:
`dmrg`, `vumps` and `idmrg` each named a module as well as a function, and the package
attribute could only be one of them. Their implementations now live in `tnkit._dmrg`,
`tnkit._vumps` and `tnkit._idmrg`, the old module paths are compatibility shims, and the
functions own the top-level names -- so `from tnkit import dmrg` gives the solver and
`from tnkit.dmrg import dmrg` keeps working.

This module stays. It is the import path that says "give me the flat surface" without
also handing over the submodules, and every existing `from tnkit.api import ...` keeps
working unchanged.
"""
from __future__ import annotations

# Re-exports: every import here is the module's product, hence the blanket noqa.
from .backend import get_backend  # noqa: F401
from .block_backend import (BlockBackend, project_to_block_mps,  # noqa: F401
                            random_block_mps, to_block_mpo, to_block_mps, to_dense_mps)
from ._dmrg import DMRGOptions, dmrg, random_mps, right_canonicalize  # noqa: F401
from .dmrg_excited import (DMRGExcitedOptions, dmrg_excited,  # noqa: F401
                           mpo_expectation, mpo_variance,
                           two_site_variance)
from ._idmrg import IDMRGOptions, idmrg  # noqa: F401
from .models import (SX, SY, SZ, magnetization_mpo, tfim_ground_energy,  # noqa: F401
                     tfim_mpo, tfim_ring_mpo)
from .models_lgt import schwinger_mpo  # noqa: F401
from .mps import MPS, as_mps  # noqa: F401
from .symmetry import ProductSymmetry, Symmetry, U1, Zn  # noqa: F401
from .observables import (all_entropies, apply_mpo, compress,  # noqa: F401
                          entanglement_entropy, local_expectations, magnetization_moments,
                          mps_overlap, neighbor_correlations, structure_factor, to_numpy,
                          two_point, two_site_entropy)
from .batch_tdvp import (BatchGeometryError, BatchKrylovError,  # noqa: F401
                         BatchTDVPOptions, evolve_batch)
from .tdvp import TDVPOptions, tdvp1_evolve, tdvp2_evolve  # noqa: F401
from .tebd import tebd_evolve  # noqa: F401
from ._vumps import UniformMPS, VUMPSOptions, vumps  # noqa: F401

#: The names `tnkit.__init__` re-exports, minus the module-name collisions.
CURATED = [
    # solvers
    "dmrg", "dmrg_excited", "idmrg", "vumps", "tebd_evolve",
    "tdvp1_evolve", "tdvp2_evolve", "evolve_batch",
    # types and settings
    "MPS", "as_mps", "DMRGOptions", "TDVPOptions", "VUMPSOptions", "UniformMPS",
    "IDMRGOptions", "DMRGExcitedOptions",
    "BatchTDVPOptions", "BatchGeometryError", "BatchKrylovError",
    # models
    "tfim_mpo", "tfim_ring_mpo", "magnetization_mpo", "tfim_ground_energy",
    "schwinger_mpo", "SX", "SY", "SZ",
    # observables
    "all_entropies", "entanglement_entropy", "local_expectations",
    "neighbor_correlations", "two_point", "structure_factor",
    "magnetization_moments", "mps_overlap", "apply_mpo", "to_numpy",
    "two_site_entropy", "compress", "mpo_expectation", "mpo_variance", "two_site_variance",
    # states and backends
    "random_mps", "right_canonicalize", "get_backend",
    # symmetries and block tensors: none unless the caller passes one
    # (provisional: may change before 1.0)
    "Symmetry", "U1", "Zn", "ProductSymmetry",
    "to_block_mpo", "to_block_mps", "project_to_block_mps", "random_block_mps",
    "to_dense_mps", "BlockBackend",
]

__all__ = list(CURATED)
