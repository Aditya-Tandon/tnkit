"""`tnkit.ift` — field-theory helpers, kept out of the engine's headline surface.

WHAT TNKIT IS. The engine is a backend-agnostic MPS toolkit: `dmrg`, `dmrg_excited`,
`tdvp`, `itdvp`, `tebd`, `idmrg`, `vumps`, `excitations`, the eigensolvers, the
truncations, the observables, the models, the backends. Someone who installs `tnkit` to
solve a spin chain needs all of it and none of this.

WHAT LIVES HERE. Two modules for finite-volume field-theory analysis, which a general MPS
user has no reason to import:

  luscher        finite-volume (Lüscher) analysis for 1+1D spin chains: the two-particle
                 quantisation condition k L + 2 delta(k) = 2 pi n, the free-fermion
                 S = -1 validation identity, and the E8 bound-state machinery.
  two_particle   the two-particle variational problem in the relative-SEPARATION
                 basis (Vanderstraeten-Verstraete-Haegeman), with the A_L-left/A_R-right
                 window convention.

`prefix_graph` is a shim over `tnkit.prefix_graph_impl`: the prefix-graph capture is the default
path of `tdvp1_evolve(graph="auto")`, so it belongs to the engine.

COMPATIBILITY. `tnkit.luscher`, `tnkit.two_particle` and `tnkit.prefix_graph` still
import and still give the same objects -- they are shims (see `tnkit/_shim.py`).

Nothing here is in `tnkit.__all__`, and `import tnkit` does not import this subpackage.
Ask for it by name::

    from tnkit.ift import luscher
    from tnkit.ift.two_particle import SeparationBasis
"""
from __future__ import annotations

from . import luscher, prefix_graph, two_particle  # noqa: F401

__all__ = ["luscher", "prefix_graph", "two_particle"]
