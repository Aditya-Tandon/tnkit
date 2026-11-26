"""The one mechanism behind `tnkit/dmrg.py`, `tnkit/vumps.py` and `tnkit/idmrg.py`.

Those three implementations were moved to `tnkit/_dmrg.py`, `tnkit/_vumps.py` and
`tnkit/_idmrg.py` so that the package can export the FUNCTIONS `dmrg`, `vumps` and
`idmrg` at the top level: a module and a function cannot both own `tnkit.dmrg`. The old module paths stay, as shims,
because existing code imports through them.

WHY THE NAMESPACE IS COPIED RATHER THAN LISTED. The tree imports forty-odd distinct names
from these three modules, including privates (`_real`, `_gauge`, `_left_environment`,
`_truncate`, `_xi_from_tensor`, `_cutn_truncate`, `_correlation_lengths`, ...) and names
the module merely re-imported (`from tnkit.vumps import update_left_env`, which `_vumps`
gets from `_dmrg`). A hand-maintained re-export list would have to track all of them, and
it would go stale silently -- a missing name only surfaces when some caller imports it.
Copying every module-level binding cannot go stale, and `tests/test_api.py` pins the property directly: the shim's namespace equals the
implementation's, and the mutable registries are the SAME objects on both sides.

WHAT A SHIM CANNOT DO, AND WHAT REPLACES IT. Copying binds names; it does not link them.
`tnkit.vumps._left_fixed_point = f` rebinds a name in the SHIM's namespace and the
implementation's own globals never see it, so the patched code would not run. Mutating a shared container is different in
kind -- `_EIGSOLVERS[...] = f` reaches the one dict both namespaces hold -- so the two
rebindable seams in `_vumps` live in a registry, `_HOOKS`, to be mutated rather than
rebound. See the note on `tnkit._vumps._HOOKS`.
"""
from __future__ import annotations

from types import ModuleType
from typing import Dict, List, Tuple

#: Module dunders, which belong to the shim itself and must not be overwritten.
_RESERVED = frozenset({
    "__name__", "__doc__", "__package__", "__loader__", "__spec__",
    "__file__", "__cached__", "__builtins__", "__path__",
})


def install(impl: ModuleType, namespace: Dict[str, object]) -> Tuple[List[str], str]:
    """Copy every module-level binding of `impl` into the calling module's namespace.

    Returns the shim's `__all__` (the public names) and a one-line provenance string,
    so the shim module can state where its contents came from.
    """
    names = [k for k in vars(impl) if k not in _RESERVED]
    for k in names:
        namespace[k] = getattr(impl, k)
    # Machine-readable provenance: every shim declares its implementation module, so the
    # test suite can COMPUTE the set of shimmed paths instead of hand-listing it. A rebind
    # through any module carrying this attribute does not reach the implementation.
    namespace["__shim_source__"] = impl.__name__
    return sorted(k for k in names if not k.startswith("_")), impl.__name__
