"""Compatibility shim -- the implementation lives in `tnkit._vumps`.

`from tnkit.vumps import ...` keeps working exactly as it did, for every name the
implementation defines or re-imports. What changed is `import tnkit.vumps as X`: the
package now exports the FUNCTION `vumps` at the top level, so that statement binds the
function rather than this module (CPython resolves `import a.b as c` by `getattr` on
`a`). Both objects carry the mutable registries an A/B driver reaches for, and they are
the same objects on both -- see `tnkit/_shim.py` for why that distinction is the whole
design, and `tests/test_api.py` for the tests that pin it.
"""
from . import _vumps as _impl
from ._shim import install as _install

__all__, _source = _install(_impl, globals())
