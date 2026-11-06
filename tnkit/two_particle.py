"""Compatibility shim -- this module moved to `tnkit.ift.two_particle`.

The field-theory helpers live in the `tnkit.ift` subpackage and are absent from
`tnkit.__all__`. The old import path keeps working for every name. Prefer
`from tnkit.ift.two_particle import ...` in new code; see `tnkit/ift/__init__.py`.
"""
from .ift import two_particle as _impl
from ._shim import install as _install

__all__, _source = _install(_impl, globals())
