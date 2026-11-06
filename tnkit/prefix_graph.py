"""Compatibility shim: the prefix-graph capture now lives in `tnkit.prefix_graph_impl`."""
from . import prefix_graph_impl as _impl
from ._shim import install as _install

__all__, _source = _install(_impl, globals())
