"""One resolver shared by every options dataclass on the public surface.

The front doors (`dmrg`, `tdvp1_evolve`, `tdvp2_evolve`) all take the same three
things: an optional options object, keyword arguments in the old flat style, and a
name to put in the error message. Keeping the merge in one place means the two
call styles cannot drift apart, and that "unknown keyword" still fails the way a
plain function would.
"""
from __future__ import annotations
import dataclasses
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar("T")


def resolve(cls: Type[T], options: Optional[Any], legacy: Dict[str, Any], where: str) -> T:
    """Merge `options` and flat keyword arguments into one options instance.

    Precedence is explicit: `options` supplies the base (its own defaults when it
    is None), and any flat keyword given at the call site overrides that field. So
    `dmrg(mpo, chi, options=tuned, verbose=True)` means "the tuned settings, but
    talkative", and existing callers -- which pass everything flat and
    never build an options object -- get exactly the dataclass defaults, which are
    the defaults the old signature carried.

    A dict is accepted where an options object is expected, because a driver that
    builds its settings from argparse or JSON has one already.
    """
    names = {f.name for f in dataclasses.fields(cls)}
    if options is None:
        base = cls()
    elif isinstance(options, cls):
        base = options
    elif isinstance(options, dict):
        bad = [k for k in options if k not in names]
        if bad:
            raise TypeError(f"{where}(): unknown option {bad[0]!r} in the options dict; "
                            f"expected one of {sorted(names)}")
        base = cls(**options)
    else:
        raise TypeError(f"{where}(): options must be a {cls.__name__}, a dict of its fields, "
                        f"or None; got {type(options).__name__}")
    if not legacy:
        return base
    for key in legacy:
        if key not in names:
            # Same wording a plain function would produce, so a typo reads the same
            # way it always did rather than as an options-machinery error.
            raise TypeError(f"{where}() got an unexpected keyword argument {key!r}")
    return dataclasses.replace(base, **legacy)
