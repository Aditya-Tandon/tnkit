"""Shape checks for MPS and MPO lists, run where the solvers take them in.

    MPS  M[i] : (left, phys, right)               rank 3
    MPO  W[i] : (left, right, phys_out, phys_in)   rank 4
    env  L0, RN : (bra, mpo, ket)                  rank 3

Bonds must chain, MPO and MPS physical legs must agree, and the outer bonds must be 1
unless boundary environments are given, in which case they must match them. einsum
broadcasts a size-1 index against any size, so an outer bond that breaks this rule does
not fail on its own: it gives wrong numbers.

The checks read shapes only (host metadata on every backend, no device synchronisation).
They cannot see a transposed tensor whose dimensions happen to coincide.

`TNKIT_CHECK` sets where they run:
    entry (default)  once per call of a solver front door
    debug            also after every sweep, where the solver's own lists must also be
                     uniform (one dtype family and one device per list)
    off              never; for timing runs, and for drivers that re-enter a solver
                     once per step

Inputs may mix dtypes and devices: the front doors cast them. `uniform=True` checks a
list that should already be cast.
"""
from __future__ import annotations

import os
from operator import attrgetter, itemgetter
from typing import Any, Optional, Sequence

MODES = ("entry", "debug", "off")
_SHAPE = attrgetter("shape")


class InvariantError(ValueError):
    """An MPS, MPO or environment has the wrong shape for where it is used."""


def mode() -> str:
    """The `TNKIT_CHECK` mode, read on every call."""
    m = os.environ.get("TNKIT_CHECK", "entry").strip().lower() or "entry"
    if m not in MODES:
        raise ValueError(f"TNKIT_CHECK={m!r}; expected one of {MODES}")
    return m


def _shapes(ts: Sequence[Any], rank: int, what: str):
    shapes = list(map(_SHAPE, ts))
    if not shapes:
        raise InvariantError(f"{what} is empty")
    if set(map(len, shapes)) != {rank}:
        i = next(i for i, s in enumerate(shapes) if len(s) != rank)
        raise InvariantError(f"{what}[{i}] has shape {tuple(shapes[i])}; expected rank "
                              f"{rank}")
    return shapes


def _chain(shapes, out_ax: int, in_ax: int, what: str) -> None:
    right = list(map(itemgetter(out_ax), shapes[:-1]))
    left = list(map(itemgetter(in_ax), shapes[1:]))
    if right != left:
        i = next(i for i, (r, l) in enumerate(zip(right, left)) if r != l)
        raise InvariantError(f"{what} bond {i}-{i + 1} does not chain: {what}[{i}] has shape "
                             f"{tuple(shapes[i])}, {what}[{i + 1}] has shape {tuple(shapes[i + 1])}")


def _family(t: Any) -> str:
    if hasattr(t, "is_complex") and hasattr(t, "device"):          # torch
        return f"torch {'complex' if t.is_complex() else 'real'} on {t.device}"
    return f"numpy {'complex' if t.dtype.kind == 'c' else 'real'} on cpu"


def _uniform(ts: Sequence[Any], what: str) -> None:
    keys = [_family(t) for t in ts]
    if len(set(keys)) > 1:
        i = next(i for i, k in enumerate(keys) if k != keys[0])
        raise InvariantError(f"{what}[{i}] is {keys[i]}; {what}[0] is {keys[0]}")


def _env(E: Any, bond: int, width: int, name: str, site: str) -> None:
    s = tuple(E.shape)
    if len(s) != 3 or s[0] != bond or s[2] != bond or s[1] != width:
        raise InvariantError(f"{name} has shape {s}; the {site} bond needs ({bond}, {width}, "
                             f"{bond})")


def check_mps(M: Sequence[Any], *, ends: bool = True, uniform: bool = False):
    """Rank-3 tensors whose bonds chain; with `ends`, outer bonds of 1. Returns the shapes."""
    shapes = _shapes(M, 3, "M")
    _chain(shapes, 2, 0, "M")
    if ends and (shapes[0][0] != 1 or shapes[-1][2] != 1):
        raise InvariantError(f"outer MPS bonds are ({shapes[0][0]}, {shapes[-1][2]}); they "
                             f"must be 1 unless boundary environments are given")
    if uniform:
        _uniform(M, "M")
    return shapes


def check_mpo(W: Sequence[Any], d: Optional[int] = None, *, ends: bool = True,
              uniform: bool = False):
    """Rank-4 tensors with square physical legs (of size `d` if given) whose bonds chain;
    with `ends`, outer bonds of 1. Returns the shapes."""
    shapes = _shapes(W, 4, "W")
    out, inn = list(map(itemgetter(2), shapes)), list(map(itemgetter(3), shapes))
    if out != inn or (d is not None and set(out) != {d}):
        i = next(i for i, s in enumerate(shapes) if s[2] != s[3] or (d is not None and s[2] != d))
        raise InvariantError(f"W[{i}] has shape {tuple(shapes[i])}; physical legs must be "
                             f"equal" + ("" if d is None else f" and of size {d}"))
    _chain(shapes, 1, 0, "W")
    if ends and (shapes[0][0] != 1 or shapes[-1][1] != 1):
        raise InvariantError(f"outer MPO bonds are ({shapes[0][0]}, {shapes[-1][1]}); they "
                             f"must be 1 unless boundary environments are given")
    if uniform:
        _uniform(W, "W")
    return shapes


def check_pair(M: Sequence[Any], W: Sequence[Any], L0: Any = None, RN: Any = None, *,
               ends: bool = True, uniform: bool = False) -> None:
    """`check_mps` and `check_mpo`, plus: equal lengths, matching physical legs, and the
    outer bonds of both against `L0`/`RN` where given (against 1 where not, with `ends`)."""
    ms = check_mps(M, ends=False, uniform=uniform)
    ws = check_mpo(W, ends=False, uniform=uniform)
    if len(ms) != len(ws):
        raise InvariantError(f"MPS has {len(ms)} sites, MPO has {len(ws)}")
    if list(map(itemgetter(1), ms)) != list(map(itemgetter(2), ws)):
        i = next(i for i, (m, w) in enumerate(zip(ms, ws)) if m[1] != w[2])
        raise InvariantError(f"site {i}: MPS physical leg {ms[i][1]}, MPO physical legs "
                             f"{tuple(ws[i][2:])}")
    if L0 is not None:
        _env(L0, ms[0][0], ws[0][0], "L0", "left")
    elif ends and (ms[0][0] != 1 or ws[0][0] != 1):
        raise InvariantError(f"left outer bonds are {ms[0][0]} (MPS) and {ws[0][0]} (MPO); "
                             f"without L0 both must be 1")
    if RN is not None:
        _env(RN, ms[-1][2], ws[-1][1], "RN", "right")
    elif ends and (ms[-1][2] != 1 or ws[-1][1] != 1):
        raise InvariantError(f"right outer bonds are {ms[-1][2]} (MPS) and {ws[-1][1]} (MPO); "
                             f"without RN both must be 1")


def entry(M: Optional[Sequence[Any]] = None, W: Optional[Sequence[Any]] = None,
          L0: Any = None, RN: Any = None, *, ends: bool = True) -> None:
    """The front-door check: whatever of `M` and `W` is given, unless `TNKIT_CHECK=off`."""
    if mode() == "off":
        return
    if M is not None and W is not None:
        check_pair(M, W, L0, RN, ends=ends)
    elif W is not None:
        check_mpo(W, ends=ends)
    elif M is not None:
        check_mps(M, ends=ends)


def sweep(M: Sequence[Any], W: Optional[Sequence[Any]] = None, L0: Any = None,
          RN: Any = None) -> None:
    """The in-sweep check, run only under `TNKIT_CHECK=debug`."""
    if mode() == "debug":
        if W is None:
            check_mps(M, ends=False, uniform=True)
        else:
            check_pair(M, W, L0, RN, uniform=True)
