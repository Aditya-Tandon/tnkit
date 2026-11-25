"""The backend adaptor for symmetric tensors, and MPOs and MPSs made of them.

`SymmetricBackend` wraps a dense backend so that the two-site DMRG and TDVP drivers, written
against ``be.*``, run on `SymmetricTensor` operands of any symmetry, abelian or not. Every tensor
the adaptor returns has all its legs in the codomain (`SymmetricTensor.canonical`), so it holds
the unit coupled sector only and its flat buffer has the dense inner product: the Krylov solvers
work on buffers unchanged. A leg's dual flag plays the part of a block leg's sign.

Tensors of the drivers, all legs in the codomain: an MPS tensor (a, s, b) has spaces
(V_a, P, dual V_b), an MPO tensor (wl, wr, s, s') has (W_l, dual W_r, P, dual P), and an
environment (bra, mpo, ket) has (V, dual W, dual V) on the left.

Bond caps count multiplets. The drivers' `chi` (and TDVP's `chi_max` and `expand`) is the number
of multiplets a bond may keep, and their profiles report bonds in multiplets; the dense bond is
the sum of dim(c) over them, so a bond of `chi` multiplets is `chi` dense states only for an
abelian symmetry. The shape checks of `tnkit.invariants` count multiplets as well.

`symmetric_mpo` projects a dense MPO onto symmetric tensors. A `SiteSpace` gives the physical
leg and the unitary from the model's basis to the leg's dense basis (sectors in order, the
states of each multiplet highest weight first, in the phases of the symmetry's CG
coefficients). Channels are labelled bond by bond from the left: each channel must be one state
of one multiplet, and channels that share an irrep are recombined so that every multiplet has
one reduced coefficient vector. A channel with weight on two sectors, or on two states of one
multiplet, raises.

`save_mps` and `load_mps` write a symmetric MPS to one ``.npz`` and read it back. The buffers
hold reduced blocks in this version's conventions (tree order, CG phases, block order), which
are provisional: a file from another tnkit version loads with a warning.

Provisional: not exported from `tnkit`; this module's API may change before 1.0.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backend import NumpyBackend
from .block_backend import BlockBackend, _live_channels, _parse
from .symmetric_tensor import (Space, SymmetricTensor, contract, from_blocks, from_dense,
                               fuse_spaces, norm, permute, random, vdot)

__all__ = ["SymmetricBackend", "SiteSpace", "spin_half_site", "symmetric_mpo",
           "random_symmetric_mps", "to_dense_mps", "edge_environments", "save_mps", "load_mps"]


def _has_sym(x) -> bool:
    if isinstance(x, SymmetricTensor):
        return True
    return isinstance(x, (list, tuple)) and any(_has_sym(i) for i in x)


class SymmetricBackend:
    """The backend adaptor for `SymmetricTensor` operands, around a dense backend `dense`.

    `einsum` (two-operand contractions without batch indices, and permutations), `vdot`, `norm`,
    `conj`, `transpose`, `zeros_like` and `array` act on symmetric tensors and return them with
    every leg in the codomain; a contraction over every leg returns a 0-d array. Calls without
    symmetric operands go to the dense backend; any other call with one raises.

    Provisional: like the rest of the backend interface, it may change before 1.0.
    """

    def __init__(self, dense):
        self.dense = dense

    @property
    def name(self) -> str:
        return f"symmetric-{self.dense.name}"

    def __getattr__(self, name):
        if name.startswith("__") or name == "dense":
            raise AttributeError(name)
        attr = getattr(self.dense, name)
        if not callable(attr):
            return attr

        def call(*args, **kw):
            if _has_sym(list(args) + list(kw.values())):
                raise NotImplementedError(f"be.{name} has no symmetric-tensor implementation")
            return attr(*args, **kw)
        return call

    def einsum(self, sub, *ops):
        if not _has_sym(ops):
            # Krylov arrays change shape at every update, so a replay per shape never pays
            if isinstance(self.dense, NumpyBackend):
                return np.einsum(sub, *ops, optimize=True)
            return self.dense.einsum(sub, *ops)
        spec = _parse(sub, len(ops)) if all(isinstance(x, SymmetricTensor) for x in ops) else None
        if spec is None:
            raise NotImplementedError(f"einsum {sub!r} on symmetric tensors: only two-operand "
                                      "contractions and permutations")
        if len(ops) == 1:
            return permute(ops[0], spec, ())
        ax_a, ax_b, perm = spec
        out = contract(ops[0], ops[1], ax_a, ax_b)
        if out.ndim == 0:
            return out.data.reshape(())
        return permute(out, perm, ())

    def array(self, x):
        """A symmetric tensor with its buffer moved to the dense backend's dtype and device."""
        if isinstance(x, SymmetricTensor):
            d = self.dense.array(x.data)
            d = np.ascontiguousarray(d) if isinstance(d, np.ndarray) else d.contiguous()
            return SymmetricTensor(x.codomain, x.domain, d, self.dense, x.sym)
        return self.dense.array(x)

    def transpose(self, x, perm):
        if isinstance(x, SymmetricTensor):
            return permute(x, tuple(perm), ())
        return self.dense.transpose(x, perm)

    def conj(self, a):
        return a.conj() if isinstance(a, SymmetricTensor) else self.dense.conj(a)

    def zeros_like(self, a):
        if isinstance(a, SymmetricTensor):
            return a.with_data(a.be.zeros_like(a.data))
        return self.dense.zeros_like(a)

    def norm(self, a):
        return norm(a) if isinstance(a, SymmetricTensor) else self.dense.norm(a)

    def vdot(self, a, b):
        if isinstance(a, SymmetricTensor) and isinstance(b, SymmetricTensor):
            return vdot(a, b)
        if _has_sym([a, b]):
            raise NotImplementedError("vdot of a symmetric tensor with a dense array")
        return self.dense.vdot(a, b)


# --------------------------------------------------------------------------- site spaces
@dataclass(frozen=True, eq=False)
class SiteSpace:
    """A physical leg: the `space` and the unitary `basis` taking a model's basis state to the
    leg's dense basis, ``x_leg = basis @ x_model``."""
    space: Space
    basis: np.ndarray

    def __post_init__(self):
        U = np.asarray(self.basis)
        if U.shape != (self.space.dim,) * 2 or not np.allclose(U.conj().T @ U, np.eye(len(U))):
            raise ValueError(f"basis must be a unitary of size {self.space.dim}")


def spin_half_site(sym=None) -> SiteSpace:
    """A spin 1/2 of SU(2) (label 1 = 2j), in the basis (up, down) of `tnkit.models`."""
    from .symmetry import SU2
    sym = SU2() if sym is None else sym
    return SiteSpace(Space(sym, (1,), (1,)), np.eye(2))


# --------------------------------------------------------------------------- MPOs
def _coupled_basis(sym, legs: Tuple[Space, ...]):
    """The dense invariant basis of the plain legs `legs`: the unitary U whose column
    (c, row r, state w) is the tree tensor of row r of coupled sector c, the rows of each c in
    layout order, and the space holding each c with its row count."""
    out = fuse_spaces(legs)
    eye = {c: np.eye(n) for c, n in zip(out.sectors, out.degeneracies)}
    U = from_blocks(eye, legs, (out,), sym=sym).to_dense()
    return np.asarray(U).reshape(-1, out.dim), out


def _decompose(sym, cols: np.ndarray, basis: np.ndarray, out: Space, tol: float, where: str):
    """For each column of `cols` (dense vectors on the legs of `basis`): its sector, state and
    coefficient vector over the rows of that sector. Raises if a column is not one state of one
    multiplet."""
    a = basis.conj().T @ cols
    found = []
    for j in range(cols.shape[1]):
        x, top = a[:, j], np.abs(a[:, j]).max()
        if top == 0:
            found.append(None)
            continue
        hits, start = [], 0
        for c, n in zip(out.sectors, out.degeneracies):
            d = sym.dim(c)
            blk = x[start:start + n * d].reshape(n, d)
            for m in np.flatnonzero(np.abs(blk).max(axis=0) > tol * top):
                hits.append((c, int(m), blk[:, m]))
            start += n * d
        if len({c for c, _, _ in hits}) > 1:
            raise ValueError(f"{where}, channel {j}: weight on sectors "
                             f"{sorted({c for c, _, _ in hits}, key=sym.sort_key)}; a channel "
                             f"must carry one irrep. Either the operator breaks the symmetry "
                             f"(for isospin SU(2): unequal flavour masses or another "
                             f"isospin-breaking term), or its compiler shares one channel "
                             f"between irreps")
        if len(hits) > 1:
            raise ValueError(f"{where}, channel {j}: weight on states {[m for _, m, _ in hits]} "
                             f"of sector {hits[0][0]}; a channel must be one state of its "
                             f"multiplet")
        found.append(hits[0])
    return found


def symmetric_mpo(mpo: Sequence[Any], sites: Sequence[SiteSpace], be=None, tol: float = 1e-10):
    """The dense MPO `mpo` (tensors ``(wl, wr, s, s')`` in the models' basis) as symmetric
    tensors with spaces ``(W_l, dual W_r, P, dual P)`` on the backend `be` (default numpy).

    Each bond's channels are relabelled from the left: the coefficients of every live channel
    in the coupled basis of (W_l, P, dual P) must lie in one sector c and one state m of it.
    For each c, the coefficient vectors of all its channels span the reduced degeneracy space
    of c on the bond, and channel j of state m becomes sum_i T[i, j] (c, i, m). The basis is
    orthonormal (by SVD of the coefficient vectors, each scaled to unit norm, so that channels
    of very different norm all count) times sqrt(dim P), the norm of the identity on P, which
    keeps T of order one along a long chain. Entries below `tol` times a channel's largest
    are zero.

    Returns ``(tensors, info)``: ``info["maps"][n]`` is T as a matrix from the original channels
    of bond n to the dense index of its space (dead channels: zero columns), so that
    ``W_reduced[n] . maps[n + 1] = maps[n] . W[n]`` in the leg's physical basis;
    ``info["errors"][n]`` is the largest deviation from that identity at site n, relative to
    the largest entry of each channel. Raises if it exceeds `tol`.
    """
    from .backend import NumpyBackend
    be = be or NumpyBackend()
    if len(sites) != len(mpo):
        raise ValueError(f"{len(sites)} site spaces for {len(mpo)} MPO tensors")
    sym = sites[0].space.sym
    live = _live_channels(mpo, 0.0)
    if int(np.shape(mpo[0])[0]) != 1 or int(np.shape(mpo[-1])[1]) != 1:
        raise ValueError("symmetric_mpo needs outer MPO bonds of one channel")
    left = Space(sym, (sym.unit(),), (1,))
    T = np.ones((1, 1))
    maps, errors, out = [T], [], []
    for n, (W, site) in enumerate(zip(mpo, sites)):
        U, P = np.asarray(site.basis), site.space
        unit = np.sqrt(P.dim)
        W = np.einsum("st,abtu,vu->absv", U, np.asarray(W), U.conj())
        Wl = np.einsum("ka,absv->ksvb", T, W)                     # rows on the reduced leg
        lv = np.flatnonzero(live[n + 1])
        basis, cs = _coupled_basis(sym, (left, P, P.dual()))
        cols = Wl[..., lv].reshape(-1, len(lv))
        hits = _decompose(sym, cols, basis, cs, tol, f"MPO site {n}")
        blocks, Tn, secs = {}, {}, []
        for c in cs.sectors:
            idx = [k for k, h in enumerate(hits) if h is not None and h[0] == c]
            if not idx:
                continue
            V = np.stack([hits[k][2] for k in idx], axis=1)
            u, s, _ = np.linalg.svd(V / np.linalg.norm(V, axis=0), full_matrices=False)
            B = u[:, s > tol * s[0]]
            blocks[c] = B * unit
            Tn[c] = [(lv[k], hits[k][1], B.conj().T @ hits[k][2] / unit) for k in idx]
            secs.append(c)
        if not secs:
            raise ValueError(f"MPO site {n} has no live right channel")
        right = Space(sym, secs, tuple(blocks[c].shape[1] for c in secs))
        T = np.zeros((right.dim, W.shape[1]), np.result_type(Wl, *blocks.values()))
        for c, o, r in zip(right.sectors, right.offsets, right.degeneracies):
            d = sym.dim(c)
            for j, m, coef in Tn[c]:
                T[o + np.arange(r) * d + m, j] = coef
        red = from_blocks(blocks, (left, P, P.dual()), (right,), sym=sym)
        dense = np.asarray(red.to_dense())
        ref = Wl[..., lv].reshape(-1, len(lv))
        dev = np.einsum("ksvr,rj->ksvj", dense, T[:, lv]).reshape(ref.shape) - ref
        top = np.abs(ref).max(axis=0)
        errors.append(float((np.abs(dev).max(axis=0) / np.where(top > 0, top, 1.0)).max()))
        if errors[-1] > tol:
            raise ValueError(f"MPO site {n}: the reduced tensor misses a channel of the dense "
                             f"one by {errors[-1]:.2e} of its largest entry")
        out.append(permute(red, (0, 3, 1, 2), ()))
        maps.append(T)
        left = right
    if left.sectors != (sym.unit(),) or left.degeneracies != (1,):
        raise ValueError(f"the MPO's last channel carries {left.sectors}; it must be the unit")
    out[-1] = out[-1] * (complex(T[0, 0]) if np.iscomplexobj(T) else float(T[0, 0]))
    maps[-1] = np.ones((1, 1))
    out = [SymmetricTensor(t.codomain, t.domain, be.array(np.asarray(t.data)), be, sym)
           for t in out]
    return out, {"maps": maps, "errors": errors}


# --------------------------------------------------------------------------- MPSs
def _grow(sym, counts, leaves):
    new: Dict = {}
    for e, k in counts.items():
        for a, n in leaves:
            for c, m in sym.fusions(e, a):
                new[c] = new.get(c, 0) + k * n * m
    return new


def random_symmetric_mps(mpo: Sequence[SymmetricTensor], chi: int, be, charge,
                         rng: Optional[np.random.Generator] = None) -> List[SymmetricTensor]:
    """A random MPS (a, s, b) on the physical legs of the symmetric MPO `mpo`, of total sector
    `charge` (held by the last bond, once), with at most `chi` multiplets on each bond and
    standard-normal reduced entries in `be`'s dtype. Not normalised.

    Bond n holds the sectors reachable from the left and, backwards, from `charge`; each gets
    multiplets in proportion to the smaller of its multiplet counts on the two sides, and with
    more candidates than `chi` those of largest count are kept."""
    rng = np.random.default_rng(0) if rng is None else rng
    sym = mpo[0].sym
    target = sym.label(charge)
    phys = [W.codomain[2] for W in mpo]
    left, right = [{sym.unit(): 1}], [{target: 1}]
    for P in phys:
        left.append(_grow(sym, left[-1], list(zip(P.sectors, P.degeneracies))))
    for P in reversed(phys):
        right.append(_grow(sym, right[-1], [(sym.dual(a), n) for a, n in
                                            zip(P.sectors, P.degeneracies)]))
    right.reverse()
    if target not in left[-1]:
        raise ValueError(f"no state of these sites has total sector {target!r}")
    bonds = []
    for lc, rc in zip(left, right):
        w = {c: min(lc[c], rc[c]) for c in lc if c in rc}
        keep = sorted(w, key=lambda c: (-w[c], sym.sort_key(c)))[:chi]
        tot = sum(w[c] for c in keep)
        dims = {c: min(w[c], max(1, int(chi * w[c] / tot))) for c in keep}
        while sum(dims.values()) > chi:
            dims[max(dims, key=dims.get)] -= 1
        bonds.append({c: d for c, d in dims.items() if d})
    spaces = [Space(sym, sorted(b, key=sym.sort_key), [b[c] for c in sorted(b, key=sym.sort_key)])
              for b in bonds]
    return [random((spaces[n], P, spaces[n + 1].dual()), (), be, rng)
            for n, P in enumerate(phys)]


def to_dense_mps(mps: Sequence[SymmetricTensor], sites: Sequence[SiteSpace]):
    """Dense tensors (a, s, b) of a symmetric MPS on its backend, with the physical legs in the
    models' basis; bond legs keep their dense index."""
    out = []
    for t, site in zip(mps, sites):
        U = t.be.array(np.asarray(site.basis).conj().T)
        out.append(t.be.einsum("st,atb->asb", U, t.to_dense()))
    return out


def edge_environments(M: Sequence[SymmetricTensor], mpo: Sequence[SymmetricTensor], be):
    """The open-boundary environments of the drivers: the identity on the outer bond's multiplet
    and the outer MPO channel, as (bra, mpo, ket) tensors."""
    dense = be.dense if isinstance(be, SymmetricBackend) else be
    a, b = M[0].codomain[0], M[-1].codomain[2]
    wl, wr = mpo[0].codomain[0].dual(), mpo[-1].codomain[1].dual()
    envs = []
    for v, w in ((a, wl), (b, wr)):
        x = np.eye(v.dim)[:, None, :]
        t, dropped = from_dense(dense.array(x), (v, w, v.dual()), (), dense)
        if dropped > 1e-24:
            raise ValueError("the outer bond and MPO channel do not admit an identity "
                             "environment")
        envs.append(t)
    return tuple(envs)


# --------------------------------------------------------------------------- files
_FORMAT = 1              # the file layout; a file of another layout is refused


def _plain(c):
    """A sector label as JSON: an int, or a list for a product label."""
    return [_plain(x) for x in c] if isinstance(c, tuple) else int(c)


def save_mps(file, mps: Sequence[SymmetricTensor]) -> None:
    """Write the symmetric MPS `mps` to one ``.npz`` at `file` (a path, to which numpy adds
    ``.npz`` if missing, or a binary file object): ``D0000``... each tensor's buffer on the host,
    ``L0000``... its legs (sectors, degeneracies, dual flags) as JSON, ``sym`` the repr of the
    symmetry, ``format`` 1 and ``tnkit`` the library version."""
    from . import __version__
    if not len(mps):
        raise ValueError("save_mps needs an MPS of at least one tensor")
    if not all(isinstance(t, SymmetricTensor) for t in mps):
        raise TypeError("save_mps takes an MPS of SymmetricTensors")
    sym = mps[0].sym
    out = {"format": np.array(_FORMAT), "sym": np.array(repr(sym)),
           "tnkit": np.array(__version__)}
    for i, t in enumerate(mps):
        if t.domain or t.sym != sym:
            raise ValueError(f"tensor {i}: an MPS tensor has all legs in the codomain and the "
                             f"symmetry of the others")
        out[f"D{i:04d}"] = np.asarray(t.be.to_numpy(t.data))
        legs = [[[_plain(c) for c in V.sectors], list(V.degeneracies), V.isdual]
                for V in t.codomain]
        out[f"L{i:04d}"] = np.array(json.dumps(legs))
    np.savez(file, **out)


def load_mps(file, sym, be=None) -> List[SymmetricTensor]:
    """The symmetric MPS written by `save_mps` to `file` (a path, ``.npz`` added if missing as
    `save_mps` adds it, or a binary file object), with its buffers on the dense backend `be`
    (default numpy, in the result type of the file's buffers). Raises if `sym` is not the file's
    symmetry or the file is not of this layout; warns if another tnkit version wrote it."""
    from . import __version__
    if isinstance(be, BlockBackend):
        raise TypeError("load_mps takes a dense backend or a SymmetricBackend")
    if isinstance(file, (str, os.PathLike)) and not os.fspath(file).endswith(".npz"):
        file = os.fspath(file) + ".npz"
    z = np.load(file)
    if not hasattr(z, "files"):
        raise ValueError("not an MPS file written by save_mps")
    with z:
        if not {"format", "sym", "tnkit"} <= set(z.files):
            raise ValueError("not an MPS file written by save_mps")
        if int(z["format"]) != _FORMAT:
            raise ValueError(f"MPS file format {int(z['format'])}; this version reads "
                             f"{_FORMAT}")
        if str(z["sym"]) != repr(sym):
            raise ValueError(f"the file holds an MPS of {z['sym']}, not of {sym!r}")
        n = sum(k.startswith("D") for k in z.files)
        data = [z[f"D{i:04d}"] for i in range(n)]
        legs = [json.loads(str(z[f"L{i:04d}"])) for i in range(n)]
        if str(z["tnkit"]) != __version__:
            warnings.warn(f"MPS file written by tnkit {z['tnkit']}, read by {__version__}: the "
                          "conventions of reduced blocks may differ", stacklevel=2)
    be = be.dense if isinstance(be, SymmetricBackend) else be
    be = be or NumpyBackend(np.result_type(*data))
    return [SymmetricTensor([Space(sym, [sym.label(c) for c in sec], deg, dual)
                             for sec, deg, dual in leg], (), be.array(d), be, sym)
            for d, leg in zip(data, legs)]
