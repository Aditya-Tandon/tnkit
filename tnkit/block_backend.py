"""The backend adaptor for block-sparse tensors, and MPOs and MPSs in block form.

`BlockBackend` wraps a dense backend so that kernels written against ``be.*`` -- the two-site and
one-site matvecs, the environment updates, the Krylov exponential -- run on `BlockTensor`
operands unchanged.

Provisional: `BlockBackend` and this module's API may change before 1.0.
"""
from __future__ import annotations

import itertools
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .blocks import BlockTensor, Leg, from_dense, random as random_blocks, tensordot
from .symmetry import Symmetry, U1

__all__ = ["BlockBackend", "mpo_charges", "to_block_mpo", "random_block_mps", "to_block_mps",
           "project_to_block_mps", "to_dense_mps"]


def _symmetry(sym: Symmetry, site_charges, where: str) -> Symmetry:
    """`sym`, after checking that it is an abelian Symmetry and that the physical charges were
    given."""
    if not isinstance(sym, Symmetry):
        raise TypeError(f"{where}: sym= must be a Symmetry such as U1(), got {sym!r}; "
                        "without a symmetry, use the dense tensors as they are")
    if sym.fusion != "unique":
        raise NotImplementedError(f"{where}: {sym} needs unique fusion, which block tensors "
                                  "require")
    if site_charges is None:
        raise ValueError(f"{where}: sym={sym} needs site_charges, the sector of every "
                         "physical state")
    return sym


def _u1_only(sym: Symmetry, site_charges, where: str) -> U1:
    sym = _symmetry(sym, site_charges, where)
    if type(sym) is not U1:
        raise NotImplementedError(f"{where} supports U(1) charges only, got {sym}")
    return sym


@lru_cache(maxsize=1024)
def _parse(sub: str, n: int):
    """``(axes_a, axes_b, perm)`` for a two-operand contraction, or the permutation for one
    operand; None for anything else (batch indices, traces, partial sums)."""
    if "->" not in sub or "." in sub:
        return None
    lhs, out = sub.replace(" ", "").split("->")
    terms = lhs.split(",")
    if len(terms) != n or any(len(set(t)) != len(t) for t in terms + [out]):
        return None
    if n == 1:
        return tuple(terms[0].index(c) for c in out) if sorted(terms[0]) == sorted(out) else None
    if n != 2:
        return None
    ta, tb = terms
    shared = [c for c in ta if c in tb]
    if set(shared) & set(out) or set(out) != set(ta) ^ set(tb):
        return None
    free = [c for c in ta if c not in shared] + [c for c in tb if c not in shared]
    return (tuple(ta.index(c) for c in shared), tuple(tb.index(c) for c in shared),
            tuple(free.index(c) for c in out))


def _has_block(x) -> bool:
    if isinstance(x, BlockTensor):
        return True
    return isinstance(x, (list, tuple)) and any(_has_block(i) for i in x)


def _densify(x):
    if isinstance(x, BlockTensor):
        return x.to_dense()
    if isinstance(x, (list, tuple)):
        return type(x)(_densify(i) for i in x)
    return x


class BlockBackend:
    """The backend adaptor for `BlockTensor` operands, around a dense backend `dense`.

    `einsum` (two-operand contractions without batch indices, and permutations), `vdot`, `norm`,
    `conj`, `transpose` and `zeros_like` act on blocks; a contraction over every leg returns a
    0-d array, as dense `einsum` does. Calls without block operands go to the dense backend.
    Anything else that receives a block tensor, also inside a list or tuple, is a dense
    fallback: it raises when `strict`, and otherwise densifies the block operands, runs on the
    dense backend, returns a dense result, and counts the call in `fallbacks`. A densified
    tensor has every leg in sector order (`BlockTensor.to_dense`).

    Provisional: exported from `tnkit` because block TDVP takes one
    (``TDVPOptions(backend=BlockBackend(be))``); like the rest of the backend interface, it may
    change before 1.0.
    """

    def __init__(self, dense, strict: bool = True):
        self.dense, self.strict, self.fallbacks = dense, strict, Counter()

    def __getattr__(self, name):
        if name.startswith("__") or name == "dense":
            raise AttributeError(name)
        attr = getattr(self.dense, name)
        if not callable(attr):
            return attr
        return lambda *args, **kw: self._dense(name, attr, args, kw)

    def _dense(self, name, fn, args, kw):
        if not _has_block(list(itertools.chain(args, kw.values()))):
            return fn(*args, **kw)
        if self.strict:
            raise NotImplementedError(f"be.{name} has no block-sparse implementation for these "
                                      f"operands (BlockBackend(strict=False) densifies and counts)")
        self.fallbacks[name] += 1
        return fn(*_densify(args), **{k: _densify(v) for k, v in kw.items()})

    def einsum(self, sub, *ops):
        if not _has_block(ops):
            return self.dense.einsum(sub, *ops)
        spec = _parse(sub, len(ops)) if all(isinstance(x, BlockTensor) for x in ops) else None
        if spec is None:
            return self._dense("einsum", self.dense.einsum, (sub,) + ops, {})
        if len(ops) == 1:
            return ops[0].transpose(spec)
        ax_a, ax_b, perm = spec
        out = tensordot(ops[0], ops[1], (ax_a, ax_b))
        return out.data.reshape(()) if out.ndim == 0 else out.transpose(perm)

    def array(self, x):
        """A block tensor with its buffer moved to the dense backend's dtype and device."""
        if isinstance(x, BlockTensor):
            d = self.dense.array(x.data)              # a real backend may return a strided view
            d = np.ascontiguousarray(d) if isinstance(d, np.ndarray) else d.contiguous()
            return BlockTensor(x.struct, d)
        return self.dense.array(x)

    def transpose(self, x, perm):
        return x.transpose(perm) if isinstance(x, BlockTensor) else self.dense.transpose(x, perm)

    def conj(self, a):
        return a.conj() if isinstance(a, BlockTensor) else self.dense.conj(a)

    def zeros_like(self, a):
        if isinstance(a, BlockTensor):
            return BlockTensor(a.struct, self.dense.zeros_like(a.data))
        return self.dense.zeros_like(a)

    def norm(self, a):
        return self.dense.norm(a.data if isinstance(a, BlockTensor) else a)

    def vdot(self, a, b):
        if isinstance(a, BlockTensor) and isinstance(b, BlockTensor):
            x, y, _ = a._aligned(b)
            return self.dense.vdot(x, y)
        return self._dense("vdot", self.dense.vdot, (a, b), {})


def _live_channels(mpo: Sequence[Any], tol: float):
    """For each MPO bond, which channels lie on a path of entries above `tol` from the first
    tensor's left channels to the last tensor's right channels. The others add nothing to the
    operator."""
    nz = [(np.abs(np.asarray(W)) > tol).any(axis=(2, 3)) for W in mpo]
    fwd = [np.ones(nz[0].shape[0], bool)]
    for m in nz:
        fwd.append(m[fwd[-1]].any(0))
    bwd = [np.ones(nz[-1].shape[1], bool)]
    for m in reversed(nz):
        bwd.append(m[:, bwd[-1]].any(1))
    return [f & b for f, b in zip(fwd, reversed(bwd))]


def mpo_charges(mpo: Sequence[Any], site_charges: Sequence[Sequence[Any]], tol: float = 0.0,
                left: Optional[Sequence[Any]] = None, *, sym: Symmetry):
    """The charge of every MPO channel under the symmetry `sym`, one per MPO bond
    (``len(mpo) + 1`` of them): int arrays for U(1), lists of labels otherwise. `sym` is
    required; without a symmetry, use the dense MPO as it is.

    ``W[wl, wr, s, s'] != 0`` moves charge ``q(s) - q(s')`` across the site, so
    ``q(wr) = q(wl) + q(s) - q(s')``, starting from `left` (default: the trivial sector on every
    left channel of the first tensor, right for a whole chain, not for a fragment). Channels that
    no path of nonzero entries connects to both ends carry nothing; they get the trivial sector
    and their entries are ignored. Raises if a channel would carry two charges; then the MPO
    cannot be written in blocks as it stands (write its terms with charged operators, such as
    S+ S- rather than X X + Y Y).
    """
    sym = _symmetry(sym, site_charges, "mpo_charges")
    live = _live_channels(mpo, tol)
    out = [[sym.unit()] * np.shape(mpo[0])[0] if left is None else [sym.label(a) for a in left]]
    for n, (W, q) in enumerate(zip(mpo, site_charges)):
        W, q = np.asarray(W), [sym.label(x) for x in q]
        a, b, s, t = np.nonzero(np.abs(W) > tol)
        on = live[n][a] & live[n + 1][b]
        found = {int(j): set() for j in np.flatnonzero(live[n + 1])}
        for i, j, x, y in zip(a[on], b[on], s[on], t[on]):
            found[j].add(sym.fuse(sym.fuse(out[-1][i], q[x]), sym.dual(q[y])))
        qr = [sym.unit()] * W.shape[1]
        for j, vals in found.items():
            if len(vals) != 1:
                raise ValueError(f"MPO site {n}, right channel {j}: charges "
                                 f"{sorted(vals, key=sym.sort_key)} (an operator that mixes "
                                 f"charges, or wrong `left` charges for a fragment?)")
            qr[j] = vals.pop()
        out.append(qr)
    return [np.asarray(c, np.int64) for c in out] if isinstance(sym, U1) else out


def to_block_mpo(mpo: Sequence[Any], site_charges: Sequence[Sequence[Any]], be, tol: float = 0.0,
                 left: Optional[Sequence[Any]] = None, *, sym: Symmetry):
    """The MPO as block tensors with legs ``(wl, wr, s, s')`` of signs ``(+1, -1, +1, -1)``, in
    `be`'s dtype and device, and for each site the dense-index orders of those legs (for
    `BlockTensor.to_dense`). `site_charges[n][s]` is the sector label of state `s` of site `n`
    under the symmetry `sym` (required; without one, use the dense MPO as it is). Channel
    charges come from `mpo_charges`; entries at or below `tol` are dropped. Channels that carry
    nothing get the trivial sector: their entries that do not fit it are dropped, the rest kept;
    either way they add nothing to the operator."""
    chans = mpo_charges(mpo, site_charges, tol, left, sym=sym)
    out, orders = [], []
    for n, W in enumerate(mpo):
        (cl, ol), (cr, orr) = (Leg.from_charges(chans[n], sym=sym),
                               Leg.from_charges(chans[n + 1], sym=sym))
        p, op = Leg.from_charges(site_charges[n], sym=sym)
        out.append(from_dense(be.array(W), (cl, cr.flip(), p, p.flip()), None,
                              (ol, orr, op, op))[0])
        orders.append((ol, orr, op, op))
    return out, orders


def random_block_mps(mpo: Sequence[BlockTensor], chi: int, be, charge: Any,
                     rng: Optional[np.random.Generator] = None):
    """A random MPS of block tensors ``(a, s, b)``, signs ``(+1, +1, -1)``, on the physical legs of
    the block MPO `mpo`, with total charge `charge` (a sector label of the MPO's symmetry;
    required), bonds of at most `chi` states, and standard-normal entries in `be`'s dtype and
    device. Not normalised.

    Bond n holds sectors reachable from the sectors kept at bond n - 1 and, backwards, from
    `charge` at the right end, so the chain always connects. Each gets states in proportion to w,
    the smaller of its numbers of basis states on the two sides (at least 1, at most w); with more
    than `chi` candidates, those of largest w are kept. Sectors that then lead nowhere are
    dropped."""
    rng = np.random.default_rng(0) if rng is None else rng
    sym = mpo[0].struct.sym
    phys = [W.legs[2] for W in mpo]
    target = sym.label(charge)

    def grow(counts, leg, sign):
        out = {}
        for c, n in counts.items():
            for a, d in zip(leg.charges, leg.dims):
                c2 = sym.fuse(c, a if sign > 0 else sym.dual(a))
                out[c2] = out.get(c2, 0) + n * d
        return out

    left, right = [{sym.unit(): 1}], [{target: 1}]
    for p in phys:
        left.append(grow(left[-1], p, 1))
    for p in reversed(phys):
        right.append(grow(right[-1], p, -1))
    right.reverse()
    if target not in left[-1]:
        raise ValueError(f"no state of these sites has total charge {target!r}")
    bonds, cand = [], {sym.unit()}
    for n, (lc, rc) in enumerate(zip(left, right)):
        w = {c: min(lc[c], rc[c]) for c in cand if c in rc}
        keep = sorted(w, key=lambda c: (-w[c], sym.sort_key(c)))[:chi]
        tot = sum(w[c] for c in keep)
        dims = {c: min(w[c], max(1, int(chi * w[c] / tot))) for c in keep}
        while sum(dims.values()) > chi:
            dims[max(dims, key=dims.get)] -= 1
        bonds.append(dims)
        if n < len(phys):
            cand = {sym.fuse(c, a) for c in keep for a in phys[n].charges}
    for n in range(len(phys) - 1, -1, -1):       # drop sectors with no kept successor
        prv = {sym.fuse(c, sym.dual(a)) for c in bonds[n + 1] for a in phys[n].charges}
        bonds[n] = {c: d for c, d in bonds[n].items() if c in prv}
    legs = []
    for b in bonds:
        cs = sorted(b, key=sym.sort_key)
        legs.append(Leg(tuple(cs), tuple(b[c] for c in cs), 1, sym))
    return [random_blocks((legs[n], p, legs[n + 1].flip()), sym.unit(), be, rng)
            for n, p in enumerate(phys)]


def to_block_mps(mps: Sequence[Any], site_charges: Sequence[Sequence[int]], be, *,
                 sym: Symmetry, weight_tol: float = 1e-6,
                 label_tol: Optional[float] = None,
                 progress: Optional[Callable[[str], None]] = None):
    """A dense MPS as block tensors with legs ``(a, s, b)`` of signs ``(+1, +1, -1)``, in `be`'s
    dtype and device, and a dict of diagnostics. `sym` is required and must be `U1()` (without a
    symmetry, use the dense MPS as it is); `site_charges[n][s]` is the integer charge of state `s`
    of site `n`.

    The state is brought to left-canonical form and each bond basis is rotated to diagonalise the
    charge to the left of the bond; the rounded eigenvalues label the bond. Entries that break the
    labels are dropped, and the state is renormalised. Raises when the dropped weight exceeds
    `weight_tol`, or (if given) when a label is off an integer by more than `label_tol`: basis
    vectors with tiny Schmidt weight can be far from any sector without costing weight.
    Diagnostics: ``label_err`` (largest distance of a charge eigenvalue
    from its label), ``charge`` (the total charge), ``fidelity`` (|<psi|P psi>|^2 with the
    normalised projected state P psi). `progress`, if given, is called with a short message after
    each pass and every 40 sites of the last one.
    """
    sym = _u1_only(sym, site_charges, "to_block_mps")
    say = progress or (lambda msg: None)
    A = [np.asarray(t, dtype=complex) for t in mps]
    N = len(A)
    Rm = np.eye(1, dtype=complex)
    for n in range(N - 1):
        T = np.tensordot(Rm, A[n], (1, 0))
        Dl, d, Dr = T.shape
        Q, Rm = np.linalg.qr(T.reshape(Dl * d, Dr))
        A[n] = Q.reshape(Dl, d, -1)
    A[-1] = np.tensordot(Rm, A[-1], (1, 0))
    A[-1] = A[-1] / np.linalg.norm(A[-1])
    say("left-canonical")
    labels, E, err = [np.zeros(1, np.int64)], np.zeros((1, 1), complex), 0.0
    for n in range(N):
        q = np.asarray(site_charges[n], dtype=float)
        X = sum(A[n][:, s, :].conj().T @ ((E + q[s] * np.eye(len(E))) @ A[n][:, s, :])
                for s in range(len(q)))
        lam, V = np.linalg.eigh((X + X.conj().T) / 2)
        if n < N - 1:
            A[n] = np.tensordot(A[n], V, (2, 0))
            A[n + 1] = np.tensordot(V.conj().T, A[n + 1], (1, 0))
        E = np.diag(lam).astype(complex)
        lab = np.rint(lam).astype(np.int64)
        err = max(err, float(np.abs(lam - lab).max()))
        labels.append(lab)
    say("bond labels")
    if label_tol is not None and err > label_tol:
        raise ValueError(f"bond charges are {err:.2e} from integers: the state is not in one "
                         f"charge sector (label_tol={label_tol})")
    legs = [Leg.from_charges(lab, sym=sym) for lab in labels]
    out, ov, nn = [], np.ones((1, 1), complex), np.ones((1, 1), complex)
    for n in range(N):
        (la, oa), (lb, ob) = legs[n], legs[n + 1]
        p, op = Leg.from_charges(site_charges[n], sym=sym)
        out.append(from_dense(be.array(A[n]), (la, p, lb.flip()), 0, (oa, op, ob))[0])
        b = np.asarray(be.to_numpy(out[-1].to_dense((oa, op, ob))))
        ov = np.einsum("ab,asc,bsd->cd", ov, A[n].conj(), b, optimize=True)   # pairwise: D^3
        nn = np.einsum("ab,asc,bsd->cd", nn, b.conj(), b)     # not pairwise: sets out[-1]
        if (n + 1) % 40 == 0 or n == N - 1:
            say(f"blocks {n + 1}/{N}")
    norm2 = float(abs(nn[0, 0]))
    fid = float(abs(ov[0, 0]) ** 2 / norm2) if norm2 > 0 else 0.0
    if fid < 1 - weight_tol:
        raise ValueError(f"the state is not in one charge sector: {1 - fid:.2e} of its weight is "
                         f"outside the labelled blocks (weight_tol={weight_tol})")
    out[-1] = out[-1] / np.sqrt(norm2)
    return out, {"label_err": err, "charge": int(labels[-1][0]), "fidelity": fid}


def project_to_block_mps(mps: Sequence[Any], site_charges: Sequence[Sequence[int]], be, chi: int,
                         *, sym: Symmetry, charge: Optional[int] = None,
                         rtol: float = 1e-14, progress: Optional[Callable[[str], None]] = None):
    """A dense MPS projected onto one total charge and compressed to bond dimension `chi`, as block
    tensors with legs ``(a, s, b)`` of signs ``(+1, +1, -1)``, and a dict of diagnostics. `sym`
    is required and must be `U1()` (without a symmetry, use the dense MPS as it is);
    `site_charges[n][s]` is the integer charge of state `s` of site `n`.

    Unlike `to_block_mps`, this does not need bond bases of definite charge, so it also takes states
    whose truncation mixed sectors inside a basis vector. The state is made right-canonical; then,
    left to right, the part of the state with each left charge is split by SVD and the `chi`
    largest singular values over all charges are kept (ties go to the lower charge; values below
    `rtol` times the largest are dropped). At the end the total charge `charge` (default: the one
    with the largest weight) is kept and the state renormalised.
    Diagnostics: ``charge``, ``weight`` (of that charge before renormalising), ``discarded`` (the
    largest weight cut at one bond) and ``fidelity`` (|<psi|result>|^2). `progress`, if given, is
    called with a short message after each pass and every 40 sites.
    """
    sym = _u1_only(sym, site_charges, "project_to_block_mps")
    say = progress or (lambda msg: None)
    A = [np.asarray(t, dtype=complex) for t in mps]
    N = len(A)
    for n in range(N - 1, 0, -1):
        Dl, d, Dr = A[n].shape
        Q, Rr = np.linalg.qr(A[n].reshape(Dl, d * Dr).conj().T)
        A[n] = Q.conj().T.reshape(-1, d, Dr)
        A[n - 1] = np.tensordot(A[n - 1], Rr.conj().T, (2, 0))
    A[0] = A[0] / np.linalg.norm(A[0])
    say("right-canonical")
    R, bonds, cut, out = {0: np.ones((1, 1), complex)}, [np.zeros(1, np.int64)], 0.0, []
    for n in range(N):
        q, lab = np.asarray(site_charges[n], dtype=np.int64), bonds[-1]
        start = {int(c): int(i) for c, i in zip(*np.unique(lab, return_index=True))}
        parts = {}
        for c, Rc in R.items():
            for s in range(len(q)):
                parts.setdefault(int(c + q[s]), []).append((c, s, Rc @ A[n][:, s, :]))
        svd = {c: np.linalg.svd(np.vstack([m for _, _, m in ps]), full_matrices=False)
               for c, ps in sorted(parts.items())}
        sv = np.concatenate([S for _, S, _ in svd.values()])
        keep = np.zeros(sv.size, bool)
        keep[np.argsort(-sv, kind="stable")[:chi]] = True     # ties: the lower charge
        keep &= sv >= rtol * sv.max()
        cut = max(cut, float(np.sum(sv[~keep] ** 2)))
        parts_keep = np.split(keep, np.cumsum([len(S) for _, S, _ in svd.values()])[:-1])
        ks = {c: int(k.sum()) for c, k in zip(svd, parts_keep)}
        T = np.zeros((len(lab), len(q), sum(ks.values())), complex)
        R, col = {}, 0
        for c, (U, S, Vh) in svd.items():
            k, row = ks[c], 0
            if k == 0:
                continue
            for cl, s, m in parts[c]:
                T[start[cl]:start[cl] + len(m), s, col:col + k] = U[row:row + len(m), :k]
                row += len(m)
            R[c], col = S[:k, None] * Vh[:k], col + k
        out.append(T)
        bonds.append(np.repeat(list(R), [ks[c] for c in R]).astype(np.int64))
        if (n + 1) % 40 == 0 or n == N - 1:
            say(f"projected {n + 1}/{N}")
    w = {c: float(np.sum(np.abs(r) ** 2)) for c, r in R.items()}
    c = max(w, key=w.get) if charge is None else int(charge)
    if w.get(c, 0.0) == 0.0:
        raise ValueError(f"the state has no weight at total charge {c}")
    sel = np.flatnonzero(bonds[-1] == c)
    out[-1] = np.tensordot(out[-1][:, :, sel], R[c] / np.sqrt(w[c]), (2, 0))
    bonds[-1] = np.full(1, c, np.int64)
    blk, ov = [], np.ones((1, 1), complex)
    for n in range(N):
        (la, oa), (lb, ob) = (Leg.from_charges(bonds[n], sym=sym),
                              Leg.from_charges(bonds[n + 1], sym=sym))
        p, op = Leg.from_charges(site_charges[n], sym=sym)
        blk.append(from_dense(be.array(out[n]), (la, p, lb.flip()), 0, (oa, op, ob))[0])
        ov = np.einsum("ab,asc,bsd->cd", ov, A[n].conj(), out[n], optimize=True)
    say("blocks")
    return blk, {"charge": int(c), "weight": w[c] / sum(w.values()), "discarded": cut,
                 "fidelity": float(abs(ov[0, 0]) ** 2)}


def to_dense_mps(mps: Sequence[BlockTensor], site_charges: Sequence[Sequence[Any]]):
    """Dense tensors of a block MPS on its backend: physical legs in the original basis order,
    bond legs in sector order."""
    return [t.to_dense((None, Leg.from_charges(q, sym=t.struct.sym)[1], None))
            for t, q in zip(mps, site_charges)]
