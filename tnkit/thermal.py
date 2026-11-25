"""Thermal states by purification.

A mixed state rho on N sites of dimension d is stored as a pure state |psi> on N fused sites of
dimension d*d: each physical site s with an ancilla a, fused index s*d + a. Then
<psi|O x 1|psi> = Tr(rho O). The infinite-temperature state is the product of the pairs
sum_s |s>|s>; cooling it in imaginary time to beta/2 under H x 1 gives
|psi(beta)> = exp(-beta H/2) x 1 |psi(0)>, with <psi(beta)|psi(beta)> = Z(beta).

The fused chain is an ordinary MPS, so every solver and observable applies to it: expectation
values through `purify` and `purify_mpo`, real-time evolution through `tdvp2_evolve` or
`TDVPEvolver` on `purify_mpo(H, twin=True)`. The twin MPO evolves the ancillas backward in time,
H x 1 - 1 x H^T, which leaves the thermal state of H unchanged and slows the growth of the bond
dimension (Karrasch, Bardarson, Moore, PRL 108, 227206 (2012)).

Block tensors. Given the integer charge q(s) of every physical basis state (`charges`, shape
(N, d)), the fused chain can be written in block tensors under `PURIFIED`, U(1) x U(1): the fused
state s*d + a has sector (q(s), q(a)), so the physical and the ancilla charge are conserved
separately. H x 1 changes only physical indices. 1 x H^T changes only ancilla indices, and since
H^T[t, s] = H[s, t] it moves the ancilla charge by minus what H moves the physical charge: its
MPO channels carry (0, -c) where those of H carry (c, 0). Both terms of the twin MPO therefore
conserve both charges. The restricted identity sum_{Q(s)=total} |s>|s> has sector
(total, total). Use ``blocks=True`` in `infinite_temperature_state` and `thermal_state`, and
`charges` in `purify_mpo` and `cool`; TDVP then needs a `BlockBackend` (`cool` wraps its backend
in one). Observables need dense tensors: `tnkit.block_backend.to_dense_mps` with
`purified_charges`.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _options
from .backend import get_backend
from .block_backend import BlockBackend, mpo_charges, to_block_mpo
from .blocks import Leg, from_dense
from .mps import MPS, MPSLike
from .symmetry import ProductSymmetry, U1
from .tdvp import TDVPOptions
from .tdvp_evolver import TDVPEvolver

PURIFIED = ProductSymmetry((U1(), U1()))
"""Sectors (physical charge, ancilla charge) of a purified chain in block tensors."""


def purify(op: np.ndarray) -> np.ndarray:
    """O x 1 on a fused site: a one-site operator for `local_expectations` and friends."""
    op = np.asarray(op)
    return np.kron(op, np.eye(op.shape[0], dtype=op.dtype))


def purified_charges(charges: Any) -> List[List[Tuple[int, int]]]:
    """The `PURIFIED` sector (q(s), q(a)) of every fused state s*d + a, per site, from the
    physical charges `charges` (shape (N, d)): the ``site_charges`` of a block purified chain."""
    return [[(x, y) for x in row for y in row] for row in _int_charges(charges).tolist()]


def _int_charges(charges: Any, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    q = np.asarray(charges)
    if q.ndim != 2 or (shape is not None and q.shape != shape):
        want = "(N, d)" if shape is None else str(shape)
        raise ValueError(f"charges must have shape {want}, got {q.shape}")
    if not np.array_equal(q, np.round(q)):
        raise ValueError("charges must be integers")
    return q.astype(int)


def purify_mpo(mpo: Sequence[Any], twin: bool = False,
               charges: Optional[Any] = None) -> List[Any]:
    """The fused-site MPO of H x 1, or with `twin` of H x 1 - 1 x H^T (the ancillas evolve
    backward in time; width doubles). `mpo` uses tnkit's (left, right, out, in) layout.

    With `charges` (physical charges, shape (N, d)), numpy block tensors under `PURIFIED`
    (`cool`, the TDVP drivers and `apply_block_mpo` move them to their own backend); raises
    unless `mpo` conserves the charge."""
    out = _purify_dense(mpo, twin)
    if charges is None:
        return out
    q = _int_charges(charges, (len(mpo), int(np.shape(mpo[0])[2])))
    try:
        mpo_charges(mpo, q, sym=U1())
    except ValueError as e:                         # keep where, drop the hint about fragments
        where = str(e).split(" (")[0]
        raise ValueError(f"the MPO does not conserve these charges: {where}") from None
    return to_block_mpo(out, purified_charges(q), get_backend("numpy"), sym=PURIFIED)[0]


def _purify_dense(mpo: Sequence[Any], twin: bool) -> List[np.ndarray]:
    out = []
    N = len(mpo)
    for i, W in enumerate(mpo):
        W = np.asarray(W)
        Dl, Dr, d, _ = W.shape
        one = np.eye(d, dtype=W.dtype)
        P = np.einsum("abst,uv->absutv", W, one).reshape(Dl, Dr, d * d, d * d)
        if not twin:
            out.append(P)
            continue
        A = np.einsum("st,abvu->absutv", one, W).reshape(Dl, Dr, d * d, d * d)
        if i == 0:
            A = -A                                  # the minus sign, once
        if N == 1:
            out.append(P + A)
        elif i == 0:
            out.append(np.concatenate([P, A], axis=1))
        elif i == N - 1:
            out.append(np.concatenate([P, A], axis=0))
        else:
            Z = np.zeros((2 * Dl, 2 * Dr, d * d, d * d), dtype=W.dtype)
            Z[:Dl, :Dr], Z[Dl:, Dr:] = P, A
            out.append(Z)
    return out


def infinite_temperature_state(N: int, d: int = 2, charges: Optional[Any] = None,
                               total: int = 0, blocks: bool = False) -> Tuple[MPS, float]:
    """The normalised purification of the identity, and log of its trace (the dimension).

    With `charges` (shape (N, d), an integer charge per site and basis state), the identity is
    restricted to states of total charge `total`: an exact MPS whose bond carries the running
    charge (bond dimension at most the number of reachable partial sums). With `blocks` (needs
    `charges`), that state in block tensors under `PURIFIED`: the bond index of running charge c
    has sector (c, c), so every block is a single number."""
    eye = np.eye(d, dtype=np.complex128).reshape(1, d * d, 1) / np.sqrt(d)
    if charges is None:
        if blocks:
            raise ValueError("blocks=True needs charges, the charge of every basis state")
        return MPS([eye.copy() for _ in range(N)]), N * float(np.log(d))
    q = _int_charges(charges, (N, d))
    tensors, bonds, dim = _restricted_identity(N, d, q, total)
    if blocks:
        tensors = _to_blocks(tensors, bonds, q)
    return MPS(tensors), float(np.log(dim))


def _restricted_identity(N: int, d: int, q: np.ndarray, total: int):
    fwd = [{0}]
    for n in range(N):
        fwd.append({c + int(x) for c in fwd[-1] for x in q[n]})
    if total not in fwd[-1]:
        raise ValueError(f"no state has total charge {total}")
    bwd = [None] * N + [{total}]
    for n in range(N - 1, -1, -1):
        bwd[n] = {c for c in fwd[n] if any(c + int(x) in bwd[n + 1] for x in q[n])}
    bonds = [sorted(b) for b in bwd]
    count = {total: 1.0}                             # number of states from each bond value
    tensors = []
    for n in range(N - 1, -1, -1):
        left, right = bonds[n], {c: j for j, c in enumerate(bonds[n + 1])}
        A = np.zeros((len(left), d * d, len(right)), dtype=np.complex128)
        new = {}
        for i, c in enumerate(left):
            for s in range(d):
                c2 = c + int(q[n, s])
                if c2 in right:
                    A[i, s * d + s, right[c2]] = 1.0
                    new[c] = new.get(c, 0.0) + count[c2]
        tensors.append(A)
        count = new
    tensors.reverse()
    dim = count[0]
    tensors[0] = tensors[0] / np.sqrt(dim)
    return tensors, bonds, dim


def _to_blocks(tensors: List[np.ndarray], bonds: List[List[int]], q: np.ndarray) -> List[Any]:
    phys = purified_charges(q)
    legs = [Leg.from_charges([(c, c) for c in b], sym=PURIFIED) for b in bonds]
    out = []
    for n, A in enumerate(tensors):
        (la, oa), (lb, ob) = legs[n], legs[n + 1]
        p, op = Leg.from_charges(phys[n], sym=PURIFIED)
        T, lost = from_dense(A, (la, p, lb.flip()), None, (oa, op, ob))
        if lost:
            raise ValueError(f"site {n}: entries outside the (Q_phys, Q_anc) blocks")
        out.append(T)
    return out


def cool(mps: MPSLike, mpo: Sequence[Any], chi: int, dbeta: float, n_steps: int,
         log_z: float = 0.0, warmup: int = 0,
         options: Optional[Union[TDVPOptions, Dict[str, Any]]] = None, method: str = "2site",
         measure: Optional[Callable[[Any], Any]] = None, meas_every: int = 1,
         beta0: float = 0.0, charges: Optional[Any] = None,
         **kwargs: Any) -> Tuple[MPS, List[Tuple[float, float, Any]]]:
    """Imaginary-time TDVP of a purified state under H x 1, `n_steps` steps of `dbeta` in beta
    (each step applies exp(-dbeta H/2)). `mpo` is the PHYSICAL MPO; it is purified here.

    Returns the normalised state and records (beta, log Z, measure(state)) every `meas_every`
    steps, starting at (beta0, log_z, ...). log Z adds 2 log of the norm each step removes, so
    truncation lowers it by the discarded weight.

    warmup : k > 0 splits the first step into steps of dbeta/2^k, dbeta/2^k, dbeta/2^(k-1), ...,
        dbeta/2. From a state of small bond dimension, two-site TDVP cannot follow terms that
        couple distant sites, and the error it makes is first order in the step while the bonds
        are small; tiny first steps let the bonds grow before beta does. Use it from infinite
        temperature with long-range Hamiltonians (`thermal_state` does, with k = 8).

    charges : the physical charges (shape (N, d)) when `mps` is in block tensors (from
        ``infinite_temperature_state(..., blocks=True)``); the MPO is then purified in blocks,
        the backend wrapped in a `BlockBackend` if it is not one, and `measure` and the result
        get block tensors. Build the state from the same table: nothing checks it beyond the
        MPO conserving it, so a table with its columns reversed is accepted whenever the MPO
        conserves that charge too, and then labels the sectors by it.

    `options`/`kwargs` are `TDVPOptions` for a `TDVPEvolver` (svd_tol, degenerate_tol, expand,
    krylov_m, backend, ...); `method` is its schedule. The step is complex (dt = -i dbeta/2), so
    CUDA graphs are off and `renorm` is False.

    Block and dense cooling agree to round-off while no cut falls inside a group of equal
    singular values and the dense state stays in its charge sector. The spectrum of
    exp(-beta H/2) pairs equal values in sectors (a, b) and (b, a); a cut between them keeps
    one, arbitrarily in a dense SVD and the lower sector in blocks. `degenerate_tol` keeps or
    drops such groups whole, by the same rule on both. A dense state also picks up weight in
    other charge sectors from round-off, and at a loose `svd_tol` cooling can let it grow,
    which alone makes the two differ."""
    o = _options.resolve(TDVPOptions, options, kwargs, "cool")
    if o.measure is not None or o.step_callback is not None:
        raise ValueError("cool takes `measure` itself; options.measure is not used")
    if o.cuda_graph is not None or o.prefix_graph is not None or o.graph == "on":
        raise ValueError("cool runs without CUDA graphs")
    if not o.canonicalize:
        raise ValueError("cool needs canonicalize=True: the norm is read off the centre at site 0")
    if hasattr(mps[0], "codomain"):
        raise ValueError("cool takes dense or block tensors; symmetric tensors are not supported")
    block = getattr(mps[0], "is_block", False)
    if block != (charges is not None):
        raise ValueError("a block state needs charges, a dense one does not take them")
    o = dataclasses.replace(o, graph="off", renorm=False)
    if block and not isinstance(get_backend(o.backend), BlockBackend):
        o = dataclasses.replace(o, backend=BlockBackend(get_backend(o.backend)))
    W = purify_mpo(mpo, charges=charges)
    ev = TDVPEvolver(mps, W, -0.5j * float(dbeta), chi, options=o)
    be = ev.be
    snap = ((lambda: list(ev.mps)) if o.keep_device or block
            else (lambda: [be.to_numpy(m) for m in ev.mps]))
    ev.prepare()
    log_z += _normalise(ev, be)
    records = [(float(beta0), log_z, measure(snap()) if measure is not None else None)]
    for k in range(n_steps):
        if k == 0 and warmup > 0:
            for h in [dbeta / 2 ** warmup] + [dbeta / 2 ** j for j in range(warmup, 0, -1)]:
                w = TDVPEvolver(ev.mps, W, -0.5j * h, chi, options=o)
                w.run(1, method)
                log_z += _normalise(w, be)
                ev.reset(w.mps)
        else:
            ev.run(1, method)
            log_z += _normalise(ev, be)
        if (k + 1) % meas_every == 0:
            records.append((float(beta0) + (k + 1) * float(dbeta), log_z,
                            measure(snap()) if measure is not None else None))
    return MPS(snap(), center=0 if ev.center == 0 else None), records


def _normalise(ev: TDVPEvolver, be: Any) -> float:
    """Scale site 0 to unit norm in place and return 2 log of the old norm. Needs the centre at
    site 0; the right environments do not involve site 0, so nothing needs rebuilding."""
    if ev.center != 0:
        ev.prepare()
    t = ev.mps[0]
    n = float(be.to_numpy(be.norm(t)))
    if getattr(t, "is_block", False):              # immutable: replace it, keep the centre
        ev.mps[0] = t * (1.0 / n)
        ev.mps.center = 0
    else:
        t *= 1.0 / n
    return 2.0 * float(np.log(n))


def thermal_state(mpo: Sequence[Any], beta: float, chi: int, dbeta: float,
                  charges: Optional[Any] = None, total: int = 0,
                  warmup: int = 8, options: Optional[Union[TDVPOptions, Dict[str, Any]]] = None,
                  method: str = "2site", blocks: bool = False,
                  **kwargs: Any) -> Tuple[MPS, float]:
    """The purified Gibbs state exp(-beta H)/Z of the physical MPO `mpo`, and log Z.

    Cools `infinite_temperature_state` (restricted to total charge `total` if `charges` is
    given) in round(beta/dbeta) steps; see `cool` for the options, `degenerate_tol` among
    them, and for when blocks and dense agree. With `blocks` (needs `charges`), in block
    tensors under `PURIFIED`."""
    n = int(round(beta / dbeta))
    if n < 0 or not np.isclose(n * dbeta, beta, rtol=1e-12, atol=1e-14):
        raise ValueError(f"beta={beta} is not a non-negative multiple of dbeta={dbeta}")
    d = int(np.asarray(mpo[0]).shape[2])
    psi, log_dim = infinite_temperature_state(len(mpo), d, charges, total, blocks)
    out, rec = cool(psi, mpo, chi, dbeta, n, log_z=log_dim, warmup=warmup, options=options,
                    method=method, charges=charges if blocks else None, **kwargs)
    return out, rec[-1][1]
