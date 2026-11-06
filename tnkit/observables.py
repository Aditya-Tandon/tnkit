"""Local observables and entanglement from an MPS, on whatever device it lives on.

All routines take a raw MPS (list of rank-3 tensors `(left, phys, right)`),
internally put a *copy* into right-canonical form (norm 1), and measure by
growing the left norm-environment. In right-canonical form the environment to the
right of the active site is the identity, so a one-site expectation is just a
contraction of the left environment, the site tensor, the operator, and the
conjugate site tensor.

Backends
--------
`dmrg(backend="torch-cuda")` returns CUDA tensors, and every routine here accepts
them: the backend is deduced from the state itself (`_backend_for`), the whole
contraction runs on that device, and only the *result* crosses to the host. That
transfer is one call per routine, not one per site -- the per-site scalars are
stacked on-device first (`Backend.stack`) so a length-N profile costs a single
synchronisation. The one other host round-trip is the Schmidt spectrum in the
entropy routines: chi floats per bond, converted deliberately so the entropy
arithmetic is bit-identical to the numpy path.

Returns are always numpy (or Python) scalars and arrays. Measurements are small
and end up in a file or a fit, so a device-resident result would only push the
transfer into the caller; `to_numpy(mps)` is here for callers who want the state
itself on the host.

Two of these routines are also methods on the `MPS` type that `dmrg` and `tdvp`
return: `mps.to_numpy()` and `mps.entropies()` call `to_numpy` and `all_entropies`
respectively. The arithmetic lives here; `tnkit.mps` only forwards.

Cost
----
Every routine canonicalises **once**. `entanglement_entropy(mps, bond)` sweeps to
`bond` from a fresh canonical copy, so asking it for a full S(l) profile is
O(N^2 chi^3); `all_entropies(mps)` returns the whole profile from ONE sweep at
O(N chi^3) and is what any profile should call. `entanglement_entropy` and
`local_expectations` also take `canon=` so a caller measuring the same state many
times pays the canonicalisation once.
"""
from __future__ import annotations
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ._dmrg import _truncate, right_canonicalize
from .mps import MPS, MPSLike, Tensor
from . import invariants


def _backend_for(mps: MPSLike) -> Any:
    """Backend matching the device and dtype the MPS already lives on.

    Deliberately duck-typed on the module name rather than `isinstance`: `tnkit`
    imports only numpy, scipy and ctypes at module scope, and that is what makes
    `pip install tnkit` work without torch. Importing torch here to ask whether a
    tensor is a torch tensor would undo it.
    """
    A = mps[0]
    if getattr(A, "is_block", False):
        raise TypeError("observables need dense tensors: convert a block MPS with "
                        "tnkit.block_backend.to_dense_mps")
    if type(A).__module__.split(".")[0] == "torch":
        from .backend import TorchBackend
        return TorchBackend(device=str(A.device), dtype=A.dtype)
    from .backend import NumpyBackend
    return NumpyBackend()             # complex128: the environment dtype the
                                      # numpy path has always used


def _copy(A: Tensor) -> Tensor:
    return A.clone() if hasattr(A, "clone") else A.copy()


def to_numpy(mps: MPSLike, be: Optional[Any] = None) -> List[np.ndarray]:
    """Host copy of an MPS whose tensors may live on a GPU. Contiguous numpy arrays.

    The escape hatch for anything this module does not cover: pass the result to
    code that is numpy-only. At most O(N chi^2) numbers, i.e. far less than one
    DMRG sweep moves, so it costs nothing measurable next to the solve.
    """
    be = be or _backend_for(mps)
    return [np.ascontiguousarray(be.to_numpy(A)) for A in mps]


def _canon(mps: MPSLike, be: Optional[Any] = None) -> MPSLike:
    """Right-canonical copy of `mps`, on its own device. Never mutates the input."""
    be = be or _backend_for(mps)
    return right_canonicalize([_copy(m) for m in mps], be=be)


def _profile(be: Any, vals: Sequence[Tensor]) -> np.ndarray:
    """One host transfer for a whole per-site profile, not one per site.

    On CUDA each `float(x)` is a synchronisation; stacking the N device scalars
    first turns a length-N profile from N stalls into one.

    `np.real` of a complex array is a stride-2 VIEW, where the base implementation
    returned a fresh contiguous buffer. Nothing observable moved -- 2000 trials up to
    N = 4096 on numpy 1.24 found no `np.sum` mismatch -- but numpy's pairwise summation
    is free to branch on stride in a future release, and `structure_factor` sums this
    array into a published number. A caller reaching for `.tobytes()` or handing the
    array to C also gets a surprise from a view. So it is made contiguous, which is
    what it was originally.
    """
    if not vals:
        return np.zeros(0)
    return np.ascontiguousarray(np.real(np.asarray(be.to_numpy(be.stack(vals)))))


def local_expectations(mps: MPSLike, op: np.ndarray,
                       canon: Optional[MPSLike] = None) -> np.ndarray:
    """<op_i> at every site for a one-site operator `op` (d*d). Returns array[N]."""
    be = _backend_for(mps)
    M = _canon(mps, be) if canon is None else canon
    op = be.array(op)
    N = len(M)
    out = []
    Lenv = be.ones((1, 1))                               # <left|left>, sites <i
    for i in range(N):
        A = M[i]
        # <op_i> = sum Lenv[a,b] A[a,s,c] op[s,t] conj(A[b,t,c])  (right bond c = identity)
        out.append(be.einsum("ab,asc,st,btc->", Lenv, A, op, be.conj(A)))
        # grow the norm environment past site i (right-canonical A)
        Lenv = be.einsum("ab,asc,bsd->cd", Lenv, A, be.conj(A))
    return _profile(be, out)


def neighbor_correlations(mps: MPSLike, opA: np.ndarray, opB: np.ndarray) -> np.ndarray:
    """<opA_i opB_{i+1}> for every bond i. Returns array[N-1]."""
    be = _backend_for(mps)
    M = _canon(mps, be)
    opA, opB = be.array(opA), be.array(opB)
    N = len(M)
    out = []
    Lenv = be.ones((1, 1))
    for i in range(N - 1):
        A, B = M[i], M[i + 1]
        # two-site ket block (a, s, t, e); right bond e is identity (right-canonical)
        theta = be.einsum("asc,cte->aste", A, B)
        # apply opA on site i, opB on site i+1 to the ket: (Opsi)_x = sum_s O_{xs} psi_s
        Otheta = be.einsum("xs,yt,aste->axye", opA, opB, theta)
        # <psi|O|psi> = sum Lenv[a,b] Otheta[a,x,y,e] conj(theta[b,x,y,e])
        out.append(be.einsum("ab,axye,bxye->", Lenv, Otheta, be.conj(theta)))
        Lenv = be.einsum("ab,asc,bsd->cd", Lenv, A, be.conj(A))
    return _profile(be, out)


def apply_mpo(mps: MPSLike, mpo: Sequence[Tensor]) -> List[Tensor]:
    """Apply an MPO to an MPS: (O|psi>). Bond dimensions multiply (chi -> chi*w).
    Tensor rule N[(a,k),s,(b,l)] = sum_{s'} W[k,l,s,s'] A[a,s',b].

    The result stays on the MPS's device; the MPO is moved to it."""
    invariants.entry(mps, mpo, ends=False)
    be = _backend_for(mps)
    out = []
    for A, W in zip(mps, mpo):
        Dl, d, Dr = (int(x) for x in A.shape)
        W = be.array(W)
        wl, wr = int(W.shape[0]), int(W.shape[1])
        Nt = be.einsum("klst,atb->aksbl", W, A)
        out.append(Nt.reshape(Dl * wl, d, Dr * wr))
    return out


def magnetization_moments(mps: MPSLike, op: np.ndarray) -> Tuple[float, float, float]:
    """Moments of the total magnetization M = sum_i op_i and the Binder cumulant.

    Uses <M^2> = ||M|psi>||^2 and <M^4> = ||M^2|psi>||^2 (M Hermitian), evaluated
    by applying the magnetization MPO. Returns (m2, m4, U) with the Binder cumulant
    U = 1 - <M^4>/(3 <M^2>^2)."""
    from .models import magnetization_mpo
    N = len(mps)
    Mmpo = magnetization_mpo(N, op)
    Mpsi = apply_mpo(mps, Mmpo)                 # M|psi>   (bond 2*chi)
    M2psi = apply_mpo(Mpsi, Mmpo)              # M^2|psi> (bond 4*chi)
    m2 = mps_overlap(Mpsi, Mpsi).real          # <M^2>
    m4 = mps_overlap(M2psi, M2psi).real        # <M^4>
    U = 1.0 - m4 / (3.0 * m2**2)
    return m2, m4, U


def mps_overlap(mps1: MPSLike, mps2: MPSLike) -> complex:
    """<psi1|psi2> by contracting the transfer network left->right. Both MPS are
    assumed normalised; the magnitude |<psi1|psi2>| is gauge-invariant.

    Both states must already be on the same device."""
    be = _backend_for(mps1)
    E = be.ones((1, 1))
    for A1, A2 in zip(mps1, mps2):
        E = be.einsum("ab,asc,bsd->cd", E, be.conj(A1), A2)
    return complex(be.to_scalar(E[0, 0]))


def two_point(mps: MPSLike, opA: np.ndarray, opB: np.ndarray, i: int, js: Sequence[int],
              canon: Optional[MPSLike] = None) -> np.ndarray:
    """<opA_i opB_j> for each j in `js` (all j > i), propagating one operator
    environment left->right through the intervening identities. Returns array.

    `canon` optionally supplies an already right-canonicalised copy of `mps`
    (as returned by `_canon`). It is a pure performance argument: callers that
    hit the same state many times -- `_structure_factor_pairwise` does it N-1
    times -- would otherwise pay a full O(N) SVD sweep per call.
    `right_canonicalize` is deterministic and `_canon` copies before mutating, so
    passing the cached form is bit-identical to recomputing it.
    """
    be = _backend_for(mps)
    js = sorted(js)
    M = _canon(mps, be) if canon is None else canon
    opA, opB = be.array(opA), be.array(opB)
    Lenv = be.ones((1, 1))                               # norm env for sites < i
    for k in range(i):
        A = M[k]
        Lenv = be.einsum("ab,asc,bsd->cd", Lenv, A, be.conj(A))
    A = M[i]                                             # apply opA at site i
    E = be.einsum("ab,ase,st,btf->ef", Lenv, A, opA, be.conj(A))
    vals = {}
    jset = set(js)
    for k in range(i + 1, js[-1] + 1):
        A = M[k]
        if k in jset:                                    # close with opB at site k
            vals[k] = be.einsum("ef,esg,st,ftg->", E, A, opB, be.conj(A))
        # propagate the environment past site k with identity
        E = be.einsum("ef,esg,fsh->gh", E, A, be.conj(A))
    return _profile(be, [vals[j] for j in js])


def _structure_factor_pairwise(mps: MPSLike, op: np.ndarray) -> Tuple[float, float]:
    """Reference O(N^2) implementation: every <op_i op_j> measured explicitly.

    Kept as the correctness oracle for `structure_factor`, which computes the
    same sum by accumulation. Too slow to use much above N ~ 200.
    """
    be = _backend_for(mps)
    N = len(mps)
    M = _canon(mps, be)              # canonicalise ONCE, not once per two_point
    opd = be.array(op)
    total = float(np.sum(local_expectations(mps, opd @ opd, canon=M)))
    for i in range(N - 1):
        total += 2.0 * float(np.sum(
            two_point(mps, op, op, i, list(range(i + 1, N)), canon=M)))
    return total / N**2, total


def structure_factor(mps: MPSLike, op: np.ndarray) -> Tuple[float, float]:
    """Squared order parameter M^2 = (1/N^2) sum_ij <op_i op_j> and the structure
    factor S = N^2 M^2 = sum_ij <op_i op_j>. Returns (M2, S).

    Diagonal terms <op_i^2> are measured explicitly. The off-diagonal double sum
    is accumulated in ONE left-to-right sweep rather than measured pair by pair:
    carrying

        D_k = sum_{i<k} (environment with op applied at site i, propagated to k)

    alongside the plain norm environment turns sum_{i<j}<op_i op_j> into O(N)
    transfer applications instead of O(N^2). Same estimator, same sum -- only the
    order of accumulation differs, so it matches the pairwise reference to
    round-off (asserted in tests) rather than bit-exactly.

    This matters: the pairwise form costs O(N^2 chi^3), which at N = 600,
    chi = 192 is ~1 h per call -- longer than the DMRG solve it is measuring.
    The single canonical form is shared with the diagonal pass, so the whole
    routine does one canonicalisation sweep, not two.
    """
    be = _backend_for(mps)
    N = len(mps)
    M = _canon(mps, be)
    opd = be.array(op)
    total = float(np.sum(local_expectations(mps, opd @ opd, canon=M)))   # <op_i^2>

    diag = []
    Lnorm = be.ones((1, 1))            # norm env over sites < k
    D = be.zeros((1, 1))               # sum_{i<k} op-env, propagated
    for k in range(N):
        A = M[k]
        # close the accumulated environment with op at site k -> sum_{i<k}<op_i op_k>
        diag.append(be.einsum("ef,esg,st,ftg->", D, A, opd, be.conj(A)))
        # D_{k+1} = (propagate D past k) + (op applied at k on the norm env)
        D = (be.einsum("ef,esg,fsh->gh", D, A, be.conj(A))
             + be.einsum("ab,ase,st,btf->ef", Lnorm, A, opd, be.conj(A)))
        Lnorm = be.einsum("ab,asc,bsd->cd", Lnorm, A, be.conj(A))
    # One transfer for the whole sweep, then the ORIGINAL accumulation order on
    # the host: summing the profile with np.sum instead would reassociate the
    # additions and move the last digits of a published number.
    for v in _profile(be, diag):
        total += 2.0 * float(v)
    return total / N**2, total


def _schmidt_sweep(M: MPSLike, be: Any, upto: int) -> Iterator[np.ndarray]:
    """Yield the Schmidt spectrum at every bond 1..upto of a right-canonical `M`.

    The shared core of `entanglement_entropy` and `all_entropies`: carry the
    orthogonality centre left to right, and the singular values at each cut fall
    out of the SVD that moves it. Yields host (numpy) spectra -- chi floats per
    bond, the one deliberate transfer in the entropy path, which keeps the
    entropy arithmetic bit-identical to what the numpy-only version computed.
    """
    d = int(M[0].shape[1])
    carry = None
    for i in range(upto):
        A = M[i]
        if carry is not None:
            A = be.einsum("ab,bsc->asc", carry, A)
        Dl, _, Dr = (int(x) for x in A.shape)
        U, S, Vh = be.svd(A.reshape(Dl * d, Dr))
        yield np.asarray(be.to_numpy(S))
        carry = be.diag(S) @ Vh


def _entropy(s: np.ndarray) -> float:
    """Von-Neumann entropy of a Schmidt spectrum (normalised here, not by the caller)."""
    s = s / np.linalg.norm(s)
    p = s ** 2
    p = p[p > 1e-16]
    return float(-np.sum(p * np.log(p)))


def entanglement_entropy(mps: MPSLike, bond: int,
                         canon: Optional[MPSLike] = None) -> float:
    """Von-Neumann entropy S = -sum lambda^2 ln lambda^2 of the Schmidt spectrum at
    `bond` (the cut between site bond-1 and site bond).

    For a whole profile call `all_entropies` instead: this routine sweeps from
    site 0 to `bond` every time, so N calls cost O(N^2 chi^3). `canon` shares one
    right-canonical copy across calls and removes the re-canonicalisation, but
    not the quadratic sweep.
    """
    be = _backend_for(mps)
    M = _canon(mps, be) if canon is None else canon
    s = None
    for s in _schmidt_sweep(M, be, bond):
        pass
    return _entropy(s)


def all_entropies(mps: MPSLike, canon: Optional[MPSLike] = None) -> np.ndarray:
    """Every bond entropy in ONE left-to-right sweep. Returns array[N-1].

    `entanglement_entropy(mps, bond)` re-canonicalises and then sweeps from site 0
    to `bond`, so asking for the full profile costs O(N^2 chi^3) -- fine at the
    N <= 64 it was written for, hours at N = 600. The Schmidt spectrum at every
    bond falls out of a single sweep: right-canonicalise once, carry the centre
    left to right, and read the singular values at each cut. Identical arithmetic,
    identical output, O(N chi^3): 33 s against hours at N = 600, chi = 192.

    Checked bond-by-bond against `entanglement_entropy` in
    `tests/test_observables_backend.py::test_all_entropies_bit_identical_to_per_bond`.
    """
    be = _backend_for(mps)
    M = _canon(mps, be) if canon is None else canon
    return np.array([_entropy(s) for s in _schmidt_sweep(M, be, len(M) - 1)])


def two_site_entropy(mps: MPSLike, canon: Optional[MPSLike] = None) -> np.ndarray:
    """S(n) = -Tr rho ln rho of the reduced density matrix of sites (n, n+1), n = 0..N-2.

    Nats, like `all_entropies`; divide by ln 2 for Papaefstathiou et al., PRD 111, 014504
    (2025), Eq. (20). One sweep; the N-1 matrices cross to the host in one transfer.
    Finite open chains only (both outer bonds 1).
    """
    if int(mps[0].shape[0]) != 1 or int(mps[-1].shape[2]) != 1:
        raise ValueError("two_site_entropy needs a finite open chain (outer bonds 1)")
    be = _backend_for(mps)
    M = _canon(mps, be) if canon is None else canon
    rhos = []
    Lenv = be.ones((1, 1))
    for i in range(len(M) - 1):
        th = be.einsum("asc,cte->aste", M[i], M[i + 1])
        rhos.append(be.einsum("ab,aste,bxye->stxy", Lenv, th, be.conj(th)))
        Lenv = be.einsum("ab,asc,bsd->cd", Lenv, M[i], be.conj(M[i]))
    out = []
    for r in np.asarray(be.to_numpy(be.stack_axis(rhos, 0))):
        d2 = r.shape[0] * r.shape[1]
        p = np.linalg.eigvalsh(r.reshape(d2, d2))
        p = p / p.sum()
        p = p[p > 1e-16]
        out.append(float(-np.sum(p * np.log(p))))
    return np.array(out)


def compress(mps: MPSLike, chi_max: int, svd_tol: float = 0.0) -> Tuple[MPS, np.ndarray]:
    """Canonical SVD compression to bond dimension <= `chi_max`.

    A right-canonical copy is swept once left to right with a truncating SVD at each bond.
    ``svd_tol > 0`` keeps the smallest rank whose relative dropped weight is at most
    ``svd_tol`` (the rule `tdvp2_evolve` uses), still capped at `chi_max`. Returns the
    normalised, left-canonical state (`center` N-1) and the relative discarded weight per
    bond, shape (N-1,). The input is not modified.
    """
    invariants.entry(mps, ends=False)
    be = _backend_for(mps)
    M = list(_canon(mps, be))
    N = len(M)
    disc = np.zeros(N - 1)
    for i in range(N - 1):
        Dl, d, Dr = (int(x) for x in M[i].shape)
        A = M[i].reshape(Dl * d, Dr)
        tot = float(be.norm(A)) ** 2
        U, S, Vh, dw = _truncate(A, chi_max, "svd", be)
        disc_rel = dw / tot
        if svd_tol > 0.0 and int(S.shape[0]) > 1:
            Sn = be.to_numpy(S)
            tail = np.cumsum((Sn ** 2)[::-1])[::-1]
            drop = tail / tot + disc_rel
            ok = np.nonzero(drop[1:] <= svd_tol)[0]
            if len(ok):
                k = int(ok[0]) + 1
                disc_rel = float(drop[k])
                U, S, Vh = U[:, :k], S[:k], Vh[:k]
        S = S / be.norm(S)
        k = int(S.shape[0])
        M[i] = U.reshape(Dl, d, k)
        M[i + 1] = be.einsum("ab,bsc->asc", be.diag(S) @ Vh, M[i + 1])
        disc[i] = disc_rel
    M[-1] = M[-1] / be.norm(M[-1])
    return MPS(M, center=N - 1), disc
