"""TEBD — Time-Evolving Block Decimation for real-time MPS dynamics.

Trotterise exp(-i H dt) into two-site gates and apply them along the chain,
truncating the bond dimension after each gate. This is contraction + SVD, so it
runs on the same backend as DMRG (GPU-ready). Time evolution grows entanglement
(the "entanglement barrier"), so it is exactly the regime that needs large chi ->
where the GPU pays off.

Trotter scheme: a symmetric (2nd-order) product built from a palindromic sweep.
Each bond carries a half-step gate exp(-i h_{i,i+1} dt/2); a forward sweep
(bonds 0..N-2) followed by a backward sweep (N-2..0) applies every bond twice in a
palindromic order -> 2nd-order accurate. The orthogonality centre is tracked so
each SVD truncation is the optimal (Schmidt) one.

For the TFIM  H = -sum sx_i sx_{i+1} - g sum sz_i,  the on-site field is split
between a site's two bonds (edge sites get full weight on their single bond) so the
bond gates reproduce H exactly.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm

from .models import SX, SZ, I2
from .backend import get_backend
from ._dmrg import right_canonicalize


def tfim_bond_gates(N, g, dt, imaginary=False):
    """Half-step two-site gates, one per bond (N-1 total), as (2,2,2,2) tensors
    G[out_i,out_j,in_i,in_j].

    imaginary=False -> exp(-i h dt/2), REAL-time evolution. Intrinsically complex.
    imaginary=True  -> exp(-h tau/2),  IMAGINARY-time evolution. The TFIM bond Hamiltonian is
                       real symmetric, so these gates are REAL -- imaginary-time TEBD needs no
                       complex arithmetic anywhere. It is not
                       norm-preserving, so `tebd_evolve` renormalises after each step; the
                       fixed point is the ground state, which makes it independently gateable
                       against the free-fermion energy.
    """
    def bond_h(wl, wr):
        return (-np.kron(SX, SX)
                - g * (wl * np.kron(SZ, I2) + wr * np.kron(I2, SZ)))
    gates = []
    for i in range(N - 1):
        wl = 1.0 if i == 0 else 0.5            # site i field weight
        wr = 1.0 if i == N - 2 else 0.5        # site i+1 field weight
        h = bond_h(wl, wr)
        if imaginary:
            G = expm(-np.asarray(h).real * dt / 2.0).real.reshape(2, 2, 2, 2)
        else:
            G = expm(-1j * h * dt / 2.0).reshape(2, 2, 2, 2)
        gates.append(G)
    return gates


def bond_gates(bond_h, dt, imaginary=False):
    """Half-step gates from per-bond Hamiltonians: each ``h`` is ``(d*d, d*d)`` on
    ``(s_i s_j)`` or ``(d, d, d, d)`` as [s_i', s_j', s_i, s_j] (the `vumps` bond_h layout),
    and the gate is ``exp(-i h dt/2)`` (``exp(-h tau/2)`` if `imaginary`), reshaped to
    ``(d, d, d, d)`` as G[out_i, out_j, in_i, in_j]."""
    gates = []
    for h in bond_h:
        h = np.asarray(h)
        if h.ndim == 4:
            h = h.reshape(h.shape[0] * h.shape[1], -1)
        d = int(round(np.sqrt(h.shape[0])))
        if h.ndim != 2 or h.shape != (d * d, d * d):
            raise ValueError(f"bond Hamiltonian of shape {h.shape}; expected (d*d, d*d) or "
                             f"(d, d, d, d)")
        G = expm(-h * dt / 2.0) if imaginary else expm(-1j * h * dt / 2.0)
        gates.append(G.reshape(d, d, d, d))
    return gates


def _apply_gate(M, i, G, chi, be, moving):
    """Apply gate G at bond (i,i+1) with the orthogonality centre there, SVD-truncate
    to chi, and move the centre ('right' -> to i+1, 'left' -> to i)."""
    theta = be.einsum("asc,ctb->astb", M[i], M[i + 1])     # (a, si, sj, b)
    theta = be.einsum("xyij,aijb->axyb", G, theta)         # apply the gate
    a, si, sj, b = (int(x) for x in theta.shape)
    U, S, Vh = be.svd(theta.reshape(a * si, sj * b))
    k = min(chi, int(S.shape[0]))
    U, S, Vh = U[:, :k], S[:k], Vh[:k]
    S = S / be.norm(S)                                     # gate is unitary -> renormalise
    if moving == "right":
        M[i] = U.reshape(a, si, k)
        M[i + 1] = (S[:, None] * Vh).reshape(k, sj, b)
    else:
        M[i] = (U * S).reshape(a, si, k)
        M[i + 1] = Vh.reshape(k, sj, b)


def _renormalise(M, be, centre=0):
    """Imaginary-time evolution is not norm-preserving; rescale the ORTHOGONALITY CENTRE.

    The backward sweep ends with the centre at site 0, so that is the tensor carrying the norm --
    normalising any other site leaves <psi|psi> != 1 and silently halves the measured energy.
    """
    nrm = be.to_scalar(be.norm(M[centre])).real
    if nrm > 0:
        M[centre] = M[centre] / nrm
    return M


def tebd_evolve(mps, N, g, chi, dt, n_steps, backend="numpy",
                measure=None, meas_every=1, imaginary=False, *, bond_h=None):
    """Evolve `mps` under the TFIM at field g for n_steps of size dt (2nd-order
    Trotter). Returns (final_mps, records) where records is a list of
    (time, measure(mps_numpy)) sampled every `meas_every` steps.

    `bond_h` replaces the TFIM: a list of N-1 two-site Hamiltonians, each ``(d*d, d*d)`` or
    ``(d, d, d, d)`` (see `bond_gates`), whose sum is H (split any on-site terms between bonds
    yourself). Pass ``g=None`` with it.

    `measure` receives the current MPS as a list of numpy tensors (observables are
    numpy-based); pass None to skip measurement.
    """
    be = get_backend(backend)
    M = right_canonicalize([be.array(m) for m in mps], be)
    if bond_h is None:
        if g is None:
            raise ValueError("tebd_evolve needs g (the TFIM field) or bond_h")
        gates = tfim_bond_gates(N, g, dt, imaginary=imaginary)
    else:
        if g is not None:
            raise ValueError("tebd_evolve: pass g=None with bond_h; the gates come from bond_h")
        if len(bond_h) != N - 1:
            raise ValueError(f"bond_h has {len(bond_h)} bonds; a chain of {N} sites has {N - 1}")
        gates = bond_gates(bond_h, dt, imaginary=imaginary)
    gates = [be.array(G) for G in gates]

    def snapshot():
        return [be.to_numpy(m) for m in M]

    records = []
    if measure is not None:
        records.append((0.0, measure(snapshot())))
    for step in range(n_steps):
        for i in range(N - 1):                 # forward sweep (centre 0 -> N-1)
            _apply_gate(M, i, gates[i], chi, be, "right")
        for i in range(N - 2, -1, -1):         # backward sweep (centre N-1 -> 0)
            _apply_gate(M, i, gates[i], chi, be, "left")
        if imaginary:
            _renormalise(M, be)
        if measure is not None and (step + 1) % meas_every == 0:
            records.append(((step + 1) * dt, measure(snapshot())))
    return M, records
