"""Exact references: free fermions for the transverse-field Ising energies and the XX quench;
sparse diagonalisation for small XXZ chains; the discarded-weight extrapolation of DMRG
energies."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sl


def tfim_modes(N, g):
    """Single-particle energies (ascending) of H = -sum X X - g sum Z on the open chain."""
    A = np.diag(np.full(N, 2.0 * g)) - np.eye(N, k=1) - np.eye(N, k=-1)
    B = -np.eye(N, k=1) + np.eye(N, k=-1)
    return np.sort(np.sqrt(np.clip(np.linalg.eigvalsh((A - B) @ (A + B)), 0.0, None)))


def tfim_energy(N, g):
    """Ground energy of H = -sum X X - g sum Z, open chain (Jordan-Wigner, exact for any N)."""
    return -0.5 * float(np.sum(tfim_modes(N, g)))


def tfim_levels(N, g, k):
    """The k lowest energies of the open Ising chain, degeneracies counted: the ground energy
    plus sums over sets of excited modes (every fermion parity is physical on an open chain)."""
    from itertools import combinations
    eps = tfim_modes(N, g)
    m = min(N, k + 6)                  # the k lowest sums use only the lowest few modes
    sums = sorted(sum(c) for r in range(m + 1) for c in combinations(eps[:m], r))
    return [tfim_energy(N, g) + float(x) for x in sums[:k]]


def extrapolate(dw, E):
    """E(dw) = E_inf + a dw fitted to every point, and with the smallest bond dimension (largest
    dw) dropped; returns (E_inf, sigma = |difference of the two|)."""
    dw, E = np.asarray(dw, float), np.asarray(E, float)
    o = np.argsort(dw)
    e_all = np.polyfit(dw[o], E[o], 1)[1]
    e_top = np.polyfit(dw[o][:-1], E[o][:-1], 1)[1] if len(dw) > 2 else np.nan
    return float(e_all), float(abs(e_all - e_top))


def xx_neel_sz(N, J, t):
    """<S^z_i(t)> after the Neel state (up on even sites) evolves under
    H = J sum (S^x S^x + S^y S^y), open chain: free fermions hopping with amplitude J/2."""
    h = 0.5 * J * (np.eye(N, k=1) + np.eye(N, k=-1))
    U = sl.expm(-1j * h * t)
    C = U.conj() @ np.diag([1.0 - i % 2 for i in range(N)]) @ U.T
    return np.real(np.diag(C)) - 0.5


def xx_neel_entropy(N, J, t, bond):
    """Entanglement entropy across `bond` (sites < bond on the left) at time t."""
    h = 0.5 * J * (np.eye(N, k=1) + np.eye(N, k=-1))
    U = sl.expm(-1j * h * t)
    C = U.conj() @ np.diag([1.0 - i % 2 for i in range(N)]) @ U.T
    nu = np.clip(np.linalg.eigvalsh(C[:bond, :bond]), 1e-300, 1 - 1e-16)
    return float(-np.sum(nu * np.log(nu) + (1 - nu) * np.log1p(-nu)))


def xxz_energy(N, J, Jz, pairs=None):
    """Ground energy of H = J sum (S^x S^x + S^y S^y) + Jz sum S^z S^z over `pairs` (default:
    the open chain), by sparse exact diagonalisation (N up to about 20)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    sp_ = sp.csr_matrix([[0.0, 1.0], [0.0, 0.0]])
    sz = sp.csr_matrix(np.diag([0.5, -0.5]))

    def op(o, i):
        return sp.kron(sp.kron(sp.identity(2 ** i), o), sp.identity(2 ** (N - i - 1)), "csr")

    H = sp.csr_matrix((2 ** N, 2 ** N))
    for i, j in pairs or [(i, i + 1) for i in range(N - 1)]:
        H = H + 0.5 * J * (op(sp_, i) @ op(sp_.T, j) + op(sp_.T, i) @ op(sp_, j)) \
            + Jz * op(sz, i) @ op(sz, j)
    return float(eigsh(H, k=1, which="SA", tol=1e-14)[0][0])
