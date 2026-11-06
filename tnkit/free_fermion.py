"""Exact free-fermion (Jordan-Wigner) ground energy of the open TFIM chain.

H = - sum_{i=1}^{N-1} sigma^x_i sigma^x_{i+1}  -  g sum_{i=1}^{N} sigma^z_i

maps to a quadratic fermion Hamiltonian. In Lieb-Schultz-Mattis form
    H = sum_ij c_i^dag A_ij c_j + (1/2) sum_ij (c_i^dag B_ij c_j^dag + h.c.) + C
the ground energy is  E0 = C + (1/2) Tr(A) - (1/2) sum_k Lambda_k,
with Lambda_k = sqrt(eigenvalues of (A-B)(A+B)).

For this TFIM: A = T (diag 2g, nearest-neighbour -1), B = Delta (antisymmetric,
B_{i,i+1} = -1), C = -g*N, which gives the compact result  E0 = -(1/2) sum_k Lambda_k.
Exact for ANY N — the reference for validating DMRG beyond the reach of exact
diagonalisation.
"""
from __future__ import annotations
import numpy as np


def tfim_free_fermion_energy(N, g):
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 2.0 * g
    for i in range(N - 1):
        A[i, i + 1] = A[i + 1, i] = -1.0        # symmetric hopping
        B[i, i + 1] = -1.0                       # antisymmetric pairing
        B[i + 1, i] = +1.0
    M = (A - B) @ (A + B)
    w = np.linalg.eigvalsh(0.5 * (M + M.T))      # symmetrise for stability
    Lam = np.sqrt(np.clip(w, 0.0, None))
    return -0.5 * float(np.sum(Lam))


# Thermodynamic-limit ground-state energy density at the critical point (g=1):
#   e_inf = -(1/pi) * int_0^pi sqrt(2 + 2 cos k) dk = -4/pi
E_DENSITY_CRITICAL = -4.0 / np.pi   # ~= -1.2732395
