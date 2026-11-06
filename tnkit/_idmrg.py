"""Infinite DMRG (iDMRG) — thermodynamic-limit ground state via a growing 2-site
unit cell. Reuses the matrix-free Lanczos + MPO environment machinery, so it is
backend-agnostic like the finite kernel.

Why: the thermodynamic limit is the *only* clean setting for finite-entanglement
scaling. On a finite chain the half-chain entropy is capped by min[(c/6)ln N,
(cκ/6)ln χ], so the χ-limited (FES) branch is only visible for N > χ^κ ≈ χ² — which
means huge N (→ overflow in the length-N canonicalisation, and cost). iDMRG has no
system size: the fixed-point iMPS at bond χ directly realises ξ(χ)~χ^κ and
S(χ)~(cκ/6)ln χ.

Algorithm (infinite-system DMRG, White): grow the chain two sites per step by
inserting a 2-site block in the middle, solving its ground state against the
accumulated left/right half-chain environments, truncating to χ, and folding the
new tensors into the environments. The per-site energy e = (E_n − E_{n−1})/2
converges to e₀; the central Schmidt spectrum converges to the bulk fixed point.

Accuracy note: pure-growth infinite-system DMRG reaches the bulk fixed point
accurately in the **gapped** phase (validated: e₀ to ~1e-4 at χ=24), but converges
slowly and sub-optimally **at criticality** (the environments and unit cell
co-converge without a wavefunction transformation). So it gives the *qualitative*
finite-entanglement scaling (S grows ~log χ) but not a clean quantitative c/κ — that
needs a variationally-optimal iMPS (VUMPS or McCulloch iDMRG).
"""
from __future__ import annotations
import dataclasses
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np

from .models import I2, SX, SZ
from .eigensolver import lanczos_ground
from ._dmrg import apply_two_site, update_left_env, update_right_env
from .backend import get_backend
from . import _options


def tfim_bulk_mpo(g):
    """Bulk (translation-invariant) TFIM MPO tensor, same 3×3 automaton as tfim_mpo."""
    W = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    W[0, 0] = I2; W[0, 1] = SX; W[0, 2] = -g * SZ; W[1, 2] = -SX; W[2, 2] = I2
    return W


def _entropy(svals):
    p = svals ** 2
    p = p[p > 1e-16]
    return float(-np.sum(p * np.log(p)))


def _xi_from_tensor(A):
    """Correlation length from the transfer matrix of a bulk tensor A=(a,s,b)."""
    a, s, b = A.shape
    T = np.einsum("asb,ASB->aAbB", A, A.conj()).reshape(a * a, b * b)
    ev = np.abs(np.linalg.eigvals(T)); ev.sort(); ev = ev[::-1]
    if ev.size < 2 or ev[1] <= 0:
        return np.nan
    return -1.0 / np.log(ev[1] / ev[0])


@dataclasses.dataclass
class IDMRGOptions:
    """Settings of `idmrg`, with the defaults of its old flat signature."""
    n_iter: int = 400
    tol: float = 1e-12
    seed: int = 0
    backend: Any = "numpy"
    verbose: bool = False


def idmrg(mpo_or_g: Any, chi: int,
          options: Optional[Union[IDMRGOptions, Dict[str, Any]]] = None,
          **legacy_kwargs: Any) -> Dict[str, Any]:
    """Infinite-system DMRG for a bulk MPO tensor, or for the TFIM at field `g`.

    `mpo_or_g` is either a number (the TFIM field) or a rank-4 bulk tensor
    ``(D, D, d, d)`` whose channel 0 starts a term and channel ``D-1`` ends one, as in
    `tfim_bulk_mpo`. Settings go in `options` or as flat keywords (`n_iter`, `tol`, `seed`,
    `backend`, `verbose`); `be=` is a deprecated alias of `backend`.

    Returns dict with e0 (energy/site), S (entanglement), xi (correlation length),
    svals, iters, e_hist, and the converged bulk tensors A,B."""
    if "be" in legacy_kwargs:
        if "backend" in legacy_kwargs:
            raise TypeError("idmrg() got both 'be' and 'backend'")
        warnings.warn("idmrg(be=...) is deprecated; use backend=...", DeprecationWarning,
                      stacklevel=2)
        legacy_kwargs["backend"] = legacy_kwargs.pop("be")
    o = _options.resolve(IDMRGOptions, options, legacy_kwargs, "idmrg")
    return _idmrg(mpo_or_g, chi, o.n_iter, o.tol, o.seed, get_backend(o.backend), o.verbose)


def _idmrg(mpo_or_g, chi, n_iter, tol, seed, be, verbose):
    nd = np.ndim(mpo_or_g)
    if nd not in (0, 4):
        raise TypeError("idmrg: mpo_or_g must be a number (the TFIM field) or a (D, D, d, d) "
                        "bulk tensor; for a finite MPO list use dmrg")
    W0 = mpo_or_g if nd == 4 else tfim_bulk_mpo(mpo_or_g)
    W = be.array(W0)
    D, d = int(W.shape[0]), int(W.shape[2])
    L = np.zeros((1, D, 1), dtype=np.complex128); L[0, 0, 0] = 1.0; L = be.array(L)       # vL
    R = np.zeros((1, D, 1), dtype=np.complex128); R[0, D - 1, 0] = 1.0; R = be.array(R)   # vR
    rng = np.random.default_rng(seed)
    theta = be.array(rng.standard_normal((1, d, d, 1)) + 1j * rng.standard_normal((1, d, d, 1)))

    E_prev = None
    e_hist = []
    S_ent = np.nan
    svals = None
    A = B = None
    for it in range(n_iter):
        shape = tuple(int(x) for x in theta.shape)          # (aL,d,d,aR)
        aL, _, _, aR = shape

        def matvec(x):
            return apply_two_site(x.reshape(shape), L, W, W, R, be).reshape(-1)

        v0 = theta.reshape(-1); v0 = v0 / be.norm(v0)
        E, v, _ = lanczos_ground(matvec, v0, be, tol=1e-11,
                                 max_iter=max(200, 4 * chi), reorth="full")
        v = v.reshape(shape)

        U, S, Vh = be.svd(v.reshape(aL * d, d * aR))
        k = min(chi, int(S.shape[0]))
        U, S, Vh = U[:, :k], S[:k], Vh[:k]
        sv = be.to_numpy(S).real; sv = sv / np.linalg.norm(sv)
        S_ent = _entropy(sv); svals = sv
        A = U.reshape(aL, d, k); B = Vh.reshape(k, d, aR)

        if E_prev is not None:
            e_hist.append((E - E_prev) / 2.0)
        E_prev = E

        L = update_left_env(L, A, W, be)                    # (k,D,k)
        R = update_right_env(R, B, W, be)                   # (k,D,k)

        # warm-start next 2-site block: reuse when the shape is stable (k saturated),
        # else a fresh random seed of the right size.
        if k == aL == aR:
            SB = be.array(np.diag(sv).astype(np.complex128)) @ B.reshape(k, -1)
            theta = be.einsum("aib,bjc->aijc", A, SB.reshape(k, d, aR))
        else:
            theta = be.array(rng.standard_normal((k, d, d, k))
                             + 1j * rng.standard_normal((k, d, d, k)))

        if verbose and it % 20 == 0 and e_hist:
            print(f"  it={it} k={k} e={e_hist[-1]:.10f} S={S_ent:.4f}", flush=True)
        if len(e_hist) >= 4 and k == chi and abs(e_hist[-1] - e_hist[-2]) < tol:
            break

    An = be.to_numpy(A)
    return {"e0": e_hist[-1] if e_hist else np.nan, "S": S_ent,
            "xi": _xi_from_tensor(An), "svals": svals, "iters": it + 1,
            "e_hist": e_hist, "A": An, "B": be.to_numpy(B)}


def mixed_field_bulk_mpo(g, h):
    """Bulk MPO for the MIXED-FIELD Ising chain

        H = - sum_i sx_i sx_{i+1}  -  g sum_i sz_i  -  h sum_i sx_i

    A longitudinal field h breaks integrability: the Jordan-Wigner transformation no longer
    gives a quadratic fermion Hamiltonian, so there is NO exact epsilon(k) to compare against.
    That is precisely the point -- it is the first model where the tangent-space excitation
    machinery (tnkit.excitations) is the only source of a dispersion relation.

    Same 3x3 automaton as tfim_bulk_mpo, with the longitudinal field added to the on-site slot.
    """
    W = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    W[0, 0] = I2; W[0, 1] = SX; W[0, 2] = -g * SZ - h * SX; W[1, 2] = -SX; W[2, 2] = I2
    return W


def mixed_field_bond_h(g, h):
    """Two-site bond operator for the mixed-field Ising chain, (d,d,d,d) as [s',t',s,t].

    Each site's field is attributed wholly to the bond on its right, which is exact for a
    uniform infinite chain (every site owns exactly one such bond).
    """
    hb = -np.kron(SX, SX) - g * np.kron(SZ, I2) - h * np.kron(SX, I2)
    # hb is (s1 s2) x (t1 t2); reshaping gives [s1_out, s2_out, s1_in, s2_in] directly, which is
    # the "stuv" layout _bond_energy_density contracts. Do NOT transpose -- an earlier version
    # did, producing e0 = -1.4252 against the exact -1.6719 at g=1.5, h=0.
    return hb.reshape(2, 2, 2, 2)


# --------------------------------------------------------------------------- #
# Symanzik-improved (tree-level) mixed-field Ising chain
# --------------------------------------------------------------------------- #
#
# WHY. The lattice realisation of Ising Field Theory carries an irrelevant-operator
# artefact.  For the 2d Ising model in a magnetic field, Caselle-Grinza-Magnoli
# (hep-th/0103263) identify the spin-4, dimension-4 operator T^2 + Tbar^2 -- the one that
# breaks rotational invariance, i.e. a *pure lattice artefact* -- as the DOMINANT
# irrelevant correction.  Poghosyan (arXiv:1908.06291) writes that perturbation as
# H_int = g I_3 + gbar Ibar_3, the third KdV charges, and gives its coupling explicitly in
# terms of a lattice anisotropy.  TTbar, the other dimension-4 candidate, is spin-0 and a
# TTbar deformation leaves infinite-volume masses invariant (it only adds a CDD factor to
# the S-matrix), so mass RATIOS can only be contaminated by T^2 + Tbar^2.
#
# THE COUNTERTERM.  On a one-particle state the charge I_3 + Ibar_3 has eigenvalue ~ k^3,
# so the coupling of T^2 + Tbar^2 in the lattice Hamiltonian is read off directly as the
# k^3 coefficient of the one-particle dispersion at the g_z = 0 critical point.  For the
# bare chain eps(k) = 4|sin(k/2)| = 2k - k^3/12: nonzero, hence the artefact.  Adding the
# three-site term  -lam * sum_j sx_j sz_{j+1} sx_{j+2}  keeps the g_z = 0 model exactly
# free-fermionic (Jordan-Wigner: sx_j sz_{j+1} sx_{j+2} = B_j A_{j+2}, quadratic) with
#
#     eps(k) = 2|f(k)|,   f(k) = -g + e^{ik} + lam e^{2ik},
#     |f(k)|^2 = (1+2lam)^2 k^2 - (1/12)(1 + 16 lam + 16 lam^2) k^4 + O(k^6),
#
# critical at g = 1 + lam (gap closes at k = 0).  The k^4 coefficient -- i.e. the
# T^2 + Tbar^2 coupling -- vanishes at
#
#     lam* = (sqrt(3) - 2)/4 = -0.06698729810778...,   g_c = 1 + lam* = (2 + sqrt(3))/4,
#
# where the sound velocity is exactly c = 2(1 + 2 lam*) = sqrt(3) and the leading residual
# is the spin-6 dimension-6 charge I_5 + Ibar_5:  |f|^2 = (3/4)k^2 - k^6/120.
# This is TREE-LEVEL Symanzik improvement: the coefficient is fixed once, at g_z = 0, and
# then used unchanged across the whole g_z grid.

SYMANZIK_LAMBDA = (np.sqrt(3.0) - 2.0) / 4.0     # = -0.0669872981077807


def improved_critical_g(lam):
    """Exact critical transverse field of the improved chain at h = 0: g_c = 1 + lam.

    The h = 0 model is free-fermionic with f(k) = -g + e^{ik} + lam e^{2ik}; the gap
    closes at k = 0 when g = 1 + lam.  For |lam| small this is the only gap closing
    (d^2|f|^2/dk^2 |_0 = 2g + 8 g lam - 2 lam > 0), so it is the Ising critical point.
    Sitting exactly here is what makes eta_latt = 0, i.e. the pure E8 point.
    """
    return 1.0 + lam


def improved_dispersion_exact(k, g, lam):
    """Exact one-particle dispersion of the h = 0 improved chain, eps(k) = 2|f(k)|."""
    k = np.asarray(k, dtype=float)
    f = -g + np.exp(1j * k) + lam * np.exp(2j * k)
    return 2.0 * np.abs(f)


def improved_ising_bulk_mpo(g, h, lam):
    """Bulk MPO (bond dimension 5) for the tree-level Symanzik-improved mixed-field chain

        H = - sum_i sx_i sx_{i+1} - g sum_i sz_i - h sum_i sx_i
            - lam sum_i sx_i sz_{i+1} sx_{i+2}

    Automaton states: 0 = start, 1 = nearest-neighbour carry, 2/3 = the two stages of the
    three-site carry, 4 = done.  Strictly upper triangular with transitions only to higher
    indices (0->1, 0->2, 2->3, 1->4, 3->4), which is what the VUMPS environment recursions
    require: they fill intermediate channels in a single increasing (left) / decreasing
    (right) pass.  lam = 0 reproduces mixed_field_bulk_mpo exactly (channels 2,3 dead).
    """
    W = np.zeros((5, 5, 2, 2), dtype=np.complex128)
    W[0, 0] = I2
    W[0, 1] = SX                      # open the nearest-neighbour bond
    W[0, 2] = SX                      # open the three-site term
    W[0, 4] = -g * SZ - h * SX        # on-site fields
    W[1, 4] = -SX                     # close the nearest-neighbour bond
    W[2, 3] = SZ                      # middle sz of the three-site term
    W[3, 4] = -lam * SX               # close the three-site term
    W[4, 4] = I2
    return W


def improved_triple_h(g, h, lam):
    """Three-site block operator, shape (2,2,2,2,2,2) as [s,t,u out | s,t,u in].

    Each site owns exactly one triple (the one it starts), so attributing its fields and
    its right-going bond to that triple is exact for a uniform infinite chain -- the same
    bookkeeping mixed_field_bond_h uses for two-site blocks.  Used only as the VUMPS
    energy READOUT (the MPO energy channel carries a harmless additive normalisation).
    """
    kron3 = lambda a, b, c: np.kron(np.kron(a, b), c)
    H = (-kron3(SX, SX, I2) - g * kron3(SZ, I2, I2) - h * kron3(SX, I2, I2)
         - lam * kron3(SX, SZ, SX))
    return H.reshape(2, 2, 2, 2, 2, 2)
