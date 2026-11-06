"""Finite-volume (Luscher) analysis for 1+1D spin chains.

Everything here is pure numpy and independent of the tensor-network stack, so it can be
unit-tested against exact free-fermion results without touching DMRG.

Conventions
-----------
Two identical particles of mass ``m`` on a periodic ring of ``L`` sites, total momentum
zero, single-particle momenta ``+k`` and ``-k``.  The two-body S-matrix is written

    S(k) = exp(2 i delta(k))

and the finite-volume quantisation condition (the 1+1D collapse of Luscher's zeta
function; equivalently the two-particle Bethe equation on a ring) is

    exp(i k L) S(k) = 1        <=>        k L + 2 delta(k) = 2 pi n ,   n integer.

Free fermions have ``S = -1`` identically, hence ``delta = pi/2`` and
``k = (2n-1) pi / L`` -- exactly the antiperiodic (Neveu-Schwarz) momentum grid of the
even-fermion-parity sector of a periodic Ising chain.  That identity is a validation
check: it is exact at finite ``L`` and finite lattice spacing, so any deviation is
bookkeeping error, not physics.

E8 (Zamolodchikov 1989) elastic A1 A1 -> A1 A1 amplitude:

    S(theta) = prod_j (sinh theta + i sin alpha_j) / (sinh theta - i sin alpha_j),
    alpha_1 = 2 pi / 3,  alpha_2 = 2 pi / 5,  alpha_3 = 2 pi / 30 = pi / 15.

Equivalent to the standard block form S_11 = f_{2/3} f_{2/5} f_{1/15} with
f_x(theta) = tanh((theta + i pi x)/2) / tanh((theta - i pi x)/2); the identity
tanh(A)/tanh(B) = (sinh(A+B) + sinh(A-B))/(sinh(A+B) - sinh(A-B)) with A+B = theta,
A-B = i pi x maps one onto the other.  The pole at theta = i alpha_j corresponds to a
bound state of mass m_j = 2 m_1 cos(alpha_j / 2), giving m_1/m_1 = 1,
m_2/m_1 = 2cos(pi/5) = 1.6180 (golden ratio) and m_3/m_1 = 2cos(pi/30) = 1.98904.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "E8_ALPHAS",
    "e8_masses",
    "e8_delta_theta",
    "e8_delta_k",
    "e8_kappa_over_m1",
    "ff_dispersion",
    "ff_ns_momenta",
    "ff_r_momenta",
    "ff_ring_spectrum_p0",
    "luscher_delta",
    "luscher_levels_from_delta",
    "k_from_energy_relativistic",
    "k_from_energy_lattice",
    "theta_from_k",
    "luscher_delta_mod_pi",
    "e8_S_imag_rapidity",
    "e8_bound_state_finite_volume",
]

# --- E8 exact S-matrix -------------------------------------------------------------

#: CDD pole locations in rapidity for A1 A1 -> A1 A1 in the E8 theory.
E8_ALPHAS = (2.0 * np.pi / 3.0, 2.0 * np.pi / 5.0, 2.0 * np.pi / 30.0)


def e8_masses() -> np.ndarray:
    """The three sub-threshold E8 masses in units of m1, from the CDD poles."""
    return np.array([2.0 * np.cos(a / 2.0) for a in E8_ALPHAS])


def e8_kappa_over_m1() -> float:
    """Binding momentum of the m3 pole in units of m1: kappa = m1 sin(alpha_3 / 2).

    Follows from m3 = 2 m1 cos(alpha_3/2) and m3 = 2 sqrt(m1^2 - kappa^2).
    """
    return float(np.sin(E8_ALPHAS[2] / 2.0))


def e8_delta_theta(theta, alphas=E8_ALPHAS) -> np.ndarray:
    """Elastic phase shift delta(theta), with S = exp(2 i delta).

    Branch: each CDD factor contributes ``arctan(sin(alpha)/sinh(theta))`` taken in
    ``(0, pi)``, so delta -> (number of factors) * pi/2 at threshold (theta -> 0+) and
    delta -> 0 as theta -> infinity.  With three factors delta(0+) = 3 pi / 2 and
    S(0) = exp(3 i pi) = -1, the fermionic value.
    """
    theta = np.asarray(theta, dtype=float)
    out = np.zeros_like(theta)
    for a in alphas:
        out = out + np.arctan2(np.sin(a), np.sinh(theta))
    return out


def theta_from_k(k, m1):
    """Rapidity difference of two particles of mass m1 with momenta +-k.

    Each particle has rapidity +-theta/2, so k = m1 sinh(theta/2).
    """
    return 2.0 * np.arcsinh(np.asarray(k, dtype=float) / m1)


def e8_delta_k(k, m1=1.0, alphas=E8_ALPHAS):
    """E8 phase shift as a function of the relative momentum k (each particle +-k)."""
    return e8_delta_theta(theta_from_k(k, m1), alphas=alphas)


# --- free fermions on a ring -------------------------------------------------------


def ff_dispersion(k, gx: float) -> np.ndarray:
    """Lattice free-fermion dispersion of H = -sum(sz sz + gx sx), eps(k) = 2|...|.

    eps(k) = 2 sqrt(1 + gx^2 - 2 gx cos k).  At gx = 1 this is 4|sin(k/2)| -> 2|k|,
    i.e. an emergent speed of light c = 2 in lattice units.
    """
    k = np.asarray(k, dtype=float)
    return 2.0 * np.sqrt(1.0 + gx**2 - 2.0 * gx * np.cos(k))


def ff_ns_momenta(N: int) -> np.ndarray:
    """Neveu-Schwarz (antiperiodic fermion) momenta of an N-site periodic Ising ring.

    These label the even-fermion-parity sector, P = prod sigma^x = +1, which contains
    the ground state (for gx > 1) and all two-particle states.
    """
    n = np.arange(N)
    return (2.0 * np.pi * (n + 0.5) / N) - np.pi  # in (-pi, pi)


def ff_r_momenta(N: int) -> np.ndarray:
    """Ramond (periodic fermion) momenta -- odd-parity sector, contains 1-particle states."""
    n = np.arange(N)
    k = 2.0 * np.pi * n / N
    return np.where(k > np.pi, k - 2.0 * np.pi, k)


def ff_ring_spectrum_p0(N: int, gx: float, nlevels: int = 8):
    """Exact zero-total-momentum, even-parity spectrum of the periodic TFIM ring.

    Returns ``(E0, gaps, ks)`` where ``E0`` is the NS vacuum energy, ``gaps[i]`` is the
    excitation energy of the i-th two-particle state above the vacuum, and ``ks[i]`` is
    the single-particle momentum (the pair carries +k and -k).
    """
    kNS = ff_ns_momenta(N)
    E0 = -0.5 * float(np.sum(ff_dispersion(kNS, gx)))
    kpos = np.sort(kNS[kNS > 0])[:nlevels]
    gaps = 2.0 * ff_dispersion(kpos, gx)
    return E0, gaps, kpos


# --- the Luscher inversion ---------------------------------------------------------


def luscher_delta(k, L, n):
    """delta = (2 pi n - k L) / 2 from the quantisation condition k L + 2 delta = 2 pi n."""
    return (2.0 * np.pi * np.asarray(n, dtype=float) - np.asarray(k, dtype=float) * L) / 2.0


def luscher_delta_mod_pi(k, L):
    """Branch-free observable: delta modulo pi, needing no level-number assignment.

    ``exp(i k L) S(k) = 1`` fixes ``2 delta = -k L`` modulo ``2 pi``, i.e. delta is
    determined modulo pi with no integer input at all.  This is the assumption-free
    form of the measurement.
    """
    return np.mod(-np.asarray(k, dtype=float) * L / 2.0, np.pi)


def luscher_levels_from_delta(delta_fn, L, m1, nmax=8, kmax=None, ngrid=200001):
    """Solve k L + 2 delta(k) = 2 pi n for the lowest ``nmax`` roots.

    ``delta_fn(k)`` must return the phase shift for relative momentum ``k``.
    Root-finding is done by sign changes on a dense grid then bisection, which is
    robust for the monotone-ish functions involved here.
    """
    kmax = kmax if kmax is not None else 4.0 * m1
    kg = np.linspace(1e-12, kmax, ngrid)
    f = kg * L + 2.0 * np.asarray(delta_fn(kg), dtype=float)
    roots = []
    ns = []
    # target values 2 pi n that are bracketed by f
    nlo = int(np.floor(f[0] / (2.0 * np.pi))) + 1
    n = nlo
    while len(roots) < nmax and n < nlo + 4 * nmax + 20:
        target = 2.0 * np.pi * n
        g = f - target
        idx = np.where(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
        if idx.size:
            i = idx[0]
            a, b = kg[i], kg[i + 1]
            for _ in range(80):
                c = 0.5 * (a + b)
                gc = c * L + 2.0 * float(delta_fn(np.array([c]))[0]) - target
                ga = a * L + 2.0 * float(delta_fn(np.array([a]))[0]) - target
                if np.sign(gc) == np.sign(ga):
                    a = c
                else:
                    b = c
            roots.append(0.5 * (a + b))
            ns.append(n)
        n += 1
    return np.array(roots), np.array(ns)


def k_from_energy_relativistic(E2, m1, c=2.0):
    """k from a two-particle energy using the continuum dispersion E = 2 sqrt(m1^2+c^2 k^2).

    ``c`` is the emergent speed of light **in lattice units**.  For the Ising chain the
    critical free-fermion dispersion is eps(k) = 4|sin(k/2)| -> 2|k|, so ``c = 2``
    exactly; a measurement in the interacting scaling window gave c = 1.9964.  Getting
    this factor wrong rescales every extracted momentum by ``c`` and therefore ruins the
    Luscher inversion -- the ``k L`` term is a *lattice* momentum times a site count.
    """
    E2 = np.asarray(E2, dtype=float)
    val = (E2 / 2.0) ** 2 - m1**2
    return np.sign(val) * np.sqrt(np.abs(val)) / c


def k_from_energy_lattice(E2, disp, kmax=np.pi, ngrid=200001, newton=4):
    """k from a two-particle energy E2 = 2 eps(k), inverting a numerical dispersion.

    ``disp(k)`` must be monotone increasing on ``[0, kmax]`` (true for the Ising chain
    in the paramagnetic phase).  A dense-grid bracket is followed by Newton refinement
    with a central-difference derivative, giving ~1e-13 accuracy.
    """
    kg = np.linspace(0.0, kmax, ngrid)
    eg = 2.0 * np.asarray(disp(kg), dtype=float)
    E2 = np.asarray(E2, dtype=float)
    k = np.interp(E2, eg, kg)
    h = 1e-6
    for _ in range(newton):
        f = 2.0 * np.asarray(disp(k), dtype=float) - E2
        df = (np.asarray(disp(k + h), dtype=float) - np.asarray(disp(k - h), dtype=float)) / h
        k = np.clip(k - f / df, 0.0, kmax)
    return k


# --- bound state in finite volume ---------------------------------------------------


def e8_S_imag_rapidity(alpha, alphas=E8_ALPHAS):
    """S(theta = i alpha), which is real: prod_j (sin alpha + sin alpha_j)/(sin alpha - sin alpha_j)."""
    alpha = np.asarray(alpha, dtype=float)
    out = np.ones_like(alpha)
    for a in alphas:
        out = out * (np.sin(alpha) + np.sin(a)) / (np.sin(alpha) - np.sin(a))
    return out


def e8_bound_state_finite_volume(L, m1=1.0, j=2, alphas=E8_ALPHAS, tol=1e-15):
    """Finite-volume mass of the j-th E8 bound state on a ring of circumference L.

    Analytic continuation of ``exp(i k L) S(k) = 1`` to ``k = i kappa`` with
    ``kappa = m1 sin(alpha/2)`` and ``theta = i alpha`` gives

        exp(-kappa L) S(i alpha) = 1   <=>   S(i alpha) = exp(m1 L sin(alpha/2)) .

    Solved by bisection for ``alpha`` just above the pole at ``alpha_j``; the mass is
    ``E(L) = 2 m1 cos(alpha/2)``.  Expanding the residue gives the asymptotic law
    ``E(L) = m_j - C m1 exp(-kappa_j L)``, a pure exponential with no power prefactor
    (unlike 3D, where Luscher's bound-state formula carries a 1/L).
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    a_j = alphas[j]
    # neighbouring pole above a_j in sin(alpha): binding grows with alpha, and the next
    # singularity of S is at the pole with the next-larger sin(alpha_i).
    sins = sorted(np.sin(np.array(alphas)))
    s_j = np.sin(a_j)
    upper = np.arcsin(min([s for s in sins if s > s_j + 1e-12], default=1.0))
    out = np.empty_like(L)
    for idx, Lv in enumerate(L):
        lo, hi = a_j + 1e-14, upper - 1e-12

        def f(al):
            return e8_S_imag_rapidity(np.array([al]), alphas)[0] - np.exp(
                m1 * Lv * np.sin(al / 2.0)
            )

        flo = f(lo)
        for _ in range(400):
            mid = 0.5 * (lo + hi)
            if np.sign(f(mid)) == np.sign(flo):
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        out[idx] = 2.0 * m1 * np.cos(0.5 * (lo + hi) / 2.0)
    return out if out.size > 1 else float(out[0])
