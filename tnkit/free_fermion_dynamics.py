"""Exact real-time dynamics of the open TFIM chain via free fermions (Majorana covariance).

    H(g) = - sum_{i=1}^{N-1} sigma^x_i sigma^x_{i+1}  -  g sum_{i=1}^{N} sigma^z_i

is quadratic under Jordan-Wigner, so a global quench g0 -> g1 is exactly solvable at ANY N and ANY
time in O(N^3) per time point. That matters because exact diagonalisation dies at N~16 while the
interesting TDVP regime is N~100+ and long times -- exactly where the entanglement barrier lives
and where we have had no ground truth at all.

Method. Write H in Majorana operators a_1..a_{2N} (a_{2j-1} = c_j + c_j^dag,
a_{2j} = i(c_j - c_j^dag)) as H = (i/4) a^T A a with A real antisymmetric. Heisenberg evolution is
then a(t) = e^{A t} a(0), so the covariance matrix

    Gamma_{jk} = (i/2) <[a_j, a_k]>          (real antisymmetric, Gamma^T = -Gamma)

evolves by an ORTHOGONAL conjugation

    Gamma(t) = R Gamma(0) R^T,   R = expm(A t)

which is unconditionally stable -- no truncation, no timestep error, exact to machine precision.
Observables are single entries of Gamma.

Conventions are checked against dense exact diagonalisation in `validate_against_ed`, which is the
only way to be sure the Jordan-Wigner sign bookkeeping is right.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm


def majorana_A(N, g):
    """Real antisymmetric A with H = (i/4) sum_{jk} A_jk a_j a_k for the open TFIM.

    With a_{2j-1} = c_j + c_j^dag and a_{2j} = i(c_j - c_j^dag) (1-based sites j):
      -g sigma^z_j            -> -g * (-i a_{2j-1} a_{2j})     [sigma^z_j = -i a_{2j-1} a_{2j}]
      -sigma^x_j sigma^x_{j+1} -> -(-i a_{2j} a_{2j+1})        [string cancels for NN bonds]
    """
    A = np.zeros((2 * N, 2 * N), dtype=np.float64)

    def add(p, q, c):
        """Add c * (i/4)*[a_p a_q] worth of coupling, antisymmetrised."""
        A[p, q] += c
        A[q, p] -= c

    # field term: -g sigma^z_j = -g * (-i a_{2j-1} a_{2j}) = i g a_{2j-1} a_{2j}
    for j in range(N):
        add(2 * j, 2 * j + 1, 2.0 * g)
    # bond term: -sigma^x_j sigma^x_{j+1} = i a_{2j} a_{2j+1}
    for j in range(N - 1):
        add(2 * j + 1, 2 * j + 2, 2.0)
    return A


def gamma_ground(N, g):
    """Covariance matrix of the ground state of H(g), from the spectral form of A."""
    A = majorana_A(N, g)
    # A real antisymmetric -> A = O (bigoplus [[0, eps_k],[-eps_k, 0]]) O^T with eps_k >= 0.
    # Use the Schur form, which gives exactly that block structure.
    from scipy.linalg import schur
    T, O = schur(A, output="real")
    G = np.zeros_like(T)
    for k in range(0, 2 * N, 2):
        # T[k, k+1] = +eps or -eps; the ground state has Gamma block = sign(eps) * [[0,1],[-1,0]]
        s = 1.0 if T[k, k + 1] >= 0 else -1.0
        G[k, k + 1] = s
        G[k + 1, k] = -s
    return O @ G @ O.T


def evolve(gamma0, N, g1, times):
    """Yield (t, Gamma(t)) for a quench to g1.

    Uses the SPECTRAL form rather than a per-timestep expm. A is real antisymmetric, so its real
    Schur form is exactly block-diagonal with 2x2 blocks [[0, e], [-e, 0]], and

        expm(A t) = O . blockdiag(rot(e_k t)) . O^T,   rot(x) = [[cos x, sin x], [-sin x, cos x]]

    so the evolution operator is orthogonal BY CONSTRUCTION at every t. A dense expm per timestep
    is both slower (one 2N x 2N expm per time point) and, at N=200, was observed to blow up at
    isolated t (|<sigma^z>| ~ 1e14 at t=0.503) while remaining fine at neighbouring t. The
    spectral form has no such failure mode and costs one Schur decomposition for the whole scan.
    """
    from scipy.linalg import schur
    A = majorana_A(N, g1)
    T, O = schur(A, output="real")
    eps = np.array([T[k, k + 1] for k in range(0, 2 * N, 2)])   # signed block frequencies
    Gt = O.T @ gamma0 @ O                                        # rotate once into the block basis
    for t in times:
        c, sn = np.cos(eps * t), np.sin(eps * t)
        R = np.zeros_like(T)
        for k in range(N):
            R[2 * k, 2 * k] = c[k];      R[2 * k, 2 * k + 1] = sn[k]
            R[2 * k + 1, 2 * k] = -sn[k]; R[2 * k + 1, 2 * k + 1] = c[k]
        yield t, O @ (R @ Gt @ R.T) @ O.T


def sigma_z(gamma, j):
    """<sigma^z_j> = +Gamma[2j, 2j+1] in this convention (0-based j; sign fixed against ED)."""
    return gamma[2 * j, 2 * j + 1]


def sigma_x_bond(gamma, j):
    """<sigma^x_j sigma^x_{j+1}> = +Gamma[2j+1, 2j+2] (0-based j; sign fixed against ED)."""
    return gamma[2 * j + 1, 2 * j + 2]


def quench_observables(N, g0, g1, times):
    """Exact <sigma^z>(t) (site-averaged) and <sigma^x sigma^x>(t) (bond-averaged) after a quench."""
    gamma0 = gamma_ground(N, g0)
    out = []
    for t, G in evolve(gamma0, N, g1, times):
        mz = float(np.mean([sigma_z(G, j) for j in range(N)]))
        mxx = float(np.mean([sigma_x_bond(G, j) for j in range(N - 1)]))
        out.append({"t": float(t), "sz": mz, "sxx": mxx})
    return out


# ---------------------------------------------------------------------------- validation
def validate_against_ed(N=8, g0=2.0, g1=1.0, times=(0.0, 0.3, 0.7, 1.5), tol=1e-9, verbose=True):
    """The Jordan-Wigner sign bookkeeping is the whole risk. Check it against dense ED."""
    from . import models as M
    from scipy.linalg import expm as dexpm
    H0 = M.tfim_dense(N, g0)
    H1 = M.tfim_dense(N, g1)
    w, V = np.linalg.eigh(H0)
    psi0 = V[:, 0]

    def op_at(op, site):
        m = np.array([[1.0 + 0j]])
        for k in range(N):
            m = np.kron(m, op if k == site else M.I2)
        return m

    SZ_ops = [op_at(M.SZ, j) for j in range(N)]
    SXX_ops = [op_at(M.SX, j) @ op_at(M.SX, j + 1) for j in range(N - 1)]

    ff = quench_observables(N, g0, g1, times)
    ok = True
    for rec in ff:
        U = dexpm(-1j * H1 * rec["t"])
        psi = U @ psi0
        mz = float(np.mean([np.real(np.vdot(psi, S @ psi)) for S in SZ_ops]))
        mxx = float(np.mean([np.real(np.vdot(psi, S @ psi)) for S in SXX_ops]))
        dz, dx = abs(mz - rec["sz"]), abs(mxx - rec["sxx"])
        ok &= (dz < tol and dx < tol)
        if verbose:
            print(f"  t={rec['t']:5.2f}  sz: ff={rec['sz']:+.10f} ed={mz:+.10f} d={dz:.2e}   "
                  f"sxx: ff={rec['sxx']:+.10f} ed={mxx:+.10f} d={dx:.2e}"
                  f"{'' if (dz < tol and dx < tol) else '   <-- MISMATCH'}")
    return ok


def apply_sigma_z(gamma, j):
    """Gamma after applying sigma^z_j -- an exact Gaussian operation.

    sigma^z_j = exp(i pi n_j) = -i gamma_{2j} gamma_{2j+1} is a product of TWO Majoranas on a
    SINGLE site, with no Jordan-Wigner string attached (unlike sigma^x/sigma^y). Conjugation
    therefore flips exactly those two Majorana rows/columns:

        sigma^z_j gamma_a sigma^z_j = -gamma_a   for a in {2j, 2j+1},  +gamma_a otherwise

    so Gamma -> R Gamma R^T with R = diag(1,...,1,-1,-1,1,...,1).

    IMPORTANT SCOPE: this is exact only when the
    state being acted on is itself a single Gaussian state in a DEFINITE PARITY SECTOR. A TFIM
    ground state is (it is a parity eigenstate), so sigma^z on a ground state is exactly
    representable. The product state (x)|+> is NOT -- it has <P> = 0, an equal superposition of
    both sectors -- which is why the earlier attempt to treat the thesis spin-flip protocol this
    way failed, with the error growing 1e-15 -> 4.8e-2.
    """
    R = np.ones(gamma.shape[0])
    R[2 * j] = -1.0
    R[2 * j + 1] = -1.0
    return (R[:, None] * gamma) * R[None, :]


def evolve_td(gamma0, N, g_of_t, times, dt_ref=0.005, magnus=True):
    """Yield (t, Gamma(t)) for a TIME-DEPENDENT field g(t) — the exact reference for a ramp.

    With H(t) quadratic at every t, the covariance still evolves by an ORTHOGONAL conjugation
    Gamma(t) = R(t) Gamma_0 R(t)^T, but now R solves dR/dt = A(t) R instead of R = expm(A t).
    R is built as a product of step propagators, each an `expm` of a real antisymmetric matrix
    and therefore orthogonal BY CONSTRUCTION — the same property that fixed the |<sigma^z>| = 5e14
    blow-up in the sudden-quench reference.

    magnus=True uses the 4th-order two-point Gauss–Legendre Magnus expansion

        Omega = (h/2)(A1 + A2) + (sqrt(3) h^2 / 12) [A2, A1],

    which costs one extra commutator per step and buys two orders over the midpoint rule. At
    dt_ref = 0.005 the self-convergence against dt_ref/2 is below 1e-13 for the ramps used here;
    `ramp_reference_convergence` measures it rather than asserting it.

    The reference must NOT reuse the TDVP integrator's own step size or splitting — that would
    make the "exact" reference share the error being measured.
    """
    times = np.asarray(times, dtype=float)
    order = np.argsort(times)
    A_of = lambda tt: majorana_A(N, float(g_of_t(tt)))
    c1, c2 = 0.5 - np.sqrt(3) / 6.0, 0.5 + np.sqrt(3) / 6.0
    R = np.eye(2 * N)
    t = 0.0
    out = {}
    for idx in order:
        tgt = float(times[idx])
        if tgt < t - 1e-15:
            raise ValueError("evolve_td needs non-negative, forward times")
        nsub = max(int(np.ceil((tgt - t) / dt_ref - 1e-12)), 0)
        for _ in range(nsub):
            h = (tgt - t) / (nsub - _)
            if magnus:
                A1, A2 = A_of(t + c1 * h), A_of(t + c2 * h)
                Om = 0.5 * h * (A1 + A2) + (np.sqrt(3) * h * h / 12.0) * (A2 @ A1 - A1 @ A2)
            else:
                Om = h * A_of(t + 0.5 * h)
            R = expm(Om) @ R
            t += h
        out[idx] = R @ gamma0 @ R.T
    for idx in range(len(times)):
        yield float(times[idx]), out[idx]


def ramp_reference_convergence(N, g_of_t, t_end, dt_ref=0.005):
    """max |Gamma(dt_ref) - Gamma(dt_ref/2)| — the reference's own self-convergence."""
    g0 = gamma_ground(N, float(g_of_t(0.0)))
    Ga = list(evolve_td(g0, N, g_of_t, [t_end], dt_ref=dt_ref))[0][1]
    Gb = list(evolve_td(g0, N, g_of_t, [t_end], dt_ref=dt_ref / 2))[0][1]
    return float(np.max(np.abs(Ga - Gb)))
