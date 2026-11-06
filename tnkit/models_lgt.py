"""The lattice Schwinger model: staggered fermions, Gauss's law eliminated, open chain.

The operator is W = 2H/(a g^2) of Bañuls, Cichy, Cirac, Jansen, JHEP 11 (2013) 158,
Eq. (2.6), on sites n = 0..N-1 with site 0 even:

    W = x sum_{n<N-1} (s+_n s-_{n+1} + s-_n s+_{n+1}) + sum_n mu_n/2 (1 + (-1)^n Z_n)
        + sum_{n<N-1} L_n^2 + charge_penalty * Q^2,

    L_n = ell_n + sum_{k<=n} q_k,   q_k = (Z_k + (-1)^k)/2,   Q = sum_k q_k,

with x = 1/(g a)^2 and mu = 2m/(g^2 a). Scalar ``mu`` and ``ell`` reproduce Eq. (2.6);
arrays give a site-dependent mass and a link-dependent background field. The basis is
tnkit's: index 0 is Z = +1. The profile helpers take ``z = local_expectations(mps, SZ)``.

Routines that take an MPS run on the state's own backend and return numpy. They accept
finite open chains only (both outer bonds 1) and raise on a window state.
"""
from __future__ import annotations

import numpy as np

from .models import SX, SZ, translation_mpo
from .observables import _backend_for, _canon, _profile
from .opsum import OpSum, compile_mpo, spin_half

SP = np.array([[0, 1], [0, 0]], dtype=np.complex128)


def _stag(N: int) -> np.ndarray:
    return (-1.0) ** np.arange(N)


def _params(N, mu, ell, charge_penalty):
    if N < 2:
        raise ValueError(f"the Schwinger chain needs N >= 2 sites, got {N}")
    if not (np.isfinite(charge_penalty) and charge_penalty >= 0):
        raise ValueError(f"charge_penalty must be finite and >= 0, got {charge_penalty}")
    mu = np.broadcast_to(np.asarray(mu, dtype=float), (N,))
    ell = np.broadcast_to(np.asarray(ell, dtype=float), (N - 1,))
    return mu, ell


def schwinger_opsum(N: int, x: float, mu, ell=0.0, charge_penalty: float = 0.0) -> OpSum:
    """W as an `OpSum` on `spin_half` sites, so callers can add terms before compiling.

    ``mu``: scalar or ``(N,)``; ``ell``: scalar or ``(N-1,)``, added to the flux on each
    link. Operator names: ``"Sp"``, ``"Sm"``, ``"Z"``, ``"Id"``.
    """
    mu, ell = _params(N, mu, ell, charge_penalty)
    s = _stag(N)
    os = OpSum()
    for n in range(N - 1):
        os += (x, "Sp", n, "Sm", n + 1)
        os += (x, "Sm", n, "Sp", n + 1)
    for n in range(N):
        os += (mu[n] / 2, "Id", n)
        os += (mu[n] / 2 * s[n], "Z", n)
    os.add_sum_of_squares(np.tril(np.full((N - 1, N), 0.5)), ell + 0.5 * np.cumsum(s)[:-1], "Z")
    if charge_penalty > 0:
        r = np.sqrt(charge_penalty)
        os.add_sum_of_squares(np.full((1, N), 0.5 * r), [0.5 * r * s.sum()], "Z")
    return os


def schwinger_mpo(N: int, x: float, mu, ell=0.0, charge_penalty: float = 0.0) -> list[np.ndarray]:
    """MPO of W (module docstring). Width <= 5; complex128, like every `tnkit.models` builder."""
    mpo, _ = compile_mpo(schwinger_opsum(N, x, mu, ell, charge_penalty), [spin_half()] * N,
                         max_width=5)
    return mpo


def schwinger_dense(N: int, x: float, mu, ell=0.0, charge_penalty: float = 0.0) -> np.ndarray:
    """Dense W, built from the definition without `OpSum`: the diagonal over bitstrings plus
    Kronecker-product hopping. Real ``(2^N, 2^N)``; for testing at small N."""
    mu, ell = _params(N, mu, ell, charge_penalty)
    s = _stag(N)
    bits = (np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1   # site 0 = most significant
    z = 1.0 - 2.0 * bits
    q = 0.5 * (z + s)
    L = ell + np.cumsum(q, axis=1)[:, :-1]
    diag = (0.5 * mu * (1 + s * z)).sum(1) + (L ** 2).sum(1) + charge_penalty * q.sum(1) ** 2
    hop = np.kron(SP.real, SP.real.T)
    hop = hop + hop.T
    H = np.diag(diag)
    for n in range(N - 1):
        H += x * np.kron(np.kron(np.eye(2 ** n), hop), np.eye(2 ** (N - n - 2)))
    return H


def charge_conjugation_mpo(N: int) -> list[np.ndarray]:
    """S_R = (prod_n X_n) T, with T the cyclic shift by one site (`translation_mpo(N, +1)`).

    Bañuls et al. 2013, Sec. 3, print S_R with X on every other site; that reading does not
    map the bare vacuum to itself, and X on every site does, so this is the operator used.
    S_R is not a symmetry of the open chain, but the sign of <S_R> on a real eigenstate tags
    it: + for the ground state and the scalar branch, - for the vector branch. Even N only.
    Width 4; complex128.
    """
    if N % 2:
        raise ValueError(f"charge_conjugation_mpo needs even N, got {N}")
    return [np.einsum("st,abtu->absu", SX, W) for W in translation_mpo(N, +1)]


def site_charge(z) -> np.ndarray:
    """q_n = (z_n + (-1)^n)/2, shape (N,). Their sum is the total charge Q."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (z + _stag(len(z)))


def particle_number(z) -> np.ndarray:
    """(1 + (-1)^n z_n)/2 in [0, 1], shape (N,). Its mean is the particle density."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1 + _stag(len(z)) * z)


def flux_profile(z, ell=0.0) -> np.ndarray:
    """<L_n> = ell_n + sum_{k<=n} q_k for links n = 0..N-2, shape (N-1,)."""
    q = site_charge(z)
    return np.broadcast_to(np.asarray(ell, dtype=float), (len(q) - 1,)) + np.cumsum(q)[:-1]


def chiral_condensate(z, x: float) -> float:
    """Sigma = (sqrt(x)/N) sum_n (-1)^n (1 + z_n)/2, the staggered condensate <psi-bar psi>/g.

    This is the operator printed in Bañuls, Cichy, Cirac, Jansen, Saito, PoS LATTICE2013
    (arXiv:1310.4118), Sec. 3. With it the free (no gauge field) ground state gives MINUS
    the positive value of that paper's Eq. (3.1), so the paper's printed numbers are
    -<Sigma>. No free-condensate subtraction or extrapolation is applied.
    """
    z = np.asarray(z, dtype=float)
    return float(np.sqrt(x) / len(z) * np.sum(_stag(len(z)) * 0.5 * (1 + z)))


def _open_chain(mps):
    if int(mps[0].shape[0]) != 1 or int(mps[-1].shape[2]) != 1:
        raise ValueError("needs a finite open chain (outer bonds 1); got outer bonds "
                         f"{int(mps[0].shape[0])} and {int(mps[-1].shape[2])}")


def flux_correlator(mps, canon=None) -> np.ndarray:
    """C^{mn} = <L_m L_n> - <L_m><L_n> for links m, n = 0..N-2, shape (N-1, N-1).

    Papaefstathiou, Knolle, Bañuls, PRD 111, 014504 (2025), Eq. (21). Independent of ell.
    One sweep carries a stack of environments, one per earlier site j with Z_j inserted,
    so all <Z_j Z_k> are read at site k by one batched contraction. Cost O(N^2 chi^3), memory
    N chi^2; one host transfer.
    """
    _open_chain(mps)
    be = _backend_for(mps)
    M = _canon(mps, be) if canon is None else canon
    N = len(M)
    z = be.array(SZ)
    zz = be.zeros((N, N))              # strict lower triangle <Z_j Z_k>, diagonal <Z_k>
    L = be.ones((1, 1))
    X = None                           # (k, D, D): environments with Z_j, j < k
    for k in range(N):
        A = M[k]
        Ac = be.conj(A)
        if X is not None:
            # pairwise contractions: a single three-tensor einsum may route through a
            # chi^4 intermediate
            XA = be.einsum("mab,asc->mbsc", X, A)
            zz[k, :k] = be.einsum("mbtc,btc->m", be.einsum("mbsc,ts->mbtc", XA, z), Ac)
            X = be.einsum("mbsc,bsd->mcd", XA, Ac)
        Zk = be.einsum("ab,asc,ts,btd->cd", L, A, z, Ac)
        zz[k, k] = be.einsum("aa->", Zk)
        L = be.einsum("ab,asc,bsd->cd", L, A, Ac)
        X = be.stack_axis([Zk] if X is None else [*X, Zk], 0)
    zz = np.real(np.asarray(be.to_numpy(zz)))
    zk = np.diag(zz).copy()
    G = np.tril(zz, -1)
    G = G + G.T + np.eye(N) - np.outer(zk, zk)
    P = np.tril(np.ones((N - 1, N)))
    return 0.25 * P @ G @ P.T


def energy_density(mps, x: float, mu, ell=0.0, charge_penalty: float = 0.0, canon=None):
    """Local pieces of <W> for the operator `schwinger_mpo(N, x, mu, ell, charge_penalty)`.

    Returns ``(hop, mass, electric, penalty)``: ``hop[n] = x <s+_n s-_{n+1} + h.c.>`` per
    bond, ``mass[n] = mu_n/2 (1 + (-1)^n <Z_n>)`` per site, ``electric[n] = <L_n^2>`` per
    link, and ``penalty = charge_penalty <Q^2>`` as one number. Their total is <W>.
    One sweep, O(N chi^3), carrying the norm, sum Z_k and (sum Z_k)^2 environments;
    one host transfer.
    """
    _open_chain(mps)
    be = _backend_for(mps)
    M = _canon(mps, be) if canon is None else canon
    N = len(M)
    mu, ell = _params(N, mu, ell, charge_penalty)
    z, sp, sm = be.array(SZ), be.array(SP), be.array(SP.T.copy())
    A0, B, C = be.ones((1, 1)), be.zeros((1, 1)), be.zeros((1, 1))
    Ep = Em = None
    zs, hop, s1, s2 = [], [], [], []
    for n in range(N):
        A = M[n]
        Ac = be.conj(A)
        if Ep is not None:
            hop.append(be.einsum("ab,asc,ts,btc->", Ep, A, sm, Ac)
                       + be.einsum("ab,asc,ts,btc->", Em, A, sp, Ac))
        TzA = be.einsum("ab,asc,ts,btd->cd", A0, A, z, Ac)
        TzB = be.einsum("ab,asc,ts,btd->cd", B, A, z, Ac)
        TA = be.einsum("ab,asc,bsd->cd", A0, A, Ac)
        C = be.einsum("ab,asc,bsd->cd", C, A, Ac) + 2 * TzB + TA
        B = be.einsum("ab,asc,bsd->cd", B, A, Ac) + TzA
        Ep = be.einsum("ab,asc,ts,btd->cd", A0, A, sp, Ac)
        Em = be.einsum("ab,asc,ts,btd->cd", A0, A, sm, Ac)
        A0 = TA
        zs.append(be.einsum("aa->", TzA))
        s1.append(be.einsum("aa->", B))
        s2.append(be.einsum("aa->", C))
    v = _profile(be, zs + hop + s1 + s2)
    zv, h = v[:N], v[N:2 * N - 1]
    S1, S2 = v[2 * N - 1:3 * N - 1], v[3 * N - 1:]
    s = _stag(N)
    c = ell + 0.5 * np.cumsum(s)[:-1]
    electric = c ** 2 + c * S1[:-1] + 0.25 * S2[:-1]
    q0 = 0.5 * s.sum()
    penalty = charge_penalty * (q0 ** 2 + q0 * S1[-1] + 0.25 * S2[-1])
    return x * h, 0.5 * mu * (1 + s * zv), electric, float(penalty)
