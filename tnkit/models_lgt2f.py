"""The two-flavour lattice Schwinger model: staggered fermions, Gauss's law eliminated, open chain.

Two fermion species ("u", "d") of equal charge on the staggered sites n = 0..N-1 (site 0 even).
The operator is the two-flavour form of W = 2H/(a g^2) of Bañuls, Cichy, Cirac, Jansen,
JHEP 11 (2013) 158, Eq. (2.6) (as in Itou, Matsumoto, Tanizaki, JHEP 11 (2023) 231, App. A):

    W = x sum_f sum_{n<N-1} (c+_{n,f} c_{n+1,f} + h.c.)
        + sum_{n,f} mu_{n,f} [ (1 - (-1)^n)/2 + (-1)^n n_{n,f} ]
        + sum_{n<N-1} L_n^2 + charge_penalty * Q^2 + isospin_penalty * I^2,

    L_n = ell_n + sum_{k<=n} q_k,   q_k = n_{k,u} + n_{k,d} - (1 - (-1)^k),   Q = sum_k q_k,

with I^2 the isospin Casimir (`isospin_opsum`), x = 1/(g a)^2 and mu = 2 m / (g^2 a) =
2 sqrt(x) m/g; the mass term is the one-flavour mu/2 (1 + (-1)^n Z) per flavour with Z = 2n - 1.
Energies are in units of a g^2 / 2, so a mass gap is M/g = Delta W / (2 sqrt(x)). The staggered
mass shift of Dempsey et al. (2022) and Itou et al. (2023, Eq. 3.12), m_lat = m - N_f g^2 a / 8,
reads mu_lat = mu_cont - N_f/4 here.

Layout: the flavours are interleaved on a chain of 2N qubits, qubit j = 2n + f with f = 0 (u)
and 1 (d), each a `spinless_fermion` site (index 0 = EMPTY, the opposite of the one-flavour
module's spin basis). Jordan-Wigner strings come from the `OpSum` compiler. Against one d = 4
site per staggered site, this keeps the MPO width and makes two-site updates cheaper, but a
same-flavour hop is next-nearest-neighbour on the qubit chain, so two-site TDVP projects it;
`merge_pairs` converts an MPO or MPS to the d = 4 layout exactly.

Exact symmetries for equal masses: total charge Q; isospin (I_z, I^2: `isospin_opsum`); on
the Q = 0 sector, the lattice P for odd N and ell = 0, and the lattice CP for even N
(`reflection_expectation`). C alone (`charge_conjugation_2f_mpo`) is not a symmetry of the
open chain; the sign of its expectation value tags states.
"""
from __future__ import annotations

import numpy as np

from .models import SX, SZ, translation_mpo
from .models_lgt import _stag
from .opsum import OpSum, compile_mpo, spinless_fermion

U, D = 0, 1


def qubit(n: int, f: int) -> int:
    """Chain index of flavour ``f`` (0 = u, 1 = d) at staggered site ``n``."""
    return 2 * n + f


def _params(N, mu, ell, charge_penalty):
    if N < 2:
        raise ValueError(f"the Schwinger chain needs N >= 2 sites, got {N}")
    if not (np.isfinite(charge_penalty) and charge_penalty >= 0):
        raise ValueError(f"charge_penalty must be finite and >= 0, got {charge_penalty}")
    mu = np.asarray(mu, dtype=float)
    if mu.ndim == 1 and mu.shape != (2,):
        raise ValueError(f"mu must be a scalar, (2,) per flavour or (N, 2); got shape {mu.shape}")
    mu = np.broadcast_to(mu, (N, 2))
    ell = np.broadcast_to(np.asarray(ell, dtype=float), (N - 1,))
    return mu, ell


def schwinger2f_opsum(N: int, x: float, mu, ell=0.0, charge_penalty: float = 0.0,
                      isospin_penalty: float = 0.0) -> OpSum:
    """W (module docstring) as an `OpSum` on 2N `spinless_fermion` sites.

    ``mu``: scalar (both flavours), ``(2,)`` (per flavour, ``mu[0]`` for u) or ``(N, 2)``;
    ``ell``: scalar or ``(N-1,)``, added to the flux on each link. ``isospin_penalty`` adds
    lambda I^2, which vanishes on isosinglets; the compiled MPO is then at most 4 wider than W's.
    With unequal masses I^2 does not commute with H, so the penalty then changes the dynamics.
    Operator names: ``"C"``, ``"Cdag"``, ``"N"``, ``"Id"``.
    """
    mu, ell = _params(N, mu, ell, charge_penalty)
    if not (np.isfinite(isospin_penalty) and isospin_penalty >= 0):
        raise ValueError(f"isospin_penalty must be finite and >= 0, got {isospin_penalty}")
    s = _stag(N)
    os = OpSum()
    for n in range(N - 1):
        for f in (U, D):
            j = qubit(n, f)
            os += (x, "Cdag", j, "C", j + 2)
            os += (x, "Cdag", j + 2, "C", j)
    const = 0.0
    for n in range(N):
        for f in (U, D):
            os += (mu[n, f] * s[n], "N", qubit(n, f))
            const += mu[n, f] * (1 - s[n]) / 2
    if const:
        os += (const, "Id", 0)
    A = np.array([[1.0 if j // 2 <= n else 0.0 for j in range(2 * N)] for n in range(N - 1)])
    os.add_sum_of_squares(A, ell - np.cumsum(1 - s)[:-1], "N")
    if charge_penalty > 0:
        r = np.sqrt(charge_penalty)
        os.add_sum_of_squares(np.full((1, 2 * N), r), [-r * (1 - s).sum()], "N")
    if isospin_penalty > 0:
        _add_isospin_squared(os, N, isospin_penalty)
    return os


def _add_isospin_squared(os: OpSum, N: int, w: float) -> None:
    """Add w I^2 to ``os``. Same operator as ``isospin_opsum(N, "I2")``, but the on-site part of
    (I+ I- + I- I+)/2 is written as (n_u + n_d)/2 - n_u n_d, so the compiler sees one operator
    per site where it would otherwise see several equal ones; this saves one MPO channel."""
    a = np.array([[0.5 if j % 2 == 0 else -0.5 for j in range(2 * N)]])
    os.add_sum_of_squares(np.sqrt(w) * a, [0.0], "N")
    for n in range(N):
        u, d = qubit(n, U), qubit(n, D)
        os += (0.5 * w, "N", u)
        os += (0.5 * w, "N", d)
        os += (-w, "N", u, "N", d)
        for m in range(N):
            if m != n:       # I+_n I-_m and I-_m I+_n commute for n != m
                os += (w, "Cdag", u, "C", d, "Cdag", qubit(m, D), "C", qubit(m, U))


def isospin_opsum(N: int, which: str, target: float = 0.0) -> OpSum:
    """Isospin on the 2N-qubit chain, as an `OpSum` (add it to W's to penalise a channel).

    ``which``: ``"Iz"`` (I_z = sum_n (n_{n,u} - n_{n,d})/2), ``"Iz2"`` ((I_z - target)^2), or
    ``"I2"`` (I^2 = I_z^2 + (I+ I- + I- I+)/2 with I+ = sum_n c+_{n,u} c_{n,d}; eigenvalues
    I(I+1)). Isospin commutes with W exactly when the two flavour masses are equal.
    """
    os = OpSum()
    if which == "Iz":
        for n in range(N):
            os += (0.5, "N", qubit(n, U))
            os += (-0.5, "N", qubit(n, D))
        return os
    a = np.array([[0.5 if j % 2 == 0 else -0.5 for j in range(2 * N)]])
    if which == "Iz2":
        os.add_sum_of_squares(a, [-float(target)], "N")
        return os
    if which != "I2":
        raise ValueError(f"which must be 'Iz', 'Iz2' or 'I2', got {which!r}")
    os.add_sum_of_squares(a, [0.0], "N")
    for n in range(N):
        for m in range(N):
            # I+ I- and I- I+, each with weight 1/2
            os += (0.5, "Cdag", qubit(n, U), "C", qubit(n, D), "Cdag", qubit(m, D), "C", qubit(m, U))
            os += (0.5, "Cdag", qubit(n, D), "C", qubit(n, U), "Cdag", qubit(m, U), "C", qubit(m, D))
    return os


def charge_base(N: int) -> int:
    """K in the packed charge c = q + K t of `site_charges_2f`: more than twice any |q| a
    tensor of the chain can carry, so c fixes q and t."""
    return 4 * N + 4


def site_charges_2f(N: int) -> list[np.ndarray]:
    """Per qubit, the packed charge c = q + K t of its two states (empty, occupied), for block
    tensors that conserve the charge Q and the isospin I_z together.

    q is the qubit's share of the staggered charge, n_j - [n odd] for qubit j = 2n + f, and
    t = 2 I_z = (1 - 2f) n_j. K = `charge_base(N)`. Every block-tensor charge rule is a signed
    sum, so conserving c conserves q and t separately.
    """
    K = charge_base(N)
    out = []
    for j in range(2 * N):
        n, f = divmod(j, 2)
        occ = np.array([0, 1])
        out.append(occ - (n % 2) + K * (1 - 2 * f) * occ)
    return out


def unpack_charge(c, N: int):
    """(q, 2 I_z) from packed charges c (`site_charges_2f`)."""
    K = charge_base(N)
    t = np.rint(np.asarray(c) / K).astype(np.int64)
    return np.asarray(c) - K * t, t


def merged(*sums: OpSum, weights=None) -> OpSum:
    """The weighted sum of several `OpSum` objects as one (weights default to 1)."""
    weights = [1.0] * len(sums) if weights is None else list(weights)
    out = OpSum()
    for os, w in zip(sums, weights):
        out.terms += [(w * c, f) for c, f in os.terms]
        out.forms += [(w * J, A, B) for J, A, B in os.forms]
    return out


def compile2f(os: OpSum, N: int, max_width: int | None = None, real: bool = False):
    """Compile an `OpSum` on the 2N-qubit chain; ``real=True`` returns float64 arrays."""
    mpo, report = compile_mpo(os, [spinless_fermion()] * (2 * N), max_width=max_width)
    if real:
        if max(float(np.abs(w.imag).max()) for w in mpo) > 0:
            raise ValueError("the MPO has a nonzero imaginary part")
        mpo = [np.ascontiguousarray(w.real) for w in mpo]
    return mpo, report


def schwinger2f_mpo(N: int, x: float, mu, ell=0.0, charge_penalty: float = 0.0,
                    isospin_penalty: float = 0.0) -> list[np.ndarray]:
    """MPO of W (module docstring) on 2N qubits; complex128, like every `tnkit.models` builder."""
    mpo, _ = compile2f(schwinger2f_opsum(N, x, mu, ell, charge_penalty, isospin_penalty), N)
    return mpo


# --------------------------------------------------------------------------- exact reference
def _occ(N2: int) -> np.ndarray:
    """Occupations of every basis state, qubit 0 most significant (as `opsum.to_dense`)."""
    return (np.arange(2 ** N2)[:, None] >> np.arange(N2 - 1, -1, -1)) & 1


def schwinger2f_sparse(N: int, x: float, mu, ell=0.0, charge_penalty: float = 0.0):
    """W as a scipy CSR matrix, built from its Pauli-string form without `OpSum`.

    Basis: 2N qubits, qubit 0 most significant, bit 1 = occupied (the `spinless_fermion`
    basis, so it equals ``to_dense(schwinger2f_mpo(...))`` entry by entry). Each hop
    c+_j c_{j+2} = sigma+_j F_{j+1} sigma-_{j+2} (F = 1 - 2n) is applied by flipping the two
    bits and reading the sign off the middle bit. For exact diagonalisation at small N.
    """
    import scipy.sparse as sp

    mu, ell = _params(N, mu, ell, charge_penalty)
    N2 = 2 * N
    s = _stag(N)
    occ = _occ(N2).astype(float)                              # (2^2N, 2N)
    nsite = occ[:, 0::2] + occ[:, 1::2]                       # n_u + n_d per staggered site
    q = nsite - (1 - s)
    L = ell + np.cumsum(q, axis=1)[:, :-1]
    mass = (mu[:, 0] * (0.5 * (1 - s) + s * occ[:, 0::2])
            + mu[:, 1] * (0.5 * (1 - s) + s * occ[:, 1::2])).sum(1)
    diag = mass + (L ** 2).sum(1) + charge_penalty * q.sum(1) ** 2
    idx = np.arange(2 ** N2)
    rows, cols, vals = [idx], [idx], [diag]
    for j in range(N2 - 2):
        bj, bk, bm = 1 << (N2 - 1 - j), 1 << (N2 - 3 - j), 1 << (N2 - 2 - j)
        # c+_j c_{j+2}: needs j empty and j+2 occupied
        src = idx[((idx & bj) == 0) & ((idx & bk) != 0)]
        dst = src ^ bj ^ bk
        sign = 1.0 - 2.0 * ((src & bm) != 0)
        rows += [dst, src]
        cols += [src, dst]
        vals += [x * sign, x * sign]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(2 ** N2, 2 ** N2))


def isospin_sparse(N: int):
    """``(Iz, Iplus)`` as scipy CSR matrices in the basis of `schwinger2f_sparse`.

    I+ = sum_n c+_{n,u} c_{n,d}: the u and d qubits of one site are adjacent, so the
    Jordan-Wigner string is trivial and the term is sigma+_{2n} sigma-_{2n+1}.
    """
    import scipy.sparse as sp

    N2 = 2 * N
    occ = _occ(N2)
    idx = np.arange(2 ** N2)
    iz = 0.5 * (occ[:, 0::2].sum(1) - occ[:, 1::2].sum(1))
    rows, cols = [], []
    for n in range(N):
        bu, bd = 1 << (N2 - 1 - 2 * n), 1 << (N2 - 2 - 2 * n)
        src = idx[((idx & bu) == 0) & ((idx & bd) != 0)]
        rows.append(src ^ bu ^ bd)
        cols.append(src)
    r, c = np.concatenate(rows), np.concatenate(cols)
    Ip = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(2 ** N2, 2 ** N2))
    return sp.diags(iz).tocsr(), Ip


def charge_sparse_diag(N: int) -> np.ndarray:
    """Total charge Q of every basis state of `schwinger2f_sparse`."""
    occ = _occ(2 * N)
    return occ.sum(1) - (1 - _stag(N)).sum()


# --------------------------------------------------------------------------- symmetry tags
def reflection_expectation(mps, other=None, particle_hole: bool = False) -> complex:
    """<a|R|b> / (|a| |b|) with R the reversal of the chain order (b defaults to a).

    On the 2N-qubit chain R maps qubit (n, f) to (N-1-n, 1-f): the site reflection
    n -> N-1-n combined with u <-> d. For equal masses on the Q = 0 sector:

    * odd N, ell = 0: R commutes with W. It is the lattice parity P (times the u <-> d swap,
      which is +1 on isosinglets and -1 on the I_z = 0 member of an isotriplet).
    * even N, ``particle_hole=True``: R is followed by X on every qubit and Z on both qubits
      of every even site. That operator commutes with W for any uniform ell; it is the
      lattice CP (times the same swap sign).

    Exact eigenstates therefore give +-1; the vacuum gives +1. Host arrays, d = 2 sites.
    """
    a = [np.asarray(m) for m in mps]
    b = a if other is None else [np.asarray(m) for m in other]
    rb = [m.transpose(2, 1, 0) for m in reversed(b)]
    if particle_hole:
        if (len(rb) // 2) % 2:
            raise ValueError("particle_hole=True needs an even number N of staggered sites")
        rb = [np.einsum("st,atb->asb", SX if (j // 2) % 2 else SZ @ SX, m)
              for j, m in enumerate(rb)]

    def inner(p, q):
        E = np.ones((1, 1), dtype=np.result_type(p[0], q[0]))
        for A, B in zip(p, q):
            # pairwise: a single three-operand einsum loops naively over all five indices
            E = np.einsum("asc,asd->cd", np.einsum("ab,asc->bsc", E, np.conj(A)), B)
        return E[0, 0]

    return complex(inner(a, rb) / np.sqrt(abs(inner(a, a)) * abs(inner(b, b))))


def charge_conjugation_2f_mpo(N: int) -> list[np.ndarray]:
    """C_lat = (prod_{n even} Z_{n,u} Z_{n,d}) (prod_j X_j) T^2 on the 2N-qubit chain.

    T is the cyclic shift by one qubit (`translation_mpo(2N, +1)`), so T^2 shifts by one
    staggered site; prod X is particle-hole on every qubit; the Z's on even sites restore the
    sign of the hop. In the bulk C_lat maps c_{n,f} -> +-c+_{n+1,f}, q_n -> -q_{n+1}, and W to
    itself; the open ends break it, so <C_lat> on an eigenstate is not +-1 but its sign tags
    the state: + for the vacuum and C-even states, - for C-odd ones. Even N only.
    Width 16; complex128.
    """
    if N % 2:
        raise ValueError(f"charge_conjugation_2f_mpo needs even N, got {N}")
    T = translation_mpo(2 * N, +1)
    T2 = []
    for A in T:
        W = np.einsum("abts,cdsu->acbdtu", A, A)
        T2.append(W.reshape(A.shape[0] ** 2, A.shape[1] ** 2, 2, 2))
    out = []
    for j, W in enumerate(T2):
        loc = SX if (j // 2) % 2 else SZ @ SX
        out.append(np.einsum("st,abtu->absu", loc, W))
    return out


def merge_pairs(tensors) -> list[np.ndarray]:
    """Group qubits (2n, 2n+1) into one d = 4 site, exactly. Takes an MPO (rank-4 tensors
    ``(Dl, Dr, d, d)``) or an MPS (rank-3 ``(chi_l, d, chi_r)``); the d = 4 index is
    ``2 * s_u + s_d``."""
    out = []
    for A, B in zip(tensors[0::2], tensors[1::2]):
        if A.ndim == 4:
            W = np.einsum("abst,bcuv->acsutv", A, B)
            out.append(W.reshape(A.shape[0], B.shape[1], 4, 4))
        else:
            M = np.einsum("asb,btc->astc", A, B)
            out.append(M.reshape(A.shape[0], 4, B.shape[2]))
    return out


# --------------------------------------------------------------------------- wave packets
PACKET_KINDS = ("u-d", "u+d", "+", "-")


def packet_opsum(N: int, kind: str, f) -> OpSum:
    """sum_{n=0}^{N-2} f_n B_n as an `OpSum`, a meson wave-packet operator on the link (n, n+1).

    With hop_f(n) = c+_{n,f} c_{n+1,f}, B_f(n) = hop_f(n) - hop_f(n)^dag:

        "u-d"  B_u - B_d                         neutral pion (isovector, I_z = 0, C-odd)
        "u+d"  B_u + B_d                         eta channel (isoscalar, C-odd)
        "+"    c+_{n,u} c_{n+1,d} - c+_{n+1,u} c_{n,d}   pi+ (raises I_z by 1)
        "-"    c+_{n,d} c_{n+1,u} - c+_{n+1,d} c_{n,u}   pi- (lowers I_z by 1)

    [I+, B_u - B_d] = -2 B_+, so on an isosinglet vacuum B_+|0> = -(1/2) I+ (B_u - B_d)|0>: the
    pi+ packet is the isospin rotation of the pi0 packet. ``f`` has length N - 1 (complex).
    """
    f = np.asarray(f)
    if f.shape != (N - 1,):
        raise ValueError(f"f must have shape ({N - 1},), got {f.shape}")
    if kind not in PACKET_KINDS:
        raise ValueError(f"kind must be one of {PACKET_KINDS}, got {kind!r}")
    pairs = {"u-d": [(U, U, 1.0), (D, D, -1.0)], "u+d": [(U, U, 1.0), (D, D, 1.0)],
             "+": [(U, D, 1.0)], "-": [(D, U, 1.0)]}[kind]
    os = OpSum()
    for n in range(N - 1):
        if f[n] == 0:
            continue
        for fa, fb, w in pairs:
            os += (w * f[n], "Cdag", qubit(n, fa), "C", qubit(n + 1, fb))
            os += (-w * f[n], "Cdag", qubit(n + 1, fa), "C", qubit(n, fb))
    return os


def wide_packet_opsum(N: int, kind: str, c: float, k: float, sigma: float, phi, lo: int = 0,
                      hi: int | None = None) -> OpSum:
    """sum_n exp(-(n - c)^2 / (2 sigma^2) + i k n) sum_{r=1..R} phi[r-1] O^r_n, a wide packet.

    The range-r pair operators generalise the link operators of `packet_opsum` (r = 1):

        "+"    O^r_n = c+_{n,u} c_{n+r,d} + (-1)^r c+_{n+r,u} c_{n,d}            (pi+)
        "-"    O^r_n = c+_{n,d} c_{n+r,u} + (-1)^r c+_{n+r,d} c_{n,u}            (pi-)
        "u-d"  O^r_n = sum_f w_f (c+_{n,f} c_{n+r,f} + (-1)^r c+_{n+r,f} c_{n,f}), w = (1, -1)
        "u+d"  as "u-d" with w = (1, 1)

    (Jordan-Wigner strings from the compiler.) A term is kept when lo <= n and n + r <= hi + 1
    (hi defaults to N - 2), so a packet stays in its half of the chain. phi = (1,) is
    `packet_opsum(N, kind, gaussian_profile(N, c, k, sigma, lo, hi))`.
    """
    if kind not in PACKET_KINDS:
        raise ValueError(f"kind must be one of {PACKET_KINDS}, got {kind!r}")
    pairs = {"u-d": [(U, U, 1.0), (D, D, -1.0)], "u+d": [(U, U, 1.0), (D, D, 1.0)],
             "+": [(U, D, 1.0)], "-": [(D, U, 1.0)]}[kind]
    hi = N - 2 if hi is None else min(hi, N - 2)
    os = OpSum()
    for n in range(max(lo, 0), hi + 1):
        a = np.exp(-(n - c) ** 2 / (2 * sigma ** 2) + 1j * k * n)
        for r, p in enumerate(phi, start=1):
            if p == 0 or n + r > hi + 1:
                continue
            for fa, fb, w in pairs:
                os += (a * p * w, "Cdag", qubit(n, fa), "C", qubit(n + r, fb))
                os += ((-1) ** r * a * p * w, "Cdag", qubit(n + r, fa), "C", qubit(n, fb))
    return os


def gaussian_profile(N: int, c: float, k: float, sigma: float, lo: int = 0, hi: int | None = None):
    """f_n = exp(-(n - c)^2 / (2 sigma^2) + i k n) on links lo..hi (inclusive), zero elsewhere."""
    hi = N - 2 if hi is None else min(hi, N - 2)
    n = np.arange(N - 1)
    f = np.exp(-(n - c) ** 2 / (2 * sigma ** 2) + 1j * k * n)
    f[(n < max(lo, 0)) | (n > hi)] = 0
    return f


# --------------------------------------------------------------------------- observables
NOP = np.diag([0.0, 1.0])
CAN = np.array([[0.0, 1.0], [0.0, 0.0]])          # annihilator in the spinless_fermion basis


def occupations(mps, canon=None) -> np.ndarray:
    """<n_{n,f}> as an (N, 2) array (column 0 = u)."""
    from .observables import local_expectations
    occ = np.real(np.asarray(local_expectations(mps, NOP, canon=canon)))
    return occ.reshape(-1, 2)


def flavour_profiles(occ, ell=0.0) -> dict:
    """From `occupations`: per-flavour charge q_f = n_f - (1 - (-1)^n)/2, total charge q, the
    isospin density i_z = (n_u - n_d)/2, and the flux <L_n> = ell + sum_{k<=n} q_k (N-1 links)."""
    occ = np.asarray(occ, dtype=float)
    s = _stag(len(occ))
    qf = occ - 0.5 * (1 - s)[:, None]
    q = qf.sum(1)
    L = np.broadcast_to(np.asarray(ell, dtype=float), (len(q) - 1,)) + np.cumsum(q)[:-1]
    return {"q_u": qf[:, 0], "q_d": qf[:, 1], "q": q, "iz": 0.5 * (occ[:, 0] - occ[:, 1]), "L": L}


def energy_density(mps, x: float, mu, ell=0.0, charge_penalty: float = 0.0, canon=None):
    """Local pieces of <W> for `schwinger2f_mpo(N, x, mu, ell, charge_penalty)`.

    Returns ``(hop, mass, electric, penalty)`` as numpy: ``hop`` (2, N-1), x <hop_f(n) + h.c.>
    per flavour and link; ``mass`` (N, 2), mu_{n,f} [(1 - (-1)^n)/2 + (-1)^n <n_{n,f}>];
    ``electric`` (N-1,), <L_n^2>; ``penalty``, charge_penalty <Q^2>. Their total is <W>.
    One sweep over the right-canonical state on the state's own backend, O(N chi^3), carrying the
    norm, sum n_j and (sum n_j)^2 environments and the open hops; one host transfer.
    """
    from .observables import _backend_for, _canon, _profile
    be = _backend_for(mps)
    M = _canon(mps, be) if canon is None else canon
    N2 = len(M)
    N = N2 // 2
    mu, ell = _params(N, mu, ell, charge_penalty)
    F, nop, cop, cdg = (be.array(o) for o in (SZ.real.copy(), NOP, CAN, CAN.T.copy()))
    A0, B, C = be.ones((1, 1)), be.zeros((1, 1)), be.zeros((1, 1))
    opened = {}                     # j -> environment with c+_j, then F_{j+1}
    hops, occs, s1, s2 = [], [], [], []

    def step(E, A, Ac, op=None):
        if op is None:
            return be.einsum("bsc,bsd->cd", be.einsum("ab,asc->bsc", E, A), Ac)
        return be.einsum("btd,btc->cd", Ac, be.einsum("bsc,ts->btc", be.einsum("ab,asc->bsc", E, A), op))

    for j in range(N2):
        A = M[j]
        Ac = be.conj(A)
        if j - 2 in opened:         # c+_j F_{j+1} c_{j+2}; hop + h.c. = 2 Re of it
            E = opened.pop(j - 2)
            t = be.einsum("bsc,ts->btc", be.einsum("ab,asc->bsc", E, A), cop)
            hops.append(be.einsum("btc,btc->", t, Ac))
        if j - 1 in opened:
            opened[j - 1] = step(opened[j - 1], A, Ac, F)
        TzA = step(A0, A, Ac, nop)
        TzB = step(B, A, Ac, nop)
        TA = step(A0, A, Ac)
        C = step(C, A, Ac) + 2 * TzB + TzA
        B = step(B, A, Ac) + TzA
        if j + 2 < N2:
            opened[j] = step(A0, A, Ac, cdg)
        A0 = TA
        occs.append(be.einsum("aa->", TzA))
        s1.append(be.einsum("aa->", B))
        s2.append(be.einsum("aa->", C))
    v = _profile(be, hops + occs + s1 + s2)
    nh = len(hops)
    hop = (2 * v[:nh]).reshape(N - 1, 2).T          # hop index j - 2 = 2n + f in order
    occ, S1, S2 = v[nh:nh + N2], v[nh + N2:nh + 2 * N2], v[nh + 2 * N2:]
    s = _stag(N)
    b = ell - np.cumsum(1 - s)[:-1]
    cut = 2 * np.arange(N - 1) + 1
    electric = b ** 2 + 2 * b * S1[cut] + S2[cut]
    mass = mu * (0.5 * (1 - s)[:, None] + s[:, None] * occ.reshape(N, 2))
    q0 = -(1 - s).sum()
    penalty = charge_penalty * (q0 ** 2 + 2 * q0 * S1[-1] + S2[-1])
    return x * hop, mass, electric, float(penalty)
