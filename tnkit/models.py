"""Spin-chain models: transverse-field Ising MPOs, related operators, and exact references.

The transverse-field Ising model (TFIM), with Pauli matrices (not spin operators S = sigma/2):

    H = - sum_i sigma^x_i sigma^x_{i+1}  -  g sum_i sigma^z_i  -  h sum_i sigma^x_i,

open boundaries unless stated, critical at g = 1 for h = 0. Sites are numbered from 0, and
the local basis is the sigma^z basis (|up>, |down>), with sigma^z = diag(1, -1).

An MPO is a list of rank-4 tensors W[site] with index order
(left_bond, right_bond, phys_out, phys_in) = (a, b, s, s').
"""
from __future__ import annotations
import numpy as np

I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
ZERO = np.zeros((2, 2), dtype=np.complex128)


def parity_mpo(N):
    """MPO for the ℤ₂ parity operator P = ∏_i σᶻ_i (bond dimension 1). [H_TFIM, P]=0;
    the disordered ground state has P=+1. Used to measure ⟨P⟩ and to target a sector."""
    return [SZ.reshape(1, 1, 2, 2).copy() for _ in range(N)]


def tfim_parity_mpo(N, g, lam):
    """MPO for H_TFIM − λ·P  (P = ∏σᶻ), a **sector-penalised** TFIM. λ splits the two
    ℤ₂ parity sectors so DMRG converges to a definite-parity ground state instead of a
    symmetry-broken cat / random sector, fixing ℤ₂ sector-jumping. λ>0
    favours P=+1. The physical energy is ⟨H_TFIM⟩ (measure with tfim_mpo)."""
    Wb = np.zeros((4, 4, 2, 2), dtype=np.complex128)
    Wb[0, 0] = I2; Wb[0, 1] = SX; Wb[0, 2] = -g * SZ; Wb[1, 2] = -SX; Wb[2, 2] = I2
    Wb[3, 3] = SZ                                   # parity string channel (decoupled)
    vL = np.array([1, 0, 0, -lam], dtype=np.complex128)
    vR = np.array([0, 0, 1, 1], dtype=np.complex128)
    mpo = []
    for i in range(N):
        if i == 0:
            mpo.append(np.einsum("a,abij->bij", vL, Wb).reshape(1, 4, 2, 2))
        elif i == N - 1:
            mpo.append(np.einsum("abij,b->aij", Wb, vR).reshape(4, 1, 2, 2))
        else:
            mpo.append(Wb.copy())
    return mpo


def tfim_mpo(N, g, h=0.0):
    """MPO of the open transverse-field Ising chain.

    H = - sum_i sigma^x_i sigma^x_{i+1} - g sum_i sigma^z_i - h sum_i sigma^x_i, with Pauli
    matrices and open boundaries. Bulk tensor and boundary vectors L = (1, 0, 0),
    R = (0, 0, 1)^T::

        W = [[ I,   sx,  -g sz - h sx],
             [ 0,    0,   -sx        ],
             [ 0,    0,    I         ]]

    Parameters
    ----------
    N : int
        Number of sites, >= 2.
    g : float
        Transverse field; the chain is critical at ``g = 1`` when ``h = 0``.
    h : float, default 0.0
        Longitudinal field, conjugate to the order parameter sigma^x; the susceptibility is
        ``d<sigma^x>/dh``.

    Returns
    -------
    list of ndarray
        N complex tensors of shape ``(1, 3, 2, 2)``, ``(3, 3, 2, 2)`` (bulk) and
        ``(3, 1, 2, 2)``, axes ``(left, right, out, in)``.
    """
    def bulk():
        W = np.zeros((3, 3, 2, 2), dtype=np.complex128)
        W[0, 0] = I2
        W[0, 1] = SX
        W[0, 2] = -g * SZ - h * SX
        W[1, 2] = -SX
        W[2, 2] = I2
        return W

    Wbulk = bulk()
    mpo = []
    for site in range(N):
        if site == 0:
            mpo.append(Wbulk[0:1, :, :, :])        # row 0  -> shape (1,3,2,2)
        elif site == N - 1:
            mpo.append(Wbulk[:, 2:3, :, :])        # col 2  -> shape (3,1,2,2)
        else:
            mpo.append(Wbulk.copy())
    return mpo


def magnetization_mpo(N, op=SX):
    """MPO of a total magnetisation ``M = sum_i op_i``, bond dimension 2.

    Bulk ``W = [[I, op], [0, I]]``: channel 0 means "op not yet placed" and 1 "placed"; the
    left boundary starts in 0 and the right one ends in 1.

    Parameters
    ----------
    N : int
        Number of sites, >= 2.
    op : ndarray, default SX
        The 2 x 2 single-site operator (Pauli sigma^x by default, so M is not divided by 2).

    Returns
    -------
    list of ndarray
        N complex tensors, axes ``(left, right, out, in)``.
    """
    W = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    W[0, 0] = I2
    W[0, 1] = op
    W[1, 1] = I2
    mpo = []
    for site in range(N):
        if site == 0:
            mpo.append(W[0:1, :, :, :])
        elif site == N - 1:
            mpo.append(W[:, 1:2, :, :])
        else:
            mpo.append(W.copy())
    return mpo


def tfim_ring_mpo(N, g, h=0.0):
    """MPO of the periodic (ring) transverse-field Ising chain, in the convention of
    `tfim_mpo`:

        H = - sum_{i=0}^{N-1} sx_i sx_{i+1 mod N} - g sum_i sz_i - h sum_i sx_i

    The wrap bond (N-1, 0) is one extra automaton channel that opens at site 0 and closes at
    site N-1, so the MPO bond dimension is 4 rather than 3. The MPS stays open: a cut of the
    ring severs two bonds, so the bond dimension needed is about the square of the open
    chain's, which is affordable when the correlation length is short. A ring is what
    finite-volume (Luscher) spectroscopy assumes: its quantisation condition
    k L + 2 delta(k) = 2 pi n holds on a periodic ring, while an open chain adds boundary
    reflection phases.

    Automaton channels: 0 = start, 1 = nearest-neighbour carry, 2 = wrap carry, 3 = done.

    Parameters
    ----------
    N : int
        Number of sites, >= 3.
    g : float
        Transverse field.
    h : float, default 0.0
        Longitudinal field.

    Returns
    -------
    list of ndarray
        N complex tensors of bond dimension 4, axes ``(left, right, out, in)``.
    """
    if N < 3:
        raise ValueError("a ring needs N >= 3 (N = 2 would double-count the bond)")

    def bulk():
        W = np.zeros((4, 4, 2, 2), dtype=np.complex128)
        W[0, 0] = I2
        W[0, 1] = SX                       # open a NN bond
        W[0, 3] = -g * SZ - h * SX         # on-site fields
        W[1, 3] = -SX                      # close a NN bond
        W[2, 2] = I2                       # carry the wrap bond through the bulk
        W[3, 3] = I2
        return W

    mpo = []
    for i in range(N):
        W = bulk()
        if i == 0:
            W[0, 2] = SX                   # open the wrap bond at site 0
            mpo.append(W[0:1].copy())      # left boundary vector = e_START
        elif i == N - 1:
            W[2, 3] = -SX                  # close the wrap bond at site N-1
            mpo.append(W[:, 3:4].copy())   # right boundary vector = e_DONE
        else:
            mpo.append(W)
    return mpo


def translation_mpo(N, direction=+1):
    """MPO for the CYCLIC SHIFT on a ring of N sites (bond dimension 4).

    ``direction=+1`` gives T|s_1..s_N> = |s_N, s_1, .., s_{N-1}>; ``-1`` gives T^dagger.

    The shift is a "delta" MPO, W[a,b,t,s] = delta(t,a) delta(b,s), whose left and
    right boundary bonds must carry the *same* value (the value being wrapped around).
    That correlated boundary is a rank-2 condition, not a product, so T is the sum of
    two bond-dimension-2 open MPOs -- equivalently one bond-dimension-4 open MPO with
    bond index (c, a): c remembers the wrapped value, a carries the shifting value.

    Why this exists: on a ring the total momentum P is a good quantum number, and the
    zero-momentum two-particle levels that the Luscher condition needs are *interleaved*
    with a dense set of moving-frame levels (at m1 L ~ 100 there are of order 100 states
    below the sixth P = 0 level).  ``H + lambda (1 - (T + T^dag)/2)`` pushes every
    P != 0 state up by ``lambda (1 - cos P)`` while leaving P = 0 exactly untouched, so
    excited-state DMRG only has to climb the levels one actually wants.  <T> is also a
    cheap post-hoc momentum diagnostic: it must come out at exp(i P) = 1.
    """
    W = np.zeros((4, 4, 2, 2), dtype=np.complex128)
    for c in range(2):
        for a in range(2):
            for b in range(2):
                # row (c,a), col (c,b): t = a, s = b
                W[2 * c + a, 2 * c + b, a, b] = 1.0
    if direction < 0:
        W = np.transpose(W, (0, 1, 3, 2)).copy()
    vL = np.zeros(4, dtype=np.complex128)
    vR = np.zeros(4, dtype=np.complex128)
    for c in range(2):
        vL[2 * c + c] = 1.0
        vR[2 * c + c] = 1.0
    mpo = []
    for i in range(N):
        if i == 0:
            mpo.append(np.einsum("a,abij->bij", vL, W).reshape(1, 4, 2, 2))
        elif i == N - 1:
            mpo.append(np.einsum("abij,b->aij", W, vR).reshape(4, 1, 2, 2))
        else:
            mpo.append(W.copy())
    return mpo


def identity_mpo(N, coeff=1.0):
    """MPO for coeff * Identity (bond dimension 1)."""
    out = [I2.reshape(1, 1, 2, 2).copy() for _ in range(N)]
    out[0] = out[0] * coeff
    return out


def mpo_sum(mpos):
    """Direct sum of MPOs (already carrying their boundary vectors): bond dims add."""
    N = len(mpos[0])
    assert all(len(m) == N for m in mpos)
    out = []
    for i in range(N):
        blocks = [m[i] for m in mpos]
        if i == 0:
            out.append(np.concatenate(blocks, axis=1))
        elif i == N - 1:
            out.append(np.concatenate(blocks, axis=0))
        else:
            Ds = [b.shape for b in blocks]
            D0 = sum(x[0] for x in Ds)
            D1 = sum(x[1] for x in Ds)
            W = np.zeros((D0, D1, 2, 2), dtype=np.complex128)
            a = b = 0
            for blk in blocks:
                W[a:a + blk.shape[0], b:b + blk.shape[1]] = blk
                a += blk.shape[0]
                b += blk.shape[1]
            out.append(W)
    return out


def momentum_projected_ring_mpo(N, g, h=0.0, lam=0.0, lam_parity=0.0):
    """H_ring + lam (1 - (T+T^dag)/2) + lam_parity (1 - P)/2, P = prod sigma^z.

    Both added terms annihilate the sector one wants (total momentum zero, and -- when
    h = 0 -- even Z2 parity), so the *penalised* energy of an exact state in that sector
    equals its physical energy, and <H'> stays a rigorous variational upper bound on the
    lowest in-sector eigenvalue.  Everything outside is pushed up by lam(1 - cos P) or by
    lam_parity.

    Sizing lam: a nonzero-momentum state at P = 2 pi j / N is lifted by
    lam (1 - cos(2 pi j / N)) ~ lam (2 pi j / N)^2 / 2.  To clear a target window of
    height W above the ground state one needs lam > 2 W N^2 / (2 pi)^2 for j = 1, i.e.
    lam grows as N^2: a moderate lam may let a low P != 0 state into a tower of levels. A
    cheaper alternative to a large lam is a moderate one and a filter of the returned tower
    on <T>, a cheap MPO expectation value.
    """
    parts = [tfim_ring_mpo(N, g, h)]
    if lam != 0.0:
        parts.append(identity_mpo(N, lam))
        parts.append([-0.5 * lam * W if k == 0 else W
                      for k, W in enumerate(translation_mpo(N, +1))])
        parts.append([-0.5 * lam * W if k == 0 else W
                      for k, W in enumerate(translation_mpo(N, -1))])
    if lam_parity != 0.0:
        parts.append(identity_mpo(N, 0.5 * lam_parity))
        parts.append([-0.5 * lam_parity * W if k == 0 else W
                      for k, W in enumerate(parity_mpo(N))])
    return mpo_sum(parts) if len(parts) > 1 else parts[0]


def longrange_mpo(N, xx_bonds, gz):
    """Finite-automaton MPO for  H = -sum_{(i,j) in xx_bonds} sx_i sx_j - gz sum_i sz_i,
    with **arbitrary-range** XX bonds (i<j). Bond dimension = 2 + max bonds crossing
    a cut. This is the reusable long-range builder (2D cylinders, lattice gauge
    theories): a bond (i,j) opens a dedicated 'carry' channel at site i (sx), holds
    identity through i+1..j-1, and closes at site j (-sx). START/DONE channels let
    terms begin/end anywhere."""
    # unique bonds (a periodic ring of length 2 would otherwise double-count a rung)
    xx_bonds = sorted(set(tuple(sorted(map(int, b))) for b in xx_bonds if b[0] != b[1]))

    def channels(p):                                  # channel label -> index at cut p
        if p == 0:
            return {"START": 0}
        if p == N:
            return {"DONE": 0}
        ob = sorted([(i, j) for (i, j) in xx_bonds if i < p <= j])
        d = {"START": 0}
        for idx, b in enumerate(ob):
            d[b] = idx + 1
        d["DONE"] = len(ob) + 1
        return d

    chan = {p: channels(p) for p in range(N + 1)}
    mpo = []
    for k in range(N):
        cin, cout = chan[k], chan[k + 1]
        W = np.zeros((len(cin), len(cout), 2, 2), dtype=np.complex128)
        if "START" in cin and "START" in cout:
            W[cin["START"], cout["START"]] = I2                 # nothing placed yet
        if "START" in cin and "DONE" in cout:
            W[cin["START"], cout["DONE"]] = -gz * SZ            # on-site field
        for (i, j) in xx_bonds:
            if i == k and (i, j) in cout:                       # open bond at site k
                W[cin["START"], cout[(i, j)]] = SX
            if j == k and (i, j) in cin:                        # close bond at site k
                W[cin[(i, j)], cout["DONE"]] = -SX
            if i < k < j and (i, j) in cin and (i, j) in cout:  # carry through
                W[cin[(i, j)], cout[(i, j)]] = I2
        if "DONE" in cin and "DONE" in cout:
            W[cin["DONE"], cout["DONE"]] = I2                   # completed term carried
        mpo.append(W)
    return mpo


def _cylinder_bonds(Lx, Ly):
    """(N, pos, unique_bonds) for an Lx*Ly cylinder (periodic in y, open in x),
    snaked (boustrophedon) into 1D. Horizontal (x) bonds become long-range (~Ly)."""
    def pos(col, row):
        return col * Ly + (row if col % 2 == 0 else Ly - 1 - row)
    P = {(c, r): pos(c, r) for c in range(Lx) for r in range(Ly)}
    bonds = set()
    for c in range(Lx):
        for r in range(Ly):
            bonds.add(tuple(sorted((P[(c, r)], P[(c, (r + 1) % Ly)]))))   # vertical
            if c < Lx - 1:
                bonds.add(tuple(sorted((P[(c, r)], P[(c + 1, r)]))))      # horizontal
    return Lx * Ly, P, sorted(bonds)


def tfim2d_cylinder(Lx, Ly, g):
    """2D TFIM MPO on an Lx*Ly cylinder. Returns (mpo, N, pos). Use Ly>=3 for a true
    cylinder (Ly=2 is a ladder — the periodic rung dedupes to a single bond)."""
    N, P, bonds = _cylinder_bonds(Lx, Ly)
    return longrange_mpo(N, bonds, g), N, P


def tfim2d_cylinder_dense(Lx, Ly, g):
    """Dense 2D-cylinder TFIM Hamiltonian (validation, small N only), same bonds."""
    N, P, bonds = _cylinder_bonds(Lx, Ly)

    def op_at(op, site):
        m = np.array([[1.0 + 0j]])
        for k in range(N):
            m = np.kron(m, op if k == site else I2)
        return m
    H = np.zeros((2 ** N, 2 ** N), dtype=np.complex128)
    for (i, j) in bonds:
        H -= op_at(SX, i) @ op_at(SX, j)
    for k in range(N):
        H -= g * op_at(SZ, k)
    return H


def mpo_to_dense(mpo):
    """Contract an MPO into the full 2^N x 2^N operator (small N only)."""
    N = len(mpo)
    # start with left boundary bond = 1
    # full[a, (s-string), (s'-string)]
    full = mpo[0]                                   # (1,b,s,s')
    full = np.transpose(full, (0, 2, 3, 1))         # (1,s,s',b)
    for k in range(1, N):
        W = np.transpose(mpo[k], (0, 2, 3, 1))      # (a,s,s',b)
        full = np.tensordot(full, W, axes=([-1], [0]))  # contract bond
        # full now (1, s..., s'..., s_k, s'_k, b)  -> reorder phys
    # full shape: (1, s1,s1', s2,s2', ..., sN,sN', b=1)
    full = full[0, ..., 0]
    N2 = full.ndim // 2
    # interleave order is (s1,s1',s2,s2',...); split into out and in
    out_axes = list(range(0, full.ndim, 2))
    in_axes = list(range(1, full.ndim, 2))
    full = np.transpose(full, out_axes + in_axes)
    dim = 2 ** N
    return full.reshape(dim, dim)


def tfim_dense(N, g):
    """The dense open-chain TFIM Hamiltonian ``H = -sum sx_i sx_{i+1} - g sum sz_i``, built
    from its action on basis states in O(N 2^N): -g sum_i sz_i is diagonal,
    ``sz_i |s> = (1 - 2 b_i(s)) |s>``, and each bond term flips bits i and i+1. Site 0 is the
    most significant bit. Returns a complex 2^N x 2^N array.
    """
    dim = 1 << N
    H = np.zeros((dim, dim), dtype=np.complex128)
    s = np.arange(dim)

    diag = np.zeros(dim)
    for i in range(N):
        diag -= g * (1.0 - 2.0 * ((s >> (N - 1 - i)) & 1))
    H[s, s] = diag

    for i in range(N - 1):
        mask = (1 << (N - 1 - i)) | (1 << (N - 2 - i))
        H[s ^ mask, s] -= 1.0
    return H


def tfim_ground_energy(N, g):
    """Exact ground-state energy of the open TFIM chain at h = 0, by dense diagonalisation
    (`tfim_dense`); for N up to about 12.

    Parameters
    ----------
    N : int
        Number of sites, >= 2.
    g : float
        Transverse field.

    Returns
    -------
    float
        The lowest eigenvalue of H.
    """
    H = tfim_dense(N, g)
    w = np.linalg.eigvalsh(H)
    return float(w[0])
