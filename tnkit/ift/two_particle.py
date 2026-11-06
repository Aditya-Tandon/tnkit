"""The two-particle variational excitation problem in the relative-SEPARATION basis.

Following Vanderstraeten, Verstraete, Haegeman, PRB 92, 125136 (2015), arXiv:1506.01008: label
two-particle states by the total momentum K and the relative separation n, keep a small reduced
basis of ell one-particle solutions per particle at n > 0, and let the n = 0 "contact" block carry
the full (d-1)D^2 tangent-space freedom.

    |chi_K(n, mu, nu)>  =  sum_j  e^{iK(j + n/2)} | ... A_L  B_mu(j)  A_L ... A_L  B_nu(j+n)  A_R ... >

with the A_L-left / A_R-right window convention (A_L to the left of EVERY insertion, A_R to the
right of every insertion) -- the earlier
A_R-in-the-middle form was a gauge bug that made the energy exactly separation-independent.

WHY THE SEPARATION BASIS
------------------------
A state bound by 0.55% has relative extent ~ 1/kappa with kappa = 0.104528 m_1, i.e. about ten
correlation lengths -- an unmissable exponential in POSITION space, and a feature 2x finer than a
practical momentum grid. An earlier O(70^2) costing used a momentum grid and was the
wrong parametrisation.

ell IS NOT THE NUMBER OF STABLE PARTICLES
-----------------------------------------
The one thing that must not be got wrong. The paper builds its reduced basis "from a number of B's
at different momenta": ell counts how many rigid tensors are needed to represent the MOMENTUM
DEPENDENCE of a single quasiparticle branch, not how many branches sit below threshold. Measured
(g=1, h=0.05): a single rigid orbital inflates the inverse
effective mass of the relative motion by 48.5%; ell = 2 gives 0.46%, ell = 3 gives 0.006%,
ell = 4 gives 2e-8. Since the m_3 binding is 0.55% of the threshold, ell = 1 is fatal and ell = 3
is ample. Cost of the contact block scales as ell^4, so ell = 3 is also the affordability line.

THE ENERGY ZERO, AND WHY IT AVOIDS A PRIOR REFERENCE BUG
--------------------------------------------------------
Every matrix element here is returned as <a|H|b> - e_vac <a|b>, with e_vac the energy of the
VACUUM window contracted through the SAME fixed F_L/F_R environments at the SAME window length.
An earlier two-particle reference bug (a localised window's energy carries an environment-dependent
offset that depends on which block structure the reference used) cannot enter, because no
absolute one-particle number is ever imported: m_1 and the two-particle levels are eigenvalues of
ONE matrix built from ONE set of environments, and every physical statement below is a difference
or ratio within that matrix.
"""
from __future__ import annotations

import numpy as np

from ..backend import PlainEinsumNumpyBackend
from ..excitations import null_space_VL


# --------------------------------------------------------------------------- #
# transfer steps with independent bra and ket
#   L, R : (bra_bond, mpo_bond, ket_bond)     Ln, Rn : (ket_bond, bra_bond)
# --------------------------------------------------------------------------- #
def _lstep(L, bra, ket, W, be):
    t1 = be.einsum("pwq,psd->wqsd", L, be.conj(bra))
    t2 = be.einsum("wqsd,wrsy->qdry", t1, W)
    return be.einsum("qdry,qye->dre", t2, ket)


def _rstep(R, bra, ket, W, be):
    t1 = be.einsum("xry,usx->ryus", R, be.conj(bra))
    t2 = be.einsum("ryus,wrsp->yuwp", t1, W)
    return be.einsum("yuwp,vpy->uwv", t2, ket)


def _lstep_b(L, bra, ket, W, be):
    """Batched left step. L is (batch, bra_bond, mpo, ket_bond); `bra`/`ket` are either a single
    (D,d,D) tensor shared by the whole batch or a (batch,D,d,D) stack."""
    bb = "b" if bra.ndim == 4 else ""
    kk = "b" if ket.ndim == 4 else ""
    t1 = be.einsum(f"bpwq,{bb}psd->bwqsd", L, be.conj(bra))
    t2 = be.einsum("bwqsd,wrsy->bqdry", t1, W)
    return be.einsum(f"bqdry,{kk}qye->bdre", t2, ket)


def _lstep_nb(Ln, bra, ket, be):
    """Batched left step for the overlap. Ln is (batch, ket_bond, bra_bond)."""
    bb = "b" if bra.ndim == 4 else ""
    kk = "b" if ket.ndim == 4 else ""
    t1 = be.einsum(f"bax,{kk}asc->bxsc", Ln, ket)
    return be.einsum(f"bxsc,{bb}xsy->bcy", t1, be.conj(bra))


def _lstep_n(Ln, bra, ket, be):
    return be.einsum("ax,asb,xsy->by", Ln, ket, be.conj(bra))


def _rstep_n(Rn, bra, ket, be):
    return be.einsum("by,asb,xsy->ax", Rn, ket, be.conj(bra))


def _c_order(x):
    """`x` in C order. `np.stack` keeps its inputs' strides, and the batched einsums round
    differently on a different layout; on torch the stack is already contiguous."""
    return x.reshape(-1).reshape(x.shape)


class SeparationBasis:
    """Matrix elements of H and of the overlap in the relative-separation two-particle basis.

    Everything is at total momentum K = 0 (the sector the E_8 masses live in). At K = 0 all the
    e^{iK...} phases are unity, so the translation sum over the pair's centre reduces to a plain
    sum over the RELATIVE shift Delta between the bra pair and the ket pair.

    THE GAUGE DOES HALF THE WORK. Both B tensors are built from V_L, the null space of A_L, so
    sum_s A_L[s]^dag B[s] = 0. Propagating the identity environment through a site where one side
    has B and the other has A_L therefore annihilates it. Consequence, and it is exact rather than
    approximate: the OVERLAP matrix elements vanish unless the leftmost insertions of bra and ket
    coincide, i.e. only Delta = 0 contributes to N. H is not protected in the same way (the MPO
    environment is not the identity), so H needs the full Delta sum -- which is why the two are
    computed by different routines below and why `check_norm_delta_selection` exists.

    `be` is the complex128 backend the window contractions run on; the default is numpy with
    plain `np.einsum`, so results do not depend on einsum plans. The assembled (H, N) blocks
    are always host arrays.
    """

    def __init__(self, AL, AR, C, W, FL, FR, orbitals, nsites, dmax=8, margin=8, be=None):
        if be is None:
            be = PlainEinsumNumpyBackend()
        if "complex128" not in str(be.dtype):
            raise TypeError(f"SeparationBasis needs a complex128 backend, got {be.dtype}")
        self.be = be
        self.AL, self.AR, self.C = be.array(AL), be.array(AR), be.array(C)
        self.W, self.FL, self.FR = be.array(W), be.array(FL), be.array(FR)
        self.D, self.d, _ = (int(n) for n in self.AL.shape)
        self.nX = self.D * (self.d - 1)
        self.NS, self.dmax, self.margin = int(nsites), int(dmax), int(margin)
        self.VL = null_space_VL(self.AL, be)

        self.orbitals = [be.array(X).reshape(self.nX, self.D) for X in orbitals]
        self.ell = len(self.orbitals)
        self.Bs = [be.einsum("asx,xb->asb", self.VL, X) for X in self.orbitals]
        self._Bst = _c_order(be.stack_axis(self.Bs, axis=0))       # (ell, D, d, D)

        self.AC = be.einsum("asb,bc->asc", self.AL, self.C)
        self._eye = be.array(np.eye(self.D))
        self.e_vac = self._vacuum_energy()

    # -- windows ------------------------------------------------------------ #
    def _win(self, ins):
        """ins = {position: orbital index}. A_L left of every insertion, A_R right of every one."""
        pmax = max(ins)
        return [self.Bs[ins[j]] if j in ins else (self.AL if j < pmax else self.AR)
                for j in range(self.NS)]

    def _vac_win(self):
        c = self.NS // 2
        return [self.AL if j < c else (self.AC if j == c else self.AR) for j in range(self.NS)]

    # -- scalar contractions ------------------------------------------------ #
    def energy(self, wa, wb):
        be, L = self.be, self.FL
        for a, b in zip(wa, wb):
            L = _lstep(L, a, b, self.W, be)
        return be.to_scalar(be.einsum("akb,akb->", L, self.FR))

    def overlap(self, wa, wb):
        be, Ln = self.be, self._eye
        for a, b in zip(wa, wb):
            Ln = _lstep_n(Ln, a, b, be)
        return be.to_scalar(be.einsum("ab,ab->", Ln, self._eye))

    def _vacuum_energy(self):
        w = self._vac_win()
        return (self.energy(w, w) / self.overlap(w, w)).real

    # -- anchor positions --------------------------------------------------- #
    def _anchor(self):
        return self.margin + self.dmax

    def _fits(self, n, npr, delta):
        j0 = self._anchor()
        lo = min(j0, j0 + delta)
        hi = max(j0 + n, j0 + delta + npr)
        return lo >= self.margin and hi <= self.NS - 1 - self.margin

    # -- two-particle / two-particle blocks --------------------------------- #
    def _batch_index(self):
        """Index arrays for the ell^4 (bra-pair, ket-pair) orbital combinations.

        Batching is not an optimisation detail, it is what makes the route affordable. The naive
        loop costs ell^4 full window sweeps per (n, n', Delta); at ell = 3, chi = 16 and a 60-site
        window that is ~10 hours for one model point. But at all but FOUR sites of the window every
        combination carries the SAME pair of tensors, so the sweep can be done once with a leading
        batch axis and only the four insertion sites need per-combination tensors.
        """
        L = self.ell
        idx = np.arange(L ** 4)
        ia, ib = idx // (L * L), idx % (L * L)
        return ia // L, ia % L, ib // L, ib % L          # mp, npp, mu, nu

    def pair_blocks(self, n, npr, dmax=None):
        """(H, N) blocks between bra separation `npr` and ket separation `n`.

        (ell^2, ell^2) matrices indexed by (mu', nu') x (mu, nu), already referred to the vacuum
        energy zero. The largest term at the edge of the Delta sum is returned so the truncation is
        checked rather than trusted -- it decays exponentially with the correlation length
        (measured ratio 0.34 per two sites at g = 1.5), so it is a scaling statement, not a
        tolerance.
        """
        be = self.be
        dmax = self.dmax if dmax is None else int(dmax)
        L, L2 = self.ell, self.ell ** 2
        j0 = self._anchor()
        mp, npp, mu, nu = (self._Bst[i] for i in self._batch_index())   # (ell^4, D, d, D)
        nb = L ** 4
        Lh0 = _c_order(be.stack_axis([self.FL] * nb, axis=0))
        Ln0 = _c_order(be.stack_axis([self._eye] * nb, axis=0))
        H = np.zeros((L2, L2), dtype=complex)
        N = np.zeros((L2, L2), dtype=complex)
        tail = 0.0

        for delta in range(-dmax, dmax + 1):
            if not self._fits(n, npr, delta):
                raise ValueError(f"window too small: n={n} n'={npr} delta={delta} NS={self.NS}")
            pa1, pa2 = j0 + delta, j0 + delta + npr
            pb1, pb2 = j0, j0 + n

            Lh, Ln = Lh0, Ln0
            for k in range(self.NS):
                bra = (mp if k == pa1 else npp if k == pa2 else
                       (self.AL if k < pa2 else self.AR))
                ket = (mu if k == pb1 else nu if k == pb2 else
                       (self.AL if k < pb2 else self.AR))
                Lh = _lstep_b(Lh, bra, ket, self.W, be)
                Ln = _lstep_nb(Ln, bra, ket, be)
            e = np.asarray(be.to_numpy(be.einsum("bakc,akc->b", Lh, self.FR)))
            o = np.einsum("baa->b", np.asarray(be.to_numpy(Ln)))
            contrib = e - self.e_vac * o
            H += contrib.reshape(L2, L2)
            N += o.reshape(L2, L2)
            if abs(delta) == dmax:
                tail = max(tail, float(np.abs(contrib).max()))
        return H, N, tail

    # -- contact (n = 0) block cross terms ---------------------------------- #
    def cross_blocks(self, n):
        """(H, N) between the FULL one-particle tangent space (bra) and separation n (ket).

        Shape (nX*D, ell^2). The sum over the bra insertion site is done with ONE forward sweep
        (bra = A_L everywhere) and ONE backward sweep (bra = A_R everywhere): the bra window for an
        insertion at j' is A_L for k < j', open at j', A_R for k > j', so the forward sweep's
        prefixes and the backward sweep's suffixes are exactly the environments needed at every j'
        simultaneously. That is what makes the cross block cost two sweeps instead of NS of them.
        """
        be, L = self.be, self.ell
        j0 = self._anchor()
        VLc = be.conj(self.VL)
        Hc = np.zeros((self.nX * self.D, L * L), dtype=complex)
        Nc = np.zeros((self.nX * self.D, L * L), dtype=complex)
        for mu in range(L):
            for nu in range(L):
                wb = self._win({j0: mu, j0 + n: nu})
                Lenv = [None] * (self.NS + 1)
                Lenv[0] = self.FL
                for k in range(self.NS):
                    Lenv[k + 1] = _lstep(Lenv[k], self.AL, wb[k], self.W, be)
                Renv = [None] * (self.NS + 1)
                Renv[self.NS] = self.FR
                for k in range(self.NS - 1, -1, -1):
                    Renv[k] = _rstep(Renv[k + 1], self.AR, wb[k], self.W, be)
                Ln = [None] * (self.NS + 1)
                Ln[0] = self._eye
                for k in range(self.NS):
                    Ln[k + 1] = _lstep_n(Ln[k], self.AL, wb[k], be)
                Rn = [None] * (self.NS + 1)
                Rn[self.NS] = self._eye
                for k in range(self.NS - 1, -1, -1):
                    Rn[k] = _rstep_n(Rn[k + 1], self.AR, wb[k], be)

                G = be.zeros((self.D, self.d, self.D))
                Gn = be.zeros((self.D, self.d, self.D))
                for jp in range(self.NS):
                    G += be.einsum("pwq,wrst,qty,cry->psc", Lenv[jp], self.W, wb[jp],
                                   Renv[jp + 1])
                    Gn += be.einsum("ax,asb,by->xsy", Ln[jp], wb[jp], Rn[jp + 1])
                ib = mu * L + nu
                # <X'|...> = sum conj(X'[x,b]) (V_L^dag G)[x,b]
                Hc[:, ib] = be.to_numpy(be.einsum("asx,asb->xb", VLc, G)).reshape(-1)
                Nc[:, ib] = be.to_numpy(be.einsum("asx,asb->xb", VLc, Gn)).reshape(-1)
        Hc = Hc - self.e_vac * Nc
        return Hc, Nc

    # -- checks ------------------------------------------------------------- #
    def check_norm_delta_selection(self, n=4, npr=4, dmax=3):
        """The gauge condition must make <bra|ket> vanish for every Delta != 0. EXACT, not a
        tolerance: if this is not satisfied the whole separation basis is mis-gauged."""
        j0 = self._anchor()
        out = []
        for delta in range(-dmax, dmax + 1):
            wa = self._win({j0 + delta: 0, j0 + delta + npr: 0})
            wb = self._win({j0: 0, j0 + n: 0})
            out.append((delta, abs(self.overlap(wa, wb))))
        return out

    def one_particle_window_element(self, X1, X2, dmax=None):
        """<Phi_0(V_L X1)| H - E_vac |Phi_0(V_L X2)> by explicit window contraction, Delta-summed.

        The point of this routine is that it must reproduce X1^dag H_eff(0) X2 from
        QuasiparticleSolver.apply, which shares NO code with it -- one contracts explicit window
        tensors against F_L/F_R, the other never forms a window. Same kind of check as the one that
        caught the two_particle_state gauge bug.
        """
        be = self.be
        dmax = self.dmax if dmax is None else int(dmax)
        j0 = self._anchor()
        B1 = be.einsum("asx,xb->asb", self.VL, be.array(X1).reshape(self.nX, self.D))
        B2 = be.einsum("asx,xb->asb", self.VL, be.array(X2).reshape(self.nX, self.D))
        tot = 0.0 + 0.0j
        for delta in range(-dmax, dmax + 1):
            wa = [B1 if j == j0 + delta else (self.AL if j < j0 + delta else self.AR)
                  for j in range(self.NS)]
            wb = [B2 if j == j0 else (self.AL if j < j0 else self.AR)
                  for j in range(self.NS)]
            tot += self.energy(wa, wb) - self.e_vac * self.overlap(wa, wb)
        return tot


def compute_pair_blocks(sep, n_comp, dmax=None, verbose=True):
    """All (H, N) blocks with 1 <= n <= n' <= n_comp, plus the largest discarded Delta term."""
    Hb, Nb, tails = {}, {}, []
    for n in range(1, n_comp + 1):
        for npr in range(n, n_comp + 1):
            h, nn, tail = sep.pair_blocks(n, npr, dmax=dmax)
            Hb[(npr, n)], Nb[(npr, n)] = h, nn
            tails.append(tail)
        if verbose:
            print(f"    pair blocks, ket separation n={n}/{n_comp}", flush=True)
    return Hb, Nb, (max(tails) if tails else 0.0)


def _get(Hb, Nb, npr, n):
    if (npr, n) in Hb:
        return Hb[(npr, n)], Nb[(npr, n)]
    return Hb[(n, npr)].conj().T, Nb[(n, npr)].conj().T


def asymptotic_hopping(Hb, Nb, n_comp, m_range):
    """T(m) = H(n0 + m, n0) read off at n0 = n_comp - m_range, with the DRIFT between
    consecutive n0 reported.

    Beyond the interaction range the blocks must depend on n' and n only through m = n' - n. That
    is a property to be MEASURED, not assumed: `drift` is how much T(m) still moves when the anchor
    n0 is shifted by one, and it is what licenses (or refuses) continuation to large n_max.
    """
    n0 = n_comp - m_range
    if n0 < 1 + m_range:
        raise ValueError("n_comp too small for the requested hopping range")
    Tm, Sm, drift = {}, {}, 0.0
    for m in range(-m_range, m_range + 1):
        h0, s0 = _get(Hb, Nb, n0 + m, n0)
        h1, s1 = _get(Hb, Nb, n0 - 1 + m, n0 - 1)
        Tm[m], Sm[m] = h0, s0
        drift = max(drift, float(np.abs(h0 - h1).max()))
    return Tm, Sm, drift, n0


def asymptotic_hopping_extrap(Hb, Nb, n_comp, m_range):
    """T(m) by RICHARDSON EXTRAPOLATION of H(n+m, n) in the anchor separation n.

    Reading T(m) off a single anchor n_0 is not accurate enough, and the reason is structural
    rather than a matter of pushing n_comp up: the residual interaction decays like exp(-n/xi) with
    xi ~ 1.5 sites, but the anchor must simultaneously satisfy n_0 + m <= n_comp, so raising the
    hopping range m_range LOWERS the anchor and the two requirements fight. Measured at
    g = 1.5, chi = 8, n_comp = 12: drift 1.3e-2 at m_range 1 (anchor 11) but 1.8e-1 at m_range 4
    (anchor 8) -- and 1.3e-2 is already the size of the m_3 binding we are trying to resolve.

    The sequence H(n+m, n) approaches its asymptote geometrically, so one Richardson step removes
    the leading term:

        T = H_n + d_n * r/(1-r),    d_n = H_n - H_{n-1},    r = ||d_n|| / ||d_{n-1}||

    The size of the correction is returned as `corr`; it IS the uncertainty on the continuation,
    and it must be small against the binding energy being tested. Reporting it is the test --
    an extrapolation whose correction is the size of the effect is not an extrapolation.
    """
    Tm, Sm = {}, {}
    corr, ratios = 0.0, []
    for m in range(-m_range, m_range + 1):
        seq = []
        for n in range(1, n_comp + 1):
            if 1 <= n + m <= n_comp:
                seq.append((n, _get(Hb, Nb, n + m, n)))
        if len(seq) < 3:
            raise ValueError(f"not enough anchors for m={m}; raise n_comp")
        (n2, (h2, s2)), (n1, (h1, s1)), (n0, (h0, s0)) = seq[-1], seq[-2], seq[-3]
        d1, d0 = h2 - h1, h1 - h0
        nd1, nd0 = np.linalg.norm(d1), np.linalg.norm(d0)
        r = float(nd1 / nd0) if nd0 > 1e-300 else 0.0
        r = min(max(r, 0.0), 0.95)
        ratios.append(r)
        dT = d1 * (r / (1.0 - r)) if r < 0.95 else d1 * 19.0
        Tm[m] = h2 + dT
        e1, e0 = s2 - s1, s1 - s0
        ne1, ne0 = np.linalg.norm(e1), np.linalg.norm(e0)
        rs = min(max(float(ne1 / ne0) if ne0 > 1e-300 else 0.0, 0.0), 0.95)
        Sm[m] = s2 + e1 * (rs / (1.0 - rs))
        corr = max(corr, float(np.abs(dT).max()))
    return Tm, Sm, corr, float(np.mean(ratios))


def hopping_range_tail(Tm, m_range):
    """|T(m)| at the edge of the retained band -- what the band truncation throws away."""
    return max(float(np.abs(Tm[m]).max()) for m in (-m_range, m_range))


def assemble(sep, Hb, Nb, n_max, n_comp, m_range):
    """(H, N, n1) in the basis {contact block} + {|n, mu, nu> : n = 1..n_max}.

    n1 = dimension of the contact block, so rows/cols [0:n1) are the full one-particle tangent
    space at K = 0 and the rest is the two-particle ladder. The contact block itself is NOT filled
    here -- it is the one-particle H_eff(0), which comes from the already-validated
    QuasiparticleSolver and is inserted by the caller.
    """
    L2 = sep.ell ** 2
    n1 = sep.nX * sep.D
    dim = n1 + L2 * n_max
    H = np.zeros((dim, dim), dtype=complex)
    N = np.zeros((dim, dim), dtype=complex)
    Tm, Sm, drift, n0 = asymptotic_hopping(Hb, Nb, n_comp, m_range)

    for n in range(1, n_max + 1):
        for npr in range(1, n_max + 1):
            m = npr - n
            if abs(m) > m_range:
                continue
            i, j = n1 + (npr - 1) * L2, n1 + (n - 1) * L2
            if n <= n_comp and npr <= n_comp:
                h, s = _get(Hb, Nb, npr, n)
            else:
                h, s = Tm[m], Sm[m]
            H[i:i + L2, j:j + L2] = h
            N[i:i + L2, j:j + L2] = s
    return H, N, n1, drift
