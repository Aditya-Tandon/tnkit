"""SU(2) coefficients (`tnkit.su2`) and the `SU2` symmetry: CG orthogonality, F and R against
contractions of CG arrays, pentagon and hexagon, 6j identities, two independent oracles (sympy,
exact; spin matrices, numerical), U(1) x SU(2) products, and the two-flavour site multiplets.
Planted errors in phase conventions must be caught."""
import itertools
from fractions import Fraction

import numpy as np
import pytest

from tnkit import su2
from tnkit.models_lgt2f import isospin_sparse, site_charges_2f, unpack_charge
from tnkit.symmetry import ProductSymmetry, SU2, U1, Zn

S = SU2()
JMAX = 8                       # 2j <= 8: spins up to 4
LABELS = list(range(JMAX + 1))
TOL = 1e-12


# --------------------------------------------------------------------------- checkers
def _outs(sym, a, b):
    return [c for c, _ in sym.fusions(a, b)]


def cg_completeness_error(sym, labels):
    """max |C^T C - 1| and |C C^T - 1| over the full coupling matrix of every a x b."""
    err = 0.0
    for a, b in itertools.product(labels, repeat=2):
        C = np.concatenate([sym.cg(a, b, c).reshape(sym.dim(a) * sym.dim(b), -1)
                            for c in _outs(sym, a, b)], axis=1)
        eye = np.eye(C.shape[0])
        err = max(err, abs(C.T @ C - eye).max(), abs(C @ C.T - eye).max())
    return err


def f_against_cg_error(sym, labels, all_m=False):
    """max |F(a,b,c,d,e,f) - <(a (b c)_f)_d | ((a b)_e c)_d>|, the overlap built from CG arrays
    (at the highest weight of d, or at every pair of weights when `all_m`)."""
    err = 0.0
    for a, b, c in itertools.product(labels, repeat=3):
        left = {(e, d): np.einsum("xye,ezm->xyzm", sym.cg(a, b, e), sym.cg(e, c, d))
                for e in _outs(sym, a, b) for d in _outs(sym, e, c)}
        right = {(f, d): np.einsum("yzf,xfm->xyzm", sym.cg(b, c, f), sym.cg(a, f, d))
                 for f in _outs(sym, b, c) for d in _outs(sym, a, f)}
        for (e, d), L in left.items():
            for f in _outs(sym, b, c):
                F = sym.F(a, b, c, d, e, f)
                if (f, d) not in right:
                    err = max(err, abs(F))
                    continue
                Rv = right[(f, d)]
                if all_m:
                    got = np.einsum("xyzm,xyzn->mn", Rv, L)
                    err = max(err, abs(got - F * np.eye(len(got))).max())
                else:
                    err = max(err, abs(np.vdot(Rv[..., 0], L[..., 0]) - F))
    return err


def r_against_cg_error(sym, labels):
    """max |cg(a,b,c)[x,y,:] - R(a,b,c) cg(b,a,c)[y,x,:]|: swapping the factors of a coupled
    pair is R times the coupled state of the swapped pair."""
    err = 0.0
    for a, b in itertools.product(labels, repeat=2):
        for c in _outs(sym, a, b):
            got = sym.cg(a, b, c) - sym.R(a, b, c) * sym.cg(b, a, c).transpose(1, 0, 2)
            err = max(err, abs(got).max())
    return err


def f_unitarity_error(sym, labels):
    err = 0.0
    for a, b, c in itertools.product(labels, repeat=3):
        for d in {d for e in _outs(sym, a, b) for d in _outs(sym, e, c)}:
            es = [e for e in _outs(sym, a, b) if d in _outs(sym, e, c)]
            fs = [f for f in _outs(sym, b, c) if d in _outs(sym, a, f)]
            M = np.array([[sym.F(a, b, c, d, e, f) for f in fs] for e in es])
            err = max(err, abs(M @ M.T - np.eye(len(es))).max(),
                      abs(M.T @ M - np.eye(len(fs))).max())
    return err


def pentagon_error(sym, labels):
    """max over trees of |F(f,c,d,e,g,h) F(a,b,h,e,f,k) - sum_l F(a,b,c,g,f,l) F(a,l,d,e,g,k)
    F(b,c,d,k,l,h)|: (((a b)_f c)_g d)_e -> (a (b (c d)_h)_k)_e by two moves or three."""
    err = 0.0
    F = sym.F
    for a, b, c, d in itertools.product(labels, repeat=4):
        for f in _outs(sym, a, b):
            for g in _outs(sym, f, c):
                for e in _outs(sym, g, d):
                    for h in _outs(sym, c, d):
                        if e not in _outs(sym, f, h):
                            continue
                        for k in _outs(sym, b, h):
                            if e not in _outs(sym, a, k):
                                continue
                            lhs = F(f, c, d, e, g, h) * F(a, b, h, e, f, k)
                            rhs = sum(F(a, b, c, g, f, l) * F(a, l, d, e, g, k)
                                      * F(b, c, d, k, l, h) for l in _outs(sym, b, c))
                            err = max(err, abs(lhs - rhs))
    return err


def hexagon_error(sym, labels):
    """Both hexagons: moving a past (b c), and c past (a b), by F R F against R F R."""
    err = 0.0
    F, R = sym.F, sym.R
    for a, b, c in itertools.product(labels, repeat=3):
        for e in _outs(sym, a, b):
            for d in _outs(sym, e, c):
                for g in _outs(sym, a, c):
                    if d not in _outs(sym, b, g):
                        continue
                    lhs = sum(F(a, b, c, d, e, f) * R(a, f, d) * F(b, c, a, d, f, g)
                              for f in _outs(sym, b, c))
                    err = max(err, abs(lhs - R(a, b, e) * F(b, a, c, d, e, g) * R(a, c, g)))
        for f in _outs(sym, b, c):
            for d in _outs(sym, a, f):
                for g in _outs(sym, c, a):
                    if d not in _outs(sym, g, b):
                        continue
                    lhs = sum(F(a, b, c, d, e, f) * R(e, c, d) * F(c, a, b, d, g, e)
                              for e in _outs(sym, a, b))
                    err = max(err, abs(lhs - R(b, c, f) * F(a, c, b, d, g, f) * R(a, c, g)))
    return err


# --------------------------------------------------------------------------- the SU2 symmetry
def test_su2_sectors():
    assert (S.unit(), S.dual(3), S.dim(3), S.fusion) == (0, 3, 4, "simple")
    assert S.fusions(2, 3) == ((1, 1), (3, 1), (5, 1))
    assert S.fuse(0, 3) == S.fuse(3, 0) == 3
    with pytest.raises(ValueError, match="outcomes"):
        S.fuse(1, 1)
    for bad in (-1, 0.5, True):
        with pytest.raises(ValueError):
            S.label(bad)
    assert S.label(np.int64(4)) == 4 and sorted([3, 0, 2], key=S.sort_key) == [0, 2, 3]
    assert SU2() == SU2() and hash(SU2()) == hash(SU2())
    for a, b in itertools.product(LABELS, repeat=2):         # dimensions add up
        assert sum(S.dim(c) * n for c, n in S.fusions(a, b)) == S.dim(a) * S.dim(b)


def test_block_tensors_still_refuse_su2():
    from tnkit import blocks as B
    leg = B.Leg((0, 1), (1, 1), 1, S)
    with pytest.raises(NotImplementedError, match="needs unique fusion"):
        B.zeros((leg, leg.flip()), 0, np.zeros(1))


@pytest.mark.parametrize("sym, q", [(S, [0, 1, 1]), (ProductSymmetry((U1(), S)), [(0, 0), (1, 1)])])
def test_block_mpo_refuses_su2(sym, q):
    from tnkit import get_backend
    from tnkit.block_backend import mpo_charges, to_block_mpo
    d = len(q)
    W = [np.eye(d).reshape(1, 1, d, d)] * 2
    for f in (lambda: mpo_charges(W, [q] * 2, sym=sym),
              lambda: mpo_charges(W, None, sym=sym),            # refused before site_charges is read
              lambda: to_block_mpo(W, [q] * 2, get_backend("numpy"), sym=sym)):
        with pytest.raises(NotImplementedError, match="mpo_charges: .* needs unique fusion"):
            f()


def test_block_mpo_refuses_undeclared_fusion():
    from tnkit.block_backend import mpo_charges
    from tnkit.symmetry import Symmetry, U1

    class Bare(Symmetry):                       # `fusion` left at None
        unit, dual, fuse, label = U1.unit, U1.dual, U1.fuse, U1.label
    W = [np.eye(2).reshape(1, 1, 2, 2)] * 2
    with pytest.raises(NotImplementedError, match="needs unique fusion"):
        mpo_charges(W, [[0, 1]] * 2, sym=Bare())


# --------------------------------------------------------------------------- CG and F, R
def test_cg_orthogonality_and_completeness():
    assert cg_completeness_error(S, LABELS) < TOL


def test_cg_arrays_are_cached_and_read_only():
    assert S.cg(2, 3, 3) is S.cg(2, 3, 3)
    with pytest.raises(ValueError):
        S.cg(2, 3, 3)[0, 0, 0] = 1.0
    assert not S.cg(2, 2, 6).any() and not S.cg(1, 1, 1).any()   # not in a x b


def test_f_against_cg():
    assert f_against_cg_error(S, LABELS) < TOL
    assert f_against_cg_error(S, LABELS[:5], all_m=True) < TOL     # independent of m


def test_r_against_cg_and_bosonic():
    assert r_against_cg_error(S, LABELS) < TOL
    for a, b in itertools.product(LABELS, repeat=2):
        for c in _outs(S, a, b):
            assert S.R(a, b, c) * S.R(b, a, c) == 1.0
    assert S.R(1, 1, 0) == -1.0 and S.R(1, 1, 2) == 1.0 and S.R(1, 1, 4) == 0.0


def test_f_unitary():
    assert f_unitarity_error(S, LABELS) < TOL


def test_pentagon():
    # all four leaves 2j <= 6, intermediates and totals unrestricted (2j <= 24)
    assert pentagon_error(S, LABELS[:7]) < TOL


@pytest.mark.slow
def test_pentagon_to_spin_4():
    assert pentagon_error(S, LABELS) < TOL               # about a minute


def test_hexagons():
    assert hexagon_error(S, LABELS) < TOL


def test_frobenius_schur_and_trivial_f():
    for a in LABELS:
        assert S.frobenius_schur(a) == pytest.approx(S.dim(a) * S.F(a, a, a, a, 0, 0), abs=TOL)
        assert S.frobenius_schur(a) == (-1) ** a
        for b, c in itertools.product(LABELS, repeat=2):           # a unit leaf recouples freely
            for d in _outs(S, b, c):
                assert S.F(0, b, c, d, b, d) == pytest.approx(1.0, abs=TOL)
                assert S.F(b, c, 0, d, d, c) == pytest.approx(1.0, abs=TOL)


def test_known_values():
    """Textbook values: two spin-1/2 into a singlet, four spin-1/2 recoupled."""
    s = 1 / np.sqrt(2)
    assert np.allclose(S.cg(1, 1, 0)[:, :, 0], [[0, s], [-s, 0]], atol=1e-15)
    assert np.allclose(S.cg(1, 1, 2)[:, :, 1], [[0, s], [s, 0]], atol=1e-15)
    assert su2.clebsch_gordan(2, 2, 1, -1, 1, 1) == pytest.approx(np.sqrt(2 / 3), abs=1e-15)
    assert su2.clebsch_gordan(2, 0, 1, 1, 1, 1) == pytest.approx(-np.sqrt(1 / 3), abs=1e-15)
    M = [[S.F(1, 1, 1, 1, e, f) for f in (0, 2)] for e in (0, 2)]
    assert np.allclose(M, [[-0.5, np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]], atol=1e-15)
    assert su2.wigner_6j(1, 1, 2, 1, 1, 0) == pytest.approx(1 / 2, abs=1e-15)   # {½ ½ 1; ½ ½ 0}
    assert su2.wigner_6j(2, 2, 2, 2, 2, 2) == pytest.approx(1 / 6, abs=1e-15)   # {1 1 1; 1 1 1}
    assert su2.cg_exact(2, 0, 1, 1, 1, 1) == Fraction(-1, 3)                  # signed square


# --------------------------------------------------------------------------- 6j identities
def _sixjs(labels):
    return [t for t in itertools.product(labels, repeat=6) if su2.sixj_exact(*t) != 0]


def test_sixj_tetrahedral_symmetry():
    for a, b, c, d, e, f in _sixjs(LABELS[:7]):
        v = su2.sixj_exact(a, b, c, d, e, f)
        cols = [(a, d), (b, e), (c, f)]
        for p in itertools.permutations(cols):              # column permutations
            assert su2.sixj_exact(*(x[0] for x in p), *(x[1] for x in p)) == v
        assert su2.sixj_exact(d, e, c, a, b, f) == v         # swap upper/lower in two columns
        assert su2.sixj_exact(a, e, f, d, b, c) == v


def test_sixj_orthogonality():
    """sum_x (2x+1)(2f+1) {a b x; c d f}{a b x; c d g} = delta_fg; exact on the diagonal."""
    for a, b, c, d in itertools.product(LABELS, repeat=4):
        xs = [x for x in range(2 * JMAX + 1) if su2.triangle(a, b, x) and su2.triangle(c, d, x)]
        fs = [f for f in range(2 * JMAX + 1) if su2.triangle(a, d, f) and su2.triangle(c, b, f)]
        for f in fs:
            assert sum((x + 1) * (f + 1) * abs(su2.sixj_exact(a, b, x, c, d, f)) for x in xs) == 1
            for g in fs:
                got = sum((x + 1) * (f + 1) * su2.wigner_6j(a, b, x, c, d, f)
                          * su2.wigner_6j(a, b, x, c, d, g) for x in xs)
                assert got == pytest.approx(float(f == g), abs=TOL)


def test_biedenharn_elliott():
    """sum_x (-1)^(S+x) (2x+1) {a b x; c d p}{c d x; e f q}{e f x; b a r}
    = {p q r; e a d}{p q r; f b c}, S = a+b+c+d+e+f+p+q+r; all six leaves 2j <= 4."""
    w, T = su2.wigner_6j, su2.triangle
    n = 0
    for a, b, c, d, e, f in itertools.product(LABELS[:5], repeat=6):
        xs = [x for x in _outs(S, a, b) if T(c, d, x) and T(e, f, x)]
        for p in (p for p in _outs(S, a, d) if T(c, b, p)):
            for q in (q for q in _outs(S, c, f) if T(e, d, q)):
                for r in (r for r in _outs(S, e, a) if T(b, f, r)):
                    lhs = sum((-1) ** ((a + b + c + d + e + f + p + q + r + x) // 2) * (x + 1)
                              * w(a, b, x, c, d, p) * w(c, d, x, e, f, q) * w(e, f, x, b, a, r)
                              for x in xs)
                    assert lhs == pytest.approx(w(p, q, r, e, a, d) * w(p, q, r, f, b, c), abs=TOL)
                    n += 1
    assert n > 10000


# --------------------------------------------------------------------------- oracles
def test_against_sympy_exactly():
    sympy = pytest.importorskip("sympy")
    from sympy.physics.wigner import clebsch_gordan, wigner_6j

    def same(sym_value, s):
        return sympy.sign(sym_value) * sym_value ** 2 == sympy.Rational(s.numerator, s.denominator)

    h = lambda x: sympy.Rational(x, 2)                       # noqa: E731
    for j1, j2 in itertools.product(LABELS, repeat=2):
        for j in _outs(S, j1, j2):
            for m1, m2 in itertools.product(range(-j1, j1 + 1, 2), range(-j2, j2 + 1, 2)):
                if abs(m1 + m2) <= j:
                    v = clebsch_gordan(h(j1), h(j2), h(j), h(m1), h(m2), h(m1 + m2))
                    assert same(v, su2.cg_exact(j1, m1, j2, m2, j, m1 + m2))
    for t in _sixjs(LABELS[:7]):
        assert same(wigner_6j(*map(h, t)), su2.sixj_exact(*t))


def _spin(j):
    """S^z and S^+ for spin j/2 in the basis m = j/2, j/2 - 1, ... (highest weight first)."""
    m = (j - 2 * np.arange(j + 1)) / 2
    sp = np.zeros((j + 1, j + 1))
    for k in range(1, j + 1):                                # S+ |m> = sqrt(j(j+1) - m(m+1)) |m+1>
        sp[k - 1, k] = np.sqrt(j / 2 * (j / 2 + 1) - m[k] * (m[k] + 1))
    return np.diag(m), sp


def _cg_by_diagonalising(j1, j2):
    """Every coupled multiplet of j1 x j2 from the total spin alone: the highest weight of total
    spin J is the M = J vector killed by S^+, its phase set by <j1 j1; j2 (J - j1)|J J> > 0
    (Condon-Shortley), the rest by S^-. Returns {J: array (j1+1, j2+1, J+1)}."""
    z1, p1 = _spin(j1)
    z2, p2 = _spin(j2)
    Z = np.kron(z1, np.eye(j2 + 1)) + np.kron(np.eye(j1 + 1), z2)
    P = np.kron(p1, np.eye(j2 + 1)) + np.kron(np.eye(j1 + 1), p2)
    out = {}
    for J in range(abs(j1 - j2), j1 + j2 + 1, 2):
        sub = np.flatnonzero(np.isclose(np.diag(Z), J / 2))
        _, s, vh = np.linalg.svd(P[:, sub])                  # null space of S^+ within M = J
        assert s[-1] < 1e-12 and (len(s) == 1 or s[-2] > 1e-6)
        v = np.zeros(Z.shape[0])
        v[sub] = vh[-1]
        v *= np.sign(v[(j2 - J + j1) // 2])                  # m1 = j1 (row 0), m2 = J - j1
        vecs = [v]
        for M in range(J, -J, -2):                           # S^- |J M> = sqrt(...) |J M-1>
            vecs.append(P.T @ vecs[-1] / np.sqrt(J / 2 * (J / 2 + 1) - M / 2 * (M / 2 - 1)))
        out[J] = np.stack(vecs, axis=1).reshape(j1 + 1, j2 + 1, J + 1)
    return out


def test_against_spin_matrices():
    for j1, j2 in itertools.product(LABELS, repeat=2):
        for J, C in _cg_by_diagonalising(j1, j2).items():
            assert abs(C - S.cg(j1, j2, J)).max() < TOL


# --------------------------------------------------------------------------- planted errors
class _FlippedSixj(SU2):
    """One 6j sign flipped: {1/2 1/2 1; 1/2 1/2 1} and everything that uses it."""

    def F(self, a, b, c, d, e, f):
        v = super().F(a, b, c, d, e, f)
        return -v if (a, b, c, d, e, f) == (1, 1, 1, 1, 2, 2) else v


class _NoExchangePhase(SU2):
    def R(self, a, b, c):
        return abs(super().R(a, b, c))


class _SwappedCGOrder(SU2):
    """CG with the factors' roles exchanged, <j2 m2; j1 m1 | j m> = (-1)^(j1+j2-j) <j1 m1; ...>.
    This rephasing of the coupled states leaves every F and R unchanged (the phases cancel
    around each tree), so only the spin-matrix oracle can tell it from Condon-Shortley."""

    def cg(self, a, b, c):
        return super().cg(a, b, c) * super().R(a, b, c)


class _OneCGFlipped(SU2):
    """The spin-1 triplet of two spin-1/2 with the opposite sign: F no longer matches."""

    def cg(self, a, b, c):
        return -super().cg(a, b, c) if (a, b, c) == (1, 1, 2) else super().cg(a, b, c)


class _AscendingM(SU2):
    def cg(self, a, b, c):
        return super().cg(a, b, c)[::-1, ::-1, ::-1]


def test_planted_errors_fail():
    assert f_against_cg_error(_FlippedSixj(), LABELS[:3]) > 0.5
    assert pentagon_error(_FlippedSixj(), LABELS[:3]) > 0.1
    assert f_unitarity_error(_FlippedSixj(), LABELS[:3]) > 0.1
    assert r_against_cg_error(_NoExchangePhase(), LABELS[:3]) > 0.5
    assert hexagon_error(_NoExchangePhase(), LABELS[:3]) > 0.5
    assert cg_completeness_error(_OneCGFlipped(), LABELS[:3]) < TOL
    assert f_against_cg_error(_OneCGFlipped(), LABELS[:3]) > 0.5
    # another CG phase convention: consistent with F and R, caught by the oracle alone
    for wrong in (_SwappedCGOrder(), _AscendingM()):
        assert cg_completeness_error(wrong, LABELS[:3]) < TOL
        oracle = _cg_by_diagonalising(2, 1)
        assert max(abs(oracle[J] - wrong.cg(2, 1, J)).max() for J in oracle) > 0.5
    assert f_against_cg_error(_SwappedCGOrder(), LABELS[:3]) < TOL
    assert r_against_cg_error(_SwappedCGOrder(), LABELS[:3]) < TOL


# --------------------------------------------------------------------------- products
QI = ProductSymmetry((U1(), SU2()))
QI_LABELS = [(q, j) for q in (-1, 0, 1) for j in range(4)]


def test_product_with_su2():
    assert QI.fusion == "simple" and ProductSymmetry((U1(), U1())).fusion == "unique"
    assert QI.fusions((1, 1), (-1, 1)) == (((0, 0), 1), ((0, 2), 1))
    assert QI.fuse((1, 0), (2, 3)) == (3, 3)
    with pytest.raises(ValueError, match="outcomes"):
        QI.fuse((1, 1), (1, 1))
    assert (QI.dual((1, 3)), QI.dim((1, 3)), QI.unit()) == ((-1, 3), 4, (0, 0))
    assert QI.frobenius_schur((1, 1)) == -1 and QI.frobenius_schur((-2, 2)) == 1
    assert sorted([(1, 1), (0, 2), (0, 0)], key=QI.sort_key) == [(0, 0), (0, 2), (1, 1)]
    assert QI.F((1, 1), (1, 1), (-1, 1), (1, 1), (2, 2), (0, 2)) == S.F(1, 1, 1, 1, 2, 2)
    assert QI.F((1, 1), (1, 1), (-1, 1), (1, 1), (1, 2), (0, 2)) == 0.0  # charges do not add
    assert QI.R((1, 1), (0, 1), (1, 0)) == -1.0
    assert np.array_equal(QI.cg((1, 2), (-1, 1), (0, 3)), S.cg(2, 1, 3))
    assert not QI.cg((1, 2), (-1, 1), (1, 3)).any()
    Z = ProductSymmetry((Zn(2), SU2()))
    assert np.array_equal(Z.cg((1, 1), (1, 1), (0, 0)), S.cg(1, 1, 0))


def test_product_with_two_su2():
    """SU(2) x SU(2): the cg arrays are Kronecker products, factor-major."""
    P = ProductSymmetry((SU2(), SU2()))
    C = P.cg((1, 2), (1, 1), (2, 1))
    assert C.shape == (6, 4, 6)
    kron = np.einsum("abc,xyz->axbycz", S.cg(1, 1, 2), S.cg(2, 1, 1)).reshape(6, 4, 6)
    assert np.array_equal(C, kron)


@pytest.mark.parametrize("sym,labels", [
    (QI, QI_LABELS),
    (ProductSymmetry((SU2(), SU2())), [(a, b) for a in range(3) for b in range(3)])])
def test_product_coefficients(sym, labels):
    assert cg_completeness_error(sym, labels) < TOL
    assert f_against_cg_error(sym, labels) < TOL
    assert r_against_cg_error(sym, labels) < TOL
    assert f_unitarity_error(sym, labels) < TOL
    assert hexagon_error(sym, labels) < TOL
    assert pentagon_error(sym, labels[:6]) < TOL


@pytest.mark.parametrize("sym,labels", [
    (U1(), [-1, 0, 2]), (Zn(3), [0, 1, 2]),
    (ProductSymmetry((U1(), Zn(2))), [(0, 0), (1, 1), (-1, 0)])])
def test_abelian_coefficients_pass_the_same_checks(sym, labels):
    assert cg_completeness_error(sym, labels) < TOL
    assert f_against_cg_error(sym, labels) < TOL
    assert r_against_cg_error(sym, labels) < TOL
    assert pentagon_error(sym, labels) < TOL and hexagon_error(sym, labels) < TOL
    assert all(sym.frobenius_schur(a) == 1 for a in labels)


# --------------------------------------------------------------------------- two-flavour site
SITE_IRREPS = {0: [(0, 0), (1, 1), (2, 0)],      # even site: (q, 2I)
               1: [(-2, 0), (-1, 1), (0, 0)]}    # odd site


def _site_operators(n):
    """Charge q, I_z and I^+ of staggered site n (qubits u, d; basis |n_u n_d>, u most
    significant, 1 = occupied) and c^dagger_u, c^dagger_d with the Jordan-Wigner string on u."""
    qs = site_charges_2f(2)                               # two sites: n = 0 even, n = 1 odd
    q_u, _ = unpack_charge(qs[2 * n], 2)
    q_d, _ = unpack_charge(qs[2 * n + 1], 2)
    q = np.add.outer(q_u, q_d).reshape(-1)
    Iz, Ip = (x.toarray() for x in isospin_sparse(1))
    sp = np.array([[0.0, 0.0], [1.0, 0.0]])               # |1><0|
    zs = np.diag([1.0, -1.0])                             # (-1)^n
    cu, cd = np.kron(sp, np.eye(2)), np.kron(zs, sp)
    return q, Iz, Ip, cu, cd


def _multiplet_basis(q, Iz, Ip, labels):
    """The Condon-Shortley basis of each (q, 2I) multiplet: its highest weight (killed by I^+),
    then I^- applied. Returns {label: (4, 2I+1)}; checks every state of the site is used."""
    out = {}
    for lab in labels:
        qq, j = lab
        sel = np.flatnonzero((q == qq) & np.isclose(np.diag(Iz), j / 2))
        null = [v for v in np.eye(4)[sel] if np.allclose(Ip @ v, 0)]
        assert len(null) == 1
        vecs = [null[0]]
        for M in range(j, -j, -2):
            w = Ip.T @ vecs[-1]
            vecs.append(w / np.sqrt(j / 2 * (j / 2 + 1) - M / 2 * (M / 2 - 1)))
        out[lab] = np.stack(vecs, axis=1)
    B = np.concatenate(list(out.values()), axis=1)
    assert np.allclose(B.T @ B, np.eye(4))
    return out


@pytest.mark.parametrize("n", [0, 1])
def test_two_flavour_site_multiplets(n):
    q, Iz, Ip, cu, cd = _site_operators(n)
    labels = SITE_IRREPS[n]
    assert [QI.label(x) for x in labels] == sorted(labels, key=QI.sort_key)
    assert sum(QI.dim(x) for x in labels) == 4
    basis = _multiplet_basis(q, Iz, Ip, labels)
    I2 = Iz @ Iz + 0.5 * (Ip @ Ip.T + Ip.T @ Ip)
    for (qq, j), V in basis.items():
        assert np.allclose(I2 @ V, j / 2 * (j / 2 + 1) * V) and np.allclose(q[:, None] * V, qq * V)
    # (c+_u, c+_d) is a (q, 2I) = (1, 1) tensor operator and (c_d, -c_u) a (-1, 1) one: by
    # Wigner-Eckart their matrix elements between multiplets are QI.cg(source, k, target) times
    # one reduced number per pair. (c_u, c_d) is not: the sign is the Condon-Shortley one.
    for k, T, ok in [((1, 1), np.stack([cu, cd]), True),          # mu = +1/2, -1/2
                     ((-1, 1), np.stack([cd.T, -cu.T]), True),
                     ((-1, 1), np.stack([cu.T, cd.T]), False)]:
        pairs, fits = 0, []
        for src, Vs in basis.items():
            for dst, Vt in basis.items():
                M = np.einsum("tm,uts,sk->kum", Vt, T, Vs)   # (m_src, mu, m_dst)
                C = QI.cg(src, k, dst)
                if not C.any():
                    assert np.allclose(M, 0)
                    continue
                red = np.vdot(C, M) / np.vdot(C, C)
                fits.append(np.allclose(M, red * C, atol=TOL) and abs(red) > 0.5)
                pairs += 1
        assert pairs == 2 and all(fits) == ok             # two hops: 0 <-> 1/2 <-> doubly occ.


def test_caches_hold_valid_labels_only():
    for g in (su2._cg, su2._sixj):
        g.cache_clear()
    su2._F_CACHE.clear()
    n_cg = n_f = 0
    for t in itertools.product(range(5), repeat=6):
        su2.cg_exact(*t), su2.fsymbol(*t)
        j1, m1, j2, m2, j, m = t
        n_cg += (m1 + m2 == m and su2.triangle(j1, j2, j)
                 and all(su2._ok_m(x, y) for x, y in ((j1, m1), (j2, m2), (j, m))))
        n_f += all(su2.triangle(*x) for x in ((t[0], t[1], t[4]), (t[0], t[3], t[5]),
                                              (t[2], t[1], t[5]), (t[2], t[3], t[4])))
    assert su2._cg.cache_info().currsize == n_cg
    assert len(su2._F_CACHE) == su2._sixj.cache_info().currsize == n_f
