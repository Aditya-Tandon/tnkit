"""Generic fusion: coefficients with multiplicity labels, read as arrays for every fusion style.

The A4 test sector (3 x 3 holds 3 twice) and every existing symmetry pass the same identities,
written once for multiplicity: CG unitarity and equivariance under explicit group matrices, F
and R against CG overlaps, F unitary, pentagon, hexagons, the unit, cups and Frobenius-Schur
signs, and the integer fusion rules; so do two other valid A4 gauges, one with R(3, 3, 3) not
diagonal and one with rephased singlet cups. Deliberately wrong coefficients must fail; one of them (a
multiplicity rotation applied to a CG array but not to F) passes everything except F against CG."""
import itertools

import numpy as np
import pytest

from oracles import fusion_checks as fc
from oracles import groups
from oracles.a4 import A4, A4RotatedCopies, A4RephasedSinglet
from tnkit import su2
from tnkit.symmetry import (SU2, U1, ProductSymmetry, Zn, cg_tensor, cup, f_from_cg, f_tensor,
                            r_from_cg, r_matrix)

TOL = 1e-12
RNG = np.random.default_rng(7)
G = A4()
QI = ProductSymmetry((U1(), SU2()))

CASES = {                          # symmetry, labels, labels for the pentagon
    "A4": (G, [0, 1, 2, 3], None),
    "A4 rotated copies": (A4RotatedCopies(), [0, 1, 2, 3], None),
    "A4 rephased singlet": (A4RephasedSinglet(), [0, 1, 2, 3], None),
    "U1": (U1(), [-1, 0, 2], None),
    "Z3": (Zn(3), [0, 1, 2], None),
    "U1xU1": (ProductSymmetry((U1(), U1())), [(0, 0), (1, -1), (-1, 2)], None),
    "SU2": (SU2(), list(range(5)), list(range(4))),
    "U1xSU2": (QI, [(q, j) for q in (-1, 0, 1) for j in range(3)], [(0, 0), (1, 1), (-1, 2)]),
    "U1xA4": (ProductSymmetry((U1(), G)), [(q, a) for q in (0, 1) for a in (1, 3)], None),
    "A4xSU2": (ProductSymmetry((G, SU2())), [(a, j) for a in (0, 2, 3) for j in (0, 1)],
               [(0, 1), (3, 0), (3, 1)]),
    "A4xA4": (ProductSymmetry((G, G)), [(0, 0), (1, 3), (3, 2), (3, 3)], [(1, 3), (3, 2)]),
}


# --------------------------------------------------------------------------- the A4 sector
def test_a4_sector():
    assert G.fusion == "generic" and G.unit() == 0
    assert [G.dual(a) for a in range(4)] == [0, 2, 1, 3]
    assert G.fusions(3, 3) == ((0, 1), (1, 1), (2, 1), (3, 2))
    assert G.multiplicity(3, 3, 3) == 2 and G.multiplicity(1, 1, 0) == 0
    assert sum(G.dim(a) ** 2 for a in range(4)) == len(groups.a4.elements()) == 12
    assert G.cg(3, 3, 3).shape == (3, 3, 3, 2) and G.F(3, 3, 3, 3, 3, 3).shape == (2, 2, 2, 2)
    assert G.fuse(1, 1) == 2 and G.fuse(1, 3) == 3
    with pytest.raises(ValueError, match="several outcomes"):
        G.fuse(3, 3)
    for bad in (-1, 4, 0.5, True):
        with pytest.raises(ValueError):
            G.label(bad)
    # 3 x 3 -> 3: the symmetric copy first, then the antisymmetric one
    assert np.allclose(r_matrix(G, 3, 3, 3), np.diag([1, -1]), atol=TOL)


def test_a4_cg_333_golden():
    """cg(3, 3, 3)[x, y, z] = (|eps_xyz|, eps_xyz) / sqrt 2, written out."""
    want = np.zeros((3, 3, 3, 2))
    for xyz, v in {(0, 1, 2): (1, 1), (1, 2, 0): (1, 1), (2, 0, 1): (1, 1),
                   (0, 2, 1): (1, -1), (2, 1, 0): (1, -1), (1, 0, 2): (1, -1)}.items():
        want[xyz] = np.array(v) / np.sqrt(2)
    assert np.abs(G.cg(3, 3, 3) - want).max() < 1e-15


def test_a4_other_gauges():
    R = r_matrix(A4RotatedCopies(), 3, 3, 3)
    assert abs(R[0, 1]) > 0.5 and abs(R - R.T).max() > 0.5          # not diagonal, not symmetric
    S = A4RephasedSinglet()
    assert cup(S, 1).tolist() == [[-1]] and cup(S, 2).tolist() == [[1]]
    assert abs(S.F(1, 2, 1, 1, 0, 0)[0, 0, 0, 0] + 1) < TOL       # kappa of 1' is -1 here


def test_a4_f_symbols_are_not_trivial():
    """Some F of A4 mixes the two copies of 3, so multiplicity labels are exercised."""
    M = G.F(3, 3, 3, 3, 3, 3).reshape(4, 4)
    assert abs(M - np.diag(np.diag(M))).max() > 0.4
    assert np.iscomplexobj(G.F(3, 3, 3, 3, 1, 3))


# --------------------------------------------------------------------------- every symmetry
@pytest.mark.parametrize("name", CASES)
def test_identities(name):
    sym, labels, pl = CASES[name]
    assert fc.fusion_rule_errors(sym, labels) == []
    errs = fc.all_errors(sym, labels, groups.elements(sym, 12, RNG), pl)
    assert max(errs.values()) < TOL, {k: v for k, v in errs.items() if v >= TOL}


def test_a4xa4_intertwines_each_factor():
    P, labels = ProductSymmetry((G, G)), [(1, 3), (3, 2), (3, 3)]
    gens = [(g, ()) for g in ((0,), (1,))] + [((), g) for g in ((0,), (1,))]
    assert fc.intertwiner_error(P, labels, gens) < TOL
    assert P.multiplicity((3, 3), (3, 3), (3, 3)) == 4


def test_f_against_cg_at_every_state():
    for sym, labels in ((G, [0, 1, 3]), (SU2(), [0, 1, 2]), (ProductSymmetry((G, SU2())),
                                                            [(3, 0), (3, 1)])):
        assert fc.f_against_cg_error(sym, labels, all_m=True) < TOL


def test_fusion_styles():
    assert ProductSymmetry((U1(), U1())).fusion == "unique" and QI.fusion == "simple"
    assert ProductSymmetry((U1(), G)).fusion == ProductSymmetry((SU2(), G)).fusion == "generic"
    P = ProductSymmetry((G, SU2()))
    assert P.fusions((3, 1), (3, 1)) == tuple(((a, j), n) for a, n in G.fusions(3, 3)
                                              for j in (0, 2))
    assert P.multiplicity((3, 1), (3, 1), (3, 2)) == 2


def test_product_multiplicity_is_factor_major():
    P = ProductSymmetry((SU2(), G))
    x = P.cg((1, 3), (1, 3), (2, 3))
    want = np.einsum("abcm,xyzn->axbyczmn", SU2().cg(1, 1, 2)[..., None], G.cg(3, 3, 3))
    assert x.shape == (6, 6, 9, 2) and np.array_equal(x, want.reshape(6, 6, 9, 2))
    F = P.F((1, 3), (1, 3), (1, 3), (1, 3), (2, 3), (2, 3))
    assert np.allclose(F, SU2().F(1, 1, 1, 1, 2, 2) * G.F(3, 3, 3, 3, 3, 3), atol=1e-15)


# --------------------------------------------------------------------------- unchanged numbers
def test_scalar_styles_unchanged():
    S = SU2()
    for t in itertools.product(range(5), repeat=6):
        F = S.F(*t)
        assert type(F) is float and F == su2.fsymbol(*t)
        x = f_tensor(S, *t)
        assert x.size == 0 or (x.shape == (1, 1, 1, 1) and x[0, 0, 0, 0] == F)
    for a, b, c in itertools.product(range(5), repeat=3):
        assert S.R(a, b, c) == su2.rsymbol(a, b, c)
        assert np.array_equal(S.cg(a, b, c), su2.cg_array(a, b, c))
        x = cg_tensor(S, a, b, c)
        assert x.shape == su2.cg_array(a, b, c).shape + (int(su2.triangle(a, b, c)),)
    assert type(QI.F((1, 1), (1, 1), (-1, 1), (1, 1), (2, 2), (0, 2))) is float
    assert type(QI.R((1, 1), (0, 1), (1, 0))) is float
    assert QI.cg((1, 2), (-1, 1), (0, 3)).shape == (3, 2, 4)


def test_su2_closed_forms_equal_cg_overlaps():
    S = SU2()
    for a, b, c in itertools.product(range(7), repeat=3):
        for e in fc.outs(S, a, b):
            for d in fc.outs(S, e, c):
                for f in fc.outs(S, b, c):
                    assert abs(f_from_cg(S, a, b, c, d, e, f) - f_tensor(S, a, b, c, d, e, f)
                               ).max(initial=0) < 1e-14
        for d in fc.outs(S, a, b):
            assert abs(r_from_cg(S, a, b, d) - r_matrix(S, a, b, d)).max() < 1e-14


def test_cups():
    assert np.allclose(cup(SU2(), 1), [[0, 1], [-1, 0]], atol=1e-15)
    assert np.allclose(cup(SU2(), 2), np.rot90(np.diag([1, -1, 1])), atol=1e-15)
    assert cup(U1(), 3).tolist() == [[1.0]] and cup(Zn(3), 1).tolist() == [[1.0]]
    assert np.allclose(cup(G, 3), np.eye(3), atol=1e-15)       # the 3 of A4 is real


def test_block_tensors_refuse_generic_fusion():
    from tnkit import blocks as B
    leg = B.Leg((0, 3), (1, 1), 1, G)
    with pytest.raises(NotImplementedError, match="needs unique fusion"):
        B.zeros((leg, leg.flip()), 0, np.zeros(1))


# --------------------------------------------------------------------------- wrong coefficients
class _RotatedCG(A4):
    """The two copies of 3 in 3 x 3 rotated by a unitary, F and R kept: a different but equally
    valid gauge for the CG arrays, which F no longer matches."""
    U = np.array([[np.cos(0.3), 1j * np.sin(0.3)], [1j * np.sin(0.3), np.cos(0.3)]])

    def cg(self, a, b, c):
        x = super().cg(a, b, c)
        return x @ self.U if (a, b, c) == (3, 3, 3) else x

    def F(self, *x):
        return A4().F(*x)

    def R(self, *x):
        return A4().R(*x)


class _SwappedVertexAxes(A4):
    """F with its first and third axes exchanged: a wrong multiplicity-index order."""

    def F(self, *x):
        F = A4().F(*x)
        return F.transpose(2, 1, 0, 3) if F.shape[0] == F.shape[2] else F


class _SwappedSinglets(A4):
    """1' and 1'' exchanged in 3 x 3: unitary, but not equivariant."""

    def cg(self, a, b, c):
        if (a, b) == (3, 3) and c in (1, 2):
            return super().cg(a, b, 3 - c)
        return super().cg(a, b, c)


class _ConjugateF(A4):
    """The complex-conjugate F: it solves the pentagon and hexagons too."""

    def F(self, *x):
        return A4().F(*x).conj()


class _NoExchangeSigns(A4):
    def R(self, *x):
        return abs(A4().R(*x))


class _SelfDualOnePrime(A4):
    def dual(self, a):
        return a if a < 3 else 3


def test_rotated_copies_fail_only_against_cg():
    bad, labels = _RotatedCG(), [0, 1, 3]
    assert fc.f_against_cg_error(bad, labels) > 0.1
    assert fc.r_against_cg_error(bad, labels) > 0.1
    gs = groups.elements(G, 12, RNG)
    for check in (fc.cg_unitarity_error, fc.f_unitarity_error, fc.pentagon_error,
                  fc.hexagon_error, fc.r_symmetric_error, fc.unit_error, fc.dual_error):
        assert check(bad, labels) < TOL, check.__name__
    assert fc.intertwiner_error(bad, labels, gs) < TOL


def test_wrong_coefficients_fail():
    labels = [0, 1, 3]
    assert fc.f_against_cg_error(_SwappedVertexAxes(), labels) > 0.1
    assert fc.pentagon_error(_SwappedVertexAxes(), labels) > 0.1
    bad = _SwappedSinglets()
    assert fc.cg_unitarity_error(bad, labels) < TOL
    assert fc.intertwiner_error(bad, [1, 2, 3], groups.elements(G, 12, RNG)) > 0.5
    assert fc.fusion_rule_errors(_SelfDualOnePrime(), [0, 1, 2, 3])
    assert fc.hexagon_error(_NoExchangeSigns(), labels) > 0.5
    assert fc.r_against_cg_error(_NoExchangeSigns(), labels) > 0.5
    conj, every = _ConjugateF(), [0, 1, 2, 3]
    assert fc.pentagon_error(conj, every) < TOL and fc.hexagon_error(conj, every) < TOL
    assert fc.f_against_cg_error(conj, every) > 0.5
