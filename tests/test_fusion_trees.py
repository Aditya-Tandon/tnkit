"""Fusion trees and their dense tensors (`tnkit.fusion_trees`).

For every leaf tuple and duality pattern below, the tree tensors over all coupled sectors form a
unitary matrix (orthonormal and complete) and each is invariant under explicit group matrices:
D on a plain leg, conj(D) on a dual leg. Symmetries: U(1), U(1) x U(1), Z_3, SU(2), U(1) x SU(2),
A4 (with multiplicity), A4 in a complex basis of its 3, whose cup is complex, and A4 with
cg(1', 1'', 1) negated, whose cups of 1' and 1'' differ. Tree tensors recouple with `f_tensor`
and `r_matrix` on every duality pattern, products with multiplicity included."""
import itertools

import numpy as np
import pytest

from oracles import groups
from oracles.a4 import A4, A4ComplexBasis, A4RephasedSinglet
from tnkit.fusion_trees import FusionTree, check, fusion_trees, tree_tensor
from tnkit.symmetry import SU2, U1, ProductSymmetry, Zn, f_tensor, r_matrix

TOL = 1e-12
RNG = np.random.default_rng(11)


QI = ProductSymmetry((U1(), SU2()))
CASES = {                               # symmetry, leaf tuples (ranks 0 to 4)
    "U1": (U1(), [(), (2,), (1, -1), (1, 0, -2), (1, 1, -1, 2)]),
    "U1xU1": (ProductSymmetry((U1(), U1())), [((1, 0), (0, -1)), ((1, 1), (-1, 0), (0, 2))]),
    "Z3": (Zn(3), [(1,), (1, 2), (1, 1, 1), (2, 1, 0, 2)]),
    "SU2": (SU2(), [(1,), (1, 1), (1, 2, 1), (1, 1, 1, 1), (2, 1, 3), (2, 2, 2, 1)]),
    "U1xSU2": (QI, [((1, 1), (-1, 1)), ((1, 1), (0, 2), (-1, 1)), ((1, 1),) * 4]),
    "A4": (A4(), [(3,), (3, 3), (1, 3), (3, 3, 3), (3, 1, 3, 3), (2, 3, 3, 3)]),
    "A4c": (A4ComplexBasis(), [(3, 3), (3, 3, 3), (1, 3, 3, 3)]),
    "A4r": (A4RephasedSinglet(), [(1, 2), (2, 1), (1, 2, 3), (3, 1, 2, 3)]),
}


def _patterns(n):
    """All duality patterns for n <= 2 leaves, else four of them (none, all, two mixed)."""
    every = list(itertools.product((False, True), repeat=n))
    return every if n <= 2 else [every[0], every[-1], every[1], every[len(every) // 2]]


def _leg_rep(sym, u, dual, g):
    """The group matrix on a leg whose leaf is u: D_u, or conj(D_(dual u)) on a dual leg."""
    return groups.rep(sym, sym.dual(u), g).conj() if dual else groups.rep(sym, u, g)


def _basis(sym, leaves, flags):
    """All tree tensors of these leaves, as columns (tree, w) of one matrix."""
    trees = fusion_trees(sym, leaves, flags)
    cols = [tree_tensor(sym, t).reshape(-1, sym.dim(t.coupled)) for t in trees]
    return trees, np.concatenate(cols, axis=1) if cols else np.zeros((1, 0))


# --------------------------------------------------------------------------- basis properties
@pytest.mark.parametrize("name", CASES)
def test_trees_are_an_orthonormal_basis(name):
    sym, leaf_sets = CASES[name]
    for leaves in leaf_sets:
        for flags in _patterns(len(leaves)):
            trees, M = _basis(sym, leaves, flags)
            assert M.shape == (int(np.prod([sym.dim(a) for a in leaves])),) * 2, (leaves, flags)
            assert abs(M.conj().T @ M - np.eye(len(M))).max() < TOL, (leaves, flags)
            assert abs(M @ M.conj().T - np.eye(len(M))).max() < TOL, (leaves, flags)
            for t in trees:
                assert tree_tensor(sym, t).shape == tuple(sym.dim(a) for a in leaves) + (
                    sym.dim(t.coupled),)


@pytest.mark.parametrize("name", CASES)
def test_trees_are_invariant(name):
    sym, leaf_sets = CASES[name]
    gs = groups.elements(sym, 6, RNG)
    for leaves in leaf_sets:
        for flags in _patterns(len(leaves)):
            for t in fusion_trees(sym, leaves, flags):
                T = tree_tensor(sym, t)
                for g in gs:
                    lhs = T
                    for i, (u, d) in enumerate(zip(leaves, flags)):
                        lhs = np.moveaxis(np.tensordot(_leg_rep(sym, u, d, g), lhs, (1, i)), 0, i)
                    rhs = np.tensordot(T, groups.rep(sym, t.coupled, g), (-1, 0))
                    assert abs(lhs - rhs).max() < TOL, (t, g)


def test_a_leg_and_its_dual_couple_to_the_identity():
    """Leaves (a, dual a), the second a dual leg: both legs live on V_a and V_a*, and the unit
    tree is delta / sqrt(dim a), the identity operator."""
    for sym, labels in [(SU2(), range(5)), (A4(), range(4)), (A4ComplexBasis(), [3]),
                        (A4RephasedSinglet(), range(4)), (QI, [(1, 1), (-2, 2)]),
                        (Zn(3), range(3))]:
        for a in labels:
            (t,) = fusion_trees(sym, (a, sym.dual(a)), (False, True), coupled=sym.unit())
            T = tree_tensor(sym, t)[..., 0]
            assert abs(T - np.eye(sym.dim(a)) / np.sqrt(sym.dim(a))).max() < TOL


def test_abelian_trees_are_trivial():
    for sym, leaves in [(U1(), (1, -2, 3)), (Zn(3), (1, 2, 2)),
                        (ProductSymmetry((U1(), U1())), ((1, 0), (0, 1), (-1, -1)))]:
        for flags in _patterns(3):
            (t,) = fusion_trees(sym, leaves, flags)
            assert tree_tensor(sym, t).tolist() == [[[[1.0]]]] and t.vertices == (0, 0)


def test_complex_basis_keeps_f_and_r():
    G, C = A4(), A4ComplexBasis()
    assert abs(G.F(3, 3, 3, 3, 3, 3) - C.F(3, 3, 3, 3, 3, 3)).max() < TOL
    assert abs(G.R(3, 3, 3) - C.R(3, 3, 3)).max() < TOL
    from tnkit.symmetry import cup
    assert abs(cup(C, 3).imag).max() > 0.1                   # the dual-leg conjugation matters


# --------------------------------------------------------------------------- recoupling
RECOUPLE = {                            # symmetry, leaf tuples (ranks 2 to 4)
    "SU2": (SU2(), [(1, 1), (1, 2, 1), (2, 1, 1, 2)]),
    "A4": (A4(), [(3, 3), (1, 3), (3, 3, 3), (2, 3, 3), (3, 3, 3, 3), (3, 1, 3, 3)]),
    "A4c": (A4ComplexBasis(), [(3, 3), (3, 3, 3), (1, 3, 3, 3)]),
    "U1xSU2": (QI, [((1, 1), (-1, 1)), ((1, 1), (0, 2), (-1, 1)), ((1, 1),) * 4]),
    "A4xSU2": (ProductSymmetry((A4(), SU2())),
               [((3, 1), (3, 1)), ((3, 1), (3, 0), (3, 1)), ((3, 1), (1, 1), (3, 0), (3, 1))]),
    "A4xA4": (ProductSymmetry((A4(), A4())),
              [((3, 3), (3, 3)), ((3, 3), (3, 3), (3, 3)), ((3, 0), (0, 3), (3, 3), (3, 0))]),
}


def _dense(sym, leaves, flags, lines, mus):
    """The tensor of the tree with these leaves, lines e_1, ..., c and vertex labels."""
    return tree_tensor(sym, FusionTree(tuple(leaves), lines[-1], tuple(flags), tuple(lines[:-1]),
                                       tuple(mus)))


def _graft(outer, i, inner):
    """Contract the coupled leg of `inner` with leg i of `outer`; inner's leaves take its place."""
    k = inner.ndim - 1
    t = np.tensordot(outer, inner, axes=(i, k))
    return np.moveaxis(t, range(outer.ndim - 1, t.ndim), range(i, i + k))


def _recoupling_error(sym, leaves, flags):
    """The largest mismatch, over every tree, of F at each pair of neighbouring vertices,
    ((e_(k-1) u_k)_(e_k) u_(k+1)) = sum F (e_(k-1) (u_k u_(k+1))_f), and of R at the first vertex,
    which swaps the first two legs."""
    err, u, fl = 0.0, leaves, flags
    swapped = (u[1], u[0]) + u[2:], (fl[1], fl[0]) + fl[2:]
    for t in fusion_trees(sym, leaves, flags):
        T, L, mus = tree_tensor(sym, t), (u[0],) + t.innerlines + (t.coupled,), t.vertices
        R = r_matrix(sym, u[0], u[1], L[1])
        S = sum(R[mus[0], nu] * _dense(sym, *swapped, L[1:], (nu,) + mus[1:]).swapaxes(0, 1)
                for nu in range(R.shape[1]))
        err = max(err, abs(T - S).max())
        for k in range(1, len(u) - 1):
            S = np.zeros_like(T)
            for f, _ in sym.fusions(u[k], u[k + 1]):
                F = f_tensor(sym, L[k - 1], u[k], u[k + 1], L[k + 1], L[k], f)
                for kap, lam in itertools.product(range(F.shape[2]), range(F.shape[3])):
                    inner = _dense(sym, u[k:k + 2], fl[k:k + 2], (f,), (kap,))
                    outer = _dense(sym, u[:k] + (f,) + u[k + 2:], fl[:k] + (False,) + fl[k + 2:],
                                   L[1:k] + L[k + 1:], mus[:k - 1] + (lam,) + mus[k + 1:])
                    S = S + F[mus[k - 1], mus[k], kap, lam] * _graft(outer, k, inner)
            err = max(err, abs(T - S).max())
    return err


@pytest.mark.parametrize("name", RECOUPLE)
def test_trees_recouple_with_f_and_r(name):
    sym, leaf_sets = RECOUPLE[name]
    for leaves in leaf_sets:
        leaves = tuple(sym.label(a) for a in leaves)
        for flags in itertools.product((False, True), repeat=len(leaves)):
            assert _recoupling_error(sym, leaves, flags) < TOL, (leaves, flags)


# --------------------------------------------------------------------------- enumeration
def test_counts_with_multiplicity():
    G = A4()
    trees = fusion_trees(G, (3, 3, 3), coupled=3)
    # 3 x 3 = 1 + 1' + 1'' + 3 + 3, and each of those x 3 holds 3 once (3 x 3 -> 3 twice)
    assert len(trees) == 3 + 2 * 2
    assert sorted(t.vertices for t in trees if t.innerlines == (3,)) == [(0, 0), (0, 1), (1, 0),
                                                                          (1, 1)]
    for sym, leaves in [(G, (3, 3, 3, 3)), (SU2(), (1, 2, 3, 2)), (QI, ((1, 1),) * 4)]:
        n_trees = {}
        for t in fusion_trees(sym, leaves):
            n_trees[t.coupled] = n_trees.get(t.coupled, 0) + 1
        want = {leaves[0]: 1}                                # multiplicities by fusion rules
        for a in leaves[1:]:
            nxt = {}
            for e, n in want.items():
                for c, m in sym.fusions(e, a):
                    nxt[c] = nxt.get(c, 0) + n * m
            want = nxt
        assert n_trees == want


def test_order():
    S = SU2()
    trees = fusion_trees(S, (1, 1, 1, 1))
    assert [(t.coupled, t.innerlines) for t in trees] == [
        (0, (0, 1)), (0, (2, 1)), (2, (0, 1)), (2, (2, 1)), (2, (2, 3)), (4, (2, 3))]
    assert fusion_trees(S, (1, 1, 1, 1), coupled=2) == tuple(t for t in trees if t.coupled == 2)
    G = A4()
    t3 = [t.vertices for t in fusion_trees(G, (3, 3, 3), coupled=3) if t.innerlines == (3,)]
    assert t3 == sorted(t3)
    (t0,) = fusion_trees(S, ())
    assert t0 == FusionTree((), 0, ()) and tree_tensor(S, t0).tolist() == [1.0]
    (t1,) = fusion_trees(S, (3,), (True,))
    assert (t1.coupled, t1.steps()) == (3, ())


def test_validation_and_caching():
    S = SU2()
    t = fusion_trees(S, (1, 1, 2), coupled=2)[0]
    assert tree_tensor(S, t) is tree_tensor(S, t)
    with pytest.raises(ValueError):
        tree_tensor(S, t)[(0,) * 4] = 1.0
    bad = [FusionTree((1, 1), 1, (False, False), (), (0,)),           # 1/2 x 1/2 has no 1/2
           FusionTree((1, 1), 0, (False, False), (), (1,)),           # vertex label too large
           FusionTree((1, 1, 1), 1, (False,) * 3, (), (0, 0)),        # inner line missing
           FusionTree((1,), 3, (False,)),                             # rank 1: c = leaf
           FusionTree((3, 3), 3, (False, False), (), (2,))]
    for tree in bad:
        sym = A4() if tree.uncoupled == (3, 3) else S
        with pytest.raises(ValueError):
            check(sym, tree)
    with pytest.raises(ValueError, match="duality flags"):
        fusion_trees(S, (1, 1), (True,))
    with pytest.raises(ValueError):
        fusion_trees(S, (1, -1))


def test_wrong_dual_legs_fail():
    """Without the cup, or with an unconjugated cup, a dual leg is not invariant."""
    C, g = A4ComplexBasis(), groups.elements(A4(), 12, RNG)
    (t,) = fusion_trees(C, (3, 3), (False, True), coupled=0)
    plain = tree_tensor(C, FusionTree((3, 3), 0, (False, False), (), (0,)))
    from tnkit.symmetry import cup
    wrong = {"no cup": plain,
             "cup not conjugated": np.einsum("mn,xnw->xmw", cup(C, 3), plain)}
    right = tree_tensor(C, t)
    for name, T in wrong.items():
        errs = [abs(np.einsum("xX,yY,XYw->xyw", C.rep(3, h), C.rep(3, h).conj(), T) - T).max()
                for h in g]
        assert max(errs) > 0.1, name
    assert max(abs(np.einsum("xX,yY,XYw->xyw", C.rep(3, h), C.rep(3, h).conj(), right)
                   - right).max() for h in g) < TOL
    # A transposed cup is invariant too on self-dual sectors (Z^T = +-Z), but for spin 1/2 it
    # gives minus the identity.
    S = SU2()
    plain = tree_tensor(S, FusionTree((1, 1), 0, (False, False), (), (0,)))
    T = np.einsum("mn,xnw->xmw", cup(S, 1).conj().T, plain)[..., 0]
    assert abs(T + np.eye(2) / np.sqrt(2)).max() < TOL
