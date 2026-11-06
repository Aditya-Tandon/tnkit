"""The Symmetry interface, block tensors for Z_n and U(1) x U(1), and the bit-identity of the
packed U(1) charges with their product-symmetry form."""
import itertools

import numpy as np
import pytest

from tnkit import blocks as B
from tnkit.backend import get_backend
from tnkit.block_backend import (BlockBackend, project_to_block_mps, to_block_mpo,
                                 to_dense_mps)
from tnkit.legs import fuse, split
from tnkit.models_lgt2f import charge_base, schwinger2f_mpo, site_charges_2f, unpack_charge
from tnkit.symmetry import ProductSymmetry, Symmetry, U1, Zn
from tnkit.tdvp import TDVPOptions
from tnkit.tdvp_evolver import TDVPEvolver

NP = get_backend("numpy")
SYMS = {"U1": (U1(), [-1, 0, 2]), "Z3": (Zn(3), [0, 1, 2]),
        "U1xZ2": (ProductSymmetry((U1(), Zn(2))), [(-1, 1), (0, 0), (0, 1), (1, 0)])}


@pytest.mark.parametrize("name", sorted(SYMS))
def test_fusion_rules(name):
    sym, labels = SYMS[name]
    e = sym.unit()
    x = next(a for a in labels if a != e)
    for a, b, c in itertools.product(labels, repeat=3):
        ab, bc = sym.fuse(a, b), sym.fuse(b, c)
        d = sym.fuse(ab, c)
        assert sym.fuse(a, e) == a and sym.fuse(a, sym.dual(a)) == e and ab == sym.fuse(b, a)
        assert d == sym.fuse(a, bc)
        assert sym.F(a, b, c, d, ab, bc) == 1.0 and sym.F(a, b, c, d, sym.fuse(ab, x), bc) == 0.0
        assert sym.R(a, b, ab) == 1.0 and sym.R(a, b, sym.fuse(ab, x)) == 0.0
        assert sym.dim(a) == 1 and sym.fusions(a, b) == ((ab, 1),)
        assert sym.cg(a, b, ab).shape == (1, 1, 1) and sym.cg(a, b, ab)[0, 0, 0] == 1.0
        assert sym.cg(a, b, sym.fuse(ab, x))[0, 0, 0] == 0.0
        assert sym.total([a, b, c], [1, -1, 1]) == sym.fuse(sym.fuse(a, sym.dual(b)), c)


def test_labels_are_checked():
    with pytest.raises(ValueError):
        U1().label(0.5)
    with pytest.raises(ValueError):
        Zn(3).label(3)
    with pytest.raises(ValueError):
        ProductSymmetry((U1(), U1())).label((1,))
    with pytest.raises(ValueError):                          # sectors out of sort order
        B.Leg(((0, 1), (0, 0)), (1, 1), 1, ProductSymmetry((U1(), U1())))
    a, _ = B.Leg.from_charges([0, 1], sym=Zn(2))
    b, _ = B.Leg.from_charges([0, 1])
    with pytest.raises(ValueError, match="same symmetry"):
        B.random((a, b), 0, NP, np.random.default_rng(0))


def test_leg_labels_belong_to_its_symmetry():
    for charges, sym in (((2, 5), Zn(3)), ((0.5,), U1()),
                         (((0, 1, 2),), ProductSymmetry((U1(), U1())))):
        with pytest.raises(ValueError):
            B.Leg(charges, (1,) * len(charges), 1, sym)
    assert B.Leg((0, 2), (1, 1), 1, Zn(3)).size == 2


def test_fusion_must_be_declared():
    class Bare(Symmetry):                       # U(1) arithmetic, `fusion` left at the default
        unit, dual, fuse, label = U1.unit, U1.dual, U1.fuse, U1.label
    leg = B.Leg((0, 1), (1, 1), 1, Bare())
    with pytest.raises(NotImplementedError, match="needs unique fusion"):
        B.random((leg, leg.flip()), 0, NP, np.random.default_rng(0))


def _legs(sym, labels, rng, n):
    """`n` random legs over `labels` (dense charges), with random signs and dense orders."""
    out = []
    for _ in range(n):
        q = [labels[i] for i in rng.integers(0, len(labels), int(rng.integers(2, 6)))]
        leg, order = B.Leg.from_charges(q, int(rng.choice([-1, 1])), sym)
        out.append((leg, order))
    return out


@pytest.mark.parametrize("name", sorted(SYMS))
def test_contraction_against_dense(name):
    sym, labels = SYMS[name]
    rng = np.random.default_rng(3)
    done = 0
    for trial in range(60):
        (la, oa), (lb, ob), (lc, oc) = _legs(sym, labels, rng, 3)
        qa = labels[int(rng.integers(len(labels)))]
        a = B.random((la, lb, lc), sym.label(qa), NP, rng)
        if a.struct.size == 0:
            continue
        lx, _ = _legs(sym, labels, rng, 1)[0]
        b = B.random((lc.flip(), lx), sym.unit(), NP, rng)
        got = B.tensordot(a, b, ((2,), (0,)))
        assert np.allclose(got.to_dense(), np.tensordot(a.to_dense(), b.to_dense(), (2, 0)),
                           atol=1e-12)
        n2 = B.tensordot(a, a.conj(), ((0, 1, 2), (0, 1, 2)))
        assert n2.charge == sym.unit()
        assert np.isclose(n2.data[0], np.vdot(a.data, a.data))
        f, rec = fuse(a, [(0, 1), (2,)], BlockBackend(NP))
        assert np.array_equal(split(f, rec, BlockBackend(NP)).to_dense(), a.to_dense())
        U, S, Vh, disc = B.truncated_svd(f, 100)
        assert disc < 1e-20
        assert np.allclose((U.scale(S, 1).to_dense() @ Vh.to_dense()), f.to_dense(), atol=1e-12)
        Q, R = B.qr(f)
        assert np.allclose(Q.to_dense() @ R.to_dense(), f.to_dense(), atol=1e-12)
        done += 1
    assert done > 20


def _relabel(t, f, sym):
    """Block tensor `t` (packed U(1) charges) with every label mapped by `f` to `sym`."""
    legs, orders = [], []
    for leg in t.legs:
        dense = np.repeat(np.arange(len(leg.charges)), leg.dims)
        new, order = B.Leg.from_charges([f(leg.charges[i]) for i in dense], leg.sign, sym)
        legs.append(new)
        orders.append(order)
    return B.from_dense(t.to_dense(), legs, f(t.charge), orders)[0]


def _hash_run(M, W, steps=4):
    ev = TDVPEvolver(M, W, 0.05, 12, TDVPOptions(backend=BlockBackend(NP), svd_tol=1e-10,
                                                  graph="off", degenerate_tol=1e-6))
    ev.run(steps, "12site")
    return [np.asarray(t.canonical().data).tobytes() for t in ev.mps], ev.mps


@pytest.fixture(scope="module")
def packed_run():
    N = 3
    W = schwinger2f_mpo(N, 4.0, 0.3)
    qs = site_charges_2f(N)
    rng = np.random.default_rng(5)
    # a random state in the Q = 0, I_z = 0 sector, from a random MPS projected by blocks
    D = [1] + [min(2 ** min(j, 2 * N - j), 6) for j in range(1, 2 * N)] + [1]
    psi = [rng.normal(size=(D[j], 2, D[j + 1])) for j in range(2 * N)]
    M, _ = project_to_block_mps(psi, qs, NP, 16, charge=0, sym=U1())
    Wb = to_block_mpo(W, qs, NP, sym=U1())[0]
    return N, M, Wb, _hash_run(list(M), Wb)


def _to_pair(N, order):
    def f(c):
        q, t = (int(x) for x in unpack_charge(c, N))
        return (t, q) if order == "tq" else (q, t)
    return f


def test_product_symmetry_reproduces_packed_charges_bitwise(packed_run):
    N, M, Wb, (ref, ref_mps) = packed_run
    P = ProductSymmetry((U1(), U1()))
    f = _to_pair(N, "tq")                    # c = q + K t sorts by (t, q)
    got, mps = _hash_run([_relabel(t, f, P) for t in M], [_relabel(w, f, P) for w in Wb])
    assert got == ref
    assert [leg.charges for leg in mps[2].legs] == [tuple(map(f, leg.charges))
                                                    for leg in ref_mps[2].legs]


def _vector(mps, qs):
    v = np.ones((1, 1))
    for t in to_dense_mps(mps, qs):
        v = np.tensordot(v, t, (v.ndim - 1, 0))
    return v.reshape(-1)


def test_swapped_sector_order_breaks_bit_identity(packed_run):
    """Sensitivity: the (q, t) order is a valid layout with the same physics, but other bits."""
    N, M, Wb, (ref, ref_mps) = packed_run
    P = ProductSymmetry((U1(), U1()))
    got, mps = _hash_run([_relabel(t, _to_pair(N, "qt"), P) for t in M],
                         [_relabel(w, _to_pair(N, "qt"), P) for w in Wb])
    assert got != ref
    K = charge_base(N)
    packed = [_relabel(t, lambda c: c[0] + K * c[1], U1()) for t in mps]
    qs = site_charges_2f(N)
    u, w = _vector(ref_mps, qs), _vector(packed, qs)
    assert abs(abs(np.vdot(u, w)) / (np.linalg.norm(u) * np.linalg.norm(w)) - 1) < 1e-12
