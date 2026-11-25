"""Symmetric tensors (`tnkit.symmetric_tensor`) against dense arrays and explicit group matrices.

On U(1), Z_3, U(1) x U(1), SU(2), U(1) x SU(2), A4, A4 x SU(2), A4 x A4 and two other A4 gauges
(a complex basis of 3, whose cup is complex, and cg(1', 1'', 1) negated, whose cups of 1' and
1'' differ), at ranks 0 to 4 with mixed codomain-domain splits and dual flags:
  * dense arrays of random tensors are invariant under group matrices that share no code with
    the trees;
  * `from_dense` is the orthogonal projector onto the fixed space of those matrices, and
    reports what it drops;
  * round trips are exact, `vdot` and `norm` equal the dense ones, the reduced blocks sit where
    the documented row order puts them, and add, scale, adjoint and conjugation match dense;
  * torch matches numpy, and on U(1) x U(1) the result matches `BlockTensor`, total charge
    included.
"""
import itertools

import numpy as np
import pytest
from scipy.linalg import block_diag

from oracles import groups
from oracles.a4 import A4, A4ComplexBasis, A4RephasedSinglet
from tnkit import blocks
from tnkit.backend import NumpyBackend
from tnkit.fusion_trees import fusion_trees, tree_tensor
from tnkit.symmetric_tensor import (Space, SymmetricTensor, from_blocks, from_dense, norm,
                                    random, vdot, zeros)
from tnkit.symmetry import SU2, U1, ProductSymmetry, Zn

TOL = 1e-12
NP = NumpyBackend()


def _spaces(sym, *legs):
    return [Space(sym, *zip(*sorted(leg.items(), key=lambda kv: sym.sort_key(kv[0]))))
            for leg in legs]


def _sym(name):
    return {"U1": U1(), "Z3": Zn(3), "U1xU1": ProductSymmetry((U1(), U1())), "SU2": SU2(),
            "U1xSU2": ProductSymmetry((U1(), SU2())), "A4": A4(), "A4c": A4ComplexBasis(),
            "A4r": A4RephasedSinglet(), "A4xSU2": ProductSymmetry((A4(), SU2())),
            "A4xA4": ProductSymmetry((A4(), A4()))}[name]


LEGS = {                     # three leg spaces per symmetry: {sector: degeneracy}
    "U1": [{-1: 1, 0: 2, 1: 1}, {-2: 1, 1: 2}, {-1: 2, 1: 1}],
    "Z3": [{0: 1, 1: 2, 2: 1}, {1: 2, 2: 1}, {0: 1, 2: 1}],
    "U1xU1": [{(0, 0): 1, (1, 0): 1, (0, 1): 2}, {(-1, 0): 1, (0, -1): 1, (1, 1): 1},
              {(0, 0): 2, (1, -1): 1}],
    "SU2": [{0: 1, 1: 2}, {1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}],
    "U1xSU2": [{(0, 0): 1, (1, 1): 1}, {(-1, 1): 1, (0, 0): 1, (0, 2): 1}, {(1, 1): 1, (2, 0): 1}],
    "A4": [{0: 1, 3: 1}, {1: 1, 2: 1, 3: 2}, {0: 1, 1: 1, 3: 1}],
    "A4c": [{0: 1, 3: 1}, {3: 2}, {1: 1, 3: 1}],
    "A4r": [{0: 1, 1: 1, 2: 1}, {1: 1, 2: 1, 3: 1}, {2: 2, 3: 1}],
    "A4xSU2": [{(0, 0): 1, (3, 1): 1}, {(1, 1): 1, (3, 0): 1}, {(0, 0): 1, (2, 1): 1}],
    "A4xA4": [{(0, 0): 1, (0, 3): 1, (3, 0): 1}, {(1, 2): 1, (3, 3): 1}, {(2, 0): 1, (3, 1): 1}],
}


def _shapes(A, B, C):
    """(codomain, domain) pairs of ranks 0 to 4, mixed splits and dual flags."""
    a, b, c = A.dual(), B.dual(), C.dual()
    return [((), ()), ((A,), ()), ((), (a,)), ((A, a), ()), ((A,), (A,)), ((b,), (B,)),
            ((), (A, C)), ((A, B), (C,)), ((a,), (B, c)), ((A, b, C), ()), ((), (a, B, c)),
            ((A, B), (c, A)), ((a, B, C), (b,)), ((A,), (B, c, a)), ((A, b, C, a), ()),
            ((), (A, B, c, a))]


CASES = [(name, i) for name in LEGS for i in range(16)]


def _case(name, i):
    sym = _sym(name)
    cod, dom = _shapes(*_spaces(sym, *LEGS[name]))[i]
    return sym, cod, dom


def _ids(case):
    return f"{case[0]}-{case[1]}"


def _rng(case):
    return np.random.default_rng([sorted(LEGS).index(case[0]), case[1]])


# --------------------------------------------------------------------------- group matrices
def _identity(sym):
    if isinstance(sym, ProductSymmetry):
        return tuple(_identity(f) for f in sym.factors)
    return {U1: 0.0, Zn: 0, SU2: np.zeros(3)}.get(type(sym), ())


def _generators(sym, rng):
    """Group elements whose common fixed space is the invariant subspace: each factor's
    generators (or random elements of a Lie group), the other factors at the identity."""
    if isinstance(sym, ProductSymmetry):
        ids = [_identity(f) for f in sym.factors]
        return [tuple(g if j == k else ids[j] for j in range(len(ids)))
                for k, f in enumerate(sym.factors) for g in _generators(f, rng)]
    if isinstance(sym, U1):
        return list(rng.uniform(0.5, 2.5, 2))
    if isinstance(sym, Zn):
        return [1]
    if isinstance(sym, SU2):
        return groups.elements(sym, 2, rng)
    return [(0,), (1,)]                                      # A4: s and t


def _leg_matrix(V, g):
    """The group matrix on one dense leg: kron(1_n, D_a) per sector, conjugated on a dual leg."""
    m = block_diag(*[np.kron(np.eye(n), groups.rep(V.sym, a, g))
                     for a, n in zip(V.sectors, V.degeneracies)])
    return m.conj() if V.isdual else m


def _act(x, cod, dom, g):
    """D(g) on every codomain leg and conj(D(g)) on every domain leg of the dense array x."""
    for i, V in enumerate(cod + dom):
        m = _leg_matrix(V, g) if i < len(cod) else _leg_matrix(V, g).conj()
        x = np.moveaxis(np.tensordot(m, x, axes=(1, i)), 0, i)
    return x


def _projector(sym, cod, dom, rng):
    """The orthogonal projector onto the common fixed space of the generators."""
    n = int(np.prod([V.dim for V in cod + dom]))
    H = np.zeros((n, n), complex)
    for g in _generators(sym, rng):
        rho = np.ones((1, 1))
        for i, V in enumerate(cod + dom):
            rho = np.kron(rho, _leg_matrix(V, g) if i < len(cod) else _leg_matrix(V, g).conj())
        H += (rho - np.eye(n)).conj().T @ (rho - np.eye(n))
    w, Q = np.linalg.eigh(H)
    assert w[w > 1e-8].min(initial=1.0) > 1e-4                  # a clear gap above the kernel
    Q = Q[:, w < 1e-8]
    return Q @ Q.conj().T


def _err(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max(initial=0.0)) / max(1.0, float(np.abs(b).max(initial=0.0)))


def _rand_dense(shape, rng):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


# --------------------------------------------------------------------------- dense tensors
def test_every_symmetry_has_nonzero_tensors_at_every_rank():
    for name in LEGS:
        ranks = set()
        for i in range(16):
            sym, cod, dom = _case(name, i)
            if zeros(cod, dom, sym=sym).layout.size:
                ranks.add(len(cod) + len(dom))
        assert ranks == {0, 1, 2, 3, 4}, name


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_dense_arrays_are_invariant(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    x = random(cod, dom, NP, rng, sym=sym).to_dense()
    assert x.shape == tuple(V.dim for V in cod + dom)
    for g in _generators(sym, rng) + groups.elements(sym, 2, rng):
        assert _err(_act(x, cod, dom, g), x) < TOL, g


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_round_trip(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    t = random(cod, dom, NP, rng, sym=sym)
    x = t.to_dense()
    t2, dropped = from_dense(x, cod, dom, NP, sym=sym)
    assert _err(t2.data, t.data) < TOL and dropped < TOL ** 2
    assert _err(t2.to_dense(), x) < TOL


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_from_dense_is_the_invariant_projector(case):
    sym, cod, dom = _case(*case)
    shape = tuple(V.dim for V in cod + dom)
    if np.prod(shape) > 2000:
        pytest.skip("dense projector too large")
    rng = _rng(case)
    y = _rand_dense(shape, rng)
    Py = (_projector(sym, cod, dom, rng) @ y.reshape(-1)).reshape(shape)
    t, dropped = from_dense(y, cod, dom, NP, sym=sym)
    assert _err(t.to_dense(), Py) < TOL
    assert abs(dropped - np.linalg.norm(y - Py) ** 2) < TOL * np.linalg.norm(y) ** 2


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_vdot_and_norm_are_the_dense_ones(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    a, b = random(cod, dom, NP, rng, sym=sym), random(cod, dom, NP, rng, sym=sym)
    A, B = a.to_dense(), b.to_dense()
    assert abs(vdot(a, b) - np.vdot(A, B)) < TOL * max(1.0, np.linalg.norm(A) * np.linalg.norm(B))
    assert abs(norm(a) - np.linalg.norm(A)) < TOL * max(1.0, np.linalg.norm(A))
    assert abs(a.norm() - norm(a)) == 0


def test_the_dimension_weight_matters():
    """Without dim(c), the reduced inner product misses the dense one."""
    sym, cod, dom = _case("SU2", 11)
    a = random(cod, dom, NP, np.random.default_rng(1), sym=sym)
    plain = sum(np.vdot(b, b) for b in a.blocks().values()).real
    assert abs(plain - norm(a) ** 2) > 0.1 * norm(a) ** 2


@pytest.mark.parametrize("name", ["SU2", "A4", "A4c", "A4r", "U1xSU2", "A4xSU2", "A4xA4"])
def test_layout_follows_the_row_order(name):
    """Every reduced entry, set alone, gives kron(e_i x e_j, sum_w X_s conj(X_f)) on the sectors
    of its leaf tuples, with rows (leaf tuple, tree, degeneracy) as the module documents."""
    sym = _sym(name)
    A, B, C = _spaces(sym, *LEGS[name])
    for cod, dom in [((A, B.dual()), (C,)), ((B,), (A.dual(), B))]:
        t = random(cod, dom, NP, np.random.default_rng(3), sym=sym)
        want = np.zeros(t.shape, complex)
        rows, cols = _order(sym, cod), _order(sym, dom)
        for c in t.sectors:
            for (r, (sl_r, s, degs_r, i)), (k, (sl_c, f, degs_c, j)) in itertools.product(
                    enumerate(rows[c]), enumerate(cols[c])):
                X, Y = tree_tensor(sym, s), tree_tensor(sym, f)
                P = np.tensordot(X, Y.conj(), axes=(-1, -1))
                e = np.zeros(degs_r + degs_c)
                e[np.unravel_index(i, degs_r) + np.unravel_index(j, degs_c)] = 1
                want[sl_r + sl_c] += t.block(c)[r, k] * np.kron(e, P)
        assert _err(t.to_dense(), want) < TOL


def _order(sym, spaces):
    """For each coupled sector, one entry per row: (dense slices, tree, degeneracies, index of
    the degeneracy state)."""
    out = {}
    pos = [np.cumsum([0] + [n * sym.dim(a) for a, n in zip(V.sectors, V.degeneracies)])
           for V in spaces]
    for idx in itertools.product(*(range(len(V.sectors)) for V in spaces)):
        secs = [V.sectors[k] for V, k in zip(spaces, idx)]
        leaves = [sym.dual(a) if V.isdual else a for V, a in zip(spaces, secs)]
        degs = tuple(V.degeneracies[k] for V, k in zip(spaces, idx))
        sl = tuple(slice(p[k], p[k + 1]) for p, k in zip(pos, idx))
        for tree in fusion_trees(sym, leaves, [V.isdual for V in spaces]):
            out.setdefault(tree.coupled, []).extend(
                (sl, tree, degs, i) for i in range(int(np.prod(degs))))
    return out


# --------------------------------------------------------------------------- operations
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_operations_match_dense(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    a, b = random(cod, dom, NP, rng, sym=sym), random(cod, dom, NP, rng, sym=sym)
    A, B = a.to_dense(), b.to_dense()
    z = 0.3 - 1.7j
    for got, want in [(a + b, A + B), (a - b, A - B), (z * a, z * A), (a * z, A * z),
                      (a / z, A / z), (-a, -A), (np.float64(2.0) * a, 2 * A)]:
        assert _err(got.to_dense(), want) < TOL
    n = len(cod)
    perm = list(range(n, A.ndim)) + list(range(n))
    adj = a.adjoint()
    assert (adj.codomain, adj.domain) == (dom, cod)
    assert _err(adj.to_dense(), A.conj().transpose(perm)) < TOL
    con = a.conj()
    assert con.codomain == tuple(V.dual() for V in cod)
    assert _err(con.to_dense(), A.conj()) < TOL
    assert _err(con.conj().data, a.data) < TOL
    assert abs(vdot(con, con) - vdot(a, a)) < TOL * max(1.0, norm(a) ** 2)


def test_construction_and_validation():
    S = SU2()
    half = Space(S, (1,), (2,))
    with pytest.raises(ValueError):
        Space(S, (2, 1), (1, 1))                                  # not in sort_key order
    with pytest.raises(ValueError):
        Space(S, (1,), (0,))
    with pytest.raises(ValueError):
        Space(S, (-1,), (1,))
    assert half.dim == 4 and half.dual().dual() == half and half.dual().leaf(1) == 1
    t = zeros([half, half], [], NP)
    assert t.sectors == (0,) and t.layout.shapes[0] == (4, 1) and not t.data.any()
    with pytest.raises(ValueError, match="no block"):
        from_blocks({2: np.ones((1, 1))}, [half, half], [], NP)
    with pytest.raises(ValueError, match="shape"):
        from_blocks({0: np.ones((3, 1))}, [half, half], [], NP)
    with pytest.raises(ValueError, match="shape"):
        from_dense(np.ones((4, 3)), [half, half], [], NP)
    with pytest.raises(ValueError, match="common symmetry"):
        zeros([], [], NP)
    with pytest.raises(ValueError, match="common symmetry"):
        zeros([half, Space(U1(), (0,), (1,))], [], NP)
    with pytest.raises(ValueError):
        t + zeros([half], [half], NP)
    with pytest.raises(ValueError, match="buffer"):
        SymmetricTensor([half], [half], np.zeros(3), NP)
    s, dropped = from_dense(np.array(2.5), [], [], NP, sym=S)        # rank 0: one 1 x 1 block
    assert s.blocks()[0].tolist() == [[2.5]] and s.to_dense().shape == () and dropped == 0


def test_real_backend():
    real = NumpyBackend(np.float64)
    sym, cod, dom = _case("SU2", 11)
    t = random(cod, dom, real, np.random.default_rng(2), sym=sym)
    assert t.data.dtype == np.float64 and t.to_dense().dtype == np.float64
    t2, dropped = from_dense(t.to_dense(), cod, dom, real, sym=sym)
    assert _err(t2.data, t.data) < TOL and dropped < TOL ** 2
    sym, cod, dom = _case("A4", 7)
    with pytest.raises(TypeError, match="imaginary"):            # A4's trees are complex
        random(cod, dom, real, np.random.default_rng(2), sym=sym).to_dense()


# --------------------------------------------------------------------------- torch
@pytest.mark.parametrize("name", list(LEGS))
def test_torch_matches_numpy(name):
    pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    tb = TorchBackend("cpu")
    for i in (0, 4, 8, 11, 13):
        sym, cod, dom = _case(name, i)
        rng = np.random.default_rng(i)
        a, b = random(cod, dom, NP, rng, sym=sym), random(cod, dom, NP, rng, sym=sym)
        ta, tb_ = (SymmetricTensor(cod, dom, tb.array(t.data), tb, sym) for t in (a, b))
        for x, y in [(ta, a), (ta.conj(), a.conj()), (ta.adjoint(), a.adjoint()),
                     (ta + tb_ * 0.5, a + b * 0.5)]:
            assert _err(tb.to_numpy(x.to_dense()), y.to_dense()) < TOL
            assert _err(tb.to_numpy(x.data), y.data) < TOL
        assert abs(complex(vdot(ta, tb_)) - vdot(a, b)) < TOL * max(1.0, norm(a) * norm(b))
        assert abs(float(norm(ta)) - norm(a)) < TOL * max(1.0, norm(a))
        y = _rand_dense(a.shape, rng)
        (t1, d1), (t2, d2) = from_dense(y, cod, dom, tb, sym=sym), from_dense(y, cod, dom, NP,
                                                                              sym=sym)
        assert _err(tb.to_numpy(t1.data), t2.data) < TOL and abs(d1 - d2) < TOL * max(1.0, d2)


# --------------------------------------------------------------------------- abelian blocks
def _block_leg(V, in_domain):
    """The block leg of a space: sign +1 on a plain codomain leg, -1 on a plain domain leg, and
    flipped by a dual flag."""
    sign = (-1 if in_domain else 1) * (-1 if V.isdual else 1)
    return blocks.Leg(V.sectors, V.degeneracies, sign, V.sym)


@pytest.mark.parametrize("i", range(16))
def test_abelian_tensors_match_block_tensors(i):
    sym, cod, dom = _case("U1xU1", i)
    rng = np.random.default_rng(i)
    legs = [_block_leg(V, False) for V in cod] + [_block_leg(V, True) for V in dom]
    for q in [sym.unit(), (1, 0), (-1, 1)]:
        if not legs:
            continue
        y = _rand_dense(tuple(V.dim for V in cod + dom), rng)
        bt, b_dropped = blocks.from_dense(y, legs, charge=q)
        charge_leg = Space(sym, (q,), (1,))                     # total charge as a domain leg
        t, dropped = from_dense(y[..., None], cod, dom + (charge_leg,), NP)
        assert _err(t.to_dense()[..., 0], bt.to_dense()) < TOL
        assert abs(dropped - b_dropped) < TOL * np.linalg.norm(y) ** 2
