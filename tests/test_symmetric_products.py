"""Symmetric tensors on products of symmetries, as U(1) x SU(2), against dense arrays and blocks.

  * Recoupling matrices are keyed on the factors with nontrivial coefficients: one matrix serves
    every charge of the same spins (cache counts), equals the matrix built on the full labels,
    and a unique-fusion factor whose coefficients are phases is kept, as is every factor of a
    product that defines its own coefficients;
  * a layout enumerates the larger side only for the coupled sectors of the smaller, and these
    are exactly the trees of the full enumeration, in the same order and with the same rows;
  * `fuse_spaces` counts rows without enumerating trees, as the trees would;
  * assignments that share a matrix share one product in the batched recoupling, on torch only
    if that takes no more batched products;
  * a two-site Hubbard term with exchange on two-colour fermion sites, in spin language, gives
    the same three-site products as U(1) block tensors and as dense arrays;
  * random tensors of ranks 3 to 6 on products of two and three symmetries, and on products in
    another gauge, permute, contract, fuse and split as dense arrays.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import pytest

from oracles.a4 import A4
from test_symmetric_recoupling import _partner, _random_split
from test_symmetric_tensor import _err
from tnkit import _recoupling as rc
from tnkit import _symmetric_layout as sl
from tnkit import blocks
from tnkit import symmetric_tensor as st
from tnkit.backend import NumpyBackend
from tnkit.symmetric_tensor import (Space, contract, from_dense, fuse, fuse_spaces, permute,
                                    random, split)
from tnkit.symmetry import SU2, U1, ProductSymmetry, Zn, f_from_cg, r_from_cg

TOL = 1e-12
NP = NumpyBackend()
U1SU2 = ProductSymmetry((U1(), SU2()))


@dataclass(frozen=True)
class PhasedU1(U1):
    """U(1) in a gauge where cg(a, b, a + b) = exp(i theta(a, b)) with theta not symmetric:
    unique fusion, but R and F are phases."""

    def cg(self, a, b, c):
        return np.full((1, 1, 1), np.exp(1j * (0.3 * a * b * b + 0.2 * a)) if c == a + b else 0)

    def F(self, a, b, c, d, e, f):
        x = f_from_cg(self, a, b, c, d, e, f)
        return complex(x.reshape(-1)[0]) if x.size else 0.0

    def R(self, a, b, c):
        x = r_from_cg(self, a, b, c)
        return complex(x.reshape(-1)[0]) if x.size else 0.0

    def frobenius_schur(self, a):
        return self.F(a, -a, a, a, 0, 0)


def _first(x):
    return complex(x.reshape(-1)[0]) if x.size else 0.0


@dataclass(frozen=True)
class GaugedProduct(ProductSymmetry):
    """A product in a gauge where cg(a, b, c) carries the phase exp(i theta(a_0, b_0)) of the first
    factor's labels, theta not symmetric and 0 on the unit: F and R are phases that no factor
    has alone, so the factors' coefficients are not the product's."""

    def cg(self, a, b, c):
        return np.exp(1j * (0.3 * a[0] * b[0] ** 2 + 0.2 * a[0] * b[0])) * super().cg(a, b, c)

    def F(self, a, b, c, d, e, f):
        return _first(f_from_cg(self, a, b, c, d, e, f))

    def R(self, a, b, c):
        return _first(r_from_cg(self, a, b, c))


GAUGED_U1SU2, GAUGED_Z3Z2 = GaugedProduct((U1(), SU2())), GaugedProduct((Zn(3), Zn(2)))


def _space(sym, degs):
    items = sorted(degs.items(), key=lambda kv: sym.sort_key(kv[0]))
    return Space(sym, [a for a, _ in items], [n for _, n in items])


def _site(sym=U1SU2):
    """A two-colour fermion site: empty, one particle (spin 1/2), two particles (singlet)."""
    return _space(sym, {(0, 0): 1, (1, 1): 1, (2, 0): 1})


def _bond(sym, charges, spins, rng):
    return _space(sym, {(q, j): int(rng.integers(1, 3)) for q in charges for j in spins
                        if rng.random() < 0.8} or {(charges[0], spins[0]): 1})


# --------------------------------------------------------------------------- one matrix per spin
def test_trivial_coefficients():
    assert U1().trivial_coefficients and Zn(3).trivial_coefficients
    assert ProductSymmetry((U1(), Zn(2))).trivial_coefficients
    assert not SU2().trivial_coefficients and not A4().trivial_coefficients
    assert not PhasedU1().trivial_coefficients and not U1SU2.trivial_coefficients
    assert not GAUGED_U1SU2.trivial_coefficients and not GAUGED_Z3Z2.trivial_coefficients
    a, phased = (1, 2, 3), ProductSymmetry((PhasedU1(), SU2()))
    for sym, part, label in [(U1SU2, SU2(), 2), (ProductSymmetry((SU2(), Zn(2), U1())), SU2(), 1),
                             (ProductSymmetry((U1(), A4(), SU2())),
                              ProductSymmetry((A4(), SU2())), (2, 3)),
                             (SU2(), SU2(), a), (phased, phased, a),
                             (GAUGED_U1SU2, GAUGED_U1SU2, a), (GAUGED_Z3Z2, GAUGED_Z3Z2, a)]:
        got, project = rc.recoupling_part(sym)
        assert got == part and project(a) == label
    assert rc.recoupling_part(U1())[0] is None


def test_product_frobenius_schur_keeps_phases():
    p = ProductSymmetry((PhasedU1(), SU2()))
    for a in [(1, 1), (-2, 2), (3, 0)]:
        want = PhasedU1().frobenius_schur(a[0]) * SU2().frobenius_schur(a[1])
        assert abs(abs(want) - 1) < TOL and abs(p.frobenius_schur(a) - want) < TOL


SYMS = {"U1xSU2": U1SU2, "SU2xZ2": ProductSymmetry((SU2(), Zn(2))),
        "U1xU1xSU2": ProductSymmetry((U1(), U1(), SU2())),
        "U1xA4xSU2": ProductSymmetry((U1(), A4(), SU2()))}


def _legs(name):
    """Three legs with several charges on each spin."""
    if name == "U1xSU2":
        return [{(q, j): 1 + (q + j) % 2 for q in (-1, 0, 1) for j in (0, 1, 2)},
                {(q, 1): 1 for q in (-1, 0, 1, 2)} | {(0, 0): 1}, {(q, 0): 2 for q in (0, 1, 2)}]
    if name == "SU2xZ2":
        return [{(j, z): 1 for j in (0, 1, 2) for z in (0, 1)}, {(1, 0): 1, (1, 1): 2},
                {(0, 1): 1, (2, 0): 1}]
    if name == "U1xU1xSU2":
        return [{(q, r, j): 1 for q in (0, 1) for r in (-1, 0) for j in (0, 1)},
                {(q, 0, 1): 1 for q in (-1, 0, 1)}, {(0, r, 0): 1 for r in (0, 1)}]
    return [{(q, a, j): 1 for q in (0, 1) for a in (0, 3) for j in (0, 1)},
            {(q, 3, 1): 1 for q in (-1, 0)}, {(0, a, 0): 1 for a in (0, 1, 3)}]


@pytest.mark.parametrize("name", list(SYMS))
def test_one_matrix_serves_every_charge(name):
    sym = SYMS[name]
    A, B, C = (_space(sym, d) for d in _legs(name))
    rng = np.random.default_rng(len(name))
    t = random((A, B.dual()), (C, A), NP, rng, sym=sym)
    spaces, n1 = t.codomain + t.domain, 2
    flags = tuple(V.isdual for V in spaces)
    project = rc.recoupling_part(sym)[1]
    for p1, p2 in [((3, 1), (2, 0)), ((0, 2, 1), (3,)), ((), (1, 0, 3, 2))]:
        st.RECOUPLING_CACHE.clear()
        u = permute(t, p1, p2)
        assert _err(u.to_dense(), t.to_dense().transpose(p1 + p2)) < TOL
        spins, n = set(), 0
        for secs in itertools.product(*(V.sectors for V in spaces)):
            leaves = tuple(V.leaf(a) for V, a in zip(spaces, secs))
            if not rc.tree_pairs(sym, leaves, flags, n1):
                continue
            n += 1
            spins.add(tuple(map(project, leaves)))
            M = rc.recoupling_matrix(sym, n1, p1, p2, leaves, flags)
            assert _err(M, rc._recoupling_matrix(sym, n1, p1, p2, leaves, flags)) < 1e-14
        stats = st.RECOUPLING_CACHE.stats
        assert len(spins) < n
        assert (len(st.RECOUPLING_CACHE), stats["misses"]) == (len(spins), len(spins))
        assert stats["hits"] == n         # the permute looks each matrix up once, the loop above


def test_recoupling_matrix_of_trivial_coefficients():
    """1 x 1 where the tree pair exists, 0 x 0 where it does not, as `_recoupling_matrix`."""
    u1z2 = ProductSymmetry((U1(), Zn(2)))
    for sym, leaves in [(U1(), (1, 1)), (U1(), (1, 2)), (u1z2, ((2, 1), (2, 1))),
                        (u1z2, ((2, 1), (2, 0)))]:
        for p1, p2 in [((0,), (1,)), ((1,), (0,)), ((), (1, 0))]:
            M = rc.recoupling_matrix(sym, 1, p1, p2, leaves, (False, False))
            assert np.array_equal(M, rc._recoupling_matrix(sym, 1, p1, p2, leaves, (False, False)))


@pytest.mark.parametrize("sym", [ProductSymmetry((PhasedU1(), SU2())), GAUGED_U1SU2],
                         ids=["PhasedU1xSU2", "GaugedU1xSU2"])
def test_phased_charge_is_not_dropped(sym):
    A, B, C = (_space(sym, d) for d in _legs("U1xSU2"))
    rng = np.random.default_rng(4)
    t = random((A, B, C.dual()), (A,), NP, rng, sym=sym)
    x = t.to_dense()
    for _ in range(4):
        p1, p2 = _random_split(rng, 4)
        assert _err(permute(t, p1, p2).to_dense(), x.transpose(p1 + p2)) < TOL


# --------------------------------------------------------------------------- restricted layouts
def _full(sym, cod, dom):
    """Sectors, block shapes and per-sector trees of the layout, from unrestricted trees."""
    row, col = sl.Side(sym, cod), sl.Side(sym, dom)
    secs = tuple(c for c in row.trees.sectors if c in col.rows)

    def trees(side, c):
        a, b = side.trees.bounds[c]
        return [(side.trees.key(k), int(side.start[k]), int(side.deg[k])) for k in range(a, b)]

    return secs, {c: (row.rows[c], col.rows[c]) for c in secs}, {
        c: (trees(row, c), trees(col, c)) for c in secs}


def _layout_cases():
    rng = np.random.default_rng(7)
    out = []
    for name, sym in [("U1xSU2", U1SU2), ("U1", U1()), ("SU2", SU2()), ("Z3", Zn(3)),
                      ("SU2xZ2", SYMS["SU2xZ2"]), ("U1xA4xSU2", SYMS["U1xA4xSU2"])]:
        legs = [_space(sym, d) for d in _random_legs(name, rng)]
        for _ in range(6):
            n = int(rng.integers(2, 6))
            spaces = [legs[int(i)] for i in rng.integers(0, len(legs), n)]
            spaces = [V.dual() if rng.random() < 0.4 else V for V in spaces]
            k = int(rng.integers(0, n + 1))
            out.append((name, sym, tuple(spaces[:k]), tuple(spaces[k:])))
    return out


def _random_legs(name, rng):
    if name == "U1":
        return [{q: int(rng.integers(1, 3)) for q in range(-2, 3)}, {0: 1, 1: 2}, {-1: 1, 2: 1}]
    if name == "SU2":
        return [{j: 1 for j in range(5)}, {1: 2}, {0: 1, 2: 1}]
    if name == "Z3":
        return [{0: 1, 1: 2, 2: 1}, {1: 1}, {0: 2, 2: 1}]
    if name == "U1xSU2":
        return [{(q, j): 1 for q in (-2, -1, 0, 1, 2) for j in (0, 1, 2, 3)},
                {(0, 0): 1, (1, 1): 1, (2, 0): 1}, {(q, j): 2 for q in (-1, 1) for j in (1, 3)}]
    return _legs(name)


@pytest.mark.parametrize("case", _layout_cases(), ids=lambda c: c[0])
def test_restricted_trees_are_the_full_trees(case):
    _, sym, cod, dom = case
    lay = sl.Layout(sym, cod, dom)
    secs, shapes, trees = _full(sym, cod, dom)
    assert lay.sectors == secs and lay.shapes == shapes
    for c in secs:
        for side, want in zip((lay.row, lay.col), trees[c]):
            a, b = side.trees.bounds[c]
            assert [(side.trees.key(k), int(side.start[k]), int(side.deg[k]))
                    for k in range(a, b)] == want


def test_larger_side_keeps_the_coupled_sectors_of_the_smaller():
    """Three bonds against one: the codomain keeps only the trees of sectors the domain has."""
    V = _space(U1SU2, {(q, j): 1 for q in range(-3, 4) for j in range(4) if (q + j) % 2 == 0})
    W = _space(U1SU2, {(q, j): 1 for q in (-1, 0, 1) for j in range(4) if (q + j) % 2 == 0})
    full = sl.Trees(U1SU2, (V.sectors,) * 3, (False,) * 3)
    st.STRUCTURE_CACHE.clear()
    lay = sl.Layout(U1SU2, (V, V, V), (W,))
    assert len(lay.row.trees.coupled) < len(full.coupled) / 3
    assert lay.row.trees.sectors == tuple(c for c in full.sectors if c in W.sectors)
    lay = sl.Layout(U1SU2, (V, V), (V, V))
    assert lay.row.trees is lay.col.trees


@pytest.mark.parametrize("name", ["U1xSU2", "SU2", "U1xA4xSU2"])
def test_fuse_spaces_counts_rows(name):
    sym = SU2() if name == "SU2" else SYMS[name]
    legs = [_space(sym, d) for d in _random_legs(name, np.random.default_rng(1))]
    for spaces in [legs, [legs[0], legs[1].dual()], [legs[2].dual(), legs[0], legs[1]]]:
        side = sl.Side(sym, tuple(spaces))
        want = Space(sym, side.trees.sectors, tuple(side.rows[c] for c in side.trees.sectors))
        assert fuse_spaces(spaces) == want


# --------------------------------------------------------------------------- batched recoupling
def test_shared_matrices_share_one_product():
    rng = np.random.default_rng(3)
    V = _space(U1SU2, {(q, j): int(rng.integers(1, 4)) for q in range(-3, 4) for j in range(5)
                       if (q + j) % 2 == 0})
    P = _site()
    t = random((V, P, P), (V,), NP, rng)
    for p1, p2 in [((0, 2, 1), (3,)), ((0, 1), (2, 3)), ((0,), (3, 2, 1))]:
        u = permute(t, p1, p2)
        op = t.layout.map(NP, ("permute", p1, p2), None)
        products = sum(s[2][0] for s in op.stacks)
        assert 0 < products < op.n_groups
        o = op.n_scalar
        for offset, size, shape, M in op.stacks:          # contiguous, in order, covering
            assert offset == o and size == np.prod(shape) and M.shape[0] == shape[0]
            o += size
        assert o == len(op.src)
        assert _err(u.to_dense(), t.to_dense().transpose(p1 + p2)) < TOL


def test_torch_merges_only_without_new_shapes():
    """On torch a merge that adds a batched product is not taken; on numpy it is."""
    torch = pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    V = _space(U1SU2, {(q, j): 1 for q in range(-2, 3) for j in range(4)})
    W = _space(U1SU2, {(q, j): 2 for q in (-1, 1) for j in (1, 3)})
    cod, dom, perm = (V.dual(),), (_site(), W, V, _site()), ((3, 1, 0, 2), (4,))
    got = []
    for be in (NP, TorchBackend("cpu", torch.complex128)):
        t = random(cod, dom, be, np.random.default_rng(0))
        u = permute(t, *perm)
        x = be.to_numpy(t.to_dense()).transpose(perm[0] + perm[1])
        assert _err(be.to_numpy(u.to_dense()), x) < TOL
        op = t.layout.map(be, ("permute",) + perm, None)
        got.append((len(op.stacks), sum(s[2][0] for s in op.stacks), op.n_groups))
    (calls, products, groups), torch_got = got
    assert products < groups and torch_got == (calls - 1, groups, groups)


# --------------------------------------------------------------------------- physics cross-check
def _site_operators():
    """c_up, c_down, (n, S) per site of a two-site chain after Jordan-Wigner, in the basis
    site 1 x site 2, each site |0>, c+_up|0>, c+_down|0>, c+_up c+_down|0>."""
    a, z, e = np.array([[0, 1], [0, 0]]), np.diag([1, -1]), np.eye(2)
    modes = []
    for m in range(4):                                # (1 up), (1 down), (2 up), (2 down)
        ops = [z] * m + [a] + [e] * (3 - m)
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        modes.append(out)
    occ = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    perm = np.zeros((16, 16))
    for bits in itertools.product((0, 1), repeat=4):
        fock = int("".join(map(str, bits)), 2)
        perm[4 * occ[bits[:2]] + occ[bits[2:]], fock] = 1
    return [perm @ c @ perm.T for c in modes]


def _hubbard_term(t=1.0, u=4.0, j=0.7):
    c1u, c1d, c2u, c2d = _site_operators()
    d = (lambda x: x.conj().T)
    hop = sum(d(x) @ y + d(y) @ x for x, y in [(c1u, c2u), (c1d, c2d)])
    n = (lambda x: d(x) @ x)
    spin = [(0.5 * (d(up) @ dn + d(dn) @ up), 0.5j * (d(dn) @ up - d(up) @ dn),
             0.5 * (n(up) - n(dn))) for up, dn in [(c1u, c1d), (c2u, c2d)]]
    exchange = sum(s1 @ s2 for s1, s2 in zip(*spin))
    h = -t * hop + 0.5 * u * (n(c1u) @ n(c1d) + n(c2u) @ n(c2d)) + j * exchange
    return np.real_if_close(h).reshape(4, 4, 4, 4)


def test_hubbard_products_match_blocks_and_dense():
    P, h = _site(), _hubbard_term()
    H, lost = from_dense(h, (P, P), (P, P), NP)
    assert lost < 1e-24
    rng = np.random.default_rng(11)
    Q = _space(U1SU2, {(3, 1): 1})                    # three particles, total spin 1/2
    psi = random((P, P, P), (Q,), NP, rng)
    leg = (lambda s: blocks.Leg((0, 1, 2), (1, 2, 1), s))
    bh = blocks.from_dense(h, [leg(1), leg(1), leg(-1), leg(-1)])[0]
    bpsi = blocks.from_dense(psi.to_dense(), [leg(1), leg(1), leg(1),
                                               blocks.Leg((3,), (2,), -1)])[0]
    x = psi.to_dense()
    y = contract(H, psi, [2, 3], [0, 1])               # H_12: (s1', s2'; s3, q)
    y = contract(H, y, [2, 3], [1, 2])                 # H_23: (s2'', s3'; s1', q)
    want = np.tensordot(h, np.tensordot(h, x, ([2, 3], [0, 1])), ([2, 3], [1, 2]))
    by = blocks.tensordot(bh, bpsi, ([2, 3], [0, 1]))
    by = blocks.tensordot(bh, by, ([2, 3], [1, 2]))
    assert _err(y.to_dense(), want) < TOL
    assert _err(by.to_dense(), want) < TOL
    z = permute(y, [2, 0, 1], [3])                     # back on the spaces of psi
    assert abs(st.vdot(psi, z) - np.vdot(x, want.transpose(2, 0, 1, 3))) < TOL * np.abs(want).sum()


def test_hubbard_term_spectrum_by_sector():
    """The blocks of H are its spectrum by (charge, spin): eigenvalues with multiplicity
    dim(c) each give the dense spectrum."""
    P, h = _site(), _hubbard_term()
    H, _ = from_dense(h, (P, P), (P, P), NP)
    ev = np.concatenate([np.repeat(np.linalg.eigvalsh(b), U1SU2.dim(c))
                         for c, b in H.blocks().items()])
    assert _err(np.sort(ev), np.linalg.eigvalsh(h.reshape(16, 16))) < TOL


# --------------------------------------------------------------------------- fuzz
FUZZ_SYMS = {"SU2xU1": (ProductSymmetry((SU2(), U1())), [(0, 1, 2), (-1, 0, 1)]),
             "Z3xSU2": (ProductSymmetry((Zn(3), SU2())), [(0, 1, 2), (0, 1, 2)]),
             "A4xU1": (ProductSymmetry((A4(), U1())), [(0, 1, 2, 3), (-1, 0, 1)]),
             "U1xU1xSU2": (SYMS["U1xU1xSU2"], [(0, 1), (-1, 0), (0, 1)]),
             "GaugedU1xSU2": (GAUGED_U1SU2, [(-1, 0, 1), (0, 1, 2)]),
             "GaugedZ3xZ2": (GAUGED_Z3Z2, [(0, 1, 2), (0, 1)])}


def _small_leg(sym, labels, rng, dim=8):
    """A random leg of dense dimension at most `dim` on products of these factor labels."""
    out, n = {}, 0
    for a in rng.permutation(list(itertools.product(*labels))).tolist():
        a = tuple(a)
        if rng.random() < 0.6 and n + sym.dim(a) <= dim:
            out[a], n = 1, n + sym.dim(a)
    return _space(sym, out or {sym.unit(): 1})


def _fuzz_cases():
    rng = np.random.default_rng(2026)
    out = []
    for rank in (3, 4, 5, 6):
        for k in range(4):
            legs = [_bond(U1SU2, (-1, 0, 1), (0, 1, 2), rng), _site(),
                    _bond(U1SU2, (0, 1), (1, 3), rng)]
            if rank >= 5:
                legs = [_space(U1SU2, {(q, j): 1 for q in (-1, 0, 1) for j in (0, 1)
                                       if rng.random() < 0.7} or {(0, 0): 1}),
                        _space(U1SU2, {(0, 0): 1, (1, 1): 1}),
                        _space(U1SU2, {(-1, 1): 1, (0, 2): 1, (1, 1): 1})]
            out.append((f"U1xSU2-rank{rank}-{k}", U1SU2, rank, legs, int(rng.integers(1 << 30))))
    for name, (sym, labels) in FUZZ_SYMS.items():
        for rank in (3, 4, 5):
            for k in range(2):
                legs = [_small_leg(sym, labels, rng) for _ in range(3)]
                out.append((f"{name}-rank{rank}-{k}", sym, rank, legs, int(rng.integers(1 << 30))))
    return out


@pytest.mark.parametrize("case", _fuzz_cases(), ids=lambda c: c[0])
def test_fuzz_against_dense(case):
    _, sym, rank, legs, seed = case
    rng = np.random.default_rng(seed)
    for _ in range(20):
        spaces = [legs[int(i)] for i in rng.integers(0, 3, rank)]
        spaces = [V.dual() if rng.random() < 0.5 else V for V in spaces]
        n1 = int(rng.integers(0, rank + 1))
        cod, dom = tuple(spaces[:n1]), tuple(spaces[n1:])
        t = random(cod, dom, NP, rng, sym=sym)
        if t.layout.size:
            break
    else:
        pytest.skip("no invariant tensor")
    x = t.to_dense()
    for _ in range(3):
        p1, p2 = _random_split(rng, rank)
        assert _err(permute(t, p1, p2).to_dense(), x.transpose(p1 + p2)) < TOL
    legs_c = sorted(int(k) for k in rng.choice(rank, int(rng.integers(1, 3)), replace=False))
    b, axes_b = _partner(rng, sym, legs_c, cod, dom)
    c = contract(t, b, legs_c, axes_b)
    assert _err(c.to_dense(), np.tensordot(x, b.to_dense(), (legs_c, axes_b))) < TOL
    if n1 >= 2:
        f = fuse(t, [0, 1])
        assert _err(split(f, 0, cod[:2]).data, t.data) < TOL
    y, lost = from_dense(x, cod, dom, NP)
    assert lost < 1e-20 and _err(y.data, t.data) < TOL
