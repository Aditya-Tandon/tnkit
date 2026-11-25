"""Recoupling, contraction, fuse and split of symmetric tensors against dense arrays.

On U(1), Z_3, U(1) x U(1), SU(2), U(1) x SU(2), A4 in four gauges, A4 x SU(2) and A4 x A4, at
ranks 1 to 5 with mixed codomain-domain splits and dual flags, with and without a total-charge
leg:
  * `permute`, `repartition`, `transpose` and `conj` equal the dense transpose, and two
    permutations in a row equal their composition;
  * recoupling matrices equal overlaps of dense trees, which share no code with the F, R and cup
    moves that build them;
  * `contract` equals `numpy.tensordot`, down to a scalar;
  * `fuse` applies the unitary onto the fused leg, and `split` undoes it;
  * torch matches numpy; abelian permutations move entries exactly, and abelian contractions
    match `BlockTensor`;
  * none of this builds a dense tree, caches respect their limits, and a change of
    degeneracies alone gives fresh buffer positions on reused trees.
Also: four distinct leg spaces at ranks 4 to 6; real tensors, and real times complex; recoupling
arrays kept per backend type; clear errors for mismatched legs, symmetries and backends; a fixed
tensor contracted with ever new partners keeps a bounded number of layouts alive, and neither
the maps nor the keys of dead partners; a permuted pair of tensors is freed without the cyclic
garbage collector once it leaves the recent layouts; real and single-precision permutes, also on
a device without 64-bit types; the caches count every cast of a map; device maps have their own
limit; every cache limit can come from the environment.
"""
import gc
import itertools
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from oracles.a4 import A4RotatedCopies
from test_symmetric_tensor import LEGS, _err, _spaces, _sym
from tnkit import _recoupling as rc
from tnkit import _symmetric_layout as sl
from tnkit import blocks
from tnkit import fusion_trees as ft
from tnkit import symmetric_tensor as st
from tnkit._bytecache import ByteCache, env_cache, nbytes
from tnkit.backend import NumpyBackend
from tnkit.symmetric_tensor import (Space, contract, from_blocks, fuse, fuse_spaces, permute,
                                    random, repartition, split, transpose, zeros)

TOL = 1e-12
NP = NumpyBackend()
NAMES = list(LEGS) + ["A4rot"]


def _sym2(name):
    return A4RotatedCopies() if name == "A4rot" else _sym(name)


def _legs(name):
    return LEGS["A4" if name == "A4rot" else name]


def _charge_leg(sym, cod, dom):
    """A one-dimensional plain domain leg whose sector makes the tensor nonzero."""
    for a in [x for V in cod + dom for x in V.sectors]:
        q = Space(sym, (a,), (1,))
        if zeros(cod, dom + (q,), NP, sym=sym).layout.size:
            return q
    return None


def _shapes(A, B, C):
    """(codomain, domain) pairs of ranks 1 to 5 with mixed splits and dual flags."""
    a, b, c = A.dual(), B.dual(), C.dual()
    return [((A,), (A,)), ((A, b), (C,)), ((a,), (B, c)), ((A, B, c), ()), ((), (a, B, C)),
            ((A, b), (C, a)), ((a, B, C), (b,)), ((A, b, C, a), (B,)), ((B, c), (a, C, b))]


CASES = [(name, i, charged) for name in NAMES for i in range(9) for charged in (False, True)]


def _case(name, i, charged):
    sym = _sym2(name)
    cod, dom = _shapes(*_spaces(sym, *_legs(name)))[i]
    if charged:
        q = _charge_leg(sym, cod, dom)
        if q is None:
            pytest.skip("no charge sector")
        dom = dom + (q,)
    return sym, tuple(cod), tuple(dom)


def _ids(case):
    return f"{case[0]}-{case[1]}" + ("-q" if case[2] else "")


def _rng(case):
    return np.random.default_rng([NAMES.index(case[0]), case[1], case[2]])


def _random_split(rng, n):
    perm = [int(k) for k in rng.permutation(n)]
    k = int(rng.integers(0, n + 1))
    return perm[:k], perm[k:]


# --------------------------------------------------------------------------- permute
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_permute_is_the_dense_transpose(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    t = random(cod, dom, NP, rng, sym=sym)
    x, n, n1 = t.to_dense(), len(t.shape), len(cod)
    spaces = cod + dom
    for _ in range(3):
        p1, p2 = _random_split(rng, n)
        u = permute(t, p1, p2)
        assert u.codomain == tuple(spaces[k] if k < n1 else spaces[k].dual() for k in p1)
        assert u.domain == tuple(spaces[k] if k >= n1 else spaces[k].dual() for k in p2)
        assert _err(u.to_dense(), x.transpose(p1 + p2)) < TOL
    for m in range(n + 1):
        assert _err(repartition(t, m).to_dense(), x) < TOL
    tt = transpose(t)
    want = x.transpose(list(range(n - 1, n1 - 1, -1)) + list(range(n1 - 1, -1, -1)))
    assert _err(tt.to_dense(), want) < TOL
    assert permute(t, range(n1), range(n1, n)) is t


@pytest.mark.parametrize("case", CASES[::3], ids=_ids)
def test_repeated_recouplings_compose(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    t = random(cod, dom, NP, rng, sym=sym)
    n = len(t.shape)
    for _ in range(3):
        p, q = _random_split(rng, n), _random_split(rng, n)
        order = p[0] + p[1]
        two = permute(permute(t, *p), *q)
        one = permute(t, [order[k] for k in q[0]], [order[k] for k in q[1]])
        assert (two.codomain, two.domain) == (one.codomain, one.domain)
        assert _err(two.data, one.data) < TOL
    inv = list(np.argsort(p[0] + p[1]))
    back = permute(permute(t, *p), inv[:len(cod)], inv[len(cod):])
    assert _err(back.data, t.data) < TOL


@pytest.mark.parametrize("name", ["SU2", "A4", "A4c", "A4r", "A4rot", "U1xSU2", "A4xSU2"])
def test_recoupling_matrices_are_overlaps_of_dense_trees(name):
    """M[i, j] = <P'_j, P_i o transpose> / dim c', from dense trees built with `tree_tensor`."""
    sym = _sym2(name)
    A, B, C = _spaces(sym, *_legs(name))
    rng = np.random.default_rng(5)
    checked = 0
    for cod, dom in [((A, B.dual()), (C,)), ((B,), (A.dual(), C)), ((A, B), (C.dual(), A))]:
        spaces = cod + dom
        n1, n = len(cod), len(spaces)
        flags = tuple(V.isdual for V in spaces)
        for p1, p2 in [_random_split(rng, n) for _ in range(3)]:
            p1, p2 = tuple(p1), tuple(p2)
            for secs in itertools.product(*(V.sectors for V in spaces)):
                leaves = tuple(V.leaf(a) for V, a in zip(spaces, secs))
                old = rc.tree_pairs(sym, leaves, flags, n1)
                if not old:
                    continue
                M = rc.recoupling_matrix(sym, n1, p1, p2, leaves, flags)
                moved = [(leaves[k], flags[k]) if (k < n1) == (j < len(p1)) else
                         (sym.dual(leaves[k]), not flags[k]) for j, k in enumerate(p1 + p2)]
                new = rc.tree_pairs(sym, tuple(x for x, _ in moved),
                                     tuple(y for _, y in moved), len(p1))
                want = np.array([[np.vdot(_pair(sym, s2, f2), _pair(sym, s, f).transpose(p1 + p2))
                                  / sym.dim(s2.coupled) for s2, f2 in new] for s, f in old])
                assert _err(M, want) < TOL
                checked += 1
    assert checked > 20


def _pair(sym, s, f):
    X, Y = ft.tree_tensor(sym, s), ft.tree_tensor(sym, f)
    return np.tensordot(X, Y.conj(), axes=(-1, -1))


# --------------------------------------------------------------------------- contract
def _partner(rng, sym, legs, cod, dom):
    """A tensor whose legs, at random places and sides, contract with `legs` of (cod, dom), plus
    one free leg; returns it and the positions of the contracted legs."""
    n1 = len(cod)
    spaces = cod + dom
    want = [spaces[k] if k < n1 else spaces[k].dual() for k in legs]  # as seen from a codomain
    extra = cod[0] if cod else dom[0]
    items = [("c", V) for V in want] + [("f", extra)]
    order = list(rng.permutation(len(items)))
    side = rng.integers(0, 2, len(items))
    bc, bd = [], []
    for j in order:
        kind, V = items[j]
        if kind == "c":
            (bc if side[j] else bd).append((j, V.dual() if side[j] else V))
        else:
            (bc if side[j] else bd).append((j, V))
    pos = {j: p for p, (j, _) in enumerate(bc + bd)}
    b = random([V for _, V in bc], [V for _, V in bd], NP, rng, sym=sym)
    return b, [pos[j] for j in range(len(legs))]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_contract_is_tensordot(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    a = random(cod, dom, NP, rng, sym=sym)
    n = len(a.shape)
    for k in range(1, min(n, 3) + 1):
        legs = [int(x) for x in rng.choice(n, k, replace=False)]
        b, axes_b = _partner(rng, sym, legs, cod, dom)
        c = contract(a, b, legs, axes_b)
        want = np.tensordot(a.to_dense(), b.to_dense(), (legs, axes_b))
        assert _err(c.to_dense(), want) < TOL * max(1.0, np.abs(want).max(initial=0))


@pytest.mark.parametrize("name", NAMES)
def test_contract_to_a_scalar_and_mismatched_legs(name):
    sym = _sym2(name)
    A, B, C = _spaces(sym, *_legs(name))
    rng = np.random.default_rng(7)
    a = random((A, B.dual()), (C,), NP, rng, sym=sym)
    b = random((C,), (A, B.dual()), NP, rng, sym=sym)
    c = contract(a, b, [0, 1, 2], [1, 2, 0])
    want = np.tensordot(a.to_dense(), b.to_dense(), ([0, 1, 2], [1, 2, 0]))
    assert c.to_dense().shape == () and abs(c.to_dense() - want) < TOL * max(1, abs(want))
    with pytest.raises(ValueError, match="dual"):
        contract(a, a, [0], [0])
    with pytest.raises(ValueError):
        contract(a, b, [0, 0], [1, 2])


# --------------------------------------------------------------------------- fuse and split
def _fuser(spaces):
    """The dense unitary U from the product of `spaces` onto their fused leg: the conjugate
    transpose of the map whose reduced blocks are identities."""
    V = fuse_spaces(spaces)
    S = from_blocks({c: np.eye(n) for c, n in zip(V.sectors, V.degeneracies)}, spaces, (V,), NP)
    S = S.to_dense().reshape(-1, V.dim)
    assert _err(S.conj().T @ S, np.eye(V.dim)) < TOL and _err(S @ S.conj().T, np.eye(len(S))) < TOL
    return S.conj().T


@pytest.mark.parametrize("case", [c for c in CASES if c[1] in (3, 5, 7, 8)], ids=_ids)
def test_fuse_and_split(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    t = random(cod, dom, NP, rng, sym=sym)
    x, n1, n = t.to_dense(), len(cod), len(t.shape)
    spaces = cod + dom
    groups = [g for g in ([0, 1], [1, 2], [2, 0], list(range(n1))) if len(g) and max(g) < n1]
    groups += [[n1 + k for k in g] for g in ([0, 1], [1, 0]) if n1 + max(g) < n]
    for axes in groups:
        legs = [spaces[k] for k in axes]
        U = _fuser(legs)
        f = fuse(t, axes)
        y = np.moveaxis(x, axes, range(len(axes))).reshape((U.shape[1],) + tuple(
            s for k, s in enumerate(x.shape) if k not in axes))
        y = np.tensordot(U if axes[0] < n1 else U.conj(), y, axes=(1, 0))
        rest = [k for k in range(n) if k not in axes or k == axes[0]]
        y = np.moveaxis(y, 0, rest.index(axes[0]))
        assert _err(f.to_dense(), y) < TOL
        back = split(f, rest.index(axes[0]), legs)
        order = [j for k in rest for j in (axes if k == axes[0] else [k])]
        want = permute(t, order[:n1], order[n1:])
        assert (back.codomain, back.domain) == (want.codomain, want.domain)
        assert _err(back.data, want.data) < TOL
    with pytest.raises(ValueError):
        fuse(t, [0, n - 1]) if 0 < n1 < n else fuse(t, [])


# --------------------------------------------------------------------------- torch
@pytest.mark.parametrize("name", NAMES)
def test_torch_matches_numpy(name):
    pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    tb = TorchBackend("cpu")
    for i in (1, 5, 8):
        sym, cod, dom = _case(name, i, False)
        rng = np.random.default_rng(i)
        a = random(cod, dom, NP, rng, sym=sym)
        ta = st.SymmetricTensor(cod, dom, tb.array(a.data), tb, sym)
        n = len(a.shape)
        p1, p2 = _random_split(rng, n)
        for x, y in [(permute(ta, p1, p2), permute(a, p1, p2)), (ta.conj(), a.conj()),
                     (fuse(ta, [0, 1]), fuse(a, [0, 1]))]:
            assert _err(tb.to_numpy(x.data), y.data) < TOL
        b, axes_b = _partner(rng, sym, [0, n - 1], cod, dom)
        tbb = st.SymmetricTensor(b.codomain, b.domain, tb.array(b.data), tb, sym)
        assert _err(tb.to_numpy(contract(ta, tbb, [0, n - 1], axes_b).data),
                    contract(a, b, [0, n - 1], axes_b).data) < TOL


# --------------------------------------------------------------------------- abelian
@pytest.mark.parametrize("case", [c for c in CASES if c[0] in ("U1", "Z3", "U1xU1")], ids=_ids)
def test_abelian_permutations_move_entries_exactly(case):
    sym, cod, dom = _case(*case)
    rng = _rng(case)
    t = random(cod, dom, NP, rng, sym=sym)
    x = t.to_dense()
    for _ in range(3):
        p1, p2 = _random_split(rng, len(t.shape))
        u = permute(t, p1, p2)
        assert np.array_equal(u.to_dense(), x.transpose(p1 + p2))
        assert np.array_equal(np.sort(u.data), np.sort(t.data))


def _block_leg(V, in_domain):
    sign = (-1 if in_domain else 1) * (-1 if V.isdual else 1)
    return blocks.Leg(V.sectors, V.degeneracies, sign, V.sym)


@pytest.mark.parametrize("i", range(9))
def test_abelian_contraction_matches_block_tensors(i):
    sym, cod, dom = _case("U1xU1", i, False)
    rng = np.random.default_rng(i)
    a = random(cod, dom, NP, rng, sym=sym)
    n = len(a.shape)
    legs = [0, n - 1] if n > 1 else [0]
    b, axes_b = _partner(rng, sym, legs, cod, dom)
    c = contract(a, b, legs, axes_b)
    ba = blocks.from_dense(a.to_dense(), [_block_leg(V, False) for V in a.codomain]
                           + [_block_leg(V, True) for V in a.domain])[0]
    bb = blocks.from_dense(b.to_dense(), [_block_leg(V, False) for V in b.codomain]
                           + [_block_leg(V, True) for V in b.domain])[0]
    assert _err(c.to_dense(), blocks.tensordot(ba, bb, (legs, axes_b)).to_dense()) < TOL


# --------------------------------------------------------------------------- caches and layout
def test_recoupling_and_contraction_build_no_dense_tree(monkeypatch):
    def refuse(*args):
        raise AssertionError("dense tree built")
    sym = _sym2("A4xSU2")
    A, B, C = _spaces(sym, *_legs("A4xSU2"))
    rng = np.random.default_rng(3)
    a = random((A, B.dual()), (C, A), NP, rng, sym=sym)
    b = random((C, A), (B,), NP, rng, sym=sym)
    st.RECOUPLING_CACHE.clear()
    st.INDEX_CACHE.clear()
    for mod, name in [(ft, "tree_tensor"), (st, "tree_tensor"), (st, "_pairs"),
                      (ft, "_tree_tensor")]:
        monkeypatch.setattr(mod, name, refuse)
    permute(a, [3, 1], [2, 0])
    contract(a, b, [2, 3], [0, 1])
    split(fuse(a, [0, 1]), 0, (A, B.dual()))
    a.conj()


def test_caches_respect_their_limits():
    sym = _sym2("A4xSU2")
    A, B, C = _spaces(sym, *_legs("A4xSU2"))
    t = random((A, B, C.dual()), (A.dual(), B), NP, np.random.default_rng(1), sym=sym)
    x = t.to_dense()
    caches = (st.RECOUPLING_CACHE, st.INDEX_CACHE, ft.TREE_CACHE)
    old = [c.max_bytes for c in caches]
    try:
        for c in caches:
            c.set_limit(2000)
        for p1, p2 in [([4, 0], [1, 2, 3]), ([2, 3, 1], [0, 4]), ([1], [4, 3, 2, 0])]:
            assert _err(permute(t, p1, p2).to_dense(), x.transpose(p1 + p2)) < TOL
            for c in caches:
                assert c.bytes <= c.max_bytes or len(c) == 1
        assert st.RECOUPLING_CACHE.stats["evictions"] > 0
        info = st.cache_info()
        assert info["recoupling"]["bytes"] == st.RECOUPLING_CACHE.bytes
        assert info["layouts"]["bytes"] > 0
    finally:
        for c, m in zip(caches, old):
            c.set_limit(m)


def test_new_degeneracies_reuse_trees_and_get_fresh_positions():
    """Same sectors, other degeneracies: no tree structure, permute structure or recoupling
    matrix is rebuilt, and the results are still right."""
    sym = _sym2("SU2")
    A, B, C = _spaces(sym, *_legs("SU2"))
    A2 = Space(sym, A.sectors, tuple(n + 2 for n in A.degeneracies))
    C2 = Space(sym, C.sectors, (3, 1, 2))
    rng = np.random.default_rng(2)
    caches = (st.STRUCTURE_CACHE, rc.PERMUTE_CACHE, st.RECOUPLING_CACHE)
    for k, (a, c) in enumerate([(A, C), (A2, C2)]):
        t = random((a, B.dual()), (c,), NP, rng, sym=sym)
        if k:
            misses = [c.stats["misses"] for c in caches]
        for p1, p2 in [([2, 0], [1]), ([1], [0, 2])]:
            assert _err(permute(t, p1, p2).to_dense(), t.to_dense().transpose(p1 + p2)) < TOL
        assert _err(split(fuse(t, [0, 1]), 0, (a, B.dual())).data, t.data) < TOL
    assert [c.stats["misses"] for c in caches] == misses


def test_layouts_are_shared_and_released():
    sym = _sym2("SU2")
    A, B, _ = _spaces(sym, *_legs("SU2"))
    a, b = zeros((A,), (B, A), NP), random((A,), (B, A), NP, np.random.default_rng(0))
    assert a.layout is b.layout
    key = (sym, (A,), (B, A))
    del a, b
    sl.RECENT_LAYOUTS.clear()
    gc.collect()
    assert key not in sl.LAYOUTS


def test_cache_info_lists_every_cache():
    sym = _sym2("SU2")
    A, B, C = _spaces(sym, *_legs("SU2"))
    t = random((A, B), (C,), NP, np.random.default_rng(0), sym=sym)
    split(fuse(permute(t, [2], [1, 0]), [1, 2]), 1, (B.dual(), A.dual()))
    info = st.cache_info()
    for name in ("recoupling", "permutes", "indices", "device_indices", "recent_layouts",
                 "structure", "dense_trees", "dense_pairs", "swaps", "bends", "bend_coefficients",
                 "fusions"):
        assert {"entries", "hits", "misses"} <= set(info[name]), name
    for name in ("swaps", "bends", "bend_coefficients", "fusions"):
        assert info[name]["max_entries"] > 0
    assert info["indices"]["bytes"] == st.INDEX_CACHE.bytes > 0
    assert info["label_tables"]["entries"] > 0 and info["label_tables"]["bytes"] > 0
    assert info["trees"]["entries"] > 0 and info["layouts"]["entries"] > 0


def test_index_cache_limit():
    sym = _sym2("U1xSU2")
    A, B, C = _spaces(sym, *_legs("U1xSU2"))
    t = random((A, B.dual(), C), (A,), NP, np.random.default_rng(4), sym=sym)
    x = t.to_dense()
    old = st.INDEX_CACHE.max_bytes
    try:
        st.INDEX_CACHE.set_limit(1)
        for _ in range(2):
            for p1, p2 in [([3, 0], [1, 2]), ([2], [0, 1, 3]), ([1, 3, 0], [2])]:
                assert _err(permute(t, p1, p2).to_dense(), x.transpose(p1 + p2)) < TOL
                assert len(st.INDEX_CACHE) == 1
        assert st.INDEX_CACHE.stats["evictions"] > 0
    finally:
        st.INDEX_CACHE.set_limit(old)


# --------------------------------------------------------------------------- four leg spaces
FOURTH = {"SU2": {1: 1, 3: 2}, "U1xSU2": {(-1, 1): 1, (1, 0): 2}, "A4": {2: 1, 3: 2},
          "A4xSU2": {(3, 0): 1, (1, 1): 1}, "U1xU1": {(1, 0): 2, (0, -1): 1}}


def _four(name):
    sym = _sym2(name)
    return sym, _spaces(sym, *_legs(name), FOURTH[name])


@pytest.mark.parametrize("name", list(FOURTH))
def test_four_leg_spaces(name):
    sym, (A, B, C, D) = _four(name)
    a, b, c, d = A.dual(), B.dual(), C.dual(), D.dual()
    shapes = [((A, D), (B, C)), ((D, b, C), (a, d)), ((A, b, C, D), (d,))]
    if name == "SU2":
        shapes.append(((A, B, D), (c, d, A)))
    rng = np.random.default_rng(list(FOURTH).index(name))
    for cod, dom in shapes:
        t = random(cod, dom, NP, rng, sym=sym)
        x, n = t.to_dense(), len(t.shape)
        for _ in range(3):
            p1, p2 = _random_split(rng, n)
            assert _err(permute(t, p1, p2).to_dense(), x.transpose(p1 + p2)) < TOL
        legs = [int(k) for k in rng.choice(n, 2, replace=False)]
        u, axes_u = _partner(rng, sym, legs, cod, dom)
        want = np.tensordot(x, u.to_dense(), (legs, axes_u))
        assert _err(contract(t, u, legs, axes_u).to_dense(), want) < TOL * max(
            1.0, np.abs(want).max())
        f = fuse(t, [0, 1])
        assert _err(split(f, 0, cod[:2]).data, t.data) < TOL
        assert _err(f.to_dense().reshape(-1), (
            np.tensordot(_fuser(cod[:2]), x.reshape((-1,) + x.shape[2:]), (1, 0)).reshape(-1)
        )) < TOL


# --------------------------------------------------------------------------- backends
RE = NumpyBackend(np.float64)


@pytest.mark.parametrize("name", ["U1", "SU2", "U1xSU2"])
def test_real_backend(name):
    """Real tensors on symmetries with real recoupling matrices, and real times complex."""
    sym = _sym2(name)
    A, B, C = _spaces(sym, *_legs(name))
    rng = np.random.default_rng(8)
    a = random((A, B.dual()), (C, A), RE, rng, sym=sym)
    assert a.data.dtype == np.float64
    x = a.to_dense()
    for p1, p2 in [([3, 0], [1, 2]), ([2], [0, 1, 3]), ([1, 3, 0], [2])]:
        u = permute(a, p1, p2)
        assert u.data.dtype == np.float64
        assert _err(u.to_dense(), x.transpose(p1 + p2)) < TOL
    b_re, axes_b = _partner(rng, sym, [0, 3], a.codomain, a.domain)
    b_re = st.SymmetricTensor(b_re.codomain, b_re.domain, b_re.data.real.copy(), RE, sym)
    b_cx = random(b_re.codomain, b_re.domain, NP, rng, sym=sym)
    a_cx = st.SymmetricTensor(a.codomain, a.domain, a.data.astype(complex), NP, sym)
    for left, right in [(a, b_re), (a, b_cx), (a_cx, b_re)]:
        c = contract(left, right, [0, 3], axes_b)
        want = np.tensordot(left.to_dense(), right.to_dense(), ([0, 3], axes_b))
        assert c.data.dtype == np.result_type(left.data, right.data)
        assert c.be.dtype == c.data.dtype
        assert _err(c.to_dense(), want) < TOL * max(1.0, np.abs(want).max())


class _Other(NumpyBackend):
    """A numpy backend of another type."""


def test_maps_are_kept_per_backend_type():
    """Same device and dtype, other backend type: its own recoupling arrays."""
    sym = _sym2("SU2")
    A, B, C = _spaces(sym, *_legs("SU2"))
    t = random((A, B), (C.dual(), A), NP, np.random.default_rng(6), sym=sym)
    be = _Other()
    u = st.SymmetricTensor(t.codomain, t.domain, t.data.copy(), be, sym)
    want = permute(t, [3, 1], [0, 2])
    got = permute(u, [3, 1], [0, 2])
    assert got.be is be
    assert {k[1][0] for k in st.INDEX_CACHE.keys() if k[0] == t.layout.uid} >= {
        "NumpyBackend", "_Other"}
    assert np.array_equal(got.data, want.data)


def test_one_map_serves_every_dtype():
    """A map built on one dtype serves another: its matrices are cast to each dtype once,
    never used in the wrong one (A4 has complex recoupling matrices)."""
    perms = [([3, 1], [2, 0]), ([2], [0, 1, 3]), ([1, 3, 0], [2])]
    for name, first, second in [("A4", np.complex64, np.complex128),
                                ("U1xSU2", np.float64, np.complex128)]:
        sym = _sym2(name)
        A, B, C = _spaces(sym, *_legs(name))
        hi = NumpyBackend(second)
        t = random((A, B.dual()), (C, A), hi, np.random.default_rng(13), sym=sym)
        lo_data = t.data.astype(first) if first is np.complex64 else t.data.real.copy()
        lo = st.SymmetricTensor(t.codomain, t.domain, lo_data, NumpyBackend(first), sym)
        x, x_lo = t.to_dense(), lo.to_dense()
        st.INDEX_CACHE.clear()
        for p1, p2 in perms:
            u, v = permute(lo, p1, p2), permute(t, p1, p2)
            assert u.data.dtype == first and v.data.dtype == second
            assert _err(u.to_dense(), x_lo.transpose(p1 + p2)) < (1e-5 if name == "A4" else TOL)
            assert _err(v.to_dense(), x.transpose(p1 + p2)) < TOL
        assert len([k for k in st.INDEX_CACHE.keys() if k[0] == t.layout.uid]) == len(perms)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_contract_batches_blocks_of_one_shape(backend):
    """Blocks of one shape that follow one another take one strided product, and blocks of one
    shape apart in the buffers one gathered product; each block gets its own product (U(1):
    the block of sector q is n_q x n_q)."""
    be = NP
    if backend == "torch":
        pytest.importorskip("torch")
        from tnkit.backend import TorchBackend
        be = TorchBackend("cpu")
    sym = _sym2("U1")
    for degs, runs, gathers in [((2, 2, 2, 3, 3), [3, 2], 0), ((2, 3, 2, 3, 2, 3, 2, 3), [], 2)]:
        V = Space(sym, tuple(range(len(degs))), degs)
        rng = np.random.default_rng(14)
        a, b = (random((V,), (V,), NP, rng) for _ in range(2))
        a, b = (st.SymmetricTensor(t.codomain, t.domain, be.array(t.data), be) for t in (a, b))
        st.INDEX_CACHE.clear()
        c = contract(a, b, [1], [0])
        want = np.asarray(be.to_numpy(a.to_dense())) @ np.asarray(be.to_numpy(b.to_dense()))
        assert _err(be.to_numpy(c.to_dense()), want) < TOL
        prod, = [op for op in st.INDEX_CACHE.values() if isinstance(op, st._Products)]
        assert sorted(r[0][0] for r in prod.runs if r[0][0] > 1) == sorted(runs)
        assert len(prod.gathers) == gathers


def test_mismatched_tensors_raise_clear_errors():
    sym = _sym2("SU2")
    A, B, C = _spaces(sym, *_legs("SU2"))
    rng = np.random.default_rng(9)
    a = random((A, B), (C,), NP, rng, sym=sym)
    A2 = Space(sym, A.sectors, tuple(n + 1 for n in A.degeneracies))
    for leg, match in [(A2, "degeneracies"), (C, "sectors"), (A.dual(), "dual")]:
        b = random((B,), (leg,), NP, rng, sym=sym)
        with pytest.raises(ValueError, match=match):
            contract(a, b, [0], [1])
    U = _spaces(_sym2("U1"), *_legs("U1"))[0]
    with pytest.raises(ValueError, match="symmetries"):
        contract(a, random((U,), (U,), NP, rng), [0], [1])
    torch = pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    tb = TorchBackend("cpu")
    ta = st.SymmetricTensor(a.codomain, a.domain, torch.as_tensor(a.data), tb, sym)
    b = random((C,), (A, B), NP, rng, sym=sym)
    with pytest.raises(TypeError, match="numpy and a torch"):
        contract(ta, b, [2], [0])
    with pytest.raises(TypeError, match="numpy and a torch"):
        ta + a


def test_byte_cache_two_threads_missing_one_key():
    import threading
    from tnkit._bytecache import ByteCache
    cache, barrier = ByteCache(1 << 20), threading.Barrier(2)

    def build():
        barrier.wait()                           # both threads miss before either stores
        return np.zeros(100)

    out = []
    threads = [threading.Thread(target=lambda: out.append(cache.get("k", build)))
               for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(cache) == 1 and cache.bytes == 800 and out[0] is out[1]


# --------------------------------------------------------------------------- layout lifetime
def _live(*keys):
    return [k for k in keys if k in sl.LAYOUTS]


def test_fixed_tensor_meets_changing_partners():
    """An operator contracted with tensors of ever new degeneracies keeps a bounded number of
    layouts alive, set by the entry limit of `RECENT_LAYOUTS`, and every product is right."""
    sym = _sym2("SU2")
    P, W = Space(sym, (1,), (1,)), Space(sym, (0, 2), (2, 1))
    rng = np.random.default_rng(10)
    M = random((W, P), (P, W), NP, rng, sym=sym)
    old = (st.RECENT_LAYOUTS.max_bytes, st.RECENT_LAYOUTS.max_entries)
    counts = []
    try:
        st.RECENT_LAYOUTS.set_limit(old[0], 4)
        gc.collect()
        base = len(sl.LAYOUTS)
        for step in range(200):
            V = Space(sym, (0, 1, 2, 3), tuple(int(n) for n in rng.integers(1, 7, 4)))
            A = random((V, P), (V,), NP, rng, sym=sym)
            x = contract(M, A, [2], [1])               # W, P, W | V, V
            y = permute(x, [3, 0, 1], [2, 4])
            if step % 10 == 0:
                want = np.tensordot(M.to_dense(), A.to_dense(), ([2], [1]))
                assert _err(y.to_dense(), want.transpose(3, 0, 1, 2, 4)) < TOL
            del A, x, y
            if step in (99, 199):
                gc.collect()
                counts.append(len(sl.LAYOUTS))
    finally:
        st.RECENT_LAYOUTS.set_limit(*old)
    assert counts[1] <= counts[0] <= base + 12


def test_permuted_pair_is_freed_without_gc():
    sym = _sym2("SU2")
    A, B, C = _spaces(sym, *_legs("SU2"))
    A = Space(sym, A.sectors, (5, 7))                  # spaces no other test uses
    gc.disable()
    try:
        t = random((A, B), (C, A), NP, np.random.default_rng(11), sym=sym)
        u = permute(t, [3, 0], [1, 2])
        v = permute(u, [1, 2], [3, 0])
        w = split(fuse(v, [0, 1]), 0, (A, B))
        assert _err(w.data, t.data) < TOL
        keys = [x.layout.key for x in (t, u, v, w)] + [fuse(v, [0, 1]).layout.key]
        uids = {x.layout.uid for x in (t, u, v, w)}
        assert _live(*keys)
        del t, u, v, w
        sl.RECENT_LAYOUTS.clear()
        assert not _live(*keys)
        assert not [k for k in st.INDEX_CACHE.keys() if k[0] in uids]
    finally:
        gc.enable()


def _wide(sym, name):
    """The three test legs of `name` with two more states in every sector."""
    return [Space(sym, V.sectors, tuple(n + 2 for n in V.degeneracies))
            for V in _spaces(sym, *_legs(name))]


@pytest.mark.parametrize("limit, dtype", [(None, np.int32), (8, np.int64)])
def test_index_width(monkeypatch, limit, dtype):
    """32-bit indices and positions into buffers below the limit, 64-bit from it; the same
    results."""
    if limit is not None:
        monkeypatch.setattr(sl, "INDEX32_LIMIT", limit)
    sym = _sym2("SU2")
    A, B, C = _wide(sym, "SU2")
    t = random((A, B.dual()), (C, A), NP, np.random.default_rng(12), sym=sym)
    assert t.layout.size > 2 ** 10
    x = t.to_dense()
    st.INDEX_CACHE.clear()
    for p1, p2 in [([3, 0], [1, 2]), ([2], [0, 1, 3])]:
        assert _err(permute(t, p1, p2).to_dense(), x.transpose(p1 + p2)) < TOL
    assert _err(split(fuse(t, [0, 1]), 0, (A, B.dual())).data, t.data) < TOL
    ops = st.INDEX_CACHE.values()
    index = [op.src for op in ops if hasattr(op, "src")] + [
        op.inv for op in ops if getattr(op, "inv", None) is not None] + [
        op.index for op in ops if hasattr(op, "index")]
    assert index and all(i.dtype == dtype for i in index)
    st.INDEX_CACHE.clear()


# --------------------------------------------------------------------------- dtypes and devices
LOW = {"float64": (np.float64, TOL), "float32": (np.float32, 1e-5),
       "complex64": (np.complex64, 1e-5)}


def _torch_cpu():
    pytest.importorskip("torch")
    from tnkit.backend import TorchBackend
    return TorchBackend("cpu")


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("name", ["A4", "A4r", "A4xSU2", "U1xSU2"])
def test_real_and_single_precision_permutes(name, backend):
    """Real and single-precision data keep their dtype under permute and match the complex
    result, on symmetries whose one-pair factors are stored complex (A4); a real tensor that
    needs a complex matrix raises."""
    tb = _torch_cpu() if backend == "torch" else None
    sym = _sym2(name)
    A, B, C = _spaces(sym, *_legs(name))
    t = random((A, B.dual()), (C, A), NP, np.random.default_rng(15), sym=sym)
    perms = [([3, 1], [2, 0]), ([2], [0, 1, 3]), ([1, 3, 0], [2])]
    for label, (dtype, tol) in LOW.items():
        data = t.data.real if np.dtype(dtype).kind == "f" else t.data
        data = data.astype(dtype)
        if tb is None:
            u = st.SymmetricTensor(t.codomain, t.domain, data, NumpyBackend(dtype), sym)
        else:
            be = type(tb)("cpu", getattr(tb.torch, label))
            u = st.SymmetricTensor(t.codomain, t.domain, tb.torch.as_tensor(data), be, sym)
        ref = st.SymmetricTensor(t.codomain, t.domain, data.astype(complex), NP, sym)
        done = 0
        for p1, p2 in perms:
            try:
                got = permute(u, p1, p2)
            except TypeError as e:
                assert "complex recoupling matrix" in str(e) and label != "complex64"
                continue
            out = got.data if tb is None else got.data.numpy()
            assert out.dtype == np.dtype(dtype), (label, p1, p2)
            assert _err(out, permute(ref, p1, p2).data) < tol
            done += 1
        assert done, label


class _No64(object):
    """torch, except that `as_tensor` refuses to make a 64-bit float or complex tensor, as on
    an MPS device."""

    def __init__(self, torch):
        self._torch = torch

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def as_tensor(self, x, dtype=None, device=None):
        d = dtype if dtype is not None else self._torch.as_tensor(np.zeros(0, x.dtype)).dtype \
            if isinstance(x, np.ndarray) else x.dtype
        if d in (self._torch.float64, self._torch.complex128):
            raise TypeError(f"no 64-bit types on this device: {d}")
        return self._torch.as_tensor(x, dtype=dtype, device=device)


def _single(be, t):
    return st.SymmetricTensor(t.codomain, t.domain, be.array(t.data), be, t.sym)


@pytest.mark.parametrize("name", ["U1xSU2", "A4"])
def test_single_precision_device_gets_no_64_bit_arrays(name):
    tb = _torch_cpu()
    be = type("No64Backend", (type(tb),), {})("cpu", tb.torch.complex64)
    be.torch = _No64(tb.torch)
    sym = _sym2(name)
    A, B, C = _spaces(sym, *_legs(name))
    rng = np.random.default_rng(16)
    a = random((A, B.dual()), (C, A), NP, rng, sym=sym)
    b = random((C, A), (B,), NP, rng, sym=sym)
    u, v = _single(be, a), _single(be, b)
    assert _err(permute(u, [3, 1], [2, 0]).data.numpy(), permute(a, [3, 1], [2, 0]).data) < 1e-5
    assert _err(contract(u, v, [2, 3], [0, 1]).data.numpy(), contract(a, b, [2, 3], [0, 1]).data
                ) < 1e-5
    assert abs(float(st.norm(u)) - st.norm(a)) < 1e-5 * st.norm(a)
    assert abs(complex(st.vdot(u, u)) - st.vdot(a, a)) < 1e-5 * st.norm(a) ** 2


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_single_and_double_precision_on_a_device(device):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()):
        pytest.skip(f"no {device} device")
    from tnkit.backend import TorchBackend
    sym = _sym2("U1xSU2")
    A, B, C = _spaces(sym, *_legs("U1xSU2"))
    rng = np.random.default_rng(17)
    a = random((A, B.dual()), (C, A), NP, rng, sym=sym)
    b = random((C, A), (B,), NP, rng, sym=sym)
    dtypes = [(torch.complex64, 1e-5)] + ([(torch.complex128, TOL)] if device == "cuda" else [])
    for dtype, tol in dtypes:
        be = TorchBackend(device, dtype)
        u, v = _single(be, a), _single(be, b)
        got = permute(u, [3, 1], [2, 0]).data
        assert got.dtype == dtype
        assert _err(got.cpu().numpy(), permute(a, [3, 1], [2, 0]).data) < tol
        assert _err(contract(u, v, [2, 3], [0, 1]).data.cpu().numpy(),
                    contract(a, b, [2, 3], [0, 1]).data) < tol
        assert abs(float(st.norm(u)) - st.norm(a)) < tol * st.norm(a)
    keys = [k for k in st.DEVICE_INDEX_CACHE.keys() if k[1][1].startswith(device)]
    assert keys and not [k for k in st.INDEX_CACHE.keys() if k[1][1].startswith(device)]
    if device == "cuda":
        total = torch.cuda.get_device_properties(be.device).total_memory
        if st.DEVICE_INDEX_CACHE.source == "default-device":
            assert st.DEVICE_INDEX_CACHE.max_bytes == min(2 ** 31, int(0.1 * total))


def _arrays(x):
    if isinstance(x, np.ndarray) or type(x).__name__ == "Tensor":
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _arrays(y)
    elif isinstance(x, dict):
        yield from _arrays(list(x.values()))


def _recount(cache):
    """The bytes of every distinct array the cached maps hold, found apart from their own
    count."""
    held = {}
    for v in cache.values():
        for a in _arrays(list(vars(v).values())):
            at = a.__array_interface__["data"][0] if isinstance(a, np.ndarray) else a.data_ptr()
            held[at] = max(held.get(at, 0), nbytes(a))
    return sum(held.values())


def test_casts_are_counted_and_limited():
    """Maps and weights grow as real, then complex, then single-precision data meet them; the
    cache counts every cast and stays within its limit."""
    sym = _sym2("SU2")
    A, B, C = _wide(sym, "SU2")
    rng = np.random.default_rng(18)
    t = random((A, B.dual(), C), (A,), NP, rng, sym=sym)
    s = random(t.codomain, t.domain, NP, rng, sym=sym)
    perms = [([3, 0], [1, 2]), ([2], [0, 1, 3]), ([1, 3, 0], [2]), ([0, 2], [3, 1])]
    st.INDEX_CACHE.clear()
    for p1, p2 in perms:
        permute(t, p1, p2)
    full = st.INDEX_CACHE.bytes
    old = st.INDEX_CACHE.max_bytes
    try:
        st.INDEX_CACHE.set_limit(int(1.25 * full))
        for dtype in (np.float64, np.complex128, np.complex64):
            data = (t.data.real if dtype is np.float64 else t.data).astype(dtype)
            u = st.SymmetricTensor(t.codomain, t.domain, data, NumpyBackend(dtype), sym)
            for p1, p2 in perms:
                permute(u, p1, p2)
                st.norm(u)
                st.vdot(u, u)
                assert st.INDEX_CACHE.bytes == _recount(st.INDEX_CACHE)
                assert st.INDEX_CACHE.bytes <= st.INDEX_CACHE.max_bytes
        assert st.INDEX_CACHE.stats["evictions"] > 0
    finally:
        st.INDEX_CACHE.set_limit(old)
    st.INDEX_CACHE.clear()
    st.norm(s)
    st.vdot(s, s)
    w, = [v for v in st.INDEX_CACHE.values() if isinstance(v, st._Weights)]
    assert w.nbytes() == 2 * w.w.nbytes                       # w, and its square root
    st.norm(_single(NumpyBackend(np.complex64), s))
    assert w.nbytes() == 2.5 * w.w.nbytes                     # and the root in float32
    assert st.INDEX_CACHE.bytes == _recount(st.INDEX_CACHE)


def test_partners_drop_their_contract_maps():
    """A fixed tensor contracted with 400 short-lived partners keeps neither their maps nor
    their keys."""
    sym = _sym2("SU2")
    P, W = Space(sym, (1,), (1,)), Space(sym, (0, 2), (2, 1))
    rng = np.random.default_rng(19)
    M = random((W, P), (P, W), NP, rng, sym=sym)
    fixed = permute(M, [0, 1, 3], [2]).layout          # the layout `contract` maps from
    old = (st.RECENT_LAYOUTS.max_bytes, st.RECENT_LAYOUTS.max_entries)
    try:
        st.RECENT_LAYOUTS.set_limit(old[0], 4)
        for _ in range(400):
            V = Space(sym, (0, 1, 2, 3), tuple(int(n) for n in rng.integers(1, 7, 4)))
            contract(M, random((V, P), (V,), NP, rng, sym=sym), [2], [1])
        gc.collect()
        live = {lay.uid for lay in sl.LAYOUTS.values()}
        keys = [k for k in st.INDEX_CACHE.keys() if k[0] == fixed.uid and k[2] == "contract"]
        assert len(keys) <= 8 and all(k[3] in live for k in keys)
        assert len(fixed.map_keys) <= 16
    finally:
        st.RECENT_LAYOUTS.set_limit(*old)


class _Props:
    total_memory = 8 * 2 ** 30


def _fake_cuda():
    torch = pytest.importorskip("torch")
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda d: _Props))
    return types.SimpleNamespace(is_torch=True, device=torch.device("cuda", 0), torch=fake)


def test_device_maps_have_their_own_limit(monkeypatch):
    """Maps on a CUDA device go to the device cache, under min(2 GiB, 10% of its memory) unless
    its limit was set; the host cache keeps its limit, even one equal to a default."""
    be = _fake_cuda()
    monkeypatch.setattr(sl, "DEVICE_INDEX_CACHE", ByteCache(2 ** 31))
    monkeypatch.setattr(sl, "INDEX_CACHE", ByteCache(2 ** 30, source="set"))
    sym = _sym2("SU2")
    A, B, _ = _spaces(sym, *_legs("SU2"))
    lay = zeros((A,), (B, A), NP).layout
    got = lay.map(be, ("probe",), lambda: np.zeros(4))
    assert lay.map(be, ("probe",), lambda: None) is got
    assert (sl.INDEX_CACHE.max_bytes, len(sl.INDEX_CACHE)) == (2 ** 30, 0)
    dev = sl.DEVICE_INDEX_CACHE
    key, = dev.keys()
    assert key[1] == ("SimpleNamespace", "cuda:0")
    assert (dev.max_bytes, dev.source) == (int(0.8 * 2 ** 30), "default-device")
    dev.set_limit(2 ** 31)
    sl.index_cache(be)
    assert (dev.max_bytes, dev.source) == (2 ** 31, "set")
    monkeypatch.setattr(sl, "DEVICE_INDEX_CACHE", env_cache("X", 2048))
    monkeypatch.setenv("TNKIT_SYMMETRIC_DEVICE_INDEX_CACHE_MB", "300")
    env = env_cache("TNKIT_SYMMETRIC_DEVICE_INDEX_CACHE_MB", 2048)
    monkeypatch.setattr(sl, "DEVICE_INDEX_CACHE", env)
    sl.index_cache(be)
    assert (env.max_bytes, env.source) == (300 * 2 ** 20, "env")


def test_cache_limits_from_the_environment():
    names = {"structure": "STRUCTURE", "indices": "INDEX", "device_indices": "DEVICE_INDEX",
             "recoupling": "RECOUPLING", "permutes": "PERMUTE", "dense_trees": "DENSE_TREE",
             "dense_pairs": "DENSE_PAIR"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("TNKIT_SYMMETRIC_")}
    env.update({f"TNKIT_SYMMETRIC_{v}_CACHE_MB": str(10 + i) for i, v in enumerate(names.values())})
    env.update(TNKIT_SYMMETRIC_RECENT_LAYOUTS_MB="5", TNKIT_SYMMETRIC_RECENT_LAYOUTS="7")
    code = ("import json; from tnkit import symmetric_tensor as st; i = st.cache_info(); "
            "print(json.dumps({k: [v['max_bytes'], v['limit_source'], v.get('max_entries')] "
            "for k, v in i.items() if 'limit_source' in v}))")
    out = json.loads(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                    text=True, check=True).stdout)
    for i, name in enumerate(names):
        assert out[name] == [(10 + i) * 2 ** 20, "env", None], name
    assert out["recent_layouts"] == [5 * 2 ** 20, "env", 7]
