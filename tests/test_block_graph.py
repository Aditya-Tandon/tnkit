"""Graph replay of block matvecs (`tnkit.block_graph`).

On the CPU a stand-in replaces the CUDA graph: capture records nothing useful (it poisons the input
and output buffers, as a real capture leaves them unwritten), and a replay recomputes the matvec
into the captured output buffer. The wrapper's own logic (layout keys, copy in, copy out, eager
fallback, what it keeps alive) is then tested exactly. The CUDA tests capture for real.
"""
import hashlib
import types
import weakref
from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tnkit import block_graph, blocks
from tnkit.api import dmrg
from tnkit.backend import get_backend
from tnkit.block_backend import BlockBackend, project_to_block_mps, to_block_mpo
from tnkit.blocks import BlockTensor
from tnkit.eigensolver import lanczos_expm
from tnkit.models_lgt2f import (compile2f, gaussian_profile, packet_opsum, schwinger2f_mpo,
                                site_charges_2f)
from tnkit.observables import apply_mpo, compress, to_numpy
from tnkit.symmetry import U1
from tnkit.tdvp import TDVPOptions
from tnkit.tdvp_evolver import TDVPEvolver

N, X, MU, DT, CHI = 6, 4.0, 0.3, 0.05, 24


def _held(v):
    """What a cache entry holds that others may share (written apart from `blocks._shared`)."""
    if isinstance(v, blocks._Index):
        yield v
    elif isinstance(v, tuple):
        for x in v:
            yield from _held(x)
    elif isinstance(v, blocks._Plan):
        yield from (x for x in (v.ia, v.ib) if x is not None)
        yield v.out
    elif isinstance(v, blocks._Gather) and v.ix is not None:
        yield v.ix


def _nbytes(x):
    if isinstance(x, blocks.Struct):
        off, st, shp = x.arrays
        return len(off.tobytes()) + len(st.tobytes()) + off.nbytes + st.nbytes + shp.nbytes
    return x.nbytes


def check_bytes():
    """Each shared object's `refs` equals its holders (cache entries and unreleased `Kept`
    records), and the counted bytes equal a recount of them once each plus gather offsets."""
    c = blocks._INDEX_CACHE
    holders = [list(_held(v)) for v in c._d.values()]
    holders += [[x for x in k.held.values() if isinstance(x, blocks._Index)] for k in c.kept]
    n = Counter(id(x) for h in holders for x in h)
    held = {id(x): x for h in holders for x in h}
    assert all(x.refs == n[i] for i, x in held.items())
    offs = sum(a.nbytes for v in c._d.values() if isinstance(v, blocks._Gather) for a in v.offs)
    assert c.bytes == sum(_nbytes(x) for x in held.values()) + offs
    return c.bytes


def fake_capture(fn, x, check=False):
    xs = x.data.clone()
    out = fn(BlockTensor(x.struct, xs))
    xs.fill_(float("nan"))
    out.data.fill_(float("nan"))

    def replay():
        if check:
            check_bytes()
        y = fn(BlockTensor(x.struct, xs))
        assert y.struct is out.struct
        out.data.copy_(y.data)
    return block_graph._Graph(types.SimpleNamespace(replay=replay), xs, out)


def checked_capture(fn, x):
    return fake_capture(fn, x, check=True)


def failing_capture(fn, x):
    raise RuntimeError("capture refused")


@pytest.fixture
def graphs(monkeypatch):
    """Graphs on, with `capture` standing in for CUDA on the CPU."""
    def use(capture):
        monkeypatch.setattr(block_graph, "wrap",
                            lambda fn, x: block_graph.MatvecGraph(fn, capture))
        block_graph.clear_stats()
    blocks.clear_cache()
    yield use
    blocks.clear_cache()
    block_graph.clear_stats()


@pytest.fixture(scope="module")
def start():
    W = schwinger2f_mpo(N, X, MU)
    _, vac, _ = dmrg(W, chi=32, n_sweeps=10, tol=1e-10)
    st = to_numpy(vac)
    for c, k, lo, hi in ((N / 4, 0.6, 0, N // 2 - 1), (3 * N / 4, -0.6, N // 2, N - 2)):
        op, _ = compile2f(packet_opsum(N, "u+d", gaussian_profile(N, c, k, 1.0, lo, hi)), N)
        st = apply_mpo(st, op)
    psi, _ = compress(st, 4096, svd_tol=1e-12)
    return [np.asarray(t) for t in to_numpy(psi)]


def evolve(psi, steps=2, method="12site"):
    be = get_backend("torch-cpu")
    qs = site_charges_2f(N)
    W = schwinger2f_mpo(N, X, MU, isospin_penalty=4.0 * 2 * np.sqrt(X))
    M, _ = project_to_block_mps(psi, qs, be, CHI, rtol=1e-10, sym=U1())
    blocks.clear_cache()
    opts = TDVPOptions(backend=BlockBackend(be), svd_tol=1e-8, chi_max=CHI, krylov_m=40,
                       graph="off", degenerate_tol=1e-4)
    ev = TDVPEvolver(M, to_block_mpo(W, qs, be, sym=U1())[0], DT, chi=CHI, options=opts)
    ev.run(steps, method=method)
    h = hashlib.sha256()
    for t in ev.mps:
        c = t.canonical()
        h.update(repr([(leg.charges, leg.dims, leg.sign) for leg in c.legs]).encode())
        h.update(np.ascontiguousarray(c.data.numpy()).tobytes())
    return h.hexdigest()


def test_off_by_default_and_never_on_the_cpu():
    x = blocks.random((blocks.Leg((0,), (2,)),), 0, get_backend("torch-cpu"),
                      np.random.default_rng(0))
    f = lambda v: v
    assert block_graph.wrap(f, x) is f
    was = block_graph.enabled()
    try:
        block_graph.set_enabled(True)
        assert block_graph.wrap(f, x) is f            # no CUDA tensor, no graph
    finally:
        block_graph.set_enabled(was)


@pytest.mark.parametrize("method", ["12site", "2site", "1site"])
def test_replayed_evolution_is_bitwise_identical(start, graphs, method):
    ref = evolve(start, method=method)
    graphs(fake_capture)
    assert evolve(start, method=method) == ref
    s = block_graph.info()
    assert s["captures"] > 0 and s["replays"] > 2 * s["captures"] and s["failed"] == 0
    assert s["replayed_gemms"] > 0


def test_failed_capture_runs_eagerly(start, graphs):
    ref = evolve(start)
    graphs(failing_capture)
    assert evolve(start) == ref
    s = block_graph.info()
    assert s["failed"] > 0 and s["captures"] == s["replays"] == 0


def lanczos_problem():
    be = BlockBackend(get_backend("torch-cpu"))
    rng = np.random.default_rng(1)
    q = lambda d: blocks.Leg((-1, 0, 1), d)
    a, s = q((2, 3, 2)), q((1, 2, 1))
    H = blocks.random((a, s, a.flip(), s.flip()), 0, be.dense, rng)
    H = H + H.conj().transpose((2, 3, 0, 1))
    x = blocks.random((a, s), 0, be.dense, rng)
    return be, (lambda v: be.einsum("abcd,cd->ab", H, v)), x, H


def test_capture_keeps_what_the_graph_reads(graphs):
    be, fn, x, H = lanczos_problem()
    mv = block_graph.MatvecGraph(fn, fake_capture)
    ref = [fn(x).data.clone() for _ in range(3)]
    out = [mv(x).data.clone() for _ in range(3)]
    g = mv._graphs[(x.struct, x.data.dtype)]
    p = blocks._plan(H.struct, x.struct, (2, 3), (0, 1))
    ixs = [i for i in (p.ia, p.ib) if i is not None]
    assert g.contractions == 1 and ixs
    assert {id(i) for i in ixs} | {id(H.data), id(g.x)} == set(g.keep.held)
    assert all(torch.equal(u, v) for u, v in zip(out, ref))
    check_bytes()
    limit = blocks.cache_info()["max_bytes"]
    try:
        blocks.set_cache_limit(1)                      # evicts all but one entry
        assert blocks.cache_info()["entries"] == 1
        assert all(i.refs >= 1 for i in ixs) and check_bytes() >= sum(i.nbytes for i in ixs)
        assert torch.equal(mv(x).data, ref[0])
    finally:
        blocks.set_cache_limit(limit)
    alive = [weakref.ref(d) for i in ixs for d in i._dev.values()]
    assert alive
    del p, ixs
    blocks.clear_cache()                               # releases the graph's record
    assert g.keep.released and not g.keep.held and blocks.cache_info()["bytes"] == 0
    assert all(r() is None for r in alive) and check_bytes() == 0
    s = dict(block_graph.info())
    assert torch.equal(mv(x).data, ref[0])             # eager again, then a new capture
    assert block_graph.info()["eager"] == s["eager"] + 1
    assert torch.equal(mv(x).data, ref[0]) and block_graph.info()["captures"] == s["captures"] + 1
    mv.close()
    assert not blocks._INDEX_CACHE.kept and check_bytes() > 0


def test_an_index_read_twice_is_held_once(graphs):
    be, fn, x, H = lanczos_problem()
    y = fn(x)
    p1 = blocks._plan(H.struct, x.struct, (2, 3), (0, 1))
    p2 = blocks._plan(H.struct, y.struct, (2, 3), (0, 1))
    assert p1.ia is p2.ia is not None                 # H's gather serves both contractions
    mv = block_graph.MatvecGraph(lambda v: fn(fn(v)), fake_capture)
    for _ in range(3):
        mv(x)
    assert mv._graphs[(x.struct, x.data.dtype)].contractions == 2
    assert p1.ia.refs == 1 + sum(p1.ia is i for v in blocks._INDEX_CACHE._d.values()
                                 for i in _held(v))
    check_bytes()
    mv.close()
    check_bytes()


def test_close_releases_and_runs_eagerly(graphs):
    be, fn, x, _ = lanczos_problem()

    def run(close):
        mv = block_graph.MatvecGraph(fn, fake_capture)
        ys = [mv(x).data.clone() for _ in range(3)]
        g = mv._graphs[(x.struct, x.data.dtype)]
        assert blocks._INDEX_CACHE.kept == {g.keep}
        before = check_bytes()
        close(mv)
        assert g.keep.released and not blocks._INDEX_CACHE.kept and check_bytes() == before
        n = block_graph.info()["eager"]
        assert torch.equal(mv(x).data, ys[0]) and block_graph.info()["eager"] == n + 1
        assert not mv._graphs
    run(lambda mv: mv.close())
    run(lambda mv: block_graph.clear())
    was = block_graph.enabled()
    try:
        block_graph.set_enabled(True)
        run(lambda mv: block_graph.set_enabled(False))
    finally:
        block_graph.set_enabled(was)


def test_solve_closes_its_wrapper(graphs):
    be, fn, x, _ = lanczos_problem()
    graphs(fake_capture)
    with block_graph.solve(fn, x) as mv:
        for _ in range(3):
            mv(x)
        assert len(blocks._INDEX_CACHE.kept) == 1
    assert mv.closed and not blocks._INDEX_CACHE.kept
    with pytest.raises(RuntimeError):
        with block_graph.solve(fn, x) as mv:
            mv(x), mv(x)
            raise RuntimeError
    assert mv.closed and not blocks._INDEX_CACHE.kept


def test_failed_capture_releases_its_record(graphs):
    be, fn, x, _ = lanczos_problem()

    def late_failure(fn, x):
        fn(x)
        assert len(next(iter(blocks._INDEX_CACHE.kept)).held) > 1
        raise RuntimeError("capture refused")
    block_graph.clear_stats()
    mv = block_graph.MatvecGraph(fn, late_failure)
    for _ in range(5):
        mv(x)
    s = block_graph.info()
    assert s["failed"] == 1 and s["eager"] == 5     # not retried
    assert not blocks._INDEX_CACHE.kept and check_bytes() > 0 and blocks._KEEP is None


def test_clear_cache_during_a_capture(graphs):
    """The record `clear_cache` released mid-capture holds nothing that is built after it."""
    be, fn, x, H = lanczos_problem()
    built = []

    def twice(v):
        y = fn(v)
        if blocks._KEEP is not None:
            blocks.clear_cache()
        p = blocks._plan(H.struct, y.struct, (2, 3), (0, 1))
        built.extend(i for i in (p.ia, p.ib) if i is not None)
        return fn(y)
    mv = block_graph.MatvecGraph(twice, fake_capture)
    mv(x), mv(x)
    g = mv._graphs[(x.struct, x.data.dtype)]
    assert built and g.keep.released and not g.keep.held
    mv.close()
    check_bytes()
    blocks.clear_cache()
    info = blocks.cache_info()
    assert info["bytes"] == 0 and info["entries"] == 0 and check_bytes() == 0
    assert all(i.refs == 0 for i in built)


def test_interrupted_capture_releases_its_record(graphs):
    be, fn, x, _ = lanczos_problem()
    calls = []

    def interrupted(v):
        y = fn(v)
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return y
    mv = block_graph.MatvecGraph(interrupted, fake_capture)
    mv(x)
    with pytest.raises(KeyboardInterrupt):
        mv(x)
    assert not blocks._INDEX_CACHE.kept and check_bytes() > 0 and blocks._KEEP is None
    assert mv._graphs[(x.struct, x.data.dtype)] is block_graph._EAGER
    n = block_graph.info()["eager"]
    assert torch.equal(mv(x).data, fn(x).data) and block_graph.info()["eager"] == n + 1
    assert not blocks._INDEX_CACHE.kept and block_graph.info()["captures"] == 0
    mv.close()


def test_close_during_its_own_capture(graphs):
    be, fn, x, _ = lanczos_problem()

    def closing(v):
        if blocks._KEEP is not None:
            mv.close()
        return fn(v)
    mv = block_graph.MatvecGraph(closing, fake_capture)
    ref = fn(x).data
    assert torch.equal(mv(x).data, ref) and torch.equal(mv(x).data, ref)
    assert mv.closed and not any(isinstance(g, block_graph._Graph) for g in mv._graphs.values())
    assert not blocks._INDEX_CACHE.kept and check_bytes() > 0
    block_graph.clear()
    assert not blocks._INDEX_CACHE.kept
    s = block_graph.info()
    assert torch.equal(mv(x).data, ref) and block_graph.info()["eager"] == s["eager"] + 1
    assert s["captures"] == s["replays"] == s["failed"] == 0


def test_env_switch_accepts_only_0_and_1(monkeypatch):
    for v, on in (("", False), ("0", False), ("1", True)):
        monkeypatch.setenv("TNKIT_BLOCK_GRAPH", v)
        assert block_graph._from_env() is on
    monkeypatch.delenv("TNKIT_BLOCK_GRAPH")
    assert block_graph._from_env() is False
    for v in ("true", "on", "2"):
        monkeypatch.setenv("TNKIT_BLOCK_GRAPH", v)
        with pytest.raises(ValueError, match="TNKIT_BLOCK_GRAPH"):
            block_graph._from_env()


@pytest.mark.parametrize("cap", [None, 1, 20000])
def test_replay_counts_what_it_keeps(start, graphs, cap):
    """Through whole evolutions, with the cache unbounded, at one entry and in between: counted
    bytes equal the recount at every replay (graphs alive) and after each solve (closed), and the
    result is that of the eager run."""
    ref = evolve(start)
    limit = blocks.cache_info()["max_bytes"]
    try:
        if cap is not None:
            blocks.set_cache_limit(cap)
        graphs(checked_capture)
        assert evolve(start) == ref
        assert not blocks._INDEX_CACHE.kept
        check_bytes()
        s = block_graph.info()
        assert s["replays"] > 2 * s["captures"] > 0
        if cap is not None:
            assert blocks.cache_info()["evictions"] > 0
    finally:
        blocks.set_cache_limit(limit)


def test_outputs_are_not_overwritten_by_later_replays(graphs):
    be, fn, x, _ = lanczos_problem()
    mv = block_graph.MatvecGraph(fn, fake_capture)
    xs = [x * (k + 1.0) for k in range(5)]
    ys = [mv(v) for v in xs]
    assert block_graph.info()["replays"] == 4
    assert all(torch.equal(y.data, fn(v).data) for y, v in zip(ys, xs))


def test_lanczos_through_the_wrapper(graphs):
    be, fn, x, _ = lanczos_problem()
    ref = lanczos_expm(fn, x, -1j * DT, be, m=12)
    mv = block_graph.MatvecGraph(fn, fake_capture)
    got = lanczos_expm(mv, x, -1j * DT, be, m=12)
    assert got.struct is ref.struct and torch.equal(got.data, ref.data)
    s = block_graph.info()
    assert s["captures"] >= 1 and s["replays"] >= 1


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def cuda(fn):
    return pytest.mark.cuda(needs_cuda(fn))


@cuda
def test_cuda_replay_is_bitwise_identical():
    be = BlockBackend(get_backend("torch-cuda"))
    rng = np.random.default_rng(1)
    q = lambda d: blocks.Leg((-1, 0, 1), d)
    a, s = q((5, 7, 4)), q((1, 2, 1))
    H = blocks.random((a, s, a.flip(), s.flip()), 0, be.dense, rng)
    H = H + H.conj().transpose((2, 3, 0, 1))
    x = blocks.random((a, s), 0, be.dense, rng)
    fn = lambda v: be.einsum("abcd,cd->ab", H, v)
    ref = lanczos_expm(fn, x, -1j * DT, be, m=12)
    block_graph.clear_stats()
    block_graph.set_enabled(True)
    try:
        got = lanczos_expm(block_graph.wrap(fn, x), x, -1j * DT, be, m=12)
    finally:
        block_graph.set_enabled(False)
    assert got.struct is ref.struct and torch.equal(got.data, ref.data)
    s = block_graph.info()
    assert s["captures"] == 1 and s["replays"] >= 1 and s["failed"] == 0


@cuda
def test_cuda_capture_of_an_unbuilt_layout_falls_back():
    be = BlockBackend(get_backend("torch-cuda"))
    rng = np.random.default_rng(2)
    a = blocks.Leg((-1, 0, 1), (3, 4, 3))
    H = blocks.random((a, a.flip()), 0, be.dense, rng)
    x = blocks.random((a,), 0, be.dense, rng)
    fn = lambda v: be.einsum("ab,b->a", H, v)
    mv = block_graph.MatvecGraph(fn)
    block_graph.clear_stats()
    ref = fn(x).data.clone()
    mv(x)
    blocks.clear_cache()                               # the capture would have to upload again
    assert torch.equal(mv(x).data, ref) and torch.equal(mv(x).data, ref)
    s = block_graph.info()
    assert s["failed"] == 1 and s["captures"] == 0 and s["eager"] == 3
