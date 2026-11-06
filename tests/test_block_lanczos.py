"""The Lanczos exponential on block tensors with less host work (`tnkit.block_lanczos`): the eager
solver's result and diagnostics, bit for bit, alone and under graph replay of the matvecs."""
import hashlib
import os
import types

import numpy as np
import pytest

from tnkit import block_graph, block_lanczos, blocks
from tnkit.api import dmrg
from tnkit.backend import get_backend
from tnkit.block_backend import BlockBackend, project_to_block_mps, to_block_mpo
from tnkit.blocks import BlockTensor
from tnkit.eigensolver import lanczos_expm
from tnkit.models_lgt2f import (compile2f, gaussian_profile, packet_opsum, schwinger2f_mpo,
                                site_charges_2f)
from tnkit.observables import apply_mpo, compress, to_numpy
from tnkit.symmetry import U1
from tnkit.tdvp import TDVPOptions, tdvp2_evolve
from tnkit.tdvp_evolver import TDVPEvolver

N, X, MU, DT, CHI = 6, 4.0, 0.3, 0.05, 24
try:
    import torch  # noqa: F401
    BACKENDS = ["numpy", "torch-cpu"]
except ImportError:
    BACKENDS = ["numpy"]


@pytest.fixture(autouse=True)
def fast():
    """Off for the reference; ``fast(True)`` switches it on for the rest of the test."""
    was = block_lanczos.enabled()
    block_lanczos.set_enabled(False)
    yield block_lanczos.set_enabled
    block_lanczos.set_enabled(was)


def host(be, x):
    return np.ascontiguousarray(be.to_numpy(x)).tobytes()


def problem(backend, seed=1):
    be = BlockBackend(get_backend(backend))
    rng = np.random.default_rng(seed)
    q = lambda d: blocks.Leg((-1, 0, 1), d)
    a, s = q((2, 3, 2)), q((1, 2, 1))
    H = blocks.random((a, s, a.flip(), s.flip()), 0, be.dense, rng)
    H = H + H.conj().transpose((2, 3, 0, 1))
    x = blocks.random((a, s), 0, be.dense, rng)
    return be, H, x


def other_layout(y):
    """`y` with legs and values unchanged, in a layout that is not the canonical one."""
    return y.transpose((1, 0)).canonical().transpose((1, 0))


@pytest.mark.skipif(os.environ.get("TNKIT_BLOCK_LANCZOS_FAST") == "0", reason="switched off")
def test_on_by_default():
    import importlib
    assert importlib.reload(block_lanczos).enabled()


def test_env_switch_accepts_only_0_and_1(monkeypatch):
    for v, on in (("", True), ("0", False), ("1", True)):
        monkeypatch.setenv("TNKIT_BLOCK_LANCZOS_FAST", v)
        assert block_lanczos._from_env() is on
    monkeypatch.delenv("TNKIT_BLOCK_LANCZOS_FAST")
    assert block_lanczos._from_env() is True
    for v in ("true", "on", "2"):
        monkeypatch.setenv("TNKIT_BLOCK_LANCZOS_FAST", v)
        with pytest.raises(ValueError, match="TNKIT_BLOCK_LANCZOS_FAST"):
            block_lanczos._from_env()


def test_import_refuses_a_bad_switch(monkeypatch):
    import importlib
    monkeypatch.setenv("TNKIT_BLOCK_LANCZOS_FAST", "yes")
    try:
        with pytest.raises(ValueError, match="TNKIT_BLOCK_LANCZOS_FAST"):
            importlib.reload(block_lanczos)
    finally:
        monkeypatch.delenv("TNKIT_BLOCK_LANCZOS_FAST")
        importlib.reload(block_lanczos)


@pytest.mark.parametrize("backend", BACKENDS)
def test_eigensolver_hands_block_vectors_over(backend, fast, monkeypatch):
    be, H, x = problem(backend)
    calls = []
    monkeypatch.setattr(block_lanczos, "lanczos_expm", lambda *a, **k: calls.append(a) or a[1])
    fn = lambda v: be.einsum("abcd,cd->ab", H, v)
    lanczos_expm(fn, x, -1j * DT, be)
    assert not calls
    fast(True)
    assert lanczos_expm(fn, x, -1j * DT, be) is x and len(calls) == 1
    dense = get_backend(backend)
    y = dense.array(np.ones(4, dtype=complex))
    lanczos_expm(lambda v: v, y, -1j * DT, dense)
    assert len(calls) == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("layout", ["canonical", "x-other", "out-other"])
@pytest.mark.parametrize("kw", [{}, {"m": 5, "adaptive": False}, {"m": 4}, {"tol": 1e-14}])
def test_solve_is_bitwise_the_eager_one(backend, layout, kw):
    be, H, x = problem(backend)
    if layout == "x-other":
        x = other_layout(x)
    out = other_layout if layout == "out-other" else (lambda y: y)
    fn = lambda v: out(be.einsum("abcd,cd->ab", H, v))
    i0, i1 = {}, {}
    ref = lanczos_expm(fn, x, -1j * DT, be, info=i0, **kw)
    got = block_lanczos.lanczos_expm(fn, x, -1j * DT, be, info=i1, **kw)
    assert got.struct is ref.struct and host(be.dense, got.data) == host(be.dense, ref.data)
    assert repr(i1) == repr(i0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_breakdown_and_zero_vector(backend):
    be, _, x = problem(backend)
    i0, i1 = {}, {}
    ref = lanczos_expm(lambda v: v * 2.0, x, -1j * DT, be, info=i0)
    got = block_lanczos.lanczos_expm(lambda v: v * 2.0, x, -1j * DT, be, info=i1)
    assert host(be.dense, got.data) == host(be.dense, ref.data) and repr(i1) == repr(i0)
    assert i1["krylov_dim"] == 1
    z = x * 0.0
    assert block_lanczos.lanczos_expm(lambda v: v, z, -1j * DT, be) is z


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("on", [False, True])
def test_diagnostics_match_a_dense_lanczos(backend, on, fast):
    """`info`, shared by both paths, against a dense Lanczos written out here; input norm 3."""
    from scipy.linalg import expm
    be, H, x = problem(backend)
    x = x * (3.0 / float(np.real(be.to_scalar(be.norm(x)))))
    fast(on)
    info = {}
    tau = -1j * DT
    lanczos_expm(lambda v: be.einsum("abcd,cd->ab", H, v), x, tau, be, info=info, est_ref_j=3)
    xd = np.asarray(be.dense.to_numpy(x.to_dense())).ravel()
    Hd = np.asarray(be.dense.to_numpy(H.to_dense())).reshape(xd.size, xd.size)
    V, alphas, betas, beta, prev = [xd / np.linalg.norm(xd)], [], [], 0.0, 0.0
    for _ in range(3):
        w = Hd @ V[-1] - beta * prev
        alphas.append(np.vdot(V[-1], w).real)
        w = w - alphas[-1] * V[-1]
        for u in V:
            w = w - np.vdot(u, w) * u
        beta, prev = np.linalg.norm(w), V[-1]
        betas.append(beta)
        V.append(w / beta)
    T = np.diag(alphas) + np.diag(betas[:2], 1) + np.diag(betas[:2], -1)
    est = betas[2] * abs(tau) * abs(expm(tau * T)[2, 0]) * 3.0
    assert info["beta1"] == pytest.approx(betas[0], rel=1e-10)
    assert info["est_ref_j"] == 3 and info["est_ref"] == pytest.approx(est, rel=1e-8)
    assert info["capped"] is False and info["krylov_dim"] >= 3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("on", [False, True])
def test_diagnostics_at_the_edges(backend, on, fast):
    """Absolute `info` at breakdown, at the cap and with a frozen-off estimate: both paths share
    the bookkeeping, so comparing them cannot catch a bug in it."""
    be, H, x = problem(backend)
    fast(on)
    fn = lambda v: be.einsum("abcd,cd->ab", H, v)
    i = {}
    lanczos_expm(lambda v: v * 2.0, x, -1j * DT, be, info=i)             # breakdown at k = 0
    assert (i["krylov_dim"], i["est_ref"], i["est_ref_j"], i["capped"]) == (1, 0.0, 1, False)
    assert i["err_est"] == float("inf") and 0 <= i["beta1"] < 1e-12
    i = {}
    lanczos_expm(fn, x, -1j * DT, be, m=2, info=i)                        # the cap decides
    assert i["capped"] is True and i["krylov_dim"] == 2
    i = {}
    lanczos_expm(fn, x, -1j * DT, be, m=3, adaptive=False, est_ref_j=5, info=i)
    assert i["err_est"] == float("inf") and i["est_ref_j"] == 0 and i["capped"] is False


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


def evolve(psi, backend, method, steps=2):
    """Hash of the final state and the step logs, timings excepted."""
    be = get_backend(backend)
    qs = site_charges_2f(N)
    W = schwinger2f_mpo(N, X, MU, isospin_penalty=4.0 * 2 * np.sqrt(X))
    M, _ = project_to_block_mps(psi, qs, be, CHI, rtol=1e-10, sym=U1())
    Wb = to_block_mpo(W, qs, be, sym=U1())[0]
    blocks.clear_cache()
    logs = []
    opts = TDVPOptions(backend=BlockBackend(be), svd_tol=1e-8, chi_max=CHI, krylov_m=40,
                       graph="off", degenerate_tol=1e-4,
                       step_callback=lambda r, o: logs.append(
                           {k: v for k, v in r.items() if not k.endswith("_s")}))
    if method == "tdvp2":
        out, _ = tdvp2_evolve(M, Wb, CHI, DT, steps, options=opts)
    else:
        ev = TDVPEvolver(M, Wb, DT, chi=CHI, options=opts)
        ev.run(steps, method=method)
        out = ev.mps
    h = hashlib.sha256(repr(logs).encode())
    for t in out:
        c = t.canonical()
        h.update(repr([(leg.charges, leg.dims, leg.sign) for leg in c.legs]).encode())
        h.update(host(be, c.data))
    return h.hexdigest()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["12site", "2site", "1site", "tdvp2"])
def test_evolution_is_bitwise_identical(start, fast, backend, method):
    ref = evolve(start, backend, method)
    fast(True)
    assert evolve(start, backend, method) == ref


def fake_capture(fn, x):
    """A CPU stand-in for a CUDA graph: capture poisons the buffers, a replay recomputes the matvec
    into the captured output."""
    xs = x.data.clone()
    out = fn(BlockTensor(x.struct, xs))
    xs.fill_(float("nan"))
    out.data.fill_(float("nan"))

    def replay():
        y = fn(BlockTensor(x.struct, xs))
        assert y.struct is out.struct
        out.data.copy_(y.data)
    return block_graph._Graph(types.SimpleNamespace(replay=replay), xs, out)


@pytest.fixture
def graphs(monkeypatch):
    """``graphs()`` turns graphs on, through `block_graph.solve`, with `fake_capture` standing in
    for CUDA."""
    def use():
        monkeypatch.setattr(block_graph, "wrap",
                            lambda fn, x: block_graph.MatvecGraph(fn, fake_capture))
        block_graph.clear_stats()
    blocks.clear_cache()
    yield use
    blocks.clear_cache()
    block_graph.clear_stats()


def counted(monkeypatch):
    """Counts the solves this path runs."""
    n, orig = [0], block_lanczos.lanczos_expm

    def run(*a, **k):
        n[0] += 1
        return orig(*a, **k)
    monkeypatch.setattr(block_lanczos, "lanczos_expm", run)
    return n


def test_solve_composes_with_graph_replay(fast, graphs, monkeypatch):
    pytest.importorskip("torch")
    be, H, x = problem("torch-cpu")
    fn = lambda v: be.einsum("abcd,cd->ab", H, v)
    i0, i1 = {}, {}
    ref = lanczos_expm(fn, x, -1j * DT, be, info=i0)
    graphs()
    fast(True)
    n = counted(monkeypatch)
    with block_graph.solve(fn, x) as f:
        assert isinstance(f, block_graph.MatvecGraph)
        got = lanczos_expm(f, x, -1j * DT, be, info=i1)
        assert blocks._INDEX_CACHE.kept
    assert not blocks._INDEX_CACHE.kept
    assert got.struct is ref.struct and host(be.dense, got.data) == host(be.dense, ref.data)
    assert repr(i1) == repr(i0)
    s = block_graph.info()
    assert s["captures"] >= 1 and s["replays"] >= 1 and s["failed"] == 0 and n[0] == 1


@pytest.mark.parametrize("cap", [None, 20000])
def test_evolution_composes_with_graph_replay(start, fast, graphs, cap, monkeypatch):
    """Graphs and this path together give the eager result, with the plan cache unbounded and
    small enough to evict while graphs are live."""
    pytest.importorskip("torch")
    ref = evolve(start, "torch-cpu", "12site")
    limit = blocks.cache_info()["max_bytes"]
    try:
        if cap is not None:
            blocks.set_cache_limit(cap)
        fast(True)
        graphs()
        n = counted(monkeypatch)
        assert evolve(start, "torch-cpu", "12site") == ref
        assert not blocks._INDEX_CACHE.kept
        s = block_graph.info()
        assert s["captures"] > 0 and s["replays"] > 2 * s["captures"] and s["failed"] == 0
        assert n[0] == s["solves"] > 0
        if cap is not None:
            assert blocks.cache_info()["evictions"] > 0
    finally:
        blocks.set_cache_limit(limit)
