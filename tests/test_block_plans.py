"""Two-level block plans: gather indices from runs, and which cache level a change of sizes or
labels hits."""
import copy
import os
import pickle
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from tnkit import blocks as B
from tnkit.backend import get_backend
from tnkit.mps import MPS
from tnkit.symmetry import ProductSymmetry, U1, Zn

NP = get_backend("numpy")


def _explicit(src0, shp, ss, dst0, ds):
    """idx[dst] = src for every element of every box, element by element."""
    n = int(sum(np.prod(s) for s in shp))
    idx = np.full(n, -1)
    for b in range(len(shp)):
        for i in np.ndindex(*shp[b]):
            idx[dst0[b] + np.dot(i, ds[b])] = src0[b] + np.dot(i, ss[b])
    return idx


@pytest.mark.parametrize("seed", range(20))
def test_runs_match_explicit_positions(seed):
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 5))
    shp = rng.integers(1, 5, size=(int(rng.integers(1, 6)), m))
    ss = np.stack([np.array(B._row_major(s))[rng.permutation(m)] for s in shp])  # transposed reads
    src0 = rng.integers(0, 50, size=len(shp)) * 100
    sizes = shp.prod(1)
    dst0 = np.cumsum(sizes) - sizes                       # boxes tile the destination
    ds = B._row_major_rows(shp)
    ix = B._Index.from_runs(B._runs(src0, shp, ss, dst0, ds), 10_000, int(sizes.sum()))
    want = _explicit(src0, shp, ss, dst0, ds)
    assert np.array_equal(ix.host, want)
    assert ix.is_identity(len(want)) == np.array_equal(want, np.arange(len(want)))
    assert ix._runs is None and ix.nbytes == ix.host.nbytes          # runs dropped once expanded
    torch = pytest.importorskip("torch")
    assert np.array_equal(ix.on(torch.zeros(1)).numpy(), want)        # copied from the host
    ix2 = B._Index.from_runs(B._runs(src0, shp, ss, dst0, ds), 10_000, int(sizes.sum()))
    assert np.array_equal(ix2.on(torch.zeros(1)).numpy(), want)
    assert ix2._runs is None and np.array_equal(ix2.host, want)       # copied from the device
    ix3 = B._Index.from_runs(B._runs(src0, shp, ss, dst0, ds), 10_000, int(sizes.sum()))
    assert np.array_equal(ix3._expand(torch, torch.device("cpu")).numpy(), want)


@pytest.mark.parametrize("mode", ["auto", "host", "device"])
@pytest.mark.parametrize("host_max", [0, 1 << 17])
def test_expansion_modes_give_the_same_index(monkeypatch, mode, host_max):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TNKIT_BLOCK_EXPAND", mode)
    monkeypatch.setattr(B, "_HOST_EXPAND_MAX", host_max)
    rng = np.random.default_rng(7)
    shp = rng.integers(1, 6, size=(9, 3))
    ss = np.stack([np.array(B._row_major(s))[rng.permutation(3)] for s in shp])
    sizes = shp.prod(1)
    args = (rng.integers(0, 40, size=9) * 200, shp, ss, np.cumsum(sizes) - sizes,
            B._row_major_rows(shp))
    want = _explicit(*args)
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for dev in devices:
        ix = B._Index.from_runs(B._runs(*args), 10_000, int(sizes.sum()))
        got = ix.on(torch.zeros(1, device=dev))
        assert got.device.type == dev.type and got.dtype == torch.int32
        assert np.array_equal(got.cpu().numpy(), want)
        assert ix._host is None or mode == "host"     # auto keeps no host copy
        assert ix._runs is None


def test_identity_needs_equal_steps():
    # two runs that start where they are written but read with step 2: positions 0, 2, 2, 4
    one = np.ones(2, np.int64)
    ix = B._Index.from_runs((np.array([0, 2]), 2 * one, np.array([0, 2]), one, 2 * one), 6, 4)
    assert not ix.is_identity(4) and ix.host.tolist() == [0, 2, 2, 4]


def test_size_change_reuses_the_symbolic_plan():
    rng = np.random.default_rng(0)
    B.clear_cache()
    p = B.Leg((0, 1), (1, 1))
    for dims in ((3, 4, 2), (5, 4, 2), (5, 4, 7)):          # sizes change, labels do not
        bond = B.Leg((-1, 0, 1), dims)
        a = B.random((bond, p, bond.flip()), 0, NP, rng)
        B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
    info = B.cache_info()
    assert info["plan_misses"] == 3 and info["label_misses"] == 1 and info["label_hits"] == 2
    bond = B.Leg((-1, 0, 2), (3, 4, 2))                      # a label changes
    a = B.random((bond, p, bond.flip()), 0, NP, rng)
    B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
    assert B.cache_info()["label_misses"] == 2


def test_operand_gather_is_reused_when_only_the_other_operand_changes():
    rng = np.random.default_rng(1)
    B.clear_cache()
    p = B.Leg((0, 1), (1, 1))
    left = B.Leg((-1, 0, 1), (3, 4, 2))
    a = B.random((left, p, left.flip()), 0, NP, rng)
    outs = []
    for dims in ((2, 2, 2), (3, 2, 2)):
        right = B.Leg((-1, 0, 1), dims)
        b = B.random((left, p.flip(), right.flip()), 0, NP, rng)
        outs.append(B.tensordot(a, b, ((2,), (0,))))
        assert np.allclose(outs[-1].to_dense(),
                           np.tensordot(a.to_dense(), b.to_dense(), (2, 0)), atol=1e-13)
    gathers = [k for k in B._INDEX_CACHE._d if k[0] == "_gather"]
    assert len([k for k in gathers if k[3] == 0]) == 1      # a's gather built once
    assert len([k for k in gathers if k[3] == 1]) == 2


def test_equal_layouts_are_one_object():
    rng = np.random.default_rng(2)
    p = B.Leg((0, 1), (1, 1))
    bond = B.Leg((-1, 0, 1), (3, 4, 2))
    a = B.random((bond, p, bond.flip()), 0, NP, rng)
    assert a.transpose((2, 0, 1)).transpose((1, 2, 0)).struct is a.struct
    out = B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
    B.clear_cache()                                          # the plan is rebuilt ...
    again = B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
    assert again.struct is out.struct                        # ... with the same output layout
    # the same labels and sector sizes with column-major blocks: another layout
    s = a.struct
    off, st, shp = s.arrays
    col = B._struct(s.legs, s.charge, s.blocks, off, B._row_major_rows(shp[:, ::-1])[:, ::-1],
                    shp, s.size, s.sym)
    assert col is not s and col.labels == s.labels
    n = shp.prod(1)                                       # blocks stored in reverse order
    rev = B._struct(s.legs, s.charge, s.blocks, s.size - np.cumsum(n), st, shp, s.size, s.sym)
    assert rev is not s and rev is not col
    t = B.BlockTensor(col, a.data)                        # the same numbers, read differently
    assert not np.array_equal(t.to_dense(), a.to_dense())
    assert np.allclose(B.tensordot(t, t.conj(), ((0, 1), (0, 1))).to_dense(),
                       np.tensordot(t.to_dense(), t.to_dense().conj(), ((0, 1), (0, 1))))


def test_plans_sharing_an_output_layout_count_it_once():
    rng = np.random.default_rng(8)
    p, x = B.Leg((0, 1), (1, 1)), B.Leg((-1, 0, 1), (3, 4, 2))
    a = B.random((x, p, x.flip()), 0, NP, rng)
    at, c = a.transpose((1, 0, 2)), a.conj()
    info = B.cache_info()
    try:
        B.clear_cache()
        B.tensordot(a, c, ((0, 1), (0, 1)))
        one = B.cache_info()["bytes"]
        B.tensordot(at, c, ((1, 0), (0, 1)))                # another plan, the same output layout
        p1 = B._plan(a.struct, c.struct, (0, 1), (0, 1))
        p2 = B._plan(at.struct, c.struct, (1, 0), (0, 1))
        out = p1.out
        assert p1 is not p2 and p2.out is out and out.refs == 2
        assert out.nbytes > 0 and B.cache_info()["bytes"] < 2 * one
        B.set_cache_limit(1)                                 # evict all but the newest entry, p2
        assert B.cache_info()["entries"] == 1 and out.refs == 1
        assert B.cache_info()["bytes"] == out.nbytes + sum(i.nbytes for i in (p2.ia, p2.ib) if i)
        B.clear_cache()
        assert out.refs == 0 and B.cache_info()["bytes"] == 0
    finally:
        B.set_cache_limit(info["max_bytes"])
        B.clear_cache()


def test_counters():
    rng = np.random.default_rng(3)
    B.clear_cache()
    p = B.Leg((0, 1), (1, 1))
    bond = B.Leg((-1, 0, 1), (3, 4, 2))
    a = B.random((bond, p, bond.flip()), 0, NP, rng)
    for _ in range(2):
        B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
    info = B.cache_info()
    plan = B._plan(a.struct, a.conj().struct, (0, 1), (0, 1))
    assert info["contractions"] == 2 and info["gemms"] == 2 * sum(g[4] > 0 for g in plan.gemms) > 0
    assert info["build_s"] > 0 and info["build_cpu_s"] > 0
    # the output block (1, 1) is allowed but y has no charge 1: a zero-filled region, not a GEMM
    x, y, z = B.Leg((0, 1), (2, 3)), B.Leg((0,), (4,)), B.Leg((0, 1), (2, 5))
    a, b = B.random((x, y.flip()), 0, NP, rng), B.random((y, z.flip()), 0, NP, rng)
    B.clear_cache()
    c = B.tensordot(a, b, ((1,), (0,)))
    assert len(B._plan(a.struct, b.struct, (1,), (0,)).gemms) == 2
    assert B.cache_info()["gemms"] == 1
    np.testing.assert_allclose(c.to_dense(), np.tensordot(a.to_dense(), b.to_dense(), (1, 0)), rtol=1e-13, atol=1e-13)


def test_build_seconds_count_nested_builds_once():
    """A plan's build runs its operands' gather builds; `build_s` must not count them twice."""
    rng = np.random.default_rng(2)
    B.clear_cache()
    leg = B.Leg(tuple(range(-4, 5)), (3,) * 9)
    a = B.random((leg, leg, leg.flip()), 0, NP, rng)
    t0 = time.perf_counter()
    B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
    assert 0 < B.cache_info()["build_s"] <= time.perf_counter() - t0


def test_self_contraction():
    """One struct as both operands: each side needs its own gather."""
    rng = np.random.default_rng(3)
    x, p = B.Leg((-1, 0, 1), (2, 3, 2)), B.Leg((0, 1), (1, 1))
    a = B.random((x, p, x.flip()), 0, NP, rng)
    B.clear_cache()
    np.testing.assert_allclose(B.tensordot(a, a, ((2,), (0,))).to_dense(),
                               np.tensordot(a.to_dense(), a.to_dense(), (2, 0)), atol=1e-13)


def test_same_structs_other_axes():
    """The same pair of structs contracted over another leg of b needs another symbolic plan."""
    rng = np.random.default_rng(4)
    x, y = B.Leg((0, 1), (2, 3)), B.Leg((0, 1, 2), (1, 2, 2))
    a = B.random((y, x), 1, NP, rng)
    b = B.random((x.flip(), x.flip()), -1, NP, rng)
    B.clear_cache()
    for j in (0, 1):
        np.testing.assert_allclose(B.tensordot(a, b, ((1,), (j,))).to_dense(),
                                   np.tensordot(a.to_dense(), b.to_dense(), (1, j)), atol=1e-13)


def test_symmetries_must_match():
    """A U(1) leg does not contract with a Z_2 leg of the same labels and dims."""
    rng = np.random.default_rng(5)
    u, z = B.Leg((0, 1), (2, 2)), B.Leg((0, 1), (2, 2), 1, Zn(2))
    a = B.random((u, u.flip()), 0, NP, rng)
    b = B.random((z, z.flip()), 0, NP, rng)
    with pytest.raises(ValueError, match="cannot contract"):
        B.tensordot(a, b, ((1,), (0,)))


def test_label_cache_limit_and_gather_keys():
    """The symbolic-plan cache has its own limit, and gather entries do not keep its plans alive."""
    rng = np.random.default_rng(6)
    info = B.cache_info()
    p = B.Leg((0, 1), (1, 1))
    try:
        B.clear_cache()
        B.set_cache_limit(info["max_bytes"], 1)
        for labels in ((-1, 0, 1), (0, 1, 2)):
            bond = B.Leg(labels, (2, 3, 2))
            a = B.random((bond, p, bond.flip()), 0, NP, rng)
            B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
        now = B.cache_info()
        assert now["label_max_bytes"] == 1 and now["label_entries"] == 1
        assert now["label_evictions"] == 1
        assert not any(isinstance(x, B._SymPlan) for k in B._INDEX_CACHE._d for x in k)
    finally:
        B.set_cache_limit(info["max_bytes"], info["label_max_bytes"])
        B.clear_cache()


def test_expand_mode_is_read_at_expansion(monkeypatch):
    torch = pytest.importorskip("torch")
    two = 2 * np.ones(2, np.int64)
    for mode, on_host in (("device", False), ("host", True)):
        monkeypatch.setenv("TNKIT_BLOCK_EXPAND", mode)
        ix = B._Index.from_runs((np.array([0, 2]), two // 2, np.array([0, 2]), two // 2, two), 4, 4)
        assert ix.on(torch.zeros(1)).tolist() == [0, 1, 2, 3]
        assert (ix._host is not None) == on_host


@pytest.mark.parametrize("mode", ["bogus", "Host"])
def test_unknown_expand_mode_raises(monkeypatch, mode):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TNKIT_BLOCK_EXPAND", mode)
    one = np.ones(2, np.int64)
    ix = B._Index.from_runs((np.array([0, 2]), one, np.array([0, 2]), one, 2 * one), 4, 4)
    with pytest.raises(ValueError, match="TNKIT_BLOCK_EXPAND"):
        ix.on(torch.zeros(1))


def test_refused_contraction_is_not_counted(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("TNKIT_BLOCK_EXPAND", "bogus")
    p = B.Leg((0, 1), (2, 3))
    a = B.random((p, p.flip()), 0, get_backend("torch-cpu"), np.random.default_rng(9))
    B.clear_cache()
    with pytest.raises(ValueError, match="TNKIT_BLOCK_EXPAND"):
        B.tensordot(a, a, ((1,), (0,)))
    info = B.cache_info()
    assert info["contractions"] == info["gemms"] == 0


def test_only_small_uploads_are_pinned(monkeypatch):
    torch = pytest.importorskip("torch")
    pinned, copied = [], []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda t: pinned.append(t.numel()) or t)
    monkeypatch.setattr(torch.Tensor, "to", lambda t, *a, **k: copied.append(k) or t)
    n = B._PIN_MAX_BYTES // 8
    for m in (n, n + 1):
        B._upload(np.zeros(m, np.int64), types.SimpleNamespace(type="cuda"))
    assert pinned == [n] and copied == [{"non_blocking": True}, {}]


def test_pickled_layouts_are_interned():
    """Pickling or copying a block tensor or a block MPS keeps its layout, and its cached plans."""
    rng = np.random.default_rng(8)
    p, x = B.Leg((0, 1), (1, 1)), B.Leg((-1, 0, 1), (3, 4, 2))
    a = B.random((x, p, x.flip()), 0, NP, rng)
    B.clear_cache()
    out = B.tensordot(a, a.conj(), ((0, 1), (0, 1))).struct
    refs = (a.struct.refs, out.refs)
    for s in (a.struct, out):
        assert pickle.loads(pickle.dumps(s)) is s and copy.deepcopy(s) is s
    assert (a.struct.refs, out.refs) == refs == (0, 1)
    m = pickle.loads(pickle.dumps(MPS([a, a])))
    assert all(t.struct is a.struct for t in m) and copy.deepcopy(a).struct is a.struct
    misses = B.cache_info()["plan_misses"]
    B.tensordot(m[1], m[0].conj(), ((0, 1), (0, 1)))
    assert B.cache_info()["plan_misses"] == misses
    z = B.Leg((0, 1, 2), (2, 3, 1), 1, Zn(3))
    uz = B.Leg(((0, 0), (1, 1), (2, 0)), (2, 3, 1), 1, ProductSymmetry((U1(), Zn(2))))
    for y, charge in ((z, 1), (uz, (1, 1))):
        t = B.random((y, y, y.flip()), charge, NP, rng)
        assert t.struct.charge != t.struct.sym.unit()
        assert pickle.loads(pickle.dumps(t)).struct is t.struct


CUDA = types.SimpleNamespace(type="cuda", index=0)
GiB = 2 ** 30


@pytest.fixture
def default_limit(monkeypatch):
    """The plan cache as at import with no variable set, its state restored afterwards."""
    monkeypatch.setattr(B._INDEX_CACHE, "max_bytes", 4 * GiB)
    monkeypatch.setattr(B._INDEX_CACHE, "source", "default-host")
    return monkeypatch


@pytest.mark.parametrize("memory, mib", [(12 * GiB, 3686), (80 * GiB, 4096)])
def test_default_limit_resolves_on_first_cuda_device(default_limit, memory, mib):
    default_limit.setattr(B, "_device_memory", lambda device: memory)
    B._INDEX_CACHE.on_device(types.SimpleNamespace(type="cpu"))
    assert B.cache_info()["limit_source"] == "default-host"
    assert B.cache_info()["max_bytes"] == 4 * GiB
    B._INDEX_CACHE.on_device(CUDA)
    info = B.cache_info()
    assert info["max_bytes"] == mib * 2 ** 20 and info["limit_source"] == "default-device"
    default_limit.setattr(B, "_device_memory", lambda device: 1 * GiB)
    B._INDEX_CACHE.on_device(CUDA)                       # resolved once
    assert B.cache_info()["max_bytes"] == mib * 2 ** 20


def test_set_cache_limit_wins(default_limit):
    default_limit.setattr(B, "_device_memory", lambda device: 12 * GiB)
    B.set_cache_limit(5 * GiB)
    B._INDEX_CACHE.on_device(CUDA)
    info = B.cache_info()
    assert info["max_bytes"] == 5 * GiB and info["limit_source"] == "set"


@pytest.mark.parametrize("value, nbytes, source", [("64", 64 * 2 ** 20, "env"),
                                                   (None, 4 * GiB, "default-host")])
def test_env_limit_wins(monkeypatch, value, nbytes, source):
    if value is None:
        monkeypatch.delenv("TNKIT_BLOCK_CACHE_MB", raising=False)
    else:
        monkeypatch.setenv("TNKIT_BLOCK_CACHE_MB", value)
    monkeypatch.setattr(B, "_device_memory", lambda device: 12 * GiB)
    c = B._env_cache("TNKIT_BLOCK_CACHE_MB", 4096)
    assert (c.max_bytes, c.source) == (nbytes, source)
    c.on_device(CUDA)
    assert c.source == ("env" if value else "default-device")


def test_import_does_not_touch_cuda():
    pytest.importorskip("torch")
    code = ("import tnkit, torch; from tnkit import blocks; i = blocks.cache_info(); "
            "assert not torch.cuda.is_initialized(); print(i['max_bytes'], i['limit_source'])")
    env = {k: v for k, v in os.environ.items() if k != "TNKIT_BLOCK_CACHE_MB"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == [str(4 * GiB), "default-host"]


@pytest.mark.cuda
def test_default_limit_on_cuda(default_limit):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    B.clear_cache()
    bond, p = B.Leg((-1, 0, 1), (2, 3, 2)), B.Leg((0, 1), (1, 1))
    a = B.random((bond, p, bond.flip()), 0, get_backend("torch-cuda"), np.random.default_rng(0))
    B.tensordot(a, a.conj(), ((0, 1), (0, 1)))
    total = torch.cuda.get_device_properties(a.data.device).total_memory
    info = B.cache_info()
    assert info["limit_source"] == "default-device"
    assert info["max_bytes"] == min(4096, int(0.3 * total / 2 ** 20)) * 2 ** 20
    B.clear_cache()
