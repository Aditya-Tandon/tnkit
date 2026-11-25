"""Recoupling maps. `_positions` against the digit-by-digit division it replaced, on numpy and
on every torch device present, with positions above 2^24 (where float32 is no longer exact) and
coefficients that took the earlier wrap-around form out of int32. The maps of every permute of
a set of layouts against digests of the maps built by the earlier code. Maps built by the
device path (forced on the CPU, and on any GPU present) against those of the host path, and
permute and contract results with them, bit for bit."""
import hashlib
import itertools

import numpy as np
import pytest

from tnkit import _recoupling as rc
from tnkit import _symmetric_layout as sl
from test_symmetric_tensor import LEGS, _shapes, _spaces, _sym
from tnkit.backend import NumpyBackend
from tnkit.symmetric_tensor import contract, permute, random, zeros
from tnkit.symmetry import SU2, U1

try:
    import torch
except ImportError:
    torch = None
HAS_CUDA = torch is not None and torch.cuda.is_available()
HAS_MPS = torch is not None and torch.backends.mps.is_available()
GPUS = [pytest.param("cuda", marks=[pytest.mark.cuda,
                                    pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")]),
        pytest.param("mps", marks=pytest.mark.skipif(not HAS_MPS, reason="no MPS device"))]
DEVICES = [None, "cpu"] + GPUS                  # None: numpy


def _reference(start, coef, shape):
    """start + sum_j digit_j coef[:, :, j], digits by division, row-major per pair."""
    E = shape.prod(axis=1)
    pair = np.repeat(np.arange(len(shape)), E)
    t = np.arange(E.sum()) - np.repeat(np.cumsum(E) - E, E)
    out = start[:, pair].astype(np.int64)
    for j in reversed(range(shape.shape[1])):
        n = shape[pair, j]
        out += (t % n) * coef[:, pair, j]
        t //= n
    return out


def _case(rng, pairs, legs, hi):
    shape = rng.integers(1, 6, size=(pairs, legs))
    shape[:, rng.integers(legs)] = 1                  # a leg of one state everywhere
    coef = rng.integers(0, hi, size=(2, pairs, legs))
    start = rng.integers(0, hi, size=(2, pairs))
    return start, coef, shape


def _check_positions(start, coef, shape, it, device):
    if device is not None and torch is None:
        pytest.skip("torch not installed")
    ref = _reference(start, coef, shape)
    g, h = rc._positions(start, coef, shape, it, device)
    for x, r in ((g, ref[0]), (h, ref[1])):
        if device is not None:
            assert x.device.type == device
            x = x.cpu().numpy()
        assert x.dtype == it and np.array_equal(x, r)
    return ref


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("legs", [1, 2, 4, 6])
def test_positions_equal_division(legs, device):
    _check_positions(*_case(np.random.default_rng(legs), 40, legs, 1000), np.int32, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("it, hi", [(np.int32, 2 ** 25), (np.int64, 2 ** 40)])
def test_positions_above_float32_range(it, hi, device):
    """Starts, coefficients and results above 2^24, results within the index dtype."""
    ref = _check_positions(*_case(np.random.default_rng(11), 400, 6, hi), it, device)
    assert 2 ** 24 < ref.max() < np.iinfo(it).max


@pytest.mark.parametrize("device", DEVICES)
def test_positions_where_earlier_form_left_int32(device):
    """First child times coefficient reaches 2.7e5 x 3e4, outside int32; the results do not."""
    rng = np.random.default_rng(7)
    shape = np.full((3, 2), 300)
    coef = np.array([[[1, 30000]] * 3, [[7000, 1]] * 3])
    start = rng.integers(0, 100, size=(2, 3))
    ref = _check_positions(start, coef, shape, np.int32, device)
    assert ref.max() < 2 ** 31 < 3 * 300 * 300 * 30000 // 2


def _splits(r):
    return [(p[:k], p[k:]) for p in itertools.permutations(range(r)) for k in range(r + 1)]


def _layouts(name, be):
    """Layouts of rank 3 and 4 from `_shapes`, degeneracies 1 to 5."""
    sym = _sym(name)
    legs = [{k: v + j + 1 for k, v in leg.items()} for j, leg in enumerate(LEGS[name])]
    return [zeros(cod, dom, be, sym).layout
            for cod, dom in _shapes(*_spaces(sym, *legs)) if len(cod) + len(dom) >= 3]


def _host(a):
    return None if a is None else np.asarray(a.cpu() if hasattr(a, "cpu") else a)


def _maps_digest(name, be):
    """(md5 of src, inv (with dtypes) and stack shapes of every permute of `_layouts`; the
    factors of the scalar entries of all those maps, in order, 1 where `scale` is None)."""
    h, scales = hashlib.md5(), []
    for lay in _layouts(name, be):
        for p1, p2 in _splits(len(lay.codomain) + len(lay.domain)):
            m = rc.Recoupler(lay, p1, p2, be)
            for a in map(_host, (m.src, m.inv)):
                h.update(b"-" if a is None else str(a.dtype).encode() + a.tobytes())
            h.update(repr([x[:3] for x in m.stacks]).encode())
            scales.append(np.ones(m.n_scalar) if m.scale is None else _host(m.scale))
    return h.hexdigest(), np.concatenate(scales)


def _weights(n):
    """Integer weights that make a sum see the order of the factors."""
    k = np.arange(n)
    return np.stack([np.ones(n), k % 7 - 3, k % 11 - 5, k % 13 - 6])


# From the division-based construction (numpy and torch-cpu agree): md5 of the index arrays,
# exact; the number of scalar factors and the sums of their real parts under `_weights`. The
# factors come from floating-point recoupling, so they differ between platforms in the last bits
# (and A4 ones carry imaginary parts of order 1e-15).
EARLIER_MAPS = {
    "U1": ("38ec8c5f4053242ff031c427642163cf", 318528, (318528.0, 0.0, -5.0, -11.0)),
    "SU2": ("60791838da68d8fbe2dc2a09ed90e246", 398016,
            (73812.5292147371, -167.4085945757957, 1218.2982163952565, 851.8511226485807)),
    "U1xSU2": ("9edae1e33c78e2c6dbe5e63f764e6823", 152064,
               (3190.6405167896546, -289.1531860660723, 278.4295153726676, 390.51473863271235)),
    "A4xSU2": ("6f8e04d555472246128848b6e2efb71d", 80064,
               (13046.737313979884, 117.44981766372736, -16.16926024087755, 47.073705549704044))}


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
@pytest.mark.parametrize("name", sorted(EARLIER_MAPS))
def test_maps_equal_earlier_maps(name, backend):
    if backend == "numpy":
        be = NumpyBackend()
    else:
        pytest.importorskip("torch")
        from tnkit.backend import TorchBackend
        be = TorchBackend("cpu")
    digest, scale = _maps_digest(name, be)
    want_digest, want_n, want_sums = EARLIER_MAPS[name]
    assert digest == want_digest and len(scale) == want_n
    assert np.abs(scale.imag).max(initial=0) < 1e-13
    terms = _weights(len(scale)) * scale.real
    assert np.all(np.abs(terms.sum(axis=1) - want_sums) <= 1e-13 * np.abs(terms).sum(axis=1))


def _device_backend(device, monkeypatch):
    """A torch backend whose maps take the device path: on "forced", the CPU with the path
    forced; else that device."""
    if torch is None:
        pytest.skip("torch not installed")
    from tnkit.backend import TorchBackend
    if device != "forced":
        return TorchBackend(device)
    be = TorchBackend("cpu")
    real = rc._on_device
    monkeypatch.setattr(rc, "_on_device", lambda b: b is be or real(b))
    return be


def _assert_same_maps(a, b, be):
    for x, y in ((a.src, b.src), (a.inv, b.inv)):
        assert (x is None) == (y is None)
        if x is not None:
            assert y.device.type == be.device.type
            x, y = _host(x), _host(y)
            assert x.dtype == y.dtype and np.array_equal(x, y)
    assert (a.scale is None) == (b.scale is None)
    if a.scale is not None:
        assert a.scale.dtype == b.scale.dtype and np.array_equal(a.scale, b.scale)
    assert [s[:3] for s in a.stacks] == [s[:3] for s in b.stacks]


@pytest.mark.parametrize("device", ["forced"] + GPUS)
@pytest.mark.parametrize("name", sorted(EARLIER_MAPS))
def test_device_maps_equal_host_maps(name, device, monkeypatch):
    be = _device_backend(device, monkeypatch)
    host = NumpyBackend()
    for lay in _layouts(name, host):
        for p1, p2 in _splits(len(lay.codomain) + len(lay.domain)):
            _assert_same_maps(rc.Recoupler(lay, p1, p2, host), rc.Recoupler(lay, p1, p2, be), be)


def _large(sym):
    """Rank-4 spaces: 1.3e6 entries (U(1)), 5.6e5 (SU(2))."""
    if isinstance(sym, U1):
        return _spaces(sym, {-1: 20, 0: 20, 1: 20}, {-1: 15, 0: 25, 1: 18},
                       {-2: 9, 0: 30, 2: 11}, {-1: 17, 1: 22})
    return _spaces(sym, {0: 12, 1: 20, 2: 9}, {1: 25, 3: 6}, {0: 14, 2: 16}, {1: 18, 3: 7})


@pytest.mark.parametrize("device", ["forced"] + GPUS)
@pytest.mark.parametrize("sym", [U1(), SU2()], ids=["U1", "SU2"])
def test_large_layout_permutes_equal_host(sym, device, monkeypatch):
    """permute on the device against numpy, on a layout large enough that a float32 rounding
    anywhere in the position arithmetic shows: U(1) bit for bit (a gather), SU(2) to float32
    rounding (batched products)."""
    be = _device_backend(device, monkeypatch)
    A, B, C, D = _large(sym)
    t = random((A, B), (C, D), NumpyBackend(np.float32), np.random.default_rng(0), sym=sym)
    u = zeros((A, B), (C, D), be, sym)._like(be.torch.as_tensor(t.data, device=be.device))
    for p1, p2 in _splits(4)[::23]:
        x = permute(t, p1, p2).data
        y = permute(u, p1, p2).data.cpu().numpy()
        if isinstance(sym, U1):
            assert x.tobytes() == y.tobytes(), (p1, p2)
        else:
            assert np.allclose(x, y, rtol=0, atol=1e-5 * np.abs(x).max()), (p1, p2)


def test_device_path_results_equal_host_results(monkeypatch):
    """permute and contract on torch-cpu, with maps from the host path and from the device
    path forced on the CPU: the same bits."""
    forced = _device_backend("forced", monkeypatch)
    from tnkit.backend import TorchBackend
    host = TorchBackend("cpu")
    sym = _sym("U1xSU2")
    rng = np.random.default_rng(5)
    legs = [{(q, s): int(rng.integers(1, 5)) for q, s in [(0, 0), (1, 1), (-1, 1), (0, 2)]}
            for _ in range(4)]
    A, B, C, D = _spaces(sym, *legs)
    tensors = [random((A, B.dual()), (C, D), be, np.random.default_rng(1), sym=sym)
               for be in (host, forced)]
    others = [random((C, D), (B,), be, np.random.default_rng(2), sym=sym)
              for be in (host, forced)]
    out = []
    for t, u in zip(tensors, others):
        sl.INDEX_CACHE.clear()
        res = [permute(t, p1, p2).data for p1, p2 in _splits(4)[::7]]
        res.append(contract(t, u, (2, 3), (0, 1)).data)
        out.append(res)
    sl.INDEX_CACHE.clear()
    assert len(out[0]) == len(out[1]) > 3
    for x, y in zip(*out):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.numpy().tobytes() == y.numpy().tobytes()
