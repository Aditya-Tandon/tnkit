"""`tnkit.legs`: fuse/split are exact inverses, report copies, and check their input."""
from __future__ import annotations

import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.legs import FuseRecord, fuse, split

BACKENDS = ["numpy"]
try:
    import torch
    BACKENDS += ["torch-cpu", pytest.param("torch-cuda", marks=[
        pytest.mark.cuda,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")])]
except ImportError:
    pass


def _rand(be, shape, seed=0):
    return be.randn(np.random.default_rng(seed), shape)


def _np(be, x):
    return np.asarray(be.to_numpy(x))


@pytest.mark.parametrize("spec", BACKENDS)
@pytest.mark.parametrize("groups", [((0, 1), (2, 3)), ((0,), (1, 2), (3,)), ((0, 2), (1, 3)),
                                    ((3, 1), (0,), (2,)), ((0, 1, 2, 3),)])
def test_split_inverts_fuse_exactly(spec, groups):
    be = get_backend(spec)
    x = _rand(be, (2, 3, 4, 5))
    y, rec = fuse(x, groups, be)
    assert tuple(y.shape) == tuple(int(np.prod([x.shape[a] for a in g])) for g in groups)
    assert np.array_equal(_np(be, split(y, rec, be)), _np(be, x))
    # the fused array is the permuted array, flattened per group, in C order
    ref = np.transpose(_np(be, x), rec.perm).reshape(tuple(y.shape))
    assert np.array_equal(_np(be, y), ref)


@pytest.mark.parametrize("spec", BACKENDS)
def test_copied_reports_allocation(spec):
    be = get_backend(spec)
    x = _rand(be, (2, 3, 4, 5))
    _, rec = fuse(x, [(0, 1), (2, 3)], be)
    assert rec.copied is False                       # in-order groups: a view
    _, rec = fuse(x, [(0, 2), (1, 3)], be)
    assert rec.copied is True                        # permuted, non-contiguous: a copy
    _, rec = fuse(be.transpose(x, (1, 0, 2, 3)), [(0, 1), (2, 3)], be)
    assert rec.copied is True                        # non-contiguous input


def test_record_fields_and_equality():
    be = get_backend("numpy")
    x = _rand(be, (2, 3, 4, 5))
    _, rec = fuse(x, [(0, 1), (2, 3)], be)
    assert rec == FuseRecord(shape=(2, 3, 4, 5), groups=((0, 1), (2, 3)))
    assert rec.perm == (0, 1, 2, 3) and rec.dims(0) == (2, 3) and rec.dims(1) == (4, 5)
    assert rec == FuseRecord((2, 3, 4, 5), ((0, 1), (2, 3)), copied=True)
    assert FuseRecord((2, 3, 4, 5), ((1, 0), (3, 2))).perm == (1, 0, 3, 2)
    with pytest.raises(AttributeError):
        rec.shape = (1,)


@pytest.mark.parametrize("spec", BACKENDS)
def test_partial_split_after_svd(spec):
    """The two-site truncation pattern: U and Vh keep one fused leg each."""
    be = get_backend(spec)
    a, s1, s2, b = 3, 2, 2, 4
    theta = _rand(be, (a, s1, s2, b))
    m, rec = fuse(theta, [(0, 1), (2, 3)], be)
    U, S, Vh = be.svd(m)
    k = int(S.shape[0])
    Ul = split(U, rec, be, legs=(0,))
    Vr = split(Vh, rec, be, legs=(1,))
    assert tuple(Ul.shape) == (a, s1, k) and tuple(Vr.shape) == (k, s2, b)
    assert np.array_equal(_np(be, Ul), _np(be, U.reshape(a, s1, k)))
    assert np.array_equal(_np(be, Vr), _np(be, Vh.reshape(k, s2, b)))
    # batched: a leading leg axis stays a group of its own
    th = _rand(be, (2, a, s1, s2, b))
    mb, recb = fuse(th, [(0,), (1, 2), (3, 4)], be)
    Ub, Sb, Vhb = be.svd_batched(mb)
    assert tuple(split(Ub, recb, be, legs=(1,)).shape) == (2, a, s1, int(Sb.shape[1]))


def test_split_checks_its_input():
    be = get_backend("numpy")
    x = _rand(be, (2, 3, 4, 5))
    y, rec = fuse(x, [(0, 1), (2, 3)], be)
    with pytest.raises(ValueError, match="fused legs"):
        split(y.reshape(-1), rec, be)
    with pytest.raises(ValueError, match="leg 0 has size"):
        split(y.reshape(20, 6), rec, be)            # same size, wrong split between legs
    with pytest.raises(ValueError, match="leg 1 has size"):
        split(np.zeros((6, 7)), rec, be, legs=(1,))
    with pytest.raises(ValueError, match="out of range"):
        split(y, rec, be, legs=(2,))
    yp, recp = fuse(x, [(1, 0), (2, 3)], be)
    with pytest.raises(ValueError, match="unpermuted"):
        split(yp, recp, be, legs=(1,))


@pytest.mark.parametrize("groups", [((0, 1), (1, 2, 3)), ((0, 1), (3,)), ((0, 1), (), (2, 3))])
def test_fuse_rejects_a_non_partition(groups):
    be = get_backend("numpy")
    with pytest.raises(ValueError, match="partition"):
        fuse(_rand(be, (2, 3, 4, 5)), groups, be)


def test_a_record_must_cover_its_shape():
    with pytest.raises(ValueError, match="partition"):
        FuseRecord((2, 3, 4, 5, 6), ((0, 1), (2, 3)))     # a leg left out
    with pytest.raises(ValueError, match="partition"):
        FuseRecord((2, 3, 4, 5), ((0, 1), (1, 3)))        # a leg twice


@pytest.mark.parametrize("spec", BACKENDS)
def test_transpose_matches_numpy(spec):
    be = get_backend(spec)
    x = _rand(be, (2, 3, 4))
    for perm in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
        assert np.array_equal(_np(be, be.transpose(x, perm)), np.transpose(_np(be, x), perm))
