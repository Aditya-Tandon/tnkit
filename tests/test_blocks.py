"""`tnkit.blocks`: block-sparse U(1) tensors agree with their dense arrays to round-off."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from tnkit import blocks as B
from tnkit.block_backend import BlockBackend, mpo_charges, to_block_mpo
from tnkit._dmrg import apply_one_site, apply_two_site, update_left_env, update_right_env
from tnkit.backend import get_backend
from tnkit.batch_tdvp import _ranks_from_spectrum
from tnkit.eigensolver import lanczos_expm
from tnkit.legs import FuseRecord, fuse, split
from tnkit.models_lgt import schwinger_mpo
from tnkit.symmetry import U1

BACKENDS = ["numpy"]
try:
    import torch
    BACKENDS += ["torch-cpu", pytest.param("torch-cuda", marks=[
        pytest.mark.cuda,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")])]
except ImportError:
    pass

BOND = B.Leg((-1, 0, 1, 2), (3, 5, 4, 1))
CH = B.Leg((-1, 0, 1), (1, 3, 1))
EVEN, ODD = B.Leg((0, 1), (1, 1)), B.Leg((-1, 0), (1, 1))     # staggered site charges


def _np(be, x):
    return np.asarray(be.to_numpy(x))


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def _site_charges(n):
    """Site n of the Schwinger chain in tnkit's basis (index 0 is Z = +1)."""
    return np.array([1, 0]) if n % 2 == 0 else np.array([0, -1])


def _block_mpo(be, N, x=1.0, mu=0.5):
    """The Schwinger MPO, and its block tensors with their legs' dense-index orders."""
    mpo = schwinger_mpo(N, x, mu)
    return (mpo,) + to_block_mpo(mpo, [_site_charges(n) for n in range(N)], be, sym=U1())


@pytest.mark.parametrize("spec", BACKENDS)
def test_dense_round_trip_and_projection(spec):
    be = get_backend(spec)
    t = B.random((BOND, EVEN, BOND.flip()), 0, be, np.random.default_rng(0))
    back, lost = B.from_dense(t.to_dense(), t.legs, 0)
    assert lost == 0.0 and np.array_equal(_np(be, back.data), _np(be, t.data))
    q = np.array([0, 1, 0, -1, 1])
    leg, order = B.Leg.from_charges(q)
    x = np.random.default_rng(1).standard_normal((5, 5))
    s, lost = B.from_dense(be.array(x), (leg, leg.flip()), 0, (order, order))
    keep = q[:, None] == q[None, :]
    assert np.array_equal(_np(be, s.to_dense((order, order))), x * keep)
    assert lost == pytest.approx(float((x[~keep] ** 2).sum()), rel=1e-12)


@pytest.mark.parametrize("spec", BACKENDS)
@pytest.mark.parametrize("case", ["one-leg", "two-legs", "charged", "empty-sector"])
def test_tensordot_matches_dense(spec, case):
    be = get_backend(spec)
    rng = np.random.default_rng(2)
    if case == "one-leg":
        a = B.random((BOND, CH.flip(), BOND.flip()), 0, be, rng)
        b = B.random((BOND, EVEN, ODD, BOND.flip()), 0, be, rng)
        axes, sub = ((2,), (0,)), "pwq,qijb->pwijb"
    elif case == "two-legs":
        a = B.random((CH, CH.flip(), EVEN, EVEN.flip()), 0, be, rng)
        b = B.random((BOND, CH.flip(), EVEN, ODD, BOND.flip()), 0, be, rng)
        axes, sub = ((0, 3), (1, 2)), "wmsi,pwijb->mspjb"
    elif case == "charged":
        a = B.random((BOND, EVEN, BOND), 1, be, rng)
        b = B.random((BOND.flip(), ODD, BOND), -2, be, rng)
        axes, sub = ((2,), (0,)), "asb,btc->astc"
    else:   # charge-1 rows of the output have no contracted sector: that block must be zero
        x, k = B.Leg((0, 1), (2, 3)), B.Leg((0,), (4,))
        a = B.random((x, k.flip()), 0, be, rng)
        b = B.random((k, x.flip()), 0, be, rng)
        axes, sub = ((1,), (0,)), "xk,ky->xy"
    out = B.tensordot(a, b, axes)
    ref = np.einsum(sub, _np(be, a.to_dense()), _np(be, b.to_dense()))
    assert out.charge == a.charge + b.charge
    assert np.abs(_np(be, out.to_dense()) - ref).max() <= 1e-13 * max(1.0, np.abs(ref).max())


def test_contracting_unmatched_legs_raises():
    be = get_backend("numpy")
    a = B.random((BOND, EVEN, BOND.flip()), 0, be, np.random.default_rng(0))
    with pytest.raises(ValueError, match="signs be opposite"):
        B.tensordot(a, a, ((2,), (2,)))


@pytest.mark.parametrize("spec", BACKENDS)
def test_transpose_moves_no_data(spec):
    be = get_backend(spec)
    t = B.random((BOND, CH.flip(), EVEN, BOND.flip()), 0, be, np.random.default_rng(3))
    p = t.transpose((3, 0, 2, 1))
    assert p.data is t.data
    assert np.array_equal(_np(be, p.to_dense()), np.transpose(_np(be, t.to_dense()), (3, 0, 2, 1)))
    c = p.canonical()
    assert np.array_equal(_np(be, c.to_dense()), _np(be, p.to_dense()))


@pytest.mark.parametrize("spec", BACKENDS)
def test_vector_algebra_across_layouts(spec):
    be = get_backend(spec)
    bb = BlockBackend(be)
    rng = np.random.default_rng(4)
    t = B.random((BOND, EVEN, BOND.flip()), 0, be, rng)
    u = B.random((BOND, EVEN, BOND.flip()), 0, be, rng)
    v = u.transpose((2, 1, 0)).canonical().transpose((2, 1, 0))     # same tensor, other layout
    assert v.struct is not u.struct
    dt, du = _np(be, t.to_dense()), _np(be, u.to_dense())
    for got, ref in [((t + v) * 2.0 - v / 4.0, (dt + du) * 2.0 - du / 4.0), (-t, -dt),
                     (0.5 * v, 0.5 * du)]:
        assert np.abs(_np(be, got.to_dense()) - ref).max() <= 1e-13
    assert abs(complex(be.to_scalar(bb.vdot(t, v))) - np.vdot(dt, du)) <= 1e-11
    assert float(bb.norm(t)) == pytest.approx(np.linalg.norm(dt), rel=1e-13)
    assert _np(be, bb.zeros_like(t).data).sum() == 0
    c = t.conj()
    assert [leg.sign for leg in c.legs] == [-1, -1, 1] and c.charge == 0
    assert np.array_equal(_np(be, c.to_dense()), np.conj(dt))
    with pytest.raises(ValueError, match="contiguous"):
        B.BlockTensor(t.struct, be.zeros((2 * t.struct.size,))[::2])


@pytest.mark.parametrize("spec", BACKENDS)
def test_two_site_matvec_at_collision_sector_shapes(spec):
    """The bond sectors of an evolved Schwinger collision state at chi = 500 ({-1: 85, 0: 242,
    1: 164, 2: 9}) scaled down by 10, the bulk Schwinger MPO, random symmetric environments."""
    be = get_backend(spec)
    rng = np.random.default_rng(5)
    bond = B.Leg((-1, 0, 1, 2), (9, 24, 16, 1))
    _, (_, _, _, _, W1, W2, *_), _ = _block_mpo(be, 10)
    ch = W1.legs[0]
    theta = B.random((bond, EVEN, ODD, bond.flip()), 0, be, rng)
    L = B.random((bond, ch.flip(), bond.flip()), 0, be, rng)
    R = B.random((bond.flip(), ch, bond), 0, be, rng)
    bb = BlockBackend(be)
    out = apply_two_site(theta, L, W1, W2, R, be=bb)
    ref = apply_two_site(*(t.to_dense() for t in (theta, L, W1, W2, R)), be=be)
    assert _rel(_np(be, out.to_dense()), _np(be, ref)) <= 1e-13
    assert not bb.fallbacks


@pytest.mark.parametrize("spec", BACKENDS)
def test_matvecs_and_environments_on_a_schwinger_chain(spec):
    """A random U(1)-symmetric MPS on N = 8 sites with the real Schwinger MPO. Environments are
    built densely from the dense tensors; their projection onto blocks drops nothing, and the
    block environment updates, two-site and one-site matvecs equal the dense ones."""
    be = get_backend(spec)
    N, rng = 8, np.random.default_rng(6)
    mpo, wb, worders = _block_mpo(be, N)
    qs = [_site_charges(n) for n in range(N)]
    cum = np.cumsum([0] + [max(q) for q in qs]), np.cumsum([0] + [min(q) for q in qs])
    bonds = []
    for n in range(N + 1):       # charges reachable from the left that can still return to 0
        hi = min(cum[0][n], -(cum[1][N] - cum[1][n]))
        lo = max(cum[1][n], -(cum[0][N] - cum[0][n]))
        qn = tuple(range(lo, hi + 1))
        bonds.append(B.Leg(qn, tuple(1 if n in (0, N) else 2 for _ in qn)))
    phys = [B.Leg.from_charges(q) for q in qs]
    mps = [B.random((bonds[n], phys[n][0], bonds[n + 1].flip()), 0, be, rng) for n in range(N)]
    dense = [m.to_dense((None, phys[n][1], None)) for n, m in enumerate(mps)]
    W = [be.array(w) for w in mpo]
    bb = BlockBackend(be)
    i = 3
    one = be.ones((1, 1, 1))
    Ld, Lb = one, B.from_dense(one, (bonds[0], wb[0].legs[0].flip(), bonds[0].flip()))[0]
    for n in range(i):
        Ld = update_left_env(Ld, dense[n], W[n], be)
        Lb = update_left_env(Lb, mps[n], wb[n], bb)
    Rd, Rb = one, B.from_dense(one, (bonds[N].flip(), wb[N - 1].legs[1].flip(), bonds[N]))[0]
    for n in range(N - 1, i + 1, -1):
        Rd = update_right_env(Rd, dense[n], W[n], be)
        Rb = update_right_env(Rb, mps[n], wb[n], bb)
    ch_l = (None, worders[i][0], None)
    ch_r = (None, worders[i + 1][1], None)
    Lp, lost_l = B.from_dense(Ld, Lb.legs, 0, ch_l)
    Rp, lost_r = B.from_dense(Rd, Rb.legs, 0, ch_r)
    assert lost_l <= 1e-24 * float(be.norm(Ld)) ** 2 and lost_r <= 1e-24 * float(be.norm(Rd)) ** 2
    assert _rel(_np(be, Lb.to_dense(ch_l)), _np(be, Ld)) <= 1e-13
    assert _rel(_np(be, Rb.to_dense(ch_r)), _np(be, Rd)) <= 1e-13

    theta = B.tensordot(mps[i], mps[i + 1], ((2,), (0,)))
    theta_d = be.einsum("asb,btc->astc", dense[i], dense[i + 1])
    ref = apply_two_site(theta_d, Ld, W[i], W[i + 1], Rd, be)
    got = apply_two_site(theta, Lb, wb[i], wb[i + 1], Rb, be=bb)
    orders = (None, phys[i][1], phys[i + 1][1], None)
    assert _rel(_np(be, got.to_dense(orders)), _np(be, ref)) <= 1e-13

    R1d = update_right_env(Rd, dense[i + 1], W[i + 1], be)
    R1b = update_right_env(Rb, mps[i + 1], wb[i + 1], bb)
    ref1 = apply_one_site(dense[i], Ld, W[i], R1d, be)
    got1 = apply_one_site(mps[i], Lb, wb[i], R1b, be=bb)
    assert _rel(_np(be, got1.to_dense((None, phys[i][1], None))), _np(be, ref1)) <= 1e-13
    assert not bb.fallbacks


def test_dense_fallback_is_counted_or_raised():
    be = get_backend("numpy")
    t = B.random((BOND, BOND.flip()), 0, be, np.random.default_rng(7))
    strict = BlockBackend(be)
    with pytest.raises(NotImplementedError, match="no block-sparse implementation"):
        strict.svd(t)
    with pytest.raises(NotImplementedError):
        strict.einsum("ab,ab->a", t, t)                   # a batch index: no block plan
    loose = BlockBackend(be, strict=False)
    U, S, Vh = loose.svd(t)
    assert np.allclose((U * S) @ Vh, t.to_dense())
    loose.einsum("ab,ab->a", t, t)
    assert loose.fallbacks == {"svd": 1, "einsum": 1}
    d = t.to_dense()
    assert loose.einsum("ab,bc->ac", d, d).shape == t.shape          # no block operand: uncounted
    assert loose.fallbacks == {"svd": 1, "einsum": 1}


@pytest.mark.parametrize("spec", BACKENDS)
@pytest.mark.parametrize("groups", [((0, 1), (2, 3)), ((0,), (1, 2, 3)), ((3, 1), (0, 2))])
def test_fuse_and_split_through_legs(spec, groups):
    be = get_backend(spec)
    t = B.random((BOND, EVEN, ODD, BOND.flip()), 0, be, np.random.default_rng(8))
    f, rec = fuse(t, groups, be)
    assert isinstance(rec, B.BlockFuseRecord) and isinstance(rec, FuseRecord)
    assert rec.shape == t.shape and rec.copied
    back = split(f, rec, be)
    assert back.struct is t.struct and np.array_equal(_np(be, back.data), _np(be, t.data))
    # the fused leg lists each charge's sub-blocks in turn, so the fused tensor is the dense
    # fused array with its rows and columns reordered by charge
    starts = [np.cumsum((0,) + leg.dims[:-1]) for leg in t.legs]
    order = []
    for g, grp in enumerate(rec.groups):
        idx = []
        for sec in rec.sectors[g]:
            for sub, _, _ in sec:
                axes = [starts[a][s] + np.arange(t.legs[a].dims[s]) for a, s in zip(grp, sub)]
                idx.extend(np.ravel_multi_index(np.ix_(*axes), [t.shape[a] for a in grp]).ravel())
        order.append(np.array(idx))
    dense_f = np.transpose(_np(be, t.to_dense()), rec.perm).reshape(f.shape)
    assert np.array_equal(_np(be, f.to_dense()), dense_f[np.ix_(*order)])
    with pytest.raises(TypeError, match="sector order"):          # a densified block tensor
        split(f.to_dense(), rec, be)


@pytest.mark.parametrize("spec", BACKENDS)
def test_split_guards_and_a_record_of_another_charge(spec):
    """A partial split needs an unpermuted fuse. A full split needs only matching fused legs:
    the record of a charge-0 tensor splits a charge-1 tensor with the same legs."""
    be = get_backend(spec)
    rng = np.random.default_rng(21)
    legs = (BOND, EVEN, ODD, BOND.flip())
    t = B.random(legs, 1, be, rng)
    f, _ = fuse(t, ((0, 1), (2, 3)), be)
    _, rec0 = fuse(B.random(legs, 0, be, rng), ((0, 1), (2, 3)), be)
    back = split(f, rec0, be)
    assert back.struct is t.struct and np.array_equal(_np(be, back.data), _np(be, t.data))
    fp, recp = fuse(t, ((1, 0), (2, 3)), be)
    with pytest.raises(ValueError, match="unpermuted"):
        split(fp, recp, be, legs=(0,))


def test_adding_tensors_of_different_charge_raises():
    """Equal legs and equal buffer sizes, so only the charge check can catch it."""
    be = get_backend("numpy")
    legs = (B.Leg((0, 1), (2, 2)), B.Leg((-1, 0, 1), (2, 2, 2), -1))
    a, b = (B.random(legs, q, be, np.random.default_rng(q)) for q in (0, 1))
    assert a.struct.size == b.struct.size
    with pytest.raises(ValueError, match="charge"):
        a + b


def test_plans_and_layouts_are_cached():
    be = get_backend("numpy")
    rng = np.random.default_rng(9)
    a = B.random((BOND, CH.flip(), BOND.flip()), 0, be, rng)
    b = B.random((BOND, EVEN, ODD, BOND.flip()), 0, be, rng)
    first = B.tensordot(a, b, ((2,), (0,)))
    info = B.cache_info()
    second = B.tensordot(B.random(a.legs, 0, be, rng), b, ((2,), (0,)))
    after = B.cache_info()
    assert after["misses"] == info["misses"] and after["hits"] > info["hits"]
    assert second.struct is first.struct


def test_mpo_charges():
    qs = [_site_charges(n) for n in range(12)]
    chans = mpo_charges(schwinger_mpo(12, 1.0, 0.5), qs, sym=U1())
    assert [c.tolist() for c in chans[:3]] == [[0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0]]
    assert chans[-1].tolist() == [0]
    mixed = [np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 2, 2))]
    mixed[0][0, 0] = [[1, 1], [0, 0]]                    # Z-diagonal and a raising term together
    mixed[1][0, 0] = np.eye(2)
    with pytest.raises(ValueError, match="right channel 0"):
        mpo_charges(mixed, qs[:2], sym=U1())


# --- review follow-ups: failure paths, randomised structure, full-size sectors ---

def test_dense_fallback_results_cannot_be_split_with_a_block_record():
    be = get_backend("numpy")
    t = B.random((BOND, EVEN, ODD, BOND.flip()), 0, be, np.random.default_rng(10))
    f, rec = fuse(t, ((0, 1), (2, 3)), be)
    loose = BlockBackend(be, strict=False)
    U, S, Vh = loose.svd(f)
    with pytest.raises(TypeError, match="BlockFuseRecord"):
        split((U * S) @ Vh, rec, be)
    with pytest.raises(TypeError):
        f.split(FuseRecord(f.shape, ((0,), (1,))))


def test_block_operands_inside_containers_are_fallbacks():
    be = get_backend("numpy")
    t = B.random((BOND, BOND.flip()), 0, be, np.random.default_rng(11))
    with pytest.raises(NotImplementedError):
        BlockBackend(be).stack([t, t])
    loose = BlockBackend(be, strict=False)
    out = loose.stack([t, t])
    assert out.dtype != object and out.shape == (2,) + t.shape
    assert np.array_equal(out[1], t.to_dense())
    assert loose.fallbacks == {"stack": 1}


@pytest.mark.parametrize("spec", BACKENDS)
def test_scalars_dtypes_and_read_only_buffer(spec):
    be = get_backend(spec)
    bb = BlockBackend(be)
    t = B.random((BOND, EVEN, BOND.flip()), 0, be, np.random.default_rng(12))
    full = bb.einsum("asb,asb->", t, t.conj())                     # every leg contracted
    assert tuple(full.shape) == () and abs(complex(full) - float(bb.norm(t)) ** 2) < 1e-10
    with pytest.raises(TypeError):
        t * be.ones((t.struct.size,))                             # an array, not a scalar
    with pytest.raises(TypeError):
        t / be.ones((1,))
    assert np.allclose(_np(be, (t * be.ones(()) * 2).data), 2 * _np(be, t.data))
    with pytest.raises(AttributeError):
        t.data = t.data
    real = B.BlockTensor(t.struct, t.data.real.contiguous() if hasattr(t.data, "is_cuda")
                         else np.ascontiguousarray(t.data.real))
    got = B.tensordot(t, real.conj(), ((0, 1, 2), (0, 1, 2)))     # complex with real
    ref = np.vdot(_np(be, real.to_dense()), _np(be, t.to_dense()))
    assert abs(complex(_np(be, got.data)[0]) - ref) < 1e-10


def _random_leg(rng, sign=None):
    n = int(rng.integers(1, 5))
    q = sorted(rng.choice(np.arange(-3, 4), n, replace=False).tolist())
    return B.Leg(tuple(q), tuple(int(d) for d in rng.integers(1, 4, n)),
                 int(sign if sign is not None else rng.choice([-1, 1])))


@pytest.mark.parametrize("spec", BACKENDS)
def test_random_contractions_and_fusions(spec):
    """Seeded random charge structures: tensordot against np.tensordot, and fuse/split."""
    be = get_backend(spec)
    rng = np.random.default_rng(13)
    nonempty = 0
    for _ in range(150):
        na, nb, nc = (int(x) for x in rng.integers(0, 3, 3))
        nc = max(nc, 1)
        shared = [_random_leg(rng) for _ in range(nc)]
        la = [_random_leg(rng) for _ in range(na)] + shared
        lb = [leg.flip() for leg in shared] + [_random_leg(rng) for _ in range(nb)]
        pa, pb = rng.permutation(len(la)), rng.permutation(len(lb))
        la, lb = [la[i] for i in pa], [lb[i] for i in pb]
        ax_a = [int(np.where(pa == na + k)[0][0]) for k in range(nc)]
        ax_b = [int(np.where(pb == k)[0][0]) for k in range(nc)]
        a = B.random(la, int(rng.integers(-1, 2)), be, rng)
        b = B.random(lb, int(rng.integers(-1, 2)), be, rng)
        got = B.tensordot(a, b, (ax_a, ax_b))
        ref = np.tensordot(_np(be, a.to_dense()), _np(be, b.to_dense()), (ax_a, ax_b))
        assert np.abs(_np(be, got.to_dense()) - ref).max() <= 1e-12 * max(1.0, np.abs(ref).max())
        nonempty += got.struct.size > 0
        if a.ndim >= 2:
            p = [int(i) for i in rng.permutation(a.ndim)]
            cut = int(rng.integers(1, a.ndim))
            groups = (tuple(p[:cut]), tuple(p[cut:]))
            f, rec = fuse(a, groups, be)
            assert np.array_equal(_np(be, split(f, rec, be).canonical().data),
                                  _np(be, a.canonical().data))
    assert nonempty > 30


def test_matvec_at_full_collision_sector_sizes():
    """The two-site matvec at the measured chi = 260 sectors, unscaled, against dense."""
    be = get_backend("numpy")
    rng = np.random.default_rng(14)
    bond = B.Leg((-2, -1, 0, 1), (6, 88, 124, 42))
    _, wb, _ = _block_mpo(be, 10)
    W1, W2 = wb[4], wb[5]
    ch = W1.legs[0]
    ts = (B.random((bond, EVEN, ODD, bond.flip()), 0, be, rng),
          B.random((bond, ch.flip(), bond.flip()), 0, be, rng), W1, W2,
          B.random((bond.flip(), ch, bond), 0, be, rng))
    got = apply_two_site(*ts, be=BlockBackend(be))
    assert _rel(got.to_dense(), apply_two_site(*(t.to_dense() for t in ts), be=be)) <= 1e-12


def test_same_dense_sizes_with_other_sectors_get_other_plans():
    be = get_backend("numpy")
    rng = np.random.default_rng(15)
    x, y = B.Leg((0, 1), (3, 5)), B.Leg((0, 1), (5, 3))           # both of size 8
    outs = []
    for leg in (x, y):
        a = B.random((leg, leg.flip()), 0, be, rng)
        b = B.random((leg, leg.flip()), 0, be, rng)
        got = B.tensordot(a, b, ((1,), (0,)))
        assert np.allclose(got.to_dense(), a.to_dense() @ b.to_dense(), atol=1e-13)
        outs.append(got.struct)
    assert outs[0] is not outs[1] and outs[0].blocks == outs[1].blocks
    assert outs[0].offsets != outs[1].offsets or outs[0].strides != outs[1].strides


@pytest.mark.parametrize("spec", BACKENDS)
def test_charged_one_site_matvec_and_environment(spec):
    be = get_backend(spec)
    rng = np.random.default_rng(16)
    bb = BlockBackend(be)
    _, wb, _ = _block_mpo(be, 10)
    W = wb[4]
    ch = W.legs[0]
    M = B.random((BOND, EVEN, BOND.flip()), 1, be, rng)            # total charge 1
    L = B.random((BOND, ch.flip(), BOND.flip()), 0, be, rng)
    R = B.random((BOND.flip(), ch, BOND), 0, be, rng)
    dense = lambda *ts: [t.to_dense() for t in ts]
    got = apply_one_site(M, L, W, R, be=bb)
    assert got.charge == 1
    assert _rel(_np(be, got.to_dense()), _np(be, apply_one_site(*dense(M, L, W, R), be=be))) <= 1e-13
    Ln = update_left_env(L, M, W, bb)
    assert _rel(_np(be, Ln.to_dense()), _np(be, update_left_env(*dense(L, M, W), be=be))) <= 1e-13


@pytest.mark.parametrize("spec", BACKENDS)
def test_lanczos_expm_through_the_block_backend(spec):
    """exp(tau H) v by Krylov on block tensors equals scipy's dense expm, for a Hermitian,
    charge-conserving H acting on a charge-0 two-leg vector."""
    be = get_backend(spec)
    bb = BlockBackend(be)
    rng = np.random.default_rng(22)
    legs = (BOND, BOND.flip())
    A = B.random(legs + (BOND.flip(), BOND), 0, be, rng)
    H = A + A.conj().transpose((2, 3, 0, 1))
    v = B.random(legs, 0, be, rng)
    tau = -0.3j
    got = lanczos_expm(lambda y: bb.einsum("abcd,cd->ab", H, y), v, tau, bb, m=30)
    n = BOND.size ** 2
    ref = expm(tau * _np(be, H.to_dense()).reshape(n, n)) @ _np(be, v.to_dense()).ravel()
    assert np.abs(_np(be, got.to_dense()).ravel() - ref).max() <= 1e-12 * np.abs(ref).max()
    assert not bb.fallbacks


@pytest.mark.skipif("torch-cpu" not in BACKENDS, reason="needs torch")
def test_buffers_at_a_storage_offset():
    """Torch buffers that are views at a non-zero storage offset, on the path that multiplies
    the operands' buffers directly, without a gather."""
    be = get_backend("torch-cpu")
    rng = np.random.default_rng(21)
    X = B.Leg((-1, 0, 1), (2, 3, 2), -1)
    K, Y = B.Leg((0, 1, 2), (3, 1, 2)), B.Leg((-1, 0, 1), (1, 2, 3))
    a, b = B.random((X, K), 1, be, rng), B.random((K.flip(), Y), -1, be, rng)
    p = B._plan(a.struct, b.struct, (1,), (0,))
    assert p.ia is None and p.ib is None

    def shifted(t, n):
        buf = torch.zeros(n + t.struct.size, dtype=t.data.dtype)
        buf[n:] = t.data
        return B.BlockTensor(t.struct, buf[n:])

    a1, b1 = shifted(a, 3), shifted(b, 5)
    assert (a1.data.storage_offset(), b1.data.storage_offset()) == (3, 5)
    got = B.tensordot(a1, b1, ((1,), (0,)))
    ref = np.tensordot(_np(be, a.to_dense()), _np(be, b.to_dense()), (1, 0))
    assert np.abs(_np(be, got.to_dense()) - ref).max() <= 1e-13


def test_cache_is_bounded_and_counted():
    be = get_backend("numpy")
    rng = np.random.default_rng(17)
    limit = B.cache_info()["max_bytes"]
    try:
        B.set_cache_limit(4_000)
        before = B.cache_info()
        for d in range(4, 12):
            leg = B.Leg((0, 1), (d, d + 1))
            a = B.random((leg, leg.flip()), 0, be, rng)
            B.tensordot(a, a.transpose((1, 0)).conj().transpose((1, 0)), ((1,), (1,)))
        info = B.cache_info()
        assert info["evictions"] > before["evictions"] and info["misses"] > before["misses"]
        assert info["bytes"] <= 4_000 or info["entries"] == 1
    finally:
        B.set_cache_limit(limit)


def test_mpo_charges_of_a_fragment():
    qs = [_site_charges(n) for n in range(12)]
    mpo = schwinger_mpo(12, 1.0, 0.5)
    chans = mpo_charges(mpo, qs, sym=U1())
    with pytest.raises(ValueError, match="fragment"):
        mpo_charges(mpo[6:], qs[6:], sym=U1())
    tail = mpo_charges(mpo[6:], qs[6:], left=chans[6], sym=U1())
    assert [c.tolist() for c in tail] == [c.tolist() for c in chans[6:]]


# --- truncated SVD ---

def _spectrum_matrix(be, rng, legs, decay=8.0, charge=0):
    """A two-leg block matrix whose singular values fall geometrically to 10**-decay,
    interleaved across sectors so the global cut mixes them."""
    m = B.random(legs, charge, be, rng).canonical()
    n = sum(min(m.struct.shape(k)) for k in range(len(m.struct.blocks)))
    vals = 10.0 ** (-decay * rng.permutation(n) / n)
    parts, i = [], 0
    for k in range(len(m.struct.blocks)):
        R, C = m.struct.shape(k)
        r = min(R, C)
        Q1 = np.linalg.qr(rng.standard_normal((R, r)) + 1j * rng.standard_normal((R, r)))[0]
        Q2 = np.linalg.qr(rng.standard_normal((C, r)) + 1j * rng.standard_normal((C, r)))[0]
        parts.append(((Q1 * vals[i:i + r]) @ Q2.conj().T).ravel())
        i += r
    return B.BlockTensor(m.struct, be.array(np.concatenate(parts)))


def _dense_truncation(dense, legs, charge, chi, svd_tol=0.0, zero_tol=0.0):
    """The rank the dense rule keeps, and how many kept values carry each bond charge. Each
    allowed block of the sector-ordered dense matrix is factorised on its own; its singular
    vectors carry the row sector's charge times the row leg's sign."""
    lr, lc = legs
    r0, c0 = np.cumsum((0,) + lr.dims), np.cumsum((0,) + lc.dims)
    vals, labels = [], []
    for r, q in enumerate(lr.charges):
        for c, p in enumerate(lc.charges):
            if lr.sign * q + lc.sign * p == charge:
                sv = np.linalg.svd(dense[r0[r]:r0[r + 1], c0[c]:c0[c + 1]], compute_uv=False)
                vals += list(sv)
                labels += [lr.sign * q] * len(sv)
    order = np.argsort(-np.array(vals), kind="stable")
    S0 = np.array(vals)[order]
    k = min(chi, len(S0))
    if zero_tol > 0:
        k = min(k, max(1, int((S0 > zero_tol * S0[0]).sum())))
    if svd_tol > 0 and k > 1:
        k = int(_ranks_from_spectrum(S0[None], k, svd_tol)[0][0])
    kept = np.array(labels)[order[:k]]
    return k, {int(q): int((kept == q).sum()) for q in np.unique(kept)}


def _check_bond(U, S, Vh, m, k, counts):
    bond = Vh.legs[0]
    assert len(S) == k and bond.sign == 1 and U.legs == (m.legs[0], bond.flip())
    assert dict(zip(bond.charges, bond.dims)) == counts
    assert U.charge == 0 and Vh.charge == m.charge and Vh.legs[1] == m.legs[1]


@pytest.mark.parametrize("spec", BACKENDS)
@pytest.mark.parametrize("method", ["svd", "gram"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("charge", [0, 1, -1, -2])
def test_truncated_svd_matches_dense(spec, method, sign, charge):
    be = get_backend(spec)
    rng = np.random.default_rng(18)
    rq, rd = (-1, 0, 1, 2), (9, 33, 41, 17)
    row = B.Leg(rq, rd, sign)
    col = B.Leg(tuple(sorted(charge - sign * q for q in rq)), (17, 41, 33, 9))
    m = _spectrum_matrix(be, rng, (row, col), decay=5.0, charge=charge)
    dense = _np(be, m.to_dense())
    U0, S0, V0 = np.linalg.svd(dense)
    for chi, tol in ((40, 0.0), (90, 0.0), (90, 1e-6), (90, 1e-3)):
        U, S, Vh, disc = B.truncated_svd(m, chi, svd_tol=tol, method=method)
        k, counts = _dense_truncation(dense, m.legs, charge, chi, svd_tol=tol)
        _check_bond(U, S, Vh, m, k, counts)
        assert np.allclose(np.sort(_np(be, S))[::-1], S0[:k], atol=1e-10 * S0[0])
        assert disc == pytest.approx(float((S0[k:] ** 2).sum()), rel=1e-6, abs=1e-20)
        rec = _np(be, U.scale(S, 1).to_dense()) @ _np(be, Vh.to_dense())
        assert np.abs(rec - (U0[:, :k] * S0[:k]) @ V0[:k]).max() <= 1e-10 * S0[0]
        for iso in (_np(be, U.to_dense()), _np(be, Vh.to_dense()).conj().T):
            assert np.abs(iso.conj().T @ iso - np.eye(k)).max() <= 1e-10


@pytest.mark.parametrize("spec", BACKENDS)
@pytest.mark.parametrize("method", ["svd", "gram"])
def test_truncated_svd_rank_at_the_tolerance_boundaries(spec, method):
    """One-element sectors with singular values 1, 1/2, 1/4, 1/8 and 0, all exact in binary:
    `zero_tol` drops values at the threshold and exact zeros, and `svd_tol` keeps the smallest
    rank whose dropped weight equals the budget."""
    be = get_backend(spec)
    vals = np.array([0.25, 0.0, 1.0, 0.125, 0.5])
    one = (1,) * len(vals)
    legs = (B.Leg((0, 1, 2, 3, 4), one, -1), B.Leg((-4, -3, -2, -1, 0), one, -1))
    m = B.BlockTensor(B.canonical(legs, 0), be.array(vals.astype(complex)))
    dense = _np(be, m.to_dense())
    w = np.sort(vals)[::-1] ** 2
    at_three = w[3:].sum() / w.sum()                 # dropped weight keeping three values
    for chi, svd_tol, zero_tol, expect in ((5, 0.0, 0.0, 5), (5, 0.0, 0.25, 2), (5, 0.0, 1e-3, 4),
                                           (1, 0.0, 0.0, 1), (5, at_three, 0.0, 3),
                                           (2, at_three, 0.0, 2)):
        U, S, Vh, disc = B.truncated_svd(m, chi, svd_tol=svd_tol, zero_tol=zero_tol,
                                         method=method)
        k, counts = _dense_truncation(dense, legs, 0, chi, svd_tol, zero_tol)
        assert k == expect
        _check_bond(U, S, Vh, m, k, counts)
        assert disc == pytest.approx(w[k:].sum(), abs=1e-15)


def test_truncated_svd_rejects_chi_or_bucket_below_one():
    m = B.random((BOND, BOND.flip()), 0, get_backend("numpy"), np.random.default_rng(0))
    with pytest.raises(ValueError, match="chi"):
        B.truncated_svd(m, 0)
    with pytest.raises(ValueError, match="bucket"):
        B.truncated_svd(m, 4, bucket=0)


@pytest.mark.parametrize("spec", BACKENDS)
def test_truncated_svd_of_a_zero_matrix_keeps_one_value(spec):
    be = get_backend(spec)
    s = B.canonical((BOND, BOND.flip()), 0)
    U, S, Vh, disc = B.truncated_svd(B.BlockTensor(s, be.zeros((s.size,))), 4, zero_tol=1e-12)
    assert len(S) == 1 and float(np.abs(_np(be, S)).max()) == 0.0 and disc == 0.0


@pytest.mark.parametrize("spec", BACKENDS)
def test_truncated_svd_driver_and_gesvd_retry(spec, monkeypatch):
    """The default SVD matches the dense one and uses gesvd on CUDA; a first attempt that does
    not converge is retried with gesvd."""
    be = get_backend(spec)
    m = _spectrum_matrix(be, np.random.default_rng(20), (BOND, BOND.flip()), decay=5.0)
    dense = _np(be, m.to_dense())
    S0 = np.linalg.svd(dense, compute_uv=False)
    lib, err = (np.linalg, np.linalg.LinAlgError) if spec == "numpy" else (
        torch.linalg, torch.linalg.LinAlgError)
    svd, drivers = lib.svd, []

    def spy(fail):
        def f(*a, **kw):
            drivers.append(kw.get("driver"))
            if fail and kw.get("driver") != "gesvd":
                raise err("SVD did not converge")
            return svd(*a, **kw)
        return f

    for fail, driver in ((False, None), (True, "gesvdj")):
        drivers.clear()
        monkeypatch.setattr(lib, "svd", spy(fail))
        U, S, Vh, disc = B.truncated_svd(m, 100, driver=driver)
        assert drivers and np.allclose(np.sort(_np(be, S))[::-1], S0, atol=1e-12)
        assert np.abs(_np(be, U.scale(S, 1).to_dense()) @ _np(be, Vh.to_dense()) - dense).max() \
            <= 1e-12
        if spec == "torch-cuda":
            assert set(drivers) == ({"gesvd"} if not fail else {"gesvdj", "gesvd"})


@pytest.mark.parametrize("spec", BACKENDS)
def test_truncation_bucketing_and_split(spec):
    """Bucketed sector sizes are multiples of the bucket, padded with exact zeros, and splitting
    U and S*Vh back to three-leg tensors reproduces the truncated two-site tensor."""
    be = get_backend(spec)
    rng = np.random.default_rng(19)
    theta = B.random((BOND, EVEN, ODD, BOND.flip()), 0, be, rng)
    f, rec = fuse(theta, ((0, 1), (2, 3)), be)
    U, S, Vh, _ = B.truncated_svd(f, 9, bucket=4, zero_tol=1e-13)
    bond = Vh.legs[0]
    assert all(d % 4 == 0 for d in bond.dims) and (np.abs(_np(be, S)) > 0).sum() == 9
    A = split(U, rec, be, legs=(0,))
    C = split(Vh.scale(S, 0), rec, be, legs=(1,))
    assert A.legs == (BOND, EVEN, bond.flip()) and C.legs == (bond, ODD, BOND.flip())
    got = B.tensordot(A, C, ((2,), (0,)))
    U1, S1, Vh1, _ = B.truncated_svd(f, 9)
    ref = B.tensordot(split(U1, rec, be, legs=(0,)), split(Vh1.scale(S1, 0), rec, be, legs=(1,)),
                      ((2,), (0,)))
    assert np.abs(_np(be, got.to_dense()) - _np(be, ref.to_dense())).max() <= 1e-12
