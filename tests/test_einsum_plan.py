"""`NumpyBackend.einsum`'s compiled plans must return exactly what `np.einsum` returns:
on a warm cache, on operands the install check never saw, and on non-contiguous views."""
import numpy as np
import pytest

from tnkit import backend as tnb
from tnkit.backend import get_backend


# The MPS contractions of the evolution and observable paths, at bond dimensions that occur in
# a real sweep: the saturated bulk (30), the ramp near a chain end (1, 2, 4), and the MPO's
# own bond (3). Boundary sites are where a wrong axis permutation is most likely to survive a
# careless check, because a length-1 axis hides an ordering mistake.
CASES = [
    ("aky,ysb->aksb", [(30, 3, 30), (30, 2, 30)]),          # apply_one_site
    ("klst,aktb->alsb", [(3, 3, 2, 2), (30, 3, 2, 30)]),
    ("bly,alsy->asb", [(30, 3, 30), (30, 3, 2, 30)]),
    ("aky,ysb->aksb", [(1, 1, 1), (1, 2, 2)]),              # the left boundary
    ("bly,alsy->asb", [(4, 3, 4), (1, 3, 2, 4)]),
    ("aky,yb->akb", [(30, 3, 30), (30, 30)]),               # apply_zero_site
    ("bkz,akz->ab", [(30, 3, 30), (30, 3, 30)]),
    ("pwq,psd->wqsd", [(30, 3, 30), (30, 2, 30)]),          # update_left_env
    ("wqsd,wrsy->qdry", [(3, 30, 2, 30), (3, 3, 2, 2)]),
    ("qdry,qye->dre", [(30, 30, 3, 2), (30, 2, 30)]),
    ("xry,usx->ryus", [(30, 3, 30), (30, 2, 30)]),          # update_right_env
    ("ryus,wrsp->yuwp", [(3, 30, 30, 2), (3, 3, 2, 2)]),
    ("yuwp,vpy->uwv", [(30, 30, 3, 2), (30, 2, 30)]),
    ("pwq,qijb->pwijb", [(30, 3, 30), (30, 2, 2, 30)]),     # apply_two_site
    ("rb,bsc->rsc", [(30, 30), (30, 2, 30)]),               # gauge transport
    ("csa,ar->csr", [(30, 2, 30), (30, 30)]),
    ("asb,bc->asc", [(2, 2, 4), (4, 4)]),
    ("ab,bsc->asc", [(30, 30), (30, 2, 30)]),
    ("asb->sba", [(30, 2, 30)]),                            # pure permutes
    ("sbr->rsb", [(2, 30, 30)]),
    ("ra->ar", [(30, 30)]),
    ("asc,ctb->astb", [(30, 2, 30), (30, 2, 30)]),          # two-site theta
    ("xyij,aijb->axyb", [(2, 2, 2, 2), (30, 2, 2, 30)]),    # a two-site gate
    ("st,atb->asb", [(2, 2), (30, 2, 30)]),
    ("asx,xb->asb", [(30, 2, 4), (4, 30)]),
    # cases the compiler must DECLINE rather than get wrong
    ("ab,asc,bsd->cd", [(30, 30), (30, 2, 30), (30, 2, 30)]),
    ("asb->ab", [(30, 2, 30)]),
    ("aa->a", [(30, 30)]),
    ("as,sb->asb", [(4, 3), (3, 4)]),                       # no summed index: an outer product
]


def _key(sub, ops):
    return (sub, tuple(o.shape for o in ops), tuple(o.dtype for o in ops))


def _operands(shapes, dtype, rng):
    out = []
    for s in shapes:
        x = rng.standard_normal(s)
        if np.issubdtype(dtype, np.complexfloating):
            x = x + 1j * rng.standard_normal(s)
        out.append(np.ascontiguousarray(x, dtype=dtype))
    return out


@pytest.mark.parametrize("sub,shapes", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_planned_einsum_is_bit_identical(sub, shapes, dtype):
    be = get_backend("numpy")
    rng = np.random.default_rng(abs(hash(sub)) % (2 ** 32))
    for _ in range(3):                       # fresh operands, same key: exercises the warm cache
        ops = _operands(shapes, dtype, rng)
        ref = np.einsum(sub, *ops, optimize=True)
        got = be.einsum(sub, *ops)
        assert got.shape == ref.shape
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref), f"{sub} {shapes} {dtype}"


@pytest.mark.parametrize("warm_seed", [1000, 1001, 1002])
def test_a_plan_built_from_one_sample_holds_for_every_other(warm_seed):
    """The install check sees ONE set of operands; the plan is then used for every set that
    ever shares those shapes. On small or degenerate shapes the two candidate GEMM orderings
    can round alike, so a sample that does not discriminate installs whichever was tried
    first. Each key is warmed from a different sample and then hammered with sixty more.
    """
    be = get_backend("numpy")
    hot = [c for c in CASES if len(c[1]) == 2][:12]
    for sub, shapes in hot:
        for dtype in (np.complex128, np.float64):
            warm = _operands(shapes, dtype, np.random.default_rng(warm_seed))
            tnb._EINSUM_PLANS.pop(_key(sub, warm), None)
            be.einsum(sub, *warm)
            rng = np.random.default_rng(warm_seed + 7)
            for _ in range(60):
                ops = _operands(shapes, dtype, rng)
                assert np.array_equal(be.einsum(sub, *ops),
                                      np.einsum(sub, *ops, optimize=True)), \
                    f"{sub} {shapes} {dtype} after warming from seed {warm_seed}"


def test_a_warm_cache_returns_what_a_cold_one_did():
    """The value AND the buffer behaviour must not depend on whether the plan was already
    built. A run whose output moved on the second call would be irreproducible against
    itself -- and the downstream QR is sensitive to the buffer even when every value in it
    is identical."""
    be = get_backend("numpy")
    rng = np.random.default_rng(11)
    sub = "aky,ysb->aksb"
    ops = _operands([(7, 3, 7), (7, 2, 5)], np.complex128, rng)
    tnb._EINSUM_PLANS.pop(_key(sub, ops), None)
    cold = be.einsum(sub, *ops)
    warm = be.einsum(sub, *ops)
    assert np.array_equal(cold, warm)
    assert cold.strides == warm.strides
    assert cold.flags.c_contiguous == warm.flags.c_contiguous


@pytest.fixture
def cold_plan_cache(monkeypatch):
    """An empty plan cache for one test; the real one is restored afterwards."""
    monkeypatch.setattr(tnb, "_EINSUM_PLANS", {})


def test_non_contiguous_operands_take_the_same_plan(cold_plan_cache):
    """Plans are keyed without strides, so a cache warmed on contiguous operands of the same
    shapes (as an earlier test does) replays a different GEMM summation order on views, which
    differs from `np.einsum` by round-off. From a cold cache the plan is built on the views
    themselves; values are compared to round-off, not bitwise."""
    be = get_backend("numpy")
    rng = np.random.default_rng(5)
    a = rng.standard_normal((30, 3, 30)).transpose(2, 1, 0)
    b = rng.standard_normal((30, 2, 30))[::-1]
    ref = np.einsum("aky,ysb->aksb", a, b, optimize=True)
    np.testing.assert_allclose(be.einsum("aky,ysb->aksb", a, b), ref, rtol=1e-14, atol=0)


def test_the_compiler_declines_what_it_cannot_do():
    """A trace, a partial sum, an outer product and a three-operand contraction are all
    delegated to `np.einsum` rather than planned. The point is not that they are slow -- it
    is that a compiler which silently accepted them would be guessing."""
    rng = np.random.default_rng(2)
    for sub, shapes in [("ab,asc,bsd->cd", [(4, 4), (4, 2, 4), (4, 2, 4)]),
                        ("asb->ab", [(4, 2, 4)]),
                        ("aa->a", [(4, 4)]),
                        ("as,sb->asb", [(4, 3), (3, 4)])]:
        ops = _operands(shapes, np.complex128, rng)
        assert tnb._build_einsum_plan(sub, tuple(o.shape for o in ops)) is None


def test_the_plan_cache_is_capped_and_stays_exact(monkeypatch):
    """Past the cap the cache is flushed, not frozen: later keys are still planned and results
    stay exact. Many distinct shapes would otherwise grow it without bound."""
    monkeypatch.setattr(tnb, "_EINSUM_PLANS", {})
    monkeypatch.setattr(tnb, "_EINSUM_PLAN_CAP", 8)
    be = get_backend("numpy")
    rng = np.random.default_rng(13)
    for chi in range(2, 40):
        ops = _operands([(chi, 3, chi), (chi, 2, chi)], np.complex128, rng)
        assert np.array_equal(be.einsum("pwq,psd->wqsd", *ops),
                              np.einsum("pwq,psd->wqsd", *ops, optimize=True))
        assert len(tnb._EINSUM_PLANS) <= 8
    assert _key("pwq,psd->wqsd", ops) in tnb._EINSUM_PLANS


def test_real_and_complex_operands_of_one_shape_are_separate_plans():
    be = get_backend("numpy")
    rng = np.random.default_rng(17)
    sub, shapes = "aky,ysb->aksb", [(6, 3, 6), (6, 2, 5)]
    keys = set()
    for dtype in (np.float64, np.complex128, np.float64, np.complex64):
        ops = _operands(shapes, dtype, rng)
        got = be.einsum(sub, *ops)
        ref = np.einsum(sub, *ops, optimize=True)
        assert got.dtype == ref.dtype and np.array_equal(got, ref)
        keys.add(_key(sub, ops))
    assert len(keys) == 3 and keys <= set(tnb._EINSUM_PLANS)
    mixed = [_operands(shapes[:1], np.float64, rng)[0],
             _operands(shapes[1:], np.complex128, rng)[0]]
    assert np.array_equal(be.einsum(sub, *mixed), np.einsum(sub, *mixed, optimize=True))


def test_large_operands_are_left_to_numpy():
    """Installing a plan costs several einsum calls and gains nothing once the GEMM dominates."""
    be = get_backend("numpy")
    ops = _operands([(120, 3, 120), (120, 2, 120)], np.complex128, np.random.default_rng(19))
    assert max(o.size for o in ops) > tnb._EINSUM_PLAN_MAX_SIZE
    sub = "pwq,psd->wqsd"
    assert np.array_equal(be.einsum(sub, *ops), np.einsum(sub, *ops, optimize=True))
    assert tnb._EINSUM_PLANS[_key(sub, ops)] is tnb._EINSUM_FALLBACK


def test_the_torch_backend_is_untouched():
    """The compiled path is a NumPy-only construction: torch resolves its own contraction
    order internally and never reaches `einsum_planned`. This pins that, so a later edit
    cannot quietly route a torch tensor through a NumPy plan."""
    torch = pytest.importorskip("torch")
    be = get_backend("torch-cpu")
    rng = np.random.default_rng(7)
    a = torch.tensor(rng.standard_normal((6, 3, 6)) + 0j)
    b = torch.tensor(rng.standard_normal((6, 2, 6)) + 0j)
    got = be.einsum("aky,ysb->aksb", a, b)
    ref = torch.einsum("aky,ysb->aksb", a, b)
    assert torch.equal(got, ref)


def test_plans_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(tnb, "_EINSUM_PLAN_ENABLED", False)
    rng = np.random.default_rng(3)
    ops = _operands([(5, 3, 5), (5, 2, 7)], np.complex128, rng)
    key = _key("aky,ysb->aksb", ops)
    tnb._EINSUM_PLANS.pop(key, None)
    got = get_backend("numpy").einsum(key[0], *ops)
    assert key not in tnb._EINSUM_PLANS
    assert np.array_equal(got, np.einsum(key[0], *ops, optimize=True))
