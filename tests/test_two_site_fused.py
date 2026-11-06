"""`apply_two_site_fused` must agree with `apply_two_site`, at the chain ends (unit bonds) and in
the bulk, with unequal physical dimensions, on every CPU backend."""
import numpy as np
import pytest

from tnkit._dmrg import apply_two_site, apply_two_site_fused, fused_two_site
from tnkit.backend import get_backend

SHAPES = [  # (a, wl, wm, wr, d1, d2, b)
    (1, 1, 5, 5, 2, 2, 2),       # left end
    (4, 5, 5, 1, 2, 2, 1),       # right end
    (7, 20, 20, 20, 2, 3, 9),    # bulk, unequal sites
]


@pytest.mark.parametrize("name", ["numpy-real", "numpy", "torch-cpu-real"])
@pytest.mark.parametrize("a,wl,wm,wr,d1,d2,b", SHAPES)
def test_fused_matches_apply_two_site(name, a, wl, wm, wr, d1, d2, b):
    if name.startswith("torch"):
        pytest.importorskip("torch")
    be = get_backend(name)
    rng = np.random.default_rng(0)
    cplx = not name.endswith("-real")

    def r(*shape):
        x = rng.standard_normal(shape)
        return be.array(x + 1j * rng.standard_normal(shape) if cplx else x)

    L, R = r(a, wl, a), r(b, wr, b)
    W1, W2 = r(wl, wm, d1, d1), r(wm, wr, d2, d2)
    theta = r(a, d1, d2, b)
    ref = be.to_numpy(apply_two_site(theta, L, W1, W2, R, be))
    got = be.to_numpy(apply_two_site_fused(theta, *fused_two_site(L, W1, W2, R, be)))
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()
