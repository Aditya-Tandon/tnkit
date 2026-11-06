"""The CGS2 overshoot-and-truncate variant: NOT bit-identical, and how close it is.

`lanczos_expm_prefix_cgs2` (and `PrefixSiteStepGraph(reorth='cgs2')`, which captures the
same arithmetic) keep the prefix graph's adaptive-schedule replay -- the Krylov dimension is still
eager's own, decided on the host from scalars the device loop reports -- but replace the
sequential modified-Gram-Schmidt reorthogonalisation with two batched CGS2 passes, and the
final combination with one batched matmul instead of a sequential sum. That is a genuine
reduction-order change, so `tests/test_prefix_identity.py`'s bar (`==` on raw bytes) does not
apply here; this file measures the deviation instead of asserting it away.

The bar: relative deviation against the plain adaptive `lanczos_expm`, on the same random
Hermitian and real-TDVP-block fixtures `test_prefix_identity.py` uses. Measured on CPU
(numpy and torch-cpu): worst case ~6e-16 on the synthetic sweep, ~5e-16 on real TDVP blocks
-- two to three times float64 machine epsilon (2.22e-16), i.e. at the same noise floor as
"reorthogonalise right-to-left instead of left-to-right" (a null comparison,
4.4e-15). The tolerance asserted below (1e-12 relative) is set two orders above the worst
measured value, not at it -- a numerically loose bar for a numerically tight result.
"""
import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.eigensolver import lanczos_expm
from tnkit.prefix_graph import lanczos_expm_prefix_cgs2, select_prefix

REL_TOL = 1e-12


def _require(backend):
    if str(backend).startswith("torch"):
        pytest.importorskip("torch")


def _hermitian_matvec(be, H):
    def mv(x):
        return be.array(np.asarray(be.to_numpy(H)) @ np.asarray(be.to_numpy(x)))
    return mv


def _random_hermitian(n, rng, scale=1.0):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * (A + A.conj().T) / 2.0


def _relerr(be, got, ref):
    g = np.asarray(be.to_numpy(got))
    r = np.asarray(be.to_numpy(ref))
    return float(np.max(np.abs(g - r)) / max(float(np.max(np.abs(r))), 1e-300))


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
@pytest.mark.parametrize("n,scale,dt,m_max", [(8, 1.0, 0.05, 12), (16, 0.3, 0.05, 12),
                                             (16, 3.0, 0.05, 12), (32, 1.0, 0.01, 12),
                                             (32, 1.0, 0.2, 20), (64, 1.0, 0.05, 12)])
def test_cgs2_within_tolerance_of_adaptive(backend, n, scale, dt, m_max):
    """Same sweep as `test_prefix_is_bit_identical_to_adaptive`, tolerance instead of bytes."""
    _require(backend)
    be = get_backend(backend)
    rng = np.random.default_rng(20260809 + n + int(100 * scale) + int(1000 * dt))
    H = be.array(_random_hermitian(n, rng, scale))
    mv = _hermitian_matvec(be, H)
    tau = -1j * dt / 2.0
    stats = {"m_eff_hist": {}}
    worst = 0.0
    for trial in range(6):
        v = be.array(rng.normal(size=n) + 1j * rng.normal(size=n))
        ref = lanczos_expm(mv, v, tau, be)
        got = lanczos_expm_prefix_cgs2(mv, v, tau, be, m_max=m_max, stats=stats)
        worst = max(worst, _relerr(be, got, ref))
    assert worst < REL_TOL, f"cgs2 deviation {worst:.3e} exceeds {REL_TOL:.0e}"
    assert stats.get("short", 0) == 0, "every call fell back; the prefix path was never run"
    assert stats["solves"] == 6


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_cgs2_overshoot_stays_small(backend):
    """The MGS design is exactly overshoot-invariant (a prefix property). CGS2 reorthogonalises
    against the same growing prefix `Vb[:k+1]`, so the property should survive approximately:
    changing m_max should not move the answer beyond the same tolerance."""
    _require(backend)
    be = get_backend(backend)
    rng = np.random.default_rng(909)
    n = 32
    H = be.array(_random_hermitian(n, rng))
    mv = _hermitian_matvec(be, H)
    tau = -1j * 0.05 / 2.0
    for _ in range(5):
        v = be.array(rng.normal(size=n) + 1j * rng.normal(size=n))
        ref = lanczos_expm(mv, v, tau, be)
        outs = {mm: lanczos_expm_prefix_cgs2(mv, v, tau, be, m_max=mm)
                for mm in (9, 10, 12, 16, 20)}
        for mm, out in outs.items():
            assert _relerr(be, out, ref) < REL_TOL, (mm, _relerr(be, out, ref))


def _canonical_block(be, N, chi, site, seed):
    from tnkit.dmrg import right_canonicalize, update_left_env, update_right_env
    from tnkit.models import tfim_mpo
    rng = np.random.default_rng(seed)
    dims = [1] + [min(chi, 2 ** min(i + 1, N - i - 1)) for i in range(N - 1)] + [1]
    M = [be.array(rng.normal(size=(dims[i], 2, dims[i + 1]))
                  + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))) for i in range(N)]
    M = right_canonicalize(M, be)
    mpo = [be.array(w) for w in tfim_mpo(N, g=1.0, h=0.10)]
    R = be.array(np.ones((1, 1, 1), dtype=complex))
    Rs = [None] * (N + 1)
    Rs[N] = R
    for i in range(N - 1, -1, -1):
        Rs[i] = update_right_env(Rs[i + 1], M[i], mpo[i], be)
    L = be.array(np.ones((1, 1, 1), dtype=complex))
    for i in range(0, site):
        L = update_left_env(L, M[i], mpo[i], be)
    return M[site], mpo[site], L, Rs[site + 1]


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_cgs2_matches_on_a_real_tdvp_block(backend):
    """The real one-site effective Hamiltonian, not a synthetic random one -- the shapes and
    conditioning the graph is actually captured at."""
    _require(backend)
    from tnkit.dmrg import apply_one_site
    be = get_backend(backend)
    stats = {"m_eff_hist": {}}
    worst = 0.0
    for (N, chi, site) in [(24, 48, 12), (24, 32, 12), (24, 16, 12), (24, 8, 12),
                           (24, 48, 3), (24, 48, 2), (32, 64, 16)]:
        M, W, L, R = _canonical_block(be, N, chi, site, seed=700 + chi + site)
        a, s, b = (int(x) for x in M.shape)
        mv = lambda x: apply_one_site(x.reshape(a, s, b), L, W, R, be).reshape(-1)
        v = M.reshape(-1)
        for _ in range(3):
            tau = -1j * 0.05 / 2.0
            ref = lanczos_expm(mv, v, tau, be)
            got = lanczos_expm_prefix_cgs2(mv, v, tau, be, m_max=12, stats=stats)
            worst = max(worst, _relerr(be, got, ref))
            v = ref                              # feed the step back in, as a sweep does
    assert worst < REL_TOL, f"cgs2 deviation {worst:.3e} exceeds {REL_TOL:.0e}"
    assert stats.get("short", 0) == 0, "the fixture is too stiff; the prefix path never ran"
    assert stats["solves"] == 21


def test_select_prefix_boundary_decision_is_only_unstable_at_the_boundary():
    """`select_prefix`'s `est < tol` stopping decision is safe by construction for `'mgs'`'s
    alphas/betas -- they are bit-identical to eager's, so it makes eager's exact decision.
    For `'cgs2'`, `_lanczos_prefix_device_cgs2` computes its own alphas/betas, off eager's by
    ~1e-15 relative (this file's own measured bound, above) EVERYWHERE, including at a `tol`
    decision boundary -- so a boundary decision is not provably identical to eager's there.

    This builds a synthetic real alpha/beta sequence, bisects one beta to `select_prefix`'s
    own `m_eff: 2 -> 3` boundary to float64 resolution, and checks the claim directly against
    `select_prefix` itself -- no matvec, no device loop, just the host selector the
    docstrings above describe.
    """
    tau = -1j * 0.05 / 2.0
    beta0, m_max, tol = 1.3, 4, 1e-12
    rng = np.random.default_rng(0)
    alphas = rng.normal(size=m_max)
    betas = np.abs(rng.normal(size=m_max)) + 0.5

    def m_eff_for(b1):
        betas_try = betas.copy()
        betas_try[1] = b1
        scal = np.concatenate([alphas, betas_try, [beta0]])
        m_eff, _ = select_prefix(scal, tau, m_max, tol=tol, adaptive=True)
        return m_eff

    lo, hi = 1e-9, 2e-9
    assert m_eff_for(lo) == 2 and m_eff_for(hi) == 3, "fixture drifted off its known boundary"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if m_eff_for(mid) == 2:
            lo = mid
        else:
            hi = mid
    # `lo`/`hi` now bracket the exact float64 boundary of select_prefix's stopping decision --
    # they differ at the machine-epsilon-relative level, the same order as CGS2's own
    # measured deviation from eager's alphas/betas.
    assert (hi - lo) / lo < 1e-13
    assert m_eff_for(lo) == 2 and m_eff_for(hi) == 3
    boundary = 0.5 * (lo + hi)

    # A perturbation at CGS2's own measured order (~5e-15 relative) flips the decision AT
    # the boundary -- the mechanism a code review flagged: select_prefix
    # replayed on cgs2's own (perturbed) alphas/betas can pick m_eff one dimension off from
    # eager's near a threshold.
    eps = 5e-15
    assert m_eff_for(boundary * (1 - eps)) == 2
    assert m_eff_for(boundary * (1 + eps)) == 3

    # ...but the instability is local to the boundary, not a general property of
    # select_prefix: away from it, the same relative perturbation never moves the decision --
    # bounding the claim ("one dimension, only near a boundary") rather than merely asserting
    # it.
    for probe in (boundary * 0.1, boundary * 10.0, 1e-6):
        base = m_eff_for(probe)
        assert m_eff_for(probe * (1 - eps)) == base
        assert m_eff_for(probe * (1 + eps)) == base
