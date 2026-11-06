"""Prefix-graph unit test, part 1: the ARITHMETIC contract, with no GPU and no graph.

WHAT THIS PROVES, AND WHY IT IS SEPARABLE. The prefix graph's claim has exactly two halves:

  (i)  overshoot-and-truncate retains EAGER'S OWN floating-point arithmetic -- the first
       m_eff Krylov vectors, the stopping decision, the small exponential and the final
       combination are the same operations on the same operands as `lanczos_expm`;
  (ii) replaying a captured CUDA graph reproduces its eager twin bit-for-bit.

Half (ii) is not in doubt: two GPU replications measured 0.000e+00, at sigma=10 and at
sigma=30, and it is re-measured on the GPU here anyway. Half (i) is the NEW claim, it is where a bug would live, and it is
device-independent -- so it is tested here, on CPU, exhaustively and for free.

The bar is `==`, not `allclose`. Anything else would be conceding the exact point the GPU
replication established: on a production-length leg an agreement of 1e-11 per step is amplified to 1e-3
by t=204, so "close" is not a weaker version of "identical", it is a different result.
"""
import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.eigensolver import lanczos_expm
from tnkit.prefix_graph import lanczos_expm_prefix


def _require(backend):
    """Skip a torch leg when torch is not installed, instead of erroring on the import.

    The library's packaging claim is that the core needs only numpy and scipy; the
    suite should be able to DEMONSTRATE that rather than blow up on collection. The
    numpy legs of every test below still run in a torch-free environment -- only the
    torch-cpu legs skip.
    """
    if str(backend).startswith("torch"):
        pytest.importorskip("torch")

def _hermitian_matvec(be, H):
    def mv(x):
        return be.array(np.asarray(be.to_numpy(H)) @ np.asarray(be.to_numpy(x)))
    return mv


def _random_hermitian(n, rng, scale=1.0):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * (A + A.conj().T) / 2.0


def _bits(be, x):
    """Raw bytes of the array -- the only comparison that is not a tolerance in disguise."""
    return np.asarray(be.to_numpy(x)).tobytes()


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
@pytest.mark.parametrize("n,scale,dt,m_max", [(8, 1.0, 0.05, 12), (16, 0.3, 0.05, 12),
                                             (16, 3.0, 0.05, 12), (32, 1.0, 0.01, 12),
                                             (32, 1.0, 0.2, 20), (64, 1.0, 0.05, 12)])
def test_prefix_is_bit_identical_to_adaptive(backend, n, scale, dt, m_max):
    """Same answer, same bits, over a sweep that moves the Krylov demand around.

    `m_max` is carried in the sweep rather than fixed, because the stiff corner
    (n=32, |tau| ||H|| large) genuinely needs a deeper space than 12 and pinning m_max there
    would silently convert the test into the eager fallback comparing `lanczos_expm` with
    itself -- which is what the `short == 0` assertion below exists to forbid.

    `scale` and `dt` are varied together because they are the two things the adaptive stop
    rule is a function of: the Saad estimate goes as |tau|^j beta_1...beta_j/(j-1)!, so a
    stiffer operator or a longer step pushes m_eff up. The sweep therefore exercises
    m_eff well below m_max (the ordinary case, where the overshoot is real) and m_eff close
    to it.
    """
    _require(backend)
    be = get_backend(backend)
    rng = np.random.default_rng(20260809 + n + int(100 * scale) + int(1000 * dt))
    H = be.array(_random_hermitian(n, rng, scale))
    mv = _hermitian_matvec(be, H)
    tau = -1j * dt / 2.0
    stats = {"m_eff_hist": {}}
    for trial in range(6):
        v = be.array(rng.normal(size=n) + 1j * rng.normal(size=n))
        ref = lanczos_expm(mv, v, tau, be)
        got = lanczos_expm_prefix(mv, v, tau, be, m_max=m_max, stats=stats)
        assert _bits(be, got) == _bits(be, ref), (
            f"prefix != adaptive bitwise ({backend}, n={n}, scale={scale}, dt={dt}, "
            f"trial={trial}); max abs diff "
            f"{np.max(np.abs(np.asarray(be.to_numpy(got)) - np.asarray(be.to_numpy(ref))))}")
    # A test that passes only because every call took the eager fallback would prove nothing
    # at all -- it would be comparing `lanczos_expm` with itself. The GPU unit gate's first run
    # did exactly that (72 fallbacks, 0 comparisons), which is why this assertion exists.
    assert stats.get("short", 0) == 0, "every call fell back; the prefix path was never run"
    assert stats["solves"] == 6


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_prefix_selects_the_same_krylov_dimension(backend):
    """The truncation length is eager's, not a re-derivation that happens to agree.

    Compared through `info['krylov_dim']`, which is the adaptive solver's own report of where
    it stopped. If the two rules ever disagreed the vectors could still match while the answer
    did not, so this is checked separately from the bits.
    """
    _require(backend)
    be = get_backend(backend)
    rng = np.random.default_rng(4242)
    n = 24
    H = be.array(_random_hermitian(n, rng))
    mv = _hermitian_matvec(be, H)
    for dt in (0.005, 0.05, 0.5):
        tau = -1j * dt / 2.0
        for _ in range(5):
            v = be.array(rng.normal(size=n) + 1j * rng.normal(size=n))
            i_ref, i_got = {}, {}
            lanczos_expm(mv, v, tau, be, info=i_ref)
            lanczos_expm_prefix(mv, v, tau, be, m_max=14, info=i_got)
            assert i_got["krylov_dim"] == i_ref["krylov_dim"], (dt, i_ref, i_got)


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_overshoot_is_invisible(backend):
    """m_max is a COST knob, not a physics knob: the answer must not depend on it at all.

    This is the prefix property stated as a test. If running further Lanczos iterations after
    the truncation point could change the retained ones -- through a reorthogonalisation that
    reaches forwards, an in-place buffer write, an accumulation that touches the tail -- this
    is the test that catches it, and it is the single most load-bearing assertion for the prefix graph.
    """
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
        bits = {mm: _bits(be, lanczos_expm_prefix(mv, v, tau, be, m_max=mm))
                for mm in (9, 10, 12, 16, 20)}
        assert len(set(bits.values())) == 1, "answer moved with m_max"
        assert next(iter(bits.values())) == _bits(be, ref)


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_short_prefix_falls_back_exactly(backend):
    """When m_max is too small the step is re-run eagerly -- never truncated early.

    A silent early truncation is the failure mode that would look like success: it would
    still produce a plausible evolved state, and the whole point is that plausible is not
    the standard. `m_max=3` here is far below the measured demand (5-8), so the fallback path
    is the one under test, and its output must still be eager's bits exactly.
    """
    _require(backend)
    be = get_backend(backend)
    rng = np.random.default_rng(77)
    n = 32
    H = be.array(_random_hermitian(n, rng))
    mv = _hermitian_matvec(be, H)
    tau = -1j * 0.05 / 2.0
    stats = {"m_eff_hist": {}}
    hits = 0
    for _ in range(6):
        v = be.array(rng.normal(size=n) + 1j * rng.normal(size=n))
        ref = lanczos_expm(mv, v, tau, be)
        got = lanczos_expm_prefix(mv, v, tau, be, m_max=3, stats=stats)
        assert _bits(be, got) == _bits(be, ref)
        hits += 1
    assert stats.get("short", 0) == hits, "the fallback did not actually engage"


def _canonical_block(be, N, chi, site, seed):
    """A genuine 1TDVP block: right-canonical MPS, real TFIM MPO, real environments.

    A random L/R/W fixture is not a weaker version of this -- it is a different operator.
    ||H_eff|| for unstructured N(0,1) environments grows like sqrt(a*D*a), so at chi=48 the
    Krylov demand exceeds any sane m_max and every call takes the eager fallback: the test
    then compares `lanczos_expm` with itself and passes vacuously. That is exactly what the
    first version of this test did, and what the `short == 0` assertion now forbids. The
    operator scale a real block exponential sees is set by canonicality (||L|| ~ 1) and by the
    MPO being a sum of O(1) local terms.
    """
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
def test_matches_on_a_real_tdvp_block(backend):
    """The same equality on the actual TDVP one-site operator, not a synthetic Hermitian.

    Random Hermitians have flat spectra; the real one-site effective Hamiltonian is a
    projected MPO with a structured spectrum and a very different Krylov demand, and the
    shapes are the (a, s, b) ones the graph will be captured at -- including sites on the
    bond-dimension ramp where a*s and b are far apart.
    """
    _require(backend)
    from tnkit.dmrg import apply_one_site
    be = get_backend(backend)
    stats = {"m_eff_hist": {}}
    for (N, chi, site) in [(24, 48, 12), (24, 32, 12), (24, 16, 12), (24, 8, 12),
                           (24, 48, 3), (24, 48, 2), (32, 64, 16)]:
        M, W, L, R = _canonical_block(be, N, chi, site, seed=700 + chi + site)
        a, s, b = (int(x) for x in M.shape)
        mv = lambda x: apply_one_site(x.reshape(a, s, b), L, W, R, be).reshape(-1)
        v = M.reshape(-1)
        for _ in range(3):
            tau = -1j * 0.05 / 2.0
            ref = lanczos_expm(mv, v, tau, be)
            got = lanczos_expm_prefix(mv, v, tau, be, m_max=12, stats=stats)
            assert _bits(be, got) == _bits(be, ref), (N, chi, site)
            v = ref                              # feed the step back in, as a sweep does
    assert stats.get("short", 0) == 0, "the fixture is too stiff; the prefix path never ran"
    assert stats["solves"] == 21


# --------------------------------------------------------------------------- end-to-end, CPU
def test_whole_sweep_is_bit_identical(monkeypatch):
    """A full 1TDVP evolution, eager vs prefix, compared as BYTES on every tensor.

    The unit tests above pin the solver. This pins the composition: the QR that moves the
    orthogonality centre, the environment updates and the bond absorption are the same calls
    in the same order either way, so substituting the solver must leave the whole sweep
    invariant -- and if any of that composition were subtly reordered, a single site-step
    would hide it while 40 sweeps over 20 sites would not.

    Run by substituting the solver rather than by capturing a graph, so it needs no GPU. The
    graph body in `prefix_graph.PrefixSiteStepGraph` performs exactly these operations; what
    the GPU test adds is only that replaying them reproduces them, which has already been
    measured at 0.000e+00.
    """
    pytest.importorskip("torch")          # torch-cpu only: skip, do not error
    import tnkit.tdvp as T
    from tnkit.models import tfim_mpo
    from tnkit.prefix_graph import lanczos_expm_prefix

    be = get_backend("torch-cpu")
    n, chi = 20, 12
    rng = np.random.default_rng(2026)
    mpo = tfim_mpo(n, g=1.0, h=0.10)
    dims = [1] + [min(chi, 2 ** min(i + 1, n - i - 1), chi) for i in range(n - 1)] + [1]
    mps0 = [(rng.normal(size=(dims[i], 2, dims[i + 1]))
             + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))) / 10.0 for i in range(n)]

    def run(use_prefix):
        if use_prefix:
            monkeypatch.setattr(
                T, "_expm_apply",
                lambda mv, x, tau, b, spec, info, krylov_m=30: lanczos_expm_prefix(
                    mv, x, tau, b, m_max=12, info=info))
        else:
            monkeypatch.undo()
        out, _ = T.tdvp1_evolve([m.copy() for m in mps0], mpo, 0.05, 8, backend="torch-cpu")
        return [np.asarray(x).tobytes() for x in out]

    ref = run(False)
    got = run(True)
    monkeypatch.undo()
    assert len(ref) == len(got)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert r == g, f"site {i} differs after 8 sweeps"
