"""Two-site DMRG and TDVP on symmetric tensors (`tnkit.symmetric_backend`) against exact results.

  * `symmetric_mpo` rebuilds the dense MPO exactly, labels the two-flavour channels as charge and
    isospin multiplets, and raises on channels that mix irreps or states;
  * the merged two-flavour site holds the isospin doublet (c+_u|0>, c+_d|0>) in the standard basis;
  * ground-state energies at full rank equal exact diagonalisation: spin-1/2 Heisenberg chains
    under SU(2), the two-flavour Schwinger model under U(1) x SU(2) in several (Q, I) sectors;
  * U(1) x U(1) through symmetric tensors gives the energy of block DMRG from the same start;
  * two-site TDVP at full rank follows exact evolution, from I = 0 and I > 0 states, and
    conserves I^2 when truncating; DMRG restarts from its own I > 0 result;
  * subspace expansion on symmetric tensors spans what it spans on dense ones, and the DMRG
    split with no expansion term cuts where `truncated_svd` does;
  * the MPO stays exact on long chains; torch on the CPU agrees with numpy;
  * inner products, expectation values and variances of block and symmetric MPSs equal the dense
    ones, per state of a multiplet; an MPS survives `save_mps` and `load_mps`, also onto torch.
"""
import io
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import expm_multiply

from tnkit.backend import NumpyBackend, get_backend
from tnkit.block_backend import BlockBackend, random_block_mps, to_block_mpo
from tnkit.block_backend import to_dense_mps as block_to_dense_mps
from tnkit.dmrg import dmrg
from tnkit.dmrg_excited import (mpo_expectation, mpo_sq_expectation, mpo_variance, mps_inner,
                                two_site_variance)
from tnkit.models_lgt2f import (charge_sparse_diag, isospin_sparse, merge_pairs,
                                schwinger2f_mpo, schwinger2f_sparse, site_spaces_2f)
from tnkit.opsum import OpSum, compile_mpo, spin_half, to_dense
from tnkit.symmetric_backend import (SiteSpace, SymmetricBackend, edge_environments, load_mps,
                                     random_symmetric_mps, save_mps, spin_half_site,
                                     symmetric_mpo, to_dense_mps)
from tnkit.symmetric_tensor import Space, concat, from_blocks, from_dense, random
from tnkit.symmetry import SU2, U1, ProductSymmetry
from tnkit.tdvp import tdvp2_evolve

X, MU = 0.6, 0.3
TIGHT = dict(n_sweeps=20, tol=1e-14, eig_tol=1e-12)


def heisenberg_mpo(N, form="pm"):
    os = OpSum()
    for i in range(N - 1):
        if form == "pm":
            os += (0.5, "Sp", i, "Sm", i + 1)
            os += (0.5, "Sm", i, "Sp", i + 1)
        else:
            os += (0.25, "X", i, "X", i + 1)
            os += (0.25, "Y", i, "Y", i + 1)
        os += (0.25, "Z", i, "Z", i + 1)
    return [np.ascontiguousarray(w.real) for w in compile_mpo(os, [spin_half()] * N)[0]]


def heisenberg_sparse(N):
    Sp = sp.csr_matrix([[0.0, 1.0], [0.0, 0.0]])
    Z = sp.diags([0.5, -0.5])

    def op(o, i):
        return sp.kron(sp.kron(sp.identity(2 ** i), o), sp.identity(2 ** (N - i - 1)), "csr")
    return sum(0.5 * (op(Sp, i) @ op(Sp.T, i + 1) + op(Sp.T, i) @ op(Sp, i + 1))
               + op(Z, i) @ op(Z, i + 1) for i in range(N - 1))


def dense_operator(mpo, sites):
    """The operator of a symmetric MPO in the models' basis."""
    out = []
    for W, s in zip(mpo, sites):
        U = np.asarray(s.basis)
        out.append(np.einsum("sa,wvst,tb->wvab", U.conj(), np.asarray(W.to_dense()), U))
    return to_dense(out)


def state_vector(mps, sites):
    v = np.ones((1, 1))
    for m in to_dense_mps(mps, sites):
        m = np.asarray(m)
        v = np.einsum("xa,asb->xsb", v, m).reshape(-1, m.shape[2])
    return v


def isospin_squared(N):
    Iz, Ip = isospin_sparse(N)
    return Iz @ Iz + 0.5 * (Ip @ Ip.T + Ip.T @ Ip)


def two_flavour(N, x=X, mu=MU):
    W = merge_pairs(schwinger2f_mpo(N, x, mu))
    return W, symmetric_mpo(W, site_spaces_2f(N))


def sector_ground(N, q, two_i, x=X, mu=MU):
    """The lowest energy of total charge q and isospin two_i / 2, by exact diagonalisation on the
    states with I_z = I (all of isospin >= I), where 40 (I^2 - I(I+1)) lifts I' > I."""
    H, I2 = schwinger2f_sparse(N, x, mu), isospin_squared(N)
    iz = isospin_sparse(N)[0].diagonal()
    idx = np.flatnonzero((charge_sparse_diag(N) == q) & (np.abs(iz - two_i / 2) < 1e-9))
    j = two_i / 2
    A = (H + 40.0 * (I2 - j * (j + 1) * sp.identity(H.shape[0])))[idx][:, idx]
    if len(idx) < 400:
        v = np.linalg.eigh(A.toarray())[1][:, 0]
    else:
        v = sla.eigsh(A, k=1, which="SA", tol=1e-14)[1][:, 0]
    h, c = H[idx][:, idx], I2[idx][:, idx]
    assert abs(np.vdot(v, c @ v).real - j * (j + 1)) < 1e-9
    return float(np.vdot(v, h @ v).real)


# --------------------------------------------------------------------------- reduced MPOs
@pytest.mark.parametrize("N", [2, 3, 4])
def test_two_flavour_mpo_is_rebuilt_exactly(N):
    W, (red, info) = two_flavour(N)
    assert max(info["errors"]) < 1e-12
    full = to_dense(W)
    assert np.abs(dense_operator(red, site_spaces_2f(N)) - full).max() < 1e-12 * np.abs(full).max()


def _product_expectation(mpo, vecs):
    """<v|W|v> of the dense MPO `mpo` on the product state of the site vectors `vecs`."""
    out = np.ones(1)
    for W, v in zip(mpo, vecs):
        out = out @ np.einsum("s,abst,t->ab", v.conj(), np.asarray(W), v)
    return complex(out[0])


@pytest.mark.parametrize("ell", [0.0, 0.3])
@pytest.mark.parametrize("penalty", [0.0, 50.0])
def test_two_flavour_mpo_holds_on_a_long_chain(ell, penalty):
    """At N = 200 the channels of one bond differ in norm by orders of magnitude, and each must
    survive the rank cut. The channel maps grow polynomially along the chain, not as 2^n."""
    N = 200
    W = merge_pairs(schwinger2f_mpo(N, X, MU, ell=ell, charge_penalty=penalty))
    sites = site_spaces_2f(N)
    red, info = symmetric_mpo(W, sites)
    assert max(info["errors"]) < 1e-12
    assert max(np.abs(T).max() for T in info["maps"]) < 1e9
    dense = [np.einsum("sa,wvst,tb->wvab", s.basis.conj(), np.asarray(t.to_dense()), s.basis)
             for t, s in zip(red, sites)]
    rng = np.random.default_rng(int(10 * ell + penalty))
    for _ in range(3):
        vecs = [rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(N)]
        vecs = [v / np.linalg.norm(v) for v in vecs]
        ref = _product_expectation(W, vecs)
        assert abs(_product_expectation(dense, vecs) - ref) < 1e-10 * abs(ref)


def test_two_flavour_channels_are_charge_and_isospin_multiplets():
    _, (red, info) = two_flavour(5)
    bond = red[1].codomain[1].dual()
    assert bond.sectors == ((-1, 1), (0, 0), (1, 1))         # hops of a doublet, both ways
    assert bond.degeneracies == (1, 3, 1)


def test_heisenberg_mpo_is_rebuilt_exactly():
    N = 6
    W = heisenberg_mpo(N)
    red, info = symmetric_mpo(W, [spin_half_site()] * N)
    assert max(info["errors"]) < 1e-12
    assert red[2].codomain[1].sectors == (0, 2)              # identities and the spin triplet
    full = to_dense(W)
    assert np.abs(dense_operator(red, [spin_half_site()] * N) - full).max() < 1e-12


def test_a_channel_that_mixes_irreps_raises():
    """The compiler shares a channel between I_z (isospin 1) and the charge (isospin 0) when the
    isospin penalty is added."""
    W = merge_pairs(schwinger2f_mpo(4, X, MU, isospin_penalty=1.0))
    with pytest.raises(ValueError, match="one irrep"):
        symmetric_mpo(W, site_spaces_2f(4))


def test_a_channel_that_mixes_states_raises():
    """X X + Y Y compiles to channels X and Y, each a mixture of the m = +1 and -1 states."""
    with pytest.raises(ValueError, match="one state"):
        symmetric_mpo(heisenberg_mpo(4, "xyz"), [spin_half_site()] * 4)


def test_wrong_in_site_string_raises():
    """With the d operators of every site multiplied by (1 - 2 n_u), (c+_u, c+_d) is no longer an
    isospin doublet and the channel labels fail."""
    W = merge_pairs(schwinger2f_mpo(4, X, MU))
    su, sd = np.divmod(np.arange(4), 2)
    flip = np.where((su[:, None] == 1) & (sd[:, None] != sd[None, :]), -1.0, 1.0)
    with pytest.raises(ValueError, match="channel"):
        symmetric_mpo([w * flip for w in W], site_spaces_2f(4))


def test_merged_site_holds_the_standard_doublet():
    """I+ = c+_u c_d on one merged site, in the leg's basis, is the Condon-Shortley raising
    operator on the doublet and zero elsewhere (a common sign on the doublet would pass);
    Q is (q0, q0 + 1, q0 + 1, q0 + 2)."""
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    F = np.diag([1.0, -1.0])
    cu, cd = np.kron(a, np.eye(2)), np.kron(F, a)            # basis 2 s_u + s_d
    for n, s in enumerate(site_spaces_2f(2)):
        U = s.basis
        Ip = U @ (cu.T @ cd) @ U.T
        ref = np.zeros((4, 4))
        ref[1, 2] = 1.0
        assert np.abs(Ip - ref).max() == 0
        q = np.diag(U @ (cu.T @ cu + cd.T @ cd) @ U.T) - 2 * (n % 2)
        assert list(q) == [x[0] for x in s.space.sectors for _ in range(SU2().dim(x[1]))]


# --------------------------------------------------------------------------- ground states
@pytest.mark.parametrize("N", [6, 10, 16])
def test_heisenberg_ground_state_equals_ed(N):
    red, _ = symmetric_mpo(heisenberg_mpo(N), [spin_half_site()] * N)
    E, mps, _ = dmrg(red, 200, charge=0, backend="numpy-real", **TIGHT)
    E0 = sla.eigsh(heisenberg_sparse(N), k=1, which="SA", tol=1e-14)[0][0]
    assert abs(E - E0) < 1e-10
    assert mps[0].sym == SU2()


@pytest.mark.parametrize("N,sector", [(4, (0, 0)), (4, (0, 2)), (4, (1, 1)), (4, (0, 4)),
                                      (6, (0, 0)), (6, (0, 2)), (6, (1, 1)), (8, (0, 0))])
def test_two_flavour_ground_state_equals_ed(N, sector):
    _, (red, _) = two_flavour(N)
    E, _, _ = dmrg(red, 400, charge=sector, backend="numpy-real", **TIGHT)
    assert abs(E - sector_ground(N, *sector)) < 1e-10


def test_u1xu1_equals_block_dmrg_at_the_same_chi():
    N, chi = 6, 16
    sym = ProductSymmetry((U1(), U1()))
    W = merge_pairs(schwinger2f_mpo(N, X, MU))
    labels = [[(q, 0), (q + 1, -1), (q + 1, 1), (q + 2, 0)] for q in [-2 * (n % 2)
                                                                       for n in range(N)]]
    sites = [SiteSpace(Space(sym, lab, (1,) * 4), np.eye(4)) for lab in labels]
    red, _ = symmetric_mpo(W, sites)
    be = get_backend("numpy-real")
    bmpo, _ = to_block_mpo(W, labels, be, sym=sym)
    bmps = random_block_mps(bmpo, chi, be, (0, 0), np.random.default_rng(1))
    smps = []
    for t in bmps:
        spaces = tuple(Space(sym, leg.charges, leg.dims, leg.sign < 0) for leg in t.legs)
        st, dropped = from_dense(t.to_dense(), spaces, (), be)
        assert dropped < 1e-24
        smps.append(st)
    kw = dict(n_sweeps=10, tol=1e-14, eig_tol=1e-12, backend="numpy-real")
    Eb, _, ib = dmrg(bmpo, chi, init_mps=bmps, **kw)
    Es, _, is_ = dmrg(red, chi, init_mps=smps, **kw)
    assert ib["discarded_weight"] > 1e-10                    # the cap acts
    assert abs(Es - Eb) < 1e-10
    # the expansion sweeps report a difference of two norms, so the totals agree less closely
    assert abs(is_["discarded_weight"] / ib["discarded_weight"] - 1) < 1e-2


# --------------------------------------------------------------------------- dynamics
def _quench(N, chi, steps, dt=0.05):
    _, (r0, _) = two_flavour(N)
    _, (r1, _) = two_flavour(N, mu=-0.4)
    sites = site_spaces_2f(N)
    _, psi, _ = dmrg(r0, 400, charge=(0, 0), backend="numpy-real", **TIGHT)
    v0 = state_vector(psi, sites)[:, 0].astype(complex)
    v0 /= np.linalg.norm(v0)
    rec = []
    out, _ = tdvp2_evolve(list(psi), r1, chi, dt, steps,
                          backend=SymmetricBackend(get_backend("numpy")),
                          measure=lambda M: rec.append(state_vector(M, sites)[:, 0]),
                          meas_every=steps)
    exact = expm_multiply(-1j * dt * steps * schwinger2f_sparse(N, X, -0.4), v0)
    v = rec[-1]
    nv = np.vdot(v, v).real
    assert abs(nv - 1) < 1e-12                               # the Schmidt values are normalised
    return 1 - abs(np.vdot(exact, v)) ** 2 / nv, np.vdot(v, isospin_squared(N) @ v).real / nv, out


def test_tdvp_at_full_rank_follows_exact_evolution():
    infid, i2, out = _quench(6, 400, 20)
    assert infid < 1e-10
    assert abs(i2) < 1e-12
    assert all(m.sym == ProductSymmetry((U1(), SU2())) for m in out)


def test_truncated_tdvp_conserves_isospin():
    infid, i2, out = _quench(6, 4, 20)
    assert infid > 1e-8                                      # the cap acts
    assert abs(i2) < 1e-12
    assert max(sum(m.codomain[2].degeneracies) for m in out) == 4


# --------------------------------------------------------------------------- torch
def test_torch_cpu_matches_numpy():
    pytest.importorskip("torch")
    N = 6
    sites = [spin_half_site()] * N
    red, _ = symmetric_mpo(heisenberg_mpo(N), sites)
    E1, m1, _ = dmrg(red, 32, charge=0, backend="numpy-real", **TIGHT)
    E2, _, _ = dmrg(red, 32, charge=0, backend="torch-cpu", **TIGHT)
    assert abs(E1 - E2) < 1e-12
    run = lambda name: tdvp2_evolve(list(m1), red, 8, 0.05, 3,
                                    backend=SymmetricBackend(get_backend(name)))[0]
    o1, o2 = run("numpy"), run("torch-cpu")
    assert o2[0].be.is_torch
    v1, v2 = state_vector(o1, sites), state_vector(o2, sites)
    assert np.abs(v1 - v2).max() < 1e-12


def test_adaptor_refuses_dense_mixing():
    be = SymmetricBackend(get_backend("numpy"))
    V = Space(SU2(), (1,), (1,))
    t = random((V, V.dual()), (), get_backend("numpy"), np.random.default_rng(0))
    with pytest.raises(NotImplementedError):
        be.einsum("ab,bb->a", t, t)
    with pytest.raises(NotImplementedError):
        be.svd(t)


def test_unequal_masses_raise_with_the_reason():
    W = merge_pairs(schwinger2f_mpo(4, X, np.array([MU, MU + 0.2])))
    with pytest.raises(ValueError, match="unequal flavour masses"):
        symmetric_mpo(W, site_spaces_2f(4))


# --------------------------------------------------------------------------- I > 0 states
def _exact_columns(v0, N, t, mu=-0.4):
    H = schwinger2f_sparse(N, X, mu)
    return np.stack([expm_multiply(-1j * t * H, v0[:, k]) for k in range(v0.shape[1])], 1)


@pytest.mark.parametrize("sector", [(1, 1), (0, 2)])
def test_tdvp_from_an_isospin_multiplet_follows_exact_evolution(sector):
    """The last bond holds the multiplet of the state: every state of it evolves exactly."""
    N, dt, steps = 4, 0.05, 10
    _, (r0, _) = two_flavour(N)
    _, (r1, _) = two_flavour(N, mu=-0.4)
    sites = site_spaces_2f(N)
    _, psi, _ = dmrg(r0, 400, charge=sector, backend="numpy-real", **TIGHT)
    v0 = state_vector(psi, sites).astype(complex)
    assert v0.shape[1] == sector[1] + 1
    out, _ = tdvp2_evolve(list(psi), r1, 400, dt, steps,
                          backend=SymmetricBackend(get_backend("numpy")))
    v, exact = state_vector(out, sites), _exact_columns(v0, N, dt * steps)
    assert np.abs(v - exact).max() < 1e-10


def test_dmrg_restarts_from_its_isospin_multiplet():
    _, (red, _) = two_flavour(4)
    E, psi, _ = dmrg(red, 400, charge=(1, 1), backend="numpy-real", **TIGHT)
    E2, _, _ = dmrg(red, 400, init_mps=list(psi), backend="numpy-real", **TIGHT)
    assert abs(E2 - sector_ground(4, 1, 1)) < 1e-10 and abs(E2 - E) < 1e-10


# --------------------------------------------------------------------------- expansion
def _right_projectors(M):
    """For each bond j >= 1, the projector onto the span of the right states of M[j:]."""
    out, R = [], np.ones((1, 1))
    for m in reversed([np.asarray(t) for t in M]):
        R = np.einsum("asb,bx->asx", m, R).reshape(m.shape[0], -1)
        out.append(R.conj().T @ R)
    return out[::-1][1:]


def test_expansion_spans_what_it_spans_on_dense_tensors():
    """With no cap anywhere, expand_right adds the unique span of H|psi> outside each basis:
    the dense and the symmetric run give the same projectors, and the state is unchanged."""
    from tnkit._dmrg import right_canonicalize
    from tnkit.expansion import expand_right
    N = 4
    W, (red, _) = two_flavour(N)
    sites = site_spaces_2f(N)
    _, psi, _ = dmrg(red, 1, charge=(0, 0), backend="numpy-real", n_sweeps=4)
    be = SymmetricBackend(get_backend("numpy"))
    M = right_canonicalize([be.array(m) for m in psi], be)
    D = [np.asarray(t).astype(complex) for t in to_dense_mps(M, sites)]
    v0 = state_vector(M, sites)
    L0, RN = edge_environments(M, red, be)
    added = expand_right(M, red, 1000, None, be, L0, RN)
    one = np.ones((1, 1, 1))
    added_d = expand_right(D, [w.astype(complex) for w in W], 1000, None, get_backend("numpy"),
                           one, one)
    assert sum(added) > 0 and added == added_d
    assert np.abs(state_vector(M, sites) - v0).max() < 1e-12
    for p, q in zip(_right_projectors(to_dense_mps(M, sites)), _right_projectors(D)):
        assert np.abs(p - q).max() < 1e-10


def test_tdvp_with_expansion_on_symmetric_tensors():
    from tnkit.tdvp import TDVPOptions
    from tnkit.tdvp_evolver import TDVPEvolver
    N = 6
    _, (r0, _) = two_flavour(N)
    _, (r1, _) = two_flavour(N, mu=-0.4)
    sites = site_spaces_2f(N)
    _, psi, _ = dmrg(r0, 2, charge=(0, 0), backend="numpy-real", n_sweeps=4)
    bonds = [m.codomain[2].degeneracies for m in psi]
    be = SymmetricBackend(get_backend("numpy"))
    runs = {}
    for n in (0, 2):
        prof = {}
        ev = TDVPEvolver(list(psi), r1, 0.05, 2,
                         options=TDVPOptions(backend=be, expand=n, profile=prof))
        ev.run(4, "1site")
        v = state_vector(ev.mps, sites)[:, 0]
        assert abs(np.vdot(v, v).real - 1) < 1e-12
        assert abs(np.vdot(v, isospin_squared(N) @ v).real) < 1e-12
        assert [m.codomain[2].degeneracies for m in ev.mps] == bonds   # cut back to the caps
        runs[n] = v
        if n:
            assert all(st["max_chi"] == 2 for st in prof["steps"])
    assert np.abs(runs[2] - runs[0]).max() > 1e-6                     # the expansion acted


def test_12site_schedule_counts_multiplets():
    """'12site' takes two-site steps exactly on the bonds below the cap, in multiplets."""
    from tnkit.tdvp import TDVPOptions
    from tnkit.tdvp_evolver import TDVPEvolver
    N, cap = 6, 5
    _, (r0, _) = two_flavour(N)
    _, (r1, _) = two_flavour(N, mu=-0.4)
    _, psi, _ = dmrg(r0, cap, charge=(0, 0), backend="numpy-real", n_sweeps=4)
    below = sum(sum(m.codomain[2].degeneracies) != cap for m in psi[:-1])
    assert 0 < below < N - 1
    prof = {}
    ev = TDVPEvolver(list(psi), r1, 0.05, cap,
                     options=TDVPOptions(backend=SymmetricBackend(get_backend("numpy")),
                                         profile=prof))
    ev.run(1, "12site")
    assert prof["steps"][0]["n_two"] == below


# --------------------------------------------------------------------------- the DMRG split
@pytest.mark.parametrize("name", ["SU2", "U1xSU2"])
def test_enriched_split_without_extra_cuts_like_truncated_svd(name):
    """alpha = 0: the density matrix x x^H ranks a multiplet by dim(c) lambda = dim(c) s^2."""
    from tnkit.expansion import enriched_split
    from tnkit.legs import fuse
    from tnkit.symmetric_linalg import truncated_svd
    if name == "SU2":
        sym = SU2()
        a, p, b = {0: 2, 1: 2, 2: 1}, {0: 1, 1: 1}, {0: 1, 1: 2, 2: 2}
    else:
        sym = ProductSymmetry((U1(), SU2()))
        a, p, b = {(0, 0): 2, (1, 1): 2, (0, 2): 1}, {(0, 0): 1, (1, 1): 1}, \
            {(1, 1): 2, (2, 0): 1, (2, 2): 1}
    order = lambda d: sorted(d, key=sym.sort_key)
    A, P, B = (Space(sym, order(d), [d[c] for c in order(d)]) for d in (a, p, b))
    be = SymmetricBackend(get_backend("numpy"))
    rng = np.random.default_rng(3)
    n_cases = 0
    for right in (True, False):
        for _ in range(10):
            x = random((A, P, P, B.dual()), (), get_backend("numpy"), rng)
            extra = random((A, P, B.dual()), (), get_backend("numpy"), rng)
            mat, _ = fuse(x, ((0, 1), (2, 3)), be)
            for chi in (1, 2, 3, 5):
                Q, _, disc = enriched_split(x, extra, 0.0, chi, be, right)
                info = {}
                U, _, _, d = truncated_svd(mat, chi, info=info)
                bond = Q.codomain[2] if right else Q.codomain[0]
                assert dict(zip(bond.sectors, bond.degeneracies)) == info["kept"]
                assert abs(disc - d) < 1e-10 * max(1.0, d)
                n_cases += 1
    assert n_cases == 80


# --------------------------------------------------------------------------- small pieces
def test_scale_on_a_middle_axis():
    sym = SU2()
    V = Space(sym, (0, 1, 2), (2, 1, 1))
    rng = np.random.default_rng(4)
    S = from_blocks({c: np.diag(rng.uniform(0.5, 2.0, n)) for c, n in zip(V.sectors,
                                                                          V.degeneracies)},
                    (V,), (V,), get_backend("numpy"), sym)               # as truncated_svd's
    for cod, dom, axis in (((V, V, V), (), 1), ((V,), (V, V), 1), ((V, V), (V,), 2)):
        t = random(cod, dom, get_backend("numpy"), rng)
        got = t.scale(S, axis)
        assert (got.codomain, got.domain) == (t.codomain, t.domain)
        want = np.moveaxis(np.tensordot(np.asarray(S.to_dense()), np.asarray(t.to_dense()),
                                        ([1], [axis])), 0, axis)
        assert np.abs(np.asarray(got.to_dense()) - want).max() < 1e-12


def test_split_checks_sizes_and_refuses_a_permuted_partial_split():
    from tnkit.legs import fuse, split
    sym = SU2()
    V, P = Space(sym, (0, 1), (2, 1)), Space(sym, (1,), (1,))
    be = SymmetricBackend(get_backend("numpy"))
    t = random((V, P, V.dual()), (), get_backend("numpy"), np.random.default_rng(5))
    m, rec = fuse(t, ((0, 1), (2,)), be)
    back = split(m, rec, be)
    assert np.abs(np.asarray(back.to_dense()) - np.asarray(t.to_dense())).max() < 1e-12
    m2, rec2 = fuse(t, ((1, 0), (2,)), be)
    assert np.abs(np.asarray(split(m2, rec2, be).to_dense())
                  - np.asarray(t.to_dense())).max() < 1e-12
    with pytest.raises(ValueError, match="unpermuted"):
        split(m2, rec2, be, legs=(0,))
    wrong = random((V, V.dual()), (), get_backend("numpy"), np.random.default_rng(6))
    with pytest.raises(ValueError, match="size"):
        split(wrong, rec, be, legs=(0,))


def test_concat_joins_a_leg_sector_by_sector():
    sym = ProductSymmetry((U1(), SU2()))
    A = Space(sym, ((0, 0), (1, 1)), (1, 2))
    B = Space(sym, ((1, 1), (2, 0)), (1, 1))
    P = Space(sym, ((0, 0), (1, 1)), (1, 1))
    C = Space(sym, ((1, 1), (2, 0), (2, 2)), (2, 1, 1))
    rng = np.random.default_rng(7)
    a = random((A, P, C.dual()), (), get_backend("numpy"), rng)
    b = random((B, P, C.dual()), (), get_backend("numpy"), rng)
    ab = concat(a, b, 0)
    leg = ab.codomain[0]
    assert leg.sectors == ((0, 0), (1, 1), (2, 0)) and leg.degeneracies == (1, 3, 1)
    x, y, z = (np.asarray(t.to_dense()) for t in (a, b, ab))
    pos = {(q, i): o + i * sym.dim(q) for q, o in zip(leg.sectors, leg.offsets)
           for i in range(4)}
    rows = lambda V, first: [pos[(q, i + first.get(q, 0))] + m for q, n in
                             zip(V.sectors, V.degeneracies) for i in range(n)
                             for m in range(sym.dim(q))]
    first = dict(zip(A.sectors, A.degeneracies))
    assert np.abs(z[rows(A, {})] - x).max() < 1e-12
    assert np.abs(z[rows(B, first)] - y).max() < 1e-12
    assert np.abs(z).sum() - np.abs(x).sum() - np.abs(y).sum() < 1e-10       # nothing else


def test_random_symmetric_mps_respects_chi():
    _, (red, _) = two_flavour(8)
    be = get_backend("numpy-real")
    for chi in (1, 3, 7):
        mps = random_symmetric_mps(red, chi, be, (0, 0))
        bonds = [sum(m.codomain[2].degeneracies) for m in mps[:-1]]
        assert max(bonds) == chi and mps[-1].codomain[2].degeneracies == (1,)


# --------------------------------------------------------------------------- observables, files
SZ2 = [1, -1]                                   # 2 S^z of (up, down)
CASES = ["U1-block", "U1-block-q2", "U1-sym", "U1-sym-q2", "SU2", "SU2-S1", "U1xSU2"]


def _case(name):
    """For one test chain: its MPO in block or symmetric tensors, the dense MPO, the sector, a
    random-MPS maker and the map of an MPS to dense tensors in the models' basis. A suffix
    q2 (2 S^z = 2) or S1 (S = 1) picks a charged sector."""
    q = 2 if name.endswith(("-q2", "-S1")) else 0
    if name.startswith("U1-block"):
        W = heisenberg_mpo(10)
        mpo, _ = to_block_mpo(W, [SZ2] * 10, get_backend("numpy-real"), sym=U1())
        return mpo, W, q, random_block_mps, lambda M: block_to_dense_mps(M, [SZ2] * 10)
    if name.startswith("U1-sym"):
        W = heisenberg_mpo(10)
        sites = [SiteSpace(Space(U1(), (-1, 1), (1, 1)), np.eye(2)[::-1])] * 10
    elif name.startswith("SU2"):
        W, sites = heisenberg_mpo(8), [spin_half_site()] * 8
    else:
        (W, _), q, sites = two_flavour(6), (0, 0), site_spaces_2f(6)
    red, _ = symmetric_mpo(W, sites)
    return red, W, q, random_symmetric_mps, lambda M: to_dense_mps(M, sites)


def _states(dM):
    """The dense MPS (numpy) of each state of the last bond of the dense MPS `dM`."""
    dM = [np.asarray(t) for t in dM]
    return [dM[:-1] + [dM[-1][..., k:k + 1]] for k in range(dM[-1].shape[2])]


def _dense_inner(dA, dB):
    return sum(mps_inner(a, b) for a, b in zip(_states(dA), _states(dB)))


def _dense_mean(f, dM, W):
    """`f` over the states of the last bond of `dM`, weighted by their norms."""
    states = _states(dM)
    n = [mps_inner(x, x) for x in states]
    return sum(f(x, W) * m for x, m in zip(states, n)) / sum(n)


@pytest.mark.parametrize("name", CASES)
def test_mps_inner_matches_dense(name):
    mpo, _, q, start, dense = _case(name)
    A = start(mpo, 6, get_backend("numpy"), q, np.random.default_rng(1))
    B = start(mpo, 3, get_backend("numpy-real"), q, np.random.default_rng(2))  # other bonds
    dA, dB = dense(A), dense(B)
    nA, nB = abs(_dense_inner(dA, dA)), abs(_dense_inner(dB, dB))
    assert abs(mps_inner(A, B) - _dense_inner(dA, dB)) < 1e-12 * np.sqrt(nA * nB)
    assert abs(mps_inner(A, A) - nA) < 1e-12 * nA
    wrong = SymmetricBackend if name.startswith("U1-block") else BlockBackend
    with pytest.raises(ValueError, match="operands need a"):
        mps_inner(A, B, wrong(get_backend("numpy")))
    for args in ((A, dB), (dA, B)):
        with pytest.raises(TypeError, match="same tensor type"):
            mps_inner(*args)


@pytest.mark.parametrize("name", CASES)
def test_expectation_and_variance_match_dense(name):
    mpo, W, q, start, dense = _case(name)
    A = start(mpo, 6, get_backend("numpy"), q, np.random.default_rng(3))
    dA = dense(A)
    e1, e2 = _dense_mean(mpo_expectation, dA, W), _dense_mean(mpo_sq_expectation, dA, W)
    scale = max(abs(e2), 1.0)
    assert abs(mpo_expectation(A, mpo) - e1) < 1e-12 * scale
    assert abs(mpo_sq_expectation(A, mpo) - e2) < 1e-12 * scale
    var, floor = mpo_variance(A, mpo)
    assert var > 1e-3 and abs(var - (e2 - e1 * e1).real) < 1e-12 * scale
    assert floor < 1e-12 * scale
    for f in (mpo_expectation, two_site_variance):
        with pytest.raises(TypeError, match="same tensor type|dense MPS and MPO"):
            f(dA, mpo)


@pytest.mark.parametrize("name", ["U1-block", "SU2-S1"])
def test_a_given_backend_is_used(name):
    class Spy(NumpyBackend):
        calls = 0

        def to_scalar(self, x):
            Spy.calls += 1
            return super().to_scalar(x)

    mpo, _, q, start, _ = _case(name)
    A = start(mpo, 6, get_backend("numpy"), q, np.random.default_rng(4))
    wrap = BlockBackend if name == "U1-block" else SymmetricBackend
    ref = mps_inner(A, A), mpo_expectation(A, mpo)
    for be in (Spy(), wrap(Spy())):                      # a dense backend, or its adaptor
        Spy.calls = 0
        got = mps_inner(A, A, be), mpo_expectation(A, mpo, be)
        assert Spy.calls == 3 and np.allclose(got, ref, rtol=1e-13, atol=0)


@pytest.mark.parametrize("name", ["U1-block", "SU2-S1"])
def test_observables_on_torch_cpu(name):
    torch = pytest.importorskip("torch")
    mpo, _, q, start, _ = _case(name)
    M = start(mpo, 6, get_backend("numpy"), q, np.random.default_rng(8))
    tb = get_backend("torch-cpu")
    wrap = BlockBackend if name == "U1-block" else SymmetricBackend
    T = [wrap(tb).array(t) for t in M]
    assert all(isinstance(t.data, torch.Tensor) for t in T)
    fs = (mpo_expectation, mpo_sq_expectation, lambda m, w, be=None: mpo_variance(m, w, be)[0])
    ref = [f(M, mpo) for f in fs] + [mps_inner(M, M)]
    scale = [abs(ref[1])] * 3 + [abs(ref[3])]
    for be in (None, tb, wrap(tb)):
        got = [f(T, mpo, be) for f in fs] + [mps_inner(T, T, be)]
        assert all(abs(g - r) < 1e-12 * c for g, r, c in zip(got, ref, scale))


@pytest.mark.parametrize("name", ["SU2", "U1xSU2"])
def test_variance_vanishes_on_a_ground_state(name):
    mpo, _, q, _, _ = _case(name)
    E, psi, _ = dmrg(mpo, 400, charge=q, backend="numpy-real", **TIGHT)
    assert mpo_variance(psi, mpo)[0] < 1e-10
    assert abs(mpo_expectation(psi, mpo) - E) < 1e-10


def test_inner_in_a_multiplet_sector():
    """(Q, 2I) = (0, 2): the last bond holds a triplet. <A|B> sums over its three states, and
    every normalised quantity is that of one state."""
    N = 4
    _, (red, _) = two_flavour(N)
    sites, H = site_spaces_2f(N), schwinger2f_sparse(N, X, MU)
    _, psi, _ = dmrg(red, 400, charge=(0, 2), backend="numpy-real", **TIGHT)
    phi = random_symmetric_mps(red, 4, get_backend("numpy"), (0, 2), np.random.default_rng(5))
    v, u = state_vector(psi, sites), state_vector(phi, sites)
    assert v.shape[1] == u.shape[1] == 3
    nv = np.vdot(v, v).real
    assert abs(mps_inner(psi, psi) - nv) < 1e-12 * nv
    E = mpo_expectation(psi, red).real
    assert abs(E - sector_ground(N, 0, 2)) < 1e-10
    assert mpo_variance(psi, red)[0] < 1e-10
    ov = abs(mps_inner(psi, phi)) / np.sqrt(mps_inner(psi, psi).real * mps_inner(phi, phi).real)
    var = mpo_variance(phi, red)[0]
    for k in range(3):
        a, b = v[:, k], u[:, k]
        assert abs(np.vdot(a, H @ a).real / np.vdot(a, a).real - E) < 1e-10
        assert abs(abs(np.vdot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b) - ov) < 1e-12
        Hb, nb = H @ b, np.vdot(b, b).real
        assert abs(np.vdot(Hb, Hb).real / nb - (np.vdot(b, Hb).real / nb) ** 2 - var) < 1e-10
    other = random_symmetric_mps(red, 4, get_backend("numpy"), (0, 0), np.random.default_rng(6))
    with pytest.raises(ValueError, match="different bonds"):
        mps_inner(psi, other)


@pytest.mark.parametrize("name,backend", [("SU2", "numpy"), ("U1-sym", "numpy-real"),
                                          ("U1xSU2", "numpy")])
def test_save_load_round_trip(name, backend, tmp_path):
    mpo, _, q, start, _ = _case(name)
    M = start(mpo, 6, get_backend(backend), q, np.random.default_rng(7))
    save_mps(tmp_path / "state.npz", M)
    buf = io.BytesIO()
    save_mps(buf, M)
    buf.seek(0)
    n, E = mps_inner(M, M).real, mpo_expectation(M, mpo)
    for L in (load_mps(tmp_path / "state.npz", M[0].sym), load_mps(buf, M[0].sym)):
        assert [t.codomain for t in L] == [t.codomain for t in M]
        assert all(a.dtype == b.dtype and np.array_equal(a.data, b.data) for a, b in zip(L, M))
        assert abs(abs(mps_inner(M, L)) / n - 1) < 1e-12
        assert abs(mpo_expectation(L, mpo) - E) < 1e-12 * max(abs(E), 1.0)


def test_load_refuses_another_symmetry(tmp_path):
    mpo, _, q, start, _ = _case("SU2")
    save_mps(tmp_path / "s.npz", start(mpo, 4, get_backend("numpy"), q))
    for sym in (U1(), ProductSymmetry((U1(), SU2()))):
        with pytest.raises(ValueError, match=r"holds an MPS of SU2\(\)"):
            load_mps(tmp_path / "s.npz", sym)
    bmpo, _, bq, bstart, _ = _case("U1-block")
    with pytest.raises(TypeError, match="SymmetricTensors"):
        save_mps(tmp_path / "b.npz", bstart(bmpo, 4, get_backend("numpy-real"), bq))


def _rewrite(src, dst, **changes):
    with np.load(src) as z:
        out = {k: z[k] for k in z.files}
    np.savez(dst, **{**out, **{k: np.array(v) for k, v in changes.items()}})


def test_load_refuses_what_save_mps_did_not_write(tmp_path):
    mpo, _, q, start, _ = _case("SU2")
    save_mps(tmp_path / "s.npz", start(mpo, 4, get_backend("numpy"), q))
    with pytest.raises(ValueError, match="at least one"):
        save_mps(tmp_path / "e.npz", [])
    np.savez(tmp_path / "other.npz", a=np.ones(3))
    np.save(tmp_path / "array.npy", np.ones(3))
    with pytest.raises(ValueError, match="not an MPS file"):
        load_mps(tmp_path / "other.npz", SU2())
    with open(tmp_path / "array.npy", "rb") as f, pytest.raises(ValueError, match="not an MPS"):
        load_mps(f, SU2())
    _rewrite(tmp_path / "s.npz", tmp_path / "f2.npz", format=2)
    with pytest.raises(ValueError, match="format 2"):
        load_mps(tmp_path / "f2.npz", SU2())
    with pytest.raises(TypeError, match="dense backend"):
        load_mps(tmp_path / "s.npz", SU2(), BlockBackend(get_backend("numpy")))


def test_load_warns_on_another_version(tmp_path):
    mpo, _, q, start, _ = _case("SU2")
    M = start(mpo, 4, get_backend("numpy"), q)
    save_mps(tmp_path / "s", M)                           # numpy adds .npz
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L = load_mps(tmp_path / "s", SU2())               # and so does load_mps
    assert all(np.array_equal(a.data, b.data) for a, b in zip(L, M))
    _rewrite(tmp_path / "s.npz", tmp_path / "old.npz", tnkit="0.0.1")
    with pytest.warns(UserWarning, match="tnkit 0.0.1"):
        load_mps(tmp_path / "old.npz", SU2())


def test_load_takes_the_result_dtype():
    mpo, _, q, start, _ = _case("SU2")
    M = start(mpo, 4, get_backend("numpy-real"), q)
    M[-1] = SymmetricBackend(get_backend("numpy")).array(M[-1]) * (1 + 1j)   # then complex
    buf = io.BytesIO()
    save_mps(buf, M)
    buf.seek(0)
    L = load_mps(buf, SU2())
    assert all(t.dtype == np.complex128 for t in L)
    assert abs(mpo_expectation(L, mpo) - mpo_expectation(M, mpo)) < 1e-12


def test_load_onto_torch_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    mpo, _, q, _, _ = _case("U1xSU2")
    _, M, _ = dmrg(mpo, 16, charge=q, backend="numpy-real", n_sweeps=4)
    save_mps(tmp_path / "s.npz", M)
    T = load_mps(tmp_path / "s.npz", M[0].sym, get_backend("torch-cpu"))
    assert all(t.be.is_torch and isinstance(t.data, torch.Tensor) for t in T)
    scale = abs(mpo_sq_expectation(M, mpo))
    for f in (mpo_expectation, mpo_sq_expectation):
        assert abs(f(T, mpo) - f(M, mpo)) < 1e-12 * scale
    assert abs(mpo_variance(T, mpo)[0] - mpo_variance(M, mpo)[0]) < 1e-12 * scale
    assert abs(mps_inner(T, T) - mps_inner(M, M)) < 1e-12
