"""DMRG on block tensors (U(1), Z_2, U(1) x U(1)) against exact diagonalisation and dense DMRG."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from tnkit import blocks as B
from tnkit.api import dmrg, dmrg_excited
from tnkit.backend import get_backend
from tnkit.dmrg_excited import dmrg_block, dmrg_tower
from tnkit.block_backend import (BlockBackend, mpo_charges, random_block_mps, to_block_mpo,
                                 to_dense_mps)
from tnkit.models import tfim_mpo
from tnkit.models_lgt2f import (charge_sparse_diag, compile2f, isospin_sparse,
                                schwinger2f_opsum, schwinger2f_sparse, site_charges_2f,
                                unpack_charge)
from tnkit.opsum import OpSum, compile_mpo, spin_half
from tnkit.symmetry import ProductSymmetry, U1, Zn

REAL = get_backend("numpy-real")
SZ2 = [1, -1]                                   # 2 S^z of (up, down)
KW = dict(n_sweeps=20, tol=1e-13, eig_tol=1e-12)


def xxz_opsum_mpo(N, Jz, form="pm"):
    """Open XXZ chain sum S.S (J = 1, anisotropy Jz) from `OpSum`, real."""
    os = OpSum()
    for i in range(N - 1):
        if form == "pm":
            os += (0.5, "Sp", i, "Sm", i + 1)
            os += (0.5, "Sm", i, "Sp", i + 1)
        else:
            os += (0.25, "X", i, "X", i + 1)
            os += (0.25, "Y", i, "Y", i + 1)
        os += (0.25 * Jz, "Z", i, "Z", i + 1)
    return [np.ascontiguousarray(w.real) for w in compile_mpo(os, [spin_half()] * N)[0]]


SP = sp.csr_matrix([[0.0, 1.0], [0.0, 0.0]])
Z = sp.csr_matrix(np.diag([1.0, -1.0]))


def site_op(o, i, N):
    return sp.kron(sp.kron(sp.identity(2 ** i), o), sp.identity(2 ** (N - i - 1)), format="csr")


def xxz_sparse(N, Jz):
    op = lambda o, i: site_op(o, i, N)
    return sum(0.5 * (op(SP, i) @ op(SP.T, i + 1) + op(SP.T, i) @ op(SP, i + 1))
               + 0.25 * Jz * op(Z, i) @ op(Z, i + 1) for i in range(N - 1))


def ground(H, mask):
    idx = np.flatnonzero(mask)
    return sla.eigsh(H[idx][:, idx], k=1, which="SA", tol=1e-14)[0][0]


def vector(mps):
    v = np.ones((1, 1))
    for t in mps:
        v = np.tensordot(v, np.asarray(t), (v.ndim - 1, 0))
    return v.reshape(-1)


def sz2(N):
    """2 S^z_total of every basis state (site 0 most significant, index 0 = up)."""
    bits = (np.arange(2 ** N)[:, None] >> np.arange(N - 1, -1, -1)) & 1
    return (1 - 2 * bits).sum(1)


@pytest.mark.parametrize("Jz", [1.0, 0.5])
def test_xxz_against_ed_and_dense(Jz):
    N = 12
    W = xxz_opsum_mpo(N, Jz)
    Wb, _ = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())
    E, psi, info = dmrg(Wb, 64, charge=0, **KW)
    Ed, _, _ = dmrg(W, 64, **KW)
    H = xxz_sparse(N, Jz)
    e0 = ground(H, sz2(N) == 0)
    assert abs(E - e0) < 1e-10 and abs(E - Ed) < 1e-10
    assert info["blocks"] and info["backend"] == "numpy-real"
    v = vector(to_dense_mps(psi, [SZ2] * N))
    assert np.vdot(v, H @ v) / np.vdot(v, v) == pytest.approx(E, abs=1e-10)
    assert np.sum(v[sz2(N) != 0] ** 2) == 0.0


def test_xxz_other_sector():
    """The lowest state with 2 S^z = 4, which dense DMRG cannot target."""
    N = 10
    Wb, _ = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())
    E, psi, _ = dmrg(Wb, 32, charge=4, **KW)
    assert psi[-1].legs[2].charges == (4,)
    assert abs(E - ground(xxz_sparse(N, 1.0), sz2(N) == 4)) < 1e-10


def test_xxz_equal_chi_truncated():
    """Below the exact bond dimension, block and dense DMRG keep the same states."""
    N, chi = 24, 24
    W = xxz_opsum_mpo(N, 1.0)
    Wb, _ = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())
    E, _, info = dmrg(Wb, chi, charge=0, **KW)
    Ed, _, _ = dmrg(W, chi, **KW)
    assert info["discarded_weight"] > 1e-8
    assert abs(E - Ed) < 1e-10


@pytest.fixture(scope="module")
def two_flavour():
    N = 5
    W, _ = compile2f(schwinger2f_opsum(N, 4.0, 0.3), N, real=True)
    H = schwinger2f_sparse(N, 4.0, 0.3).real
    Iz, _ = isospin_sparse(N)
    e0 = ground(H, (charge_sparse_diag(N) == 0) & (np.round(2 * Iz.diagonal()) == 0))
    return N, W, e0


def test_two_flavour_product_symmetry(two_flavour):
    """U(1) x U(1) with label tuples (2 I_z, q), and the same as one packed U(1) charge."""
    N, W, e0 = two_flavour
    packed = site_charges_2f(N)
    pairs = [list(zip(*(x.tolist() for x in unpack_charge(c, N)[::-1]))) for c in packed]
    sym = ProductSymmetry((U1(), U1()))
    E, psi, _ = dmrg(to_block_mpo(W, pairs, REAL, sym=sym)[0], 64, charge=(0, 0), **KW)
    Ep, _, _ = dmrg(to_block_mpo(W, packed, REAL, sym=U1())[0], 64, charge=0, **KW)
    Ed, _, _ = dmrg(W, 64, **KW)
    assert abs(E - e0) < 1e-10 and abs(Ep - e0) < 1e-10 and abs(Ed - e0) < 1e-10
    assert psi[0].struct.sym == sym and psi[-1].legs[2].charges == ((0, 0),)
    v = vector(to_dense_mps(psi, pairs))           # the state itself, not only its energy
    H = schwinger2f_sparse(N, 4.0, 0.3).real
    assert np.vdot(v, H @ v) / np.vdot(v, v) == pytest.approx(e0, abs=1e-10)


def test_tfim_z2():
    N, g = 12, 0.9
    W = [np.ascontiguousarray(w.real) for w in tfim_mpo(N, g)]
    Wb, _ = to_block_mpo(W, [[0, 1]] * N, REAL, sym=Zn(2))
    E, _, _ = dmrg(Wb, 64, charge=0, **KW)
    E1, _, _ = dmrg(Wb, 64, charge=1, **KW)
    X = SP + SP.T
    H = -sum(site_op(X, i, N) @ site_op(X, i + 1, N) for i in range(N - 1)) \
        - g * sum(site_op(Z, i, N) for i in range(N))
    odd = (sz2(N) // 2 - N // 2) % 2 != 0          # parity of the number of down spins
    assert abs(E - ground(H, ~odd)) < 1e-10 and abs(E1 - ground(H, odd)) < 1e-10
    assert E1 > E + 1e-3


def test_backends_agree():
    pytest.importorskip("torch")
    N = 10
    Wb, _ = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())
    E, _, _ = dmrg(Wb, 32, charge=0, **KW)
    Et, psi, _ = dmrg(Wb, 32, charge=0, backend="torch-cpu", **KW)
    Eb, _, _ = dmrg(Wb, 32, charge=0, backend=BlockBackend(get_backend("numpy")), **KW)
    assert abs(Et - E) < 1e-10 and abs(Eb - E) < 1e-10
    assert not isinstance(psi[0].data, np.ndarray)


def test_mpo_charges_dead_and_mixed_channels():
    N = 6
    W = xxz_opsum_mpo(N, 1.0)          # the compiled MPO has channels that carry nothing
    assert np.abs(W[0][:, -1]).max() == 0
    chans = mpo_charges(W, [SZ2] * N, sym=U1())
    assert chans[0].tolist() == [0] and sorted(chans[1].tolist()) == [-2, 0, 0, 0, 2]
    Wb, orders = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())
    for w, wb, o in zip(W, Wb, orders):
        np.testing.assert_array_equal(wb.to_dense(o), w)
    with pytest.raises(ValueError, match="mixes charges"):
        mpo_charges(xxz_opsum_mpo(N, 1.0, form="xy"), [SZ2] * N, sym=U1())
    # a channel cut off from the left end: its entries further right must not set charges
    k = int(np.flatnonzero(chans[1] == 2)[0])
    W[0] = W[0].copy()
    W[0][:, k] = 0
    cut = [w.copy() for w in W]
    cut[1][k] = 0
    Wb, orders = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())
    for w, wb, o in zip(cut, Wb, orders):
        np.testing.assert_array_equal(wb.to_dense(o), w)


def test_random_block_mps():
    N, chi = 8, 6
    Wb, _ = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())
    M = random_block_mps(Wb, chi, REAL, charge=2, rng=np.random.default_rng(1))
    assert [t.legs[0].charges for t in M][:2] == [(0,), (-1, 1)]
    assert M[-1].legs[2].charges == (2,) and max(t.shape[2] for t in M) <= chi
    assert all(t.charge == 0 for t in M)
    for a, b in zip(M[:-1], M[1:]):
        assert a.legs[2] == b.legs[0].flip()
    with pytest.raises(ValueError, match="total charge"):
        random_block_mps(Wb, chi, REAL, charge=1)


def test_refusals():
    N = 6
    W = xxz_opsum_mpo(N, 1.0)
    Wb, _ = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())
    with pytest.raises(ValueError, match="block MPO"):
        dmrg(W, 8, charge=0)
    with pytest.raises(ValueError, match="trunc='svd'"):
        dmrg(Wb, 8, trunc="density")
    with pytest.raises(ValueError, match="lanczos-mixed"):
        dmrg(Wb, 8, eig="lanczos-mixed")
    with pytest.raises(ValueError, match="not a sector label"):
        dmrg(Wb, 8, charge=(0, 0))
    pairs = to_block_mpo(W, [[(1, 0), (-1, 1)]] * N, REAL, sym=ProductSymmetry((U1(), Zn(2))))[0]
    with pytest.raises(ValueError, match="not a sector label"):
        dmrg(pairs, 8, charge=0)
    for f in (dmrg_excited, lambda w, c: dmrg_block(w, c, 2), lambda w, c: dmrg_tower(w, c, 2)):
        with pytest.raises(ValueError, match="takes a dense MPO"):
            f(Wb, 8)
    M = random_block_mps(Wb, 8, REAL, charge=0)
    with pytest.raises(ValueError, match="block init_mps"):
        dmrg(W, 8, init_mps=M)
    with pytest.raises(ValueError, match="block init_mps"):
        dmrg(Wb, 8, init_mps=dmrg(W, 8, n_sweeps=1)[1])
    with pytest.raises(ValueError, match="exactly one of charge="):
        dmrg(Wb, 8, init_mps=M, charge=0)
    E, _, _ = dmrg(Wb, 8, init_mps=M, **KW)
    assert E == pytest.approx(ground(xxz_sparse(N, 1.0), sz2(N) == 0), abs=1e-10)
    Mc = [B.BlockTensor(t.struct, t.data.astype(complex)) for t in M]    # real on a real backend
    assert dmrg(Wb, 8, init_mps=Mc, **KW)[0] == pytest.approx(E, abs=1e-12)
    assert isinstance(M[0], B.BlockTensor)


def test_block_mps_refuses_dense_methods():
    N = 6
    E, psi, _ = dmrg(to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())[0], 8, charge=0)
    for f in (psi.to_numpy, psi.entropies, lambda: np.asarray(psi[0])):
        with pytest.raises(TypeError, match="to_dense_mps"):
            f()


@pytest.mark.parametrize("kw", [dict(eig="lanczos-restart"), dict(eig="lobpcg"),
                                dict(eig="scipy-host"), dict(warm_start=False),
                                dict(eig="lobpcg", backend="torch-cpu")])
def test_other_eigensolvers(kw):
    if kw.get("backend") == "torch-cpu":
        pytest.importorskip("torch")
    N = 8
    Wb, _ = to_block_mpo(xxz_opsum_mpo(N, 0.5), [SZ2] * N, REAL, sym=U1())
    E, _, _ = dmrg(Wb, 16, charge=2, **KW, **kw)
    assert abs(E - ground(xxz_sparse(N, 0.5), sz2(N) == 2)) < 1e-10


@pytest.mark.parametrize("q", [0, 2])
def test_xxz_z4(q):
    """2 S^z mod 4: each Z_4 sector holds two U(1) sectors."""
    N = 8
    Wb, _ = to_block_mpo(xxz_opsum_mpo(N, 1.0), [[1, 3]] * N, REAL, sym=Zn(4))
    E, psi, _ = dmrg(Wb, 32, charge=q, **KW)
    assert psi[-1].legs[2].charges == (q,)
    assert abs(E - ground(xxz_sparse(N, 1.0), sz2(N) % 4 == q)) < 1e-10


def test_random_block_mps_keeps_only_reachable_sectors():
    N = 4
    W, _ = compile2f(schwinger2f_opsum(N, 4.0, 0.3), N, real=True)
    Wb, _ = to_block_mpo(W, site_charges_2f(N), REAL, sym=U1())
    M = random_block_mps(Wb, 2, REAL, charge=0, rng=np.random.default_rng(0))
    for t in M:
        a, p, b = t.legs
        assert b.charges and set(b.charges) <= {x + s for x in a.charges for s in p.charges}
        assert set(a.charges) <= {y - s for y in b.charges for s in p.charges}


@pytest.mark.parametrize("q", [3, -3])
def test_chi_one_start_connects(q):
    """At chi = 1 sectors picked bond by bond used not to chain."""
    N = 9
    Wb, _ = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())
    E, psi, _ = dmrg(Wb, 1, charge=q, n_sweeps=2)
    assert psi[-1].legs[2].charges == (q,) and np.isfinite(E)
    assert max(t.shape[2] for t in psi) == 1


@pytest.mark.parametrize("block", [False, True])
def test_cold_start_is_used(block):
    """warm_start=False starts each local solve from a random vector, which costs matvecs."""
    N = 8
    W = xxz_opsum_mpo(N, 1.0)
    mpo = to_block_mpo(W, [SZ2] * N, REAL, sym=U1())[0] if block else W
    warm = dmrg(mpo, 16, n_sweeps=2, tol=0.0, charge=0 if block else None)[2]["matvec_calls"]
    cold = dmrg(mpo, 16, n_sweeps=2, tol=0.0, warm_start=False,
                charge=0 if block else None)[2]["matvec_calls"]
    assert cold > warm


def test_random_block_mps_bond_cap_with_many_small_sectors():
    """Spin 1: many small sectors each get one state, so the cap takes more than one pass."""
    N, chi = 9, 7
    Wb, _ = to_block_mpo([np.eye(3).reshape(1, 1, 3, 3)] * N, [[2, 0, -2]] * N, REAL, sym=U1())
    M = random_block_mps(Wb, chi, REAL, charge=6)
    assert max(sum(t.legs[2].dims) for t in M) <= chi


def test_complex_torch_start_on_real_torch_backend():
    """A complex torch start on a real torch backend is a strided view; it must be copied."""
    torch = pytest.importorskip("torch")
    N = 6
    Wb, _ = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())
    M = random_block_mps(Wb, 8, REAL, charge=0)
    Mc = [B.BlockTensor(t.struct, torch.tensor(t.data.astype(complex))) for t in M]
    E = dmrg(Wb, 8, init_mps=Mc, backend="torch-cpu-real", n_sweeps=4)[0]
    assert E == pytest.approx(dmrg(Wb, 8, init_mps=M, n_sweeps=4)[0], abs=1e-10)
