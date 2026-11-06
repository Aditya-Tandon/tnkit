"""TDVP on block tensors with the charge and I_z of the two-flavour chain, against dense TDVP."""
import itertools

import numpy as np
import pytest

from tnkit import blocks as B
from tnkit._dmrg import _cut, right_canonicalize
from tnkit.api import dmrg
from tnkit.backend import get_backend
from tnkit.block_backend import (BlockBackend, mpo_charges, project_to_block_mps, to_block_mpo,
                                 to_block_mps, to_dense_mps)
from tnkit.models_lgt2f import (charge_sparse_diag, compile2f, gaussian_profile, isospin_opsum,
                                isospin_sparse, packet_opsum, schwinger2f_mpo, site_charges_2f,
                                unpack_charge)
from tnkit.observables import apply_mpo, compress, to_numpy
from tnkit.symmetry import U1
from tnkit.tdvp import TDVPOptions, tdvp1_evolve, tdvp2_evolve
from tnkit.tdvp_evolver import TDVPEvolver

N, X, MU, DT = 4, 4.0, 0.3, 0.05
DENSE = get_backend("numpy")
BB = BlockBackend(DENSE)


def vector(mps):
    v = np.ones((1, 1), complex)
    for t in mps:
        v = np.tensordot(v, np.asarray(t), (v.ndim - 1, 0))
    return v.reshape(-1)


def infidelity(u, w):
    return 1 - abs(np.vdot(u, w)) ** 2 / (np.vdot(u, u).real * np.vdot(w, w).real)


def packets(kinds):
    """The DMRG vacuum with one packet per half: an isoscalar pair, or pi+ alone."""
    W = schwinger2f_mpo(N, X, MU)
    _, vac, _ = dmrg(W, chi=64, n_sweeps=20, tol=1e-12)
    st = to_numpy(vac)
    for kind, (c, k, lo, hi) in zip(kinds, [(0.5, 0.2, 0, 1), (2.0, -0.2, 2, 2)]):
        op, _ = compile2f(packet_opsum(N, kind, gaussian_profile(N, c, k, 1.0, lo, hi)), N)
        st = apply_mpo(st, op)
    psi, _ = compress(st, 4096, svd_tol=1e-12)
    return W, [np.asarray(t) for t in to_numpy(psi)]


@pytest.fixture(scope="module")
def eta():
    W, psi = packets(["u+d", "u+d"])
    qs = site_charges_2f(N)
    Mb, info = to_block_mps(psi, qs, DENSE, sym=U1())
    return W, psi, to_block_mpo(W, qs, DENSE, sym=U1())[0], Mb, info, qs


def dense(mps, qs):
    return vector(to_dense_mps(mps, qs))


def sectors(v):
    Iz, Ip = isospin_sparse(N)
    m = (np.round(2 * Iz.diagonal()) == 0) & (charge_sparse_diag(N) == 0)
    w = np.where(m, v, 0)
    I2 = Iz @ Iz + 0.5 * (Ip @ Ip.T + Ip.T @ Ip)
    return 1 - np.sum(np.abs(w) ** 2) / np.sum(np.abs(v) ** 2), np.vdot(w, I2 @ w).real / np.vdot(w, w).real


def test_packed_charges():
    qs = site_charges_2f(N)
    q, t = unpack_charge(np.concatenate(qs), N)
    assert set(q.tolist()) <= {-1, 0, 1} and set(t.tolist()) <= {-1, 0, 1}
    occ = np.array([[0, 1]] * (2 * N))
    assert (t.reshape(-1, 2)[0::2] == occ[0::2]).all() and (t.reshape(-1, 2)[1::2] == -occ[1::2]).all()
    for W in (schwinger2f_mpo(N, X, MU), compile2f(isospin_opsum(N, "I2"), N)[0]):
        mpo_charges(W, qs, sym=U1())                       # raises if a channel carries two charges
        Wb, orders = to_block_mpo(W, qs, DENSE, sym=U1())
        for w, wb, o in zip(W, Wb, orders):
            np.testing.assert_array_equal(wb.to_dense(o), w)


def test_state_conversion(eta):
    W, psi, Wb, Mb, info, qs = eta
    assert info["charge"] == 0 and info["label_err"] < 1e-10
    assert info["fidelity"] == pytest.approx(1, abs=1e-12)
    assert infidelity(vector(psi), dense(Mb, qs)) < 1e-13
    _, pip = packets(["+"])
    Mp, ip = to_block_mps(pip, qs, DENSE, sym=U1())
    assert unpack_charge(ip["charge"], N) == (0, 2)
    with pytest.raises(ValueError, match="charge sector"):
        to_block_mps(_mixed(psi, pip), qs, DENSE, sym=U1())


def test_projected_conversion(eta):
    """A state with a small pi+ admixture: the projection keeps exactly the Q = 0, I_z = 0 part
    when chi is large, and cuts to chi otherwise."""
    W, psi, Wb, Mb, info, qs = eta
    _, pip = packets(["+"])
    mix = _mixed(psi, [0.1 * t if n == 0 else t for n, t in enumerate(pip)])
    v = vector(mix)
    Iz, _ = isospin_sparse(N)
    p0 = np.where((np.round(2 * Iz.diagonal()) == 0) & (charge_sparse_diag(N) == 0), v, 0)
    weight = np.sum(np.abs(p0) ** 2) / np.sum(np.abs(v) ** 2)
    assert 1e-4 < 1 - weight < 0.1
    Mp, ip = project_to_block_mps(mix, qs, DENSE, 64, sym=U1())
    assert ip["charge"] == 0 and ip["discarded"] < 1e-24
    assert ip["weight"] == pytest.approx(weight, abs=1e-12)
    assert ip["fidelity"] == pytest.approx(weight, abs=1e-12)
    assert infidelity(p0, dense(Mp, qs)) < 1e-12
    Mt, it = project_to_block_mps(mix, qs, DENSE, 4, sym=U1())
    assert max(int(t.shape[2]) for t in Mt) <= 4 and it["discarded"] > 0
    assert it["fidelity"] < ip["fidelity"]
    Mpsi = project_to_block_mps(psi, qs, DENSE, 64, sym=U1())[0]
    assert infidelity(vector(psi), dense(Mpsi, qs)) < 1e-12


def test_projection_keeps_chi_at_ties():
    """(|01> + |10>)/sqrt(2): two left charges tie at the first bond, and chi = 1 keeps one."""
    a0, a1 = np.zeros((1, 2, 2)), np.zeros((2, 2, 1))
    a0[0, 0, 0] = a0[0, 1, 1] = 1
    a1[0, 1, 0] = a1[1, 0, 0] = 2 ** -0.5
    M, info = project_to_block_mps([a0, a1], [[0, 1], [0, 1]], DENSE, 1, sym=U1())
    assert int(M[0].shape[2]) == 1 and M[0].legs[2].charges == (0,)
    assert info["fidelity"] == pytest.approx(0.5) and info["discarded"] == pytest.approx(0.5)


def test_projection_drops_values_below_rtol():
    """(|010> + |100>)/sqrt(2): the charge-1 part at the second bond has rank one, so its second
    singular value (zero up to rounding) is dropped even though chi allows it."""
    a = [np.zeros((1, 2, 2)), np.zeros((2, 2, 1)), np.zeros((1, 2, 1))]
    a[0][0, 0, 0] = a[0][0, 1, 1] = 2 ** -0.5
    a[1][0, 1, 0] = a[1][1, 0, 0] = 1
    a[2][0, 0, 0] = 1
    M, info = project_to_block_mps(a, [[0, 1]] * 3, DENSE, 4, sym=U1())
    assert [int(t.shape[2]) for t in M] == [2, 1, 1]


def _mixed(a, b):
    """An MPS of a + b (bond dimensions add)."""
    out = []
    for n, (x, y) in enumerate(zip(a, b)):
        if n == 0:
            out.append(np.concatenate([x, y], axis=2))
        elif n == len(a) - 1:
            out.append(np.concatenate([x, y], axis=0))
        else:
            z = np.zeros((x.shape[0] + y.shape[0], 2, x.shape[2] + y.shape[2]), complex)
            z[:x.shape[0], :, :x.shape[2]], z[x.shape[0]:, :, x.shape[2]:] = x, y
            out.append(z)
    return out


def test_qr_and_right_canonical(eta):
    _, _, _, Mb, _, qs = eta
    m = Mb[3].fuse(((0, 1), (2,)))[0]
    for right in (False, True):
        a, b = B.qr(m, right)
        np.testing.assert_allclose(B.tensordot(a, b, ((1,), (0,))).to_dense(), m.to_dense(),
                                   atol=1e-14)
        iso = a.to_dense() if not right else b.to_dense().conj().T
        np.testing.assert_allclose(iso.conj().T @ iso, np.eye(iso.shape[1]), atol=1e-14)
    M = right_canonicalize(list(Mb), BB)
    assert infidelity(dense(M, qs), dense(Mb, qs)) < 1e-14
    for t, q in zip(M[1:], qs[1:]):
        d = np.asarray(t.to_dense((None, B.Leg.from_charges(q)[1], None)))
        np.testing.assert_allclose(np.einsum("asb,csb->ac", d, d.conj()), np.eye(d.shape[0]),
                                   atol=1e-13)


@pytest.mark.parametrize("method, n", [("2tdvp", 10), ("1tdvp", 4)])
def test_integrators_match_dense(eta, method, n):
    W, psi, Wb, Mb, _, qs = eta
    if method == "2tdvp":
        a = tdvp2_evolve(list(psi), W, 64, DT, n, TDVPOptions(svd_tol=1e-12))[0]
        b = tdvp2_evolve(list(Mb), Wb, 64, DT, n, TDVPOptions(backend=BB, svd_tol=1e-12))[0]
    else:
        a = tdvp1_evolve(list(psi), W, DT, n, TDVPOptions(graph="off"))[0]
        b = tdvp1_evolve(list(Mb), Wb, DT, n, TDVPOptions(backend=BB, graph="off"))[0]
    assert infidelity(vector(a), dense(b, qs)) < 1e-12


@pytest.mark.parametrize("cap, tol", [(6, 0.0), (8, 1e-8), (12, 1e-8), (16, 0.0)])
def test_per_bond_schedule_matches_dense(eta, cap, tol):
    W, psi, Wb, Mb, _, qs = eta
    o = dict(svd_tol=1e-10, degenerate_tol=tol, graph="off")
    a = TDVPEvolver(list(psi), W, DT, cap, TDVPOptions(**o))
    b = TDVPEvolver(list(Mb), Wb, DT, cap, TDVPOptions(backend=BB, **o))
    a.run(20, "12site")
    b.run(20, "12site")
    assert [t.shape for t in a.mps] == [t.shape for t in b.mps]
    assert infidelity(vector(a.mps), dense(b.mps, qs)) < 1e-12


def test_blocks_do_not_leak_where_dense_does(eta):
    # one-site TDVP far below the needed bond dimension: dense TDVP amplifies round-off out of
    # the sector, block tensors cannot leave it
    W, psi, Wb, Mb, _, qs = eta
    a = tdvp1_evolve(list(psi), W, DT, 50, TDVPOptions(graph="off"))[0]
    b = tdvp1_evolve(list(Mb), Wb, DT, 50, TDVPOptions(backend=BB, graph="off"))[0]
    leak_a, _ = sectors(vector(a))
    leak_b, _ = sectors(dense(b, qs))
    assert leak_a > 1e-3
    assert abs(leak_b) < 1e-14


def test_refusals(eta):
    W, psi, Wb, Mb, _, qs = eta
    with pytest.raises(ValueError, match="fast_expm"):
        tdvp1_evolve(list(Mb), Wb, DT, 1, TDVPOptions(backend=BB, fast_expm=8, graph="off"))
    with pytest.raises(ValueError, match="fast_expm"):
        TDVPEvolver(list(Mb), Wb, DT, 8, TDVPOptions(backend=BB, fast_expm=8, graph="off"))
    with pytest.raises(ValueError, match="open boundaries"):
        tdvp2_evolve(list(Mb), Wb, 8, DT, 1, TDVPOptions(backend=BB, L0=np.ones((1, 1, 1))))
    with pytest.raises(ValueError, match="trunc='svd'"):
        tdvp2_evolve(list(Mb), Wb, 8, DT, 1, TDVPOptions(backend=BB, trunc="randomized"))
    for mps, mpo, be in ((Mb, Wb, DENSE), (Mb, W, BB), (psi, Wb, BB)):
        with pytest.raises(ValueError, match="block MPS, a block MPO"):
            tdvp2_evolve(list(mps), mpo, 8, DT, 1, TDVPOptions(backend=be))
        with pytest.raises(ValueError, match="block MPS, a block MPO"):
            tdvp1_evolve(list(mps), mpo, DT, 1, TDVPOptions(backend=be, graph="off"))
        with pytest.raises(ValueError, match="block MPS, a block MPO"):
            TDVPEvolver(list(mps), mpo, DT, 8, TDVPOptions(backend=be, graph="off"))


def test_block_truncation_keeps_groups_across_sectors():
    # equal singular values in two different sectors: the cut never separates them
    l1, _ = B.Leg.from_charges([0, 0, 0, 1, 1, 1])
    m = B.from_dense(np.zeros((6, 6), complex), (l1, l1.flip()))[0]
    rng = np.random.default_rng(1)
    S0, S1 = [3.0, 2.0, 1.0], [2.5, 2.0, 0.5]
    data = []
    for S in (S0, S1):
        u = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        v = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        data.append(((u * S) @ v.T).reshape(-1))
    m = B.BlockTensor(m.struct, np.concatenate(data).astype(complex))
    info = {}
    U, S, Vh, disc = B.truncated_svd(m, 3, degenerate_tol=1e-8, info=info)
    assert len(S) == 2 and info["pinned"] and disc == pytest.approx(4 + 4 + 1 + 0.25)
    info = {}
    U, S, Vh, _ = B.truncated_svd(m, 3, info=info)
    assert len(S) == 3 and not info["pinned"]


def _block_matrix(rng, sectors):
    l, _ = B.Leg.from_charges([c for c, s in enumerate(sectors) for _ in s])
    m = B.from_dense(np.zeros((l.size, l.size), complex), (l, l.flip()))[0]
    data = []
    for s in sectors:
        u = np.linalg.qr(rng.normal(size=(len(s), len(s))))[0]
        v = np.linalg.qr(rng.normal(size=(len(s), len(s))))[0]
        data.append(((u * s) @ v.T).reshape(-1))
    return B.BlockTensor(m.struct, np.concatenate(data).astype(complex))


def test_block_truncation_follows_the_dense_rule():
    # ties across sectors and values under zero_tol: the block cut, its pin and its dropped
    # weight are those of a dense SVD of the same matrix cut by `_cut`
    rng = np.random.default_rng(0)
    for _ in range(30):
        vals = rng.uniform(0.01, 1, size=5)
        sectors = [np.sort(rng.choice(vals, size=rng.integers(1, 5)))[::-1] for _ in range(3)]
        sectors[0] = np.concatenate([sectors[0], [1e-9]])
        m = _block_matrix(rng, sectors)
        s = np.linalg.svd(m.to_dense(), compute_uv=False)
        tot = float(np.sum(s ** 2))
        for chi, svd_tol, deg, zero_tol, bucket in itertools.product(
                (1, 3, 5, 20), (0.0, 1e-3, 0.2), (0.0, 0.3), (0.0, 1e-3, 0.5), (1, 4)):
            info = {}
            _, S, _, disc = B.truncated_svd(m, chi, svd_tol=svd_tol, degenerate_tol=deg,
                                            zero_tol=zero_tol, bucket=bucket, info=info)
            live = max(1, int(np.sum(s > zero_tol * s[0]))) if zero_tol else len(s)
            k, pinned, rel = _cut(s[:live], chi, svd_tol, deg, np.sum(s[live:] ** 2) / tot, tot)
            assert int(np.count_nonzero(S)) == k and info["pinned"] == pinned
            assert disc / tot == pytest.approx(rel, abs=1e-12)


def test_state_conversion_normalises_and_checks_labels():
    _, psi = packets(["u+d", "u+d"])
    _, pip = packets(["+"])
    qs = site_charges_2f(N)
    pip = [pip[0] * 1e-4] + list(pip[1:])          # 1e-8 of the weight in another sector
    Mx, info = to_block_mps(_mixed(psi, pip), qs, DENSE, sym=U1())
    v = vector([t.to_dense() for t in Mx])
    assert np.vdot(v, v).real == pytest.approx(1, abs=1e-13)
    assert info["fidelity"] > 1 - 1e-7 and info["label_err"] > 1e-8
    with pytest.raises(ValueError, match="integers"):
        to_block_mps(_mixed(psi, pip), qs, DENSE, label_tol=1e-8, sym=U1())


def test_torch_blocks_match_numpy_blocks(eta):
    torch = pytest.importorskip("torch")
    W, psi, Wb, Mb, _, qs = eta
    tb = get_backend("torch-cpu")
    bt = BlockBackend(tb)
    Mt, _ = to_block_mps(psi, qs, tb, sym=U1())
    Wt, _ = to_block_mpo(W, qs, tb, sym=U1())
    o = dict(svd_tol=1e-10, graph="off")
    a = TDVPEvolver(list(Mb), Wb, DT, 6, TDVPOptions(backend=BB, **o))
    b = TDVPEvolver(Mt, Wt, DT, 6, TDVPOptions(backend=bt, **o))
    a.run(10, "12site")
    b.run(10, "12site")
    assert isinstance(b.mps[0].data, torch.Tensor)
    vb = vector([tb.to_numpy(t) for t in to_dense_mps(b.mps, qs)])
    assert infidelity(dense(a.mps, qs), vb) < 1e-12
