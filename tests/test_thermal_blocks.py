"""Purified thermal states in U(1) x U(1) block tensors against the dense path."""
import numpy as np
import pytest

from tnkit import SZ, apply_mpo, compress, local_expectations, mpo_expectation, tdvp2_evolve
from tnkit.backend import get_backend
from tnkit.block_backend import (BlockBackend, apply_block_mpo, mpo_charges, to_block_mpo,
                                 to_dense_mps)
from tnkit.blocks import qr, tensordot
from tnkit.models import tfim_mpo
from tnkit.models_lgt import schwinger_charges, schwinger_mpo, schwinger_opsum
from tnkit.opsum import OpSum, compile_mpo, spin_half
from tnkit.symmetry import U1
from tnkit.thermal import (PURIFIED, cool, infinite_temperature_state, purified_charges, purify,
                           purify_mpo, thermal_state)

X, MU = 4.0, 0.1
BB = BlockBackend(get_backend("numpy"))
PZ = purify(SZ)


def complex_schwinger(N, theta=0.7):
    """The Schwinger chain with hopping x (e^{i theta} s+ s- + h.c.): charge-conserving, and
    H^T != H, so the twin MPO's transpose matters."""
    os = schwinger_opsum(N, 0.0, MU)
    for n in range(N - 1):
        os += (X * np.exp(1j * theta), "Sp", n, "Sm", n + 1)
        os += (X * np.exp(-1j * theta), "Sm", n, "Sp", n + 1)
    return compile_mpo(os, [spin_half()] * N)[0]


def dense(m, q):
    return [np.asarray(t) for t in to_dense_mps(m, purified_charges(q))]


def vector(mps):
    v = np.ones((1, 1), complex)
    for t in mps:
        v = np.tensordot(v, np.asarray(t), (v.ndim - 1, 0))
    return v.reshape(-1)


def sector_weight(v, q, sector):
    """Weight of the fused-basis vector `v` outside (Q_phys, Q_anc) = `sector`."""
    N, d = q.shape
    idx = np.indices((d * d,) * N).reshape(N, -1)
    qp = sum(q[n][idx[n] // d] for n in range(N))
    qa = sum(q[n][idx[n] % d] for n in range(N))
    out = (qp != sector[0]) | (qa != sector[1])
    return np.sum(np.abs(v[out]) ** 2) / np.sum(np.abs(v) ** 2)


def observables(m, H):
    """Energy and <Z_n> of a dense purified state."""
    return mpo_expectation(m, purify_mpo(H)).real, local_expectations(m, PZ).real


@pytest.mark.parametrize("twin", [False, True])
def test_block_mpo_is_the_dense_mpo(twin):
    """Exact entries, and channel charges (c, 0) for H x 1 and (0, -c) for 1 x H^T, where c are
    the channel charges of H."""
    N = 5
    q = schwinger_charges(N)
    pq = purified_charges(q)
    for H in (schwinger_mpo(N, X, MU, ell=0.3), complex_schwinger(N)):
        W = purify_mpo(H, twin=twin)
        Wb = purify_mpo(H, twin=twin, charges=q)
        orders = to_block_mpo(W, pq, get_backend("numpy"), sym=PURIFIED)[1]
        for w, wb, o in zip(W, Wb, orders):
            np.testing.assert_array_equal(wb.to_dense(o), w)
        for n, (h, c) in enumerate(zip(mpo_charges(H, q, sym=U1()),
                                       mpo_charges(W, pq, sym=PURIFIED))):
            want = [(int(x), 0) for x in h]
            if twin and 0 < n < N:
                want += [(0, -int(x)) for x in h]
            assert c == want
    with pytest.raises(ValueError, match="does not conserve these charges: MPO site 0"):
        purify_mpo(tfim_mpo(N, 0.8), charges=q)            # X does not conserve the charge


def test_identity_in_blocks():
    N = 6
    q = schwinger_charges(N)
    for total in (0, 1):
        psi, ld = infinite_temperature_state(N, 2, q, total)
        pb, lb = infinite_temperature_state(N, 2, q, total, blocks=True)
        assert lb == ld
        assert all(set(t.struct.shape(k)) == {1} for t in pb for k in range(len(t.struct.blocks)))
        assert pb[-1].legs[2].charges == ((total, total),)
        v, vb = vector(psi), vector(dense(pb, q))
        assert np.abs(v - vb).max() < 1e-15
        assert sector_weight(vb, q, (total, total)) == 0
    with pytest.raises(ValueError, match="needs charges"):
        infinite_temperature_state(N, 2, blocks=True)


def test_cooling_matches_dense():
    N, dbeta = 8, 0.1
    H = schwinger_mpo(N, X, MU)
    q = schwinger_charges(N)
    out = {}
    for blocks in (False, True):
        psi, ld = infinite_temperature_state(N, 2, q, 0, blocks=blocks)
        conv = (lambda m: observables(dense(m, q), H)) if blocks else (lambda m: observables(m, H))
        _, rec = cool(psi, H, 256, dbeta, 6, log_z=ld, warmup=8, meas_every=3, measure=conv,
                      charges=q if blocks else None, svd_tol=1e-12)
        out[blocks] = rec
    assert len(out[True]) == 3
    for (b0, lz0, (E0, z0)), (b1, lz1, (E1, z1)) in zip(out[False], out[True]):
        assert b0 == b1
        assert abs(E1 - E0) < 1e-10 and abs(lz1 - lz0) < 1e-10
        assert abs((lz1 + b1 * E1) - (lz0 + b0 * E0)) < 1e-10      # entropy
        assert np.abs(z1 - z0).max() < 1e-10
    rho, lz = thermal_state(H, 0.3, 256, dbeta, charges=q, blocks=True, svd_tol=1e-12)
    assert lz == pytest.approx(out[True][1][1], abs=1e-12)
    assert rho[-1].legs[2].charges == ((0, 0),)
    assert sector_weight(vector(dense(rho, q)), q, (0, 0)) == 0


def test_degenerate_tol_reaches_both_paths():
    """A degenerate_tol large enough to move cuts changes the result, the same way on both."""
    N = 6
    H, q = schwinger_mpo(N, 2.0, 0.3), schwinger_charges(N)
    lz = {}
    for dtol in (0.0, 0.3):
        for blocks in (False, True):
            psi, ld = infinite_temperature_state(N, 2, q, blocks=blocks)
            lz[dtol, blocks] = cool(psi, H, 8, 0.1, 5, log_z=ld, warmup=4, degenerate_tol=dtol,
                                    charges=q if blocks else None)[1][-1][1]
        assert abs(lz[dtol, True] - lz[dtol, False]) < 1e-12
    assert abs(lz[0.3, False] - lz[0.0, False]) > 1e-3


@pytest.fixture(scope="module")
def thermal():
    N = 6
    H = complex_schwinger(N)
    q = schwinger_charges(N)
    rd, _ = thermal_state(H, 0.7, 64, 0.05, charges=q, svd_tol=1e-12)
    rb, _ = thermal_state(H, 0.7, 64, 0.05, charges=q, blocks=True, svd_tol=1e-12)
    return N, H, q, rd, rb


def test_twins_keep_the_block_thermal_state_static(thermal):
    """Energy and bond dimension stay put (the drift is the svd_tol cut, the same as dense); the
    MPO without twins grows the bonds."""
    N, H, q, rd, rb = thermal
    run = lambda m, W, be, conv: tdvp2_evolve(
        list(m), W, 64, 0.05, 20, svd_tol=1e-12, backend=be, meas_every=10,
        measure=lambda x: (observables(conv(x), H)[0], max(int(t.shape[2]) for t in x)))[1]
    rec = run(rb, purify_mpo(H, twin=True, charges=q), BB, lambda m: dense(m, q))
    ref = run(rd, purify_mpo(H, twin=True), "numpy", lambda m: m)
    (E0, c0) = rec[0][1]
    for (_, (E, c)), (_, (Ed, cd)) in zip(rec, ref):
        assert abs(E - E0) < 2e-9 and abs(E - Ed) < 1e-12 and c == cd == c0
    grow = run(rb, purify_mpo(H, charges=q), BB, lambda m: dense(m, q))
    assert grow[-1][1][1] > c0


def packet(N, charged, f=None):
    """sum_n f_n s+_n s-_{n+1} (neutral) or sum_n f_n s+_n (charge +1), Gaussian by default."""
    if f is None:
        f = np.exp(-0.5 * (np.arange(N) - (N - 1) / 2) ** 2 + 0.8j * np.arange(N))
    os = OpSum()
    for n in range(N - (0 if charged else 1)):
        os += (f[n], "Sp", n) if charged else (f[n], "Sp", n, "Sm", n + 1)
    return compile_mpo(os, [spin_half()] * N)[0]


@pytest.mark.parametrize("charged", [False, True])
def test_packet_on_blocks_matches_dense(thermal, charged):
    N, H, q, rd, rb = thermal
    Bo = packet(N, charged)
    exact = vector(apply_mpo(rd, purify_mpo(Bo)))
    a, _ = compress(apply_mpo(rd, purify_mpo(Bo)), 256)
    b, norm, err = apply_block_mpo(list(rb), purify_mpo(Bo, charges=q), 256)
    assert err < 1e-20 and b.center == 0
    assert norm == pytest.approx(np.linalg.norm(exact), rel=1e-12)
    sector = (1, 0) if charged else (0, 0)
    assert b[-1].legs[2].charges == (sector,)
    assert np.abs(vector(dense(b, q)) * norm - exact).max() < 1e-12      # phase included
    assert np.abs(vector(a) - vector(dense(b, q))).max() < 1e-12
    Wd, Wb = purify_mpo(H, twin=True), purify_mpo(H, twin=True, charges=q)
    za = tdvp2_evolve(list(a), Wd, 64, 0.05, 10, svd_tol=1e-12, meas_every=5,
                      measure=lambda m: local_expectations(m, PZ).real)[1]
    final, zb = tdvp2_evolve(list(b), Wb, 64, 0.05, 10, svd_tol=1e-12, backend=BB,
                             meas_every=5,
                             measure=lambda m: local_expectations(dense(m, q), PZ).real)
    for (t0, z0), (t1, z1) in zip(za, zb):
        assert t0 == t1 and np.abs(z1 - z0).max() < 1e-10
    assert final[-1].legs[2].charges == (sector,)
    assert sector_weight(vector(dense(final, q)), q, sector) == 0


def left_canonical(m):
    m = list(m)
    for i in range(len(m) - 1):
        f, rec = m[i].fuse(((0, 1), (2,)))
        Q, R = qr(f)
        m[i] = Q.split(rec, legs=(0,))
        m[i + 1] = tensordot(R.split(rec, legs=(1,)), m[i + 1], ((1,), (0,)))
    return m


def infidelity(exact, v):
    return 1 - abs(np.vdot(exact, v)) ** 2 / (np.vdot(exact, exact).real * np.vdot(v, v).real)


@pytest.mark.parametrize("chi", [6, 12])
@pytest.mark.parametrize("gauge", ["right", "left"])
def test_truncated_packet_as_good_as_compress(thermal, chi, gauge):
    """Cut to chi, the block state is within 2x of dense `compress` in 1 - F whatever the gauge
    of the input, the returned error is that 1 - F, and the returned norm is that of O|psi>."""
    N, H, q, rd, rb = thermal
    Bo = packet(N, True)
    exact = vector(apply_mpo(rd, purify_mpo(Bo)))
    start = list(rb) if gauge == "right" else left_canonical(rb)
    b, norm, err = apply_block_mpo(start, purify_mpo(Bo, charges=q), chi)
    a, _ = compress(apply_mpo(rd, purify_mpo(Bo)), chi)
    f_block, f_dense = infidelity(exact, vector(dense(b, q))), infidelity(exact, vector(a))
    assert max(int(t.shape[2]) for t in b) == chi
    assert 1e-8 < f_block <= 2 * f_dense
    assert err == pytest.approx(f_block, rel=1e-6)
    assert norm == pytest.approx(np.linalg.norm(exact), rel=1e-12)
    assert abs(norm - 1) > 0.1                          # so that norm and norm^2 differ
    assert abs(np.vdot(vector(dense(b, q)), vector(dense(b, q))) - 1) < 1e-12


def test_chained_packets_as_good_as_compress(thermal):
    N, H, q, rd, rb = thermal
    f = np.exp(-0.5 * (np.arange(N) - 1) ** 2 + 0.8j * np.arange(N))
    B1, B2 = packet(N, False, f), packet(N, False, f[::-1].conj())
    exact = vector(apply_mpo(apply_mpo(rd, purify_mpo(B1)), purify_mpo(B2)))
    b = apply_block_mpo(list(rb), purify_mpo(B1, charges=q), 10)[0]
    b = apply_block_mpo(b, purify_mpo(B2, charges=q), 10)[0]
    a = compress(apply_mpo(rd, purify_mpo(B1)), 10)[0]
    a = compress(apply_mpo(a, purify_mpo(B2)), 10)[0]
    f_block, f_dense = infidelity(exact, vector(dense(b, q))), infidelity(exact, vector(a))
    assert 1e-8 < f_block <= 2 * f_dense


def test_packet_svd_tol_as_compress(thermal):
    N, H, q, rd, rb = thermal
    Bo = packet(N, False)
    exact = vector(apply_mpo(rd, purify_mpo(Bo)))
    a, _ = compress(apply_mpo(rd, purify_mpo(Bo)), 256, svd_tol=1e-4)
    b, _, err = apply_block_mpo(list(rb), purify_mpo(Bo, charges=q), 256, svd_tol=1e-4)
    assert [int(t.shape[2]) for t in b] == [int(t.shape[2]) for t in a]
    assert max(int(t.shape[2]) for t in b) < max(int(t.shape[2]) for t in rb)
    assert err == pytest.approx(infidelity(exact, vector(dense(b, q))), rel=1e-6)
    assert 1e-6 < err < (N - 1) * 1e-4


def test_packet_into_an_empty_sector():
    N = 4
    q = schwinger_charges(N)
    top = int(q.max(1).sum())
    psi, _ = infinite_temperature_state(N, 2, q, top, blocks=True)
    with pytest.raises(ValueError, match="charge sector is empty"):
        apply_block_mpo(list(psi), purify_mpo(packet(N, True), charges=q), 8)


def test_refusals():
    N = 4
    H = schwinger_mpo(N, X, MU)
    q = schwinger_charges(N)
    pb, _ = infinite_temperature_state(N, 2, q, blocks=True)
    pd, _ = infinite_temperature_state(N, 2, q)
    with pytest.raises(ValueError, match="block state needs charges"):
        cool(pb, H, 16, 0.1, 1)
    with pytest.raises(ValueError, match="block state needs charges"):
        cool(pd, H, 16, 0.1, 1, charges=q)
    with pytest.raises(ValueError, match=r"shape \(4, 2\), got \(3, 2\)"):
        purify_mpo(H, charges=q[:-1])
    with pytest.raises(ValueError, match="must be integers"):
        purify_mpo(H, charges=q + 0.5)
    with pytest.raises(ValueError, match="needs charges"):
        thermal_state(H, 0.2, 16, 0.1, blocks=True)


def test_normalising_keeps_the_environments(monkeypatch):
    """Scaling site 0 of a block state keeps the centre there, so cooling builds the
    environments once, as the dense path does."""
    from tnkit.tdvp_evolver import TDVPEvolver
    rebuild, calls = TDVPEvolver._rebuild, []
    monkeypatch.setattr(TDVPEvolver, "_rebuild", lambda ev: (calls.append(1), rebuild(ev))[1])
    N = 4
    H, q = schwinger_mpo(N, X, MU), schwinger_charges(N)
    count = {}
    for blocks in (False, True):
        calls.clear()
        psi, ld = infinite_temperature_state(N, 2, q, blocks=blocks)
        cool(psi, H, 16, 0.1, 4, log_z=ld, charges=q if blocks else None)
        count[blocks] = len(calls)
    assert count[True] == count[False] == 1


@pytest.fixture
def torch_cpu():
    pytest.importorskip("torch")
    return BlockBackend(get_backend("torch-cpu"))


def test_packet_on_torch(thermal, torch_cpu):
    N, H, q, rd, rb = thermal
    W = purify_mpo(packet(N, True), charges=q)
    b, norm, err = apply_block_mpo(list(rb), W, 6)
    bt, norm_t, err_t = apply_block_mpo([torch_cpu.array(t) for t in rb], W, 6)
    assert type(bt[0].data).__module__.split(".")[0] == "torch"
    host = [type(t)(t.struct, t.data.numpy()) for t in bt]
    assert np.abs(vector(dense(host, q)) - vector(dense(b, q))).max() < 1e-12
    assert norm_t == pytest.approx(norm, rel=1e-12) and err_t == pytest.approx(err, rel=1e-9)


def test_cooling_on_torch(torch_cpu):
    N = 4
    H, q = schwinger_mpo(N, X, MU), schwinger_charges(N)
    psi, ld = infinite_temperature_state(N, 2, q, blocks=True)
    rec = {}
    for be in ("numpy", "torch-cpu"):
        rho, rec[be] = cool(psi, H, 16, 0.1, 3, log_z=ld, charges=q, svd_tol=1e-12, backend=be)
    assert type(rho[0].data).__module__.split(".")[0] == "torch"
    assert abs(rec["torch-cpu"][-1][1] - rec["numpy"][-1][1]) < 1e-12
