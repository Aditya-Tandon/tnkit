"""OpSum -> MPO compiler: exactness against dense Hamiltonians, the hand-written builders
in `tnkit.models`, and the Schwinger-model rank structure."""
from __future__ import annotations

import time
from functools import reduce

import numpy as np
import pytest
import scipy.sparse as sp

from oracles.schwinger_rank_check import coeffs, dense_H, hand_mpo, opschmidt_rank
from tnkit.backend import get_backend
from tnkit.models import (SX, longrange_mpo, magnetization_mpo, mpo_to_dense, tfim_mpo,
                          tfim_ring_mpo)
from tnkit.opsum import (MPOWidthError, OpSum, SiteType, _cover, compile_mpo, qudit,
                         spin_half, spinless_fermion, to_dense)


def _same_bits(a, b):
    """Equal shape, dtype and bits, except that -0.0 and +0.0 count as equal."""
    return a.shape == b.shape and a.dtype == b.dtype and (a + 0.0).tobytes() == (b + 0.0).tobytes()


def _dense_ref(os, sites, sparse=False):
    """Bosonic reference: each term as a Kronecker product of its per-site operator products."""
    kron = (lambda a, b: sp.kron(a, b, format="csr")) if sparse else np.kron
    H = 0
    for coef, factors in os.terms:
        local = [site.ops["Id"] for site in sites]
        for name, s in factors:
            local[s] = local[s] @ sites[s].ops[name]
        H = H + coef * reduce(kron, [sp.csr_matrix(m) if sparse else m for m in local])
    for J, A, B in os.forms:
        for i, j in zip(*np.nonzero(J)):
            local = [site.ops["Id"] for site in sites]
            local[i], local[j] = sites[i].ops[A], sites[j].ops[B]
            H = H + J[i, j] * reduce(kron, [sp.csr_matrix(m) if sparse else m for m in local])
    return H.toarray() if sparse else H


def _random_opsum(rng, N, names, body, n_terms, complex_coef, form):
    os = OpSum()
    for _ in range(n_terms):
        c = rng.normal() + (1j * rng.normal() if complex_coef else 0)
        args = [c]
        for _ in range(body):
            args += [names[rng.integers(len(names))], int(rng.integers(N))]
        os += tuple(args)
    if form:                                    # rank-2 long-range form: exercises the SVD pass
        u, v = rng.normal(size=(2, N)), rng.normal(size=(2, N))
        os.add_quadratic_form(np.triu(u.T @ v, 1), names[0], names[-1])
    return os


def _assert_exact(mpo, H):
    assert np.abs(to_dense(mpo) - H).max() <= 1e-12 * np.abs(H).max()


def _schwinger(N, x, mu, l, lam=0.0):
    os = OpSum()
    for n in range(N - 1):
        os += (x, "Sp", n, "Sm", n + 1)
        os += (x, "Sm", n, "Sp", n + 1)
    for n in range(N):
        os += (mu / 2, "Id", n)
        os += (mu / 2 * (-1) ** n, "Z", n)
    A = np.tril(np.full((N - 1, N), 0.5))
    b = l + 0.5 * np.cumsum([(-1) ** k for k in range(N - 1)])
    os.add_sum_of_squares(A, b, "Z")
    if lam:
        stag = 0.5 * sum((-1) ** k for k in range(N))
        os.add_sum_of_squares(np.full((1, N), 0.5 * lam ** 0.5), [lam ** 0.5 * stag], "Z")
    return os


@pytest.mark.parametrize("N", [2, 5, 8, 10])
@pytest.mark.parametrize("body", [2, 3, 4])
@pytest.mark.parametrize("complex_coef", [False, True])
def test_dense_equality_spin_half(N, body, complex_coef):
    rng = np.random.default_rng(1000 * N + 10 * body + complex_coef)
    names = ["X", "Y", "Z", "Sp", "Sm"] if complex_coef else ["X", "Z", "Sp", "Sm"]
    sites = [spin_half()] * N
    os = _random_opsum(rng, N, names, body, 12, complex_coef, form=N > 2)
    mpo, _ = compile_mpo(os, sites)
    _assert_exact(mpo, _dense_ref(os, sites))


@pytest.mark.slow
def test_dense_equality_n12():
    rng = np.random.default_rng(12)
    sites = [spin_half()] * 12
    os = _random_opsum(rng, 12, ["X", "Y", "Z", "Sp", "Sm"], 3, 20, True, form=True)
    mpo, _ = compile_mpo(os, sites)
    _assert_exact(mpo, _dense_ref(os, sites, sparse=True))


@pytest.mark.parametrize("d", [3, 5])
def test_dense_equality_qudit(d):
    rng = np.random.default_rng(d)
    ops = {k: rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)) for k in ("A", "B", "C")}
    sites = [qudit(d, ops)] * 5
    os = _random_opsum(rng, 5, ["A", "B", "C"], 3, 10, True, form=True)
    mpo, _ = compile_mpo(os, sites)
    assert [W.shape[2] for W in mpo] == [d] * 5
    _assert_exact(mpo, _dense_ref(os, sites))


def test_fermions_match_dense_jordan_wigner():
    N, rng = 6, np.random.default_rng(7)
    f = spinless_fermion()
    Id, F = f.ops["Id"], f.ops["F"]
    c = [reduce(np.kron, [F] * i + [f.ops["C"]] + [Id] * (N - 1 - i)) for i in range(N)]
    cd = [m.conj().T for m in c]
    os, H = OpSum(), 0
    for i in range(N - 1):
        t = rng.normal() + 1j * rng.normal()
        os += (t, "Cdag", i, "C", i + 1)
        os += (np.conj(t), "Cdag", i + 1, "C", i)          # written out of site order
        H = H + t * cd[i] @ c[i + 1] + np.conj(t) * cd[i + 1] @ c[i]
    for i, j in [(0, 3), (1, 5), (2, 2)]:
        V = rng.normal()
        os += (V, "N", i, "N", j)
        H = H + V * cd[i] @ c[i] @ cd[j] @ c[j]
    os += (0.7, "C", 4, "Cdag", 1, "Cdag", 0, "C", 3)       # four fermions, unordered
    H = H + 0.7 * c[4] @ cd[1] @ cd[0] @ c[3]
    mpo, _ = compile_mpo(os, [f] * N)
    _assert_exact(mpo, H)


def test_schwinger_in_fermion_language_equals_spin_language():
    N, x, mu = 8, 1.3, 0.4
    os = OpSum()
    for n in range(N - 1):                    # c+_n c_{n+1} + h.c. = s-_n s+_{n+1} + h.c.
        os += (x, "Cdag", n, "C", n + 1)
        os += (x, "Cdag", n + 1, "C", n)
    for n in range(N):                        # Z = F = 1 - 2N
        os += (mu / 2, "Id", n)
        os += (mu / 2 * (-1) ** n, "F", n)
    A = np.tril(np.full((N - 1, N), 0.5))
    b = 0.5 * np.cumsum([(-1) ** k for k in range(N - 1)])
    os.add_sum_of_squares(A, b, "F")
    mpo, _ = compile_mpo(os, [spinless_fermion()] * N)
    _assert_exact(mpo, dense_H(N, x, mu, 0.0))


@pytest.mark.parametrize("builder", ["tfim", "tfim_h", "magnetization", "longrange"])
def test_tfim_family_bit_for_bit(builder):
    N, g, h = 8, 0.7, 0.3
    os = OpSum()
    if builder == "magnetization":
        for i in range(N):
            os += (1.0, "X", i)
        ref = magnetization_mpo(N, SX)
    elif builder == "longrange":
        bonds = [(i, i + 3) for i in range(N - 3)]
        for i, j in bonds:
            os += (-1.0, "X", i, "X", j)
        for i in range(N):
            os += (-g, "Z", i)
        ref = longrange_mpo(N, bonds, g)
    else:
        h = h if builder == "tfim_h" else 0.0
        for i in range(N - 1):
            os += (-1.0, "X", i, "X", i + 1)
        for i in range(N):
            os += (-g, "Z", i)
            if h:
                os += (-h, "X", i)
        ref = tfim_mpo(N, g, h)
    mpo, _ = compile_mpo(os, [spin_half()] * N)
    assert all(_same_bits(a, b) for a, b in zip(mpo, ref))
    assert np.array_equal(to_dense(mpo), mpo_to_dense(ref))


def test_tfim_ring():
    """The wrap channel shares its opening operator with the first bond, so the compiled ring
    is one channel narrower at the edges; in the bulk it is the builder with the nearest-
    neighbour and wrap channels swapped."""
    N, g, h = 8, 0.9, 0.2
    os = OpSum()
    for i in range(N):
        os += (-1.0, "X", i, "X", (i + 1) % N)
        os += (-g, "Z", i)
        os += (-h, "X", i)
    mpo, rep = compile_mpo(os, [spin_half()] * N)
    ref = tfim_ring_mpo(N, g, h)
    assert rep["width"] == [1, 3] + [4] * (N - 3) + [3, 1]
    assert np.abs(to_dense(mpo) - mpo_to_dense(ref)).max() < 1e-13
    perm = [0, 2, 1, 3]
    assert all(_same_bits(mpo[k], ref[k][perm][:, perm]) for k in range(2, N - 2))


@pytest.mark.parametrize("N,l,mu,lam", [(6, 0.0, 2e-5, 0.0), (8, 0.0, 2e-5, 0.0),
                                        (10, 0.0, 2e-5, 0.0), (8, 0.3, 0.5, 0.0),
                                        (8, 0.0, 2e-5, 100.0)])
def test_schwinger_widths_match_rank_check(N, l, mu, lam):
    """Inner widths equal the operator-Schmidt ranks, except that an end bond may be one
    wider: START and DONE stay outside the SVD, and there a carrier lies in their span."""
    mpo, rep = compile_mpo(_schwinger(N, 1.0, mu, l, lam), [spin_half()] * N)
    H = dense_H(N, 1.0, mu, l, lam=lam)
    _assert_exact(mpo, H)
    ranks = [opschmidt_rank(H, N, p) for p in range(1, N)]
    widths = rep["width"][1:-1]
    assert ranks == [4] + [5] * (N - 3) + [4]
    assert widths[1:-1] == ranks[1:-1]
    assert all(r <= w <= r + 1 for r, w in zip(ranks, widths))


@pytest.mark.parametrize("N", [12, 50, 100, 400])
def test_schwinger_charge_form_has_cross_rank_one(N):
    J, _, _ = coeffs(N, 1.0, 0.0, 0.0)
    os = OpSum()
    os.add_quadratic_form(J, "Z", "Z")
    _, rep = compile_mpo(os, [spin_half()] * N)
    assert max(rep["numerical"]) == 1
    assert max(rep["structural"]) == 2


def test_real_problem_detection():
    N = 10
    mpo, rep = compile_mpo(_schwinger(N, 1.0, 2e-5, 0.0), [spin_half()] * N, dtype="auto")
    assert rep["dtype"] == "float64" and all(W.dtype == np.float64 for W in mpo)
    assert get_backend("auto", mpo=mpo).name == "numpy-real"
    mpo_c, _ = compile_mpo(_schwinger(N, 1.0, 2e-5, 0.0), [spin_half()] * N)
    assert mpo_c[0].dtype == np.complex128
    assert get_backend("auto", mpo=mpo_c).name == "numpy-real"
    os = OpSum()
    os += (1.0, "Y", 0, "Y", 1)
    mpo_y, _ = compile_mpo(os, [spin_half()] * 2, dtype="auto")
    assert mpo_y[0].dtype == np.complex128


def test_width_guard_names_the_first_wide_bond():
    N, rng = 12, np.random.default_rng(3)
    os = OpSum()
    os.add_quadratic_form(np.triu(rng.normal(size=(N, N)), 1), "Z", "Z")
    _, rep = compile_mpo(os, [spin_half()] * N)
    assert rep["width"][1:-1] == [2 + min(p, N - p) for p in range(1, N)]
    with pytest.raises(MPOWidthError, match=r"bond 5 \(between sites 4 and 5\) needs width 7"):
        compile_mpo(os, [spin_half()] * N, max_width=6)
    compile_mpo(_schwinger(N, 1.0, 2e-5, 0.0), [spin_half()] * N, max_width=5)


@pytest.mark.parametrize("bad", [
    lambda os: os.__iadd__((1.0, "X")),
    lambda os: os.add_quadratic_form(np.ones((3, 3)), "Z", "Z"),
    lambda os: compile_mpo(os.__iadd__((1.0, "Q", 0)), [spin_half()] * 3),
    lambda os: compile_mpo(os.__iadd__((1.0, "X", 3)), [spin_half()] * 3),
    lambda os: os.__iadd__((1.0, "X", 1.5)),
    lambda os: os.__iadd__((1.0, "X", "1")),
    lambda os: compile_mpo(os, []),
])
def test_bad_input_raises(bad):
    with pytest.raises(ValueError):
        bad(OpSum())


@pytest.mark.slow
def test_schwinger_n1000_against_hand_automaton():
    """Compile time and exactness at N = 1000, on a random product state."""
    N = 1000
    t0 = time.perf_counter()
    mpo, rep = compile_mpo(_schwinger(N, 1.0, 2e-5, 0.0), [spin_half()] * N, dtype="auto")
    elapsed = time.perf_counter() - t0
    assert elapsed < 30 and max(rep["width"]) == 5
    rng = np.random.default_rng(0)
    v = rng.normal(size=(N, 2)) + 1j * rng.normal(size=(N, 2))
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    def expval(m):
        E = np.ones((1,))
        for W, x in zip(m, v):
            E = E @ np.einsum("abst,s,t->ab", W, x.conj(), x)
        return E[0]

    ref = expval(hand_mpo(N, 1.0, 2e-5, 0.0))
    assert abs(expval(mpo) - ref) <= 1e-12 * abs(ref)
    print(f"N=1000: compile {elapsed:.1f} s, max|W| = {max(np.abs(W).max() for W in mpo):.4g}, "
          f"largest dropped singular value {max(rep['dropped']):.2e}")


def test_complex128_keeps_tiny_imaginary_parts():
    os = OpSum()
    os += (1.0, "X", 0)
    os += (1e-13j, "X", 1)
    H = to_dense(compile_mpo(os, [spin_half()] * 2)[0])
    assert H[0, 2] == 1.0 and H[0, 1] == 1e-13j
    mpo, rep = compile_mpo(os, [spin_half()] * 2, dtype="auto")
    assert rep["dtype"] == "float64" and to_dense(mpo)[0, 1] == 0


def test_auto_dtype_sees_the_jordan_wigner_parity():
    """A lone C_3 inserts F on sites 0..2; a complex F must make the MPO complex."""
    f = spinless_fermion()
    site = SiteType({**f.ops, "F": np.diag([1, 1j])}, f.fermionic)
    os = OpSum()
    os += (1.0, "C", 3)
    assert compile_mpo(os, [site] * 4, dtype="auto")[1]["dtype"] == "complex128"
    assert compile_mpo(os, [f] * 4, dtype="auto")[1]["dtype"] == "float64"


def test_cover_is_iterative_on_long_augmenting_paths():
    """Row r is adjacent to columns r-1, r: matching row r walks back through every row, so
    a recursive search needs depth n, above Python's default limit of 1000."""
    n = 1500
    adj = {0: [0], **{r: [r - 1, r] for r in range(1, n)}}
    rows, cols = _cover(list(range(n)), adj)
    assert rows == list(range(n)) and cols == []


def _edge_case(case):
    os, ref = OpSum(), OpSum()
    if case == "n1":
        os += (0.3, "X", 0)
        os += (1.2, "Id", 0)
    elif case == "identity":
        os += (2.5, "Id", 0)
        os += (-0.5, "Id", 3)
    elif case in ("merge", "cancel"):
        os += (0.2, "Z", 0)
        ref += (0.2, "Z", 0)
        os += (0.7, "Z", 1, "X", 2)
        os += (0.7 if case == "merge" else -0.7, "X", 2, "Z", 1)
        if case == "merge":
            ref += (1.4, "Z", 1, "X", 2)
    elif case == "zero":
        os += (0.0, "X", 0, "X", 1)
        os += (0.4, "Z", 2)
        ref += (0.4, "Z", 2)
    return os, (os if case in ("n1", "identity") else ref)


@pytest.mark.parametrize("case", ["n1", "identity", "merge", "cancel", "zero"])
def test_edge_cases_match_dense(case):
    sites = [spin_half()] * (1 if case == "n1" else 4)
    os, ref = _edge_case(case)
    mpo, rep = compile_mpo(os, sites)
    _assert_exact(mpo, _dense_ref(ref, sites))
    assert max(rep["width"]) <= (3 if case in ("merge",) else 2)


def test_odd_parity_fermion_term():
    f = spinless_fermion()
    os = OpSum()
    os += (0.9, "C", 3)
    os += (0.4, "Cdag", 1)
    H = (0.9 * reduce(np.kron, [f.ops["F"]] * 3 + [f.ops["C"]])
         + 0.4 * reduce(np.kron, [f.ops["F"], f.ops["Cdag"], f.ops["Id"], f.ops["Id"]]))
    _assert_exact(compile_mpo(os, [f] * 4)[0], H)


@pytest.mark.parametrize("N,b", [(1, [0.3]), (5, [0.3, -1.1])])
def test_sum_of_squares_with_offset(N, b):
    A = np.random.default_rng(N).normal(size=(len(b), N))
    os = OpSum()
    os.add_sum_of_squares(A, b, "Z")
    z = np.array([[1 - 2 * ((k >> (N - 1 - s)) & 1) for s in range(N)] for k in range(2 ** N)])
    _assert_exact(compile_mpo(os, [spin_half()] * N)[0], np.diag(((z @ A.T + b) ** 2).sum(1)))


def test_rank_tol_truncates_a_weak_second_channel():
    N, rng = 10, np.random.default_rng(5)
    u, v, w, z = rng.normal(size=(4, N))
    os = OpSum()
    os.add_quadratic_form(np.triu(np.outer(u, v) + 1e-11 * np.outer(w, z), 1), "Z", "Z")
    H = _dense_ref(os, [spin_half()] * N)
    mpo, rep = compile_mpo(os, [spin_half()] * N)
    cut, rep_cut = compile_mpo(os, [spin_half()] * N, rank_tol=1e-9)
    assert max(rep["numerical"]) == 2 and max(rep_cut["numerical"]) == 1
    assert max(rep["dropped"]) < 1e-14 < max(rep_cut["dropped"]) < 1e-9
    _assert_exact(mpo, H)
    assert 0 < np.abs(to_dense(cut) - H).max() <= 1e-9 * np.abs(H).max()


def test_schwinger_n100_compiles_under_one_second():
    t0 = time.perf_counter()
    _, rep = compile_mpo(_schwinger(100, 1.0, 2e-5, 0.0), [spin_half()] * 100)
    assert time.perf_counter() - t0 < 1.0 and max(rep["width"]) == 5
