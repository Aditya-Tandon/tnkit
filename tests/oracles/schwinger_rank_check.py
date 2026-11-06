"""Schwinger-model MPO rank check: dense Hamiltonian, hand-written width-5 automaton, ranks.

W = 2H/(a g^2), open chain, Gauss's law eliminated:
W = x sum (s+_n s-_{n+1} + h.c.) + mu/2 sum (1 + (-1)^n Z_n)
    + sum_{n=0}^{N-2} (l + 1/2 sum_{k<=n} ((-1)^k + Z_k))^2  [+ lam Q^2].

Imported by the compiler tests as an independent reference; run directly to print the table.
"""
import numpy as np

from tnkit.models import mpo_to_dense

I = np.eye(2); Z = np.diag([1., -1.]); Sp = np.array([[0, 1], [0, 0.]]); Sm = Sp.T


def op(o, i, N):
    m = np.array([[1.]])
    for k in range(N):
        m = np.kron(m, o if k == i else I)
    return m


def coeffs(N, x, mu, l):
    """Z_i Z_j weights (i<j), linear Z weights and constant of the expanded W."""
    c = np.array([l + 0.5 * sum((-1) ** k for k in range(n + 1)) for n in range(N - 1)])
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = 0.5 * (N - 1 - j)
    h = np.array([c[k:].sum() for k in range(N)]) + 0.5 * mu * np.array([(-1) ** n for n in range(N)])
    C = (c ** 2).sum() + 0.25 * sum(n + 1 for n in range(N - 1)) + mu * N / 2
    return J, h, C


def dense_H(N, x, mu, l, electric_only=False, lam=0.0):
    """W directly from its definition, by Kronecker products."""
    D = 2 ** N; H = np.zeros((D, D)); Zs = [op(Z, i, N) for i in range(N)]
    for n in range(N - 1):
        Ln = l * np.eye(D) + 0.5 * sum(((-1) ** k) * np.eye(D) + Zs[k] for k in range(n + 1))
        H += Ln @ Ln
    if electric_only:
        return H
    for n in range(N):
        H += 0.5 * mu * (np.eye(D) + (-1) ** n * Zs[n])
    for n in range(N - 1):
        H += x * (op(Sp, n, N) @ op(Sm, n + 1, N) + op(Sm, n, N) @ op(Sp, n + 1, N))
    if lam:
        Q = sum(0.5 * (Zs[k] + (-1) ** k * np.eye(D)) for k in range(N))
        H += lam * Q @ Q
    return H


def hand_mpo(N, x, mu, l):
    """Width-5 automaton: 0 START, 1 charge carrier, 2/3 hopping carriers, 4 DONE."""
    J, h, C = coeffs(N, x, mu, l)
    mpo = []
    for n in range(N):
        W = np.zeros((5, 5, 2, 2))
        W[0, 0] = I; W[4, 4] = I
        W[0, 1] = Z; W[1, 1] = I; W[1, 4] = 0.5 * (N - 1 - n) * Z
        W[0, 2] = Sp; W[2, 4] = x * Sm; W[0, 3] = Sm; W[3, 4] = x * Sp
        W[0, 4] = h[n] * Z + (C * I if n == 0 else 0 * I)
        if n == 0:
            W = W[0:1]
        if n == N - 1:
            W = W[:, 4:5]
        mpo.append(W)
    return mpo


def opschmidt_rank(H, N, p, tol=1e-10):
    """Operator-Schmidt rank of H across the cut between sites p-1 and p."""
    T = H.reshape([2] * (2 * N))
    L = list(range(p)) + [N + k for k in range(p)]
    R = list(range(p, N)) + [N + k for k in range(p, N)]
    s = np.linalg.svd(T.transpose(L + R).reshape(4 ** p, 4 ** (N - p)), compute_uv=False)
    return int((s > tol * s[0]).sum())


if __name__ == "__main__":
    x, mu, l = 1.0, 2e-5, 0.0
    for N in (4, 6, 8, 10):
        Hd = dense_H(N, x, mu, l); Hm = mpo_to_dense(hand_mpo(N, x, mu, l))
        Ee = dense_H(N, x, mu, l, True)
        print(f"N={N} |H_dense - H_handMPO|max={np.abs(Hd - Hm).max():.2e}  "
              f"opSchmidt full={[opschmidt_rank(Hd, N, p) for p in range(1, N)]} "
              f"electric={[opschmidt_rank(Ee, N, p) for p in range(1, N)]}")
    N = 8; Hd = dense_H(N, 1.0, 0.5, 0.3)
    print("l=0.3,mu=0.5 N=8 err", np.abs(Hd - mpo_to_dense(hand_mpo(N, 1.0, 0.5, 0.3))).max(),
          [opschmidt_rank(Hd, N, p) for p in range(1, N)])
    for N in (6, 8, 10):
        Hd = dense_H(N, 1.0, 2e-5, 0.0, lam=100.0)
        print(f"penalty lam=100: N={N}", [opschmidt_rank(Hd, N, p) for p in range(1, N)])
    N = 10; D = 2 ** N; w, v = np.linalg.eigh(dense_H(N, 1.0, 2e-5, 0.0))
    Q = sum(0.5 * (op(Z, k, N) + (-1) ** k * np.eye(D)) for k in range(N))
    print(f"vacuum N=10 lam=0: <Q> = {abs(v[:, 0] @ Q @ v[:, 0]):.1e}")
    for N in (12, 50, 100, 400):
        J, _, _ = coeffs(N, 1, 0, 0); Js = J + J.T
        r = [np.linalg.matrix_rank(Js[:p, p:], tol=1e-10) for p in range(1, N)]
        print(f"N={N} cross-block rank of ZZ coeff: max={max(r)} min={min(r)}  "
              f"full-matrix rank={np.linalg.matrix_rank(Js)}")
    for N in (12, 100):
        print(f"N={N} one-channel-per-pair width at middle cut:",
              2 + sum(1 for i in range(N) for j in range(i + 1, N) if i < N // 2 <= j))
