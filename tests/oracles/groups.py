"""Explicit group matrices for the test symmetries, a reference that shares no code with their
coefficients: `rep(sym, a, g)` is the unitary matrix of sector `a` at the group element `g`, and
`elements(sym, n, rng)` draws group elements (every element of a finite group when n allows).
Products take one element per factor and Kronecker products, factor-major."""
import numpy as np
from scipy.linalg import expm

from tnkit.symmetry import SU2, U1, ProductSymmetry, Zn

from . import a4


def spin_matrices(j):
    """S^z, S^+, S^- for spin j/2, basis m = j/2, j/2 - 1, ... (highest weight first)."""
    m = (j - 2 * np.arange(j + 1)) / 2
    sp = np.zeros((j + 1, j + 1))
    for k in range(1, j + 1):
        sp[k - 1, k] = np.sqrt(j / 2 * (j / 2 + 1) - m[k] * (m[k] + 1))
    return np.diag(m), sp, sp.T


def rep(sym, a, g):
    if isinstance(sym, U1):
        return np.array([[np.exp(1j * a * g)]])
    if isinstance(sym, Zn):
        return np.array([[np.exp(2j * np.pi * a * g / sym.n)]])
    if isinstance(sym, SU2):
        sz, sp, sm = spin_matrices(a)
        sx, sy = (sp + sm) / 2, (sp - sm) / 2j
        return expm(-1j * (g[0] * sx + g[1] * sy + g[2] * sz))
    if isinstance(sym, a4.A4):
        return sym.rep(a, g)
    if isinstance(sym, ProductSymmetry):
        out = np.ones((1, 1))
        for s, x, h in zip(sym.factors, a, g):
            out = np.kron(out, rep(s, x, h))
        return out
    raise TypeError(f"no group matrices for {sym}")


def elements(sym, n, rng):
    if isinstance(sym, U1):
        return list(rng.uniform(0, 2 * np.pi, n))
    if isinstance(sym, Zn):
        return list(range(sym.n)) if n >= sym.n else list(rng.integers(0, sym.n, n))
    if isinstance(sym, SU2):
        return list(rng.normal(size=(n, 3)) * 2)
    if isinstance(sym, a4.A4):
        e = a4.elements()
        return list(e) if n >= len(e) else [e[i] for i in rng.choice(len(e), n, replace=False)]
    if isinstance(sym, ProductSymmetry):
        per = [elements(s, n, rng) for s in sym.factors]
        return [tuple(p[i % len(p)] for p in per) for i in range(max(map(len, per)))]
    raise TypeError(f"no group elements for {sym}")
