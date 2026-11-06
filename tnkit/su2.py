"""SU(2) coupling coefficients, exact and cached.

Spins are passed as twice their value, so every label is a non-negative integer: j = 1/2 is 1,
j = 1 is 2. A projection m of spin j is likewise 2m, one of j, j - 2, ..., -j.

Each coefficient is computed in rational arithmetic (Racah's formulas, integer factorials) and
kept as a *signed square* s, a `Fraction` with coefficient = sign(s) sqrt(|s|). The float
versions are correct to about one ulp. Exact values, F symbols and CG arrays are cached.

Conventions (j, m in ordinary units below):
  * Clebsch-Gordan <j1 m1; j2 m2 | j m>: Condon-Shortley. Real, and <j1 j1; j2 (j - j1) | j j>
    > 0. Exchange symmetry <j2 m2; j1 m1 | j m> = (-1)^(j1 + j2 - j) <j1 m1; j2 m2 | j m>.
  * Wigner 6j {a b c; d e f}: Racah's single sum; invariant under the 24 tetrahedral
    symmetries.
  * Array layout of `cg_array(j1, j2, j)`: shape (2j1+1, 2j2+1, 2j+1), index k <-> m = j - k
    (highest weight first, the order of S^z = diag(j, j-1, ..., -j)).
  * Coupled basis of a tree: |(j1 j2)_j, m> = sum <j1 m1; j2 m2 | j m> |j1 m1> |j2 m2>.
  * F-symbol: the change of basis between the two trees with leaves a, b, c and total d,

        |((a b)_e c)_d, m> = sum_f F(a, b, c, d, e, f) |(a (b c)_f)_d, m>,

    F(a, b, c, d, e, f) = (-1)^(a + b + c + d) sqrt((2e + 1)(2f + 1)) {a b e; c d f}.
    Real and orthogonal in (e, f); independent of m.
  * R-symbol: swapping the two factors of a coupled pair,
    P |(a b)_c, m> = R(a, b, c) |(b a)_c, m> with P|x>|y> = |y>|x>, R(a, b, c) = (-1)^(a + b - c).
    Bosonic: R(a, b, c) R(b, a, c) = 1.
  * Frobenius-Schur indicator (-1)^(2j): +1 for integer spin, -1 for half-integer spin. It is
    the sign picked up when a leg is bent twice, kappa_a = (2a + 1) F(a, a, a, a, 0, 0).

Disallowed couplings (triangle or parity violated, |m| > j, m1 + m2 != m) give 0.

Provisional: not exported from `tnkit`; this module's API may change before 1.0.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial, sqrt

import numpy as np


def triangle(a: int, b: int, c: int) -> bool:
    """True when spins a/2, b/2, c/2 can couple: |a - b| <= c <= a + b and a + b + c even."""
    return a >= 0 and b >= 0 and c >= 0 and abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def _value(s: Fraction) -> float:
    return sqrt(s) if s >= 0 else -sqrt(-s)


def _signed_square(x: Fraction, rest: Fraction) -> Fraction:
    """The signed square of x sqrt(rest), rest >= 0."""
    return x * abs(x) * rest


def _ok_m(j: int, m: int) -> bool:
    return abs(m) <= j and (j + m) % 2 == 0


def cg_exact(j1: int, m1: int, j2: int, m2: int, j: int, m: int) -> Fraction:
    """<j1 m1; j2 m2 | j m> as a signed square (doubled labels)."""
    if (m1 + m2 != m or not triangle(j1, j2, j)
            or not (_ok_m(j1, m1) and _ok_m(j2, m2) and _ok_m(j, m))):
        return Fraction(0)
    return _cg(j1, m1, j2, m2, j, m)


@lru_cache(maxsize=None)
def _cg(j1, m1, j2, m2, j, m):
    f = factorial
    a, b, c = (j1 + j2 - j) // 2, (j1 - m1) // 2, (j2 + m2) // 2
    d, e = (j - j2 + m1) // 2, (j - j1 - m2) // 2
    total = Fraction(0)
    for k in range(max(0, -d, -e), min(a, b, c) + 1):
        total += Fraction((-1) ** k, f(k) * f(a - k) * f(b - k) * f(c - k) * f(d + k) * f(e + k))
    rest = Fraction((j + 1) * f((j + j1 - j2) // 2) * f((j - j1 + j2) // 2) * f(a)
                    * f((j + m) // 2) * f((j - m) // 2) * f((j1 - m1) // 2) * f((j1 + m1) // 2)
                    * f((j2 - m2) // 2) * f((j2 + m2) // 2),
                    f((j1 + j2 + j) // 2 + 1))
    return _signed_square(total, rest)


def _delta2(a: int, b: int, c: int) -> Fraction:
    f = factorial
    return Fraction(f((a + b - c) // 2) * f((a - b + c) // 2) * f((b + c - a) // 2),
                    f((a + b + c) // 2 + 1))


def _sixj_ok(a, b, c, d, e, f) -> bool:
    return triangle(a, b, c) and triangle(a, e, f) and triangle(d, b, f) and triangle(d, e, c)


def sixj_exact(a: int, b: int, c: int, d: int, e: int, f: int) -> Fraction:
    """{a b c; d e f} as a signed square (doubled labels)."""
    return _sixj(a, b, c, d, e, f) if _sixj_ok(a, b, c, d, e, f) else Fraction(0)


@lru_cache(maxsize=None)
def _sixj(a, b, c, d, e, f):
    triads = ((a, b, c), (a, e, f), (d, b, f), (d, e, c))
    lo = max(sum(t) // 2 for t in triads)
    hi = min((a + b + d + e) // 2, (a + c + d + f) // 2, (b + c + e + f) // 2)
    fac = factorial
    total = Fraction(0)
    for t in range(lo, hi + 1):
        den = fac((a + b + d + e) // 2 - t) * fac((a + c + d + f) // 2 - t) \
            * fac((b + c + e + f) // 2 - t)
        for x in triads:
            den *= fac(t - sum(x) // 2)
        total += Fraction((-1) ** t * fac(t + 1), den)
    rest = Fraction(1)
    for x in triads:
        rest *= _delta2(*x)
    return _signed_square(total, rest)


def clebsch_gordan(j1: int, m1: int, j2: int, m2: int, j: int, m: int) -> float:
    """<j1 m1; j2 m2 | j m> (doubled labels)."""
    return _value(cg_exact(j1, m1, j2, m2, j, m))


def wigner_6j(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    """{a b c; d e f} (doubled labels)."""
    return _value(sixj_exact(a, b, c, d, e, f))


def fsymbol_exact(a: int, b: int, c: int, d: int, e: int, f: int) -> Fraction:
    """F(a, b, c, d, e, f) as a signed square (doubled labels); see the module docstring."""
    s = sixj_exact(a, b, e, c, d, f)
    if s == 0:
        return s
    return s * (-1) ** ((a + b + c + d) // 2) * (e + 1) * (f + 1)


_F_CACHE: dict = {}


def fsymbol(a: int, b: int, c: int, d: int, e: int, f: int) -> float:
    """Recoupling coefficient |((a b)_e c)_d> -> |(a (b c)_f)_d> (doubled labels)."""
    key = (a, b, c, d, e, f)
    try:
        return _F_CACHE[key]
    except KeyError:
        pass
    if not _sixj_ok(a, b, e, c, d, f):          # invalid labels are not cached
        return 0.0
    value = _F_CACHE[key] = _value(fsymbol_exact(a, b, c, d, e, f))
    return value


def rsymbol(a: int, b: int, c: int) -> float:
    """Exchange phase of a x b -> c (doubled labels); 0 if c is not in a x b."""
    if not triangle(a, b, c):
        return 0.0
    return -1.0 if (a + b - c) // 2 % 2 else 1.0


@lru_cache(maxsize=None)
def cg_array(j1: int, j2: int, j: int) -> np.ndarray:
    """All <j1 m1; j2 m2 | j m> as a read-only array of shape (j1+1, j2+1, j+1), index k <-> m =
    j - 2k in doubled units (zeros when j is not in j1 x j2)."""
    out = np.zeros((j1 + 1, j2 + 1, j + 1))
    if triangle(j1, j2, j):
        for x in range(j1 + 1):
            for y in range(j2 + 1):
                m = j1 + j2 - 2 * (x + y)
                if abs(m) <= j:
                    out[x, y, (j - m) // 2] = clebsch_gordan(j1, j1 - 2 * x, j2, j2 - 2 * y, j, m)
    out.flags.writeable = False
    return out
