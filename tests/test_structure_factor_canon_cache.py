"""two_point's `canon` argument must be a bit-identical no-op.

structure_factor calls two_point N-1 times on the same state; each call used to
re-run a full right-canonicalisation sweep, making the measurement O(N^2) SVD
sweeps instead of O(N). The cached form must reproduce the uncached numbers
EXACTLY -- not to a tolerance -- or the speedup has silently changed physics.
"""
import numpy as np
import pytest

import tnkit.models as M
import tnkit.observables as ob
from tnkit.dmrg import dmrg

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def _uncached_structure_factor(mps, op):
    """The pre-optimisation code path, verbatim."""
    N = len(mps)
    total = float(np.sum(ob.local_expectations(mps, op @ op)))
    for i in range(N - 1):
        total += 2.0 * float(np.sum(ob.two_point(mps, op, op, i,
                                                 list(range(i + 1, N)))))
    return total / N ** 2, total


@pytest.mark.parametrize("N,chi", [(12, 16), (24, 24)])
def test_canon_cache_is_bit_identical(N, chi):
    """Caching the canonical form inside the pairwise sum must change nothing."""
    _, mps, _ = dmrg(M.tfim_mpo(N, 1.0), chi=chi, n_sweeps=40, tol=1e-13,
                     reorth="full", eig_stop="residual")
    m2_new, s_new = ob._structure_factor_pairwise(mps, SX)
    m2_old, s_old = _uncached_structure_factor(mps, SX)
    assert m2_new == m2_old, f"M2 drifted: {m2_new!r} vs {m2_old!r}"
    assert s_new == s_old


def test_two_point_explicit_canon_matches():
    N, chi = 16, 16
    _, mps, _ = dmrg(M.tfim_mpo(N, 1.0), chi=chi, n_sweeps=40, tol=1e-13,
                     reorth="full", eig_stop="residual")
    canon = ob._canon(mps)
    js = list(range(3, N))
    a = ob.two_point(mps, SX, SX, 2, js)
    b = ob.two_point(mps, SX, SX, 2, js, canon=canon)
    assert np.array_equal(a, b)


def test_structure_factor_matches_free_fermion_at_N48():
    """Guards the value itself, not just self-consistency (published value 0.2670743049)."""
    _, mps, _ = dmrg(M.tfim_mpo(48, 1.0), chi=32, n_sweeps=60, tol=1e-13,
                     reorth="full", eig_stop="residual")
    m2, _ = ob.structure_factor(mps, SX)
    assert abs(m2 - 0.2670743049) < 1e-8


@pytest.mark.parametrize("N,chi", [(16, 16), (32, 24), (64, 32)])
def test_accumulated_structure_factor_matches_pairwise(N, chi):
    """The O(N) accumulation must reproduce the O(N^2) pairwise sum.

    Different summation order, so this is a round-off comparison, not equality
    -- but it must sit at machine epsilon, not merely 'close'.
    """
    _, mps, _ = dmrg(M.tfim_mpo(N, 1.0), chi=chi, n_sweeps=60, tol=1e-13,
                     reorth="full", eig_stop="residual")
    m2_acc, s_acc = ob.structure_factor(mps, SX)
    m2_pair, s_pair = ob._structure_factor_pairwise(mps, SX)
    assert abs(m2_acc - m2_pair) / m2_pair < 1e-13
    assert abs(s_acc - s_pair) / s_pair < 1e-13
