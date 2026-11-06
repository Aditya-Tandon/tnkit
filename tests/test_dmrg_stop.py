"""DMRG's relative energy-change stop (`rtol`) and how it meets subspace expansion."""
import numpy as np

from tnkit.api import DMRGOptions, dmrg
from tnkit.block_backend import to_block_mpo
from tnkit.models import tfim_mpo
from tnkit.symmetry import U1

from test_block_dmrg import REAL, SZ2, ground, sz2, xxz_opsum_mpo, xxz_sparse
from test_dmrg_expansion import neel


def test_rtol_on_by_default_and_ends_a_drifting_run():
    assert DMRGOptions().rtol == 1e-10
    W = [np.real(t) for t in tfim_mpo(64, 1.0)]
    E0, _, i0 = dmrg(W, 8, n_sweeps=10, tol=1e-13, rtol=0.0)
    E1, _, i1 = dmrg(W, 8, n_sweeps=10, tol=1e-13)
    assert i0["energies"][:i1["sweeps"]] == i1["energies"]          # same run, cut short
    assert i1["sweeps"] < i0["sweeps"] and abs(E1 - E0) < 1e-9 * abs(E0)
    assert len(i1["sweep_s"]) == i1["sweeps"]


def test_rtol_is_not_read_while_expanding():
    """One-site DMRG from the Neel state is stuck: its energy is flat, and any energy-change
    rule stops it there. With expansion the rule waits for two unbiased sweeps, by which time
    the state has left the plateau."""
    N = 10
    W = to_block_mpo(xxz_opsum_mpo(N, 1.0), [SZ2] * N, REAL, sym=U1())[0]
    E_ed = ground(xxz_sparse(N, 1.0), sz2(N) == 0)
    kw = dict(sites=1, n_sweeps=20, tol=0.0, rtol=1e-3)
    E0, _, i0 = dmrg(W, 32, init_mps=neel(N, REAL), expand=0, **kw)
    assert i0["sweeps"] == 2 and E0 > E_ed + 0.5
    E1, _, i1 = dmrg(W, 32, init_mps=neel(N, REAL), expand=4, **kw)
    assert i1["sweeps"] >= 6 and abs(E1 - E_ed) < 1e-6
