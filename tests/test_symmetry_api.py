"""The public symmetry API: no symmetry unless the caller passes one; an explicit U(1), Z_2 or
product symmetry runs on block tensors; a symmetry without charges, or charges without a symmetry,
is refused."""
import numpy as np
import pytest

import tnkit
from tnkit import api
from tnkit import (ProductSymmetry, U1, Zn, dmrg, get_backend, project_to_block_mps,
                   random_block_mps, tfim_mpo, to_block_mpo, to_block_mps)
from tnkit.block_backend import mpo_charges
from tnkit.blocks import BlockTensor
from tnkit.models_lgt2f import compile2f, schwinger2f_opsum, site_charges_2f, unpack_charge

REAL = get_backend("numpy-real")
KW = dict(n_sweeps=10, tol=1e-12, eig_tol=1e-12)
SZ2 = [1, -1]                                   # 2 S^z of (up, down)
NEW = ("Symmetry", "U1", "Zn", "ProductSymmetry", "to_block_mpo", "to_block_mps",
       "project_to_block_mps", "random_block_mps", "to_dense_mps", "BlockBackend")


def heisenberg_mpo(N):
    """Open Heisenberg chain sum S.S in S+ S- form, real."""
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    sz, eye = np.diag([0.5, -0.5]), np.eye(2)
    W = np.zeros((5, 5, 2, 2))
    W[0, 0] = W[4, 4] = eye
    W[0, 1], W[0, 2], W[0, 3] = 0.5 * sp, 0.5 * sp.T, sz
    W[1, 4], W[2, 4], W[3, 4] = sp.T, sp, sz
    return [W[:1]] + [W] * (N - 2) + [W[:, 4:]]


def test_exported_from_tnkit():
    from tnkit import block_backend, symmetry
    for name in ("Symmetry", "U1", "Zn", "ProductSymmetry"):
        assert getattr(tnkit, name) is getattr(symmetry, name) and name in tnkit.__all__
    for name in ("to_block_mpo", "to_block_mps", "project_to_block_mps", "random_block_mps",
                 "to_dense_mps", "BlockBackend"):
        assert getattr(tnkit, name) is getattr(block_backend, name) and name in tnkit.__all__
        assert name in block_backend.__all__
    assert not any(name.startswith("_") for name in tnkit.__all__ if name != "__version__")


def test_block_module_exports_exist():
    from tnkit import block_backend, block_graph, block_lanczos, blocks
    for mod in (blocks, block_backend, block_graph, block_lanczos):
        assert all(hasattr(mod, name) for name in mod.__all__), mod.__name__


def test_new_names_in_both_surfaces():
    assert set(NEW) <= set(api.CURATED) and set(NEW) <= set(tnkit.__all__)


def test_mpo_charges_needs_a_symmetry():
    W = [np.eye(2).reshape(1, 1, 2, 2)] * 4
    with pytest.raises(TypeError, match="'sym'"):
        mpo_charges(W, [SZ2] * 4)
    with pytest.raises(ValueError, match="needs site_charges"):
        mpo_charges(W, None, sym=U1())


def test_default_is_dense():
    W = heisenberg_mpo(8)
    E, psi, info = dmrg(W, 16, **KW)
    assert info["blocks"] is False
    assert all(isinstance(t, np.ndarray) for t in psi)
    assert dmrg(W, 16, charge=None, **KW)[0] == E


def _two_flavour(N=3):
    W, _ = compile2f(schwinger2f_opsum(N, 4.0, 0.3), N, real=True)
    pairs = [list(zip(*(x.tolist() for x in unpack_charge(c, N)[::-1])))
             for c in site_charges_2f(N)]
    return W, pairs


@pytest.mark.parametrize("case", ["u1", "z2", "u1xu1"])
def test_explicit_symmetry_runs_on_blocks(case):
    if case == "u1":
        W, sym, q, charge = heisenberg_mpo(8), U1(), [SZ2] * 8, 0
    elif case == "z2":
        W, sym, q, charge = [np.real(w) for w in tfim_mpo(8, 0.9)], Zn(2), [[0, 1]] * 8, 0
    else:
        (W, q), sym, charge = _two_flavour(), ProductSymmetry((U1(), U1())), (0, 0)
    E, psi, info = dmrg(to_block_mpo(W, q, REAL, sym=sym)[0], 32, charge=charge, **KW)
    assert info["blocks"] is True
    assert isinstance(psi[0], BlockTensor) and psi[0].struct.sym == sym
    assert E == pytest.approx(dmrg(W, 32, **KW)[0], abs=1e-9)     # the ground state's sector


def test_symmetry_without_charges_is_refused():
    W = heisenberg_mpo(6)
    psi = [np.ones((1, 2, 1))] * 6
    with pytest.raises(ValueError, match="needs site_charges"):
        to_block_mpo(W, None, REAL, sym=U1())
    with pytest.raises(ValueError, match="needs site_charges"):
        to_block_mps(psi, None, REAL, sym=U1())
    Wb, _ = to_block_mpo(W, [SZ2] * 6, REAL, sym=U1())
    with pytest.raises(ValueError, match="exactly one of charge="):
        dmrg(Wb, 8)
    with pytest.raises(TypeError, match="charge"):
        random_block_mps(Wb, 8, REAL)


def test_charges_without_symmetry_are_refused():
    W = heisenberg_mpo(6)
    psi = [np.eye(2)[s % 2].reshape(1, 2, 1) for s in range(6)]
    q = [SZ2] * 6
    with pytest.raises(TypeError, match="'sym'"):
        to_block_mpo(W, q, REAL)
    with pytest.raises(TypeError, match="must be a Symmetry"):
        to_block_mpo(W, q, REAL, sym=None)
    with pytest.raises(TypeError, match="'sym'"):
        to_block_mps(psi, q, REAL)
    with pytest.raises(TypeError, match="'sym'"):
        project_to_block_mps(psi, q, REAL, 8)
    with pytest.raises(ValueError, match="needs a block MPO"):
        dmrg(W, 8, charge=0)
    with pytest.raises(NotImplementedError, match="U\\(1\\) charges only"):
        to_block_mps(psi, [[0, 1]] * 6, REAL, sym=Zn(2))


def test_projection_needs_u1_and_charges():
    psi = [np.eye(2)[s % 2].reshape(1, 2, 1) for s in range(6)]
    with pytest.raises(NotImplementedError, match="U\\(1\\) charges only"):
        project_to_block_mps(psi, [[0, 1]] * 6, REAL, 8, sym=Zn(2))
    with pytest.raises(ValueError, match="needs site_charges"):
        project_to_block_mps(psi, None, REAL, 8, sym=U1())
