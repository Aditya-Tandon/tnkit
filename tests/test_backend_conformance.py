"""Every backend method against numpy, on the edge cases of a tensor-contraction interface:
zero-mode (scalar) operands and outputs, outer products, full contractions, batch (Hadamard)
indices, complex and mixed real/complex operands, and malformed calls. The block backend is
checked on the operations it implements and must refuse the rest."""
import numpy as np
import pytest

from tnkit import blocks as B
from tnkit.backend import get_backend
from tnkit.block_backend import BlockBackend

CUDA = pytest.param("torch-cuda", marks=pytest.mark.cuda)
DENSE = ["numpy", "torch-cpu", CUDA]


def _backend(spec):
    if spec.startswith("torch"):
        torch = pytest.importorskip("torch")
        if spec == "torch-cuda" and not torch.cuda.is_available():
            pytest.skip("no CUDA device")
    return get_backend(spec)


def _np(be, x):
    return np.asarray(be.to_numpy(x))


def _close(got, ref, tol=1e-12):
    ref = np.asarray(ref)
    assert np.shape(got) == ref.shape
    scale = max(1.0, float(np.max(np.abs(ref)))) if ref.size else 1.0
    assert np.max(np.abs(np.asarray(got) - ref), initial=0.0) <= tol * scale


def _rand(rng, shape, real=False):
    x = rng.normal(size=shape)
    return x if real else x + 1j * rng.normal(size=shape)


EINSUM_CASES = [
    ("ij,jk->ik", [(3, 4), (4, 5)]),            # matrix product
    ("ijk,kjl->il", [(2, 3, 4), (4, 3, 2)]),    # two contracted modes, permuted
    ("ij,jk->ki", [(3, 4), (4, 2)]),            # permuted output
    ("ij,ij->", [(3, 4), (3, 4)]),              # full contraction: scalar output
    ("i,j->ij", [(3,), (4,)]),                  # outer product
    ("ab,cd->cabd", [(2, 3), (2, 2)]),          # outer product, permuted
    (",ij->ij", [(), (3, 4)]),                  # scalar operand
    (",->", [(), ()]),                          # scalars only
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 2)]),   # batch index
    ("ab,ab->ab", [(3, 4), (3, 4)]),            # Hadamard product
    ("ab,ab->b", [(3, 4), (3, 4)]),             # batch index kept, one summed
    ("ijk->kij", [(2, 3, 4)]),                  # permutation only
    ("ij->", [(3, 4)]),                         # sum of all entries
    ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)]),  # three operands
    ("i,i->", [(0,), (0,)]),                    # zero extent
]


# real x complex cases that torch.einsum refuses (a sum over an index of two operands, of nonzero
# extent); it promotes the others, like numpy
TORCH_NO_PROMOTE = {"ij,jk->ik", "ijk,kjl->il", "ij,jk->ki", "ij,ij->", "bij,bjk->bik",
                    "ab,ab->b", "ij,jk,kl->il"}


@pytest.mark.parametrize("spec", DENSE)
@pytest.mark.parametrize("sub,shapes", EINSUM_CASES)
@pytest.mark.parametrize("mix", ["complex", "real-complex"])
def test_einsum(spec, sub, shapes, mix, request):
    be = _backend(spec)
    if mix == "real-complex" and be.is_torch and sub in TORCH_NO_PROMOTE:
        request.applymarker(pytest.mark.xfail(
            raises=RuntimeError, strict=True,
            reason="torch.einsum does not promote a real operand to complex in a contraction"))
    rng = np.random.default_rng(len(sub))
    xs = [_rand(rng, s, real=(mix == "real-complex" and i == 0)) for i, s in enumerate(shapes)]
    ops = [be.array(x) if mix == "complex" or i else _real(be, x) for i, x in enumerate(xs)]
    _close(_np(be, be.einsum(sub, *ops)), np.einsum(sub, *xs))


def _real(be, x):
    """`x` as a real array of the backend's library and device."""
    if be.is_torch:
        import torch
        return torch.as_tensor(np.asarray(x, dtype=np.float64), device=be.device)
    return np.asarray(x, dtype=np.float64)


@pytest.mark.parametrize("spec", DENSE)
@pytest.mark.parametrize("sub,shapes", [("ij,jk->ik", [(3, 4), (5, 2)]),   # extents disagree
                                        ("ij,jk->ik", [(3, 4)]),           # operand missing
                                        ("ij->ijk", [(3, 4)])])            # unknown output mode
def test_einsum_rejects_malformed_calls(spec, sub, shapes):
    be = _backend(spec)
    ops = [be.array(np.ones(s)) for s in shapes]
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        be.einsum(sub, *ops)


@pytest.mark.parametrize("spec", DENSE)
def test_other_methods(spec):
    be = _backend(spec)
    rng = np.random.default_rng(1)
    a, b = _rand(rng, (4, 3)), _rand(rng, (4, 3))
    A, Bm = be.array(a), be.array(b)
    _close(_np(be, be.conj(A)), a.conj())
    _close(_np(be, be.transpose(be.array(_rand(rng, (2, 3, 4))), (2, 0, 1))).shape, (4, 2, 3))
    _close(_np(be, be.diag(be.array(a[:, 0]))), np.diag(a[:, 0]))
    _close(complex(_np(be, be.vdot(A, Bm))), np.vdot(a, b))
    _close(float(_np(be, be.norm(A))), np.linalg.norm(a))
    _close(_np(be, be.zeros_like(A)), np.zeros_like(a))
    _close(_np(be, be.ones((2, 3))), np.ones((2, 3)))
    _close(_np(be, be.zeros((2, 0))), np.zeros((2, 0)))
    assert be.to_scalar(be.vdot(A, Bm)) == pytest.approx(np.vdot(a, b), rel=1e-13)
    _close(_np(be, be.stack([be.vdot(A, A), be.vdot(A, Bm)])), [np.vdot(a, a), np.vdot(a, b)])
    _close(_np(be, be.stack_axis([A, Bm], axis=0)), np.stack([a, b]))
    _close(_np(be, be.norm_axis(A, axis=0)), np.linalg.norm(a, axis=0))
    x = be.array(a[:, 0])
    _close(_np(be, be.mv(be.array(b.T), x)), b.T @ a[:, 0])
    _close(_np(be, be.vm(x, A)), a[:, 0] @ a)
    g = _np(be, be.guard_denom(be.array(np.array([0.0, 1e-300, 2.0]).astype(complex)).real))
    _close(g, [1e-200, 1e-200, 2.0])
    d, e = rng.normal(size=5), rng.normal(size=4)
    T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    from scipy.linalg import expm
    dv, ev = be.array(d), be.array(e)
    _close(_np(be, be.tridiag(dv, ev)), T)
    _close(_np(be, be.expm(be.array(T * 0.3j))), expm(T * 0.3j))
    _close(np.asarray(be.expm_host(be.array(T * 0.3j))), expm(T * 0.3j))
    _close(_np(be, be.expm_sym_tridiag_eigh(_real(be, d), _real(be, e), -0.2j)),
           expm(-0.2j * T))
    _close(_np(be, be.expm_sym_tridiag_eigh_batched(_real(be, np.stack([d, d])),
                                                    _real(be, np.stack([e, e])), -0.2j)),
           np.stack([expm(-0.2j * T)] * 2))
    _close(_np(be, be.expm_taylor(be.array(T * 0.3j), 6)), expm(T * 0.3j), tol=1e-11)
    U, S, Vh = be.svd(A)
    _close(_np(be, U) * _np(be, S) @ _np(be, Vh), a)
    _close(_np(be, S), np.linalg.svd(a, compute_uv=False))
    Ub, Sb, Vhb = be.svd_batched(be.array(np.stack([a, b])))
    _close(_np(be, Sb), np.linalg.svd(np.stack([a, b]), compute_uv=False))
    Q, R = be.qr(A)
    _close(_np(be, Q) @ _np(be, R), a)
    H = a.conj().T @ a
    w, V = be.eigh(be.array(H))
    _close(_np(be, w), np.linalg.eigvalsh(H))
    _close(_np(be, be.astype(A, "low")), a.astype(np.complex64), tol=1e-6)
    _close(_np(be, be.randn(np.random.default_rng(7), (3,))),
           get_backend("numpy").randn(np.random.default_rng(7), (3,)))
    be.sync()


# --- the block backend ---

def _block_pair(rng):
    p, _ = B.Leg.from_charges([0, 1])
    bond, _ = B.Leg.from_charges([-1, 0, 0, 1, 1, 2])
    a = B.random((bond, p, bond.flip()), 0, get_backend("numpy"), rng)
    b = B.random((bond, p.flip(), bond.flip()), 0, get_backend("numpy"), rng)
    return a, b


@pytest.mark.parametrize("spec", DENSE)
def test_block_backend(spec):
    dense = _backend(spec)
    bb = BlockBackend(dense)
    rng = np.random.default_rng(2)
    a0, b0 = _block_pair(rng)
    a, b = bb.array(a0), bb.array(b0)
    ad, bd = a0.to_dense(), b0.to_dense()
    D = lambda t: _np(dense, t.to_dense())
    cases = [("asb,bpc->aspc", (a, b), (ad, bd)),            # one contracted mode
             ("asb,csb->ac", (a, a.conj()), (ad, ad.conj())),  # two contracted, permuted
             ("asb,asb->", (a, a.conj()), (ad, ad.conj())),     # full contraction: scalar out
             ("asb,cpd->asbcpd", (a, b), (ad, bd)),           # outer product
             ("asb->bsa", (a,), (ad,))]                       # permutation
    for sub, ops, ref in cases:
        got = bb.einsum(sub, *ops)
        _close(D(got) if hasattr(got, "struct") else _np(dense, got), np.einsum(sub, *ref))
    _close(complex(_np(dense, bb.vdot(a, a))), np.vdot(ad, ad))
    _close(float(_np(dense, bb.norm(a))), np.linalg.norm(ad))
    _close(D(bb.conj(a)), ad.conj())
    _close(D(bb.transpose(a, (2, 0, 1))), ad.transpose(2, 0, 1))
    _close(D(bb.zeros_like(a)), np.zeros_like(ad))
    for sub, ops in [("asb,asb->asb", (a, a)),           # batch indices
                     ("asb,bpc,cqd->aspqd", (a, b, b)),   # three operands
                     ("asa->s", (a,))]:                   # trace
        with pytest.raises(NotImplementedError):
            bb.einsum(sub, *ops)
    with pytest.raises(NotImplementedError):
        bb.svd(a)
    with pytest.raises(ValueError, match="signs"):
        bb.einsum("asb,asb->", a, a)                     # same signs cannot contract
    loose = BlockBackend(dense, strict=False)
    _close(_np(dense, loose.einsum("asb,asb->asb", a, a)), ad * ad)
    assert loose.fallbacks["einsum"] == 1
