"""DMRG: the sweep, its local eigensolve and truncation, and the MPS helpers it uses.

Each local update solves the ground-state problem of one or two sites with a matrix-free
eigensolver (`tnkit.eigensolver`) and, for two sites, cuts the optimised block back to bond
dimension chi with an SVD, which also lets the bonds grow.

Tensor axes (as in `tnkit.mps.MPS`):
  MPS  M[i]: (left_bond, phys, right_bond)               = (a, s, b)
  MPO  W[i]: (mpo_left, mpo_right, phys_out, phys_in)     = (wl, wr, s, s')
  envs L[i]: (bra_bond, mpo_bond, ket_bond) at the left edge of site i
       R[i]: (bra_bond, mpo_bond, ket_bond) at the right edge of site i

All heavy work (contractions, SVD, the eigensolve) goes through a backend object (numpy, or
torch on the CPU, CUDA or Apple MPS), so ``dmrg(..., backend="torch-cuda")`` runs on an NVIDIA
GPU in double precision. The default is 'auto': a real dtype when the problem is real, complex
otherwise (see `DMRGOptions` and `get_backend`). The contraction helpers take an optional `be`
(default numpy, complex).
"""
from __future__ import annotations
import dataclasses
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .eigensolver import (lanczos_ground, lanczos_ground_restart, lanczos_ground_mixed,
                          lobpcg_ground, scipy_host_ground, check_reorth_periodic)
from .backend import NumpyBackend, get_backend
from .blocks import BlockTensor, from_dense
from .block_backend import BlockBackend, random_block_mps
from .symmetric_backend import SymmetricBackend, edge_environments, random_symmetric_mps
from .symmetric_tensor import SymmetricTensor
from ._sparse import linalg_of, symmetry_of
from .expansion import enriched_split
from .legs import fuse, split
from . import invariants
from .mps import MPS, MPSLike, MutableMPS, Tensor
from . import _options
from . import linalg as tl

_EIGSOLVERS = {"lanczos": lanczos_ground, "lanczos-restart": lanczos_ground_restart,
               "lobpcg": lobpcg_ground,
               "scipy-host": scipy_host_ground}   # scipy's Lanczos on the host
_NP = NumpyBackend()
_BLOCK_EXPAND = 4        # opening expansion sweeps on block MPOs when expand=None


def _cutn_truncate(theta, chi):
    """Truncated SVD via cuTensorNet `tensor.decompose` (hybrid approach: tnkit's matrix-free eig +
    env kept, only the SVD/truncation swapped to NVIDIA-native). `theta` is a device 2D array
    (torch-cuda or cupy); decompose accepts/returns the same array type, so no host round-trip
    and no numpy/cupy seam. Returns (U[a s1, k], S[k], Vh[k, s2 b], discarded_weight).

    Requires the `cuquantum` (cuTensorNet) package and a CUDA-device `theta`; the whole point of
    the path is to stay on-device, so a host (numpy) array is a configuration error, not something
    to silently round-trip. Both conditions raise an actionable error rather than a cryptic one
    from deep inside cuTensorNet."""
    try:
        from cuquantum.tensornet.tensor import decompose, SVDMethod
    except ImportError as e:                               # cuquantum not installed in this env
        raise RuntimeError(
            "trunc='cutn' needs the cuquantum (cuTensorNet) package. Install it into the same "
            "env as the CUDA backend (e.g. `pip install cuquantum-python-cu12`), or use "
            "trunc='randomized-mixed' for a pure-tnkit GPU truncation.") from e
    if isinstance(theta, np.ndarray):                      # host array -> no device SVD to accelerate
        raise RuntimeError(
            "trunc='cutn' requires a CUDA-device two-site block (backend='torch-cuda'); got a "
            "host numpy array. Use trunc='svd' on the numpy/torch-cpu backends.")
    if hasattr(theta, "contiguous"):                       # torch tensors must be contiguous
        theta = theta.contiguous()
    U, S, Vh, info = decompose("ij->ix,xj", theta, method=SVDMethod(max_extent=int(chi)),
                               return_info=True)
    disc = float(getattr(info, "discarded_weight", 0.0) or 0.0)
    return U, S, Vh, disc


def _cut(s, chi, svd_tol=0.0, degenerate_tol=0.0, disc0=0.0, tot=None):
    """Where to cut a descending spectrum `s` (numpy). Returns (k, pinned, disc).

    The cap keeps min(chi, len(s)) values. `svd_tol` > 0 then keeps the fewest (at least one)
    whose relative dropped weight, disc0 + sum(s[k:]^2) / tot, stays within it. `disc0` is the
    relative weight a factorisation already dropped before returning `s`; `tot` is the total
    sum(sigma^2), sum(s^2) by default.

    degenerate_tol > 0 keeps or drops a group of values whole. The group at a cut is every
    value within degenerate_tol, relative to the larger, of the value next to the cut: the cap
    cut moves down to the start of the group of s[k], the `svd_tol` cut up to the end of the
    group of s[k-1]. A group that starts at s[0] is split instead, since moving down would keep
    nothing, and a cap cut among zero values stays where it is. `pinned` is True when the cap
    cut moved below the cap and `svd_tol` did not cut lower. `disc` is the relative dropped
    weight."""
    n = len(s)
    tot = tot or float(np.sum(s ** 2)) or 1.0
    k = min(chi, n)
    if degenerate_tol > 0 and k < n and s[k] > 0:
        j = k
        while j > 0 and s[j - 1] - s[k] <= degenerate_tol * s[j - 1]:
            j -= 1
        k = j or k
    pinned = k < min(chi, n)
    disc = disc0 + float(np.sum(s[k:] ** 2)) / tot
    if svd_tol > 0.0 and k > 1:
        tail = np.cumsum((s[:k] ** 2)[::-1])[::-1]       # tail[j] = sum_{l>=j} sigma_l^2
        drop = tail / tot + disc                          # relative loss if we keep j values
        ok = np.nonzero(drop[1:] <= svd_tol)[0]           # j = 1 .. k-1 (never drop to rank 0)
        if len(ok):
            c = j = int(ok[0]) + 1
            if degenerate_tol > 0:
                while j < k and s[c - 1] - s[j] <= degenerate_tol * s[c - 1]:
                    j += 1
            if j < k:
                k, pinned, disc = j, False, float(drop[j])
    return k, pinned, disc


def _truncate(theta, chi, trunc, be, rng=None, iso=None):
    """Rank-chi factorisation U, S, Vh of a two-site matrix `theta` + discarded weight.

    THE single truncation dispatch for the whole package: two-site DMRG (`dmrg`) and two-site
    TDVP (`tnkit.tdvp.tdvp2_evolve`) both call it, so a new truncation method is added once and
    both the statics and the dynamics get it. It was a closure inside `dmrg` until two-site TDVP
    needed the same dispatch in the time-evolution path.

    trunc : 'svd' | 'gram' | 'randomized' | 'randomized-mixed' | 'density' | 'cutn'. See `dmrg`
        for what each one is and when it is worth using.
    iso   : which factor the caller keeps as a canonical site: 'left' (U, sweeping right) or
        'right' (Vh, sweeping left). Required by 'gram', which makes only that factor an exact
        isometry (see `linalg.gram_truncation`); the SVD paths make both and ignore it.

    DISCARDED WEIGHT CONVENTION -- not uniform, and callers must know which they got:
    every tnkit path returns the ABSOLUTE sum(sigma_dropped^2); 'cutn' returns cuTensorNet's
    own `discarded_weight`, which is already normalised by sum(sigma^2). Divide by
    ||theta||^2 = sum(sigma^2) to compare the tnkit numbers against it.
    """
    if trunc == "svd":
        U, S, Vh = be.svd(theta)
        keep = min(chi, int(S.shape[0]))
        Snp = be.to_numpy(S)
        disc = float(np.sum(Snp[keep:] ** 2))
        return U[:, :keep], S[:keep], Vh[:keep], disc
    if trunc == "gram":                                # eigh of the smaller Gram matrix
        if iso is None:
            raise ValueError("trunc='gram' needs iso='left' or 'right'")
        return tl.gram_truncation(theta, chi, be, iso)
    if trunc == "randomized":                          # on-device range finder
        return tl.randomized_svd_device(theta, chi, be, rng=rng)
    if trunc == "randomized-mixed":                    # mixed precision: fp32 sketch + fp64 core
        return tl.randomized_svd_device(theta, chi, be, rng=rng, mixed=True)
    if trunc == "cutn":                                # cuTensorNet native truncated SVD
        return _cutn_truncate(theta, chi)
    if trunc not in tl.TRUNCATIONS:
        raise ValueError(f"unknown trunc {trunc!r}; expected one of 'svd', 'gram', 'randomized', "
                         f"'randomized-mixed', 'density', 'cutn'")
    Un, Sn, Vhn, disc = tl.TRUNCATIONS[trunc](be.to_numpy(theta), chi)   # host round-trip (density)
    return be.array(Un), be.array(Sn), be.array(Vhn), disc


# --------------------------------------------------------------------------- #
# contractions (be.einsum dispatches: numpy uses optimize=True -> BLAS; torch
# optimises internally and stays on-device)
# --------------------------------------------------------------------------- #
def apply_two_site(theta, L, W1, W2, R, be=None):
    """(H_eff theta)[a,s1,s2,b] for the two-site block. See module docstring."""
    be = be or _NP
    x1 = be.einsum("pwq,qijb->pwijb", L, theta)      # contract ket left bond
    x2 = be.einsum("wmsi,pwijb->pmsjb", W1, x1)      # W1 over (wl,s1')
    x3 = be.einsum("mrtj,pmsjb->pstrb", W2, x2)      # W2 over (wm,s2')
    out = be.einsum("drb,pstrb->pstd", R, x3)        # R over (wr,ket right)
    return out


def fused_two_site(L, W1, W2, R, be=None):
    """The two-site effective Hamiltonian as three matrices for `apply_two_site_fused`: L as
    (bra mpo, ket), the MPO pair as S[(s1 s2 wr), (wl s1' s2')] and R as (bra, mpo ket).
    Built once per local problem."""
    be = be or _NP
    p, wl, q = L.shape
    wr, d1, d2 = W2.shape[1], W1.shape[2], W2.shape[2]
    S = be.einsum("wmsi,mrtj->strwij", W1, W2).reshape(d1 * d2 * wr, wl * d1 * d2)
    return L.reshape(p * wl, q), S, R.reshape(R.shape[0], wr * R.shape[2])


def apply_two_site_fused(theta, Lm, S, Rm):
    """`apply_two_site` on the matrices of `fused_two_site` (dense tensors only): two GEMMs and
    one batched GEMM, with no transposes of the large intermediates."""
    q, d1, d2, b = theta.shape
    x = (Lm @ theta.reshape(q, -1)).reshape(-1, S.shape[1], b)
    x = S @ x                                        # (p, s1 s2 wr, b), batched over p
    return (x.reshape(-1, Rm.shape[1]) @ Rm.T).reshape(-1, d1, d2, Rm.shape[0])


def apply_one_site(M, L, W, R, be=None):
    """(H_eff M)[a,s,b] for a single-site tensor M=(a,s,b). Used by TDVP's one-site
    back-evolution. L=(bra,mpo,ket), W=(wl,wr,s,s'), R=(bra,mpo,ket)."""
    be = be or _NP
    x1 = be.einsum("aky,ysb->aksb", L, M)            # L[a,k,a'] M[a',s,b]
    x2 = be.einsum("klst,aktb->alsb", W, x1)         # W over (k, s')
    out = be.einsum("bly,alsy->asb", R, x2)          # R over (l, ket right)
    return out


def apply_zero_site(C, L, R, be=None):
    """(H_eff C)[a,b] for the BOND matrix C=(a,b) sitting between two sites.

    The zero-site effective Hamiltonian contracts the left and right environments directly over
    the MPO bond, with no W tensor in between. It is what one-site TDVP back-evolves: 1TDVP
    alternates a forward site update with a backward BOND update, which is what makes it exactly
    bond-dimension preserving (no SVD, no truncation).
    L=(bra,mpo,ket), R=(bra,mpo,ket).
    """
    be = be or _NP
    x1 = be.einsum("aky,yb->akb", L, C)              # L[a,k,a'] C[a',b]
    return be.einsum("bkz,akz->ab", R, x1)           # R[b,k,b'] over (k, b')


def update_left_env(L, A, W, be=None):
    """Grow the left environment past a left-canonical tensor A[i]=(a,s,c)."""
    be = be or _NP
    t1 = be.einsum("pwq,psd->wqsd", L, A.conj())
    t2 = be.einsum("wqsd,wrsy->qdry", t1, W)
    t3 = be.einsum("qdry,qye->dre", t2, A)
    return t3


def update_right_env(R, B, W, be=None):
    """Grow the right environment past a right-canonical tensor B[i]=(u,s,v)."""
    be = be or _NP
    t1 = be.einsum("xry,usx->ryus", R, B.conj())
    t2 = be.einsum("ryus,wrsp->yuwp", t1, W)
    t3 = be.einsum("yuwp,vpy->uwv", t2, B)
    return t3


def _edge_envs(M, mpo, be, L0=None, RN=None):
    """The boundary environments: given, or trivial (one element; for block tensors, with the
    legs of the outer bonds and MPO channels)."""
    if not getattr(M[0], "is_block", False):
        return (be.ones((1, 1, 1)) if L0 is None else be.array(L0),
                be.ones((1, 1, 1)) if RN is None else be.array(RN))
    if L0 is not None or RN is not None:
        raise ValueError("block tensors need open boundaries (L0 = RN = None)")
    if isinstance(M[0], SymmetricTensor):
        return edge_environments(M, mpo, be)
    one = be.dense.ones((1, 1, 1))
    a, b = M[0].legs[0], M[-1].legs[2]
    return (from_dense(one, (a, mpo[0].legs[0].flip(), a.flip()))[0],
            from_dense(one, (b, mpo[-1].legs[1].flip(), b.flip()))[0])


# --------------------------------------------------------------------------- #
# canonicalisation
# --------------------------------------------------------------------------- #
def random_mps(N: int, chi: int, d: int = 2, rng: Optional[Any] = None,
               be: Optional[Any] = None) -> List[Tensor]:
    """Random MPS. Tensors are seeded in numpy then moved to the backend, so the
    initial state is identical across backends (deterministic cross-checks)."""
    be = be or _NP
    rng = np.random.default_rng(0) if rng is None else rng
    dims = [1] + [min(chi, d ** min(i, N - i)) for i in range(1, N)] + [1]
    return [be.randn(rng, (dims[i], d, dims[i + 1]))
            for i in range(N)]


def right_canonicalize(M: MutableMPS, be: Optional[Any] = None) -> MutableMPS:
    """Make every tensor right-canonical (sum_s B B^H = I), center at site 0.

    Mutates `M` in place and returns the same object (an `MPS` in, an `MPS` out, with
    `center` 0).

    The center tensor is renormalised at every step. Without it, each site pushes a
    U*S factor (scale ~sqrt(d*chi) ~ 10) into its left neighbour, so the *global*
    scale of the running center grows geometrically — ~1e235 by N=200, and its
    sum-of-squares norm overflows to inf around N~200 (state collapses to 0 -> NaN).
    The overall scale is physically irrelevant here (DMRG works with a normalised
    state), and rescaling the center leaves the isometric Vh factors untouched, so
    right-canonical form is preserved exactly (verified: ||sum_s B B^H - I|| ~ 1e-15
    up to N=300). If an absolute norm were ever needed it would be accumulated as a
    separate log-domain scalar, never left to inflate the tensors."""
    be = be or _NP
    N = len(M)
    if getattr(M[0], "is_block", False):
        return _right_canonicalize_blocks(M, be)
    for i in range(N - 1, 0, -1):
        Dl, d, Dr = M[i].shape
        U, S, Vh = be.svd(M[i].reshape(Dl, d * Dr))
        M[i] = Vh.reshape(-1, d, Dr)
        M[i - 1] = be.einsum("asb,bc->asc", M[i - 1], U * S)   # column-scale U by S
        M[i - 1] = M[i - 1] / be.norm(M[i - 1])                # cap the running scale
    M[0] = M[0] / be.norm(M[0])
    if isinstance(M, MPS):
        M.center = 0
    return M


def _right_canonicalize_blocks(M: MutableMPS, be: Any) -> MutableMPS:
    """`right_canonicalize` for block tensors, by an LQ per block instead of an SVD."""
    for i in range(len(M) - 1, 0, -1):
        f, rec = fuse(M[i], ((0,), (1, 2)), be)
        Lm, Q = linalg_of(f).qr(f, right=True)
        M[i], Lm = split(Q, rec, be, legs=(1,)), split(Lm, rec, be, legs=(0,))
        M[i - 1] = be.einsum("asb,bc->asc", M[i - 1], Lm)
        M[i - 1] = M[i - 1] / be.norm(M[i - 1])
    M[0] = M[0] / be.norm(M[0])
    if isinstance(M, MPS):
        M.center = 0
    return M


# --------------------------------------------------------------------------- #
# DMRG
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DMRGOptions:
    """Settings of `dmrg`, apart from `mpo` and `chi`.

    Every field can also be passed to `dmrg` as a keyword, with the same default. One object
    can be reused across a scan, and `dataclasses.replace` gives variants that differ in one
    field. It is not hashable, and serialises to JSON only while every field is a scalar (not
    when `init_mps` holds a state or `backend` a backend object).

    Parameters
    ----------
    n_sweeps : int, default 8
        Maximum number of full sweeps (left to right, then back).
    tol, rtol : float, default 1e-10 each
        The run stops once the energy changes between sweeps by less than
        ``max(tol, rtol * abs(E))``, the looser of the two. So a `tol` below ``rtol * abs(E)``
        has no effect, a tighter stop needs ``rtol=0`` too, and ``tol=0, rtol=0`` runs all
        `n_sweeps`. E is the last local eigenvalue of the sweep; at a truncated `chi` it keeps
        moving by about the truncation error, which grows with N, so an absolute `tol` alone
        below that level runs every sweep.
    eig_tol : float, default 1e-9
        Tolerance of the local eigensolver, read against `eig_stop`.
    eig_stop : {'residual', 'energy'}, default 'residual'
        What `eig_tol` measures; see the notes of `dmrg`. Under 'energy' the state is
        converged only to about ``sqrt(eig_tol)``.
    eig_tol_warmup : float or None, default None
        If set, the local tolerance of the first `warmup_sweeps` sweeps.
    warmup_sweeps : int, default 2
        Sweeps that use `eig_tol_warmup`.
    eig : str, default 'lanczos-restart'
        Local eigensolver. 'lanczos-restart': thick-restart Lanczos, whose basis is capped
        (`eigensolver.lanczos_ground_restart`). 'lanczos': Lanczos with an unbounded basis.
        'lanczos-mixed': low-precision matvecs, then a double-precision refinement (dense
        tensors only). 'lobpcg'. 'scipy-host': scipy's Lanczos on the host around the device
        matvec, a reference for the cost of a host eigensolver.
    warm_start : bool, default True
        Start each local solve from the current block (True) or from a random vector.
    reorth : {'full', 'none'}, default 'full'
        Reorthogonalisation of 'lanczos'. 'periodic' raises with ``eig_stop='residual'`` and
        warns otherwise: it loses orthogonality on long local solves
        (`eigensolver.check_reorth_periodic`).
    max_iter : int or None, default None
        Cap on the Krylov dimension. None uses ``min(block size, max(200, 4 chi))``. A two-site
        block has about ``4 chi^2`` entries, and a fixed cap of 200 starves the local solve at
        large `chi`, so the energy stalls above its converged value.
    restart_m, restart_k : int, default 40 and 10
        Basis size and kept Ritz vectors of 'lanczos-restart'.
    reorth_alg : {'cgs2', 'cgs2v2', 'mgs'}, default 'cgs2'
        Reorthogonalisation kernel of 'lanczos-restart': two batched classical Gram-Schmidt
        passes ('cgs2v2' conjugates the vector instead of the basis), or sequential modified
        Gram-Schmidt.
    mixed_tol_lo : float, default 1e-4
        Residual at which 'lanczos-mixed' switches to double precision.
    trunc : str, default 'svd'
        How a two-site block is cut back to `chi`. 'svd': exact SVD. 'gram': GEMM and eigh of
        the smaller Gram matrix, faster on GPUs; double precision and dense tensors only; each
        cut matches 'svd' to rounding but not bit for bit, and singular values below about
        1e-8 of the largest are not resolved, so do not read small entanglement spectra from
        it. 'randomized', 'randomized-mixed': randomized SVD. 'cutn': cuTensorNet. 'density' is
        deprecated and means 'svd'. Block and symmetric tensors take 'svd' only.
    backend : str or backend, default 'auto'
        'numpy', 'torch-cpu', 'torch-cuda' or 'torch-mps', each optionally with '-real', a
        backend object, or 'auto': 'numpy-real' when `mpo` (and `init_mps`) are real to the
        tolerance of `get_backend`, 'numpy' otherwise. A real backend takes half the memory and
        fewer flops at the same accuracy on a real Hamiltonian. Pass 'numpy' for a complex
        state whatever the MPO.
    init_mps : MPS or None, default None
        Starting state, for example the result at a nearby coupling. None starts from a random
        MPS drawn with `seed`.
    charge : sector label or None, default None
        Block and symmetric MPOs: the total charge of the state sought, a label of the MPO's
        symmetry. Under SU(2) a label is twice the spin, so 0 is a singlet and 2 a triplet.
        Required unless `init_mps` is given (its charge is then used); None for a dense MPO.
    seed : int, default 0
        Seeds the random start and the randomized truncations.
    verbose : bool, default False
        Print the energy after every sweep.
    sites : {1, 2}, default 2
        Sites optimised per step. One-site steps are cheaper but change a bond's size or
        sectors only through `expand`. Both run on dense, block and symmetric tensors.
    expand : int or None, default None
        Number of opening sweeps with subspace expansion. None: 0 on a dense MPO and
        ``max(0, min(4, n_sweeps - 2))`` on a block or symmetric MPO; the value used is
        ``info["expand"]``. In those sweeps the bond basis left by each step is the `chi`
        leading eigenvectors of ``x x^H + alpha^2 P P^H``, where x is the optimised centre and
        P is the MPO and environment of the side that keeps the basis applied to x, less its
        part inside the span of x (Hubig et al., PRB 91, 155115 (2015); with ``sites=2`` this
        is White's density-matrix correction, except that White's P is not projected). The
        basis then gains the states H couples to, in their own sectors, so a sector the state
        lacks can fill. Must be less than `n_sweeps`. The stop is read only from the second
        sweep after the expansion ends, so a run does at least ``expand + 2`` sweeps.
        Expanding sweeps cut with an exact SVD whatever `trunc` says. Expansion can lead to a
        different fixed point, not always a lower one: at small `chi` it can end above the
        plain run. It cannot leave a start that is an exact eigenstate while `warm_start` is
        True.
    expand_alpha, expand_decay : float, default 3e-3 and 0.3
        alpha in sweep ``k < expand`` is ``expand_alpha * expand_decay**k``, and 0 afterwards,
        so the last sweeps are unbiased. P scales with H, so alpha is in units of 1/energy.
    """

    n_sweeps: int = 8
    tol: float = 1e-10
    trunc: str = "svd"
    eig_tol: float = 1e-9
    eig: str = "lanczos-restart"
    warm_start: bool = True
    reorth: Union[str, bool] = "full"
    eig_tol_warmup: Optional[float] = None
    warmup_sweeps: int = 2
    seed: int = 0
    init_mps: Optional[MPSLike] = None
    backend: Any = "auto"
    max_iter: Optional[int] = None
    verbose: bool = False
    restart_m: int = 40
    restart_k: int = 10
    mixed_tol_lo: float = 1e-4
    eig_stop: str = "residual"
    reorth_alg: str = "cgs2"
    charge: Any = None
    sites: int = 2
    expand: Optional[int] = None
    expand_alpha: float = 3e-3
    expand_decay: float = 0.3
    rtol: float = 1e-10


def dmrg(mpo: Sequence[Tensor], chi: Optional[int] = None,
         options: Optional[Union[DMRGOptions, Dict[str, Any]]] = None,
         **legacy_kwargs: Any) -> Tuple[float, MPS, Dict[str, Any]]:
    """Ground state of an MPO by DMRG, two-site by default.

    Parameters
    ----------
    mpo : sequence of tensors
        The Hamiltonian, one tensor per site with axes (left, right, out, in): dense arrays
        of shape ``(wl, wr, d, d)``, block tensors (`to_block_mpo`) or symmetric tensors
        (`tnkit.symmetric_backend.symmetric_mpo`). Sites are numbered from 0.
    chi : int
        Maximum bond dimension, >= 1. On symmetric tensors it counts multiplets, so the dense
        bond can be larger.
    options : DMRGOptions or dict, optional
        Settings; see `DMRGOptions` for every field, its default and its range.
    **legacy_kwargs
        Any field of `DMRGOptions`, passed flat. A keyword overrides the same field of
        `options`.

    Returns
    -------
    energy : float
        The local eigenvalue of the last step of the last sweep.
    mps : MPS
        The state: a list of rank-3 tensors ``(left, phys, right)`` on the backend used, with
        its orthogonality centre at site 0 (``mps.center == 0``) and every other site
        right-canonical. Dense states also have ``.entropies()`` and ``.to_numpy()``; block
        and symmetric states keep their tensor type (their ``to_dense_mps`` converts them).
    info : dict
        ``sweeps`` (number run), ``energies`` (energy after each sweep), ``sweep_s`` (wall
        time of each sweep, seconds), ``backend`` (its name), ``blocks`` (True for block and
        symmetric tensors), ``eig``, ``discarded_weight`` (summed over every cut),
        ``matvec_calls``, ``phase_time`` (seconds in 'eig', 'trunc' and 'env'),
        ``mixed_counts``, ``sites``, ``expand`` (expansion sweeps used) and ``expand_alphas``
        (alpha of each sweep, 0 where none).

    Notes
    -----
    Two call styles run the same code::

        E, mps, info = dmrg(mpo, chi=64, n_sweeps=20)                     # flat
        E, mps, info = dmrg(mpo, 64, DMRGOptions(n_sweeps=20))            # grouped

    Symmetries (provisional). With an MPO of block tensors (`to_block_mpo`, which takes the
    sector of every physical state and an abelian `Symmetry`), the sweep runs on block-sparse
    tensors and stays in the sector `charge`::

        Wb, _ = to_block_mpo(W, [[1, -1]] * N, get_backend("numpy-real"), sym=U1())  # 2 S^z
        E, mps, info = dmrg(Wb, 64, charge=0)

    The start is a random block MPS of total charge `charge` (or `init_mps`, a block MPS),
    and truncation keeps the `chi` largest singular values over all sectors together.
    `backend` names the dense backend under the blocks, or is a `BlockBackend`. Block runs
    expand the subspace in their opening sweeps by default (`expand`); with or without it, a
    block run at small `chi` can stop at a different fixed point than dense DMRG. An MPO of
    symmetric tensors (`tnkit.symmetric_backend.symmetric_mpo`, any symmetry, SU(2) included)
    runs the same way on a `SymmetricBackend`; `chi` then counts multiplets and `charge` is
    the total sector (under SU(2), twice the total spin). On block and symmetric tensors only
    ``trunc='svd'`` is supported, 'lanczos-mixed' is not, and `dmrg_block` and
    ``dmrg_tower(pair_solve=True)`` take dense MPOs only. One-site DMRG (``sites=1``) is
    supported on both.

    The local stopping rule (`eig_stop`). 'residual' (the default) stops the local
    eigensolver on the Ritz residual ``norm(H x - e x)``, the convergence of the eigenvector,
    which is what DMRG uses: x goes into the truncation and into the MPS. 'energy', the usual
    textbook rule, stops on the change of the Rayleigh quotient e. The Rayleigh quotient is
    quadratically accurate in the eigenvector error, so under 'energy' the state is converged
    only to about ``sqrt(eig_tol)``: at ``eig_tol=1e-10`` the residual is about 1e-5 against
    about 1e-11 under 'residual'. The gap grows with `chi`, and 'residual' costs little extra
    time because better local solves need fewer sweeps. Nothing in the output of a run under
    'energy' shows the difference, so compare runs only under the same rule.
    """
    # `chi` is a keyword for the drivers that call dmrg(mpo, **kw); both spellings
    # at once used to be a TypeError from the interpreter and stays one here,
    # rather than silently picking a bond dimension the caller did not mean.
    if chi is not None and "chi" in legacy_kwargs:
        raise TypeError("dmrg() got multiple values for argument 'chi'")
    if chi is None and "chi" not in legacy_kwargs:
        raise TypeError("dmrg() missing required argument 'chi' (bond dimension)")
    chi = legacy_kwargs.pop("chi", chi)
    o = _options.resolve(DMRGOptions, options, legacy_kwargs, "dmrg")
    tl.warn_if_deprecated(o.trunc)
    return _dmrg(mpo, chi, o)


def _check_sector(mpo, charge, init_mps, trunc, eig):
    """The block-MPO rules on `charge`, `init_mps`, `trunc` and `eig` of `dmrg` and
    `dmrg_excited`; returns whether `mpo` is a block MPO."""
    block = getattr(mpo[0], "is_block", False)
    if init_mps is not None and getattr(init_mps[0], "is_block", False) != block:
        raise ValueError("a block MPO needs a block init_mps, and a dense MPO a dense one")
    if block and (trunc != "svd" or eig == "lanczos-mixed"):
        raise ValueError("a block MPO needs trunc='svd' and an eig other than 'lanczos-mixed'")
    if charge is not None and not block:
        raise ValueError("charge= needs a block MPO: convert the MPO with to_block_mpo(W, "
                         "site_charges, be, sym=...) first")
    if block and (charge is None) == (init_mps is None):
        raise ValueError("a block MPO needs exactly one of charge= (the total charge of the state "
                         "sought) and a block init_mps (which fixes the charge)")
    if charge is not None:
        try:
            symmetry_of(mpo[0]).label(charge)
        except TypeError:
            raise ValueError(f"charge={charge!r} is not a sector label of "
                             f"{symmetry_of(mpo[0])}") from None
    return block


def _block_backend(spec: Any, mpo: Sequence[Any], init_mps: Optional[MPSLike]):
    """The `BlockBackend` for block tensors, or the `SymmetricBackend` for symmetric ones:
    `spec` is one, or names the dense backend under it ('auto': real when the data are real)."""
    wrap = SymmetricBackend if isinstance(mpo[0], SymmetricTensor) else BlockBackend
    if isinstance(spec, wrap):
        return spec
    if isinstance(spec, (BlockBackend, SymmetricBackend)):
        raise ValueError(f"{type(mpo[0]).__name__} operands need a {wrap.__name__}")
    data = lambda ts: None if ts is None else [t.data for t in ts]
    return wrap(get_backend(spec, mpo=data(mpo), init_mps=data(init_mps)))


def _dmrg(mpo: Sequence[Tensor], chi: int, o: DMRGOptions) -> Tuple[float, MPS, Dict[str, Any]]:
    """The sweep itself. The options are unpacked into the names the body has always used,
    so the dense arithmetic is that of the flat-signature version; block tensors take the
    branches marked `block`."""
    n_sweeps, tol, trunc, eig_tol = o.n_sweeps, o.tol, o.trunc, o.eig_tol
    eig, warm_start, reorth = o.eig, o.warm_start, o.reorth
    eig_tol_warmup, warmup_sweeps = o.eig_tol_warmup, o.warmup_sweeps
    seed, init_mps, backend = o.seed, o.init_mps, o.backend
    max_iter, verbose = o.max_iter, o.verbose
    restart_m, restart_k, mixed_tol_lo = o.restart_m, o.restart_k, o.mixed_tol_lo
    eig_stop, reorth_alg = o.eig_stop, o.reorth_alg

    invariants.entry(init_mps, mpo)
    block = _check_sector(mpo, o.charge, init_mps, trunc, eig)
    sites = o.sites
    if sites not in (1, 2):
        raise ValueError(f"sites must be 1 or 2, got {sites!r}")
    if sites == 1 and eig == "lanczos-mixed":
        raise ValueError("sites=1 does not take eig='lanczos-mixed'")
    expand = o.expand
    if expand is not None and (isinstance(expand, bool)
                               or not isinstance(expand, (int, np.integer))):
        raise TypeError(f"expand must be an int or None, got {expand!r}")
    if expand is None:
        expand = min(_BLOCK_EXPAND, max(0, n_sweeps - 2)) if block else 0
    if not 0 <= expand < n_sweeps:
        raise ValueError(f"expand must be in [0, n_sweeps), so that the last sweep is unbiased; "
                         f"got expand={expand!r}, n_sweeps={n_sweeps!r}")
    if expand and not (o.expand_alpha > 0 and 0 < o.expand_decay <= 1):
        raise ValueError("expand needs expand_alpha > 0 and 0 < expand_decay <= 1 "
                         "(pass expand=0 to turn expansion off)")
    be = (_block_backend(backend, mpo, init_mps) if block
          else get_backend(backend, mpo=mpo, init_mps=init_mps))
    N = len(mpo)
    d = mpo[0].shape[2]
    mpo = [be.array(W) for W in mpo]
    if init_mps is not None:
        M = right_canonicalize([be.array(m) for m in init_mps], be)
    elif block:
        start = random_symmetric_mps if isinstance(mpo[0], SymmetricTensor) else random_block_mps
        M = right_canonicalize(start(mpo, chi, be.dense, o.charge, np.random.default_rng(seed)),
                               be)
    else:
        M = right_canonicalize(random_mps(N, chi, d, np.random.default_rng(seed), be), be)
    if eig not in _EIGSOLVERS and eig != "lanczos-mixed":
        raise ValueError(f"unknown eig solver {eig!r}; expected one of "
                         f"{sorted(_EIGSOLVERS) + ['lanczos-mixed']}")
    eig_fn = _EIGSOLVERS.get(eig)   # None for 'lanczos-mixed' (handled inline in solve_block)
    # reorth='periodic' is refused for every solver (`check_reorth_periodic`); 'lanczos'
    # checks it itself, and the other solvers ignore `reorth`, so the check is made here once.
    if eig != "lanczos":
        check_reorth_periodic(reorth, eig_stop)
    mixed_counts = {"lo": 0, "hi": 0}
    _rng = np.random.default_rng(seed + 1)

    # environments
    L = [None] * (N + 1)
    R = [None] * (N + 1)
    L[0], R[N] = _edge_envs(M, mpo, be)
    for i in range(N - 1, 0, -1):
        R[i] = update_right_env(R[i + 1], M[i], mpo[i], be)
    # R[i] is the env at the right edge of site i-1; the block (i,i+1) uses L[i], R[i+2]

    energies = []
    total_disc = 0.0
    matvec_calls = 0
    phase_t = {"eig": 0.0, "trunc": 0.0, "env": 0.0}

    def truncate(theta, iso):
        if block:
            return linalg_of(theta).truncated_svd(theta, chi)   # chi largest over all sectors
        return _truncate(theta, chi, trunc, be, _rng, iso)   # shared with tdvp2_evolve

    def eigsolve(x0, apply, cur_eig_tol, apply_lo=None):
        """Lowest eigenpair of the local problem `apply`, started from `x0` (a tensor)."""
        nonlocal matvec_calls
        shape = tuple(int(n) for n in x0.shape)
        counter = {"n": 0}
        if block:                                  # the solver sees the flat block buffer
            x0 = x0.canonical()
            wrap = (x0.with_data if isinstance(x0, SymmetricTensor)
                    else (lambda d, s=x0.struct: BlockTensor(s, d)))
            def matvec(x):
                counter["n"] += 1
                return apply(wrap(x)).canonical().data
        else:
            def matvec(x):
                counter["n"] += 1
                return apply(x.reshape(shape)).reshape(-1)
        if apply_lo is not None:
            def matvec_lo(x):
                counter["n"] += 1
                return apply_lo(x.reshape(shape)).reshape(-1)

        local_dim = int(x0.data.shape[0]) if block else int(np.prod(shape))
        if warm_start:
            v0 = x0.data if block else x0.reshape(-1)
        else:
            v0 = be.randn(_rng, local_dim)
        if eig == "lanczos":
            eig_kwargs = {"reorth": reorth, "stop": eig_stop}
        elif eig == "lanczos-restart":
            eig_kwargs = {"m": restart_m, "k": restart_k, "reorth_alg": reorth_alg}
        else:
            eig_kwargs = {}
        mi = max_iter if max_iter is not None else min(local_dim, max(200, 4 * chi))
        be.sync(); t0 = time.perf_counter()
        if eig == "lanczos-mixed":
            e, x, minfo = lanczos_ground_mixed(
                matvec_lo, matvec, v0, be, tol=cur_eig_tol, max_iter=mi,
                m=restart_m, k=restart_k, tol_lo=mixed_tol_lo, counts=mixed_counts)
        else:
            e, x, minfo = eig_fn(matvec, v0, be, tol=cur_eig_tol, max_iter=mi, **eig_kwargs)
        be.sync(); phase_t["eig"] += time.perf_counter() - t0
        matvec_calls += counter["n"]
        return e, (wrap(x) if block else x.reshape(shape))

    def solve_block(i, sweep_dir, cur_eig_tol, alpha):
        nonlocal total_disc
        W1, W2 = mpo[i], mpo[i + 1]
        Li, Ri = L[i], R[i + 2]
        theta0 = be.einsum("asc,ctb->astb", M[i], M[i + 1])
        if block:
            apply = lambda t: apply_two_site(t, Li, W1, W2, Ri, be)
        else:
            op = fused_two_site(Li, W1, W2, Ri, be)
            apply = lambda t: apply_two_site_fused(t, *op)
        apply_lo = None
        # Mixed-precision inner matvec: operands cast to complex64 once per block; torch.einsum
        # then runs the contraction on tensor cores, with the eigensolver's fp64 accumulation
        # preserved. The fp64 `apply` is the guardrail/refinement operator.
        if eig == "lanczos-mixed":
            Li_lo, W1_lo, W2_lo, Ri_lo = (be.astype(Li, "low"), be.astype(W1, "low"),
                                          be.astype(W2, "low"), be.astype(Ri, "low"))
            apply_lo = lambda t: be.astype(apply_two_site(be.astype(t, "low"), Li_lo, W1_lo,
                                                          W2_lo, Ri_lo, be), "high")
        e, x = eigsolve(theta0, apply, cur_eig_tol, apply_lo)

        t0 = time.perf_counter()
        if alpha and sweep_dir == "right":
            P = be.einsum("wmsi,pwijb->psmjb", W1, be.einsum("pwq,qijb->pwijb", Li, x))
            M[i], C, disc = enriched_split(x, P, alpha, chi, be, True)
            M[i + 1] = C / be.norm(C)
        elif alpha:
            P = be.einsum("wrjt,astdr->aswjd", W2, be.einsum("drb,astb->astdr", Ri, x))
            M[i + 1], C, disc = enriched_split(x, P, alpha, chi, be, False)
            M[i] = C / be.norm(C)
        else:
            mat, rec = fuse(x, ((0, 1), (2, 3)), be)   # (a s1, s2 b)
            U, S, Vh, disc = truncate(mat, "left" if sweep_dir == "right" else "right")
            S = S / be.norm(S)                         # keep state normalised
            if sweep_dir == "right":
                M[i] = split(U, rec, be, legs=(0,))
                SV = Vh.scale(S, 0) if block else S[:, None] * Vh     # row-scale Vh by S
                M[i + 1] = split(SV, rec, be, legs=(1,))
            else:
                M[i] = split(U.scale(S, 1) if block else U * S, rec, be, legs=(0,))  # U by S
                M[i + 1] = split(Vh, rec, be, legs=(1,))
        be.sync(); phase_t["trunc"] += time.perf_counter() - t0
        total_disc += disc

        t0 = time.perf_counter()
        if sweep_dir == "right":
            L[i + 1] = update_left_env(L[i], M[i], W1, be)
        else:
            R[i + 1] = update_right_env(R[i + 2], M[i + 1], W2, be)
        be.sync(); phase_t["env"] += time.perf_counter() - t0
        return e

    def solve_site(i, sweep_dir, cur_eig_tol, alpha):
        """One-site step at site i; the centre then moves to i + 1 (right) or i - 1 (left)."""
        nonlocal total_disc
        W, Li, Ri = mpo[i], L[i], R[i + 1]
        e, x = eigsolve(M[i], lambda t: apply_one_site(t, Li, W, Ri, be), cur_eig_tol)
        t0 = time.perf_counter()
        if alpha:
            if sweep_dir == "right":
                P = be.einsum("klst,aktb->aslb", W, be.einsum("aky,ysb->aksb", Li, x))
            else:
                P = be.einsum("klst,atbl->aksb", W, be.einsum("bly,asy->asbl", Ri, x))
            A, C, disc = enriched_split(x, P, alpha, chi, be, sweep_dir == "right")
        elif sweep_dir == "right":
            mat, rec = fuse(x, ((0, 1), (2,)), be)
            U, S, Vh, disc = truncate(mat, "left")
            A = split(U, rec, be, legs=(0,))
            C = split(Vh.scale(S, 0) if block else S[:, None] * Vh, rec, be, legs=(1,))
        else:
            mat, rec = fuse(x, ((0,), (1, 2)), be)
            U, S, Vh, disc = truncate(mat, "right")
            A = split(Vh, rec, be, legs=(1,))
            C = split(U.scale(S, 1) if block else U * S, rec, be, legs=(0,))
        if sweep_dir == "right":
            M[i], nxt = A, be.einsum("kb,bsc->ksc", C, M[i + 1])
            M[i + 1] = nxt / be.norm(nxt)
        else:
            M[i], nxt = A, be.einsum("asb,bk->ask", M[i - 1], C)
            M[i - 1] = nxt / be.norm(nxt)
        be.sync(); phase_t["trunc"] += time.perf_counter() - t0
        total_disc += disc
        t0 = time.perf_counter()
        if sweep_dir == "right":
            L[i + 1] = update_left_env(L[i], M[i], W, be)
        else:
            R[i] = update_right_env(R[i + 1], M[i], W, be)
        be.sync(); phase_t["env"] += time.perf_counter() - t0
        return e

    if sites == 2:
        sweep_right, sweep_left = range(0, N - 1), range(N - 2, -1, -1)
        step = solve_block
    else:
        sweep_right, sweep_left = range(0, N - 1), range(N - 1, 0, -1)
        step = solve_site
    alphas, sweep_s = [], []
    prev_E = None
    for sweep in range(n_sweeps):
        t_sweep = time.perf_counter()
        # inner-tolerance scheduling: loose local solves in early sweeps (default off)
        cur_eig_tol = (eig_tol_warmup if (eig_tol_warmup is not None
                                          and sweep < warmup_sweeps) else eig_tol)
        alpha = o.expand_alpha * o.expand_decay ** sweep if sweep < expand else 0.0
        alphas.append(alpha)
        for i in sweep_right:                       # left -> right
            E = step(i, "right", cur_eig_tol, alpha)
        for i in sweep_left:                        # right -> left
            E = step(i, "left", cur_eig_tol, alpha)
        invariants.sweep(M, mpo)
        energies.append(E)
        be.sync(); sweep_s.append(time.perf_counter() - t_sweep)
        if verbose:
            print(f"  sweep {sweep+1:2d}: E = {E:.12f}  (chi={chi}, trunc={trunc}, {be.name})")
        if prev_E is not None and abs(E - prev_E) < max(tol, o.rtol * abs(E)):
            break
        prev_E = None if alpha else E

    info = {"sweeps": sweep + 1, "energies": energies, "backend": be.name, "blocks": block,
            "eig": eig, "discarded_weight": total_disc, "matvec_calls": matvec_calls,
            "phase_time": phase_t, "mixed_counts": mixed_counts, "sites": sites,
            "expand": expand, "expand_alphas": alphas, "sweep_s": sweep_s}
    return E, MPS(M, center=0), info             # the last sweep ends at site 0
