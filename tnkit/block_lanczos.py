"""The Lanczos exponential on block tensors with less host work per iteration (on by default).

`lanczos_expm` returns, bit for bit, what `eigensolver.lanczos_expm` returns for a `BlockTensor`
vector and a `BlockBackend`. When switched on, `eigensolver.lanczos_expm` hands block vectors to it,
so its callers, and anything that wraps it, are unchanged. The kernels, their operands and their
order are the eager loop's; only the host work around them changes:
  - when two operands share one layout (the usual case after the first iteration), the vector
    algebra runs on their flat buffers, with no `BlockTensor` or `BlockBackend` call per
    operation; operands in different layouts are aligned by `BlockTensor` as before;
  - a projection coefficient multiplies a buffer from the buffer's side (``u * c``), so torch does
    not first try, and fail, ``Tensor * BlockTensor``;
  - alpha stays on the device and is read once per iteration together with beta, not on its own
    before the re-orthogonalisation, so the host issues the rest of the iteration without waiting
    for the matvec.
Alpha as a 0-d tensor multiplies a complex buffer as the Python float does: both become
``complex(alpha, 0)``. On CUDA torch applies a host scalar as the left operand, so the device
scalar goes on the left there.

With `block_graph` on as well, `matvec` is the solve's graph wrapper (`block_graph.solve`), which
this path calls exactly as the eager loop does, so the two compose: graphs replay the matvecs and
this path issues the work between them.

Switch: `set_enabled`, or ``TNKIT_BLOCK_LANCZOS_FAST=0`` to use the eager loop (values other than
``0`` and ``1`` raise).
"""
from __future__ import annotations

import os

import numpy as np

from . import eigensolver
from .backend import TorchBackend
from .blocks import BlockTensor

__all__ = ["set_enabled", "enabled"]


def _from_env() -> bool:
    v = os.environ.get("TNKIT_BLOCK_LANCZOS_FAST", "")
    if v not in ("", "0", "1"):
        raise ValueError(f"TNKIT_BLOCK_LANCZOS_FAST must be 0 or 1, got {v!r}")
    return v != "0"


_ON = _from_env()


def set_enabled(on: bool) -> None:
    """Turn this path on or off (default: on, or off with ``TNKIT_BLOCK_LANCZOS_FAST=0``)."""
    global _ON
    _ON = bool(on)


def enabled() -> bool:
    return _ON


def _kernels(dense, x):
    """(vdot, norm, read) for buffers like `x`; ``read(a, b)`` returns two 0-d values as floats
    with one device-to-host copy."""
    if isinstance(x, np.ndarray) or not (type(dense).vdot is TorchBackend.vdot
                                         and type(dense).norm is TorchBackend.norm):
        return dense.vdot, dense.norm, lambda a, b: (float(np.real(a)), float(np.real(b)))
    import torch
    return torch.vdot, torch.linalg.norm, lambda a, b: torch.stack((a, b)).tolist()


def _pair(a, sa, b, sb):
    """The buffers `a` and `b` (layouts `sa`, `sb`) in one layout, and that layout."""
    if sa is sb:
        return a, b, sa
    return BlockTensor(sa, a)._aligned(BlockTensor(sb, b))


def lanczos_expm(matvec, v, tau, ops, m=30, tol=1e-12, adaptive=None, info=None, est_ref_j=2):
    """`eigensolver.lanczos_expm`'s solve for a `BlockTensor` `v` and a `BlockBackend` `ops`: the
    same arguments, result and `info`."""
    dense = ops.dense
    vdot, norm, read = _kernels(dense, v.data)
    left = bool(getattr(v.data, "is_cuda", False))
    beta0 = dense.to_scalar(norm(v.data)).real
    if not np.isfinite(beta0) or beta0 < 1e-300:
        return v
    mon = eigensolver._ExpmMonitor(tau, beta0, tol, adaptive, info, est_ref_j)
    V = [BlockTensor(v.struct, v.data / beta0)]
    alphas, betas = [], []
    beta = 0.0
    ps, pd = v.struct, dense.zeros_like(v.data)
    capped = False
    for k in range(m):
        y = matvec(V[-1])
        a, b, s = _pair(y.data, y.struct, pd * beta, ps)
        w = a - b
        u = V[-1]
        a, b, _ = _pair(u.data, u.struct, w, s)
        alpha = vdot(a, b).real
        a, b, s = _pair(w, s, alpha * u.data if left else u.data * alpha, u.struct)
        w = a - b
        for u in V:                                  # full reorthogonalisation
            a, b, _ = _pair(u.data, u.struct, w, s)
            a, b, s = _pair(w, s, u.data * vdot(a, b), u.struct)
            w = a - b
        alpha, beta = read(alpha, norm(w))
        alphas.append(alpha)
        stop = mon.check(k, alphas, betas, beta)
        if stop == "breakdown":
            break
        betas.append(beta)
        ps, pd = V[-1].struct, V[-1].data
        V.append(BlockTensor(s, w / beta))
        if stop:
            break
    else:
        capped = bool(mon.adaptive)
    c = mon.finish(alphas, betas, capped)
    s, out = V[0].struct, dense.zeros_like(V[0].data)
    for coeff, u in zip(c, V):
        a, b, s = _pair(out, s, u.data * complex(coeff), u.struct)
        out = a + b
    return BlockTensor(s, out)
