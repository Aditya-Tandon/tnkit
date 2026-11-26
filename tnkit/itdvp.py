"""TDVP for UNIFORM infinite MPS (iTDVP).

Real-time evolution directly in the thermodynamic limit. The state is a single translation-
invariant tensor in mixed gauge {AL, AR, AC, C}; there are no boundaries, so there is no
finite-size revival to out-run and no N to inflate.

WHY
---
Excitations spread at most at the Lieb-Robinson speed (v_max = 2 for the TFIM at g = 1), and a
finite chain of N sites shows a revival at about t_rev = N / v_max, so a finite run needs N
large enough that t_rev exceeds the times of interest, or it records boundary reflections. In
the thermodynamic limit that cost disappears.

It does not lift the entanglement barrier: the reachable time still grows only like log chi.
An infinite system removes finite-size artefacts; it does not let a fixed chi hold more
entanglement.

THE INTEGRATOR, AND A SIGN THAT IS EASY TO GET WRONG
----------------------------------------------------
    AC <- exp(-i dt H_AC) AC
    C  <- exp(-i dt H_C ) C          # FORWARD, not backward
    AL, AR <- polar gauge from (AC, C)

Both tensors evolve forward. By analogy with finite one-site TDVP, where the bond tensor is
explicitly evolved backward (-P_bond in the projector), this looks wrong, but it is not.

The reason it differs: in this formulation the subtraction is carried by the GAUGE EXTRACTION, not
by the bond time step. AL is recovered as argmin ||AC - AL C||, i.e. AL ~ AC C^-1, so evolving both
factors forward and then dividing already realises the difference of the two generators. Counter-
rotating C as well applies the subtraction twice.

With the wrong sign the state stops being a consistent uniform MPS: the gauge error
eps = ||AC - AL C|| grows without bound while the norm and energy still look plausible, and
the observables drift away from the exact result after a few steps. eps is the diagnostic to
watch.
"""
from __future__ import annotations

import numpy as np

from .backend import get_backend
from ._dmrg import apply_one_site
from .eigensolver import lanczos_expm
from ._idmrg import tfim_bulk_mpo
from .models import SX, SZ
from ._vumps import (vumps, _left_environment, _right_environment, _gauge, _real)


def _observables(AC, AR, be):
    """<sigma^z> on the centre site and <sigma^x sigma^x> on the centre bond."""
    sx, sz = be.array(SX), be.array(SZ)
    ACn = AC / be.norm(AC)
    z = _real(be, be.einsum("asb,st,atb->", ACn.conj(), sz, ACn))
    x1 = _real(be, be.einsum("asb,st,atb->", ACn.conj(), sx, ACn))
    th = be.einsum("asc,ctb->astb", AC, AR)
    th = th / be.norm(th)
    xx = _real(be, be.einsum("astb,su,tv,auvb->", th.conj(), sx, sx, th))
    return float(z), float(x1), float(xx)


def _entropy(C, be):
    s = be.svd(C)[1]
    s = np.asarray(_to_np(s), dtype=float)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 1e-16]
    return float(-np.sum(p * np.log(p)))


def _to_np(x):
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)


def _itdvp_sub(AL, AR, AC, C, W, dt, be, env_tol, m_krylov, FL=None, FR=None):
    """One first-order sub-step. If FL/FR are supplied they are used instead of being recomputed
    from the current state -- that is what makes the midpoint rule below possible."""
    if FL is None:
        FL, _ = _left_environment(AL, C, W, be, env_tol=env_tol)
    if FR is None:
        FR, _ = _right_environment(AR, C, W, be, env_tol=env_tol)

    shape_ac, D = AC.shape, C.shape[0]
    mv_ac = lambda x: apply_one_site(x.reshape(shape_ac), FL, W, FR, be).reshape(-1)
    # FORWARD, like AC -- see the module docstring. The gauge extraction below carries the
    # bond subtraction; counter-rotating C here would apply it twice and eps would diverge.
    mv_c = lambda x: be.einsum("xwa,ab,ywb->xy", FL, x.reshape(D, D), FR).reshape(-1)

    ACn = lanczos_expm(mv_ac, AC.reshape(-1), -1j * dt, be, m=m_krylov).reshape(shape_ac)
    Cn = lanczos_expm(mv_c, C.reshape(-1), -1j * dt, be, m=m_krylov).reshape(D, D)

    ACn = ACn / be.norm(ACn)
    Cn = Cn / be.norm(Cn)
    ALn, ARn, eps = _gauge(ACn, Cn, be)
    return ALn, ARn, ACn, Cn, eps


def itdvp_step(AL, AR, AC, C, W, dt, be, env_tol=1e-10, m_krylov=30, order=1):
    """One first-order step of uniform TDVP. Returns (AL, AR, AC, C, eps).

    The environments are computed once from the current state and held fixed across the step;
    that freezing is the source of the O(dt) error. ``order=2`` raises `NotImplementedError`:
    swapping the order of the AC and C exponentials changes nothing (both start from the same
    tensors with the same environments), and recomputing FL/FR at a midpoint and applying them
    to the original tensors mixes gauges, since environments belong to a particular AL/AR.
    """
    if order == 1:
        return _itdvp_sub(AL, AR, AC, C, W, dt, be, env_tol, m_krylov)
    if order == 2:
        raise NotImplementedError(
            "order=2 is not implemented. Swapping the AC and C exponentials is a no-op (both "
            "start from the same tensors with the same frozen environments), and the O(dt) error "
            "comes from freezing FL/FR across the step. Recomputing FL/FR at a midpoint and "
            "applying them to the original tensors mixes gauges, since environments belong to a "
            "particular AL/AR. A second-order scheme must centre the environments without mixing "
            "gauges.")
    raise ValueError(f"itdvp_step: order must be 1 or 2, got {order}")


def itdvp_quench(g0, g1, chi, tmax, dt=0.05, meas_every=1, backend="numpy",
                 vumps_iter=400, vumps_tol=1e-12, env_tol=1e-10, m_krylov=30,
                 verbose=False, order=1):
    """Ground state of the infinite TFIM at g0, then evolve under H(g1).

    Returns a dict of arrays: t, sz, sx, sxsx, S, eps.
    """
    be = get_backend(backend)
    gs = vumps(g0, chi, n_iter=vumps_iter, tol=vumps_tol, backend=backend)
    AL, AR, C = gs["AL"], gs["AR"], gs["C"]
    AC = be.einsum("asb,bc->asc", AL, C)
    W = be.array(tfim_bulk_mpo(g1))

    nsteps = int(round(tmax / dt))
    out = {k: [] for k in ("t", "sz", "sx", "sxsx", "S", "eps")}

    def record(t, eps):
        z, x, xx = _observables(AC, AR, be)
        out["t"].append(t); out["sz"].append(z); out["sx"].append(x)
        out["sxsx"].append(xx); out["S"].append(_entropy(C, be)); out["eps"].append(eps)

    record(0.0, float("nan"))
    for n in range(1, nsteps + 1):
        AL, AR, AC, C, eps = itdvp_step(AL, AR, AC, C, W, dt, be,
                                        env_tol=env_tol, m_krylov=m_krylov, order=order)
        if n % meas_every == 0:
            record(n * dt, float(eps))
            if verbose:
                print(f"  t={n*dt:6.3f}  sz={out['sz'][-1]:+.8f}  S={out['S'][-1]:.5f}",
                      flush=True)
    return {k: np.array(v) for k, v in out.items()}
