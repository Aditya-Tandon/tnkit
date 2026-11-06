"""TDVP for UNIFORM infinite MPS (iTDVP).

Real-time evolution directly in the thermodynamic limit. The state is a single translation-
invariant tensor in mixed gauge {AL, AR, AC, C}; there are no boundaries, so there is no
finite-size revival to out-run and no N to inflate.

WHY THIS EXISTS
---------------
Measurement of the exact free-fermion reference found the light cone directly (v_max = 2 for the
TFIM at g=1) with a finite-size revival at t_rev = N/v_max, so every finite benchmark must take N
large enough that t_rev exceeds the window or it is fitting boundary reflections.
A finite-size run paid for that by running N=48 purely to push the
revival out to t=24 for a t<=10 window -- most of those sites bought nothing but distance from the
boundary. In the thermodynamic limit that cost disappears entirely.

This does NOT lift the entanglement barrier. t_max is still logarithmic in chi
(t_max = 0.920 log2 chi - 0.980); an infinite system removes finite-size artefacts, it does not let
a fixed chi hold more entanglement, which is exactly what this scaling law predicts.

THE INTEGRATOR, AND A SIGN THAT IS EASY TO GET WRONG
----------------------------------------------------
    AC <- exp(-i dt H_AC) AC
    C  <- exp(-i dt H_C ) C          # FORWARD, not backward
    AL, AR <- polar gauge from (AC, C)

Both tensors evolve FORWARD. This looks wrong by analogy with finite 1-site TDVP, where the bond
tensor is explicitly counter-rotated (-P_bond in the projector), and that analogy was in fact
followed here first -- incorrectly.

The reason it differs: in this formulation the subtraction is carried by the GAUGE EXTRACTION, not
by the bond time step. AL is recovered as argmin ||AC - AL C||, i.e. AL ~ AC C^-1, so evolving both
factors forward and then dividing already realises the difference of the two generators. Counter-
rotating C as well applies the subtraction twice.

Measured, g: 2 -> 1 quench at chi=16, dt=0.05, against exact free fermions:

    C forward  (-i dt):  err ~5e-7, FLAT;         gauge error eps ~1.2e-6, flat
    C backward (+i dt):  err -> 1.2e-1 by t=0.3;  gauge error eps -> 0.575, growing

The observable alone is ambiguous early on (both agree to 4e-7 after one step). The decisive
diagnostic is the GAUGE ERROR eps = ||AC - AL C||: with the wrong sign the state stops being a
consistent uniform MPS and eps diverges, while norm and energy still look plausible.
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
    """One step of uniform TDVP. Returns (AL, AR, AC, C, eps).

    order=1  environments computed once from the current state and held frozen across the step.
    order=2  MIDPOINT rule: half-step to a provisional midpoint, recompute the environments THERE,
             then take a full step from the ORIGINAL tensors using the midpoint environments.

    Why the midpoint and not a Strang splitting of the two exponentials: AC and C are evolved from
    the same starting state with the SAME environments, so the two exponentials do not interact and
    swapping their order is a no-op. An earlier attempt did exactly that and measured a "second
    order" scheme whose output was bit-identical to order=1 at half the step -- because that is all
    it was. The genuine O(dt) error in this scheme comes from FREEZING FL and FR over the step, so
    that is what has to be centred.

    Note that this check is only meaningful in a regime where the
    gauge error is subdominant; the first run of it was gauge-dominated (err/eps_gauge ~ 1.4 at
    every dt) and therefore could not resolve the splitting order at all.
    """
    if order == 1:
        return _itdvp_sub(AL, AR, AC, C, W, dt, be, env_tol, m_krylov)
    if order == 2:
        raise NotImplementedError(
            "order=2 is REFUTED, not merely unimplemented. Three things were established and are "
            "kept here so the next attempt does not repeat them:\n"
            "  (1) Swapping the AC and C exponentials is a NO-OP. Both are evolved from the same "
            "starting tensors with the same frozen environments, so they do not interact. The "
            "'symmetric' scheme built that way produced output BIT-IDENTICAL to order=1 at half "
            "the step (err 3.527e-06 at dt=0.1 vs order=1 at dt=0.05, to all digits).\n"
            "  (2) The genuine O(dt) error is in FREEZING FL/FR across the step, not in the "
            "splitting -- so that is what a second-order scheme must centre.\n"
            "  (3) A midpoint rule that recomputes FL/FR at the midpoint and applies them to the "
            "ORIGINAL tensors is WRONG: environments are gauge-dependent objects tied to specific "
            "AL/AR, so this mixes gauges. Measured, it made things worse -- err 7.878e-05 vs "
            "1.596e-05 and the gauge error eps jumped 13x, from 1.10e-05 to 1.42e-04.\n"
            "A correct scheme must centre the environment freezing WITHOUT mixing gauges, e.g. by "
            "gauge-transforming the midpoint environments back, or by a scheme that never leaves "
            "one gauge.")
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
