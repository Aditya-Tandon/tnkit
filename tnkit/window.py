"""A nonuniform WINDOW on an infinite, absorbing background (Milsted et al. 2013).

A finite window of tensors is embedded in an infinite uniform iMPS background, with the boundary
environments of that background held FIXED. Quasiparticles that leave the window are absorbed into
the background instead of reflecting off a hard edge.

WHY
---
Measurement of the exact free-fermion reference found the light cone (v_max = 2 for the TFIM at
g=1) and the finite-size revival t_rev = N/v_max. Every finite benchmark since has had to buy
sites purely as distance from the boundary -- one such run used N=48
to push the revival to t=24 for a t<=10 window. Here the window covers only the region that is
actually nonuniform.

WHAT THE BACKGROUND MUST BE
---------------------------
The construction is only valid when the background is STATIONARY, i.e. an eigenstate of the SAME
Hamiltonian the window evolves under. So this handles a LOCAL perturbation on a ground state --
not a global quench, where the background itself would evolve and a fixed boundary environment
would be simply wrong. That is why the thesis's spin-flip protocol does not apply here:
(x)|+> is not an eigenstate of H(g>0), so it is a global quench, not a local perturbation on a
stationary background.
"""
from __future__ import annotations

import numpy as np

from .backend import PlainEinsumNumpyBackend, get_backend
from .mps import MPS
from .observables import _profile
from ._idmrg import tfim_bulk_mpo
from ._vumps import vumps, _left_environment, _right_environment


def build_window(g, chi, n_left, n_right, backend="numpy", vumps_iter=400, vumps_tol=1e-12,
                 env_tol=1e-10, op=None, gs=None, mpo=None, bond_h=None, triple_h=None):
    """Ground state of the infinite TFIM at g, with a local operator applied at the centre.

    Returns (window_tensors, FL, FR, W, background_gs).

    window = [AL]*n_left + [AC (op applied)] + [AR]*n_right, which is exactly mixed-canonical
    with the orthogonality centre at index n_left. FL/FR are the environments of the infinite
    left/right halves and stay FIXED for the whole evolution -- that is what makes the background
    absorbing rather than reflecting.

    `bond_h`/`triple_h` are forwarded to `vumps` when it has to solve the background here. A
    non-TFIM `mpo` without one of them now raises `EnergyReadoutError` rather than returning a
    TFIM e0 for a mixed-field chain -- see tnkit/vumps.py. Callers that already hold a converged
    `gs` are unaffected (no VUMPS call is made).
    """
    be = get_backend(backend)
    if gs is None:
        gs = vumps(g, chi, n_iter=vumps_iter, tol=vumps_tol, backend=backend, mpo=mpo,
                   bond_h=bond_h, triple_h=triple_h)
    AL, AR, C = gs["AL"], gs["AR"], gs["C"]
    # The old signature hardwired the h = 0 TFIM MPO here, so every mixed-field caller
    # got boundary environments FL/FR built from a
    # DIFFERENT Hamiltonian than the bulk it then evolved under. The background is only absorbing
    # -- indeed only stationary -- if the environments solve the same operator, so at h != 0 the
    # window edges were being driven by a mismatched boundary term. Pass `mpo` and they match.
    W = be.array(tfim_bulk_mpo(g) if mpo is None else mpo)

    FL, _ = _left_environment(AL, C, W, be, env_tol=env_tol)
    FR, _ = _right_environment(AR, C, W, be, env_tol=env_tol)

    AC = be.einsum("asb,bc->asc", AL, C)
    if op is not None:
        AC = be.einsum("st,atb->asb", be.array(op), AC)

    window = [AL] * n_left + [AC] + [AR] * n_right
    return MPS([be.to_numpy(t) for t in window], center=n_left), be.to_numpy(FL), be.to_numpy(FR), \
        be.to_numpy(W), gs


def window_mpo(W, n_sites):
    """Bulk MPO repeated across the window. Boundaries are carried by FL/FR, not by the MPO."""
    return [W] * n_sites


def _on_backend(window):
    """(backend, complex128 tensors) for a window. Torch tensors stay on their device; numpy
    input uses plain `np.einsum`, so results do not depend on einsum plans."""
    A = window[0]
    if type(A).__module__.split(".")[0] == "torch":
        from .backend import TorchBackend
        be = TorchBackend(device=str(A.device))
    else:
        be = PlainEinsumNumpyBackend()
    return be, [be.array(m) for m in window]


def _norm_envs(be, M):
    """Left and right norm environments, closed by D x D identities at the window edges."""
    N = len(M)
    Lenv = [None] * (N + 1)
    Lenv[0] = be.array(np.eye(int(M[0].shape[0])))
    for i in range(N):
        Lenv[i + 1] = be.einsum("ab,asc,bsd->cd", Lenv[i], be.conj(M[i]), M[i])
    Renv = [None] * (N + 1)
    Renv[N] = be.array(np.eye(int(M[-1].shape[2])))
    for i in range(N - 1, -1, -1):
        Renv[i] = be.einsum("asc,bsd,cd->ab", be.conj(M[i]), M[i], Renv[i + 1])
    return Lenv, Renv


def window_expectations(window, op):
    """<op_j> at every site of a window embedded in an infinite background.

    The ordinary `local_expectations` closes the chain with 1x1 boundary vectors, which is simply
    wrong here: the window's edge tensors carry a full bond of dimension D onto the background.
    The semi-infinite halves contract to the IDENTITY for norm purposes (the left block is built
    from left-canonical AL, the right from right-canonical AR), so the correct closure is a
    D x D identity at each end -- not a 1-vector, and not FL/FR either (those carry the MPO index
    and belong to the energy, not to observables).

    Using the wrong closure is not a small error: it showed up as <sigma^z> = 0.5357 at the window
    edge against 0.5522 in the bulk, i.e. a fake 3% boundary feature in a state that is exactly
    uniform there.
    """
    be, M = _on_backend(window)
    op = be.array(op)
    Lenv, Renv = _norm_envs(be, M)
    out = []
    for j in range(len(M)):
        num = be.einsum("ab,asc,st,btd,cd->", Lenv[j], be.conj(M[j]), op, M[j], Renv[j + 1])
        den = be.einsum("ab,asc,bsd,cd->", Lenv[j], be.conj(M[j]), M[j], Renv[j + 1])
        out.append(num / den)
    return _profile(be, out)


def window_neighbor_correlations(window, opA, opB):
    """<opA_j opB_{j+1}> on every bond j = 0..N-2 of a window on an infinite background.

    The two-site sibling of `window_expectations`, and it exists for exactly the same reason: the
    ordinary `tnkit.observables.neighbor_correlations` closes the chain with 1x1 boundary vectors
    and right-canonicalises, both of which are wrong for a window whose edge tensors carry a full
    bond of dimension D onto the background. The correct closure is the D x D identity at each end
    (the semi-infinite halves are built from left-/right-canonical AL/AR, so they contract to the
    identity for norm purposes). The Lenv/Renv machinery below is `window_expectations`' verbatim.

    Validated against `neighbor_correlations` on a bond-1 open MPS, where the two closures
    coincide.
    """
    be, M = _on_backend(window)
    opA, opB = be.array(opA), be.array(opB)
    Lenv, Renv = _norm_envs(be, M)
    out = []
    for j in range(len(M) - 1):
        # two-site ket block (a, s, t, e)
        theta = be.einsum("asc,cte->aste", M[j], M[j + 1])
        Oth = be.einsum("xs,yt,bste->bxye", opA, opB, theta)
        num = be.einsum("ab,axye,bxyf,ef->", Lenv[j], be.conj(theta), Oth, Renv[j + 2])
        den = be.einsum("ab,axye,bxyf,ef->", Lenv[j], be.conj(theta), theta, Renv[j + 2])
        out.append(num / den)
    return _profile(be, out)


def grow_window(window, AL, AR, n_left=1, n_right=1):
    """Prepend `n_left` vacuum tensors AL and append `n_right` vacuum tensors AR.

    THE ECONOMY, and the reason growth is nearly free here: the boundary environments FL and FR do
    NOT have to be recomputed. FL is the fixed point of the left transfer built from AL, so pushing
    the window boundary one site further left -- and filling the new slot with AL itself -- leaves
    that fixed point invariant. Same on the right with AR. This is only true because the background
    is STATIONARY; for a global quench the background would evolve and every growth step would need
    fresh environments (and the fixed-environment construction would be wrong anyway).

    Consequence worth testing rather than assuming: a growth event must be a NO-OP on every
    observable. The state is physically unchanged, only its representation is wider. See
    tests/test_window.py::test_growth_does_not_perturb_the_state.
    """
    be, M = _on_backend(window)
    c = getattr(window, "center", None)
    return MPS([be.array(AL)] * int(n_left) + M + [be.array(AR)] * int(n_right),
               center=None if c is None else c + int(n_left))


def edge_deviations(window, op, background):
    """(left, right): |<op_j> - background| at the two outermost sites of the window."""
    vals = window_expectations(window, op)
    return abs(float(vals[0]) - background), abs(float(vals[-1]) - background)


def edge_deviation(window, op, background):
    """max |<op_j> - background| over the two outermost sites of the window.

    This is the growth trigger: while it is below threshold the disturbance has not reached the
    edge and the window is wide enough; once it rises the window must grow or the outgoing
    excitation starts being absorbed instead of tracked.
    """
    return max(edge_deviations(window, op, background))


def evolve_growing_window(window, FL, FR, W, dt, n_steps, be_name="numpy",
                          op=None, background=None, AL=None, AR=None,
                          grow_thresh=1e-6, grow_by=2, max_sites=None,
                          measure=None, meas_every=1, grow_every=1, symmetric=True):
    """Evolve a window with 1TDVP, widening it whenever the disturbance reaches an edge.

    Returns (window, records, sizes) where `sizes` logs (t, n_sites) so the growth history is
    inspectable -- a window that never grows, or grows every step, both indicate a badly chosen
    threshold rather than a result.

    The edges are tested every `grow_every` steps (each test is a full expectation sweep), so
    `grow_by` must exceed the distance the disturbance travels in `grow_every * dt`.
    `symmetric=True` grows both sides when either edge trips; `False` grows only the side
    that tripped.
    """
    from .tdvp import tdvp1_evolve

    if op is None or background is None or AL is None or AR is None:
        raise ValueError("evolve_growing_window needs op, background, AL and AR to decide growth")

    records, sizes = [], [(0.0, len(window))]
    if measure is not None:
        records.append((0.0, measure(window)))

    # 1TDVP needs the orthogonality centre at site 0. A built window has it at its centre, and
    # prepending AL puts left-canonical tensors in front of it, so the first step and every
    # step after a left growth re-canonicalise. Appending AR keeps site 0 the centre.
    recanon = True
    for step in range(n_steps):
        window, rec = tdvp1_evolve(window, window_mpo(W, len(window)), dt, 1,
                                   backend=be_name, L0=FL, RN=FR, canonicalize=recanon)
        recanon = False
        t = (step + 1) * dt

        if (step + 1) % grow_every == 0:
            dev_l, dev_r = edge_deviations(window, op, background)
            if symmetric:
                dev_l = dev_r = max(dev_l, dev_r)
            add_l = grow_by if dev_l > grow_thresh else 0
            add_r = grow_by if dev_r > grow_thresh else 0
            if (add_l or add_r) and (max_sites is None
                                     or len(window) + add_l + add_r <= max_sites):
                window = grow_window(window, AL, AR, n_left=add_l, n_right=add_r)
                recanon = bool(add_l)
        sizes.append((t, len(window)))

        if measure is not None and (step + 1) % meas_every == 0:
            records.append((t, measure(window)))

    return window, records, sizes
