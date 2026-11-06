"""tnkit: dense tensors on numpy or torch, and U(1) block tensors (`tnkit.blocks`).

Settings are tnkit's defaults except where the workload fixes them (bond cap, tolerances,
sweeps, ramp) and where the settings file sets them (w["opts"]: `DMRGOptions`,
`DMRGExcitedOptions` or `TDVPOptions` fields, passed through). DMRG always passes its stop and
expansion explicitly (`PINNED`, overridden by the settings), so a change of tnkit's defaults
cannot change a benchmark: the absolute stop `tol` of the workload, `rtol` 0 and `expand` 0
unless the settings say otherwise. DMRG and TDVP use the same MPO arrays built here; block runs
label the physical states by 2 S_z (XXZ) or by the packed two-flavour charge (collision).
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from common import Clock, Unsupported, bonds, couplings, neel, schedule_of, stop_tol, sync

NAME = "tnkit"
SP = np.array([[0, 1], [0, 0]], dtype=float)
SM = SP.T.copy()
SZ = np.diag([0.5, -0.5])
I2 = np.eye(2)


def version():
    import tnkit
    return getattr(tnkit, "__version__", None)


def xxz_mpo(N, J, Jz):
    """H = J sum (S+ S- + S- S+)/2 + Jz sum Sz Sz, legs (wl, wr, s, s'), real."""
    W = np.zeros((5, 5, 2, 2))
    W[0, 0], W[0, 1], W[0, 2], W[0, 3] = I2, SP, SM, SZ
    W[1, 4], W[2, 4], W[3, 4], W[4, 4] = 0.5 * J * SM, 0.5 * J * SP, Jz * SZ, I2
    return prune([W[:1]] + [W.copy() for _ in range(N - 2)] + [W[:, 4:]])


def pairs_mpo(N, pairs, J, Jz):
    """H = sum over `pairs` (i < j) of J (S+ S- + S- S+)/2 + Jz Sz Sz as an MPO: channel 0 is
    "nothing placed", (o, k) is "operator o placed k sites ago", the last is "done". Every
    channel carries one operator type, so one U(1) charge."""
    R = max(j - i for i, j in pairs)
    ops = [(SP, 0.5 * J * SM), (SM, 0.5 * J * SP), (SZ, Jz * SZ)]
    D = 2 + 3 * R
    ch = lambda o, k: 1 + o * R + (k - 1)            # noqa: E731
    coupled = set(pairs)
    mpo = []
    for s in range(N):
        W = np.zeros((D, D, 2, 2))
        W[0, 0] = W[-1, -1] = I2
        for o, (a, b) in enumerate(ops):
            W[0, ch(o, 1)] = a
            for k in range(1, R + 1):
                if k < R:
                    W[ch(o, k), ch(o, k + 1)] = I2
                if (s - k, s) in coupled:
                    W[ch(o, k), D - 1] = b
        mpo.append(W)
    mpo[0], mpo[-1] = mpo[0][:1], mpo[-1][:, -1:]
    return prune(mpo)


def prune(mpo):
    """Drop MPO channels that carry nothing on a bond (block tensors need every channel to have
    one charge, which an all-zero channel has not)."""
    mpo = [w.copy() for w in mpo]
    changed = True
    while changed:
        changed = False
        for n in range(len(mpo) - 1):
            a, b = mpo[n], mpo[n + 1]
            keep = [j for j in range(a.shape[1]) if np.any(a[:, j]) and np.any(b[j])]
            if len(keep) < a.shape[1]:
                mpo[n], mpo[n + 1], changed = a[:, keep], b[keep], True
    return mpo


def mpo_for(w):
    if w["model"] == "xxz":
        return xxz_mpo(w["N"], w["J"], w["Jz"])
    if w["model"] == "heis_cyl":
        return pairs_mpo(w["N"], bonds(w), *couplings(w))
    if w["model"] == "tfim":
        from tnkit.models import tfim_mpo
        return [np.real(t) for t in tfim_mpo(w["N"], w["g"])]
    raise ValueError(w["model"])


def check(w, sym):
    if sym == "su2":
        raise Unsupported("tnkit has no SU(2) tensors")
    if w["kind"] == "excited" and sym != "dense":
        raise Unsupported("tnkit's excited-state DMRG is dense only")
    if w["kind"] == "dmrg" and sym != "dense":
        if w["model"] not in ("xxz", "heis_cyl"):
            raise Unsupported(f"no U(1) charges defined for model {w['model']!r}")
        try:
            from tnkit.block_backend import random_block_mps  # noqa: F401
        except ImportError:
            raise Unsupported("this tnkit has no block-tensor DMRG") from None


@contextmanager
def sweep_clock(backend):
    """Device-synced time stamps at the end of every DMRG sweep (tnkit's `_dmrg` calls
    `invariants.sweep` there once per sweep)."""
    import tnkit._dmrg as D
    real, stamps = D.invariants, []

    class Stamp:
        def __getattr__(self, name):
            return getattr(real, name)

        def sweep(self, *a, **k):
            sync(backend)
            stamps.append(time.perf_counter())
            return real.sweep(*a, **k)

    D.invariants = Stamp()
    try:
        yield stamps
    finally:
        D.invariants = real


PINNED = {"rtol": 0.0, "expand": 0}               # tnkit's defaults are 1e-10 and 4 (blocks)


def dmrg(w, chi, backend, sym):
    from tnkit.api import dmrg as run
    be = backend + "-real"                       # real Hamiltonian: tnkit's real dtype
    mpo, kw = mpo_for(w), {}
    if sym == "u1":                              # block MPO, ground state at S_z = 0
        from tnkit.backend import get_backend
        from tnkit.block_backend import to_block_mpo
        from tnkit.symmetry import U1
        mpo = to_block_mpo(mpo, [[1, -1]] * w["N"], get_backend(be), sym=U1())[0]
        kw = {"charge": 0}
    opts = dict(w.get("opts", {}))
    pin = {k: opts.pop(k, v) for k, v in PINNED.items()}
    stages = schedule_of(w, chi)
    energies, sweeps, mps = [], 0, None
    with Clock(backend) as c, sweep_clock(backend) as stamps:
        t0 = time.perf_counter()
        for k, (b, n) in enumerate(stages):
            last = k == len(stages) - 1 and not w.get("fixed_sweeps")
            stop = {"tol": stop_tol(w), "rtol": pin["rtol"]} if last else {"tol": 0.0, "rtol": 0.0}
            start = {"init_mps": mps} if mps is not None else kw
            grow = {"expand": pin["expand"] if k == 0 else 0}       # expansion opens the run only
            E, mps, info = run(mpo, chi=b, n_sweeps=n, backend=be,
                               **{**opts, **stop, **start, **grow})
            energies += [float(e) for e in info["energies"]]
            sweeps += int(info["sweeps"])
    sweep_s = list(np.diff([t0] + stamps)) if len(stamps) == sweeps else None
    return {"wall_s": c.s, "energy": float(E), "energies": energies, "sweeps": sweeps,
            "sweep_s": sweep_s, "max_bond": max(int(t.shape[2]) for t in mps),
            "discarded_weight": float(info.get("discarded_weight", np.nan)),
            "dtype": "float64",
            "settings": {"backend": be, "stages": stages,
                         "tol": 0.0 if w.get("fixed_sweeps") else stop_tol(w),
                         "rtol": 0.0 if w.get("fixed_sweeps") else pin["rtol"],
                         "expand": pin["expand"],
                         "stop": "fixed sweeps, no energy stop" if w.get("fixed_sweeps") else
                         "last stage: |dE| < max(tol, rtol |E|); earlier stages: none",
                         "fixed_sweeps": w.get("fixed_sweeps"), "opts": opts,
                         "eig": "lanczos-restart (default)", "eig_stop": "residual (default)",
                         "init": "random MPS at the first stage's bond (default)" if sym ==
                         "dense" else "random block MPS at the first stage's bond, 2 S_z = 0",
                         "update": "two-site", "blocks": sym == "u1"}}


def excited(w, chi, backend, sym):
    """The lowest `nstates` levels, one after another, each deflated against those before."""
    from tnkit.api import dmrg_excited
    be = backend + "-real"
    mpo, opts = mpo_for(w), dict(w.get("opts", {}))
    states, energies, sweeps = [], [], []
    with Clock(backend) as c:
        for _ in range(w["nstates"]):
            E, M, info = dmrg_excited(mpo, chi, prev_states=list(states), n_sweeps=w["max_sweeps"],
                                      tol=w["e_tol"], backend=be, **opts)
            states.append(M)
            energies.append(float(E))
            sweeps.append(int(info.get("sweeps", 0)))
    return {"wall_s": c.s, "energies_k": energies, "sweeps_k": sweeps, "dtype": "float64",
            "settings": {"backend": be, "n_sweeps": w["max_sweeps"], "tol": w["e_tol"],
                         "opts": opts, "deflate": "penalty, weight 1 (defaults)"}}


def _block_state(N, mpo, be):
    from tnkit.block_backend import BlockBackend, to_block_mpo, to_block_mps
    from tnkit.symmetry import U1
    qs = [np.array([1, -1])] * N                 # 2 S_z of (up, down)
    psi = [np.eye(2)[s].reshape(1, 2, 1).astype(complex) for s in neel(N)]
    M, _ = to_block_mps(psi, qs, be, sym=U1())
    W, _ = to_block_mpo([t.astype(complex) for t in mpo], qs, be, sym=U1())
    return M, W, qs, BlockBackend(be)


def tdvp(w, chi, backend, sym):
    from tnkit.backend import get_backend
    from tnkit.observables import local_expectations
    from tnkit.tdvp import TDVPOptions
    from tnkit.tdvp_evolver import TDVPEvolver
    N, steps = w["N"], int(round(w["T"] / w["dt"]))
    mpo = xxz_mpo(N, w["J"], w["Jz"])
    be = get_backend(backend)
    opts = dict(svd_tol=w["svd_tol"], chi_max=chi, **w.get("opts", {}))
    if sym == "u1":
        M, W, qs, ebe = _block_state(N, mpo, be)
        opts.update(backend=ebe, graph="off")
    elif sym == "dense":
        M = [np.eye(2)[s].reshape(1, 2, 1).astype(complex) for s in neel(N)]
        W, qs = [t.astype(complex) for t in mpo], None
        opts.update(backend=backend)
    else:
        raise ValueError(sym)
    step_s = []
    with Clock(backend) as c:
        ev = TDVPEvolver(M, W, w["dt"], chi, TDVPOptions(**opts))
        for _ in range(steps):
            with Clock(backend) as cs:
                ev.run(1, "12site")
            step_s.append(cs.s)
    mps = ev.mps
    if qs is not None:
        from tnkit.block_backend import to_dense_mps
        mps = [np.asarray(be.to_numpy(t)) for t in to_dense_mps(mps, qs)]
    sz = np.real(local_expectations(mps, SZ.astype(complex)))
    return {"wall_s": c.s, "step_s": step_s, "steps": steps, "sz": [float(x) for x in sz],
            "bonds": [int(t.shape[2]) for t in ev.mps][:-1], "dtype": "complex128",
            "settings": {"evolver": "TDVPEvolver '12site' (two-site below the cap, one-site at it)",
                         "svd_tol": w["svd_tol"], "chi_max": chi, "krylov_m": "30 (default)",
                         "opts": w.get("opts", {}),
                         "graph": opts.get("graph", "auto (default)"),
                         "blocks": sym == "u1"}}


def collision(w, chi, backend, sym):
    """Steps of the two-flavour collision continued from a saved dense state (with its MPO)."""
    from tnkit.backend import get_backend
    from tnkit.block_backend import (BlockBackend, project_to_block_mps, to_block_mpo,
                                     to_dense_mps)
    from tnkit.models_lgt2f import site_charges_2f
    from tnkit.observables import local_expectations
    from tnkit.symmetry import U1
    from tnkit.tdvp import TDVPOptions
    from tnkit.tdvp_evolver import TDVPEvolver
    psi, W = load_collision(w)
    be = get_backend(backend)
    opts = dict(svd_tol=w["svd_tol"], chi_max=chi, krylov_m=w["krylov_m"], graph="off")
    setup = {}
    if sym == "u1":
        qs = site_charges_2f(w["N"])
        with Clock(backend) as c0:
            M, conv = project_to_block_mps(psi, qs, be, chi, sym=U1())
            dense = [np.array(be.to_numpy(t)) for t in to_dense_mps(M, qs)]
            M, again = project_to_block_mps(dense, qs, be, chi, rtol=1e-10, sym=U1())
            Wb = to_block_mpo(W, qs, be, sym=U1())[0]
        setup = {"projection_s": c0.s, "fidelity": conv["fidelity"],
                 "fidelity_second_pass": again["fidelity"]}
        ev_args = (M, Wb)
        opts.update(backend=BlockBackend(be))
    elif sym == "dense":
        qs, ev_args = None, (psi, W)
        opts.update(backend=backend, canonicalize=False)
    else:
        raise ValueError(sym)
    step_s = []
    with Clock(backend) as c:
        ev = TDVPEvolver(*ev_args, w["dt"], chi, TDVPOptions(**opts))
        for _ in range(w["steps"]):
            with Clock(backend) as cs:
                ev.run(1, "12site")
            step_s.append(cs.s)
    mps = ev.mps
    if qs is not None:
        mps = [np.asarray(be.to_numpy(t)) for t in to_dense_mps(mps, qs)]
    n = np.real(local_expectations(mps, np.diag([0.0, 1.0]).astype(complex)))
    return {"wall_s": c.s, "step_s": step_s, "steps": w["steps"], "n": [float(x) for x in n],
            "bonds": [int(t.shape[2]) for t in ev.mps][:-1], "dtype": "complex128",
            "setup": setup,
            "settings": {"evolver": "TDVPEvolver '12site'", "svd_tol": w["svd_tol"],
                         "chi_max": chi, "krylov_m": w["krylov_m"], "graph": "off",
                         "blocks": sym == "u1", "start": "saved dense state; blocks: "
                         "project_to_block_mps twice (second pass rtol 1e-10), as tf_collision"}}


def load_collision(w):
    """The saved state (M0000..) and MPO (W0000..) npz files named by the workload."""
    import os
    st = np.load(os.path.expandvars(w["state"]))
    wz = np.load(os.path.expandvars(w["mpo"]))
    psi = [st[k] for k in sorted(k for k in st.files if k.startswith("M"))]
    W = [wz[k] for k in sorted(k for k in wz.files if k.startswith("W"))]
    return psi, W


def warmup(kind, backend, sym):
    """Imports, device context, BLAS and solver set-up on a tiny problem of the same kind."""
    if kind == "excited":
        excited({"model": "tfim", "N": 8, "g": 1.0, "nstates": 2, "max_sweeps": 2,
                 "e_tol": 1e-8}, 4, backend, "dense")
    elif kind == "dmrg":
        dmrg({"model": "xxz", "N": 8, "J": 1.0, "Jz": 1.0, "max_sweeps": 2, "e_tol": 1e-8}, 4,
             backend=backend, sym=sym)
    else:
        tdvp({"N": 8, "J": 1.0, "Jz": 0.0, "dt": 0.05, "T": 0.1, "svd_tol": 1e-10}, 4, backend,
             "u1" if sym == "u1" else "dense")
