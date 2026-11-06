"""YASTN (yastn.tn.mps): dense or U(1) tensors, numpy or torch backend.

Settings follow the YASTN documentation: two-site DMRG with `energy_tol`, TDVP method '12site'
(two-site where a bond can grow, one-site at the cap) with Hermitian `expmv`, everything else
at YASTN's defaults. YASTN's `opts_svd['tol']` cuts singular values below tol times the largest;
the workload's `sv_tol` sets it. The settings file (w["opts"]) may set `method`, `opts_eigs`,
`opts_svd` (merged over the workload's), `opts_expmv`, `project_penalty` (excited states), and for
DMRG `two_site_sweeps` (k '2site' sweeps per call, then '1site') and `rtol` (stop the last stage
once |dE| < rtol |E|), as in YASTN's rendered examples.
"""
from __future__ import annotations

import numpy as np

from common import bonds as pairs
from common import Clock, Unsupported, couplings, cuda_backend, laps, neel, options, \
    schedule_of, stop_tol

OPTS = ("method", "opts_eigs", "opts_svd", "opts_expmv", "project_penalty", "two_site_sweeps",
        "rtol")

NAME = "yastn"


def version():
    from importlib import metadata
    return metadata.version("yastn")


def check(w, sym):
    if sym == "su2":
        raise Unsupported("SU(2) is not used for YASTN in this suite")
    if w.get("model") == "tfim" and sym == "u1":
        raise Unsupported("the Ising chain has no U(1) symmetry")


def _cfg(backend, sym, dtype):
    kw = {"sym": "U1" if sym == "u1" else "dense", "default_dtype": dtype}
    if backend == "numpy":
        kw["backend"] = "np"
    else:
        kw.update(backend="torch", default_device="cuda" if cuda_backend(backend) else "cpu")
    return kw


def _ops(backend, sym, dtype):
    import yastn
    ops = yastn.operators.Spin12(**_cfg(backend, sym, dtype))
    ops.random_seed(seed=0)
    return ops


def _hamiltonian(w, ops):
    import yastn.tn.mps as mps
    N = w["N"]
    bonds = range(N - 1)
    if w["model"] in ("xxz", "xx"):
        terms = [mps.Hterm(0.5 * w["J"], (i, i + 1), (ops.sp(), ops.sm())) for i in bonds]
        terms += [mps.Hterm(0.5 * w["J"], (i, i + 1), (ops.sm(), ops.sp())) for i in bonds]
        if w.get("Jz", 0.0):
            terms += [mps.Hterm(w["Jz"], (i, i + 1), (ops.sz(), ops.sz())) for i in bonds]
    elif w["model"] == "heis_cyl":
        J, Jz = couplings(w)
        terms = []
        for i, j in pairs(w):
            terms += [mps.Hterm(0.5 * J, (i, j), (ops.sp(), ops.sm())),
                      mps.Hterm(0.5 * J, (i, j), (ops.sm(), ops.sp())),
                      mps.Hterm(Jz, (i, j), (ops.sz(), ops.sz()))]
    elif w["model"] == "tfim":                   # Pauli operators are twice the spin operators
        terms = [mps.Hterm(-4.0, (i, i + 1), (ops.sx(), ops.sx())) for i in bonds]
        terms += [mps.Hterm(-2.0 * w["g"], (i,), (ops.sz(),)) for i in range(N)]
    else:
        raise ValueError(w["model"])
    return mps.generate_mpo(mps.product_mpo(ops.I(), N), terms)


def dmrg(w, chi, backend, sym):
    import yastn
    import yastn.tn.mps as mps
    opts = options(w, OPTS)
    ops = _ops(backend, sym, "float64")
    H = _hamiltonian(w, ops)
    I = mps.product_mpo(ops.I(), w["N"])
    stages = schedule_of(w, chi)
    opts_eigs = opts.get("opts_eigs")
    energies, sweep_s, svds = [], [], []
    with Clock(backend) as c:
        psi = mps.random_mps(I, n=0 if sym == "u1" else None, D_total=stages[0][0])
        for k, (b, n) in enumerate(stages):
            last = k == len(stages) - 1 and not w.get("fixed_sweeps")
            opts_svd = {"D_total": b, "tol": w.get("sv_tol", 1e-14), **opts.get("opts_svd", {})}
            svds.append(opts_svd)
            k2 = opts.get("two_site_sweeps")
            method = yastn.Method("2site") if k2 else opts.get("method", "2site")
            it = mps.dmrg_(psi, H, method=method,
                           energy_tol=stop_tol(w) if last else None, max_sweeps=n,
                           iterator_step=1, opts_svd=opts_svd,
                           **({"opts_eigs": opts_eigs} if opts_eigs else {}))
            es, ts = [], []
            for out, dt in laps(it, backend):
                es.append(float(np.real(out.energy)))
                ts.append(dt)
                if k2 and out.sweeps == k2:
                    method.update_("1site")
                if last and "rtol" in opts and abs(out.denergy) < opts["rtol"] * abs(out.energy):
                    break
            energies += es[:int(out.sweeps)]             # the iterator may yield once more
            sweep_s += ts[:int(out.sweeps)]
    return {"wall_s": c.s, "energy": energies[-1], "energies": energies,
            "sweeps": len(energies), "sweep_s": sweep_s,
            "max_bond": max(psi.get_bond_dimensions()), "dtype": "float64",
            "settings": {"method": opts.get("method", "2site"), "stages": stages,
                         "two_site_sweeps": opts.get("two_site_sweeps"),
                         "rtol": 0.0 if w.get("fixed_sweeps") else opts.get("rtol", 0.0),
                         "energy_tol": None if w.get("fixed_sweeps") else stop_tol(w),
                         "stop": "fixed sweeps, no energy stop" if w.get("fixed_sweeps") else
                         "last stage: dmrg_'s energy_tol (absolute), or the caller stops once "
                         "|denergy| < rtol |energy|; earlier stages: none",
                         "opts_svd": svds,
                         "opts_eigs": opts_eigs or "default {'hermitian': True, 'ncv': 3, "
                         "'which': 'SR'}", "fixed_sweeps": w.get("fixed_sweeps"),
                         "init": "random_mps(D_total=first stage)" +
                         (", n=0" if sym == "u1" else "")}}


def excited(w, chi, backend, sym):
    """The lowest `nstates` levels, each by DMRG projected against the states found before."""
    import yastn.tn.mps as mps
    opts = options(w, OPTS)
    ops = _ops(backend, sym, "float64")
    H = _hamiltonian(w, ops)
    I = mps.product_mpo(ops.I(), w["N"])
    opts_svd = {"D_total": chi, "tol": w.get("sv_tol", 1e-14), **opts.get("opts_svd", {})}
    penalty = opts.get("project_penalty")
    states, energies, sweeps = [], [], []
    with Clock(backend) as c:
        for _ in range(w["nstates"]):
            psi = mps.random_mps(I, D_total=chi)
            project = [(penalty, p) for p in states] if penalty else list(states)
            out = mps.dmrg_(psi, H, project=project, method=opts.get("method", "2site"),
                            energy_tol=w["e_tol"], max_sweeps=w["max_sweeps"],
                            opts_svd=opts_svd, **({"opts_eigs": opts["opts_eigs"]}
                                                  if "opts_eigs" in opts else {}))
            states.append(psi)
            energies.append(float(np.real(mps.vdot(psi, H, psi))))
            sweeps.append(int(out.sweeps))
    return {"wall_s": c.s, "energies_k": energies, "sweeps_k": sweeps, "dtype": "float64",
            "settings": {"method": opts.get("method", "2site"), "energy_tol": w["e_tol"],
                         "max_sweeps": w["max_sweeps"], "opts_svd": opts_svd,
                         "project": f"previous states, penalty {penalty or 'default'}",
                         "init": "random_mps(D_total=chi)"}}


def tdvp(w, chi, backend, sym):
    import yastn.tn.mps as mps
    ops = _ops(backend, sym, "complex128")
    H = _hamiltonian(w, ops)
    psi = mps.product_mps([ops.vec_z(1 if s == 0 else -1) for s in neel(w["N"])])
    steps = int(round(w["T"] / w["dt"]))
    times = tuple(k * w["dt"] for k in range(steps + 1))
    opts = options(w, OPTS)
    opts_svd = {"D_total": chi, "tol": w["sv_tol"], **opts.get("opts_svd", {})}
    opts_expmv = {"hermitian": True, **opts.get("opts_expmv", {})}
    method = opts.get("method", "12site")
    with Clock(backend) as c:
        gen = mps.tdvp_(psi, H, times=times, dt=w["dt"], u=1j, method=method,
                        opts_svd=opts_svd, opts_expmv=opts_expmv)
        step_s = [dt for _, dt in laps(gen, backend)]
    sz = mps.measure_1site(psi, ops.sz(), psi)
    return {"wall_s": c.s, "step_s": step_s, "steps": steps,
            "sz": [float(np.real(sz[i])) for i in range(w["N"])],
            "bonds": [int(b) for b in psi.get_bond_dimensions()][1:-1], "dtype": "complex128",
            "settings": {"method": method, "opts_svd": opts_svd,
                         "opts_expmv": opts_expmv, "order": "2nd (default)"}}


# --- the two-flavour collision: tnkit's saved state and MPO, written as YASTN tensors -------

def _tensor(cfg, arr, charges, signature, skip_zero=False):
    """A YASTN tensor from a dense array whose index i on leg k carries charge charges[k][i]
    (None for a dense config)."""
    import itertools
    import yastn
    T = yastn.Tensor(config=cfg, s=signature)
    if charges is None:
        T.set_block(Ds=arr.shape, val=arr)
        return T
    groups = []
    for q in charges:
        u = np.unique(q)
        groups.append([(int(c), np.flatnonzero(q == c)) for c in u])
    for combo in itertools.product(*groups):
        if sum(s * c for s, (c, _) in zip(signature, combo)) != 0:
            continue
        block = arr[np.ix_(*[ix for _, ix in combo])]
        if skip_zero and not np.any(block):
            continue
        T.set_block(ts=tuple(c for c, _ in combo), Ds=block.shape, val=block)
    return T


def u1_inputs(psi, W, N, chi):
    """The U(1) collision as (array, charges per leg, signature) per tensor, for `_tensor`: the
    saved state projected by tnkit (as the tnkit block arm) and the MPO (legs wl, s, wr, s') with
    its bond charges. Needs tnkit only; returns (state, MPO, diagnostics)."""
    from tnkit.backend import get_backend
    from tnkit.block_backend import mpo_charges, project_to_block_mps, to_dense_mps
    from tnkit.models_lgt2f import site_charges_2f
    from tnkit.symmetry import U1
    qs = site_charges_2f(N)
    nb = get_backend("numpy")
    M, conv = project_to_block_mps(psi, qs, nb, chi, sym=U1())
    dense = [np.array(t) for t in to_dense_mps(M, qs)]
    M, again = project_to_block_mps(dense, qs, nb, chi, rtol=1e-10, sym=U1())
    ch = mpo_charges(W, qs, sym=U1())
    state = []
    for t in M:
        qa, qp, qb = (np.repeat(leg.charges, leg.dims) for leg in t.legs)
        state.append((np.asarray(t.to_dense()), (-qa, qp, -qb), (-1, 1, 1)))
    mpo = [(np.transpose(Wn, (0, 2, 1, 3)), (-ch[n], qs[n], -ch[n + 1], qs[n]), (-1, 1, 1, -1))
           for n, Wn in enumerate(W)]
    return state, mpo, {"fidelity": conv["fidelity"], "fidelity_second_pass": again["fidelity"]}


def collision(w, chi, backend, sym):
    import yastn
    import yastn.tn.mps as mps
    from tnkit.models_lgt2f import site_charges_2f
    from tnkit_runner import load_collision
    psi_np, W = load_collision(w)
    N2 = len(psi_np)
    cfg = yastn.make_config(**_cfg(backend, sym, "complex128"))
    setup = {}
    with Clock(backend) as c0:
        if sym == "u1":
            state, mpo, setup = u1_inputs(psi_np, W, w["N"], chi)
            psi, H = mps.Mps(N2), mps.Mpo(N2)
            for n, (arr, q, s) in enumerate(state):
                psi[n] = _tensor(cfg, arr, q, s)
            for n, (arr, q, s) in enumerate(mpo):
                H[n] = _tensor(cfg, arr, q, s, skip_zero=True)
        else:
            psi = mps.Mps(N2)
            for n, t in enumerate(psi_np):
                psi[n] = _tensor(cfg, np.asarray(t), None, (-1, 1, 1))
            H = mps.Mpo(N2)
            for n, Wn in enumerate(W):
                H[n] = _tensor(cfg, np.transpose(Wn, (0, 2, 1, 3)), None, (-1, 1, 1, -1))
        psi.canonize_(to="first")
    setup["setup_s"] = c0.s
    times = tuple(k * w["dt"] for k in range(w["steps"] + 1))
    opts_svd = {"D_total": chi, "tol": w["sv_tol"]}
    with Clock(backend) as c:
        gen = mps.tdvp_(psi, H, times=times, dt=w["dt"], u=1j, method="12site",
                        opts_svd=opts_svd, opts_expmv={"hermitian": True})
        step_s = [dt for _, dt in laps(gen, backend)]
    nq = (np.diag([0.0, 1.0]) if sym == "dense" else None)
    n = []
    for i in range(N2):
        if sym == "dense":
            op = _tensor(cfg, nq.astype(complex), None, (1, -1))
        else:
            q = site_charges_2f(w["N"])[i]
            op = _tensor(cfg, np.diag([0.0, 1.0]).astype(complex), (q, q), (1, -1),
                         skip_zero=True)
        v = mps.measure_1site(psi, op, psi, sites=i)
        n.append(float(np.real(v[i] if isinstance(v, dict) else v)))
    return {"wall_s": c.s, "step_s": step_s, "steps": w["steps"], "n": n,
            "bonds": [int(b) for b in psi.get_bond_dimensions()][1:-1], "dtype": "complex128",
            "setup": setup,
            "settings": {"method": "12site", "opts_svd": opts_svd,
                         "opts_expmv": {"hermitian": True}, "start": "tnkit's saved dense state;"
                         " U(1): tnkit's projection (as the tnkit block arm), as YASTN U1 tensors"
                         " of the packed U(1)xU(1) charge"}}


def warmup(kind, backend, sym):
    if kind == "excited":
        excited({"model": "tfim", "N": 8, "g": 1.0, "nstates": 2, "max_sweeps": 2,
                 "e_tol": 1e-8}, 4, backend, sym)
        return
    tdvp({"model": "xx", "N": 8, "J": 1.0, "Jz": 0.0, "dt": 0.05, "T": 0.1, "sv_tol": 1e-6}, 4,
         backend, sym)
    if kind == "dmrg":
        dmrg({"model": "xxz", "N": 8, "J": 1.0, "Jz": 1.0, "max_sweeps": 2, "e_tol": 1e-8}, 4,
             backend=backend, sym=sym)
