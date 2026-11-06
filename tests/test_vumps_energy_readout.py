"""Regression tests for the VUMPS energy-readout guard (the mixed-field e0 bug).

THE BUG. `vumps()` reads e0 off LOCAL expectation values, not off the MPO energy channel (the
channel carries an additive normalisation that is harmless for the optimisation and unreliable as
an absolute energy). The readout dispatches on the *operator the caller supplied*:

    triple_h  -> _triple_energy_density        three-site models
    bond_h    -> _bond_energy_density          the caller's own nearest-neighbour operator
    neither   -> _energy_density(..., g)       the TFIM formula  -<sx sx> - g<sz>

and the fall-through fired whenever a caller passed a custom `mpo` without the matching operator.
Every mixed-field caller did exactly that -- `vumps(g, chi, mpo=mixed_field_bulk_mpo(g, h))` --
so `gs["e0"]` came back SHORT BY EXACTLY h<sx> while the state itself was perfectly correct. At
the production point g_x = 1, g_z = 0.10 that is

    reported  e0 = -1.26819106      true e0 = -1.35074499      h<sx> = 0.08255394.

It mattered because e0's consumer is the anti-aliasing MPO shift `Ws[0,2] -= e0*I` of a
time-evolution window, whose whole purpose is to put the window vacuum at zero energy density. A short
e0 left it at -h<sx> per site -- -28.06 rather than -0.74 at N_s = 331 -- i.e. ~1/38 of the
intended margin.

THE FIX is loud, not silent: a custom non-TFIM `mpo` with neither `bond_h` nor
`triple_h` now raises `EnergyReadoutError`. Auto-correcting from the MPO would have fixed this
instance and hidden the next one.

Run:  python tests/test_vumps_energy_readout.py     (no pytest dependency; prints PASS/FAIL)
"""
import ast
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tnkit.backend import get_backend                                   # noqa: E402
from tnkit.idmrg import (mixed_field_bulk_mpo, mixed_field_bond_h,      # noqa: E402
                         tfim_bulk_mpo)
from tnkit.models import SX, SZ                                         # noqa: E402
from tnkit.vumps import EnergyReadoutError, vumps                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G, H, CHI = 1.0, 0.10, 8


def _independent_e0(gs, g, h):
    """e0 from local expectations, written out here from the Pauli matrices alone.

    Deliberately does NOT call _bond_energy_density or anything else vumps uses, so a bug in the
    readout cannot cancel against a bug in the check.
    """
    AC = np.asarray(get_backend("numpy").to_numpy(gs["AC"]))
    AR = np.asarray(get_backend("numpy").to_numpy(gs["AR"]))
    ACn = AC / np.linalg.norm(AC)
    sz = np.einsum("asb,st,atb->", ACn.conj(), SZ, ACn).real
    sx = np.einsum("asb,st,atb->", ACn.conj(), SX, ACn).real
    th = np.einsum("asc,ctb->astb", AC, AR)
    th = th / np.linalg.norm(th)
    sxsx = np.einsum("astb,su,tv,auvb->", th.conj(), SX, SX, th).real
    return -sxsx - g * sz - h * sx, sx


def test_mixed_field_e0_matches_independent_computation():
    """With bond_h, e0 is the FULL mixed-field energy density, to solver tolerance."""
    gs = vumps(G, CHI, n_iter=600, tol=1e-12, mpo=mixed_field_bulk_mpo(G, H),
               bond_h=mixed_field_bond_h(G, H))
    e0_indep, _ = _independent_e0(gs, G, H)
    err = abs(float(np.real(gs["e0"])) - e0_indep)
    assert err < 1e-12, f"e0 {gs['e0']!r} vs independent {e0_indep!r} (err {err:.3e})"


def test_mixed_field_without_bond_h_raises():
    """The bug itself: mixed-field MPO, no bond_h. Must be loud, and must name the fix."""
    try:
        vumps(G, CHI, n_iter=5, mpo=mixed_field_bulk_mpo(G, H))
    except EnergyReadoutError as e:
        assert "bond_h" in str(e), f"the error must name the fix, got: {e}"
    else:
        raise AssertionError("vumps returned an e0 for a mixed-field MPO with no bond_h")


def test_the_guard_costs_nothing_when_it_passes():
    """It fires BEFORE the optimisation loop, so a mis-specified run dies immediately."""
    import time
    t0 = time.perf_counter()
    try:
        vumps(G, 64, n_iter=600, mpo=mixed_field_bulk_mpo(G, H))
    except EnergyReadoutError:
        pass
    assert time.perf_counter() - t0 < 5.0, "the guard ran after the optimisation, not before"


def test_explicit_tfim_mpo_is_still_accepted():
    """mpo=tfim_bulk_mpo(g) IS the TFIM at that g -- the TFIM readout is valid, no bond_h needed.

    The guard compares against the operator, not against the mere presence of `mpo=`, so callers
    that hand VUMPS the same Hamiltonian it would have built itself are unaffected.
    """
    a = vumps(1.5, CHI, n_iter=300, tol=1e-12, seed=1, mpo=tfim_bulk_mpo(1.5))
    b = vumps(1.5, CHI, n_iter=300, tol=1e-12, seed=1)
    assert abs(float(np.real(a["e0"])) - float(np.real(b["e0"]))) < 1e-12


def test_tfim_mpo_at_the_wrong_g_is_caught():
    """The failure mode the shape of the check buys for free: mpo and g disagreeing."""
    try:
        vumps(1.5, CHI, n_iter=5, mpo=tfim_bulk_mpo(1.2))
    except EnergyReadoutError:
        return
    raise AssertionError("vumps accepted an MPO at a different g than its own g argument")


def test_ignore_reproduces_the_pre_fix_number_exactly():
    """The escape hatch must reproduce the OLD e0 bit-for-bit -- old results stay reproducible.

    And the deficit must be exactly h<sx>, which is the reproduction of the bug as a measurement
    rather than an assertion about it.
    """
    gs = vumps(G, CHI, n_iter=600, tol=1e-12, mpo=mixed_field_bulk_mpo(G, H),
               on_mpo_mismatch="ignore")
    e0_indep, sx = _independent_e0(gs, G, H)
    deficit = e0_indep - float(np.real(gs["e0"]))
    assert abs(deficit + H * sx) < 1e-12, \
        f"deficit {deficit:.12f} is not -h<sx> = {-H * sx:.12f}"


def test_no_caller_in_the_repo_passes_mpo_without_a_bond_operator():
    """Static sweep: this is what stops the bug recurring at a NEW call site.

    Any `vumps(...)` call with `mpo=` and neither `bond_h=`/`triple_h=` nor an explicit
    `on_mpo_mismatch=` would raise at runtime -- possibly hours into a run. Catch it here.
    Conditional forms like `mpo=W if h else None` count as passing an MPO.

    Widened after a review found three holes in it. It used `os.listdir`, so a driver
    in a subdirectory was never parsed; it covered only three directories, so `bench/`
    was outside it entirely; and it read `mpo` only as a literal keyword at the call, so
    `kw = dict(g=..., mpo=..., bond_h=...)` followed by `vumps(chi=chi, **kw)` was
    invisible. It now walks the source trees recursively and resolves a `**name` against the
    dict literal assigned to `name` in the same file, flagging any it cannot resolve for
    manual review rather than passing it. The runtime guard always caught these; what was
    lost was the fire-at-commit-time property that is the whole reason the sweep exists.
    """
    def _is_tfim_mpo(node):
        """`mpo=tfim_bulk_mpo(...)` IS the TFIM -- the guard passes it, so the sweep must too."""
        return (isinstance(node, ast.Call)
                and (getattr(node.func, "id", None) or getattr(node.func, "attr", None))
                == "tfim_bulk_mpo")

    def _dict_literals(tree):
        """name -> the keys of any `name = dict(...)` or `name = {...}` in the file.

        A driver may build its keywords before calling: `kw = dict(g=..., mpo=...,
        bond_h=...)` then `vumps(chi=chi, **kw)`. That call is invisible to a sweep that
        only reads literal keywords, and such a call is correct only if its author
        remembered. Collect the keys so a
        `**name` call can be judged rather than waved through.
        """
        out = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = getattr(node.targets[0], "id", None)
            if tgt is None:
                continue
            v = node.value
            if isinstance(v, ast.Dict):
                out.setdefault(tgt, set()).update(
                    k.value for k in v.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str))
            elif (isinstance(v, ast.Call)
                  and getattr(v.func, "id", None) == "dict"):
                out.setdefault(tgt, set()).update(k.arg for k in v.keywords if k.arg)
        return out

    offenders = []
    for sub in ("tnkit", "experiments", "tests", "bench", "scripts"):
        top = os.path.join(ROOT, sub)
        if not os.path.isdir(top):
            continue
        for root, _dirs, files in os.walk(top):
            for fn in sorted(files):
                if not fn.endswith(".py") or fn == os.path.basename(__file__):
                    continue                   # this file exercises the guard on purpose
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, ROOT)
                try:
                    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
                except SyntaxError:                                # pragma: no cover
                    continue
                built = _dict_literals(tree)
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    fname = (getattr(node.func, "id", None)
                             or getattr(node.func, "attr", None))
                    if fname != "vumps":
                        continue
                    kw = {k.arg: k.value for k in node.keywords if k.arg}
                    # fold in every `**name` whose dict literal we can see
                    unresolved = False
                    for k in node.keywords:
                        if k.arg is not None:
                            continue
                        name = getattr(k.value, "id", None)
                        if name in built:
                            kw.update({key: None for key in built[name]})
                        else:
                            unresolved = True
                    if unresolved:
                        offenders.append(f"{rel}:{node.lineno} (vumps(**...) not resolvable"
                                         " -- read it by hand)")
                        continue
                    if "mpo" in kw and kw["mpo"] is not None and _is_tfim_mpo(kw["mpo"]):
                        continue
                    if "mpo" in kw and not (set(kw)
                                            & {"bond_h", "triple_h", "on_mpo_mismatch"}):
                        offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, ("vumps(mpo=...) without bond_h/triple_h at: " + ", ".join(offenders))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:                                     # noqa
            failed += 1; print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
