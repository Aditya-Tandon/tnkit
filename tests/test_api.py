"""The public API surface: version, curated namespace, options objects, MPS type.

These tests exist because a refactor of the library changed how the library
is CALLED without changing what it computes, and the only way to say that honestly
is to assert it. Three properties are pinned here:

  1. `__version__` is whatever `pyproject.toml` declares -- never a second copy of
     the number that can drift from it.
  2. The old flat call style and the new options objects produce BIT-IDENTICAL
     results. Existing scripts use the flat style and will not be rewritten, so
     "equivalent" has to mean equal bytes, not close.
  3. `dmrg` and the TDVP integrators return an `MPS`, which is a `list` subclass,
     so every script that treats the result as a list of rank-3 tensors is
     unaffected.
"""
import ast
import importlib
import os
import re
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tnkit                                                          # noqa: E402
from tnkit import api                                                 # noqa: E402
from tnkit.dmrg import DMRGOptions, dmrg                              # noqa: E402
from tnkit.models import SX, tfim_mpo                                 # noqa: E402
from tnkit.mps import MPS, as_mps                                     # noqa: E402
from tnkit.observables import all_entropies, structure_factor, to_numpy   # noqa: E402
from tnkit.tdvp import TDVPOptions, tdvp1_evolve, tdvp2_evolve        # noqa: E402
from tnkit.vumps import VUMPSOptions, vumps                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_version():
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"""\s*version\s*=\s*["']([^"']+)["']""", line)
            if m:
                return m.group(1)
    raise AssertionError("pyproject.toml declares no version")


def _bytes(mps):
    return [np.ascontiguousarray(np.asarray(A)).tobytes() for A in to_numpy(mps)]


# --------------------------------------------------------------------------- version
def test_version_matches_pyproject():
    # The declaring file and the distribution metadata are the only two sources
    # `_detect_version` will read, and they must agree with each other, so this
    # holds both in a checkout and in an installed environment. The
    # checkout wins when there is one; see the stale-metadata test below.
    assert tnkit.__version__ == _pyproject_version()


def test_version_falls_back_without_installed_metadata():
    # Experiment drivers run from the repository root off sys.path, where there is
    # no distribution metadata at all. The fallback must find the declaration.
    import importlib.metadata as md
    real = md.version
    try:
        md.version = lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name))
        assert tnkit._detect_version() == _pyproject_version()
    finally:
        md.version = real


def test_version_prefers_the_checkout_over_stale_installed_metadata():
    """A stale `tnkit.egg-info/` must not decide what version a driver stamps.

    Planting a PKG-INFO at 0.0.99 made a checkout whose pyproject.toml says otherwise
    report 0.0.99, and every driver that records `tnkit.__version__` into its JSON
    provenance would have stamped a version that was not the code it ran, silently.
    The checkout now takes precedence.
    """
    import importlib.metadata as md
    real = md.version
    try:
        md.version = lambda name: "0.0.99"
        assert tnkit._detect_version() == _pyproject_version()
    finally:
        md.version = real


def test_version_never_raises(tmp_path, monkeypatch):
    # Neither source available: a placeholder, not an exception. `import tnkit`
    # must not be able to fail because of a version lookup.
    import importlib.metadata as md
    real = md.version
    try:
        md.version = lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name))
        monkeypatch.setattr(tnkit, "__file__", str(tmp_path / "tnkit" / "__init__.py"))
        assert tnkit._detect_version() == "0+unknown"
    finally:
        md.version = real


# --------------------------------------------------------------------------- surface
def test_curated_names_all_exist():
    for name in api.CURATED:
        assert hasattr(api, name), f"tnkit.api is missing {name}"
    assert sorted(api.__all__) == sorted(api.CURATED)


def test_package_reexports_everything_it_can():
    """Nothing on the curated surface is unreachable from `import tnkit` alone.

    This is the property the private-module layout exists to deliver. It used to be
    expressed through a public `api.SHADOWED_BY_MODULE` list, empty and kept alive only
    by a test asserting it stayed empty; the attribute is gone and the property is
    stated directly.
    """
    assert set(api.CURATED) <= set(tnkit.__all__), (
        set(api.CURATED) - set(tnkit.__all__))
    for name in api.CURATED:
        assert hasattr(tnkit, name), f"tnkit does not export {name}"
        assert getattr(tnkit, name) is getattr(api, name), f"{name} is a different object"
        assert name in tnkit.__all__, f"{name} missing from tnkit.__all__"


def test_all_names_resolve():
    for name in tnkit.__all__:
        assert hasattr(tnkit, name), f"tnkit.__all__ advertises a missing {name}"


def test_the_three_solvers_are_functions_at_the_top_level():
    """`from tnkit import dmrg` gives the SOLVER.

    An earlier layout could not deliver it: `dmrg`, `vumps` and `idmrg` each named a module as
    well as a function. The implementations moved to `tnkit._dmrg`, `tnkit._vumps`
    and `tnkit._idmrg`, the old module paths became shims, and the functions took the
    package attributes.
    """
    from tnkit.api import dmrg as api_dmrg, idmrg as api_idmrg, vumps as api_vumps
    for name, fn in (("dmrg", api_dmrg), ("idmrg", api_idmrg), ("vumps", api_vumps)):
        top = getattr(tnkit, name)
        assert callable(top) and not isinstance(top, types.ModuleType)
        assert top is fn, f"tnkit.{name} is not tnkit.api.{name}"
        assert name in tnkit.__all__
    # `import a.b as c` resolves by getattr on the package, so this now binds the function
    import tnkit.dmrg as D
    assert D is api_dmrg


def test_the_old_module_paths_still_import_every_name():
    """Existing scripts import through `tnkit.dmrg`/`vumps`/`idmrg`. None moves.

    The shim copies the implementation's whole namespace rather than re-exporting a
    hand-written list, so this asserts the property directly instead of spot-checking
    the names someone remembered.
    """
    for public, private in (("tnkit.dmrg", "tnkit._dmrg"),
                            ("tnkit.vumps", "tnkit._vumps"),
                            ("tnkit.idmrg", "tnkit._idmrg")):
        shim, impl = sys.modules[public], importlib.import_module(private)
        assert isinstance(shim, types.ModuleType)
        missing = [k for k in vars(impl)
                   if not k.startswith("__") and not hasattr(shim, k)]
        assert not missing, f"{public} is missing {missing}"
        for k in vars(impl):
            if not k.startswith("__"):
                assert getattr(shim, k) is getattr(impl, k), f"{public}.{k} is a copy"
    # the imports the tree actually performs, including privates and re-imports
    from tnkit.dmrg import dmrg, DMRGOptions, apply_two_site, _truncate, _EIGSOLVERS  # noqa: F401
    from tnkit.vumps import (vumps, VUMPSOptions, _real, _gauge, _left_environment,  # noqa: F401
                             update_left_env, EnergyReadoutError, _HOOKS)
    from tnkit.idmrg import (idmrg, tfim_bulk_mpo, mixed_field_bond_h,  # noqa: F401
                             _xi_from_tensor, SYMANZIK_LAMBDA)


def test_the_mutable_registries_are_one_object_on_every_path():
    """Identity, on all three paths a driver can reach: this is why mutation survives.

    A driver reaches `_EIGSOLVERS` through
    `import tnkit.dmrg as D`, which is now the function; the library reads it as a
    module global of `tnkit._dmrg`; a reader might come through the shim. All three
    must be the same dict, or a patch lands somewhere nobody reads.
    """
    import tnkit._dmrg as impl_d
    import tnkit._vumps as impl_v
    assert tnkit.dmrg._EIGSOLVERS is impl_d._EIGSOLVERS
    assert sys.modules["tnkit.dmrg"]._EIGSOLVERS is impl_d._EIGSOLVERS
    assert tnkit.vumps._HOOKS is impl_v._HOOKS
    assert sys.modules["tnkit.vumps"]._HOOKS is impl_v._HOOKS
    assert set(impl_v._HOOKS) == {"left_fixed_point", "energy_channel"}


def test_a_patched_eigensolver_really_runs(monkeypatch):
    """The driver's own idiom, and the patch must be CALLED -- not merely installed.

    Identity of a dict proves the patch is visible; it does not prove the sweep went
    through it. The failure this guards is an A/B arm reporting a number for code it
    never ran, which is silent by construction, so the assertion is a call count.
    """
    import tnkit.dmrg as D                       # the function, not the module
    calls = []
    base = D._EIGSOLVERS["scipy-host"]
    try:
        D._EIGSOLVERS["scipy-host"] = (
            lambda mv, v0, ops, **k: (calls.append(1), base(mv, v0, ops, **k))[1])
        E, mps, _ = dmrg(tfim_mpo(8, 1.0), chi=6, n_sweeps=2, eig="scipy-host")
    finally:
        D._EIGSOLVERS["scipy-host"] = base
    assert calls, "the patched eigensolver was never called: the arm ran UNPATCHED"
    assert isinstance(mps, MPS) and np.isfinite(E)


def test_a_patched_vumps_hook_really_runs():
    """The seam a driver uses to replay the legacy VUMPS fixed point.

    Two halves. A counting pass-through must be called and must leave the answer
    alone -- that says the registry sits on the real code path. Then a hook that
    raises must reach the caller, which says the registry is the ONLY path: if the
    module still called its own global, the exception could not escape.
    """
    from tnkit.vumps import _HOOKS
    import tnkit._vumps as impl

    clean = vumps(1.5, chi=6, n_iter=40, tol=1e-9, seed=1)["e0"]

    calls = []
    real = _HOOKS["left_fixed_point"]
    try:
        _HOOKS["left_fixed_point"] = (
            lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        patched = vumps(1.5, chi=6, n_iter=40, tol=1e-9, seed=1)["e0"]
    finally:
        _HOOKS["left_fixed_point"] = real
    assert calls, "the patched fixed point was never called: the arm ran UNPATCHED"
    assert patched == clean, "a pass-through hook must not move the answer"

    class _Reached(RuntimeError):
        pass

    def _boom(*a, **k):
        raise _Reached("the registry is on the path")

    try:
        _HOOKS["energy_channel"] = _boom
        with pytest.raises(_Reached):
            vumps(1.5, chi=6, n_iter=5, tol=1e-9, seed=1)
    finally:
        _HOOKS["energy_channel"] = impl._energy_channel


def test_rebinding_a_module_global_does_not_reach_the_implementation():
    """The negative control, and the whole reason `_HOOKS` exists.

    Drivers used to write `V._left_fixed_point = f` through
    `import tnkit.vumps as V`. Through a shim -- or through the exported function --
    that assignment cannot reach `tnkit._vumps`' own globals, so the legacy arm would
    have run the CURRENT code while reporting itself as legacy. Pinned here so the
    old idiom cannot creep back in believing it works.
    """
    shim = sys.modules["tnkit.vumps"]
    import tnkit._vumps as impl
    sentinel = object()
    old = shim._left_fixed_point
    try:
        shim._left_fixed_point = sentinel
        assert impl._left_fixed_point is not sentinel, \
            "rebinding the shim reached the implementation; the negative control is void"
        assert impl._HOOKS["left_fixed_point"] is not sentinel
        # and the solver is unmoved by it
        assert np.isfinite(vumps(1.5, chi=6, n_iter=20, tol=1e-9, seed=1)["e0"])
    finally:
        shim._left_fixed_point = old


def test_the_shim_set_and_the_modules_that_stay_modules():
    """Which public paths are shims, and that `tnkit.tdvp` and `tnkit.eigensolver` stay modules.

    A rebind `<alias>.<name> = ...` through a shim never reaches the implementation (above), so
    callers that patch module globals rely on this split: patching through `tnkit.tdvp` or
    `tnkit.eigensolver` works only while no top-level function of that name shadows them.
    """
    import pkgutil
    shims = {m.name for m in pkgutil.iter_modules(tnkit.__path__) if not m.name.startswith("_")
             and getattr(importlib.import_module(f"tnkit.{m.name}"), "__shim_source__", None)}
    assert {"dmrg", "vumps", "idmrg", "luscher", "two_particle", "prefix_graph"} <= shims, \
        f"shim discovery broke -- found only {sorted(shims)}"
    for name in ("tdvp", "eigensolver"):
        assert name not in shims
        assert isinstance(getattr(tnkit, name, None), types.ModuleType), \
            f"tnkit.{name} is no longer the module; patches through it would silently do nothing"


# --------------------------------------------------------------------------- options
HISTORICAL_DMRG_KWARGS = [
    "n_sweeps", "tol", "trunc", "eig_tol", "eig", "warm_start", "reorth",
    "eig_tol_warmup", "warmup_sweeps", "seed", "init_mps", "backend", "max_iter",
    "verbose", "restart_m", "restart_k", "mixed_tol_lo", "eig_stop", "reorth_alg",
]
NEW_DMRG_FIELDS = ["charge"]                  # added after the flat signature; default None
LATER_DMRG_DEFAULTS = {"sites": 2, "expand": None, "expand_alpha": 3e-3, "expand_decay": 0.3,
                       "rtol": 1e-10}


def test_dmrg_options_cover_the_historical_signature():
    """Every keyword the flat signature took is a field with the same name.

    Renaming one would break scripts silently-ish (a TypeError at the top of a
    12-hour run), so the historical list is written out here rather than derived.
    """
    import dataclasses
    fields = [f.name for f in dataclasses.fields(DMRGOptions)]
    assert sorted(fields) == sorted(HISTORICAL_DMRG_KWARGS + NEW_DMRG_FIELDS
                                    + list(LATER_DMRG_DEFAULTS))
    defaults = {f.name: f.default for f in dataclasses.fields(DMRGOptions)}
    assert all(defaults[k] is None for k in NEW_DMRG_FIELDS)
    assert all(defaults[k] == v for k, v in LATER_DMRG_DEFAULTS.items())
    assert defaults["eig_stop"] == "residual"      # the default that matters most
    assert defaults["reorth"] == "full"
    assert defaults["n_sweeps"] == 8 and defaults["tol"] == 1e-10


@pytest.mark.parametrize("chi", [12])
def test_dmrg_flat_and_options_are_bit_identical(chi):
    N, g = 14, 1.0
    flat = dict(n_sweeps=12, tol=1e-12, eig_tol=1e-11, trunc="svd", reorth="full",
                seed=3, eig_stop="residual", warm_start=True, max_iter=64)
    E1, m1, i1 = dmrg(tfim_mpo(N, g), chi=chi, **flat)
    E2, m2, i2 = dmrg(tfim_mpo(N, g), chi, DMRGOptions(**flat))
    assert E1 == E2, f"energies differ: {E1!r} vs {E2!r}"
    assert _bytes(m1) == _bytes(m2), "states differ bit-for-bit"
    assert i1["sweeps"] == i2["sweeps"] and i1["energies"] == i2["energies"]
    assert i1["matvec_calls"] == i2["matvec_calls"]


def test_dmrg_options_dict_and_flat_override():
    N, g, chi = 12, 1.0, 8
    base = DMRGOptions(n_sweeps=6, tol=1e-12, seed=5)
    E1, m1, _ = dmrg(tfim_mpo(N, g), chi, base)
    E2, m2, _ = dmrg(tfim_mpo(N, g), chi, dict(n_sweeps=6, tol=1e-12, seed=5))
    E3, m3, _ = dmrg(tfim_mpo(N, g), chi, DMRGOptions(n_sweeps=6, tol=1e-12, seed=0), seed=5)
    assert E1 == E2 == E3
    assert _bytes(m1) == _bytes(m2) == _bytes(m3)


def test_dmrg_rejects_nonsense():
    mpo = tfim_mpo(6, 1.0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'not_a_knob'"):
        dmrg(mpo, chi=4, not_a_knob=1)
    with pytest.raises(TypeError, match="missing required argument 'chi'"):
        dmrg(mpo)
    with pytest.raises(TypeError, match="options must be"):
        dmrg(mpo, 4, options="fast")
    with pytest.raises(TypeError, match="multiple values for argument 'chi'"):
        dmrg(mpo, 4, chi=8)
    # chi as a keyword is how `dmrg(mpo, **kw)` drivers call it
    E, mps, _ = dmrg(mpo, chi=4, n_sweeps=2)
    assert isinstance(mps, MPS)


def test_tdvp_flat_and_options_are_bit_identical():
    N, g, chi = 10, 1.0, 8
    _, mps0, _ = dmrg(tfim_mpo(N, g), chi=chi, n_sweeps=6, tol=1e-12)
    quench = tfim_mpo(N, 1.5)
    a1, r1 = tdvp1_evolve(list(mps0), quench, 0.05, 4, backend="numpy", meas_every=2)
    a2, r2 = tdvp1_evolve(list(mps0), quench, 0.05, 4,
                          TDVPOptions(backend="numpy", meas_every=2))
    assert _bytes(a1) == _bytes(a2) and r1 == r2

    b1, _ = tdvp2_evolve(list(mps0), quench, chi, 0.05, 4, trunc="svd", svd_tol=0.0, seed=2)
    b2, _ = tdvp2_evolve(list(mps0), quench, chi, 0.05, 4,
                         TDVPOptions(trunc="svd", svd_tol=0.0, seed=2))
    assert _bytes(b1) == _bytes(b2)


def test_tdvp_options_refuse_the_other_integrators_knobs():
    N, g, chi = 8, 1.0, 6
    _, mps0, _ = dmrg(tfim_mpo(N, g), chi=chi, n_sweeps=4)
    quench = tfim_mpo(N, 1.5)
    with pytest.raises(TypeError, match="two-site integrator"):
        tdvp1_evolve(list(mps0), quench, 0.05, 1, svd_tol=1e-8)
    with pytest.raises(TypeError, match="one-site integrator"):
        tdvp2_evolve(list(mps0), quench, chi, 0.05, 1, cuda_graph=object())
    # but a SHARED bundle carrying both integrators' fields is the recommended
    # 2TDVP-then-1TDVP protocol, and must be accepted
    shared = TDVPOptions(svd_tol=1e-8, trunc="svd", backend="numpy")
    out, _ = tdvp1_evolve(list(mps0), quench, 0.05, 1, shared)
    assert isinstance(out, MPS)


# --------------------------------------------------------------------------- VUMPS options
HISTORICAL_VUMPS_KWARGS = [
    "n_iter", "tol", "seed", "init", "verbose", "backend", "env_tol", "local_tol",
    "m_local_cap", "trace", "mpo", "bond_h", "triple_h", "on_mpo_mismatch",
    "xi_on_padding", "xi_rank_tol",
]


def test_vumps_options_cover_the_historical_signature():
    """Every keyword the flat signature took is a field with the same name and default."""
    import dataclasses
    fields = [f.name for f in dataclasses.fields(VUMPSOptions)]
    assert sorted(fields) == sorted(HISTORICAL_VUMPS_KWARGS)
    d = {f.name: f.default for f in dataclasses.fields(VUMPSOptions)}
    assert d["n_iter"] == 200 and d["tol"] == 1e-10 and d["seed"] == 0
    assert d["backend"] == "numpy" and d["env_tol"] == 1e-9 and d["local_tol"] == 1e-10
    # the two guards' defaults are the ones that decide whether a wrong number is
    # returned quietly, so they are pinned by name rather than left to the dataclass
    assert d["on_mpo_mismatch"] == "raise"
    assert d["xi_on_padding"] == "deflate"


def test_vumps_flat_and_options_are_bit_identical():
    """The two call styles are equal to the byte, not to a tolerance."""
    flat = dict(n_iter=120, tol=1e-11, seed=1, env_tol=1e-10, backend="numpy")
    a = vumps(1.5, chi=8, **flat)
    b = vumps(1.5, 8, VUMPSOptions(**flat))
    assert a["e0"] == b["e0"], f"{a['e0']!r} vs {b['e0']!r}"
    assert a["S"] == b["S"] and a["eps"] == b["eps"] and a["iters"] == b["iters"]
    for k in ("AL", "AR", "AC", "C"):
        assert np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes(), f"{k} differs"


def test_vumps_options_dict_and_flat_override():
    base = dict(n_iter=100, tol=1e-11, seed=1)
    a = vumps(1.5, 8, VUMPSOptions(**base))
    b = vumps(1.5, 8, dict(base))
    c = vumps(1.5, 8, VUMPSOptions(n_iter=100, tol=1e-11, seed=0), seed=1)
    assert a["e0"] == b["e0"] == c["e0"]


def test_vumps_rejects_nonsense():
    with pytest.raises(TypeError, match="unexpected keyword argument 'not_a_knob'"):
        vumps(1.5, chi=4, not_a_knob=1)
    with pytest.raises(TypeError, match="missing required argument 'chi'"):
        vumps(1.5)
    with pytest.raises(TypeError, match="options must be"):
        vumps(1.5, 4, options="fast")
    with pytest.raises(TypeError, match="multiple values for argument 'chi'"):
        vumps(1.5, 4, chi=8)
    # `g` and `chi` both as keywords, as older call sites do
    gs = vumps(g=1.5, chi=6, n_iter=20, tol=1e-9, seed=1)
    assert "e0" in gs


def test_no_call_site_passes_more_positionals_than_the_front_doors_keep():
    """Compatibility measured by parsing every call site, not by grep.

    `dmrg`, `vumps` and the two TDVP integrators all end their positional run
    where the options object begins. A driver that passed one more positional
    argument would now silently hand it in as `options` and raise -- at the top of
    a twelve-hour run. So the whole tree is parsed instead of trusted.
    """
    # the number of positional arguments each front door kept BEFORE its `options` slot
    limits = {"dmrg": 2, "vumps": 2, "tdvp1_evolve": 4, "tdvp2_evolve": 5}
    over, grouped_outside_tests = [], []
    for sub in ("tnkit", "experiments", "tests", "bench"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for dirpath, _, files in os.walk(d):
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
                except SyntaxError:                                    # pragma: no cover
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                    if name not in limits:
                        continue
                    where = f"{os.path.relpath(path, ROOT)}:{node.lineno} ({name}, " \
                            f"{len(node.args)} positional)"
                    if len(node.args) > limits[name] + 1:      # past the options slot: broken
                        over.append(where)
                    elif len(node.args) == limits[name] + 1 and sub != "tests":
                        grouped_outside_tests.append(where)
    assert not over, "call sites passing more positionals than the front door has: " + ", ".join(over)
    # Nothing outside the suite passes an options object positionally, so every
    # historical call site is still reading its arguments by the name it always used.
    assert not grouped_outside_tests, \
        "unexpected positional options object at: " + ", ".join(grouped_outside_tests)


# --------------------------------------------------------------------------- MPS type
def test_dmrg_returns_an_mps_that_is_still_a_list():
    N, chi = 12, 8
    E, mps, _ = dmrg(tfim_mpo(N, 1.0), chi=chi, n_sweeps=6, tol=1e-12)
    assert isinstance(mps, MPS) and isinstance(mps, list)
    assert len(mps) == N
    assert [A.shape for A in mps] == [A.shape for A in list(mps)]
    assert mps[0].shape[0] == 1 and mps[-1].shape[2] == 1        # open boundaries
    for A, B in zip(mps, mps[1:]):                               # slicing still works
        assert A.shape[2] == B.shape[0]
    plain = list(mps)                                            # and round-trips
    assert as_mps(plain) is not plain and isinstance(as_mps(plain), MPS)
    assert as_mps(mps) is mps


def test_mps_methods_delegate_to_observables():
    _, mps, _ = dmrg(tfim_mpo(16, 1.0), chi=16, n_sweeps=8, tol=1e-12)
    assert np.array_equal(mps.entropies(), all_entropies(mps))
    host = mps.to_numpy()
    assert isinstance(host, MPS)
    for A, B in zip(host, to_numpy(mps)):
        assert np.array_equal(A, B)
    # measuring must not gauge the caller's state (the observables' contract, and
    # the reason the methods can be called in any order)
    before = _bytes(mps)
    mps.entropies(); mps.to_numpy()
    assert _bytes(mps) == before


def test_observables_accept_a_bare_list_and_an_mps_alike():
    _, mps, _ = dmrg(tfim_mpo(14, 1.0), chi=12, n_sweeps=8, tol=1e-12)
    bare = list(mps)
    assert np.array_equal(all_entropies(mps), all_entropies(bare))
    assert structure_factor(mps, SX) == structure_factor(bare, SX)


def test_mps_repr_is_a_summary():
    _, mps, _ = dmrg(tfim_mpo(10, 1.0), chi=8, n_sweeps=4)
    r = repr(mps)
    assert r.startswith("MPS(N=10") and "chi_max=" in r
    assert len(r) < 120                       # not 2 N chi^2 numbers
    assert repr(MPS()) == "MPS(empty)"
