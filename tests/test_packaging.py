"""The packaging claim, verified rather than assumed.

`pyproject.toml` makes torch an extra, and that only works because torch is
imported lazily -- inside `get_backend()` and `TorchBackend.__init__`, never at
module scope. It is the kind of property that holds until someone adds a
convenience import at the top of a file, so it is asserted here.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBE = r"""
import sys, importlib

class Forbid:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("tnkit imported torch at module scope: " + name)
        return None

sys.meta_path.insert(0, Forbid())
import tnkit
for m in ("tnkit.backend", "tnkit.dmrg", "tnkit.observables", "tnkit.models",
          "tnkit.linalg", "tnkit.eigensolver", "tnkit.tdvp", "tnkit.tebd",
          "tnkit.idmrg", "tnkit.vumps", "tnkit.excitations", "tnkit.window",
          "tnkit.krylov", "tnkit.itdvp", "tnkit.dmrg_excited", "tnkit.free_fermion",
          "tnkit.cudagraph", "tnkit.cusolver_gesvdr", "tnkit.luscher",
          "tnkit.two_particle", "tnkit.prefix_graph", "tnkit.prefix_graph_impl",
          "tnkit.free_fermion_dynamics", "tnkit.opsum", "tnkit.models_lgt",
          # the private implementations and the shim helper
          "tnkit._dmrg", "tnkit._vumps", "tnkit._idmrg", "tnkit._shim", "tnkit._options",
          "tnkit.api", "tnkit.mps",
          # the field-theory subpackage must not drag torch in either
          "tnkit.ift", "tnkit.ift.luscher", "tnkit.ift.two_particle",
          "tnkit.ift.prefix_graph"):
    importlib.import_module(m)
assert "torch" not in sys.modules, sorted(k for k in sys.modules if k.startswith("torch"))
print("OK")
"""


def test_import_tnkit_does_not_pull_in_torch():
    """Importing every tnkit module with torch forbidden must still succeed.

    A meta-path finder that raises on `import torch` is stricter than checking
    `sys.modules` afterwards: it also catches a module that imports torch inside
    a try/except and swallows the result.
    """
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_pyproject_declares_a_thin_core():
    """numpy and scipy only; torch lives in the extras."""
    import tomllib
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    deps = {d.split(">")[0].split("=")[0].strip() for d in cfg["project"]["dependencies"]}
    assert deps == {"numpy", "scipy"}
    extras = cfg["project"]["optional-dependencies"]
    assert "torch" in extras and "cuda" in extras
    # the engine and its field-theory subpackage, listed rather than auto-discovered
    assert cfg["tool"]["setuptools"]["packages"] == ["tnkit", "tnkit.ift"]


def test_the_distribution_is_the_library_and_nothing_else(tmp_path):
    """Build a real sdist and a real wheel, and look inside them.

    Every top-level directory of the checkout other than `tnkit/` (benchmarks, tests, docs,
    and whatever else the checkout holds) stays in git and out of a `pip install`.
    `MANIFEST.in` prunes the sdist and `[tool.setuptools] packages`
    bounds the wheel, but neither is worth much as a declaration: this builds both and
    asserts on the members, which is the only form of the claim that cannot rot.
    """
    import tarfile
    import zipfile
    try:
        import setuptools.build_meta as backend
    except Exception:                                          # pragma: no cover
        import pytest
        pytest.skip("no PEP 517 setuptools backend available")

    # setuptools writes `tnkit.egg-info/` and `build/` into the SOURCE tree, and the
    # egg-info carries a PKG-INFO that `importlib.metadata` will then discover off
    # sys.path. Left behind, it becomes stale the moment `pyproject.toml`'s version
    # changes, and `tnkit.__version__` -- which reads installed metadata first -- starts
    # reporting the old number in this very checkout. That is exactly the "stale install"
    # failure `test_version_matches_pyproject` exists to catch, and it would be this test
    # that caused it. So anything this test creates, this test removes.
    import shutil
    residue = [os.path.join(REPO, d) for d in ("tnkit.egg-info", "build")]
    pre_existing = {p for p in residue if os.path.exists(p)}

    out = str(tmp_path)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        sdist = backend.build_sdist(out)
        wheel = backend.build_wheel(out)
    finally:
        os.chdir(cwd)
        for p in residue:
            if p not in pre_existing and os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)

    with tarfile.open(os.path.join(out, sdist)) as tf:
        sdist_members = [m.split("/", 1)[1] for m in tf.getnames() if "/" in m]
    with zipfile.ZipFile(os.path.join(out, wheel)) as zf:
        wheel_members = zf.namelist()

    forbidden = tuple(d + "/" for d in os.listdir(REPO)
                      if os.path.isdir(os.path.join(REPO, d)) and d not in ("tnkit", "docs")
                      and not d.endswith(".egg-info") and d not in ("build", ".git"))
    for label, members in (("sdist", sdist_members), ("wheel", wheel_members)):
        bad = [m for m in members if m.startswith(forbidden)]
        assert not bad, f"the {label} ships more than the library: {bad[:8]}"

    # and it does ship the library, both packages, plus the files that make it usable
    assert "tnkit/__init__.py" in wheel_members
    assert "tnkit/ift/luscher.py" in wheel_members, "the ift subpackage is not packaged"
    for f in ("LICENSE", "README.md", "CHANGELOG.md", "pyproject.toml"):
        assert f in sdist_members, f"the sdist is missing {f}"
