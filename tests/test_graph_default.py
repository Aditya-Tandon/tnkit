"""Prefix-graph capture as the default: the `graph` resolution logic, and its bit-identity on CPU.

WHAT THIS PINS. `tnkit.tdvp.resolve_graph_mode` is a pure function of (`graph`, `backend`,
`cuda_available`) with no GPU dependency at all (`cuda_available` overrides the live probe),
so every branch of the 'auto'/'on'/'off' rule is exercised here without a GPU. What is NOT
re-tested here: that a captured graph replays bit-identically to its eager twin (measured
on a GPU at 0.000e+00), and `tests/test_prefix_identity.py` already pins the
(device-independent) arithmetic contract the prefix graph replays. This file is only
about which path gets CHOSEN, on which backend, by default.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from tnkit.backend import get_backend
from tnkit.block_backend import BlockBackend
from tnkit.models import tfim_mpo
from tnkit.tdvp import GRAPH_MODES, resolve_graph_mode, tdvp1_evolve


# --------------------------------------------------------------------------- resolve_graph_mode
@pytest.mark.parametrize("backend", ["numpy", "torch-cpu", "torch-mps"])
def test_auto_never_captures_off_cuda(backend):
    """'auto' on any non-CUDA backend name always resolves 'eager', regardless of the CUDA
    probe -- a torch-cuda BUILD with no visible device must fall back exactly the same way a
    numpy/torch-cpu/torch-mps backend does."""
    assert resolve_graph_mode("auto", backend) == "eager"
    assert resolve_graph_mode("auto", backend, cuda_available=True) == "eager"


def test_auto_on_torch_cuda_follows_the_live_cuda_probe():
    assert resolve_graph_mode("auto", "torch-cuda", cuda_available=False) == "eager"
    assert resolve_graph_mode("auto", "torch-cuda", cuda_available=True) == "capture"


def test_off_is_always_eager():
    for backend in ("numpy", "torch-cpu", "torch-mps", "torch-cuda"):
        assert resolve_graph_mode("off", backend) == "eager"
        assert resolve_graph_mode("off", backend, cuda_available=True) == "eager"


def test_on_requires_torch_cuda_with_a_device_or_raises():
    """'on' never silently falls back -- an explicit request that could not be honoured is a
    configuration error, not a quiet no-op. This IS this project's CUDA-path test: it proves
    the guard fires correctly with no GPU present, i.e. the import/fallback behaviour, which
    is all a machine with no CUDA device can verify."""
    assert resolve_graph_mode("on", "torch-cuda", cuda_available=True) == "capture"
    for backend in ("numpy", "torch-cpu", "torch-mps"):
        with pytest.raises(ValueError, match="graph='on' requires"):
            resolve_graph_mode("on", backend, cuda_available=True)
    with pytest.raises(ValueError, match="graph='on' requires"):
        resolve_graph_mode("on", "torch-cuda", cuda_available=False)


def test_unrecognised_graph_mode_raises():
    assert set(GRAPH_MODES) == {"auto", "on", "off"}
    with pytest.raises(ValueError, match="graph must be one of"):
        resolve_graph_mode("sometimes", "numpy")


def test_auto_probes_torch_cuda_availability_when_not_overridden(monkeypatch):
    """With no override, 'auto' asks `torch.cuda.is_available()` -- and on a machine with no
    torch installed at all (this project's core dependency is only numpy/scipy), the probe
    must fail closed to 'eager' rather than raising ImportError."""
    import tnkit.tdvp as T
    monkeypatch.setattr(T, "_cuda_capture_available", lambda: False)
    assert T.resolve_graph_mode("auto", "torch-cuda") == "eager"
    monkeypatch.setattr(T, "_cuda_capture_available", lambda: True)
    assert T.resolve_graph_mode("auto", "torch-cuda") == "capture"


def _cuda(real=False, index=None, is_torch=True):
    return types.SimpleNamespace(is_torch=is_torch,
                                 device=types.SimpleNamespace(type="cuda", index=index),
                                 is_complex=not real)


def _torch(spec):
    pytest.importorskip("torch")
    return get_backend(spec)                            # needs no GPU or MPS to build


def _torch_cuda_object(real=False):
    return _torch("torch-cuda-real" if real else "torch-cuda")


@pytest.mark.parametrize("make", [_cuda, lambda: _cuda(index=0), _torch_cuda_object,
                                  lambda: _torch("torch-cuda:0"), lambda: "torch-cuda:0"],
                         ids=["fake-object", "fake-object-0", "TorchBackend", "TorchBackend-0",
                              "indexed-spec"])
def test_cuda_backend_objects_and_indexed_specs_follow_the_torch_cuda_rule(make):
    """The rule reads the backend's device and dtype, not the literal string 'torch-cuda'."""
    b = make()
    assert resolve_graph_mode("auto", b, cuda_available=True) == "capture"
    assert resolve_graph_mode("on", b, cuda_available=True) == "capture"
    assert resolve_graph_mode("auto", b, cuda_available=False) == "eager"
    with pytest.raises(ValueError, match="graph='on' requires.*cuda_available=False"):
        resolve_graph_mode("on", b, cuda_available=False)


@pytest.mark.parametrize("make", [
    lambda: "torch-cuda-real", lambda: "torch-cuda:0-real", lambda: _cuda(real=True),
    lambda: _torch_cuda_object(real=True), lambda: BlockBackend(_torch_cuda_object()),
    lambda: "auto", lambda: get_backend("numpy"),
    lambda: "torch-cuda:1", lambda: "torch-cuda:1-real", lambda: _torch("torch-cuda:1"),
    lambda: _cuda(index=1), lambda: _torch("torch-mps"), lambda: _cuda(is_torch=False),
    lambda: "torch-cuda:", lambda: "torch-cuda:abc", lambda: "torch-cudax"],
    ids=["real-spec", "indexed-real-spec", "real-fake", "real-TorchBackend", "block",
         "auto-spec", "NumpyBackend", "cuda1-spec", "cuda1-real-spec", "cuda1-TorchBackend",
         "cuda1-fake", "mps-TorchBackend", "not-torch-fake", "empty-index", "bad-index",
         "bad-device"])
def test_real_block_numpy_and_other_device_backends_never_capture(make):
    """Capture buffers take the backend dtype, so a real one would drop the imaginary part of
    exp(-i dt H); block TDVP takes no step graphs; 'auto' resolves to a numpy spec; capture
    records on the current CUDA device only; a spec `get_backend` rejects is no target."""
    b = make()
    assert resolve_graph_mode("auto", b, cuda_available=True) == "eager"
    with pytest.raises(ValueError, match="graph='on' requires"):
        resolve_graph_mode("on", b, cuda_available=True)


def test_on_refusal_names_the_backend():
    pytest.importorskip("torch")
    for b in ("torch-cuda:1", get_backend("torch-cuda:1")):
        with pytest.raises(ValueError, match=r"got backend='torch-cuda:1'\. "):
            resolve_graph_mode("on", b, cuda_available=True)
    with pytest.raises(ValueError, match=r"got backend='torch-cpu'\. "):
        resolve_graph_mode("on", get_backend("torch-cpu"))
    with pytest.raises(ValueError, match=r"got backend='BlockBackend\(torch-cuda\)'\. "):
        resolve_graph_mode("on", BlockBackend(get_backend("torch-cuda")), cuda_available=True)


def test_probe_runs_only_for_a_capture_target(monkeypatch):
    import tnkit.tdvp as T
    calls = []
    monkeypatch.setattr(T, "_cuda_capture_available", lambda: calls.append(1) or True)
    for b in ("numpy", "torch-cpu", "torch-cuda-real", get_backend("numpy"), _cuda(real=True)):
        assert T.resolve_graph_mode("auto", b) == "eager"
    assert calls == []
    assert T.resolve_graph_mode("auto", _cuda()) == "capture" and calls == [1]


# --------------------------------------------------------------------------- tdvp1_evolve wiring
def _small_problem(backend):
    n, chi = 10, 6
    rng = np.random.default_rng(2026)
    mpo = tfim_mpo(n, g=1.0, h=0.10)
    dims = [1] + [min(chi, 2 ** min(i + 1, n - i - 1), chi) for i in range(n - 1)] + [1]
    mps0 = [(rng.normal(size=(dims[i], 2, dims[i + 1]))
             + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))) / 10.0 for i in range(n)]
    return mps0, mpo


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_default_graph_matches_explicit_off_bit_for_bit_on_cpu(backend):
    """The claim this project cares about most: leaving `graph` at its default ('auto')
    changes NOTHING on a backend that cannot capture. Trivial on CPU (both resolve 'eager'),
    which is exactly the point -- this is the one comparison that needs no GPU to be
    meaningful, and it is the one every CPU-only run of this pipeline actually exercises.
    """
    if backend.startswith("torch"):
        pytest.importorskip("torch")
    mps0, mpo = _small_problem(backend)

    out_default, _ = tdvp1_evolve([m.copy() for m in mps0], mpo, 0.05, 6, backend=backend)
    out_off, _ = tdvp1_evolve([m.copy() for m in mps0], mpo, 0.05, 6, backend=backend,
                              graph="off")
    assert len(out_default) == len(out_off)
    for i, (d, o) in enumerate(zip(out_default, out_off)):
        assert np.asarray(d).tobytes() == np.asarray(o).tobytes(), f"site {i} differs"


@pytest.mark.parametrize("spec", ["numpy", "torch-cpu"])
def test_default_graph_on_a_backend_object_matches_explicit_off(spec):
    if spec.startswith("torch"):
        pytest.importorskip("torch")
    from tnkit.tdvp_evolver import TDVPEvolver
    mps0, mpo = _small_problem(spec)
    be = get_backend(spec)
    assert resolve_graph_mode("auto", be, cuda_available=True) == "eager"
    a, _ = tdvp1_evolve([m.copy() for m in mps0], mpo, 0.05, 3, backend=be)
    b, _ = tdvp1_evolve([m.copy() for m in mps0], mpo, 0.05, 3, backend=spec, graph="off")
    assert [np.asarray(x).tobytes() for x in a] == [np.asarray(x).tobytes() for x in b]
    ev = TDVPEvolver([m.copy() for m in mps0], mpo, 0.05, 6, backend=be)
    assert ev._graphs == (None, None)


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_graph_on_raises_on_a_backend_that_cannot_capture(backend):
    """Import/fallback-only verification of the CUDA path on a machine with no GPU: forcing
    `graph='on'` on numpy/torch-cpu must fail loudly, before any tensor work happens, rather
    than silently running eager."""
    if backend.startswith("torch"):
        pytest.importorskip("torch")
    mps0, mpo = _small_problem(backend)
    with pytest.raises(ValueError, match="graph='on' requires"):
        tdvp1_evolve([m.copy() for m in mps0], mpo, 0.05, 2, backend=backend, graph="on")


def test_graph_field_is_one_site_only():
    """`graph` (and its sizing knobs) belong to `tdvp1_evolve`, exactly like `cuda_graph` and
    `prefix_graph` already did -- passing it to `tdvp2_evolve` must raise the same loud,
    named error the existing one-site-only fields already raise, not be silently ignored."""
    from tnkit.tdvp import tdvp2_evolve
    n, chi = 8, 4
    rng = np.random.default_rng(1)
    mpo = tfim_mpo(n, g=1.0, h=0.10)
    dims = [1] + [min(chi, 2 ** min(i + 1, n - i - 1), chi) for i in range(n - 1)] + [1]
    mps0 = [(rng.normal(size=(dims[i], 2, dims[i + 1]))) for i in range(n)]
    with pytest.raises(TypeError, match="graph"):
        tdvp2_evolve(mps0, mpo, chi, 0.05, 1, backend="numpy", graph="off")


_NO_TORCH = """
import sys
import numpy as np
from tnkit.models import tfim_mpo
from tnkit.tdvp import resolve_graph_mode, tdvp1_evolve

rng = np.random.default_rng(0)
mps = [rng.normal(size=(1 if i == 0 else 4, 2, 1 if i == 5 else 4)) for i in range(6)]
assert "torch" not in sys.modules, "importing tnkit already imports torch"
tdvp1_evolve(mps, tfim_mpo(6, g=1.0, h=0.1), 0.05, 2, backend="numpy")
assert "torch" not in sys.modules, "tdvp1_evolve on numpy imported torch"
resolve_graph_mode("auto", "torch-cuda")            # control: this probe does import torch
assert "torch" in sys.modules
"""


_NO_TORCH_ON = """
import sys
from tnkit.backend import get_backend
from tnkit.tdvp import resolve_graph_mode

for b in ("numpy", "numpy-real", get_backend("numpy"), get_backend("numpy-real")):
    assert resolve_graph_mode("auto", b) == "eager"
    try:
        resolve_graph_mode("on", b)
    except ValueError as e:
        assert "graph='on' requires" in str(e) and "cuda_available" not in str(e), e
    else:
        raise AssertionError(f"graph='on' with {b!r} did not raise")
assert "torch" not in sys.modules, "the graph='on' refusal imported torch"
"""


def test_graph_on_refusal_on_numpy_does_not_import_torch():
    """Numpy specs and objects need no CUDA probe, so neither 'auto' nor the 'on' refusal may
    import torch."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", _NO_TORCH_ON], capture_output=True, text=True,
                       timeout=100)
    assert r.returncode == 0, r.stderr


def test_numpy_tdvp1_does_not_import_torch():
    """Only a torch-cuda backend may probe CUDA; numpy never pays for the torch import. The
    control at the end shows the probe would have imported torch had it run."""
    import importlib.util
    import subprocess
    import sys
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed, so the check could not fail")
    r = subprocess.run([sys.executable, "-c", _NO_TORCH], capture_output=True, text=True,
                       timeout=100)
    assert r.returncode == 0, r.stderr
