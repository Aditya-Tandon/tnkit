"""Shared pieces of the library benchmark: the spec, the run record, timing, environment capture.

A run is one process: one library, backend, symmetry, workload, bond dimension and repeat. It
warms up on a tiny instance of the same workload, times the real one, and appends one JSON line.
Runners return raw outputs only (energies, profiles, times); errors against references are
computed afterwards by `analyse.py`, so no runner needs the reference code or its environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

SCHEMA = "tnbench/1"


class Unsupported(Exception):
    """A run this library cannot do (recorded as `unsupported`; the arm leaves the workload)."""


def require_data(w):
    """Refuse a workload whose input files sit under an unset environment variable.

    The `tf_collision` workload continues a saved two-flavour Schwinger state (`state`, `mpo`:
    npz files under `$TNBENCH_DATA`). Those files are not distributed with the suite.
    """
    for key in ("state", "mpo"):
        if key in w and "$" in os.path.expandvars(w[key]):
            raise Unsupported(f"{w[key]} needs TNBENCH_DATA set to the directory holding the saved "
                              "state and MPO of this workload, which are not distributed")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_spec(path):
    with open(path, "rb") as f:
        raw = f.read()
    spec = json.loads(raw)
    spec["_sha256"] = hashlib.sha256(raw).hexdigest()
    spec["_path"] = os.path.abspath(path)
    return spec


def load_settings(spec):
    """Per-library settings named by the spec ("settings", relative to the spec), or by
    TNBENCH_SETTINGS; ({}, None, None) for a spec without them."""
    path = os.environ.get("TNBENCH_SETTINGS") or spec.get("settings")
    if not path:
        return {}, None, None
    path = os.path.join(os.path.dirname(spec["_path"]), path)
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), os.path.abspath(path)


def settings_for(settings, lib, kind, workload):
    """The settings of one run: the library's settings for this kind of run, overridden by
    those for this workload."""
    s = settings.get(lib, {})
    return {**s.get(kind, {}), **s.get(workload, {})}


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cuda_backend(backend):
    return backend.startswith("torch-cuda")


def sync(backend):
    if cuda_backend(backend):
        import torch
        torch.cuda.synchronize()


class Clock:
    """Wall time with a device sync on both sides."""

    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        sync(self.backend)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.backend)
        self.s = time.perf_counter() - self.t0


def laps(iterable, backend):
    """Yield (item, seconds since the previous item, or since the start), device-synced."""
    sync(backend)
    last = time.perf_counter()
    for item in iterable:
        sync(backend)
        now = time.perf_counter()
        yield item, now - last
        last = time.perf_counter()


def set_torch_threads(n):
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(n)


def reset_gpu_peak(backend):
    if cuda_backend(backend):
        import torch
        torch.cuda.reset_peak_memory_stats()


def peaks(backend):
    out = {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    if sys.platform == "darwin":                 # ru_maxrss is bytes on macOS, kB on Linux
        out["peak_rss_mb"] /= 1024.0
    if cuda_backend(backend):
        import torch
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return out


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()


def _cgroup_cpus():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        return None


def _versions(names):
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def environment(backend):
    """Interpreter, packages, threads and hardware of this process."""
    env = {
        "python": platform.python_version(), "executable": sys.executable,
        "packages": _versions(["numpy", "scipy", "torch", "yastn", "physics-tenpy", "qtealeaves",
                               "qredtea", "h5py", "block2", "psutil"]),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                    "MKL_NUM_THREADS")},
        "hostname": socket.gethostname(), "cpu_model": _cpu_model(), "nproc": os.cpu_count(),
        "cgroup_cpus": _cgroup_cpus(),
        "code_sha": os.environ.get("TN_HEAD_SHA"), "image_id": os.environ.get("TN_IMAGE_ID"),
        "env_name": os.environ.get("TNBENCH_ENV"),
        "node_selector": os.environ.get("TNBENCH_NODE_SELECTOR"),
    }
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts")
        env["blas"] = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:  # noqa: BLE001 -- old numpy has no mode="dicts"
        env["blas"] = None
    try:
        import torch
        env["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except ImportError:
        pass
    if "cuda" in backend:
        env["nvidia_smi_L"] = _cmd(["nvidia-smi", "-L"])
        env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        env["nvidia_visible_devices"] = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    return env


def append_record(path, rec):
    """One JSON line, written in one call and synced (network filesystems truncate partial
    writes)."""
    line = json.dumps(rec, default=float) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def neel(N):
    """Up on even sites, down on odd (index 0 = up = Sz +1/2)."""
    return [0 if i % 2 == 0 else 1 for i in range(N)]


def bonds(w):
    """Coupled pairs (i < j) of a spin model: nearest neighbours of an open chain, or of an
    Lx x Ly cylinder (open in x, periodic in y) on the column-major path i = x * Ly + y."""
    if w["model"] in ("xxz", "xx"):
        return [(i, i + 1) for i in range(w["N"] - 1)]
    if w["model"] == "heis_cyl":
        Lx, Ly = w["Lx"], w["Ly"]
        if Ly < 3 or Lx * Ly != w["N"]:
            raise ValueError("heis_cyl needs Ly >= 3 and N = Lx * Ly")
        s = [(x * Ly + y, x * Ly + (y + 1) % Ly) for x in range(Lx) for y in range(Ly)]
        s += [(x * Ly + y, (x + 1) * Ly + y) for x in range(Lx - 1) for y in range(Ly)]
        return sorted((min(a, b), max(a, b)) for a, b in s)
    raise ValueError(w["model"])


def couplings(w):
    """(J, Jz) of an XXZ-type model: H = sum over bonds of J (SxSx + SySy) + Jz SzSz."""
    if w["model"] == "heis_cyl":
        return w["J"], w["J"]
    return w["J"], w.get("Jz", 0.0)


def schedule_of(w, chi):
    """DMRG stages as (bond dimension, sweeps): the workload's ramp (fractions of chi, two sweeps
    each by default), then `max_sweeps` at chi. A `fixed_sweeps` workload is one stage."""
    if w.get("fixed_sweeps"):
        return [(chi, w["fixed_sweeps"])]
    ramp = [(max(1, int(round(chi * f))), n) for f, n in w.get("ramp", [])]
    return ramp + [(chi, w["max_sweeps"])]


def stop_tol(w):
    """Absolute energy-change stop of the final stage: e_tol, or e_tol_rel times the workload's
    energy scale (a fixed constant, so that every library stops on the same rule)."""
    if "e_tol_rel" in w:
        return w["e_tol_rel"] * w["energy_scale"]
    return w["e_tol"]


def sweep_time(r):
    """(mean seconds per sweep over sweeps 2-3 of a DMRG record, whether it is the whole run
    divided by its sweeps instead). Sweep times summing to more than 1.5x the wall time are read
    as cumulative."""
    s = r.get("sweep_s")
    if s and len(s) >= 2:
        s = [float(x) for x in s]
        if sum(s) > 1.5 * r["wall_s"]:
            s = [b - a for a, b in zip([0.0] + s, s)]
        return sum(s[1:3]) / len(s[1:3]), False
    return r["wall_s"] / max(r.get("sweeps") or 1, 1), True


def options(w, allowed):
    """The library settings of this run (`run.py` puts them in w["opts"]); unknown keys raise,
    so a misspelt or misplaced setting fails the smoke instead of being ignored."""
    opts = dict(w.get("opts", {}))
    bad = sorted(set(opts) - set(allowed))
    if bad:
        raise ValueError(f"unknown settings {bad}; this runner takes {sorted(allowed)}")
    return opts
