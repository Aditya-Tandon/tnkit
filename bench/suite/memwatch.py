"""Memory and GPU-sharing samples taken from outside the timed process.

`Watch(pid)` samples, every 0.5 s on a thread of the scheduler, the resident memory of the run's
whole process tree (from /proc, so it covers Julia children) and the memory in use on this pod's
GPU or MIG instance (NVML, through ctypes: the image has no NVML package). It never touches the
timed process. Standalone, it checks that the GPU is idle when nothing of ours runs on it:

    python3 bench/suite/memwatch.py idle --seconds 60 --out cotenancy_start.json

A MIG instance is exclusive by construction; on a full GPU, memory in use, a utilisation above 0
or a compute process while we run nothing means another tenant.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import threading
import time

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _children():
    """{ppid: [pid, ...]} from /proc."""
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            out.setdefault(ppid, []).append(int(d))
    return out


def tree_rss_mb(pid):
    """Resident memory of `pid` and all its descendants, in MB (None off Linux)."""
    if not os.path.isdir("/proc"):
        return None
    kids, todo, total = _children(), [pid], 0
    while todo:
        p = todo.pop()
        todo += kids.get(p, [])
        try:
            with open(f"/proc/{p}/statm") as f:
                total += int(f.read().split()[1]) * PAGE
        except (OSError, IndexError, ValueError):
            pass
    return total / 2 ** 20


def visible_uuid():
    """The MIG instance's UUID if the pod has one, else the GPU's (from nvidia-smi -L)."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    mig = re.findall(r"UUID: (MIG-[0-9a-f-]+)", out)
    gpu = re.findall(r"UUID: (GPU-[0-9a-f-]+)", out)
    return (mig[0] if mig else (gpu[0] if gpu else None)), (gpu[0] if gpu else None)


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Nvml:
    """One device (a MIG instance or a whole GPU), found by UUID."""

    def __init__(self, uuid):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        self.uuid, self.h = uuid, ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.h)) != 0:
            self.h = self._mig_by_scan(uuid)

    def _mig_by_scan(self, uuid):
        parent, n, buf = ctypes.c_void_p(), ctypes.c_uint(), ctypes.create_string_buffer(96)
        self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(parent))
        self.lib.nvmlDeviceGetMaxMigDeviceCount(parent, ctypes.byref(n))
        for i in range(n.value):
            h = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetMigDeviceHandleByIndex(parent, i, ctypes.byref(h)) == 0:
                self.lib.nvmlDeviceGetUUID(h, buf, 96)
                if buf.value.decode() == uuid:
                    return h
        raise OSError(f"no NVML handle for {uuid}")

    def used_mb(self):
        m = _Mem()
        return m.used / 2 ** 20 if self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m)) == 0 \
            else None

    def total_mb(self):
        m = _Mem()
        return m.total / 2 ** 20 if self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m)) == 0 \
            else None

    def n_procs(self):
        n = ctypes.c_uint(0)
        rc = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.h, ctypes.byref(n), None)
        return n.value if rc in (0, 7) else None          # 7: buffer too small, count is set

    def util(self):
        u = _Util()
        return u.gpu if self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) == 0 \
            else None                                      # not supported on a MIG instance


def gpu():
    """An Nvml for this pod's device, or None (no GPU, or NVML unusable: recorded as such)."""
    uuid, _ = visible_uuid()
    if not uuid:
        return None
    try:
        return Nvml(uuid)
    except OSError:
        return None


def sample(dev):
    if dev is None:
        return {}
    return {"gpu_used_mb": dev.used_mb(), "gpu_procs": dev.n_procs(), "gpu_util": dev.util()}


class Watch(threading.Thread):
    """Peak tree RSS and peak device memory while `pid` runs; `stop()` returns the summary."""

    def __init__(self, pid, dev, period=0.5):
        super().__init__(daemon=True)
        self.pid, self.dev, self.period = pid, dev, period
        self.before = sample(dev)
        self.halt = threading.Event()
        self.rss, self.used, self.n = 0.0, 0.0, 0

    def run(self):
        while not self.halt.is_set():
            r = tree_rss_mb(self.pid)
            u = self.dev.used_mb() if self.dev else None
            self.rss = max(self.rss, r or 0.0)
            self.used = max(self.used, u or 0.0)
            self.n += 1
            self.halt.wait(self.period)

    def stop(self):
        self.halt.set()
        self.join(timeout=5)
        return {"peak_tree_rss_mb": self.rss, "peak_gpu_used_mb": self.used if self.dev else None,
                "samples": self.n, "period_s": self.period, "before": self.before}


def idle(seconds, out):
    """Device samples once a second while this pod runs nothing on the GPU."""
    uuid, parent = visible_uuid()
    dev = gpu()
    s = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        s.append({"t": round(time.time() - t0, 1), **sample(dev)})
        time.sleep(1.0)
    used = [x["gpu_used_mb"] for x in s if x.get("gpu_used_mb") is not None]
    util = [x["gpu_util"] for x in s if x.get("gpu_util") is not None]
    procs = [x["gpu_procs"] for x in s if x.get("gpu_procs") is not None]
    rec = {"uuid": uuid, "parent_uuid": parent, "mig": bool(uuid and uuid.startswith("MIG-")),
           "nvml": dev is not None, "total_mb": dev.total_mb() if dev else None,
           "max_used_mb": max(used, default=None), "max_util": max(util, default=None),
           "max_procs": max(procs, default=None), "samples": s}
    rec["foreign_seen"] = bool((rec["max_used_mb"] or 0) > 100 or (rec["max_util"] or 0) > 0
                               or (rec["max_procs"] or 0) > 0)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("idle", {k: v for k, v in rec.items() if k != "samples"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["idle"])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idle(a.seconds, a.out)
