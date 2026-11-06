"""Python side of the Julia probe pod (standard library, numpy, scipy and torch only).

    python3 bench/suite/julia_probe.py net   --out net.json
    python3 bench/suite/julia_probe.py fetch URL PATH
    python3 bench/suite/julia_probe.py ref   --out ref.json      # tnkit and exact energies
    python3 bench/suite/julia_probe.py calib --out calib.json    # torch GEMM and SVD rates on CUDA
    python3 bench/suite/julia_probe.py check --part cpu|gpu --dir OUTDIR [--cpu-dir CPUDIR]
                                                                 # verdict.json; exit 0 iff PASS

The reference is the Heisenberg chain of the Julia smoke (N 16, chi 64): tnkit two-site DMRG,
dense and U(1), on numpy, and sparse exact diagonalisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-10                       # |E(ITensor) - E(tnkit)|, and |E(tnkit) - E(exact)|
URLS = {
    "julia_s3": "https://julialang-s3.julialang.org/bin/versions.json",
    "pkg_server": "https://pkg.julialang.org/registries",
    "github": "https://github.com/JuliaRegistries/General",
    "github_release_asset": "https://github.com/JuliaBinaryWrappers/CUDA_Runtime_jll.jl/releases/"
                            "download/CUDA_Runtime-v0.25.0+3/"
                            "CUDA_Runtime.v0.25.0.x86_64-linux-gnu-cuda+13.1.tar.gz",
    "pypi": "https://pypi.org/simple/numpy/",
}


def net(out):
    rec = {}
    for name, url in URLS.items():
        t = time.time()
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-1023",
                                                       "User-Agent": "julia-probe"})
            with urllib.request.urlopen(req, timeout=20) as r:
                n = len(r.read(1024))
                rec[name] = {"ok": True, "status": r.status, "bytes": n, "final_url": r.geturl()[:200]}
        except Exception as e:  # noqa: BLE001  (any failure is the measurement)
            rec[name] = {"ok": False, "error": repr(e)[:300]}
        rec[name]["s"] = round(time.time() - t, 2)
        print("net", name, rec[name])
    dump(out, rec)


def fetch(url, path):
    h, n, t = hashlib.sha256(), 0, time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            h.update(b)
            f.write(b)
            n += len(b)
    print(f"fetched {n} bytes in {time.time() - t:.1f} s, sha256 {h.hexdigest()}")


def ref(out, N=16, chi=64):
    sys.path[:0] = [HERE, os.path.join(HERE, "runners"), os.path.dirname(os.path.dirname(HERE))]
    import tnkit_runner as tk
    from reference import xxz_energy
    w = {"kind": "dmrg", "model": "xxz", "N": N, "J": 1.0, "Jz": 1.0, "max_sweeps": 20,
         "e_tol": 1e-12}
    rec = {"N": N, "chi": chi, "E_exact": xxz_energy(N, 1.0, 1.0)}
    for sym in ("dense", "u1"):
        r = tk.dmrg(w, chi, backend="numpy", sym=sym)
        rec[sym] = {k: r[k] for k in ("energy", "sweeps", "max_bond", "wall_s")}
        rec[sym]["minus_exact"] = r["energy"] - rec["E_exact"]
    dump(out, rec)
    print("ref", json.dumps(rec))


def calib(out):
    import torch
    rec = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    if rec["cuda_available"]:
        dev = torch.device("cuda")

        def rate(f, flops, reps=5):
            f()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                f()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            m = sorted(ts)[(reps - 1) // 2]
            return {"median_s": m, "tflops": flops / m / 1e12}

        n, m = 4096, 1024
        a, b = (torch.rand(n, n, dtype=torch.float64, device=dev) for _ in range(2))
        z, y = (torch.rand(n, n, dtype=torch.complex128, device=dev) for _ in range(2))
        s, sz = torch.rand(m, m, dtype=torch.float64, device=dev), torch.rand(m, m, dtype=torch.complex128, device=dev)
        rec["dgemm_4096"] = rate(lambda: a @ b, 2.0 * n ** 3)
        rec["zgemm_4096"] = rate(lambda: z @ y, 8.0 * n ** 3)
        rec["dsvd_1024"] = rate(lambda: torch.linalg.svd(s), 22.0 * m ** 3, reps=3)
        rec["zsvd_1024"] = rate(lambda: torch.linalg.svd(sz), 88.0 * m ** 3, reps=3)
        rec["device"] = torch.cuda.get_device_name(0)
    dump(out, rec)
    print("calib", json.dumps(rec))


def load(d, name):
    try:
        with open(os.path.join(d, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def lines(d, name):
    try:
        with open(os.path.join(d, name)) as f:
            return [x for x in f.read().split("\n") if x]
    except OSError:
        return None


def steps(d):
    rows = {}
    try:
        with open(os.path.join(d, "steps.tsv")) as f:
            for line in f:
                name, rc, sec = line.rstrip("\n").split("\t")[:3]
                rows[name] = {"rc": int(rc), "s": float(sec)}
    except OSError:
        pass
    return rows


def arm(smoke, name, e_ref, cuda=False):
    """One ITensor arm against the tnkit energy of the same symmetry."""
    r = (smoke or {}).get(name)
    if not r:
        return {"pass": False, "why": "not run"}
    out = {"status": r.get("status"), "energy": r.get("energy"), "wall_s_first": r.get("wall_s"),
           "wall_s_warm": r.get("wall_s_warm"), "sweeps": r.get("sweeps"), "max_bond": r.get("max_bond")}
    if r.get("status") != "ok":
        out.update(error=(r.get("error") or "")[:500], **{"pass": False})
        if r.get("retry_without_noise"):
            out["without_noise"] = arm({"x": r["retry_without_noise"]}, "x", e_ref, cuda)
        return out
    en = r.get("energy")
    out["minus_tnkit"] = en - e_ref if en is not None and e_ref is not None else None
    ok = out["minus_tnkit"] is not None and abs(out["minus_tnkit"]) <= TOL
    if cuda:
        out.update(on_device=r.get("on_device"), float64=r.get("float64"))
        ok = ok and bool(r.get("on_device")) and bool(r.get("float64"))
    out["pass"] = bool(ok)
    return out


def arms(d, name, e, cuda=False):
    """Dense and QN arms of one smoke file against tnkit dense and U(1)."""
    s = load(d, name)
    return {"dense": arm(s, "dense", e["dense"], cuda), "qn": arm(s, "qn", e["u1"], cuda)}


def reuse(d, e, devices):
    """Offline reuse of an unpacked depot: precompiled before loading, nothing compiled, smoke passes."""
    pre = (load(d, "loadcheck.json") or {}).get("precompiled_before_load") or {}
    new = lines(d, "depot_new_files.txt")
    smoke = {dv: arms(d, f"smoke_{dv}_reuse.json", e, dv == "cuda") for dv in devices}
    gated = [smoke["cpu"]["dense"], smoke["cpu"]["qn"]] + ([smoke["cuda"]["dense"]] if "cuda" in smoke else [])
    return {"precompiled_before_load": pre, "new_files": new, "smoke": smoke,
            "all_precompiled": bool(pre) and all(x is True for x in pre.values()),
            "no_recompile": new is not None and not any("/compiled/" in x for x in new),
            "smoke_pass": all(a["pass"] for a in gated)}


def check(d, cpu_dir, part):
    """Pass/fail summary. cpu: the CPU gates from cpu_dir. gpu: those plus the GPU gates from d."""
    st, rf = steps(cpu_dir), load(cpu_dir, "ref.json") or {}
    e = {s: (rf.get(s) or {}).get("energy") for s in ("dense", "u1")}
    ok = lambda *names: any(st.get(n, {}).get("rc") == 0 for n in names)  # noqa: E731
    cpu, ru = arms(cpu_dir, "smoke_cpu.json", e), reuse(cpu_dir, e, ["cpu"])
    mk = load(cpu_dir, "mpskit.json")
    if mk and mk.get("energy") is not None and e["dense"] is not None:
        mk["minus_tnkit"] = mk["energy"] - e["dense"]
    g = {"julia_tarball_verified": ok("julia", "unpack"),
         "depot_built": ok("unpack") or (ok("instantiate", "instantiate_git")
                                         and ok("precompile", "precompile_retry1")),
         "tnkit_reference_exact": all(abs((rf.get(s) or {}).get("minus_exact", 1.0)) <= TOL
                                      for s in ("dense", "u1")),
         "cpu_dense": cpu["dense"]["pass"], "cpu_qn": cpu["qn"]["pass"],
         "depot_tarball_readback": ok("tar_depot", "unpack"),
         "cpu_reuse_all_precompiled": ru["all_precompiled"], "cpu_reuse_no_recompile": ru["no_recompile"],
         "cpu_reuse_smoke": ru["smoke_pass"]}
    v = {"part": part, "steps_cpu": st, "ref": rf, "net_cpu": load(cpu_dir, "net.json"), "cpu": cpu,
         "cpu_reuse": ru, "mpskit": mk}
    if part == "gpu":
        gs, sg = steps(d), load(d, "smoke_cuda.json") or {}
        cuda, gu = arms(d, "smoke_cuda.json", e, cuda=True), reuse(d, e, ["cpu", "cuda"])
        xn = lines(d, "depot_new_files_xnode.txt") or []
        v.update(steps_gpu=gs, net_gpu=load(d, "net.json"), cuda=cuda, gpu_reuse=gu,
                 device={k: sg.get(k) for k in ("device_name", "device_uuid", "parent_uuid", "runtime_version",
                                                "driver_version", "total_memory_gb", "cutensor_functional",
                                                "load_cuda_s", "load_itensors_s")},
                 cross_node={"loadcheck": load(d, "loadcheck_xnode.json"),
                             "compiled_files_added": sum("/compiled/" in x for x in xn),
                             "artifact_files_added": sum("/artifacts/" in x for x in xn),
                             "cpu": arms(d, "smoke_cpu_xnode.json", e)})
        g.update(cuda_functional=bool(sg.get("cuda_functional")), cuda_dense=cuda["dense"]["pass"],
                 gpu_depot_tarball_readback=gs.get("tar_depot", {}).get("rc") == 0,
                 gpu_reuse_all_precompiled=gu["all_precompiled"], gpu_reuse_no_recompile=gu["no_recompile"],
                 gpu_reuse_smoke=gu["smoke_pass"])
        q = cuda["qn"]
        v["qn_on_cuda"] = ("works" if q["pass"] else "works only without noise"
                           if (q.get("without_noise") or {}).get("pass") else "fails")
        v["cuda_ready"] = g["depot_built"] and g["cuda_functional"] and g["cuda_dense"]
    v["gates"], v["pass"] = g, all(g.values())
    dump(os.path.join(d, "verdict.json"), v)
    print("JPROBE RESULT" if part == "gpu" else "JPROBE CPU RESULT",
          "PASS" if v["pass"] else "FAIL", json.dumps(g))
    if part == "gpu":
        print("JPROBE QN-ON-CUDA", v["qn_on_cuda"], json.dumps(v["cuda"]["qn"])[:600])
        print("JPROBE CUDA-READY", v["cuda_ready"], "cross-node:",
              json.dumps({k: v["cross_node"][k] for k in ("compiled_files_added", "artifact_files_added")}))
    if mk:
        print("JPROBE MPSKIT", mk.get("status"), mk.get("minus_tnkit"))
    return 0 if v["pass"] else 1


def dump(path, rec):
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["net", "fetch", "ref", "calib", "check"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--cpu-dir", help="check: the cpu part's directory (default --dir)")
    ap.add_argument("--part", choices=["cpu", "gpu"], default="gpu", help="check: which gates")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        return fetch(*a.args)
    if a.cmd == "check":
        return check(a.dir, a.cpu_dir or a.dir, a.part)
    return {"net": net, "ref": ref, "calib": calib}[a.cmd](a.out)


if __name__ == "__main__":
    sys.exit(main())
