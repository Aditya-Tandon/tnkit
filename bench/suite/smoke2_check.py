"""Check a smoke run of suite 2 against exact answers (run where the smoke ran; standard library and numpy).

    python3 bench/suite/smoke2_check.py --spec bench/suite/suite_2_smoke.json --pod POD DIR

Every run must end ok, unsupported, or skipped_gate (an arm whose environment did not pass its
own check). Where the smoke's bond dimension is exact, the result must match exact
diagonalisation or free fermions: DMRG energies to 1e-8 relative, excited levels to 1e-7 of
|E_0|, the quench to 1e-5 in <S^z_i>; any DMRG energy must not lie below the exact one. GPU runs
must see a GPU, and the memory sampler's records are summarised. Exit status 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import bonds, couplings, load_spec  # noqa: E402
from reference import tfim_energy, tfim_levels, xx_neel_sz, xxz_energy  # noqa: E402


def exact_chi(rec, N):
    """The bond dimension above which an MPS of N spins 1/2 is exact (SU(2): multiplets)."""
    return comb(N // 2, N // 4) if rec["sym"] == "su2" else 2 ** (N // 2)


def check(rec, w):
    """None if the run is right, else why not."""
    k = w["kind"]
    exact = rec["chi"] >= exact_chi(rec, w["N"])
    if k == "dmrg":
        e0 = tfim_energy(w["N"], w["g"]) if w["model"] == "tfim" else \
            xxz_energy(w["N"], *couplings(w), pairs=bonds(w))
        d = (rec["energy"] - e0) / abs(e0)
        if d < -1e-10:
            return f"energy below exact by {-d:.1e} (relative)"
        if exact and not w.get("fixed_sweeps") and d > 1e-8:
            return f"energy {d:.1e} (relative) above exact at an exact bond dimension"
    elif k == "excited":
        ex = tfim_levels(w["N"], w["g"], w["nstates"])
        d = max(abs(a - b) for a, b in zip(rec["energies_k"], ex)) / abs(ex[0])
        if exact and d > 1e-7:
            return f"levels off by {d:.1e} of |E_0| at an exact bond dimension"
    elif k == "tdvp":
        d = float(np.max(np.abs(np.asarray(rec["sz"]) - xx_neel_sz(w["N"], w["J"], w["T"]))))
        if exact and d > 1e-5:
            return f"<S^z_i> off by {d:.1e} at an exact cap"
    elif k == "collision":
        d = abs(float(np.sum(rec["n"])) - w["N"])
        if d > 1e-5:
            return f"total charge off by {d:.1e}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--pod", required=True)
    ap.add_argument("dir")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    gpu = spec["pods"][a.pod]["hardware"] == "gpu"
    recs = [json.loads(line) for line in open(os.path.join(a.dir, "runs.jsonl"))]
    bad, gated = [], set()
    for r in recs:
        tag = f"{r['lib']}/{r['backend']}/{r['sym']} {r['workload']} chi {r['chi']} rep {r['rep']}"
        if r["status"] == "skipped_gate":
            gated.add(f"{r['lib']}/{r['backend']}/{r['sym']}")
            continue
        if r["status"] == "unsupported":
            continue
        if r["status"] != "ok":
            bad.append(f"{tag}: {r['status']} {str(r.get('error', ''))[:300]}")
            continue
        why = check(r, spec["workloads"][r["workload"]])
        if why:
            bad.append(f"{tag}: {why}")
        if gpu and "GPU" not in str((r.get("env") or {}).get("nvidia_smi_L")):
            bad.append(f"{tag}: no GPU seen")
    mem = os.path.join(a.dir, "memory.jsonl")
    if os.path.exists(mem):
        ms = [json.loads(line) for line in open(mem)]
        gpu_mb = [m["peak_gpu_used_mb"] for m in ms if m.get("peak_gpu_used_mb")]
        print(f"memory sampler: {len(ms)} runs; GPU memory sampled in {len(gpu_mb)}; "
              f"peak tree RSS max {max((m['peak_tree_rss_mb'] for m in ms), default=0):.0f} MB")
        if gpu and not gpu_mb:
            print("WARNING: no GPU memory samples (NVML unusable here); torch counters remain")
    print(f"smoke: {len(recs)} runs; gated off: {sorted(gated) or 'none'}; bad: {len(bad)}")
    for b in bad:
        print("  BAD", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
