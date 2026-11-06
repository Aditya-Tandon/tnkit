"""Score the runs: errors against references, validity, time to each accuracy, ratios.

    python bench/suite/analyse.py --spec bench/suite/suite.json --runs DIR/*/runs.jsonl \
        --out summary.json --table table.md

Rules (fixed before any data):
- A run's error: DMRG, |E - E_ref| / |E_ref| (exact free fermions for the Ising chain; for the
  Heisenberg chain the chi-512 reference of `confirmed_reference`: the lowest energy that a run
  of another library matches within CONFIRM_REL; with no such pair the reference is unconfirmed
  and no Heisenberg run is scored); TDVP, max_i |<S^z_i(T)> - exact|; collision,
  max_j |<n_j> - <n_j>(tnkit, same symmetry, repeat 1)|.
- A DMRG energy below the reference by more than `agree_rel` is a void run (not variational).
- Cells: one arm at one chi on one pod and device (MIG profile, or CPU model); runs of one arm on
  different pods or devices are never pooled. A cell is complete if every one of its runs is ok
  and repeats 1..R are all there (R: the repeats the pod planned for the workload, read from the
  scheduler's plan.json beside the runs file; without one, the highest base repeat recorded).
- Spread reruns (repeats above R) count only if every arm of that workload, chi, pod and device
  whose repeats 1..R are complete finished all of them; otherwise every arm there is scored on
  repeats 1..R, so that the arms of one comparison share their repeat count.
- Time to accuracy eps: the median wall time over repeats at the smallest chi whose error (worst
  repeat) is at most eps; the bars are the fastest and slowest repeat. No interpolation. Only
  complete cells count: climbing the ladder stops at the first incomplete cell, reported as
  "timeout" (if a run there timed out or was skipped after one) or "incomplete".
- Collision step time: mean of the steps after `exclude_first`, per repeat; median over repeats.
- Block against dense tnkit DMRG (Heisenberg workloads, including the `perf` ones): seconds per
  sweep (wall / sweeps) and wall time, medians over repeats, as dense / block (above 1: blocks
  faster); energy difference block - dense of the median energies, same chi and pod.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import load_spec  # noqa: E402
from reference import tfim_energy, xx_neel_sz  # noqa: E402

LIB_NAMES = {"tnkit": "tnkit", "yastn": "YASTN", "tenpy": "TeNPy", "qtea": "Quantum TEA"}
CONFIRM_REL = 1e-12          # two libraries' chi-512 energies agreeing this closely confirm it
OFF = 100                    # rep offset of the sampler-off check runs (schedule.SAMPLER_OFF)
TIMEOUT = {"timeout", "skipped_after_timeout"}
NOT_RUN = {"unsupported": {"unsupported", "skipped_unsupported"}, "not run: gate": {"skipped_gate"}}


def hardware(rec):
    return "gpu" if "cuda" in rec["backend"] else "cpu"


def device_of(rec):
    """The MIG profile (or GPU name) of a GPU run, the CPU model of a CPU run; None without an
    environment record (timeouts and skips)."""
    env = rec.get("env")
    if not env:
        return None
    if "cuda" in rec["backend"]:
        m = re.search(r"MIG (\d+g\.\d+gb)", str(env.get("nvidia_smi_L")))
        return m.group(1) if m else (env.get("gpu_name") or "gpu")
    return env.get("cpu_model") or "cpu"


def placer(recs, label=device_of):
    """A function record -> label(record), where a record without an environment takes the one
    label of its pod's other records (or "unknown" if they have several or none)."""
    seen = defaultdict(set)
    for r in recs:
        if r.get("env"):
            seen[r.get("pod")].add(label(r))
    pod = {p: s.pop() if len(s) == 1 else "unknown" for p, s in seen.items()}
    return lambda r: label(r) if r.get("env") else pod.get(r.get("pod"), "unknown")


def env_name(rec):
    return rec.get("env_name") or (rec.get("env") or {}).get("env_name") or "main"


def arm_of(rec):
    return (rec["lib"], rec["backend"], rec["sym"], env_name(rec))


def planned_reps(path):
    """{workload: the highest base repeat planned} from a scheduler's plan.json ({} if absent)."""
    try:
        with open(path) as f:
            order = json.load(f)["order"]
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for rep, wl, *_ in order:
        if rep < OFF:
            out[wl] = max(out.get(wl, 0), rep)
    return out


def load(paths):
    """The records of each runs file. One pod writes one file, so a record without `pod` takes
    the name of its file's directory; `planned_reps` comes from the plan.json beside it."""
    recs = []
    for p in paths:
        d = os.path.dirname(os.path.abspath(p))
        pod, planned = os.path.basename(d), planned_reps(os.path.join(d, "plan.json"))
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["pod"] = r.get("pod") or pod
                    if r.get("workload") in planned:
                        r["planned_reps"] = planned[r["workload"]]
                    recs.append(r)
    return recs


def confirmed_reference(pts, rel=CONFIRM_REL):
    """pts: (energy, lib, sym). The lowest energy that a run of a different library matches
    within `rel` (relative), or None (unconfirmed) if no two libraries agree."""
    ok = [e for e, lib, _ in pts
          if len({b for f, b, _ in pts if abs(f - e) <= rel * abs(e)} | {lib}) >= 2]
    e0 = min(ok, default=None)
    agree = [] if e0 is None else \
        sorted({f"{lib}/{sym}" for e, lib, sym in pts if abs(e - e0) <= rel * abs(e0)})
    return {"energy": e0, "confirmed": e0 is not None, "agreeing": agree, "confirm_rel": rel,
            "lowest": min(e for e, _, _ in pts)}


def references(spec, recs):
    out = {}
    for name, w in spec["workloads"].items():
        if w["kind"] == "dmrg" and w["model"] == "tfim":
            out[name] = {"energy": tfim_energy(w["N"], w["g"]), "source": "exact free fermions"}
        elif w["kind"] == "dmrg" and "reference" in w:
            ref = w["reference"]
            pts = [(r["energy"], r["lib"], r["sym"]) for r in recs
                   if r["workload"] == name and r["chi"] == ref["chi"] and r["status"] == "ok"
                   and r.get("rep", 1) < OFF]
            if not pts:
                out[name] = {"energy": None, "confirmed": False, "source": "missing"}
                continue
            c = confirmed_reference(pts)
            out[name] = {**c, "chi": ref["chi"], "all": pts, "source":
                         f"chi-{ref['chi']} energy, confirmed by two libraries" if c["confirmed"]
                         else f"UNCONFIRMED: no two libraries agree within {c['confirm_rel']:g}"}
    return out


def error_of(rec, w, refs, coll_ref):
    if w["kind"] == "dmrg":
        e0 = refs.get(rec["workload"], {}).get("energy")
        return None if e0 is None else (rec["energy"] - e0) / abs(e0)
    if w["kind"] == "tdvp":
        ex = xx_neel_sz(w["N"], w["J"], w["T"])
        return float(np.max(np.abs(np.asarray(rec["sz"]) - ex)))
    ref = coll_ref.get(rec["sym"])
    if ref is None:
        return None
    return float(np.max(np.abs(np.asarray(rec["n"]) - np.asarray(ref["n"]))))


def complete(rs, base):
    """Repeats 1..base all present and every run ok, except spread reruns (rep > base) that the
    pod's budget skipped."""
    rs = [r for r in rs if r["rep"] <= base or r["status"] != "skipped_budget"]
    return all(r["status"] == "ok" for r in rs) and \
        set(range(1, base + 1)) <= {r["rep"] for r in rs} and base >= 1


def cells_of(spec, recs, arm=None, share=None):
    """{(workload, hardware, pod, device, arm, chi): records} and {same key: base repeats}, the
    sampler-off runs left out. `share` labels the hardware (default: cpu or gpu)."""
    arm, hw, dev = arm or arm_of, placer(recs, share or hardware), placer(recs)
    cells = defaultdict(list)
    for r in recs:
        if r["workload"] in spec["workloads"] and r.get("rep", 1) < OFF:
            cells[(r["workload"], hw(r), r.get("pod"), dev(r), arm(r), r["chi"])].append(r)
    top = defaultdict(int)
    for k, rs in cells.items():
        planned = [r["planned_reps"] for r in rs if r.get("planned_reps")]
        top[k[:5]] = max([top[k[:5]]] + (planned or [r["rep"] for r in rs
                                                     if r["rep"] <= spec["repeats"]]))
    base = {k: top[k[:5]] for k in cells}
    groups = defaultdict(list)                   # the arms of one comparison
    for k in cells:
        groups[k[:4] + k[5:]].append(k)
    for ks in groups.values():
        reruns = {r["rep"] for k in ks for r in cells[k] if r["rep"] > base[k]}
        live = [k for k in ks if complete([r for r in cells[k] if r["rep"] <= base[k]], base[k])]
        if reruns and not all(reruns <= {r["rep"] for r in cells[k] if r["rep"] > base[k]
                                         and r["status"] != "skipped_budget"} for k in live):
            for k in ks:
                cells[k] = [r for r in cells[k] if r["rep"] <= base[k]]
    return cells, base


def score(spec, recs):
    refs = references(spec, recs)
    coll_ref = {r["sym"]: r for r in recs if r["kind"] == "collision" and r["status"] == "ok"
                and r["lib"] == "tnkit" and r["rep"] == 1}
    cells, base = cells_of(spec, recs)
    rows = []
    for key, rs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        wl, hw, pod, dev, arm, chi = key
        w = spec["workloads"][wl]
        ok = [r for r in rs if r["status"] == "ok"]
        row = {"workload": wl, "hardware": hw, "pod": pod, "device": dev, "lib": arm[0],
               "backend": arm[1], "sym": arm[2], "env": arm[3], "chi": chi, "n_ok": len(ok),
               "n_runs": len(rs), "complete": complete(rs, base[key]),
               "statuses": sorted({r["status"] for r in rs}),
               "errors": [r.get("error") for r in rs if r.get("error")][:2]}
        if ok:
            errs = [error_of(r, w, refs, coll_ref) for r in ok]
            walls = [r["wall_s"] for r in ok]
            row.update(err=None if any(e is None for e in errs) else float(np.max(np.abs(errs))),
                       err_signed=errs, wall=walls, wall_med=float(np.median(walls)),
                       wall_min=float(np.min(walls)), wall_max=float(np.max(walls)),
                       peak_rss_mb=max(r.get("peak_rss_mb", 0) for r in ok),
                       peak_gpu_mb=max(r.get("peak_gpu_mb", 0) or 0 for r in ok),
                       lib_version=ok[0].get("lib_version"), dtype=ok[0].get("dtype"),
                       numpy=(ok[0].get("env") or {}).get("packages", {}).get("numpy"))
            if w["kind"] == "dmrg":
                tol = w.get("reference", {}).get("agree_rel", 1e-12)
                row["void"] = any(e is not None and e < -tol for e in errs)
                row["sweeps"] = [r.get("sweeps") for r in ok]
                row["energies"] = [r["energy"] for r in ok]
            if w["kind"] == "tdvp":                  # S^z_total is conserved: 0 for the Neel state
                row["sz_total_dev"] = max(abs(float(np.sum(r["sz"]))) for r in ok)
            if w["kind"] == "collision":
                k = w["exclude_first"]
                per = [float(np.mean(r["step_s"][k:])) for r in ok]
                row.update(step_s=per, step_med=float(np.median(per)),
                           step_min=float(np.min(per)), step_max=float(np.max(per)),
                           bond_mean=[float(np.mean(r["bonds"])) for r in ok],
                           charge=[float(np.sum(r["n"])) for r in ok],
                           charge_dev=max(abs(float(np.sum(r["n"])) - w["N"]) for r in ok),
                           void=row["err"] is None or row["err"] > w["accuracy"]["eps"][0])
        rows.append(row)
    return refs, rows


ARM = ("hardware", "pod", "device", "lib", "backend", "sym", "env")


def climb(rs, eps):
    """(status, cell) for one ladder (cells sorted by chi): ("reached", the smallest complete
    cell whose worst repeat meets eps), else where the climb stopped: "error" (every run
    failed), "timeout" or "incomplete" at the first incomplete cell, "unconfirmed reference" at
    the first unscored one, or "not reached"."""
    for r in rs:
        if not r["complete"]:
            why = next((k for k, v in NOT_RUN.items() if set(r["statuses"]) <= v), None)
            if not why and set(r["statuses"]) <= {"error", "crash"}:
                why = "error"
            return why or ("timeout" if TIMEOUT & set(r["statuses"]) else "incomplete"), r
        if r.get("void"):
            continue
        if r.get("err") is None:
            return "unconfirmed reference", r
        if r["err"] <= eps:
            return "reached", r
    return "not reached", None


def time_to_accuracy(spec, rows, keys=ARM, ladder_only=True):
    """Per arm and eps (`climb`); `ladder_only` leaves out bond dimensions outside the
    workload's ladder (suite 1's chi-512 reference runs)."""
    out = []
    groups = defaultdict(list)
    for r in rows:
        w = spec["workloads"][r["workload"]]
        if w["kind"] != "collision" and not w.get("perf") and \
                (r["chi"] in w["chis"] or not ladder_only):
            groups[(r["workload"],) + tuple(r[k] for k in keys)].append(r)
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["chi"])
        for eps in spec["workloads"][key[0]]["accuracy"]["eps"]:
            status, at = climb(rs, eps)
            hit = at if status == "reached" else None
            out.append({"workload": key[0], **dict(zip(keys, key[1:])), "eps": eps,
                        "status": status, "reached": hit is not None,
                        "chi": at and at["chi"], "n_ok": at and at["n_ok"],
                        "n_runs": at and at["n_runs"],
                        **{k: hit and hit[k] for k in ("wall_med", "wall_min", "wall_max")},
                        "best_err": min((r["err"] for r in rs if r.get("err") is not None
                                         and r["complete"]), default=None)})
    return out


def tta_cell(t, fmt):
    """One time-to-accuracy table cell."""
    if t["reached"]:
        return (f"{fmt(t['wall_med'])} (χ {t['chi']}; n {t['n_ok']}; "
                f"{fmt(t['wall_min'])}–{fmt(t['wall_max'])})")
    if t["status"] in ("timeout", "incomplete"):
        return f"{t['status']} at χ {t['chi']} ({t['n_ok']} of {t['n_runs']} ok)"
    if t["status"] == "error":
        return f"error at χ {t['chi']} (all {t['n_runs']} runs)"
    if t["status"] == "unconfirmed reference":
        return "reference unconfirmed"
    if t["status"] in NOT_RUN:
        return t["status"]
    return "not reached" + (f" (best {t['best_err']:.1e})" if t["best_err"] is not None else "")


def collision_tests(rows):
    """Collision step ratios. T1: r_lib = t(tnkit, faster of dense and U(1)) / t(YASTN U(1));
    YASTN-AHEAD if r_lib >= 1.5, TNKIT-AHEAD if r_lib <= 1/1.5, else PARITY. T2: r_sym =
    t(tnkit dense) / t(tnkit U(1)); BLOCKS-PAY if r_sym >= 1.5. Void arms make a test VOID."""
    c = {(r["lib"], r["sym"]): r for r in rows
         if r["workload"] == "tf_collision" and r.get("complete") and not r.get("void")}
    out = {}
    tk = [c[k] for k in (("tnkit", "dense"), ("tnkit", "u1")) if k in c]
    if tk and ("yastn", "u1") in c:
        best = min(tk, key=lambda r: r["step_med"])
        r1 = best["step_med"] / c[("yastn", "u1")]["step_med"]
        out["T1"] = {"r_lib": r1, "tnkit_best": best["sym"],
                     "verdict": "YASTN-AHEAD" if r1 >= 1.5 else
                     ("TNKIT-AHEAD" if r1 <= 1 / 1.5 else "PARITY")}
    else:
        out["T1"] = {"verdict": "VOID", "why": "a needed arm is missing or failed its check"}
    if ("tnkit", "dense") in c and ("tnkit", "u1") in c:
        r2 = c[("tnkit", "dense")]["step_med"] / c[("tnkit", "u1")]["step_med"]
        out["T2"] = {"r_sym": r2, "verdict": "BLOCKS-PAY" if r2 >= 1.5 else "BLOCKS-DO-NOT-PAY"}
    else:
        out["T2"] = {"verdict": "VOID", "why": "a needed arm is missing or failed its check"}
    if ("yastn", "dense") in c and ("yastn", "u1") in c:
        out["yastn_r_sym"] = c[("yastn", "dense")]["step_med"] / c[("yastn", "u1")]["step_med"]
    return out


def block_vs_dense(spec, rows):
    """tnkit block against dense DMRG on the Heisenberg workloads, per hardware, N and chi."""
    cells = defaultdict(dict)
    for r in rows:
        w = spec["workloads"][r["workload"]]
        if r["lib"] == "tnkit" and w["kind"] == "dmrg" and w.get("model") == "xxz" \
                and r.get("complete"):
            cells[(r["hardware"], r["pod"], w["N"], r["chi"])][r["sym"]] = r
    out = []
    for (hw, pod, N, chi), d in sorted(cells.items(), key=str):
        if not {"u1", "dense"} <= set(d):
            continue
        sps = {k: float(np.median([t / n for t, n in zip(d[k]["wall"], d[k]["sweeps"])]))
               for k in ("u1", "dense")}
        out.append({"hardware": hw, "pod": pod, "N": N, "chi": chi,
                    "s_per_sweep_block": sps["u1"],
                    "s_per_sweep_dense": sps["dense"], "sweep_ratio": sps["dense"] / sps["u1"],
                    "wall_ratio": d["dense"]["wall_med"] / d["u1"]["wall_med"],
                    "sweeps_block": d["u1"]["sweeps"], "sweeps_dense": d["dense"]["sweeps"],
                    "dE": float(np.median(d["u1"]["energies"]) -
                                np.median(d["dense"]["energies"]))})
    return out


def _fmt_t(s):
    return "–" if s is None else (f"{s:.3g} s" if s < 100 else f"{s:.0f} s")


def table(spec, tta, rows, refs, bvd=()):
    lines = []
    for wl, w in spec["workloads"].items():
        if w.get("perf"):
            continue
        for share in sorted({tuple(r[k] for k in ARM[:3]) for r in rows
                             if r["workload"] == wl}, key=str):
            hw, pod, dev = share
            title = f"{wl}, {hw.upper()} (pod {pod}, {dev})"
            if w["kind"] == "collision":
                rs = [r for r in rows if r["workload"] == wl and
                      tuple(r[k] for k in ARM[:3]) == share]
                lines += [f"\n#### {title}\n",
                          "| library | tensors | s per step (median, min–max) | check | peak GPU MB |",
                          "|---|---|---|---|---|"]
                for r in sorted(rs, key=lambda r: (r["lib"], r["sym"])):
                    if not r.get("complete"):
                        lines.append(f"| {LIB_NAMES[r['lib']]} | {r['sym']} | "
                                     f"incomplete: {r['n_ok']} of {r['n_runs']} ok "
                                     f"({', '.join(r['statuses'])}) | | |")
                        continue
                    lines.append(f"| {LIB_NAMES[r['lib']]} {r['lib_version']} | {r['sym']} | "
                                 f"{r['step_med']:.1f} ({r['step_min']:.1f}–{r['step_max']:.1f}) | "
                                 f"{'void' if r['void'] else 'ok'} ({r['err']:.1e}) | "
                                 f"{r['peak_gpu_mb']:.0f} |")
                continue
            ts = [t for t in tta
                  if t["workload"] == wl and tuple(t[k] for k in ARM[:3]) == share]
            if not ts:
                continue
            eps = w["accuracy"]["eps"]
            lines += [f"\n#### {title}: time to accuracy\n",
                      "| library | tensors | " + " | ".join(f"to {e:.0e}" for e in eps) + " |",
                      "|---|---|" + "---|" * len(eps)]
            arms = sorted({(t["lib"], t["backend"], t["sym"], t["env"]) for t in ts})
            for lib, be, sym, env in arms:
                cells = []
                for e in eps:
                    t = next(x for x in ts if (x["lib"], x["backend"], x["sym"], x["env"], x["eps"])
                             == (lib, be, sym, env, e))
                    cells.append(tta_cell(t, _fmt_t))
                star = "*" if env != "main" else ""
                lines.append(f"| {LIB_NAMES[lib]}{star} | {sym} | " + " | ".join(cells) + " |")
    if bvd:
        lines += ["\n#### tnkit block against dense DMRG, Heisenberg chain\n",
                  "| hardware | pod | N | χ | s/sweep dense | s/sweep block | dense / block per sweep | "
                  "dense / block wall | sweeps dense, block | E block − dense |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for b in bvd:
            lines.append(f"| {b['hardware'].upper()} | {b['pod']} | {b['N']} | {b['chi']} | "
                         f"{_fmt_t(b['s_per_sweep_dense'])} | {_fmt_t(b['s_per_sweep_block'])} | "
                         f"{b['sweep_ratio']:.2f} | {b['wall_ratio']:.2f} | "
                         f"{b['sweeps_dense']}, {b['sweeps_block']} | {b['dE']:.1e} |")
    lines.append("\n\\* run in a different software environment (its own numpy pin); same hardware.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=None)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    recs = load(a.runs)
    refs, rows = score(spec, recs)
    tta = time_to_accuracy(spec, rows)
    tests = collision_tests(rows)
    bvd = block_vs_dense(spec, rows)
    summary = {"spec": spec["name"], "spec_sha256": spec["_sha256"], "references": refs,
               "rows": rows, "time_to_accuracy": tta, "tests": tests, "block_vs_dense": bvd,
               "n_records": len(recs)}
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    md = table(spec, tta, rows, refs, bvd)
    if a.table:
        with open(a.table, "w") as f:
            f.write(md + "\n")
    print(md)
    print(json.dumps({"references": {k: {kk: v[kk] for kk in v if kk != "all"}
                                     for k, v in refs.items()}, "tests": tests}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
