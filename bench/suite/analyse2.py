"""Score suite 2: errors against references, validity, time to each accuracy, throughput,
scaling fits, memory, thread scaling and the sampler's cost.

    python bench/suite/analyse2.py --spec bench/suite/suite_2.json --runs DIR/*/runs.jsonl \
        [--extra-ref SUITE1/cpu-ref/runs.jsonl] --out summary.json --table table.md

Rules (fixed before any data):
- Hardware is the pod's share: gpu-1g or gpu-3g (from the MIG profile every GPU run records),
  cpu4 or cpu16 (from the cgroup quota); an arm is library, backend, tensors, environment and
  threads. Cells are also keyed by pod and device (MIG profile or CPU model), so runs on
  different pods or CPU models are never pooled. Runs at rep >= 100 (the sampler-off checks) are
  left out of everything but the sampler's cost. A cell counts only if complete (every run ok,
  repeats 1..R all there), as in `analyse.py`.
- Errors. DMRG on the chain: |E - E_ref| / |E_ref|, E_ref the chi-512 reference (xxz_ref, or
  xxz_dmrg in an extra file) confirmed by two libraries within 1e-12 relative
  (`analyse.confirmed_reference`; unconfirmed: no chain run is scored); a run more than 1e-10
  below E_ref is void. Cylinder: E_ref = E_inf of a straight-line fit of the
  block2 SU(2) reference runs' last-sweep energy against their last-sweep discarded weight,
  sigma = |E_inf - the fit without the smallest bond dimension|; accepted if sigma / |E_inf| <=
  3e-5 and no run lies below E_inf - 2 sigma; a run below E_inf - 3 sigma is void. If it is not
  accepted, errors are taken against the lowest energy of any run and the table says so (a
  variational bound, not a reference). Excited: max_k |E_k - exact| / |E_0|. Quench:
  max_i |<S^z_i(T)> - exact|. Collision: as suite 1.
- Throughput (`perf` workloads with fixed sweeps): seconds per sweep = mean of sweeps 2-3 of each
  repeat (sweep times summing to more than 1.5x the wall time are read as cumulative; wall /
  sweeps where a library reports no sweep times, flagged), median over repeats;
  valid only if the energy after the sweeps is at most 1e-6 (relative) above E_ref and not 1e-10
  below it.
- Time to accuracy: as suite 1 (median wall over repeats at the smallest complete chi whose worst
  repeat meets eps; no interpolation; the climb stops at a timeout or incomplete cell), per
  hardware, pod, device and arm.
- Scaling: log t = log a + p log chi over the top three valid chi points, fitted only if t rises at
  least 4x per doubling of chi at each step there; sigma_p and a 68% band by bootstrap over
  repeats (resampling one repeat per point, 2000 draws, seed 0).
- Threads: t(4 threads) / t(16 threads) per arm and cell on the 16-core pod (medians).
- Memory: per cell, the largest peak over repeats of the process tree's resident memory and of
  the device's memory in use (both sampled from outside the run), beside the libraries' own
  counters.
- Sampler cost: per checked cell and arm, median wall with the sampler on / off.
- Cross-pod offset of the tnkit dense control (`--control-runs`): per workload and hardware, the
  median over shared bond dimensions of (median wall here) / (median wall there), against only
  runs that stopped on the same rule. Here the control's DMRG stops on rtol 1e-10, so its offset
  is taken against an earlier rtol arm (the tuned-settings runs), not
  against suite 1's control, which stopped on an absolute 1e-10 and ran more sweeps.
- Labels: block2 SU(2) bond dimensions count multiplets (2S+1 states each). TeNPy's stop
  (`max_E_err`) is relative to max(|E|, 1); on the excited-state row it is that much looser than
  the other libraries' absolute e_tol, and the table marks it.
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

from analyse import (CONFIRM_REL, OFF, cells_of, collision_tests, complete,  # noqa: E402
                     confirmed_reference, load, time_to_accuracy, tta_cell)
from common import load_spec, sweep_time  # noqa: E402
from reference import extrapolate, tfim_levels, xx_neel_sz  # noqa: E402

NAMES = {"tnkit": "tnkit", "yastn": "YASTN", "tenpy": "TeNPy", "qtea": "Quantum TEA",
         "block2": "block2", "itensor": "ITensor"}


def hardware(r):
    env = r.get("env") or {}
    if "cuda" in r["backend"]:
        m = re.search(r"MIG (\dg)\.\d+gb", str(env.get("nvidia_smi_L")))
        return f"gpu-{m.group(1)}" if m else "gpu-full"
    q = env.get("cgroup_cpus")
    return f"cpu{int(round(q))}" if q else "cpu"


def arm_of(r):
    env = r.get("env_name") or (r.get("env") or {}).get("env_name") or "main"
    return (r["lib"], r["backend"], r["sym"], env, r.get("threads"))


def chain_reference(spec, recs, workloads=("xxz_ref", "xxz_dmrg"), chi=512, agree=CONFIRM_REL):
    pts = [(r["energy"], r["lib"], r["sym"]) for r in recs if r["workload"] in workloads
           and r["chi"] == chi and r["status"] == "ok" and r.get("rep", 1) < OFF]
    if not pts:
        return {"energy": None, "confirmed": False, "source": "missing"}
    c = confirmed_reference(pts, agree)
    return {**c, "points": pts, "source": f"chi-{chi} energy of {list(workloads)}, confirmed by "
            "two libraries" if c["confirmed"] else f"UNCONFIRMED: no two libraries agree "
            f"within {agree:g}"}


def cylinder_reference(w, recs):
    ref = w["reference"]
    pts = [r for r in recs if r["workload"] == ref["workload"] and r["status"] == "ok"
           and r.get("discarded_weights")]
    out = {"points": [(r["chi"], r["discarded_weights"][-1], r["energy"]) for r in pts]}
    if len(pts) >= 3:
        e_inf, sigma = extrapolate([r["discarded_weights"][-1] for r in pts],
                                   [r["energy"] for r in pts])
        out.update(energy=e_inf, sigma=sigma, sigma_rel=sigma / abs(e_inf))
        below = [r["energy"] for r in recs if r["workload"] == "heis_cyl6" and
                 r["status"] == "ok" and r["energy"] < e_inf - 2 * sigma]
        out["runs_below_2sigma"] = len(below)
        out["accepted"] = bool(out["sigma_rel"] <= ref["sigma_rel_max"] and not below)
    else:
        out["accepted"] = False
    if not out["accepted"]:
        cyl = [r["energy"] for r in recs if r["workload"] == "heis_cyl6" and r["status"] == "ok"]
        out.update(energy=min(cyl + [r["energy"] for r in pts], default=None), sigma=0.0,
                   source="NOT ACCEPTED: errors against the lowest energy of any run "
                          "(a variational bound)")
    else:
        out["source"] = "block2 SU(2), extrapolated in the discarded weight"
    return out


def references(spec, recs):
    refs = {}
    for name, w in spec["workloads"].items():
        if w["kind"] == "dmrg" and w["model"] == "xxz":
            refs[name] = chain_reference(spec, recs)
        elif w["kind"] == "dmrg" and w["model"] == "heis_cyl" and "reference" in w:
            refs[name] = cylinder_reference(w, recs)
        elif w["kind"] == "excited":
            refs[name] = {"levels": tfim_levels(w["N"], w["g"], w["nstates"]),
                          "source": "exact free fermions"}
    return refs


def error_of(r, w, refs, coll_ref):
    k = w["kind"]
    if k == "dmrg":
        e0 = refs.get(r["workload"], {}).get("energy")
        return None if e0 is None else (r["energy"] - e0) / abs(e0)
    if k == "excited":
        ex = refs[r["workload"]]["levels"]
        return max(abs(a - b) for a, b in zip(r["energies_k"], ex)) / abs(ex[0])
    if k == "tdvp":
        return float(np.max(np.abs(np.asarray(r["sz"]) - xx_neel_sz(w["N"], w["J"], w["T"]))))
    ref = coll_ref.get(r["sym"])
    return None if ref is None else float(np.max(np.abs(np.asarray(r["n"]) - ref["n"])))


def void_of(r, w, refs, err):
    if err is None or w["kind"] != "dmrg":
        return False
    ref = refs.get(r["workload"], {})
    if w["model"] == "heis_cyl" and ref.get("accepted"):
        return r["energy"] < ref["energy"] - 3 * ref["sigma"]
    if w.get("fixed_sweeps"):
        v = w.get("validity", {})
        return err < -v.get("below_rel", 1e-10) or err > v.get("above_rel", 1e-6)
    return err < -1e-10


def memory_index(paths):
    idx = {}
    for p in paths:
        m = os.path.join(os.path.dirname(p), "memory.jsonl")
        if os.path.exists(m):
            for x in load([m]):
                idx[(x["workload"], x["chi"], x["rep"], x["lib"], x["backend"], x["sym"],
                     x["env_name"], x["threads"])] = x
    return idx


def score(spec, recs, mem, refs):
    coll_ref = {r["sym"]: r for r in recs if r["kind"] == "collision" and r["status"] == "ok"
                and r["lib"] == "tnkit" and r["rep"] == 1}
    cells, base = cells_of(spec, recs, arm=arm_of, share=hardware)
    rows = []
    for key, rs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        wl, hw, pod, dev, arm, chi = key
        w = spec["workloads"][wl]
        ok = [r for r in rs if r["status"] == "ok"]
        row = {"workload": wl, "hardware": hw, "pod": pod, "device": dev, "lib": arm[0],
               "backend": arm[1], "sym": arm[2], "env": arm[3], "threads": arm[4], "chi": chi,
               "n_ok": len(ok), "n_runs": len(rs), "complete": complete(rs, base[key]),
               "statuses": sorted({r["status"] for r in rs}),
               "errors": [str(r.get("error"))[:300] for r in rs if r.get("error")][:2]}
        if ok:
            errs = [error_of(r, w, refs, coll_ref) for r in ok]
            walls = [r["wall_s"] for r in ok]
            m = [mem.get((wl, chi, r["rep"], r["lib"], r["backend"], r["sym"], arm[3],
                          r.get("threads"))) or {} for r in ok]
            row.update(err=None if any(e is None for e in errs) else float(np.max(np.abs(errs))),
                       err_signed=errs, wall=walls, wall_med=float(np.median(walls)),
                       wall_min=float(np.min(walls)), wall_max=float(np.max(walls)),
                       void=any(void_of(r, w, refs, e) for r, e in zip(ok, errs)),
                       peak_rss_mb=max(max(r.get("peak_rss_mb") or 0,
                                           r.get("peak_rss_mb_julia") or 0) for r in ok),
                       peak_gpu_mb_lib=max(r.get("peak_gpu_mb") or 0 for r in ok),
                       peak_tree_rss_mb=max((x["peak_tree_rss_mb"] for x in m
                                             if x.get("peak_tree_rss_mb")), default=None),
                       peak_device_mb=max((x["peak_gpu_used_mb"] for x in m
                                           if x.get("peak_gpu_used_mb")), default=None),
                       lib_version=ok[0].get("lib_version"),
                       settings_sha256=ok[0].get("settings_sha256"))
            if w["kind"] == "dmrg":
                row["sweeps"] = [r.get("sweeps") for r in ok]
                row["energies"] = [r["energy"] for r in ok]
                st = [sweep_time(r) for r in ok]
                row["s_per_sweep"] = [t for t, _ in st]
                row["s_per_sweep_med"] = float(np.median(row["s_per_sweep"]))
                row["s_per_sweep_whole_run"] = any(f for _, f in st)
            if w["kind"] == "tdvp":
                row["sz_total_dev"] = max(abs(float(np.sum(r["sz"]))) for r in ok)
            if w["kind"] == "collision":
                k = w["exclude_first"]
                per = [float(np.mean(r["step_s"][k:])) for r in ok]
                row.update(step_s=per, step_med=float(np.median(per)),
                           step_min=float(np.min(per)), step_max=float(np.max(per)),
                           charge_dev=max(abs(float(np.sum(r["n"])) - w["N"]) for r in ok),
                           void=row["err"] is None or row["err"] > w["accuracy"]["eps"][0])
        rows.append(row)
    return refs, rows


KEYS = ("hardware", "pod", "device", "lib", "backend", "sym", "env", "threads")


def scaling_fit(chis, reps, draws=2000, seed=0):
    """p of t = a chi^p over the top three chi points if t rises >= 4x per doubling there."""
    if len(chis) < 3:
        return None
    c, t = np.asarray(chis[-3:], float), [np.asarray(x, float) for x in reps[-3:]]
    med = np.array([np.median(x) for x in t])
    steps = np.log2(c[1:] / c[:-1])
    if np.any(med[1:] / med[:-1] < 4.0 ** steps):
        return {"fitted": False, "why": "not cost-dominated by chi (< 4x per doubling)"}
    p = np.polyfit(np.log(c), np.log(med), 1)[0]
    rng = np.random.default_rng(seed)
    boot = [np.polyfit(np.log(c), np.log([rng.choice(x) for x in t]), 1) for _ in range(draws)]
    ps = np.array([b[0] for b in boot])
    grid = np.geomspace(c[0], c[-1], 40)
    curves = np.array([np.exp(b[1]) * grid ** b[0] for b in boot])
    return {"fitted": True, "p": float(p), "sigma_p": float(np.std(ps)), "chis": c.tolist(),
            "band_chi": grid.tolist(), "band_lo": np.percentile(curves, 16, 0).tolist(),
            "band_hi": np.percentile(curves, 84, 0).tolist(),
            "line": (np.exp(np.polyfit(np.log(c), np.log(med), 1)[1]) * grid ** p).tolist()}


def throughput(spec, rows):
    out = []
    groups = defaultdict(list)
    for r in rows:
        w = spec["workloads"][r["workload"]]
        if w.get("fixed_sweeps") and r.get("complete") and r.get("err") is not None \
                and not r.get("void"):                   # validity needs the reference
            groups[(r["workload"],) + tuple(r[k] for k in KEYS)].append(r)
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["chi"])
        out.append({"workload": key[0], **dict(zip(KEYS, key[1:])),
                    "chis": [r["chi"] for r in rs],
                    "s_per_sweep": [r["s_per_sweep"] for r in rs],
                    "s_per_sweep_med": [r["s_per_sweep_med"] for r in rs],
                    "whole_run": any(r["s_per_sweep_whole_run"] for r in rs),
                    "peak_device_mb": [r.get("peak_device_mb") for r in rs],
                    "peak_gpu_mb_lib": [r.get("peak_gpu_mb_lib") for r in rs],
                    "fit": scaling_fit([r["chi"] for r in rs], [r["s_per_sweep"] for r in rs])})
    return out


def thread_ratios(rows):
    cells = defaultdict(dict)
    for r in rows:
        if r["hardware"] == "cpu16" and r.get("complete"):
            cells[(r["workload"], r["chi"], r["lib"], r["sym"], r["env"], r["pod"],
                   r["device"])][r["threads"]] = r
    return [{"workload": k[0], "chi": k[1], "lib": k[2], "sym": k[3], "env": k[4],
             "pod": k[5], "device": k[6],
             "t4": d[4]["wall_med"], "t16": d[16]["wall_med"],
             "t4_over_t16": d[4]["wall_med"] / d[16]["wall_med"]}
            for k, d in sorted(cells.items(), key=str) if 4 in d and 16 in d]


def sampler_cost(spec, recs):
    d = defaultdict(lambda: ([], []))
    for r in recs:
        if r["status"] == "ok":
            d[(r["workload"], r["chi"], arm_of(r)[:4], r["rep"] % OFF)][int(r["rep"] >= OFF)] \
                .append(r["wall_s"])
    out = defaultdict(lambda: ([], []))
    for (wl, chi, arm, _), (on, off) in d.items():
        if on and off:
            out[(wl, chi, arm)][0].extend(on)
            out[(wl, chi, arm)][1].extend(off)
    return [{"workload": k[0], "chi": k[1], "arm": "/".join(k[2]),
             "on_over_off": float(np.median(v[0]) / np.median(v[1])), "n": len(v[1])}
            for k, v in out.items()]


YES_NO = {True: "yes", False: "no", None: "–"}


def control_offset(recs, ctrl, workloads=("xxz_dmrg", "xx_quench")):
    """Offsets of the tnkit dense control against `ctrl` records with the same stop (rtol)."""
    def walls(rs):
        d = defaultdict(list)
        for r in rs:
            if r["lib"] == "tnkit" and r["sym"] == "dense" and r["workload"] in workloads \
                    and r["status"] == "ok" and r.get("rep", 1) < OFF:
                hw = re.sub(r"^cpu\d*$", "cpu", hardware(r))
                rtol = (r.get("settings") or {}).get("rtol", 0.0) if r["kind"] == "dmrg" else None
                d[(r["workload"], hw, rtol, r["chi"])].append(r["wall_s"])
        return d
    here, there = walls(recs), walls(ctrl)
    out = []
    for wl, hw, rtol in sorted({k[:3] for k in here}, key=str):
        chis = sorted(k[3] for k in here if k[:3] == (wl, hw, rtol) and k in there)
        ratios = [float(np.median(here[(wl, hw, rtol, c)]) / np.median(there[(wl, hw, rtol, c)]))
                  for c in chis]
        off = float(np.median(ratios)) if ratios else None
        out.append({"workload": wl, "hardware": hw, "rtol": rtol, "chis": chis, "ratios": ratios,
                    "offset": off,
                    "within_0.8_1.25": None if off is None else 0.8 <= off <= 1.25})
    return out


def _t(s):
    return "–" if s is None else (f"{s:.3g} s" if s < 100 else f"{s:.0f} s")


def _name(lib, sym, kind):
    return (NAMES[lib] + ("*" if lib == "qtea" else "") +
            ("†" if lib == "tenpy" and kind == "excited" else ""),
            "su2 (χ: multiplets)" if sym == "su2" else sym)


def table(spec, tta, rows, refs, thr, tr, sc, off=()):
    L = []
    for wl, w in spec["workloads"].items():
        ts = [t for t in tta if t["workload"] == wl]
        for share in sorted({tuple(t[k] for k in KEYS[:3]) for t in ts}, key=str):
            eps = w["accuracy"]["eps"]
            L += [f"\n#### {wl}, {share[0]} (pod {share[1]}, {share[2]}): time to accuracy\n",
                  "| library | tensors | threads | " + " | ".join(f"to {e:.0e}" for e in eps) + " |",
                  "|---|---|---|" + "---|" * len(eps)]
            mine = [t for t in ts if tuple(t[k] for k in KEYS[:3]) == share]
            arms = sorted({(t["lib"], t["sym"], t["env"], t["threads"]) for t in mine}, key=str)
            for lib, sym, env, th in arms:
                cells = []
                for e in eps:
                    t = next(x for x in mine if (x["lib"], x["sym"], x["env"], x["threads"],
                                                 x["eps"]) == (lib, sym, env, th, e))
                    cells.append(tta_cell(t, _t))
                name, tensors = _name(lib, sym, w["kind"])
                L.append(f"| {name} | {tensors} | {th} | " + " | ".join(cells) + " |")
    if thr:
        L += ["\n#### Throughput: seconds per sweep (median; sweeps 2–3)\n",
              "| workload | hardware | library | tensors | χ: s/sweep | device peak MB | p ± σ |",
              "|---|---|---|---|---|---|---|"]
        for t in thr:
            f = t["fit"] or {}
            fit = f"{f['p']:.2f} ± {f['sigma_p']:.2f}" if f.get("fitted") else \
                (f.get("why") or "fewer than 3 chi points")
            name, tensors = _name(t["lib"], t["sym"], "dmrg")
            L.append(f"| {t['workload']} | {t['hardware']} | {name} | {tensors} | " +
                     ", ".join(f"{c}: {_t(s)}" for c, s in zip(t["chis"], t["s_per_sweep_med"])) +
                     (" (whole run / sweeps)" if t["whole_run"] else "") + " | " +
                     ", ".join("–" if m is None else f"{m:.0f}" for m in t["peak_device_mb"]) +
                     f" | {fit} |")
    coll = [r for r in rows if spec["workloads"][r["workload"]]["kind"] == "collision"]
    if coll:
        L += ["\n#### Collision step\n", "| hardware | library | tensors | s per step (min–max) | check |",
              "|---|---|---|---|---|"]
        for r in coll:
            if r.get("complete"):
                L.append(f"| {r['hardware']} | {NAMES[r['lib']]} | {r['sym']} | {r['step_med']:.1f} "
                         f"({r['step_min']:.1f}–{r['step_max']:.1f}) | "
                         f"{'void' if r['void'] else 'ok'} ({r['err']:.1e}) |")
            else:
                L.append(f"| {r['hardware']} | {NAMES[r['lib']]} | {r['sym']} | incomplete: "
                         f"{r['n_ok']} of {r['n_runs']} ok ({', '.join(r['statuses'])}) | |")
    if tr:
        L += ["\n#### Threads on the 16-core pod: t(4) / t(16)\n", "| workload | χ | library | tensors | t(4) | t(16) | ratio |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {x['workload']} | {x['chi']} | {NAMES[x['lib']]} | {x['sym']} | {_t(x['t4'])} | "
              f"{_t(x['t16'])} | {x['t4_over_t16']:.2f} |" for x in tr]
    if sc:
        L += ["\n#### Memory sampler: wall time on / off\n", "| workload | χ | arm | on / off | n |",
              "|---|---|---|---|---|"]
        L += [f"| {x['workload']} | {x['chi']} | {x['arm']} | {x['on_over_off']:.3f} | {x['n']} |"
              for x in sc]
    if off:
        L += ["\n#### Cross-pod offset of the tnkit dense control (same stop rule only)\n",
              "| workload | hardware | rtol | χ shared | offset (median ratio) | "
              "within 0.8–1.25 |",
              "|---|---|---|---|---|---|"]
        L += [f"| {x['workload']} | {x['hardware']} | {x['rtol']} | {x['chis'] or '–'} | "
              f"{'–' if x['offset'] is None else format(x['offset'], '.2f')} | "
              f"{YES_NO[x['within_0.8_1.25']]} |" for x in off]
    L.append("\nReferences: " + json.dumps({k: {kk: v[kk] for kk in v if kk not in
                                                 ("points", "levels")}
                                             for k, v in refs.items()}, default=float))
    L.append("\n\\* Quantum TEA runs in its own software environment (its own numpy pin); same hardware.")
    L.append("\n† TeNPy's stop (max_E_err) is relative to max(|E|, 1), so on the excited-state "
             "row it is looser than the other libraries' absolute e_tol by about |E_0|.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--extra-ref", nargs="*", default=[],
                    help="runs used only for the chain reference (suite 1's cpu-ref)")
    ap.add_argument("--control-runs", nargs="*", default=[],
                    help="earlier runs for the tnkit control's cross-pod offset (same stop only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=None)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    recs = load(a.runs)
    refs = references(spec, recs)
    extra = [r for r in load(a.extra_ref) if r["workload"] == "xxz_dmrg"]
    if extra:                                    # the chain reference pools suite 1's cpu-ref
        pooled = chain_reference(spec, recs + extra)
        for name, w in spec["workloads"].items():
            if w["kind"] == "dmrg" and w["model"] == "xxz":
                refs[name] = pooled
    refs, rows = score(spec, recs, memory_index(a.runs), refs)
    tta = time_to_accuracy(spec, rows, KEYS, ladder_only=False)
    thr = throughput(spec, rows)
    tr = thread_ratios(rows)
    sc = sampler_cost(spec, recs)
    off = control_offset(recs, load(a.control_runs)) if a.control_runs else []
    tests = collision_tests([dict(r, workload="tf_collision") for r in rows
                             if r["workload"] == "tf_collision" and r["hardware"] == "gpu-3g"])
    summary = {"spec": spec["name"], "spec_sha256": spec["_sha256"], "references": refs,
               "rows": rows, "time_to_accuracy": tta, "throughput": thr, "threads": tr,
               "sampler": sc, "control_offset": off, "tests_3g": tests, "n_records": len(recs)}
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1, default=float)
    md = table(spec, tta, rows, refs, thr, tr, sc, off)
    if a.table:
        with open(a.table, "w") as f:
            f.write(md + "\n")
    print(md)
    print(json.dumps({"tests_3g": tests}, indent=1, default=float))


if __name__ == "__main__":
    main()
