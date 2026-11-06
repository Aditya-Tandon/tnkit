"""Figures of suite 2 from `analyse2.py`'s summary (style: vizstyle, dpi 400).

    python bench/suite/plot2.py summary.json --spec bench/suite/suite_2.json --outdir figures/

- bench2-<workload>.png: wall time against delivered error, one panel per hardware share, one
  line per arm through its bond dimensions, bars from fastest to slowest repeat, the fixed
  targets dotted.
- bench2-throughput.png: seconds per sweep against chi (log-log) per GPU share, bars over
  repeats, the power-law fit with its 68% bootstrap band where the rule allows one;
  beside it the device memory peak against chi with the 12 and 47 GB slice limits.
- bench2-threads.png: t(4 threads) / t(16 threads) per arm on the 16-core pod.
- bench2-glance.png: tnkit's time to the middle target over the best other library's, per
  workload and hardware share (below 1: tnkit faster).
- bench2-collision.png: the collision step on the 3g slice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vizstyle import CAT, INK, STATUS, apply_default, grid_y  # noqa: E402

COLOR = {"tnkit": CAT[0], "yastn": CAT[3], "tenpy": CAT[4], "qtea": CAT[7], "itensor": CAT[5],
         "block2": CAT[2]}
NAME = {"tnkit": "tnkit", "yastn": "YASTN", "tenpy": "TeNPy", "qtea": "Quantum TEA",
        "itensor": "ITensor", "block2": "block2"}
LS = {"dense": "-", "u1": "--", "su2": ":"}
SYM = {"dense": "dense", "u1": "U(1)", "su2": "SU(2)"}
HW = {"gpu-1g": "GPU, MIG 1g.12gb", "gpu-3g": "GPU, MIG 3g.47gb", "gpu-full": "GPU, full H100",
      "cpu4": "CPU, 4 cores", "cpu16": "CPU, 16 cores", "cpu": "CPU"}
XLABEL = {"dmrg": "relative energy error", "excited": "max level error / $|E_0|$",
          "tdvp": r"max error in $\langle S^z_i\rangle$"}


def label(r):
    star = "*" if r["env"] == "qtea" else ""
    th = f", {r['threads']} thr" if r.get("threads") == 4 and r.get("hardware") == "cpu16" else ""
    return f"{NAME[r['lib']]}{star}, {SYM[r['sym']]}{th}"


def tradeoff(summary, wl, kind, eps, title, path):
    rows = [r for r in summary["rows"] if r["workload"] == wl and r.get("complete")
            and r.get("err") is not None and not r.get("void")]
    hws = [h for h in HW if any(r["hardware"] == h for r in rows)]
    if not hws:
        return
    fig, axes = plt.subplots(1, len(hws), figsize=(8 * len(hws), 6.5), squeeze=False)
    for ax, hw in zip(axes[0], hws):
        arms = sorted({(r["lib"], r["sym"], r["env"], r["threads"]) for r in rows
                       if r["hardware"] == hw}, key=str)
        for arm in arms:
            rs = sorted((r for r in rows if r["hardware"] == hw and
                         (r["lib"], r["sym"], r["env"], r["threads"]) == arm),
                        key=lambda r: r["chi"])
            x = [max(abs(r["err"]), 1e-16) for r in rs]
            y = [r["wall_med"] for r in rs]
            ax.errorbar(x, y, yerr=[[r["wall_med"] - r["wall_min"] for r in rs],
                                    [r["wall_max"] - r["wall_med"] for r in rs]],
                        color=COLOR[arm[0]], ls=LS[arm[1]], marker="o", capsize=3,
                        label=label(rs[0]))
            for xi, yi, r in zip(x, y, rs):
                ax.annotate(f"{r['chi']}", (xi, yi), textcoords="offset points", xytext=(4, 4),
                            fontsize=11, color=INK["secondary"])
        for e in eps:
            ax.axvline(e, color=INK["muted"], ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(XLABEL[kind] + "  (more accurate →)")
        ax.set_ylabel("wall time (s)")
        ax.set_title(HW[hw])
        grid_y(ax)
        ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def throughput(summary, path):
    thr = [t for t in summary["throughput"] if t["hardware"].startswith("gpu")]
    hws = [h for h in HW if any(t["hardware"] == h for t in thr)]
    if not hws:
        return
    fig, axes = plt.subplots(1, len(hws) + 1, figsize=(8 * (len(hws) + 1), 6.5), squeeze=False)
    for ax, hw in zip(axes[0], hws):
        for t in (t for t in thr if t["hardware"] == hw):
            c, med = np.array(t["chis"]), np.array(t["s_per_sweep_med"])
            lo = med - np.array([min(x) for x in t["s_per_sweep"]])
            hi = np.array([max(x) for x in t["s_per_sweep"]]) - med
            f = t["fit"] or {}
            lab = label(t) + (f", p = {f['p']:.2f} ± {f['sigma_p']:.2f}" if f.get("fitted") else "")
            ax.errorbar(c, med, yerr=[lo, hi], color=COLOR[t["lib"]], ls="none", marker="o",
                        capsize=3, label=lab)
            if f.get("fitted"):
                ax.plot(f["band_chi"], f["line"], color=COLOR[t["lib"]], ls=LS[t["sym"]], lw=2)
                ax.fill_between(f["band_chi"], f["band_lo"], f["band_hi"],
                                color=COLOR[t["lib"]], alpha=0.18, lw=0)
            else:
                ax.plot(c, med, color=COLOR[t["lib"]], ls=LS[t["sym"]], lw=1, alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_ylabel("seconds per sweep")
        ax.set_title(HW[hw])
        grid_y(ax)
        ax.legend(fontsize=11, loc="upper left")
    ax = axes[0][-1]
    for t in thr:
        m = [(c, x) for c, x in zip(t["chis"], t["peak_device_mb"]) if x]
        if m:
            ax.plot([c for c, _ in m], [x / 1024 for _, x in m], color=COLOR[t["lib"]],
                    ls=LS[t["sym"]], marker="o" if t["hardware"] == "gpu-3g" else "s",
                    label=f"{label(t)}, {t['hardware'][4:]}")
    for gb, name in ((12, "1g.12gb"), (47, "3g.47gb")):
        ax.axhline(gb, color=INK["muted"], ls=":", lw=1.5)
        ax.text(0.99, gb, name, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                color=INK["secondary"], fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel("device memory peak (GB)")
    ax.set_title("Memory on the slice")
    grid_y(ax)
    ax.legend(fontsize=10, loc="lower right")
    fig.suptitle("Heisenberg chain, N = 64: cost per sweep")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def threads(summary, path):
    tr = summary["threads"]
    if not tr:
        return
    tr = sorted(tr, key=lambda x: (x["workload"], x["lib"], x["sym"]))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(tr) + 3), 6.5))
    xs = np.arange(len(tr))
    ax.bar(xs, [x["t4_over_t16"] for x in tr], color=[COLOR[x["lib"]] for x in tr], width=0.65)
    ax.axhline(4.0, color=INK["muted"], ls=":", lw=1.5)
    ax.text(xs[-1] + 0.4, 4.0, " ideal", va="center", color=INK["secondary"], fontsize=12)
    ax.axhline(1.0, color=INK["baseline"], lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{NAME[x['lib']]} {SYM[x['sym']]}\n{x['workload']} χ {x['chi']}"
                        for x in tr], rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("t(4 threads) / t(16 threads)")
    ax.set_title("Speed-up from 4 to 16 cores (16-core pod)")
    grid_y(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def glance(summary, spec, path):
    """tnkit's best arm against the best other arm, time to the middle target."""
    tta = summary["time_to_accuracy"]
    wls = [w for w, x in spec["workloads"].items() if len(x["accuracy"]["eps"]) == 3]
    hws = [h for h in HW if any(t["hardware"] == h for t in tta)]
    if not wls or not hws:
        return
    V = np.full((len(wls), len(hws)), np.nan)
    for i, wl in enumerate(wls):
        mid = spec["workloads"][wl]["accuracy"]["eps"][1]
        for j, hw in enumerate(hws):
            ts = [t for t in tta if (t["workload"], t["hardware"], t["eps"]) == (wl, hw, mid)
                  and t["reached"]]
            tk = [t["wall_med"] for t in ts if t["lib"] == "tnkit"]
            ot = [t["wall_med"] for t in ts if t["lib"] != "tnkit"]
            if tk and ot:
                V[i, j] = min(tk) / min(ot)
    fig, ax = plt.subplots(figsize=(2.6 * len(hws) + 3, 1.0 * len(wls) + 2))
    for i in range(len(wls)):
        for j in range(len(hws)):
            v = V[i, j]
            c = INK["grid"] if np.isnan(v) else (STATUS["good"] if v < 1 else STATUS["critical"])
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.92, color=c,
                                       alpha=1.0 if np.isnan(v) else 0.35))
            ax.text(j + 0.48, i + 0.46, "–" if np.isnan(v) else f"{v:.2g}", ha="center",
                    va="center", color=INK["primary"], fontsize=15)
    ax.set_xlim(0, len(hws))
    ax.set_ylim(len(wls), 0)
    ax.set_xticks(np.arange(len(hws)) + 0.48)
    ax.set_xticklabels([HW[h] for h in hws], fontsize=12)
    ax.set_yticks(np.arange(len(wls)) + 0.46)
    ax.set_yticklabels(wls, fontsize=12)
    ax.set_title("tnkit time / best other library, middle target (below 1: tnkit faster)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--spec", default=os.path.join(HERE, "suite_2.json"))
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args(argv)
    apply_default()
    summary = json.load(open(a.summary))
    spec = json.load(open(a.spec))
    os.makedirs(a.outdir, exist_ok=True)
    titles = {"xxz_dmrg": "Heisenberg chain ground state (DMRG)",
              "xx_quench": "XX chain after a Néel quench (TDVP)",
              "heis_cyl6": "Heisenberg cylinder 12 × 6 ground state (DMRG)",
              "xx_quench_large": "XX chain N = 48 after a Néel quench, T = 12 (TDVP)",
              "tfim_excited": "Ising chain N = 48, four lowest levels (DMRG)"}
    for wl, w in spec["workloads"].items():
        if w["kind"] != "collision" and w["accuracy"]["eps"]:
            tradeoff(summary, wl, w["kind"], w["accuracy"]["eps"], titles.get(wl, wl),
                     os.path.join(a.outdir, f"bench2-{wl}.png"))
    throughput(summary, os.path.join(a.outdir, "bench2-throughput.png"))
    threads(summary, os.path.join(a.outdir, "bench2-threads.png"))
    glance(summary, spec, os.path.join(a.outdir, "bench2-glance.png"))
    from plot import collision
    coll = {"rows": [r for r in summary["rows"] if r.get("hardware") == "gpu-3g"]}
    collision(coll, os.path.join(a.outdir, "bench2-collision.png"))


if __name__ == "__main__":
    main()
