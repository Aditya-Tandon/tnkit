"""Figures from `analyse.py`'s summary: wall time against delivered error, and collision steps.

    python bench/suite/plot.py summary.json --outdir figures/

One figure per DMRG/TDVP workload, panels CPU and GPU: each arm is a line through its bond
dimensions, wall time (median, bars fastest to slowest repeat) against its error; dashed
verticals mark the fixed accuracy targets. No fits. One bar figure for the collision step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vizstyle import CAT, INK, apply_default  # noqa: E402

COLOR = {"tnkit": CAT[0], "yastn": CAT[3], "tenpy": CAT[4], "qtea": CAT[7]}
NAME = {"tnkit": "tnkit", "yastn": "YASTN", "tenpy": "TeNPy", "qtea": "Quantum TEA"}
TITLE = {"tfim_dmrg": "Ising chain ground state (DMRG)",
         "xxz_dmrg": "Heisenberg chain ground state (DMRG)",
         "xx_quench": "XX chain after a Néel quench (TDVP)"}
XLABEL = {"dmrg": "relative energy error", "tdvp": "max error in ⟨Sᶻᵢ(T)⟩"}


def label(r):
    star = "*" if r["env"] != "main" else ""
    sym = "U(1)" if r["sym"] == "u1" else "dense"
    return f"{NAME[r['lib']]}{star}, {sym}"


def tradeoff(summary, wl, kind, eps, path):
    rows = [r for r in summary["rows"] if r["workload"] == wl and r.get("complete")
            and r.get("err") is not None and not r.get("void")]
    hws = [h for h in ("cpu", "gpu") if any(r["hardware"] == h for r in rows)]
    if not hws:
        return
    fig, axes = plt.subplots(1, len(hws), figsize=(7.5 * len(hws), 6.2), squeeze=False)
    for ax, hw in zip(axes[0], hws):
        arms = sorted({(r["lib"], r["sym"], r["env"]) for r in rows if r["hardware"] == hw})
        for lib, sym, env in arms:
            rs = sorted((r for r in rows if r["hardware"] == hw and
                         (r["lib"], r["sym"], r["env"]) == (lib, sym, env)), key=lambda r: r["chi"])
            x = [max(r["err"], 1e-16) for r in rs]
            y = [r["wall_med"] for r in rs]
            yerr = [[r["wall_med"] - r["wall_min"] for r in rs],
                    [r["wall_max"] - r["wall_med"] for r in rs]]
            ax.errorbar(x, y, yerr=yerr, color=COLOR[lib], ls="-" if sym == "dense" else "--",
                        marker="o" if env == "main" else "s", ms=6, capsize=3, lw=1.8,
                        label=label(rs[0]))
            for xi, yi, r in zip(x, y, rs):
                ax.annotate(f"{r['chi']}", (xi, yi), textcoords="offset points", xytext=(4, 4),
                            fontsize=9, color=INK["secondary"])
        for e in eps:
            ax.axvline(e, color=INK["muted"], ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(XLABEL[kind] + " (more accurate →)")
        ax.set_ylabel("wall time (s)")
        ax.set_title(hw.upper())
        ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(TITLE.get(wl, wl))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def collision(summary, path):
    rows = [r for r in summary["rows"] if r["workload"] == "tf_collision" and r.get("complete")]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["step_med"], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(rows) + 1.8))
    ys = range(len(rows))
    ax.barh(list(ys), [r["step_med"] for r in rows], color=[COLOR[r["lib"]] for r in rows],
            xerr=[[r["step_med"] - r["step_min"] for r in rows],
                  [r["step_max"] - r["step_med"] for r in rows]], capsize=4, height=0.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([label(r) + (" (void)" if r.get("void") else "") for r in rows])
    ax.set_xlabel("wall time per TDVP step (s)")
    ax.set_title("Two-flavour collision step, bond cap 256, one GPU slice")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--spec", default=os.path.join(HERE, "suite.json"))
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args(argv)
    apply_default()
    summary = json.load(open(a.summary))
    spec = json.load(open(a.spec))
    os.makedirs(a.outdir, exist_ok=True)
    for wl, w in spec["workloads"].items():
        if w["kind"] == "collision":
            collision(summary, os.path.join(a.outdir, f"bench-{wl}.png"))
        else:
            tradeoff(summary, wl, w["kind"], w["accuracy"]["eps"],
                     os.path.join(a.outdir, f"bench-{wl}.png"))


if __name__ == "__main__":
    main()
