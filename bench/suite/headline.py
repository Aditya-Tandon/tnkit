"""The measured cells behind BENCHMARKS.md, and its headline figure.

    python bench/suite/headline.py extract SUMMARY_2 SUMMARY_2U --out bench/suite/scored_cells.tsv
    python bench/suite/headline.py plot bench/suite/scored_cells.tsv --out docs/figures/benchmarks.png

`extract` reads the summaries that `analyse2.py` (suite_2.json) and `analyse.py` (suite_2u.json)
write and keeps one line per measured cell: time to each accuracy target, seconds per collision
step, and seconds per sweep. `plot` draws, per workload and hardware, tnkit's fastest time to the
middle target over the fastest other library's, with bounds from the repeat ranges.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COLS = ["spec", "measure", "workload", "hardware", "device", "lib", "sym", "threads", "eps",
        "status", "chi", "med_s", "min_s", "max_s"]
HW_2U = {"gpu": "gpu-1g", "cpu": "cpu4"}          # suite 2u ran one GPU share and one CPU share


def _cells(summary, spec):
    s = json.load(open(summary))
    hw = lambda h: HW_2U.get(h, h)
    out = []
    for t in s["time_to_accuracy"]:
        out.append(dict(spec=spec, measure="time_to_eps", workload=t["workload"],
                        hardware=hw(t["hardware"]), device=t["device"], lib=t["lib"],
                        sym=t["sym"], threads=t.get("threads", ""), eps=t["eps"],
                        status=t["status"], chi=t.get("chi") or "",
                        med_s=t.get("wall_med", ""), min_s=t.get("wall_min", ""),
                        max_s=t.get("wall_max", "")))
    for r in s["rows"]:
        if r["workload"] == "tf_collision" and r.get("complete"):
            out.append(dict(spec=spec, measure="s_per_step", workload=r["workload"],
                            hardware=hw(r["hardware"]), device=r["device"], lib=r["lib"],
                            sym=r["sym"], threads=r.get("threads", ""), eps="", status="ok",
                            chi=r["chi"], med_s=r["step_med"], min_s=r["step_min"],
                            max_s=r["step_max"]))
    for t in s.get("throughput", []):
        for chi, reps, med in zip(t["chis"], t["s_per_sweep"], t["s_per_sweep_med"]):
            out.append(dict(spec=spec, measure="s_per_sweep", workload=t["workload"],
                            hardware=t["hardware"], device=t["device"], lib=t["lib"],
                            sym=t["sym"], threads=t["threads"], eps="", status="ok", chi=chi,
                            med_s=med, min_s=min(reps), max_s=max(reps)))
    return out


def extract(summaries, path):
    rows = _cells(summaries[0], "suite_2.json") + _cells(summaries[1], "suite_2u.json")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def load(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        for k in ("med_s", "min_s", "max_s"):
            r[k] = float(r[k]) if r[k] else None
        r["eps"] = float(r["eps"]) if r["eps"] else None
    return rows


# (label, workload, hardware, middle target or None for the collision step)
ROWS = [
    ("GPU, 1g slice", [("Heisenberg chain, ground state", "xxz_dmrg", 1e-8),
                       ("Ising chain, 4 lowest levels", "tfim_excited", 1e-8),
                       ("XX quench, N = 32", "xx_quench", 1e-3)], "gpu-1g"),
    ("GPU, 3g slice", [("XX quench, N = 48", "xx_quench_large", 1e-3),
                       ("Schwinger collision step", "tf_collision", None),
                       ("Heisenberg 12 × 6 cylinder", "heis_cyl6", 3e-4)], "gpu-3g"),
    ("CPU, 4 cores", [("XX quench, N = 32", "xx_quench", 1e-3),
                      ("Ising chain, 4 lowest levels", "tfim_excited", 1e-8),
                      ("Heisenberg chain, ground state", "xxz_dmrg", 1e-8)], "cpu4"),
    ("CPU, 16 cores", [("Heisenberg chain, ground state", "xxz_dmrg", 1e-8),
                       ("Heisenberg 12 × 6 cylinder\n(tnkit χ 256, 4 threads;\nblock2 χ 128, 16 threads)",
                        "heis_cyl6", 3e-4)], "cpu16"),
]
NAME = {"yastn": "YASTN", "tenpy": "TeNPy", "qtea": "Quantum TEA", "itensor": "ITensor",
        "block2": "block2"}
SYM = {"dense": "dense", "u1": "U(1)", "su2": "SU(2)"}


def ratios(rows, wl, hw, eps):
    """tnkit's fastest cell over the fastest other library's, with repeat-range bounds; the same
    against the fastest library without SU(2) symmetry; and the names of both."""
    if eps is None:
        cs = [r for r in rows if r["spec"] == "suite_2.json" and r["measure"] == "s_per_step"
              and r["hardware"] == hw]
    else:
        cs = [r for r in rows if r["spec"] == "suite_2.json" and r["measure"] == "time_to_eps"
              and r["workload"] == wl and r["hardware"] == hw and r["eps"] == eps
              and r["status"] == "reached"]
    tk = min((r for r in cs if r["lib"] == "tnkit"), key=lambda r: r["med_s"])
    out = []
    for others in ([r for r in cs if r["lib"] != "tnkit"],
                   [r for r in cs if r["lib"] != "tnkit" and r["sym"] != "su2"]):
        if not others:
            out.append(None)
            continue
        o = min(others, key=lambda r: r["med_s"])
        out.append((tk["med_s"] / o["med_s"], tk["min_s"] / o["max_s"], tk["max_s"] / o["min_s"],
                    f"{NAME[o['lib']]} {SYM[o['sym']]}"))
    return out


def plot(path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    from vizstyle import CAT, INK, apply

    apply()
    rows = load(path)
    n = sum(len(g) for _, g, _ in ROWS)
    fig, ax = plt.subplots(figsize=(7.2, 4.9), constrained_layout=True)
    y, labels, seps = n - 1, [], []
    for gi, (group, items, hw) in enumerate(ROWS):
        for label, wl, eps in items:
            best, nosu2 = ratios(rows, wl, hw, eps)
            labels.append((y, label))
            if best is None:
                ax.text(1.15, y, "only tnkit finished", va="center",
                        ha="left", fontsize=7.5, color=INK["secondary"])
            else:
                r, lo, hi, who = best
                win, tie = hi < 1, lo <= 1 <= hi
                c = INK["muted"] if tie else CAT[0] if win else CAT[5]
                ax.errorbar(r, y, xerr=[[r - lo], [hi - r]], fmt="o", color=c, ms=5.5,
                            capsize=2, elinewidth=1.0, zorder=3)
                txt = (f"tie with {who} ({r:.2f}×)" if tie else
                       f"{1 / r:.3g}× faster than {who}" if win else f"{r:.3g}× slower than {who}")
                if nosu2 and nosu2[3] != who:
                    r2 = nosu2[0]
                    ax.plot(r2, y, "o", mfc=INK["surface"], mec=c, mew=1.2, ms=5.5, zorder=3)
                    txt += f"; {r2:.3g}× vs {nosu2[3]}"
                ax.text(1.02, y, txt, va="center", ha="left", fontsize=7.5,
                        transform=ax.get_yaxis_transform(), color=INK["primary"] if win else INK["secondary"])
            y -= 1
        if gi < len(ROWS) - 1:
            seps.append(y + 0.5)
    for s in seps:
        ax.axhline(s, color=INK["baseline"], lw=0.7, zorder=1)
    ax.axvline(1, color=INK["muted"], lw=0.9, ls="--", zorder=1)
    starts = [n - 1]
    for _, items, _ in ROWS[:-1]:
        starts.append(starts[-1] - len(items))
    for (group, _, _), y0 in zip(ROWS, starts):
        ax.text(0.055, y0 + 0.47, group, fontsize=8, color=INK["secondary"], style="italic",
                va="top")
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([t for _, t in labels], fontsize=8)
    ax.set_ylim(-0.6, n - 0.45)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 120)
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("tnkit time ÷ fastest other library  (below 1: tnkit faster)")
    ax.grid(axis="x", which="major", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("tnkit 0.2.0rc1 against five libraries, middle accuracy target", fontsize=10)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=400)
    plt.close(fig)
    from PIL import Image
    # a flat-colour plot: an adaptive palette keeps it under 300 KB at full resolution
    Image.open(buf).convert("RGB").quantize(64).save(out, optimize=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("summaries", nargs=2)
    e.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("cells")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        extract(a.summaries, a.out)
    else:
        plot(a.cells, a.out)


if __name__ == "__main__":
    main()
