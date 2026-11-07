"""Compare the two JSONs written by fused_dmrg_check.py (unfused, fused): per case the final
energies and |dE|/|E|, the largest per-sweep relative energy difference, and the per-sweep speed-up
(median sweep time, unfused over fused). Flags any final |dE|/|E| above 1e-10.

usage: python fused_dmrg_compare.py UNFUSED.json FUSED.json OUT.json
"""
import json, statistics, sys


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    assert not a["fused_matvec_in_library"] and b["fused_matvec_in_library"], "versions swapped"
    rows = []
    for x, y in zip(a["cases"], b["cases"]):
        assert (x["case"], x["chi"]) == (y["case"], y["chi"])
        dE = abs(x["E"] - y["E"]) / abs(x["E"])
        per = [abs(p - q) / abs(p) for p, q in zip(x["energies"], y["energies"])]
        su = statistics.median(x["sweep_s"]) / statistics.median(y["sweep_s"])
        rows.append(dict(case=x["case"], chi=x["chi"], E_unfused=x["E"], E_fused=y["E"],
                         E_per_site_unfused=x["E_per_site"], E_per_site_fused=y["E_per_site"],
                         rel_dE_final=dE, max_rel_dE_per_sweep=max(per), rel_dE_per_sweep=per,
                         energies_unfused=x["energies"], energies_fused=y["energies"],
                         median_sweep_s_unfused=statistics.median(x["sweep_s"]),
                         median_sweep_s_fused=statistics.median(y["sweep_s"]), sweep_speedup=su,
                         flag=dE > 1e-10))
        print(f"{x['case']:8s} chi {x['chi']:4d}  E/N {x['E_per_site']:.15f} / {y['E_per_site']:.15f}  "
              f"|dE|/|E| {dE:.2e}  max per-sweep {max(per):.2e}  sweep {statistics.median(x['sweep_s']):.2f} s "
              f"-> {statistics.median(y['sweep_s']):.2f} s ({su:.2f}x){'  FLAG > 1e-10' if dE > 1e-10 else ''}")
    json.dump(dict(unfused=a["label"], fused=b["label"], rows=rows,
                   any_flag=any(r["flag"] for r in rows)), open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
