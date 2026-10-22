"""Partial film renewal in Demo (revision of 22 Oct 2026): the minimum films and the
evaporation against the renewed fraction of the film per shot.

Reads the repeated-shot runs with partial renewal, results/revision/li_renewal/Demo_d<film>um_r<fraction>/Demo.json
(scripts/run_li_multi.py renew=<fraction>: the retained lithium keeps its heat, the film temperature is carried
forward), and, as the fully renewed case, the production runs runs/li_multi/Demo_d<film>um/Demo.json.  Computed:
fractions 0.03 and 0.3 at 31.6, 56.2 and 100 um (chosen on 22 Oct 2026); intermediate fractions and
Prototype are not computed.  The minimum films for the 925 K and 873 K limits are interpolated as in Section 2.7
(linearly in temperature, logarithmically in thickness) between the computed films; a limit not crossed inside
31.6-100 um is reported as below or above that range.

Usage: python scripts/li_renewal_summary.py [--out=results/revision/li_renewal]
"""
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILMS = ("31.6", "56.2", "100")
FRACTIONS = ("0.03", "0.3")
LIMITS = (925.0, 873.0)


def row(f):
    d = json.loads(f.read_text(encoding="utf-8"))
    return {"delta_um": d["delta_um"], "n_shots": d["n_shots"], "converged": d["converged"], "status": d["status"],
            "peak_steel_K": d["last_shot_peak_steel_K"], "preshot_front_K": d["last_preshot_steel_front_K"],
            "swing_K": d["last_shot_peak_steel_K"] - d["last_preshot_steel_front_K"],
            "evaporated_kg_m2_per_shot": d["last_shot_vaporized_kg_m2"],
            "film_T_after_renewal_K": d.get("last_film_T_after_renewal_K"),
            "energy_rel_residual": d["energy_rel_residual"], "source": str(f.relative_to(ROOT)).replace("\\", "/")}


def minimum(rows, limit):
    rows = sorted(rows, key=lambda r: r["delta_um"])
    T = [r["peak_steel_K"] for r in rows]
    if all(t <= limit for t in T):
        return {"film_um": None, "note": f"below {rows[0]['delta_um']:.3g} um (every computed film passes)"}
    if all(t > limit for t in T):
        return {"film_um": None, "note": f"above {rows[-1]['delta_um']:.3g} um (no computed film passes)"}
    for a, b in zip(rows[:-1], rows[1:]):
        if a["peak_steel_K"] > limit >= b["peak_steel_K"]:
            w = (a["peak_steel_K"] - limit) / (a["peak_steel_K"] - b["peak_steel_K"])
            return {"film_um": math.exp(math.log(a["delta_um"]) + w * (math.log(b["delta_um"]) - math.log(a["delta_um"]))),
                    "between_um": [a["delta_um"], b["delta_um"]]}
    return {"film_um": None, "note": "not monotonic"}


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "li_renewal"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cases = {"1 (full renewal, production)": [row(ROOT / "runs" / "li_multi" / f"Demo_d{d}um" / "Demo.json") for d in FILMS]}
    for r in FRACTIONS:
        files = [ROOT / "results" / "revision" / "li_renewal" / f"Demo_d{d}um_r{r}" / "Demo.json" for d in FILMS]
        cases[r] = [row(f) for f in files if f.exists()]
    res = {"note": __doc__.split("Usage:")[0].strip(), "cases": {}}
    for r, rows in cases.items():
        res["cases"][r] = {"runs": rows, "minimum_film": {f"{lv:.0f}K": minimum(rows, lv) for lv in LIMITS} if len(rows) == 3
                           else "incomplete"}
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["renewed_fraction", "delta_um", "n_shots", "converged", "peak_steel_K", "preshot_front_K", "swing_K",
                    "evaporated_kg_m2_per_shot", "film_T_after_renewal_K"])
        for r, c in res["cases"].items():
            for x in c["runs"]:
                w.writerow([r, f"{x['delta_um']:.4g}", x["n_shots"], x["converged"], f"{x['peak_steel_K']:.2f}",
                            f"{x['preshot_front_K']:.2f}", f"{x['swing_K']:.2f}", f"{x['evaporated_kg_m2_per_shot']:.4e}",
                            "" if x["film_T_after_renewal_K"] is None else f"{x['film_T_after_renewal_K']:.2f}"])
    for r, c in res["cases"].items():
        print(r, c["minimum_film"])
        for x in c["runs"]:
            print(f"   {x['delta_um']:7.2f} um  {x['n_shots']:4d} shots conv={x['converged']}  peak {x['peak_steel_K']:.1f} K  "
                  f"pre {x['preshot_front_K']:.1f} K  evap {x['evaporated_kg_m2_per_shot']:.3e}  film {x['film_T_after_renewal_K']}")


if __name__ == "__main__":
    main()
