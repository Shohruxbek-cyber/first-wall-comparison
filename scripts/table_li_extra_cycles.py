"""Collect the fully recorded extra cycles of the repeated-shot lithium runs (scripts/li_extra_cycle.py) into one
table (revision of 21 Oct 2026; Table E.3 of the article): per scenario and film, the
pre-shot, peak and time-mean temperature of the steel front face, the time per shot above 925 K and 873 K, the
energy partition of the absorbed energy (latent and sensible heat of the vapor, enthalpy removed with the replaced
film, heat to the coolant) and the time-mean heat flux through the steel to the coolant; and (22 Oct 2026) the per-shot swing of the steel front face and the heat-flux factors of the energy entering the
steel (worst window; square-pulse equivalent of the swing), a pulsed-load indicator for which no steel design rule
was found.

Usage: python scripts/table_li_extra_cycles.py [--out=results/revision/li_extra_cycle]
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["scenario", "delta_um", "source_run", "shot_number", "preshot_front_K", "peak_front_K", "peak_steel_K",
        "time_mean_front_K", "us_above_925K", "us_above_873K", "vapor_latent_pct", "vapor_sensible_pct",
        "film_replacement_pct", "to_coolant_pct", "conducted_Li_to_steel_pct", "absorbed_J_m2",
        "q_coolant_MW_m2", "evaporated_kg_m2", "steel_swing_K", "F_steel_worst_window_MW_m2_s05",
        "F_steel_square_equiv_MW_m2_s05"]


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "li_extra_cycle"))
    out = out if out.is_absolute() else ROOT / out
    rows = []
    for f in sorted((ROOT / "results" / "revision" / "li_extra_cycle").glob("*/*_extra_cycle.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        fr = r["fractions_of_absorbed"]
        rows.append({"scenario": r["scenario"], "delta_um": round(r["delta_um"], 3), "source_run": r["source_run"],
                     "shot_number": r["shot_number"], "preshot_front_K": r["preshot_front_K"], "peak_front_K": r["peak_front_K"],
                     "peak_steel_K": r["peak_steel_K"], "time_mean_front_K": r["time_mean_front_K"],
                     "us_above_925K": r["time_above_K_s"]["925"] * 1e6, "us_above_873K": r["time_above_K_s"]["873"] * 1e6,
                     "vapor_latent_pct": 100 * fr["vapor_latent"], "vapor_sensible_pct": 100 * fr["vapor_sensible"],
                     "film_replacement_pct": 100 * fr["film_replacement"], "to_coolant_pct": 100 * fr["to_coolant"],
                     "conducted_Li_to_steel_pct": 100 * fr["conducted_Li_to_steel"],
                     "absorbed_J_m2": r["energy_per_shot_J_m2"]["absorbed"],
                     "q_coolant_MW_m2": r["time_mean_heat_flux_into_steel_W_m2"]["to_coolant"] / 1e6,
                     "evaporated_kg_m2": r["evaporated_kg_m2"],
                     "steel_swing_K": r.get("steel_face_pulse", {}).get("swing_K"),
                     "F_steel_worst_window_MW_m2_s05": r.get("steel_face_pulse", {}).get("F_worst_window_MW_m2_s05"),
                     "F_steel_square_equiv_MW_m2_s05": r.get("steel_face_pulse", {}).get("F_square_pulse_equivalent_MW_m2_s05")})
    rows.sort(key=lambda x: (x["scenario"], x["delta_um"]))
    with open(out / "table_extra_cycles.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    for x in rows:
        print(f"{x['scenario']:9s} {x['delta_um']:8.2f} um  pre {x['preshot_front_K']:6.1f}  peak front {x['peak_front_K']:6.1f}  "
              f"peak steel {x['peak_steel_K']:6.1f}  mean {x['time_mean_front_K']:6.1f}  >925 {x['us_above_925K']:6.1f} us  "
              f">873 {x['us_above_873K']:6.1f} us | lat {x['vapor_latent_pct']:4.1f} sen {x['vapor_sensible_pct']:4.1f} "
              f"film {x['film_replacement_pct']:4.1f} cool {x['to_coolant_pct']:4.1f} | q {x['q_coolant_MW_m2']:.3f} MW/m2 "
              f"| evap {x['evaporated_kg_m2']:.3e} | swing {x['steel_swing_K'] or float('nan'):.1f} K, "
              f"F {x['F_steel_worst_window_MW_m2_s05'] or float('nan'):.2f} / {x['F_steel_square_equiv_MW_m2_s05'] or float('nan'):.2f}"
              f"  ({x['source_run']})")


if __name__ == "__main__":
    main()
