"""Erosion lifetime of the 1 mm tungsten armor in Demo from stored outputs (revision of 20 Oct 2026).

Combines the physical sputtering per shot (Eckstein fits, without and with the proxies for the
carbon and gold ions; runs/sputtering/sputtering.json) with the evaporation per shot of the last
shot of the 600-shot run (difference of the last two cumulative values in
runs/w_multi/baseline/Demo_per_shot.csv), and gives the evaporation-only loss of the two
550-shot runs with a degraded conductivity (runs/w_damage/multi/).  Full-power years use the
shots per year of the sputtering script.  No solver run is involved; melt motion and splashing
are not modelled, so a melting surface may lose more.  Also (22 Oct 2026) the
shots and full-power years until only the 100 um of tungsten that Garoz et al. (2016) found
sufficient to protect the steel is left, i.e. until 0.9 mm has eroded.

Usage: python scripts/diag_erosion_lifetime.py [--out=results/revision/erosion]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NM_PER_MM = 1e6
GAROZ_SUFFICIENT_NM = 1e5  # 100 um, Garoz et al. (2016)

if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "erosion"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((ROOT / "runs" / "sputtering" / "sputtering.json").read_text(encoding="utf-8"))["scenarios"]["Demo"]
    shots_per_fpy = sp["shots_to_erode_1mm_sourced"] / sp["full_power_years_to_erode_1mm_sourced"]
    r = np.loadtxt(ROOT / "runs" / "w_multi" / "baseline" / "Demo_per_shot.csv", delimiter=",", skiprows=1)
    evap = float(r[-1, 5] - r[-2, 5])  # nm evaporated in shot 600 (cumulative column)
    s0 = sp["sourced_nm_per_shot"]
    p_lo, p_hi = sp["with_proxies_nm_per_shot_range"]
    rates = {"sputtering (H and He isotopes)": s0,
             "sputtering with C and Au proxies, low": p_lo,
             "sputtering with C and Au proxies, high": p_hi,
             "sputtering + evaporation": s0 + evap,
             "sputtering with proxies (high) + evaporation": p_hi + evap}
    runs_n = {}
    for label, sub in (("conductivity 0.5", "kirr0.5_conv"), ("conductivity 0.5 with a 5 um skin at 0.25", "Heskin_conv")):
        dm = json.loads((ROOT / "runs" / "w_damage" / "multi" / sub / "Demo.json").read_text(encoding="utf-8"))
        last = dm["last_shot_ablated_nm"]
        rates[f"{label}: evaporation, last shot of the run"] = last
        rates[f"{label}: evaporation + sputtering"] = last + s0
        runs_n[label] = dm["n_shots"]
    res = {"shots_per_full_power_year": shots_per_fpy, "evaporation_nm_per_shot_last_shot": evap,
           "last_shot_of_the_600_shot_run": int(r[-1, 0]), "damaged_runs_n_shots": runs_n,
           "cases": {k: {"nm_per_shot": v, "shots_to_1mm": NM_PER_MM / v,
                         "full_power_years": NM_PER_MM / v / shots_per_fpy,
                         "shots_to_100um_left": (NM_PER_MM - GAROZ_SUFFICIENT_NM) / v,
                         "full_power_years_to_100um_left": (NM_PER_MM - GAROZ_SUFFICIENT_NM) / v / shots_per_fpy}
                     for k, v in rates.items()}}
    (out / "erosion_lifetime.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
