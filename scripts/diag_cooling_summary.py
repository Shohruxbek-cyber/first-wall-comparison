"""Summary of the check with the coolant 3.5 mm behind the plasma-facing surface (revision of
20 Oct 2026): tungsten 1 mm on 2.5 mm of steel, lithium film on 3.5 mm of
steel, everything else as in the production runs.  Collects the stored outputs of
results/cooling_3p5mm/ and runs/li_scan/T600K_steel_thickness-0.0035/ and interpolates the minimum
lithium film (925 K, as in the production scan: linear in temperature, logarithmic in thickness)
for single shots and for the repeated-shot runs that exist.

Usage: python scripts/diag_cooling_summary.py
"""
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "results" / "cooling_3p5mm"
LIMIT = 925.0


def crossing(pairs, limit=LIMIT):
    pairs = sorted(pairs)
    for (d1, T1), (d2, T2) in zip(pairs, pairs[1:]):
        if T1 >= limit > T2:
            f = (T1 - limit) / (T1 - T2)
            return math.exp(math.log(d1) + f * math.log(d2 / d1))
    return None


if __name__ == "__main__":
    res = {"tungsten": {}, "lithium": {}}
    for f in sorted(glob.glob(str(C / "w_single" / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        res["tungsten"][f"{d['scenario']} single shot"] = {"peak_surface_K": d["peak_surface_T_K"],
                                                          "preshot2_surface_K": d.get("pre_shot2_surface_T_K")}
    for f in sorted(glob.glob(str(C / "w_multi_*" / "*.json"))):
        if f.endswith("code_version.json"):
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        res["tungsten"][f"{d['scenario']} repeated shots"] = {k: d.get(k) for k in (
            "n_shots", "converged", "last_preshot_K", "last_shot_peak_K", "max_interface_T_K", "last_shot_interface_T_K")}
    for scen in ("Prototype", "Demo"):
        scan = json.loads((ROOT / "runs" / "li_scan" / "T600K_steel_thickness-0.0035" / f"{scen}.json").read_text(encoding="utf-8"))
        single = [(r["delta_um"], r["peak_substrate_K"]) for r in scan["rows"] if r["status"] == "ok"]
        multi = []
        for f in sorted(glob.glob(str(C / f"li_multi_{scen}_d*" / f"{scen}.json"))):
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            multi.append({"delta_um": d["delta_um"], "n_shots": d["n_shots"], "converged": d["converged"],
                          "last_shot_peak_steel_K": d["last_shot_peak_steel_K"], "max_peak_steel_K": d["max_peak_steel_K"],
                          "last_preshot_steel_front_K": d["last_preshot_steel_front_K"]})
        conv = [(m["delta_um"], m["max_peak_steel_K"]) for m in multi if m["converged"]]
        res["lithium"][scen] = {"single_shot_min_film_um": crossing(single), "repeated_runs": multi,
                                "repeated_min_film_um": crossing(conv) if len(conv) >= 2 else None}
    (C / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
