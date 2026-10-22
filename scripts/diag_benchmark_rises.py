"""Benchmark ratios as ratios of temperature rises (revision of 21 Oct 2026): Tables 3 and E.2
compared the Blanchard-Martin and Raffray peaks as ratios of Celsius temperatures; this gives the ratios of the
rises above each calculation's initial temperature instead.

  * Blanchard and Martin (2005): initial temperature 400 C ("400 C was used as the initial condition for the
    thermal analyses"), the same in validation/bm2005.py; their finite-element peaks 2554 / 1393 / 849 C at the
    surface, 10 and 25 um, as stored with the benchmark (runs/validation/bm2005.json, targets_C).
  * Raffray (2005): coolant 500 C (Fig. 7 caption), the initial temperature of validation/raffray2005.py; his
    digitized 100 um trace starts at 501.5 C (data/raffray2005_fig7_DIGITIZED.json); his peaks read by eye from
    Fig. 7, ~2915 C at the surface and ~1525 C at 10 um.

Usage: python scripts/diag_benchmark_rises.py [--out=results/revision/benchmark_rises]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "benchmark_rises"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    bm = json.loads((ROOT / "runs" / "validation" / "bm2005.json").read_text(encoding="utf-8"))
    base = next(r for r in bm if r.get("label") == "BM props, Table-1 loads (baseline)")["per_shot_C"][0]
    tgt = next(r for r in bm if "targets_C" in r)["targets_C"]
    rf = json.loads((ROOT / "runs" / "validation" / "raffray2005.json").read_text(encoding="utf-8"))
    dig = json.loads((ROOT / "data" / "raffray2005_fig7_DIGITIZED.json").read_text(encoding="utf-8"))
    res = {"note": __doc__.split("Usage:")[0].strip(), "Blanchard_Martin": {}, "Raffray": {}}
    T0 = 400.0
    for key, lab in (("Ts", "surface"), ("T10", "10 um"), ("T25", "25 um")):
        ref, this = float(tgt[key]), float(base[key])
        res["Blanchard_Martin"][lab] = {"reference_C": ref, "this_work_C": this, "initial_C": T0,
                                       "ratio_C": this / ref, "ratio_K": (this + 273.15) / (ref + 273.15),
                                       "ratio_of_rises": (this - T0) / (ref - T0)}
    T0 = 500.0
    rfb, rft = rf["baseline"], rf["targets_fig7"]
    for key, lab in (("Ts_peak_C", "surface"), ("10um_C", "10 um")):
        ref, this = float(rft[key]), float(rfb[key])
        res["Raffray"][lab] = {"reference_C": ref, "this_work_C": this, "initial_C": T0,
                              "ratio_C": this / ref, "ratio_K": (this + 273.15) / (ref + 273.15),
                              "ratio_of_rises": (this - T0) / (ref - T0)}
    res["Raffray"]["digitized_100um_first_point_C"] = float(dig["100um"][1][0])
    (out / "benchmark_rises.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for w in ("Blanchard_Martin", "Raffray"):
        for lab, r in res[w].items():
            if isinstance(r, dict):
                print(w, lab, {k: round(v, 4) for k, v in r.items()})
    print("Raffray digitized 100 um trace starts at", res["Raffray"]["digitized_100um_first_point_C"], "C")


if __name__ == "__main__":
    main()
