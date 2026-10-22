"""Shots to the helium thresholds for given helium-model run folders (revision of 20 Oct 2026),
with the same interpolation as scripts/make_he_results.py (ife_wall.he.analysis.shots_to).

Usage: python scripts/he_case_thresholds.py runs/he_retention/Demo_dpa_path [more folders]
Writes <folder>/thresholds.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.he import analysis as an  # noqa: E402

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        d = Path(arg) if Path(arg).is_absolute() else ROOT / arg
        t = np.genfromtxt(d / "totals.csv", delimiter=",", names=True)
        res = {"shots_run": int(t["shot"][-1]),
               "retained_fraction_last": float(t["He_retained"][-1] / t["He_in"][-1]),
               "shots_to": {n: an.shots_to(v, t["shot"], t["He_retained"]) for n, v in an.THRESHOLDS.items()}}
        (d / "thresholds.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(d.name, json.dumps({k: (v["value"], v["kind"]) for k, v in res["shots_to"].items()}, default=float))
