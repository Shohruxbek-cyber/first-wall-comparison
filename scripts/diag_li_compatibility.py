"""Lithium film thickness at which the peak steel temperature reaches the lithium/ferritic-steel
compatibility limit of about 600 C (873 K) given by Raffray (2005, p. 181) instead of the 925 K
structural limit (revision of 20 Oct 2026).  Same interpolation as
the production minimum film (linear in temperature, logarithmic in thickness, between the
computed thicknesses), on the stored single-shot scan (runs/summary/summary.json) and the
repeated-shot runs (runs/li_multi/summary.json); also lists the pre-shot steel front-face
temperatures of the repeated-shot runs, the sustained part of the interface temperature.

Usage: python scripts/diag_li_compatibility.py [--out=results/revision/li_compatibility]
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMITS = (873.0, 925.0)


def crossing(rows, key, limit):
    rows = sorted((r for r in rows if r.get(key) is not None), key=lambda r: r["delta_um"])
    for a, b in zip(rows, rows[1:]):
        Ta, Tb = a[key], b[key]
        if Ta >= limit > Tb:
            f = (Ta - limit) / (Ta - Tb)
            return math.exp(math.log(a["delta_um"]) + f * math.log(b["delta_um"] / a["delta_um"]))
    return None


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "li_compatibility"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    single = json.loads((ROOT / "runs" / "summary" / "summary.json").read_text(encoding="utf-8"))["Li"]
    multi = json.loads((ROOT / "runs" / "li_multi" / "summary.json").read_text(encoding="utf-8"))
    res = {}
    for scen in ("Prototype", "Demo"):
        srows = [r for r in single[scen]["rows"] if r["status"] == "ok"]
        mrows = multi[scen]["rows"]
        res[scen] = {"single_shot_min_film_um": {f"{L:g} K": crossing(srows, "peak_substrate_K", L) for L in LIMITS},
                     "repeated_shot_min_film_um": {f"{L:g} K": crossing(mrows, "peak_K", L) for L in LIMITS},
                     "repeated_shot_preshot_front_K": {f"{r['delta_um']:.3g}": r["last_preshot_front_K"] for r in mrows},
                     "repeated_shot_peak_K": {f"{r['delta_um']:.3g}": r["peak_K"] for r in mrows}}
    (out / "li_compatibility.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
