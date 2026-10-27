"""Experimental-scenario helium counts for Table 4 (revision of 21 Oct 2026), from outputs that
already exist: the implanted helium per shot (runs/he_retention/results/sources.json, full retention) and the
500-shot retention-model run of Supplementary Section S7 (runs/he_retention/Experimental_baseline/totals.csv; each shot from
300 K, the Experimental scenario being single shots).  Shot counts interpolated linearly between shots, as for
Prototype and Demo.  Also the count for burn 4He alone and the helium per shot that 1e3 shots to 1e19 m^-2 imply,
for the comparison with Gonzalez-Arrabal et al. (2020).

Usage: python scripts/diag_experimental_helium.py [--out=results/revision/experimental_helium]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def crossing(shots, retained, level):
    i = int(np.argmax(retained >= level))
    if retained[i] < level:
        return None
    if i == 0:
        return float(level / retained[0])
    return float(shots[i - 1] + (level - retained[i - 1]) / (retained[i] - retained[i - 1]))


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "experimental_helium"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads((ROOT / "runs" / "he_retention" / "results" / "sources.json").read_text(encoding="utf-8"))["Experimental"]
    tot = np.genfromtxt(ROOT / "runs" / "he_retention" / "Experimental_baseline" / "totals.csv", delimiter=",", names=True)
    per_shot = src["He_implanted_total"]
    res = {"note": __doc__.split("Usage:")[0].strip(), "He_implanted_per_shot_m2": per_shot,
           "He_implanted_burn_per_shot_m2": src["He_implanted_burn"], "model_shots_run": int(tot["shot"][-1])}
    for lv in (1e18, 1e19):
        k = f"{lv:.0e}"
        res[f"full_retention_shots_to_{k}"] = lv / per_shot
        res[f"burn_He_only_shots_to_{k}"] = lv / src["He_implanted_burn"]
        res[f"model_shots_to_{k}"] = crossing(tot["shot"], tot["He_retained"], lv)
    res["He_per_shot_for_1e3_shots_to_1e19"] = 1e19 / 1e3
    # local helium content (mobile + trapped, appm) at the stored profiles, as in scripts/make_he_results.py
    z = np.load(ROOT / "runs" / "he_retention" / "Experimental_baseline" / "profiles.npz", allow_pickle=False)
    res["peak_local_He"] = {}
    for k in sorted({f.split("_")[0] for f in z.files}, key=int):
        tot = z[f"{k}_He_trapped"] + z[f"{k}_He"]
        res["peak_local_He"][k] = {"appm": float(tot.max()), "depth_um": float(z[f"{k}_x"][int(np.argmax(tot))] * 1e6)}
    (out / "experimental_helium.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
