"""Drift left at the end of the repeated Demo tungsten runs (revision of 21 Oct 2026, final number check).

The repeated runs stop at the quasi-steady criterion (pre-shot surface rising by less than 0.05 K per
shot) or at a fixed shot count, and the article quotes their last shot.  As for the interface of the
600-shot Demo run in Section 4.2 (fit to shots 201-600, 938 K), T(n) = T_inf - A exp(-(n - n0)/tau) is
fitted to the last two thirds of the shots of each run (of its neutron-heated shots, for the runs
resumed with neutron heating after shot 600) for the pre-shot surface, the peak surface and the peak
interface temperature; T_inf minus the last shot is the drift left at the end of the run.
Runs no solver.  Usage: python scripts/diag_quasi_steady_drift.py [--out=results/revision/quasi_steady_drift]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
RUNS = {  # label: (per-shot file, shots before the segment that is fitted)
    "Demo, production (600 shots)": ("runs/w_multi/baseline/Demo_per_shot.csv", 0),
    "Demo, neutron heating, steel factor 0.5": ("results/neutron/w_multi_Demo_steel0.5/Demo_per_shot.csv", 600),
    "Demo, neutron heating, steel factor 1": ("results/neutron/w_multi_Demo_steel1/Demo_per_shot.csv", 600),
    "Demo, coolant 3.5 mm behind the surface": ("results/cooling_3p5mm/w_multi_Demo/Demo_per_shot.csv", 0),
}
COLUMNS = ("preshot_surface_T_K", "peak_surface_T_K", "peak_interface_T_K")


def asymptote(n, T):
    """Fit T_inf - A exp(-(n - n[0])/tau); returns T_inf, A, tau."""
    x = n - n[0]
    p0 = (T[-1] + 1.0, max(T[-1] - T[0], 1e-3) + 1.0, max(x[-1] / 3.0, 1.0))
    p, _ = curve_fit(lambda x, Ti, A, tau: Ti - A * np.exp(-x / tau), x, T, p0=p0, maxfev=20000)
    return float(p[0]), float(p[1]), float(p[2])


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "quasi_steady_drift"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for label, (path, before) in RUNS.items():
        d = np.genfromtxt(ROOT / path, delimiter=",", names=True)
        n = d["shot"]
        last = int(n[-1])
        start = before + (last - before) // 3  # fit shots start+1 ... last
        sel = n > start
        r = {"file": path, "last_shot": last, "fitted_shots": [start + 1, last]}
        for c in COLUMNS:
            Ti, A, tau = asymptote(n[sel], d[c][sel])
            r[c] = {"last_shot": float(d[c][-1]), "asymptote": Ti, "drift_left": Ti - float(d[c][-1]), "tau_shots": tau}
        res[label] = r
        print(label, "| fitted shots", r["fitted_shots"])
        for c in COLUMNS:
            q = r[c]
            print(f"   {c}: last {q['last_shot']:.2f} K, asymptote {q['asymptote']:.2f} K, drift left {q['drift_left']:+.2f} K, "
                  f"tau {q['tau_shots']:.0f} shots")
    (out / "quasi_steady_drift.json").write_text(json.dumps({
        "note": "exponential fit T_inf - A exp(-(n - n0)/tau) to the last two thirds of each run (of its neutron-heated "
                "shots); drift_left = asymptote - last shot", "runs": res}, indent=1), encoding="utf-8")
