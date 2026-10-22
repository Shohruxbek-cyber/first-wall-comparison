"""Converged pre-shot state of the repeated-shot tungsten runs (revision of 22 Oct 2026).

Reads the final checkpoint of runs/w_multi/baseline (Demo after 600 shots, Prototype after 100: the state just
before the next shot) and gives the tungsten surface temperature, the W/steel interface, the temperature of the
last steel cell (its centre, which the per-shot record calls the back face) and the true back-face temperature of
the steel, T_face = T_c + q/h with q the coolant flux of the solver's own boundary conductance.  These are the
hot-start temperature of the single-shot studies and the start of the helium-model cycles (--Tback of
scripts/record_he_thermal.py, which places it at the face).  Also the same quantities of the last recorded
pre-shot state of the per-shot table.

Usage: python scripts/diag_converged_state.py [--out=results/revision/converged_state]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import DEFAULT_ION_SPECTRA  # noqa: E402
from ife_wall.scenarios import SCENARIOS, Options, w_case  # noqa: E402
from ife_wall.thermal import EnergyLedger  # noqa: E402


def state(name, run_dir):
    s = w_case(SCENARIOS[name], Options())
    z = np.load(run_dir / f"{name}_checkpoint.npz", allow_pickle=False)
    s.m, s.H, s.layer = z["m"].copy(), z["H"].copy(), z["layer"].copy()
    s.m_target0 = z["m_target0"].copy()
    s.t = float(z["t"])
    s.ledger = EnergyLedger(**json.loads(str(z["ledger"])))
    s._frac_cache = None
    T = s.T
    k = s._by_layer(lambda mat, sl: mat.k(T[sl]))
    r_half = s.dx(T) / (2 * k)
    _, G_back, _ = s._conductances(T)
    q = G_back * (T[-1] - s.back.T_coolant)
    i = int(np.argmax(s.layer == 1)) - 1                      # last tungsten cell
    T_if = (T[i] / r_half[i] + T[i + 1] / r_half[i + 1]) / (1 / r_half[i] + 1 / r_half[i + 1])
    rows = np.loadtxt(run_dir / f"{name}_per_shot.csv", delimiter=",", skiprows=1)
    return {"shots_done": int(len(z["rows"])), "t_s": s.t,
            "surface_K": float(s.surface_temperature(T)), "interface_K": float(T_if),
            "last_steel_cell_centre_K": float(T[-1]), "back_face_K": float(s.back.T_coolant + q / s.back.h),
            "coolant_flux_W_m2": float(q), "last_steel_cell_thickness_mm": float(s.dx(T)[-1] * 1e3),
            "per_shot_table_last_row": {"shot": int(rows[-1, 0]), "preshot_surface_K": float(rows[-1, 1]),
                                        "preshot_last_steel_cell_K": float(rows[-1, 4])}}


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "converged_state"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "runs" / "w_multi" / "baseline"
    res = {"note": __doc__.split("Usage:")[0].strip(), "ion_spectra": DEFAULT_ION_SPECTRA,
           "scenarios": {n: state(n, run_dir) for n in ("Demo", "Prototype")}}
    (out / "converged_state.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n, r in res["scenarios"].items():
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
