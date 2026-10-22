"""Diagnostic (revision of 20 Oct 2026): first-cell-centre
temperature T0 against the reported surface temperature Ts during a shot, and the effect on
the Hertz-Knudsen-Langmuir evaporation if it were evaluated at Ts instead of T0.

The production solver evaluates the evaporation flux j at T0 (thermal.ThermalSolver.step) and
reports Ts = T0 + q dx0 / (2 k0) with q = -j L_v (ThermalSolver.surface_temperature).  This
script reruns production cases with the UNCHANGED solver and a record callback that, after every
accepted step, stores T0, Ts, j(T0) and j(Ts).  The mass evaporated with j(T0) is the solver's
own (it uses j(T_new[0]) dt per step); the mass with j(Ts) is a first-order, non-self-consistent
estimate (lower evaporation would also cool the surface less).

Cases (as in the production drivers): W Demo single shot from 600 K and from the converged
converged pre-shot state, integrated to 50 us (where run_w_single.py takes the ablated depth); Li film on
steel in Demo at T_li = 600 K for delta = 1 mm and delta = 31.6 um (DELTAS[12] and DELTAS[6]
of run_li_scan.py), integrated to 50 us and then to 5 ms as run_li_scan.py does.

Usage: python scripts/diag_T0_vs_Ts.py [--out=results/revision/T0_vs_Ts]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import SCENARIOS, Options, converged_preshot, li_case, w_case  # noqa: E402

DELTAS = np.geomspace(1e-6, 1e-3, 13)  # identical to scripts/run_li_scan.py


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "no-git"


def run_case(label, s, parts):
    mat = s.materials[s.layer[0]]
    vap, alpha = mat.vapor, s.alpha_evap
    rec = {"t": [], "T0": [], "Ts": [], "j0": [], "js": []}
    prev = {"t": s.t}
    acc = {"M0": 0.0, "Ms": 0.0}

    def cb(sv):
        T = sv.T
        T0 = float(T[0])
        Ts = sv.surface_temperature(T)
        j0 = float(vap.hkl_flux(T0, alpha))
        js = float(vap.hkl_flux(Ts, alpha))
        dt = sv.t - prev["t"]
        prev["t"] = sv.t
        acc["M0"] += j0 * dt
        acc["Ms"] += js * dt
        for k, v in (("t", sv.t), ("T0", T0), ("Ts", Ts), ("j0", j0), ("js", js)):
            rec[k].append(v)

    t0 = time.time()
    status = "ok"
    try:
        for t_end, dt0 in parts:
            s.run(t_end, dt0=dt0, dT_target=5.0, record=cb)
    except RuntimeError as e:
        status = f"stopped at t={s.t:.3e} s: {e}"
    T0 = np.array(rec["T0"])
    Ts = np.array(rec["Ts"])
    i = int(np.argmax(T0))
    p_sat_peak = float(vap.p_sat(np.array(T0[i])))
    out = {"case": label, "status": status, "steps": len(T0), "t_end_s": s.t,
           "peak_T0_K": float(T0.max()), "peak_Ts_K": float(Ts.max()),
           "T0_minus_Ts_at_peak_T0_K": float(T0[i] - Ts[i]), "max_T0_minus_Ts_K": float((T0 - Ts).max()),
           "p_sat_at_peak_T0_Pa": p_sat_peak,
           "mass_evaporated_solver_kg_m2": s.mass_evaporated,
           "mass_with_j_at_T0_recomputed_kg_m2": acc["M0"],
           "mass_with_j_at_Ts_first_order_kg_m2": acc["Ms"],
           "ratio_Ts_over_T0": (acc["Ms"] / acc["M0"]) if acc["M0"] > 0 else None,
           "alpha_evap": alpha, "surface_cell_thickness_m": float(s.dx()[0]), "wall_s": time.time() - t0}
    return out


if __name__ == "__main__":
    outdir = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), ROOT / "results" / "revision" / "T0_vs_Ts"))
    outdir.mkdir(parents=True, exist_ok=True)
    demo = SCENARIOS["Demo"]
    results = []
    results.append(run_case("W Demo, from 600 K, to 50 us", w_case(demo, Options()), [(50e-6, 1e-13)]))
    T_hot = converged_preshot("Demo", legacy=921.1)
    results.append(run_case(f"W Demo, from {T_hot:.1f} K, to 50 us", w_case(demo, Options(T_init=T_hot)), [(50e-6, 1e-13)]))
    for idx in (12, 6):
        d = float(DELTAS[idx])
        results.append(run_case(f"Li Demo, delta = {d * 1e6:.4g} um, T_li = 600 K, to 5 ms", li_case(demo, d, 600.0, Options()),
                                [(50e-6, 1e-13), (5e-3, 1e-9)]))
    meta = {"script": "scripts/diag_T0_vs_Ts.py", "git": git_hash(), "date": time.strftime("%Y-%m-%d %H:%M"), "results": results}
    (outdir / "T0_vs_Ts.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for r in results:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
