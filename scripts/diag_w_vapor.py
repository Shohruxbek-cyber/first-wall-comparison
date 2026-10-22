"""Vapor shielding of the tungsten armor (revision of 20 Oct 2026).

(a) Coupled Demo single shot from 600 K with the shielding option on, to 50 us, with the
    deposition fractions recomputed after every accepted step, so that the column always equals
    the mass evaporated so far.  The production solver recomputes them only when the evaporated
    mass has changed by more than 1 % of the surface-cell mass (1.9e-7 kg/m^2 for W), so in the
    stored tungsten runs the column (at most 2.2e-11 kg/m^2) was never applied.
(b) Static share of the ion energy stopped by a column held in front of the production tungsten
    mesh from the start, for the per-shot evaporated mass of stored tungsten cases, up to the
    hottest one (degraded conductivity with a 5 um skin at 0.25).

Usage: python scripts/diag_w_vapor.py [--out=results/vapor_shielding/tungsten]
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import SCENARIOS, Options, w_case  # noqa: E402
from ife_wall.vapor import VaporShieldedSource  # noqa: E402

DEMO = SCENARIOS["Demo"]
RHO_W_REF = 19250.0  # kg/m^3, as in the drivers (nm per shot -> kg/m^2)
CASES = ((0.0, 1.0), (1.0, 1.0), (0.0, 2.0), (1.0, 2.0))  # (f_rr, s)
STORED = {  # per-shot evaporated depth (nm) of stored tungsten runs: file, key
    "Demo single shot from 600 K": ("runs/w_single/baseline/Demo.json", "ablated_nm_per_shot"),
    "Demo degraded conductivity 0.5, last shot of the run": ("runs/w_damage/multi/kirr0.5_conv/Demo.json", "last_shot_ablated_nm"),
    "Demo conductivity 0.5 with a 5 um skin at 0.25, last shot of the run": ("runs/w_damage/multi/Heskin_conv/Demo.json",
                                                                          "last_shot_ablated_nm"),
}


def coupled(f_rr, sf):
    s = w_case(DEMO, Options(vapor_shielding=True, vapor_f_rr=f_rr, vapor_stopping_factor=sf))
    w = s.sources[1]
    pk = [0.0]

    def rec(sv):
        pk[0] = max(pk[0], sv.surface_temperature())
        sv._frac_cache = None  # recompute the deposition (and the column) before the next step

    t0 = time.time()
    s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec)
    v = w.finalize(s.t)
    return {"f_rr": f_rr, "stopping_factor": sf, "peak_surface_T_K": pk[0],
            "evaporated_kg_m2": s.mass_evaporated, "vapor": v, "wall_s": time.time() - t0}


def static_share(M, sf):
    s = w_case(DEMO, Options())
    base = s.sources[1]
    w = VaporShieldedSource(base, stopping_factor=sf, fixed_column=M)
    w.fractions(s)
    E = base.fluence
    return float(E @ w.vapor_share / E.sum())


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "vapor_shielding" / "tungsten"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    prod = json.loads((ROOT / "runs" / "w_single" / "baseline" / "Demo.json").read_text(encoding="utf-8"))
    res = {"production_peak_surface_T_K": prod["peak_surface_T_K"], "static": {}, "coupled_refresh_every_step": []}
    for label, (f, key) in STORED.items():
        nm = json.loads((ROOT / f).read_text(encoding="utf-8"))[key]
        M = nm * 1e-9 * RHO_W_REF
        res["static"][label] = {"nm_per_shot": nm, "column_kg_m2": M,
                                "share_of_ion_energy_s1": static_share(M, 1.0), "share_of_ion_energy_s2": static_share(M, 2.0)}
    (out / "w_vapor.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for f_rr, sf in CASES:
        r = coupled(f_rr, sf)
        r["peak_change_K"] = r["peak_surface_T_K"] - prod["peak_surface_T_K"]
        res["coupled_refresh_every_step"].append(r)
        (out / "w_vapor.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f_rr, sf, r["peak_surface_T_K"], r["peak_change_K"], r["vapor"]["ion_energy_in_vapor_J_m2"], flush=True)
    print(json.dumps(res["static"], indent=1))
