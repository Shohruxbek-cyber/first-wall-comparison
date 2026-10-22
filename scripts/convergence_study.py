"""Grid and time-step convergence on the production cases (revision of 20 Oct 2026).

Cases: (a) the Demo tungsten single shot from 600 K (peak surface temperature, evaporated depth);
(b) one converged Demo tungsten cycle, started from the saved state after 600 shots
(runs/w_multi/verify_2026-10-05/Demo_checkpoint.npz; on a different grid the temperature profile
is interpolated in depth onto the new cells) -- pre-shot surface temperature at the end of the
cycle, peak surface and peak interface temperature; (c) the Demo lithium single shot at 31.6 um
(peak steel temperature, evaporated mass) and at 17.8 um, which bracket the single-shot
minimum film (interpolated as in the production scan).
Settings: baseline; surface cell halved and doubled (Options.dx0; the Li surface cell is
10 x dx0); front-layer growth 1.025 and 1.10 (baseline 1.05; steel growth scaled alike);
every time-step limit halved and doubled (temperature target, 2 % rule, x-ray pulse fraction,
ion-window fraction).  Observed orders from the three levels of each refinement.

Usage: python scripts/convergence_study.py [--out=results/revision/convergence]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import SCENARIOS, Options, interface_index, li_case, w_case  # noqa: E402

DEMO = SCENARIOS["Demo"]
DELTAS = np.geomspace(1e-6, 1e-3, 13)
SETTINGS = {  # name: (dx0 factor, growth, steel growth, time-step factor)
    "baseline": (1.0, 1.05, 1.15, 1.0),
    "dx0_half": (0.5, 1.05, 1.15, 1.0), "dx0_double": (2.0, 1.05, 1.15, 1.0),
    "growth_1.025": (1.0, 1.025, 1.075, 1.0), "growth_1.10": (1.0, 1.10, 1.30, 1.0),
    "dt_half": (1.0, 1.05, 1.15, 0.5), "dt_double": (1.0, 1.05, 1.15, 2.0),
}
CKPT = ROOT / "runs" / "w_multi" / "verify_2026-10-05" / "Demo_checkpoint.npz"


def options(setting):
    fdx, g, gs, _ = SETTINGS[setting]
    return Options(dx0=1e-9 * fdx, grid_growth=g, grid_growth_steel=gs)


def scale_steps(s, c):
    """Multiply the source step limits by c (x-ray pulse fraction, ion-window fraction)."""
    if c == 1.0:
        return
    s.sources[0].dt_fraction *= c
    ions = s.sources[1]
    f = ions.dt_limit
    ions.dt_limit = lambda t, _f=f, _c=c: _f(t) * _c


def run(s, t_end, dt0, c, rec):
    s.run(t_end, dt0=dt0, dT_target=5.0 * c, dt_rel=0.02 * c, record=rec)


def w_single(setting):
    c = SETTINGS[setting][3]
    s = w_case(DEMO, options(setting))
    scale_steps(s, c)
    pk = [0.0]
    run(s, 50e-6, 1e-13, c, lambda sv: pk.__setitem__(0, max(pk[0], sv.surface_temperature())))
    return {"peak_surface_K": pk[0], "evaporated_nm": s.mass_evaporated / 19250.0 * 1e9, "cells": len(s.m)}


def w_cycle(setting):
    c = SETTINGS[setting][3]
    z = np.load(CKPT, allow_pickle=False)
    ref = w_case(DEMO, Options())
    ref.m, ref.H, ref.layer = z["m"].copy(), z["H"].copy(), z["layer"].copy()
    s = w_case(DEMO, options(setting))
    if setting in ("baseline", "dt_half", "dt_double"):
        s.m, s.H, s.layer = ref.m.copy(), ref.H.copy(), ref.layer.copy()
    else:  # interpolate the converged temperature profile in depth onto the new cells, layer by layer
        T_ref, x_ref = ref.T, ref.x_centers()
        x_new = s.x_centers()
        T_new = np.interp(x_new, x_ref, T_ref)
        s.H = s._by_layer(lambda mat, sel: mat.h_of_T(T_new[sel]))
    s.mass_evaporated = float(z["mass_evaporated"])
    s._frac_cache = None
    scale_steps(s, c)
    i_if = interface_index(s)
    pk = {"Ts": 0.0, "Tif": 0.0}

    def rec(sv):
        pk["Ts"] = max(pk["Ts"], sv.surface_temperature())
        pk["Tif"] = max(pk["Tif"], float(sv.T[i_if]))

    pre = s.surface_temperature()
    run(s, 50e-6, 1e-13, c, rec)
    run(s, 0.1, 1e-9, c, rec)
    return {"preshot_start_K": pre, "preshot_next_K": s.surface_temperature(), "peak_surface_K": pk["Ts"],
            "peak_interface_K": pk["Tif"], "cells": len(s.m)}


def li_single(setting, i_delta):
    c = SETTINGS[setting][3]
    s = li_case(DEMO, float(DELTAS[i_delta]), 600.0, options(setting))
    scale_steps(s, c)
    pk = {"Tsub": 0.0}
    rec = lambda sv: pk.__setitem__("Tsub", max(pk["Tsub"], float(sv.T[sv.layer == 1].max())))  # noqa: E731
    run(s, 50e-6, 1e-13, c, rec)
    run(s, 5e-3, 1e-9, c, rec)
    return {"peak_steel_K": pk["Tsub"], "evaporated_kg_m2": s.mass_evaporated, "cells": len(s.m)}


def crossing(d1, T1, d2, T2, limit=925.0):
    f = (T1 - limit) / (T1 - T2)
    return float(np.exp(np.log(d1) + f * np.log(d2 / d1))) * 1e6


def orders(res, key_path):
    """Observed order from (coarse, base, fine) for the three refinements."""
    def get(name):
        v = res[name]
        for k in key_path:
            v = v[k]
        return v
    out = {}
    for label, coarse, fine in (("surface cell", "dx0_double", "dx0_half"), ("growth", "growth_1.10", "growth_1.025"),
                                ("time step", "dt_double", "dt_half")):
        fc, fb, ff = get(coarse), get("baseline"), get(fine)
        d1, d2 = abs(fc - fb), abs(fb - ff)
        out[label] = math.log2(d1 / d2) if d1 > 0 and d2 > 0 else None
    return out


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "convergence"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name in SETTINGS:
        t0 = time.time()
        r = {"w_single": w_single(name), "w_cycle": w_cycle(name), "li_31.6um": li_single(name, 6), "li_17.8um": li_single(name, 5)}
        r["li_min_film_um"] = crossing(DELTAS[5], r["li_17.8um"]["peak_steel_K"], DELTAS[6], r["li_31.6um"]["peak_steel_K"])
        r["wall_s"] = time.time() - t0
        res[name] = r
        (out / "convergence.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(name, json.dumps({k: v for k, v in r.items() if k != "wall_s"}), flush=True)
    keys = {"W single, peak surface": ("w_single", "peak_surface_K"), "W single, evaporated depth": ("w_single", "evaporated_nm"),
            "W cycle, next pre-shot": ("w_cycle", "preshot_next_K"), "W cycle, peak surface": ("w_cycle", "peak_surface_K"),
            "W cycle, peak interface": ("w_cycle", "peak_interface_K"), "Li 31.6 um, peak steel": ("li_31.6um", "peak_steel_K"),
            "Li 31.6 um, evaporated": ("li_31.6um", "evaporated_kg_m2"), "Li minimum film": ("li_min_film_um",)}
    summary = {label: {"values": {n: (res[n][k[0]][k[1]] if len(k) == 2 else res[n][k[0]]) for n in SETTINGS},
                       "observed_order": orders(res, k)} for label, k in keys.items()}
    (out / "convergence_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
