"""One more shot cycle of a stored repeated-shot lithium run, fully recorded (revision of 21 Oct 2026).

Loads <run_dir>/<scenario>_checkpoint.npz (read only) and <scenario>.json, rebuilds the case with the stored
options, replaces the film and fires the next shot exactly as scripts/run_li_multi.py does (clock reset, to 50 us
at dt0 = 1e-13 s, then to the shot period at dt0 = 1e-9 s), and records at every solver step the steel front-face
temperature (first steel cell, as in the per-shot record), the lithium surface temperature, the hottest lithium
cell and its depth below the surface.  Reports
  * the time-mean front-face temperature over the cycle and the time per shot above 925 K and 873 K (M1);
  * the energy partition of the shot from the ledger (absorbed; latent and sensible heat leaving with the vapor;
    enthalpy removed by the film replacement at the start of the shot; conducted into the steel; deposited
    directly in the steel; to the coolant) and the time-mean heat flux into the steel (M4);
  * the histories needed for the surface pressure (M6): surface temperature against time, the peak of the
    hottest lithium cell and its depth;
  * the pulsed load on the steel face (22 Oct 2026): the per-shot swing of the
    front-face temperature, and the heat-flux factor of the energy entering the steel, E_in(t) = change of the
    steel's stored energy + heat to the coolant (exact by the steel's energy balance), in the worst-window form
    F = max over windows of dE_in / sqrt(dt), and as the square-pulse equivalent of the swing,
    F_eq = swing * sqrt(pi k rho c) / 2 with the steel properties at the pre-shot front-face temperature.
Usage: python scripts/li_extra_cycle.py <run_dir> <scenario> [--out=DIR]   (default DIR: <run_dir>_extra_cycle)
"""
import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.li_multishot import LiLedger, li_multishot_case, reset_film, steel_index  # noqa: E402
from ife_wall.scenarios import SCENARIOS, Options  # noqa: E402

LIMITS = (925.0, 873.0)


def time_above(t, T, level):
    """Time with T > level, linear interpolation inside each step."""
    tot = 0.0
    for i in range(1, len(t)):
        a, b, dt = T[i - 1] - level, T[i] - level, t[i] - t[i - 1]
        if a > 0 and b > 0:
            tot += dt
        elif a > 0 or b > 0:
            tot += dt * (max(a, b) / (abs(a) + abs(b)))
    return tot


def worst_window_F(t, E):
    """max over t1 < t2 of (E(t2) - E(t1)) / sqrt(t2 - t1), W s^1/2 m^-2 (E cumulative, J/m^2)."""
    best, arg = 0.0, (0.0, 0.0)
    for i in range(len(t) - 1):
        dt = t[i + 1:] - t[i]
        ok = dt > 0
        if not ok.any():
            continue
        f = np.where(ok, (E[i + 1:] - E[i]) / np.sqrt(np.where(ok, dt, 1.0)), -np.inf)
        j = int(np.argmax(f))
        if f[j] > best:
            best, arg = float(f[j]), (float(t[i]), float(t[i + 1 + j]))
    return best, arg


def steel_pulse(s, t, Tf, E_in):
    """M3: per-shot swing of the steel front face and the heat-flux factors of the energy entering the steel."""
    k0 = int(np.argmax(t > 0)) if np.any(t > 0) else 0
    tt, EE = t[k0 - 1 if k0 > 0 else 0:], E_in[k0 - 1 if k0 > 0 else 0:]
    F, (t1, t2) = worst_window_F(tt, EE)
    steel = s.materials[1]
    T0 = np.array([float(Tf[0])])
    krc = float(steel.k(T0)[0] * steel.rho(T0)[0] * steel.cp(T0)[0]) if hasattr(steel, "cp") else None
    swing = float(Tf.max() - Tf[0])
    return {"swing_K": swing, "t_peak_front_s": float(t[int(np.argmax(Tf))]),
            "F_worst_window_MW_m2_s05": F / 1e6, "worst_window_s": [t1, t2],
            "E_in_per_shot_J_m2": float(E_in[-1]),
            "F_square_pulse_equivalent_MW_m2_s05": (swing * np.sqrt(np.pi * krc) / 2 / 1e6) if krc else None,
            "steel_k_rho_c_at_preshot_front": krc,
            "note": ("pulsed-load indicator only: no pulsed (thermal-fatigue) design rule for steel was found; the "
                     "tungsten rule N = exp(100/F) of Mitteau et al. does not apply to steel")}


def main():
    run_dir, name = Path(sys.argv[1]), sys.argv[2]
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    out = next((Path(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("--out=")), None)
    out = (out if out.is_absolute() else ROOT / out) if out else run_dir.parent / (run_dir.name + "_extra_cycle")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run_dir / f"{name}.json").read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(Options)}
    opt = Options(**{k: v for k, v in meta["options"].items() if k in fields})
    scen = SCENARIOS[name]
    delta, T_li, T_c, li_lv = meta["delta_um"] * 1e-6, meta["T_li_K"], meta["T_c_K"], meta["li_lv"]
    P = 1.0 / scen.rep_Hz
    s = li_multishot_case(scen, delta, T_li, opt, li_lv=li_lv, T_coolant=T_c)
    z = np.load(run_dir / f"{name}_checkpoint.npz", allow_pickle=False)
    s.m, s.H, s.layer = z["m"].copy(), z["H"].copy(), z["layer"].copy()
    s.m_target0 = z["m_target0"].copy()
    s.mass_evaporated, s.n_rezones = float(z["mass_evaporated"]), int(z["n_rezones"])
    s.ledger = LiLedger(**json.loads(str(z["ledger"])))
    s._frac_cache = None
    n_done = len(z["rows"])
    L0 = copy.deepcopy(s.ledger)
    removed = reset_film(s, T_li)
    s.t = 0.0
    for src in s.sources:
        if hasattr(src, "start_shot"):
            src.start_shot(s)
    stored0 = s.stored_energy()
    hist = {"t": [], "T_front": [], "T_surface": [], "T_li_max": [], "li_max_depth_um": [], "T_steel_max": [],
            "E_steel_in_J_m2": []}
    fe0 = s.layer == 1
    E_fe0 = float(np.sum(s.m[fe0] * s.H[fe0]))
    cool0 = s.ledger.to_coolant

    def rec(sv):
        T = sv.T
        i_fe = steel_index(sv)
        li = sv.layer == 0
        hist["t"].append(sv.t)
        hist["T_front"].append(float(T[i_fe]))
        hist["T_steel_max"].append(float(T[sv.layer == 1].max()))
        fe = sv.layer == 1
        hist["E_steel_in_J_m2"].append(float(np.sum(sv.m[fe] * sv.H[fe])) - E_fe0 + sv.ledger.to_coolant - cool0)
        if li.any():
            hist["T_surface"].append(float(sv.surface_temperature(T)))
            k = int(np.argmax(np.where(li, T, -np.inf)))
            xm = np.cumsum(sv.m / sv.materials[0].rho(T) * li)  # depth of cell ends in the film (m)
            hist["T_li_max"].append(float(T[k]))
            hist["li_max_depth_um"].append(float((xm[k] - 0.5 * sv.m[k] / float(sv.materials[0].rho(np.array(T[k])))) * 1e6))
        else:
            hist["T_surface"].append(float("nan")); hist["T_li_max"].append(float("nan")); hist["li_max_depth_um"].append(float("nan"))

    T0 = s.T
    rec(s)
    s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec)
    s.run(P, dt0=1e-9, dT_target=5.0, record=rec)
    L1 = s.ledger
    t = np.array(hist["t"])
    Tf = np.array(hist["T_front"])
    dL = {k: getattr(L1, k) - getattr(L0, k) for k in ("deposited", "to_coolant", "vaporized_latent", "vaporized_sensible",
                                                        "removed_with_film", "incident", "passed_through")}
    dL["through_interface_Li_steel"] = L1.through_interface[0] - L0.through_interface[0]
    absorbed = dL["deposited"]
    i_fe0 = steel_index(s)
    res = {
        "source_run": (str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir)).replace("\\", "/"),
        "scenario": name, "delta_um": delta * 1e6,
        "shot_number": n_done + 1, "shot_period_s": P,
        "preshot_front_K": float(Tf[0]), "peak_front_K": float(Tf.max()), "peak_steel_K": float(np.max(hist["T_steel_max"])),
        "stored_last_shot_peak_steel_K": meta["last_shot_peak_steel_K"],
        "time_mean_front_K": float(np.trapezoid(Tf, t) / (t[-1] - t[0])),
        "time_above_K_s": {f"{lv:.0f}": time_above(t, Tf, lv) for lv in LIMITS},
        "energy_per_shot_J_m2": {**dL, "removed_with_film_at_shot_start": removed, "absorbed": absorbed,
                                 "stored_change": s.stored_energy() - stored0},
        "fractions_of_absorbed": {"vapor_latent": dL["vaporized_latent"] / absorbed, "vapor_sensible": dL["vaporized_sensible"] / absorbed,
                                  "film_replacement": removed / absorbed, "to_coolant": dL["to_coolant"] / absorbed,
                                  "conducted_Li_to_steel": dL["through_interface_Li_steel"] / absorbed},
        "time_mean_heat_flux_into_steel_W_m2": {"to_coolant": dL["to_coolant"] / P,
                                                "conducted_from_film": dL["through_interface_Li_steel"] / P},
        "peak_Li_surface_K": float(np.nanmax(hist["T_surface"])), "peak_Li_cell_K": float(np.nanmax(hist["T_li_max"])),
        "depth_of_Li_peak_um_at_its_max": float(np.array(hist["li_max_depth_um"])[int(np.nanargmax(hist["T_li_max"]))]),
        "evaporated_kg_m2": float(s.mass_evaporated - float(z["mass_evaporated"])),
        "n_steps": len(t), "first_steel_cell_index_at_end": i_fe0,
    }
    res["steel_face_pulse"] = steel_pulse(s, t, Tf, np.array(hist["E_steel_in_J_m2"]))
    tag = f"{name}"
    (out / f"{tag}_extra_cycle.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savetxt(out / f"{tag}_extra_cycle_history.csv", np.column_stack([hist[k] for k in hist]), delimiter=",",
               header=",".join(hist), comments="")
    print(json.dumps({k: res[k] for k in ("scenario", "delta_um", "preshot_front_K", "peak_front_K", "peak_steel_K",
                                          "stored_last_shot_peak_steel_K", "time_mean_front_K", "time_above_K_s",
                                          "fractions_of_absorbed", "peak_Li_surface_K", "peak_Li_cell_K")}, indent=1))


if __name__ == "__main__":
    main()
