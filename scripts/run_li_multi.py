"""Production (added 2026-10-06): multi-shot Li film, complete replacement each shot, steel
carried forward.  NEW CODE PATH: scripts/run_li_scan.py and the W drivers are untouched.

Per shot k = 0 ... n-1:
  * k > 0: the Li layer is rebuilt (full thickness, uniform T_li); the steel keeps its profile
    (ife_wall.li_multishot.reset_film);
  * the solver clock is reset to 0, so every shot is integrated exactly as the single-shot
    scan: to 50 us at dt0 = 1e-13 s, then to 1/f at dt0 = 1e-9 s (or to --t_end, for the
    verification against the single-shot scan, which integrates to 5 ms);
  * recorded per shot: pre-shot steel front-face T, peak steel T, peak Li surface T, pre-shot
    back-face T, Li evaporated in the shot, enthalpy removed with the replaced film, the
    cumulative ledger residual, burn-through flag.

Usage: python scripts/run_li_multi.py <Prototype|Demo> <n_shots> delta_um=<um> [T_li=600]
         [T_c=600] [li_lv=davison_slope] [option=value ...] [--out=DIR] [--converge]
         [--resume] [--t_end=SECONDS]
  --converge  stop once the pre-shot steel front-face T rises < 0.05 K/shot over the last 10 %
              of shots (checked every 10 shots, at least 100 shots): the W criterion, applied
              to the steel
  --resume    continue from DIR/<scenario>_checkpoint.npz (written every 10 shots)
  --t_end     integrate each shot only to this time instead of 1/f (verification)
  --return-latent  bounding case of 21 Oct 2026: after the first 50 us of each shot, the latent
              heat of the lithium evaporated in that shot is returned to the film surface as a uniform flux
              until the end of the period (off by default; the default path is unchanged)
  renew=<f>   partial renewal (22 Oct 2026): only the fraction f (0 < f <= 1) of the film
              is replaced each shot, the evaporated mass is made up, and the retained lithium keeps its heat
              (ife_wall.li_multishot.renew_film); without it the film is replaced completely (default path unchanged)
Output: runs/li_multi/<tag>/<scenario>.json (+ <scenario>_per_shot.csv, figure), or DIR.
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ife_wall.li_multishot import LiLedger, deposition_split, li_multishot_case, renew_film, reset_film, steel_index  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402
from ife_wall.thermal import FrontBC  # noqa: E402
from run_w_single import git_hash, parse_overrides  # noqa: E402

CONV_TOL, CONV_MIN = 0.05, 100
LIMIT_K = 925.0
COLS = ("shot,preshot_steel_front_T_K,peak_steel_T_K,peak_Li_surface_T_K,preshot_back_T_K,"
        "vaporized_kg_m2_this_shot,cum_vaporized_kg_m2,removed_with_film_J_m2,energy_rel_residual_cum,burn_through")


def flag(name):
    v = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return v[0] if v else None


converge = "--converge" in sys.argv[1:]
return_latent = "--return-latent" in sys.argv[1:]
resume = "--resume" in sys.argv[1:]
t_end_override = flag("t_end")
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
name, n = argv[0], int(argv[1])
kv = dict(a.split("=", 1) for a in argv[2:])
if "delta_index" in kv:  # the scan's own grid value, bit-identical to scripts/run_li_scan.py
    delta = float(np.geomspace(1e-6, 1e-3, 13)[int(kv.pop("delta_index"))])
else:
    delta = float(kv.pop("delta_um")) * 1e-6
T_li = float(kv.pop("T_li", 600.0))
T_c = float(kv.pop("T_c", 600.0))
li_lv = kv.pop("li_lv", "davison_slope")
renew = float(kv.pop("renew")) if "renew" in kv else None
opt = parse_overrides([f"{k}={v}" for k, v in kv.items()])
scen = SCENARIOS[name]
if scen.rep_Hz is None and t_end_override is None:
    raise SystemExit(f"{name} has no repetition rate; give --t_end")
P = 1.0 / scen.rep_Hz if scen.rep_Hz else None
t_end = float(t_end_override) if t_end_override else P
tag = "_".join([f"d{delta * 1e6:g}um", f"T{T_li:.0f}K", f"Tc{T_c:.0f}K"]
               + ([f"lv-{li_lv}"] if li_lv != "davison_slope" else []) + [f"{k}-{v}" for k, v in kv.items()]
               + ([f"tend-{t_end_override}"] if t_end_override else []) + (["return-latent"] if return_latent else [])
               + ([f"renew-{renew:g}"] if renew is not None else []))
d = Path(flag("out")) if flag("out") else ROOT / "runs" / "li_multi" / tag
d.mkdir(parents=True, exist_ok=True)
ckpt = d / f"{name}_checkpoint.npz"

s = li_multishot_case(scen, delta, T_li, opt, li_lv=li_lv, T_coolant=T_c)
split = deposition_split(s)
rho_li = float(s.materials[0].rho(np.array(T_li)))
m_li0 = float(s.m[s.layer == 0].sum())
rows = []
film_T = []  # film temperature after each renewal (renew runs only)
t0 = time.time()
if resume and ckpt.exists():
    z = np.load(ckpt, allow_pickle=False)
    s.m, s.H, s.layer = z["m"].copy(), z["H"].copy(), z["layer"].copy()
    s.m_target0 = z["m_target0"].copy()
    s.mass_evaporated, s.n_rezones = float(z["mass_evaporated"]), int(z["n_rezones"])
    s.ledger = LiLedger(**json.loads(str(z["ledger"])))
    s._frac_cache = None
    rows = [list(r) for r in z["rows"]]
    film_T = list(z["film_T"]) if "film_T" in z.files else []
    print(f"resumed from {ckpt} at shot {len(rows)}", flush=True)


def save_checkpoint():
    np.savez(ckpt, m=s.m, H=s.H, layer=s.layer, m_target0=s.m_target0, mass_evaporated=s.mass_evaporated,
             n_rezones=s.n_rezones, ledger=json.dumps(dataclasses.asdict(s.ledger)), rows=np.array(rows),
             **({"film_T": np.array(film_T)} if renew is not None else {}))


def fit_asymptote(shots, y):
    """Exponential approach y = T_inf - A exp(-n/tau) fitted to the last two thirds of the run,
    as scripts/make_reports.py does for the W interface (Section 5.2)."""
    try:
        from scipy.optimize import curve_fit
        k = len(shots) // 3
        p, _ = curve_fit(lambda x, Ti, A, tau: Ti - A * np.exp(-x / tau), shots[k:], y[k:],
                         p0=(y[-1] + 10, y[-1] - y[0], len(shots) / 3), maxfev=20000)
        return {"T_inf_K": float(p[0]), "tau_shots": float(p[2]), "fitted_shots": [int(shots[k]), int(shots[-1])],
                "label": "EXTRAPOLATION: exponential approach fitted to the last two thirds of the run"}
    except Exception as e:  # noqa: BLE001
        return {"fit_error": str(e)}


def write_outputs(status, converged, slope):
    r = np.array(rows)
    np.savetxt(d / f"{name}_per_shot.csv", r, delimiter=",", header=COLS, comments="")
    above = r[:, 2] > LIMIT_K
    out = {"scenario": name, "delta_um": delta * 1e6, "T_li_K": T_li, "T_c_K": T_c, "h_W_m2K": s.back.h,
           "rep_Hz": scen.rep_Hz, "shot_period_s": P, "integrated_to_s_per_shot": t_end, "li_lv": li_lv,
           "options": {k: v for k, v in vars(opt).items() if k != "extra"}, "git": git_hash(),
           "model": ("film replaced at full thickness and T_li at the start of every shot after the first; "
                     "steel enthalpy profile carried forward; coolant fixed at T_c; the enthalpy removed with "
                     "the replaced film is booked in the ledger (modelling choice, see ife_wall/li_multishot.py)"),
           "deposition_per_shot_J_m2": split, "n_shots": len(r), "max_n_shots": n, "status": status,
           "converged": bool(converged), "preshot_front_rise_per_shot_last10pct_K": slope,
           "convergence_criterion": (f"pre-shot steel front-face T rise < {CONV_TOL} K/shot over the last 10 % "
                                     f"of shots (min {CONV_MIN} shots)"),
           "first_shot_peak_steel_K": float(r[0, 2]), "last_shot_peak_steel_K": float(r[-1, 2]),
           "max_peak_steel_K": float(r[:, 2].max()), "last_preshot_steel_front_K": float(r[-1, 1]),
           "first_shot_steel_above_925K": int(r[np.argmax(above), 0]) if above.any() else None,
           "first_shot_peak_Li_surface_K": float(r[0, 3]), "last_shot_peak_Li_surface_K": float(r[-1, 3]),
           "first_shot_vaporized_kg_m2": float(r[0, 5]), "last_shot_vaporized_kg_m2": float(r[-1, 5]),
           "cum_vaporized_kg_m2": float(r[-1, 6]), "removed_with_film_total_J_m2": float(s.ledger.removed_with_film),
           "energy_rel_residual": float(s.ledger.residual()[1]), "max_abs_energy_rel_residual_per_shot": float(np.abs(r[:, 8]).max()),
           "any_burn_through": bool(r[:, 9].any()), "wall_s_this_session": time.time() - t0,
           "ledger_cumulative_J_m2": dataclasses.asdict(s.ledger)}
    if return_latent:
        out["return_latent"] = ("bounding case: latent heat of the lithium evaporated in each shot returned to the film "
                                "surface as a uniform flux from 50 us to the end of the period")
    if renew is not None:
        out["renewal_fraction"] = renew
        out["model"] = (f"partial renewal: {renew:g} of the film left after each shot replaced by fresh lithium at T_li "
                        "and the evaporated mass made up, the retained lithium keeping its enthalpy, mixed uniformly "
                        "(ife_wall.li_multishot.renew_film); steel enthalpy profile carried forward; coolant fixed at "
                        "T_c; the enthalpy removed with the exchanged lithium is booked in the ledger")
        out["film_T_after_renewal_K"] = [float(x) for x in film_T]
        out["last_film_T_after_renewal_K"] = float(film_T[-1]) if film_T else None
        out["last_shot_steel_swing_K"] = float(r[-1, 2] - r[-1, 1])
    if len(r) >= 30:
        out["fit_peak_steel"] = fit_asymptote(r[:, 0], r[:, 2])
        out["fit_preshot_front"] = fit_asymptote(r[:, 0], r[:, 1])
    (d / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


converged, slope, status = False, float("nan"), "ok"
for k in range(len(rows), n):
    if k == 0:
        removed = 0.0
    elif renew is None:
        removed = reset_film(s, T_li)
    else:
        removed, Tf = renew_film(s, T_li, renew)
        film_T.append(Tf)
    s.t = 0.0
    for src in s.sources:  # vapor shielding (optional): the column restarts every shot
        if hasattr(src, "start_shot"):
            src.start_shot(s)
    i_fe = steel_index(s)
    T = s.T
    pre_front, pre_back = float(T[i_fe]), float(T[-1])
    vap0 = s.mass_evaporated
    pk = {"Ts": 0.0, "Tsub": 0.0}

    def rec(sv):
        T = sv.T
        if np.any(sv.layer == 0):
            pk["Ts"] = max(pk["Ts"], sv.surface_temperature(T))
        pk["Tsub"] = max(pk["Tsub"], float(T[sv.layer == 1].max()))

    try:
        lat0 = s.ledger.vaporized_latent
        s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec)
        if return_latent:
            q_ret = (s.ledger.vaporized_latent - lat0) / (t_end - s.t)
            s.front = FrontBC(kind="flux", flux=lambda tt, q=q_ret: q)
        s.run(t_end, dt0=1e-9, dT_target=5.0, record=rec)
        if return_latent:
            s.front = FrontBC()
    except RuntimeError as e:
        status = f"stopped in shot {k + 1} at t={s.t:.3e} s: {e}"
    s.ledger.stored_final = s.stored_energy()
    m_left = float(s.m[s.layer == 0].sum()) if np.any(s.layer == 0) else 0.0
    rows.append([k + 1, pre_front, pk["Tsub"], pk["Ts"], pre_back, s.mass_evaporated - vap0, s.mass_evaporated,
                 removed, float(s.ledger.residual()[1]), float(m_left < 0.01 * m_li0)])
    if (k + 1) % 10 == 0 or k == n - 1 or status != "ok":
        r = np.array(rows)
        m = min(max(len(r) // 10, 2), len(r))
        slope = float((r[-1, 1] - r[-m, 1]) / (m - 1)) if m >= 2 else float("nan")
        converged = len(r) >= CONV_MIN and abs(slope) < CONV_TOL
        write_outputs(status, converged, slope)
        save_checkpoint()
        print(f"{name} d={delta * 1e6:g} um shot {k + 1}: pre-front {r[-1, 1]:.1f} K, peak steel {r[-1, 2]:.1f} K, "
              f"Li surf {r[-1, 3]:.0f} K, vap {r[-1, 5]:.3e} kg/m2, dT_pre/shot {slope:.3f} K"
              f"{' CONVERGED' if converged else ''}, resid {r[-1, 8]:.1e}, {time.time() - t0:.0f}s", flush=True)
        if status != "ok" or (converge and converged):
            break

r = np.array(rows)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(r[:, 0], r[:, 1], label="steel front face, just before shot")
ax[0].plot(r[:, 0], r[:, 4], label="steel back face, before shot")
ax[0].set(xlabel="shot number", ylabel="temperature (K)", title=f"{name}, Li {delta * 1e6:g} um: pre-shot temperatures ({scen.rep_Hz:g} Hz)")
ax[0].legend()
ax[1].plot(r[:, 0], r[:, 2], label="peak steel")
ax[1].plot(r[:, 0], r[:, 3], label="peak Li surface")
ax[1].axhline(LIMIT_K, ls="--", c="k", lw=0.8)
ax[1].text(1, LIMIT_K + 15, "ODS-RAFM limit 925 K", fontsize=8)
ax[1].set(xlabel="shot number", ylabel="temperature (K)", title="per-shot peaks")
ax[1].legend()
ax[2].plot(r[:, 0], r[:, 5])
ax[2].set(xlabel="shot number", ylabel="Li evaporated per shot (kg/m$^2$)", title="evaporation per shot")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(d / f"{name}_multishot.{ext}", dpi=150)
print("done:", status, flush=True)
