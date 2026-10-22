"""Production: multi-shot tungsten response at the design rep rate.

Records, for every shot: surface T just before the shot, peak surface T, peak W/steel
interface T, steel back-face T before the shot, cumulative ablation, melt depth and W lost in
that shot.  Diagnostics (He-layer history, depth table, gradient, stress proxy) are kept
for the most recent shot, so whichever shot ends the run has them.

Usage: python scripts/run_w_multi.py <Prototype|Demo> <n_shots> [option=value ...]
                                     [--out=DIR] [--converge] [--resume]
  n_shots      number of shots, or the maximum number with --converge
  --converge   stop early once the pre-shot surface temperature rises < 0.05 K/shot over the
               last 10 % of shots (checked every 10 shots, at least 100 shots)
  --resume     continue from DIR/<scenario>_checkpoint.npz (written every 10 shots).  Each
               shot restarts its time stepping from dt0, so a resumed run reproduces an
               uninterrupted one.
Output: runs/w_multi/<tag>/<scenario>.json (+ per-shot CSV, figure), or DIR with --out.
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
from ife_wall.diagnostics import ShotRecorder  # noqa: E402
from ife_wall.scenarios import SCENARIOS, interface_index, w_case  # noqa: E402
from ife_wall.thermal import EnergyLedger  # noqa: E402
from run_w_single import k_status, parse_overrides, save_diagnostics  # noqa: E402

CONV_TOL, CONV_MIN = 0.05, 100
outdir = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")]
converge = "--converge" in sys.argv[1:]
resume = "--resume" in sys.argv[1:]
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
name, n = argv[0], int(argv[1])
args = argv[2:]
opt = parse_overrides(args)
tag = "baseline" if not args else "_".join(a.replace("=", "-") for a in args)
scen = SCENARIOS[name]
P = 1.0 / scen.rep_Hz
s = w_case(scen, opt, n_shots=n)
d = Path(outdir[0]) if outdir else ROOT / "runs" / "w_multi" / tag
d.mkdir(parents=True, exist_ok=True)
ckpt = d / f"{name}_checkpoint.npz"
rows = []
t0 = time.time()
if resume and ckpt.exists():
    z = np.load(ckpt, allow_pickle=False)
    s.m, s.H, s.layer = z["m"].copy(), z["H"].copy(), z["layer"].copy()
    s.m_target0 = z["m_target0"].copy()
    s.t, s.mass_evaporated, s.n_rezones = float(z["t"]), float(z["mass_evaporated"]), int(z["n_rezones"])
    s.ledger = EnergyLedger(**json.loads(str(z["ledger"])))
    s._frac_cache = None
    rows = [list(r) for r in z["rows"]]
    print(f"resumed from {ckpt} at shot {len(rows)}", flush=True)


def save_checkpoint():
    np.savez(ckpt, m=s.m, H=s.H, layer=s.layer, m_target0=s.m_target0, t=s.t, mass_evaporated=s.mass_evaporated,
             n_rezones=s.n_rezones, ledger=json.dumps(dataclasses.asdict(s.ledger)), rows=np.array(rows))


last = None  # ShotRecorder of the most recent shot
converged = False
for k in range(len(rows), n):
    pre_Ts = s.surface_temperature()
    pre_back = float(s.T[-1])
    pk = {"Ts": 0.0, "Tif": 0.0, "melt": 0.0}
    abl0 = s.mass_evaporated
    last = ShotRecorder(s, s.t, P)

    def rec(sv):
        pk["Ts"] = max(pk["Ts"], sv.surface_temperature())
        pk["Tif"] = max(pk["Tif"], float(sv.T[interface_index(sv)]))
        pk["melt"] = max(pk["melt"], sv.melt_depth())
        last.update(sv)

    s.run(k * P + 50e-6, dt0=1e-13, dT_target=5.0, record=rec)
    s.run((k + 1) * P, dt0=1e-9, dT_target=5.0, record=rec)
    rows.append([k + 1, pre_Ts, pk["Ts"], pk["Tif"], pre_back, s.mass_evaporated / 19250 * 1e9,
                 pk["melt"] * 1e6, (s.mass_evaporated - abl0) / 19250 * 1e9])
    if (k + 1) % 10 == 0 or k == n - 1:
        s.ledger.stored_final = s.stored_energy()
        r = np.array(rows)
        np.savetxt(d / f"{name}_per_shot.csv", r, delimiter=",",
                   header="shot,preshot_surface_T_K,peak_surface_T_K,peak_interface_T_K,preshot_back_T_K,cum_ablated_nm,"
                          "melt_depth_um,ablated_nm_this_shot", comments="")
        # convergence: change of the pre-shot surface T over the last 10 % of shots
        m = max(len(r) // 10, 2)
        slope = (r[-1, 1] - r[-m, 1]) / (m - 1)
        converged = len(r) >= CONV_MIN and abs(slope) < CONV_TOL
        ks, sks = k_status(opt)
        out = {"scenario": name, "n_shots": len(r), "rep_Hz": scen.rep_Hz, "options": {kk: v for kk, v in vars(opt).items() if kk != "extra"},
               "k_irr": opt.k_irr, "skin_um": opt.skin_um, "skin_factor": opt.skin_factor, "T_init": opt.T_init,
               "k_status": ks, "skin_status": sks,
               "first_shot_peak_K": r[0, 2], "last_shot_peak_K": r[-1, 2], "last_preshot_K": r[-1, 1],
               "preshot_rise_per_shot_last10pct_K": slope, "converged": bool(converged),
               "convergence_criterion": f"pre-shot rise < {CONV_TOL} K/shot over the last 10 % of shots (min {CONV_MIN} shots)",
               "max_n_shots": n, "max_interface_T_K": float(r[:, 3].max()), "last_shot_interface_T_K": r[-1, 3],
               "last_shot_melt_depth_um": r[-1, 6], "max_melt_depth_um": float(r[:, 6].max()),
               "last_shot_ablated_nm": r[-1, 7], "cum_ablated_nm": r[-1, 5],
               "energy_rel_residual": float(s.ledger.residual()[1]), "wall_s_this_session": time.time() - t0}
        out["newton_this_session"] = s.newton_stats()  # counters added 16 Oct 2026; cumulative since this process started
        out["diagnostics_last_shot"] = last.results()
        save_diagnostics(d, name, {"_maxT_profile": last.profile_rows(), "diagnostics": out["diagnostics_last_shot"]})
        (d / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        save_checkpoint()
        print(f"{name} shot {k + 1}: pre {r[-1, 1]:.1f} K, peak {r[-1, 2]:.0f} K, interface {r[-1, 3]:.0f} K, "
              f"melt {r[-1, 6]:.3f} um, dT_pre/shot {slope:.3f} K{' CONVERGED' if converged else ''}, "
              f"{time.time() - t0:.0f}s", flush=True)
        if converge and converged:
            break

r = np.array(rows)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(r[:, 0], r[:, 1], label="surface, just before shot")
ax[0].plot(r[:, 0], r[:, 4], label="steel back face, before shot")
ax[0].set(xlabel="shot number", ylabel="temperature (K)", title=f"{name}: pre-shot temperatures ({scen.rep_Hz:g} Hz)")
ax[0].legend()
ax[1].plot(r[:, 0], r[:, 2], label="peak W surface")
ax[1].plot(r[:, 0], r[:, 3], label="peak W/steel interface")
ax[1].axhline(925, ls="--", c="k", lw=0.8)
ax[1].text(1, 940, "ODS-RAFM limit 925 K", fontsize=8)
ax[1].set(xlabel="shot number", ylabel="temperature (K)", title="per-shot peaks")
ax[1].legend()
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(d / f"{name}_multishot.{ext}", dpi=150)
