"""Production: single-shot tungsten response for the HiPER scenarios.

For each scenario: surface T(t), peak surface T and time, melt depth (max), ablated depth per
shot (nm, at 19.25 g/cm^3), energy ledger.  Runs until 50 us after ignition (all ions
deposited; surface back below ~2000 K in all cases), then continues to the next-shot time
(1/rep) for Prototype/Demo to record the pre-shot surface temperature of shot 2.

Usage: python scripts/run_w_single.py [variant=value ...]  (e.g. xray_tau=1e-10 alpha_evap=0.1)
Outputs: runs/w_single/<tag>/<scenario>.json, history CSV, and a T(t) figure.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import SCENARIOS, Options, w_case  # noqa: E402

RHO_W_REF = 19250.0


def parse_overrides(args):
    opt = Options()
    for a in args:
        k, v = a.split("=")
        cur = getattr(opt, k)
        setattr(opt, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v) if cur is not None else float(v))
    return opt


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def k_status(opt):
    """Provenance strings for the damaged-conductivity options (study of 2026-09-29)."""
    if opt.k_irr == 1.0:
        k = "undamaged (Tolias 2017)"
    elif abs(opt.k_irr - 0.5) < 1e-12:
        k = "SOURCED (conductivity about halved after irradiation: Reza et al. 2020; Saito et al. 2026)"
    else:
        k = "ESTIMATED (sensitivity)"
    if opt.skin_um <= 0:
        sk = "no separate skin"
    elif opt.skin_factor == 1.0:
        sk = "annealed skin: ASSUMPTION (defects heal each shot)"
    else:
        sk = opt.extra.get("skin_status", "ESTIMATED (helium-bubble skin factor bracketed)")
    return k, sk


def single(name, opt):
    from ife_wall.diagnostics import ShotRecorder
    scen = SCENARIOS[name]
    s = w_case(scen, opt)
    hist = []
    peak = {"Ts": 0.0, "t": 0.0, "melt": 0.0}
    cycle = 1.0 / scen.rep_Hz if scen.rep_Hz else None
    recd = ShotRecorder(s, 0.0, cycle)

    def rec(sv):
        Ts = sv.surface_temperature()
        hist.append((sv.t, Ts, sv.mass_evaporated))
        if Ts > peak["Ts"]:
            peak.update(Ts=Ts, t=sv.t)
        peak["melt"] = max(peak["melt"], sv.melt_depth())
        recd.update(sv)

    t0 = time.time()
    s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec)
    abl_nm = s.mass_evaporated / RHO_W_REF * 1e9
    ks, sks = k_status(opt)
    out = {"scenario": name, "yield_MJ": scen.yield_J / 1e6, "R_m": scen.R,
           "T_init_K": scen.T_init if opt.T_init is None else float(opt.T_init),
           "options": {k: v for k, v in vars(opt).items() if k != "extra"},
           "k_irr": opt.k_irr, "skin_um": opt.skin_um, "skin_factor": opt.skin_factor, "T_init": opt.T_init,
           "k_status": ks, "skin_status": sks,
           "peak_surface_T_K": peak["Ts"], "t_peak_us": peak["t"] * 1e6, "melt_depth_um": peak["melt"] * 1e6,
           "ablated_nm_per_shot": abl_nm, "above_6000K_extrapolated": peak["Ts"] > 6000,
           "ledger_J_m2": {"incident": s.ledger.incident, "deposited": s.ledger.deposited,
                           "not_absorbed": s.ledger.passed_through, "vaporized": s.ledger.vaporized}}
    s.ledger.stored_final = s.stored_energy()
    out["energy_rel_residual"] = float(s.ledger.residual()[1])
    out["newton_to_50us"] = s.newton_stats()  # counters added 16 Oct 2026 (skin sweep); no effect on the numerics
    if scen.rep_Hz:
        s.run(1.0 / scen.rep_Hz, dt0=1e-9, dT_target=5.0, record=rec)  # full cycle: time averages
        out["pre_shot2_surface_T_K"] = s.surface_temperature()
        s.ledger.stored_final = s.stored_energy()
        out["energy_rel_residual_full_cycle"] = float(s.ledger.residual()[1])
        out["newton_full_cycle"] = s.newton_stats()  # cumulative over both parts
    out["diagnostics"] = recd.results()
    out["_maxT_profile"] = recd.profile_rows()
    out["wall_s"] = time.time() - t0
    return out, np.array(hist)


def save_diagnostics(d, name, out):
    """<scenario>_maxT_profile.csv and <scenario>_He_layer.csv next to the JSON."""
    prof = out.pop("_maxT_profile", None)
    if prof is not None:
        np.savetxt(d / f"{name}_maxT_profile.csv", prof, delimiter=",", header="depth_um,max_T_K", comments="")
    he = out.get("diagnostics", {}).get("He_layer")
    if he:
        with open(d / f"{name}_He_layer.csv", "w", encoding="utf-8") as f:
            f.write("depth_um,time_above_1000K_us,time_above_900K_us,peak_K,time_avg_K\n")
            for r in he:
                f.write(f"{r['depth_um']},{r['time_above_1000K_us']},{r['time_above_900K_us']},{r['peak_K']},{r['time_avg_K']}\n")


if __name__ == "__main__":
    outdir = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")]
    argv = [a for a in sys.argv[1:] if not a.startswith("--out=")]
    args = [a for a in argv if "=" in a]
    names = [a for a in argv if "=" not in a] or list(SCENARIOS)
    opt = parse_overrides(args)
    tag = "baseline" if not args else "_".join(a.replace("=", "-") for a in args)
    d = Path(outdir[0]) if outdir else ROOT / "runs" / "w_single" / tag
    d.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        out, h = single(name, opt)
        out["git"] = git_hash()
        out["run_date"] = time.strftime("%Y-%m-%d %H:%M")
        save_diagnostics(d, name, out)
        (d / f"{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        np.savetxt(d / f"{name}_history.csv", h, delimiter=",", header="t_s,Ts_K,evaporated_kg_m2", comments="")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k not in ("options", "ledger_J_m2")}), flush=True)
        ax.semilogx(h[:, 0] * 1e6, h[:, 1], label=f"{name} ({out['yield_MJ']:.0f} MJ, R = {out['R_m']} m)")
    ax.axhline(3695, ls="--", c="k", lw=0.8)
    ax.text(0.012, 3730, "W melting 3695 K", fontsize=8)
    ax.axhspan(1300, 1800, color="0.9", zorder=0)
    ax.text(0.012, 1830, "recrystallization 1300–1800 K", fontsize=8)
    ax.set(xlim=(1e-2, 50), xlabel=r"time after ignition ($\mu$s)", ylabel="W surface temperature (K)",
           title=f"W armor, single shot ({tag}; no vapour shielding)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(d / f"surface_T.{ext}", dpi=150)
