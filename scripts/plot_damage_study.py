"""Figures and tables for the damaged-conductivity study (runs H and I).

Reads runs/w_damage/single/<case>/<scenario>.json (+ _history.csv, _maxT_profile.csv) and
runs/w_damage/multi/<case>/Demo*.  Writes runs/w_damage/{fig*.png,pdf, summary.csv/.md,
depth_temperatures.csv/.md, fhf_check.csv/.md, neutron_wall_loading.json} and png copies
in docs/figures/damage/.
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.library import W_TM, tungsten  # noqa: E402

D = ROOT / "runs" / "w_damage"
FIGCOPY = ROOT / "docs" / "figures" / "damage"
FIGCOPY.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11})
K_SWEEP = (1.0, 0.85, 0.75, 0.6, 0.5, 0.4)
LIMIT = 925.0


def load(case, scen="Demo"):
    p = D / "single" / case / f"{scen}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def hist(case, scen="Demo"):
    p = D / "single" / case / f"{scen}_history.csv"
    return np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2) if p.exists() else None


def prof(case, scen="Demo"):
    p = D / "single" / case / f"{scen}_maxT_profile.csv"
    return np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2) if p.exists() else None


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(D / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    fig.savefig(FIGCOPY / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


cases = sorted(p.name for p in (D / "single").iterdir() if p.is_dir()) if (D / "single").exists() else []
status = json.loads((D / "status.json").read_text(encoding="utf-8")) if (D / "status.json").exists() else {}


def best_C(thr, start):
    """Use the self-consistency pass if one was run."""
    for c in (f"C_annealed_d{thr}_{start}_pass2", f"C_annealed_d{thr}_{start}"):
        if load(c):
            return c
    return None


he_cases = sorted(c for c in cases if c.startswith("D_Heskin"))
worst_cases = sorted(c for c in cases if c.startswith("E_worst"))

# ---------------- Figure 1: peak surface T vs k_irr ------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 5.8))
for start, c, lab in (("600K", "C0", "start from 600 K (baseline convention)"), ("hot", "C3", "hot start 921.1 K (converged pre-shot T)")):
    k, T = [], []
    for kk in K_SWEEP:
        r = load(f"A_kirr{kk}_{start}")
        if r:
            k.append(kk)
            T.append(r["peak_surface_T_K"])
    ax.plot(k, T, "o-", c=c, label=lab)
mk = {"600K": "s", "hot": "D"}
for start in ("600K", "hot"):
    for thr, col in (("1300", "C2"), ("1800", "C4")):
        cc = best_C(thr, start)
        if cc:
            r = load(cc)
            ax.plot(0.5, r["peak_surface_T_K"], mk[start], c=col, ms=8, mfc="none", mew=2,
                    label=f"annealed skin to the {thr} K depth ({r['skin_um']:.1f} µm), {'hot' if start == 'hot' else '600 K'}")
for c in he_cases:
    r = load(c)
    bulk = 0.5 if "bulk0.5" in c else 1.0
    saito = "Saito" in c
    ax.plot(bulk, r["peak_surface_T_K"], "^" if saito else "v", c="C1" if saito else "0.55", ms=9,
            label=f"He-bubble skin 5 µm, skin k × {r['skin_factor']:.3g}, bulk × {bulk}, hot"
                  + (" (Saito f_He)" if saito else " (ESTIMATED f_He)"))
for c in (x for x in cases if x.startswith("E_worst")):
    r = load(c)
    ax.plot(0.5, r["peak_surface_T_K"], "*", c="k", ms=14,
            label=f"worst credible (bound): bulk × 0.5 + He skin × {r['skin_factor']:.3g}, hot")
ax.axhline(W_TM, ls="--", c="k", lw=1)
ax.text(0.995, W_TM + 25, "W melts (3695 K)", ha="left", fontsize=9)
ax.axhspan(1300, 1800, color="0.9", zorder=0)
ax.text(0.995, 1830, "recrystallization 1300–1800 K", ha="left", fontsize=9)
ax.axvline(0.5, ls=":", c="0.4")
ax.text(0.497, 2250, "about halved after\nirradiation", fontsize=9, va="bottom", ha="left")
ax.set(xlabel="solid-W conductivity factor k_irr (1 = undamaged)", ylabel="peak W surface temperature (K)",
       title="Demo (154 MJ, 6.5 m), single shot: effect of damaged conductivity", ylim=(1200, None))
ax.invert_xaxis()
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
save(fig, "fig1_peakT_vs_kirr")

# ---------------- Figure 2: surface T vs time (hot start) -----------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2))
sel = [("A_kirr1.0_hot", "undamaged"), ("A_kirr0.5_hot", "uniform damage k × 0.5"),
       (best_C("1300", "hot"), "annealed skin (to 1300 K depth) + bulk × 0.5")]
if he_cases:
    hc = [c for c in he_cases if "bulk0.5" in c]
    if hc:
        sel.append((hc[0], f"He-bubble skin ({load(hc[0])['skin_factor']:.3g}) + bulk × 0.5"))
for c, lab in sel:
    h = hist(c) if c else None
    if h is not None:
        ax.semilogx(h[:, 0] * 1e6, h[:, 1], label=lab)
ax.axhline(W_TM, ls="--", c="k", lw=1)
ax.set(xlim=(1e-2, 1e5), xlabel="time after ignition (µs)", ylabel="W surface temperature (K)",
       title="Demo, hot start (921.1 K): surface temperature over one 100 ms cycle")
ax.legend(fontsize=9)
save(fig, "fig2_surfaceT_vs_time")

# ---------------- Figure 3: He-layer time above 1000 K ---------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sel3 = [("A_kirr1.0_hot", "undamaged, hot start"), ("A_kirr0.5_hot", "uniform k × 0.5, hot start"),
        ("A_kirr1.0_600K", "undamaged, 600 K start"), ("A_kirr0.5_600K", "uniform k × 0.5, 600 K start")]
if he_cases:
    sel3 += [(c, f"He-bubble skin {load(c)['skin_factor']:.3g}, bulk {0.5 if 'bulk0.5' in c else 1.0}") for c in he_cases]
for c, lab in sel3:
    r = load(c)
    if r:
        he = r["diagnostics"]["He_layer"]
        ax.semilogy([x["depth_um"] for x in he], [max(x["time_above_1000K_us"], 1e-3) for x in he], "o-", label=lab)
ax.set(xlabel="depth (µm)", ylabel="time above 1000 K per shot (µs)",
       title="He-implanted layer: time above 1000 K per shot (Demo)")
ax.text(0.02, 0.02, "He fuzz forms only above ~1000 K", transform=ax.transAxes, fontsize=9)
ax.legend(fontsize=7.5)
save(fig, "fig3_He_layer_time_above_1000K")

# ---------------- Figure 4: melt depth and W lost per shot vs k_irr ---------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for start, c in (("600K", "C0"), ("hot", "C3")):
    k, m, a = [], [], []
    for kk in K_SWEEP:
        r = load(f"A_kirr{kk}_{start}")
        if r:
            k.append(kk)
            m.append(r["melt_depth_um"])
            a.append(r["ablated_nm_per_shot"])
    ax[0].plot(k, m, "o-", c=c, label=start.replace("600K", "600 K start").replace("hot", "hot start"))
    ax[1].semilogy(k, a, "o-", c=c, label=start.replace("600K", "600 K start").replace("hot", "hot start"))
ax[0].set(xlabel="k_irr", ylabel="melt depth per shot (µm)", title="Demo: melt depth")
ax[1].set(xlabel="k_irr", ylabel="W evaporated per shot (nm)", title="Demo: W lost per shot (evaporation)")
for x in ax:
    x.invert_xaxis()
    x.legend()
save(fig, "fig4_melt_and_loss_vs_kirr")

# ---------------- Figure 5: multi-shot per-shot peaks (4 panels, see finalize_multishot.py) ------
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_multishot import make_fig5  # noqa: E402

make_fig5(D, FIGCOPY, np.loadtxt(ROOT / "runs/w_multi/baseline/Demo_per_shot.csv", delimiter=",", skiprows=1, ndmin=2))

# ---------------- Figure 6: max-T depth profiles --------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for c, lab, col in (("A_kirr1.0_hot", "undamaged, hot start", "C0"), ("A_kirr0.5_hot", "k × 0.5, hot start", "C3"),
                    ("A_kirr1.0_600K", "undamaged, 600 K start", "C0"), ("A_kirr0.5_600K", "k × 0.5, 600 K start", "C3")):
    p = prof(c)
    if p is not None:
        ax.semilogx(p[:, 0], p[:, 1], c=col, ls="-" if "hot" in c else "--", label=lab)
        r = load(c)["diagnostics"]
        for thr in (1300, 1800):
            ax.plot(r[f"d_{thr}_um"], thr, "o", c=col)
for thr in (1300, 1800):
    ax.axhline(thr, ls=":", c="0.5", lw=0.8)
ax.axhline(W_TM, ls="--", c="k", lw=1)
ax.set(xlabel="depth (µm)", ylabel="maximum temperature during the shot (K)", xlim=(1e-3, 200),
       title="Demo: peak temperature vs depth (dots: 1300 K and 1800 K crossings)")
ax.legend(fontsize=9)
save(fig, "fig6_maxT_depth_profiles")

# ---------------- tables ---------------------------------------------------------------------
rows = []
for c in cases:
    for scen in ("Demo", "Prototype"):
        r = load(c, scen)
        if not r:
            continue
        he = {x["depth_um"]: x["time_above_1000K_us"] for x in r["diagnostics"]["He_layer"]}
        est = []
        if r["k_status"].startswith("ESTIMATED"):
            est.append("k_irr")
        if r["skin_um"] > 0:
            est.append("skin (" + r["skin_status"].split(":")[0].split(" (")[0] + ")")
        rows.append({"case": c, "scenario": scen, "k_irr": r["k_irr"], "skin_um": round(r["skin_um"], 3),
                     "skin_factor": r["skin_factor"], "T_init_K": r["T_init_K"],
                     "peak_surface_T_K": round(r["peak_surface_T_K"], 1), "t_peak_us": round(r["t_peak_us"], 3),
                     "melt_depth_um": round(r["melt_depth_um"], 3), "W_lost_nm_per_shot": f"{r['ablated_nm_per_shot']:.3e}",
                     "preshot_T_next_shot_K": round(r.get("pre_shot2_surface_T_K", float("nan")), 1),
                     "max_interface_T_K_multishot": "",
                     "t_above_1000K_1um_us": round(he[1.0], 1), "t_above_1000K_2um_us": round(he[2.0], 1),
                     "t_above_1000K_4um_us": round(he[4.0], 1),
                     "elastic_stress_proxy_MPa": round(r["diagnostics"]["elastic_stress_proxy_MPa"]),
                     "max_gradient_K_per_um": round(r["diagnostics"]["max_near_surface_gradient_K_per_um"]),
                     "energy_rel_residual": f"{r['energy_rel_residual']:.1e}",
                     "flags": ";".join((["above_6000K_extrapolated"] if r["above_6000K_extrapolated"] else []) + (["ESTIMATED: " + ",".join(est)] if est else []))})
for tag in ("kirr0.5_conv", "Heskin_conv"):
    p = D / "multi" / tag / "Demo.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        he = {x["depth_um"]: x["time_above_1000K_us"] for x in m["diagnostics_last_shot"]["He_layer"]} if "diagnostics_last_shot" in m else {}
        rows.append({"case": f"F_multi_{tag} ({m['n_shots']} shots, last shot)", "scenario": "Demo", "k_irr": m["k_irr"],
                     "skin_um": m["skin_um"], "skin_factor": m["skin_factor"], "T_init_K": 600.0,
                     "peak_surface_T_K": round(m["last_shot_peak_K"], 1), "t_peak_us": "",
                     "melt_depth_um": round(m["last_shot_melt_depth_um"], 3), "W_lost_nm_per_shot": f"{m['last_shot_ablated_nm']:.3e}",
                     "preshot_T_next_shot_K": round(m["last_preshot_K"], 1), "max_interface_T_K_multishot": round(m["max_interface_T_K"], 1),
                     "t_above_1000K_1um_us": round(he.get(1.0, float("nan")), 1), "t_above_1000K_2um_us": round(he.get(2.0, float("nan")), 1),
                     "t_above_1000K_4um_us": round(he.get(4.0, float("nan")), 1),
                     "elastic_stress_proxy_MPa": round(m["diagnostics_last_shot"]["elastic_stress_proxy_MPa"]) if "diagnostics_last_shot" in m else "",
                     "max_gradient_K_per_um": round(m["diagnostics_last_shot"]["max_near_surface_gradient_K_per_um"]) if "diagnostics_last_shot" in m else "",
                     "energy_rel_residual": f"{m['energy_rel_residual']:.1e}",
                     "flags": f"converged={abs(m['preshot_rise_per_shot_last10pct_K']) < 0.05} (last-10% rise {m['preshot_rise_per_shot_last10pct_K']:.3f} K/shot)"})


def write_table(name, rows_):
    if not rows_:
        return
    keys = list(rows_[0].keys())
    with open(D / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows_]
    (D / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


write_table("summary", rows)

# Depth table: single-shot hot start (steady-state cycle, see run B) baseline vs uniform 0.5,
# plus the damaged multi-shot last shot
bk = []
srcs = [("A_kirr1.0_hot", "undamaged, one cycle from the converged 921.1 K state"),
        ("A_kirr0.5_hot", "uniform k × 0.5, one cycle from 921.1 K")]
for c, lab in srcs:
    r = load(c)
    if r:
        for depth, v in r["diagnostics"]["depth_table"].items():
            bk.append({"case": lab, "depth": depth, "pre_shot_K": round(v["pre_shot_K"], 1), "peak_K": round(v["peak_K"], 1),
                       "time_avg_over_100ms_cycle_K": round(v["time_avg_K"], 1) if v["time_avg_K"] else ""})
p = D / "multi" / "kirr0.5_conv" / "Demo.json"
if p.exists():
    m = json.loads(p.read_text(encoding="utf-8"))
    if "diagnostics_last_shot" in m:
        for depth, v in m["diagnostics_last_shot"]["depth_table"].items():
            bk.append({"case": f"uniform k × 0.5, multi-shot, last of {m['n_shots']} shots", "depth": depth,
                       "pre_shot_K": round(v["pre_shot_K"], 1), "peak_K": round(v["peak_K"], 1),
                       "time_avg_over_100ms_cycle_K": round(v["time_avg_K"], 1) if v["time_avg_K"] else ""})
# one extra shot from each run's final checkpoint (run_extra_shot.py): same table + melt timing
for tag, lab in (("kirr0.5_conv", "uniform k × 0.5"), ("Heskin_conv", "He skin × 0.25 + bulk × 0.5")):
    p = D / "multi" / f"{tag}_extra_shot" / "Demo_extra_shot.json"
    if p.exists():
        e = json.loads(p.read_text(encoding="utf-8"))
        for depth, v in e["diagnostics"]["depth_table"].items():
            bk.append({"case": f"{lab}, multi-shot, extra shot {e['shot_number']} from the final state", "depth": depth,
                       "pre_shot_K": round(v["pre_shot_K"], 1), "peak_K": round(v["peak_K"], 1),
                       "time_avg_over_100ms_cycle_K": round(v["time_avg_K"], 1) if v["time_avg_K"] else "",
                       "melt_duration_us": round(e["melt_duration_us"], 2) if depth == "surface" else "",
                       "max_melt_depth_um": round(e["max_melt_depth_um"], 3) if depth == "surface" else ""})
keys = list(dict.fromkeys(k for r in bk for k in r))
bk = [{k: r.get(k, "") for k in keys} for r in bk]
write_table("depth_temperatures", bk)

# ---------------- run H: heat-flux-factor check ----------------------------------------------
W = tungsten()
F_HF = 44.4e6  # W m^-2 s^0.5, GA2020 Table I (Demo)
fh = []
for start, T0, case_k1, case_k05 in (("600K", 600.0, "A_kirr1.0_600K", "A_kirr0.5_600K"), ("hot", 921.1, "A_kirr1.0_hot", "A_kirr0.5_hot")):
    for k_fac, case in ((1.0, case_k1), (0.5, case_k05)):
        r = load(case)
        if not r:
            continue
        Tm = 0.5 * (T0 + min(r["peak_surface_T_K"], W_TM))  # mean temperature over the rise
        k, rho, cp = float(W.k(np.array(Tm))) * k_fac, float(W.rho(np.array(Tm))), float(W.cp(np.array(Tm)))
        dT_an = 2 * F_HF / math.sqrt(math.pi * k * rho * cp)
        dT_code = r["peak_surface_T_K"] - T0
        fh.append({"start": start, "k_factor": k_fac, "T_eval_K": round(Tm), "k_W_mK": round(k, 1), "rho": round(rho),
                   "cp": round(cp, 1), "dT_analytic_K": round(dT_an), "dT_code_K": round(dT_code),
                   "ratio_analytic_over_code": round(dT_an / dT_code, 3),
                   "note": "analytic = surface flux on a semi-infinite solid (no melting); code deposits ions over um "
                           "and includes latent heat of melting" + ("; code case melts" if r["melt_depth_um"] > 0 else "")})
write_table("fhf_check", fh)

# ---------------- run I: neutron wall loading ------------------------------------------------
nwl = {}
for scen, (Y, R, rep) in {"Demo": (154e6, 6.5, 10.0), "Prototype": (50e6, 6.5, 1.0)}.items():
    nwl[scen] = {"neutron_wall_loading_MW_m2": 0.71 * Y * rep / (4 * math.pi * R ** 2) / 1e6,
                 "basis": "0.71 x yield x rep / (4 pi R^2); neutron share 71 % from GA2020 Sec. II"}
nwl["W_dpa_per_MWyr_m2"] = "MISSING (needs a neutronics calculation)"
nwl["context"] = ("estimate by comparison with tokamak conditions: about 0.3 dpa within days; "
                  "GA2020 Table I: HiPER Demo fatigue-limited life ~1e6 shots = 28 h")
(D / "neutron_wall_loading.json").write_text(json.dumps(nwl, indent=1), encoding="utf-8")
print(json.dumps(nwl, indent=1))
print("figures:", sorted(p.name for p in D.glob("fig*.png")))
print("summary rows:", len(rows))
