"""Alloy study driver (2026-10-13): single-shot matrix of tungsten grades with recrystallization
and DBTT diagnostics, a W_pure regression check, summary tables and figures.  Writes only under
results/alloys/ (default).  The production scripts and runs/ are untouched.

Usage:
  python scripts/run_alloy_study.py single [--out=DIR] [variants=a,b] [scen=Demo,Prototype] [starts=cold,hot]
  python scripts/run_alloy_study.py summary [--out=DIR] [--he_factor=F]
Multi-shot runs use the unchanged production script with the new options, e.g.
  python scripts/run_w_multi.py Demo 600 w_material=W_ODS_La2O3 alloy_k_proxy=0.8 --converge
         --out=results/alloys/multi/Demo_W_ODS_La2O3_p0.8

Variants (k factors on the Tolias 2017 pure-W conductivity):
  W_pure             baseline (factor 1)
  W_K                sourced factor 1.0 (Nogami 2021 / Ma 2022a / Fukuda 2018): thermally identical to W_pure
  W_K_k0.9, W_K_k1.1 sensitivity +/-10 % via the existing k_scale option
  W_ODS_La2O3_p1.0/0.9/0.8  PROXY conductivity factors (no sourced k(T) for W-1%La2O3)
Hot start: T_init = converged pre-shot surface temperature of the pure-W multi-shot baselines
(runs/w_multi/baseline: Demo 921.1 K after 600 shots at 10 Hz, Prototype 609.5 K after 100 shots at 1 Hz).
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ife_wall import alloys  # noqa: E402
from ife_wall.diagnostics import RxSpec, ShotRecorder  # noqa: E402
from ife_wall.scenarios import SCENARIOS, Options, converged_preshot, w_case  # noqa: E402
from run_w_single import RHO_W_REF, git_hash, k_status, save_diagnostics  # noqa: E402

VARIANTS = {
    "W_pure": dict(w_material="W_pure"),
    "W_K": dict(w_material="W_K"),
    "W_K_k0.9": dict(w_material="W_K", k_scale=0.9),
    "W_K_k1.1": dict(w_material="W_K", k_scale=1.1),
    "W_ODS_La2O3_p1.0": dict(w_material="W_ODS_La2O3", alloy_k_proxy=1.0),
    "W_ODS_La2O3_p0.9": dict(w_material="W_ODS_La2O3", alloy_k_proxy=0.9),
    "W_ODS_La2O3_p0.8": dict(w_material="W_ODS_La2O3", alloy_k_proxy=0.8),
}
HOT = {n: converged_preshot(n, legacy=v) for n, v in {"Demo": 921.1, "Prototype": 609.5}.items()}
Q_SET = (322e3, 336e3, 350e3)  # Shah 2021 range and its mid value
Q_MID_KEY = "336000"
T_ONSET_S = 3600.0  # the sourced T_rx values are 1 h isochronal-anneal onsets (Nogami 2021)
BASELINE = ROOT / "results" / "baseline_pureW" / "w_single"
REG_KEYS = ("peak_surface_T_K", "t_peak_us", "melt_depth_um", "ablated_nm_per_shot", "energy_rel_residual",
            "pre_shot2_surface_T_K", "energy_rel_residual_full_cycle", "diagnostics", "ledger_J_m2")


def rx_spec(key):
    thr = alloys.recrystallization_thresholds(key)
    return RxSpec(thresholds=thr, Q=Q_SET, T_ref=alloys.GRADES[key].T_rx.value)


def single_rx(name, opt, rx):
    """run_w_single.single() with the recrystallization bookkeeping; the solver calls are identical."""
    scen = SCENARIOS[name]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = w_case(scen, opt)
        f_alloy = alloys.conductivity_factor(opt)
    hist = []
    peak = {"Ts": 0.0, "t": 0.0, "melt": 0.0}
    cycle = 1.0 / scen.rep_Hz if scen.rep_Hz else None
    recd = ShotRecorder(s, 0.0, cycle, rx=rx)

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
           "grade": opt.w_material, "thermal_status": alloys.thermal_status(opt),
           "k_factor_effective": f_alloy * opt.k_scale,
           "warnings": sorted({str(x.message) for x in w}),
           "k_irr": opt.k_irr, "skin_um": opt.skin_um, "skin_factor": opt.skin_factor, "T_init": opt.T_init,
           "k_status": ks, "skin_status": sks,
           "peak_surface_T_K": peak["Ts"], "t_peak_us": peak["t"] * 1e6, "melt_depth_um": peak["melt"] * 1e6,
           "ablated_nm_per_shot": abl_nm, "above_6000K_extrapolated": peak["Ts"] > 6000,
           "ledger_J_m2": {"incident": s.ledger.incident, "deposited": s.ledger.deposited,
                           "not_absorbed": s.ledger.passed_through, "vaporized": s.ledger.vaporized}}
    s.ledger.stored_final = s.stored_energy()
    out["energy_rel_residual"] = float(s.ledger.residual()[1])
    if scen.rep_Hz:
        s.run(1.0 / scen.rep_Hz, dt0=1e-9, dT_target=5.0, record=rec)
        out["pre_shot2_surface_T_K"] = s.surface_temperature()
        s.ledger.stored_final = s.stored_energy()
        out["energy_rel_residual_full_cycle"] = float(s.ledger.residual()[1])
    out["diagnostics"] = recd.results()
    out["rx_spec"] = {"thresholds_K": list(rx.thresholds), "Q_J_per_mol": list(rx.Q), "T_ref_K": rx.T_ref}
    out["_maxT_profile"] = recd.profile_rows()
    out["_rx_profile"] = (recd.rx_profile_rows(), recd.rx_profile_header())
    out["wall_s"] = time.time() - t0
    return out, np.array(hist)


def run_single(outdir, variants, scens, starts):
    d = outdir / "single"
    for label in variants:
        for name in scens:
            for start in starts:
                if start == "hot" and name not in HOT:
                    continue
                extra = {"T_init": HOT[name]} if start == "hot" else {}
                opt = Options(**VARIANTS[label], **extra)
                sub = d / label
                sub.mkdir(parents=True, exist_ok=True)
                stem = f"{name}_{start}"
                if (sub / f"{stem}.json").exists():
                    print(f"skip {label}/{stem} (exists)", flush=True)
                    continue
                out, h = single_rx(name, opt, rx_spec(opt.w_material))
                out["git"], out["run_date"], out["variant"], out["start"] = git_hash(), time.strftime("%Y-%m-%d %H:%M"), label, start
                prof, hdr = out.pop("_rx_profile")
                np.savetxt(sub / f"{stem}_rx_profile.csv", prof, delimiter=",", header=hdr, comments="")
                save_diagnostics(sub, stem, out)
                (sub / f"{stem}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
                np.savetxt(sub / f"{stem}_history.csv", h, delimiter=",", header="t_s,Ts_K,evaporated_kg_m2", comments="")
                rx = out["diagnostics"]["recrystallization"]
                key_mid = f"{rx['thresholds_K'][1]:g}"
                t_above = rx["surface"]["time_above_s"][key_mid] * 1e6
                t_eq = rx["surface"]["t_eq_s"][Q_MID_KEY]
                print(f"{label:18s} {stem:18s} peak {out['peak_surface_T_K']:.1f} K  melt {out['melt_depth_um']:.3f} um  "
                      f"d1300 {out['diagnostics']['d_1300_um']:.2f} um  t>Trx {t_above:.3f} us  t_eq {t_eq:.3e} s  "
                      f"{out['wall_s']:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------ summary
def shots_to_onset(t_eq_per_shot, Q, T_ref, dT_ref=0.0):
    """Shots until the Arrhenius-equivalent time at T_ref + dT_ref reaches the 1 h onset."""
    if t_eq_per_shot <= 0:
        return np.inf
    fac = np.exp(Q / alloys.R_GAS * (1.0 / (T_ref + dT_ref) - 1.0 / T_ref))  # t_eq(T_ref') = fac * t_eq(T_ref)
    return T_ONSET_S / (t_eq_per_shot * fac)


def load_single(outdir):
    res = {}
    for f in sorted((outdir / "single").glob("*/*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        res[(j["variant"], j["scenario"], j["start"])] = j
    return res


def _strip_rx(d):
    return {k: v for k, v in d.items() if k != "recrystallization"}


def regression(res, outdir):
    rep = {"reference": str(BASELINE.relative_to(ROOT)), "keys": list(REG_KEYS),
           "csv_files": ["history", "maxT_profile", "He_layer"], "scenarios": {}}
    ok_all = True
    for name in SCENARIOS:
        r = res.get(("W_pure", name, "cold"))
        ref_f = BASELINE / f"{name}.json"
        if r is None or not ref_f.exists():
            rep["scenarios"][name] = "missing"
            ok_all = False
            continue
        ref = json.loads(ref_f.read_text(encoding="utf-8"))
        per = {}
        for k in REG_KEYS:
            if k in ref:
                a, b = r.get(k), ref[k]
                per[k] = (_strip_rx(a) == b) if k == "diagnostics" else (a == b)
        for suffix in ("history", "maxT_profile", "He_layer"):
            a = (outdir / "single" / "W_pure" / f"{name}_cold_{suffix}.csv").read_text(encoding="utf-8")
            b = (BASELINE / f"{name}_{suffix}.csv").read_text(encoding="utf-8")
            per[f"{suffix}_csv"] = (a == b)
        rep["scenarios"][name] = per
        ok_all &= all(per.values())
    rep["bit_identical"] = bool(ok_all)
    (outdir / "regression_W_pure.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def he_counts():
    f = ROOT / "runs" / "source_term" / "he_shot_count.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text(encoding="utf-8"))
    from ife_wall.ions import DEFAULT_ION_SPECTRA
    key = f"({DEFAULT_ION_SPECTRA})"
    return {scen: j[f"{scen} {key}"]["all He"] for scen in SCENARIOS if f"{scen} {key}" in j}


def summary(outdir, he_factor=None):
    res = load_single(outdir)
    reg = regression(res, outdir)
    rows = []
    for (label, name, start), j in res.items():
        d = j["diagnostics"]
        rx = d["recrystallization"]
        T_ref = rx["T_ref_K"]
        thr = rx["thresholds_K"]
        k_lo, k_mid, k_hi = (f"{t:g}" for t in thr)
        teq = rx["surface"]["t_eq_s"]
        t1um = next(x for x in rx["He_layer"] if x["depth_um"] == 1.0)
        he1um = next(x for x in d["He_layer"] if x["depth_um"] == 1.0)
        row = {"variant": label, "grade": j["grade"], "scenario": name, "start": start, "T_init_K": j["T_init_K"],
               "k_factor": j["k_factor_effective"], "thermal_status": j["thermal_status"],
               "peak_surface_T_K": j["peak_surface_T_K"], "t_peak_us": j["t_peak_us"], "melt_depth_um": j["melt_depth_um"],
               "ablated_nm_per_shot": j["ablated_nm_per_shot"], "pre_shot2_surface_T_K": j.get("pre_shot2_surface_T_K"),
               "d_1300_um": d["d_1300_um"], "d_1800_um": d["d_1800_um"], "T_rx_K": T_ref,
               "time_above_Trx_surface_us": rx["surface"]["time_above_s"][k_mid] * 1e6,
               "time_above_Trx_m100_surface_us": rx["surface"]["time_above_s"][k_lo] * 1e6,
               "time_above_Trx_p100_surface_us": rx["surface"]["time_above_s"][k_hi] * 1e6,
               "time_above_Trx_1um_us": t1um["time_above_s"][k_mid] * 1e6,
               "depth_above_Trx_um": rx["depth_any_time_above_um"][k_mid],
               "depth_above_Trx_m100_um": rx["depth_any_time_above_um"][k_lo],
               "depth_above_Trx_p100_um": rx["depth_any_time_above_um"][k_hi],
               "t_eq_surface_s_Q336": teq[Q_MID_KEY], "t_eq_1um_s_Q336": t1um["t_eq_s"][Q_MID_KEY],
               "shots_to_onset_surface": shots_to_onset(teq[Q_MID_KEY], 336e3, T_ref),
               "shots_to_onset_surface_Q322": shots_to_onset(teq["322000"], 322e3, T_ref),
               "shots_to_onset_surface_Q350": shots_to_onset(teq["350000"], 350e3, T_ref),
               "shots_to_onset_surface_Trx_m100": shots_to_onset(teq[Q_MID_KEY], 336e3, T_ref, -100.0),
               "shots_to_onset_surface_Trx_p100": shots_to_onset(teq[Q_MID_KEY], 336e3, T_ref, +100.0),
               "shots_to_onset_1um": shots_to_onset(t1um["t_eq_s"][Q_MID_KEY], 336e3, T_ref),
               "He_1um_time_above_1000K_us": he1um["time_above_1000K_us"],
               "elastic_stress_proxy_MPa": d["elastic_stress_proxy_MPa"],
               "pre_shot_T_K": j["T_init_K"], "dbtt_regime": alloys.dbtt_regime(j["T_init_K"], j["grade"]),
               "wall_s": j["wall_s"]}
        # recrystallized depth after N shots: deepest grid point with N * t_eq(depth) >= 1 h (Q mid column)
        prof = np.loadtxt(outdir / "single" / label / f"{name}_{start}_rx_profile.csv", delimiter=",", skiprows=1)
        teq_prof = prof[:, 1 + len(thr) + 1]
        for N in (1e4, 1e5, 1e6, 1e7):
            idx = np.nonzero(N * teq_prof >= T_ONSET_S)[0]
            row[f"rx_depth_um_after_{N:.0e}_shots"] = float(prof[idx[-1], 0]) if len(idx) else 0.0
        rows.append(row)
    multi = {}
    mdir = outdir / "multi"
    for f in (sorted(mdir.glob("*/*.json")) if mdir.exists() else []):
        j = json.loads(f.read_text(encoding="utf-8"))
        multi[f.parent.name] = {k: j.get(k) for k in ("scenario", "n_shots", "converged", "last_preshot_K", "last_shot_peak_K",
                                                      "max_interface_T_K", "max_melt_depth_um", "cum_ablated_nm",
                                                      "preshot_rise_per_shot_last10pct_K", "energy_rel_residual", "options")}
    he = he_counts()
    sbs = side_by_side(rows, he)
    out = {"git": git_hash(), "date": time.strftime("%Y-%m-%d %H:%M"), "regression_W_pure": reg, "single": rows, "multi": multi,
           "he_shot_counts_pure_W": he, "he_threshold_factor": he_factor, "side_by_side": sbs, "li_doc_source": LI_DOC_SOURCE,
           "notes": ["thermal results of W_K equal those of W_pure by construction (sourced k factor 1.0)",
                     "W_ODS_La2O3 conductivity is a PROXY band, not a measurement",
                     "shots_to_onset = 3600 s / Arrhenius-equivalent time per shot at T_rx (isokinetic, single Q, no incubation)",
                     "dbtt_regime compares the pre-shot (inter-shot minimum) temperature with sourced DBTT values; it is not a fracture model"]}
    (outdir / "summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    write_md(outdir, out)
    figures(outdir, res, rows)
    figures_extra(outdir, rows, he, sbs)
    print(json.dumps(reg, indent=1))
    print(f"summary: {len(rows)} single runs, {len(multi)} multi runs -> {outdir / 'summary.md'}")


LI_DOC = {  # Li film results of the full-results document (revision of 5-6 Oct 2026: Table 11 and the summary table)
    "Experimental": {"min_um_single": 7.0, "evap_kg_m2": 6.8e-7},
    "Prototype": {"min_um_single": 12.0, "evap_kg_m2": 7.9e-6},
    "Demo": {"min_um_single": 31.0, "evap_kg_m2": 7.1e-4},
}
LI_DOC_SOURCE = "the lithium results of the paper (Table 5) (minimum film for steel < 925 K, single shot on 600 K steel; thick-film evaporation)"
W_LIMIT_INTERFACE_K = 925.0


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def side_by_side(rows, he):
    """W armor (this task) beside the Li film results (document and runs/li_multi), per scenario."""
    sput = _load_json(ROOT / "runs" / "sputtering" / "sputtering.json")
    li_multi = _load_json(ROOT / "runs" / "li_multi" / "summary.json")

    def pick(variant, scen, start):
        return next((r for r in rows if r["variant"] == variant and r["scenario"] == scen and r["start"] == start), None)

    out = {}
    for scen in SCENARIOS:
        conv = "hot" if scen in HOT else "cold"
        wp, wk = pick("W_pure", scen, conv), pick("W_K", scen, conv)
        o10, o08 = pick("W_ODS_La2O3_p1.0", scen, conv), pick("W_ODS_La2O3_p0.8", scen, conv)
        wm = _load_json(ROOT / "runs" / "w_multi" / "baseline" / f"{scen}.json")
        sp = sput["scenarios"][scen]["sourced_nm_per_shot"] if sput else None
        ev = wp["ablated_nm_per_shot"] if wp else None
        d = {"scenario": scen, "W_state": "converged inter-shot state" if conv == "hot" else "first shot",
             "W_peak_surface_K_pure": wp["peak_surface_T_K"] if wp else None,
             "W_peak_surface_K_ODS_p0.8": o08["peak_surface_T_K"] if o08 else None,
             "W_melt_um_max": max(x["melt_depth_um"] for x in (wp, wk, o10, o08) if x) if wp else None,
             "W_evap_nm_per_shot_pure": ev, "W_evap_nm_per_shot_ODS_p0.8": o08["ablated_nm_per_shot"] if o08 else None,
             "W_sputter_nm_per_shot": sp,
             "W_shots_1mm_evap_plus_sputter_pure": (1e6 / (ev + sp)) if (ev is not None and sp) else None,
             "W_He_shots_1e18": he[scen]["shots_1e18"] if he else None, "W_He_shots_1e19": he[scen]["shots_1e19"] if he else None,
             "W_rx_onset_shots_pure": wp["shots_to_onset_surface"] if wp else None,
             "W_rx_onset_shots_WK": wk["shots_to_onset_surface"] if wk else None,
             "W_rx_onset_shots_ODS_band": [o08["shots_to_onset_surface"], o10["shots_to_onset_surface"]] if (o08 and o10) else None,
             "W_interface_max_K": wm["max_interface_T_K"] if wm else None,
             "W_interface_vs_925K": (f"{wm['max_interface_T_K'] - W_LIMIT_INTERFACE_K:+.0f} K" if wm else "single shot"),
             "Li_min_film_um_single": LI_DOC[scen]["min_um_single"],
             "Li_min_film_um_multishot": li_multi[scen]["min_delta_um_multishot"] if (li_multi and scen in li_multi) else None,
             "Li_evap_kg_m2_per_shot": LI_DOC[scen]["evap_kg_m2"]}
        out[scen] = d
    return out


def figures_extra(outdir, rows, he, sbs):
    fd = outdir / "figures"
    labels = ["W_pure", "W_K", "W_K_k0.9", "W_K_k1.1", "W_ODS_La2O3_p1.0", "W_ODS_La2O3_p0.9", "W_ODS_La2O3_p0.8"]

    def val(variant, scen, start, key):
        r = next((r for r in rows if r["variant"] == variant and r["scenario"] == scen and r["start"] == start), None)
        return r[key] if r else np.nan

    # (a) ablation per shot
    sput = _load_json(ROOT / "runs" / "sputtering" / "sputtering.json")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(labels))
    for i, (scen, start, c) in enumerate((("Demo", "cold", "tab:orange"), ("Demo", "hot", "tab:red"), ("Prototype", "hot", "tab:blue"))):
        ax.bar(x + (i - 1) * 0.27, [val(v, scen, start, "ablated_nm_per_shot") for v in labels], 0.27, color=c,
               label=f"evaporation, {scen}, {start} start")
    if sput:
        for scen, c in (("Demo", "tab:red"), ("Prototype", "tab:blue")):
            ax.axhline(sput["scenarios"][scen]["sourced_nm_per_shot"], c=c, ls="--", lw=0.9,
                       label=f"sputtering, {scen} (Eckstein yields, grade-independent)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="thickness lost per shot (nm)", title="W ablation per shot: evaporation (model) and sputtering (published yields)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"fig_alloy_ablation_per_shot.{ext}", dpi=150)
    plt.close(fig)

    # (b) shot-count indicators, Demo
    d = sbs.get("Demo")
    if d and he:
        names, vals, lo, hi, cols = [], [], [], [], []

        def add(name, v, vlo=None, vhi=None, c="0.4"):
            if v is None or not np.isfinite(v):
                return
            names.append(name)
            vals.append(v)
            lo.append(v - vlo if vlo is not None else 0.0)
            hi.append(vhi - v if vhi is not None else 0.0)
            cols.append(c)

        add("retained He reaches 1e18 m⁻² (all grades)", d["W_He_shots_1e18"], c="tab:green")
        add("retained He reaches 1e19 m⁻² (all grades)", d["W_He_shots_1e19"], c="tab:green")
        for v, lab, c in (("W_pure", "W_pure", "k"), ("W_K", "W_K", "tab:blue"), ("W_ODS_La2O3_p1.0", "W_ODS_La2O3 (proxy k 1.0)", "tab:red"),
                          ("W_ODS_La2O3_p0.8", "W_ODS_La2O3 (proxy k 0.8)", "tab:red")):
            n0 = val(v, "Demo", "hot", "shots_to_onset_surface")
            band = [val(v, "Demo", "hot", k) for k in ("shots_to_onset_surface_Q322", "shots_to_onset_surface_Q350",
                                                      "shots_to_onset_surface_Trx_m100", "shots_to_onset_surface_Trx_p100")]
            add(f"recrystallization onset at the surface, {lab}", n0, min(band), max(band), c)
        add("1 mm consumed by evaporation + sputtering (W_pure)", d["W_shots_1mm_evap_plus_sputter_pure"], c="0.5")
        fig, ax = plt.subplots(figsize=(9, 4.6))
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=[lo, hi], color=cols, alpha=0.8, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set(xlabel="shots (10 Hz: 1e4 shots = 17 min, 1e6 = 28 h)",
               title="Demo: shot-count indicators (none is a computed life; bars = Q 322–350 kJ/mol and T_rx ±100 K)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fd / f"fig_alloy_shots_to_failure_Demo.{ext}", dpi=150)
        plt.close(fig)

    # (c) W vs Li side by side
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    peaks = [val(v, "Demo", "hot", "peak_surface_T_K") for v in labels]
    ax[0].bar(x, peaks, color=["k", "tab:blue", "tab:blue", "tab:blue", "tab:red", "tab:red", "tab:red"], alpha=0.8)
    ax[0].axhline(3695, ls="--", c="k", lw=0.8)
    ax[0].text(0.1, 3705, "W melting 3695 K", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax[0].set(ylim=(2800, 3800), ylabel="peak W surface temperature (K)", title="W, Demo, from the converged 921 K state")
    scens = ["Experimental", "Prototype", "Demo"]
    xs = np.arange(3)
    for i, (key, lab, c) in enumerate((("W_rx_onset_shots_pure", "recrystallization onset, W_pure", "k"),
                                       ("W_rx_onset_shots_WK", "recrystallization onset, W_K", "tab:blue"))):
        ax[1].bar(xs + (i - 1) * 0.22, [sbs[s][key] or np.nan for s in scens], 0.22, color=c, alpha=0.8, label=lab)
    ax[1].bar(xs + 0.22, [sbs[s]["W_rx_onset_shots_ODS_band"][1] if sbs[s]["W_rx_onset_shots_ODS_band"] else np.nan for s in scens], 0.22,
              color="tab:red", alpha=0.8, label="recrystallization onset, W_ODS_La2O3 (proxy k 1.0; k 0.8 lower)")
    if he:
        ax[1].plot(xs, [he[s]["shots_1e18"] for s in scens], "v", c="tab:green", label="He 1e18 m⁻² (all grades)")
        ax[1].plot(xs, [he[s]["shots_1e19"] for s in scens], "^", c="tab:green", label="He 1e19 m⁻² (all grades)")
    ax[1].set_yscale("log")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(scens, fontsize=8)
    ax[1].set(ylim=(1, 1e21), ylabel="shots", title="W shot-count indicators (not lives)")
    ax[1].legend(fontsize=6.5, loc="upper left")
    w = 0.35
    ax[2].bar(xs - w / 2, [sbs[s]["Li_min_film_um_single"] for s in scens], w, color="tab:purple", alpha=0.6, label="single shot (document)")
    ax[2].bar(xs + w / 2, [sbs[s]["Li_min_film_um_multishot"] or np.nan for s in scens], w, color="tab:purple", alpha=1.0,
              label="multi-shot, steel carried forward (6 Oct 2026)")
    for i, s in enumerate(scens):
        ax[2].text(i, 2, f"evap. {sbs[s]['Li_evap_kg_m2_per_shot']:.1e} kg/m² per shot", ha="center", fontsize=7, rotation=90, color="w")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(scens, fontsize=8)
    ax[2].set(ylabel="minimum Li film thickness for steel < 925 K (µm)", title="Li film (published results, unchanged by this task)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"fig_W_vs_Li_side_by_side.{ext}", dpi=150)
    plt.close(fig)


def fmt_n(x):
    if x is None or not np.isfinite(x):
        return "inf"
    return f"{x:.2e}" if x >= 1e4 else f"{x:.0f}"


def write_md(outdir, out):
    reg_word = "bit-identical" if out["regression_W_pure"]["bit_identical"] else "DIFFERENCES"
    L = [f"# Alloy study: summary ({out['date']}, commit {out['git']})", "",
         f"W_pure regression against results/baseline_pureW: **{reg_word}** "
         f"(keys: {', '.join(out['regression_W_pure']['keys'])}; history, max-T profile and He-layer CSVs).", ""]
    for start in ("cold", "hot"):
        L += [f"## Single shots, {start} start", "",
              "| variant | scenario | T_init (K) | k factor | status | peak surface (K) | melt (µm) | evap. (nm/shot) | d(1300 K) (µm) | T_rx (K) | "
              "time > T_rx, surface (µs) [−100 K / +100 K] | depth > T_rx (µm) | t_eq/shot at T_rx (s) | "
              "shots to 1 h onset, surface [Q 322 / 350 kJ; T_rx −100 / +100 K] | shots to onset, 1 µm | rx depth after 1e5 / 1e6 / 1e7 shots (µm) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in out["single"]:
            if r["start"] != start:
                continue
            L.append(f"| {r['variant']} | {r['scenario']} | {r['T_init_K']:.1f} | {r['k_factor']:.2f} | {r['thermal_status'].split(' (')[0]} | "
                     f"{r['peak_surface_T_K']:.1f} | {r['melt_depth_um']:.3f} | {r['ablated_nm_per_shot']:.2e} | {r['d_1300_um']:.2f} | {r['T_rx_K']:.0f} | "
                     f"{r['time_above_Trx_surface_us']:.3f} [{r['time_above_Trx_m100_surface_us']:.3f} / {r['time_above_Trx_p100_surface_us']:.3f}] | "
                     f"{r['depth_above_Trx_um']:.2f} | {r['t_eq_surface_s_Q336']:.2e} | {fmt_n(r['shots_to_onset_surface'])} "
                     f"[{fmt_n(r['shots_to_onset_surface_Q322'])} / {fmt_n(r['shots_to_onset_surface_Q350'])}; "
                     f"{fmt_n(r['shots_to_onset_surface_Trx_m100'])} / {fmt_n(r['shots_to_onset_surface_Trx_p100'])}] | "
                     f"{fmt_n(r['shots_to_onset_1um'])} | {r['rx_depth_um_after_1e+05_shots']:.2f} / {r['rx_depth_um_after_1e+06_shots']:.2f} / "
                     f"{r['rx_depth_um_after_1e+07_shots']:.2f} |")
        L.append("")
    keys = sorted({k for r in out["single"] for k in r["dbtt_regime"]})
    L += ["## Inter-shot (pre-shot) temperature against the sourced DBTT values", "",
          "| grade | scenario | start | pre-shot T (K) | " + " | ".join(keys) + " |", "|---|---|---|---|" + "---|" * len(keys)]
    seen = set()
    for r in out["single"]:
        sig = (r["grade"], r["scenario"], r["start"])
        if sig in seen:
            continue
        seen.add(sig)
        L.append(f"| {r['grade']} | {r['scenario']} | {r['start']} | {r['pre_shot_T_K']:.1f} | "
                 + " | ".join(r["dbtt_regime"].get(k, "–") for k in keys) + " |")
    L.append("")
    if out["multi"]:
        L += ["## Multi-shot runs (production script `run_w_multi.py` with the alloy options)", "",
              "| run | scenario | shots | converged | last pre-shot surface (K) | last peak (K) | max interface (K) | max melt (µm) | cum. evaporation (nm) | residual |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, m in out["multi"].items():
            L.append(f"| {k} | {m['scenario']} | {m['n_shots']} | {m['converged']} | {m['last_preshot_K']:.1f} | {m['last_shot_peak_K']:.0f} | "
                     f"{m['max_interface_T_K']:.0f} | {m['max_melt_depth_um']:.3f} | {m['cum_ablated_nm']:.2e} | {m['energy_rel_residual']:.1e} |")
        L.append("")
    if out["he_shot_counts_pure_W"]:
        f = out["he_threshold_factor"]
        f_word = f"{f}" if f else "not set: option off"
        from ife_wall.ions import DEFAULT_ION_SPECTRA
        L += [f"## Helium damage-fluence indicator (existing machinery, `scripts/he_shot_count.py`, all He, {DEFAULT_ION_SPECTRA} spectra)", "",
              "Shots for the retained He fluence to reach the GA2020 pulsed-damage range 1e18–1e19 m⁻². The alloy option `he_threshold_factor` "
              "multiplies the threshold fluence; **no sourced value exists for W-K or ODS-W**, so the 'on' columns are only the scaling rule "
              f"(factor {f_word}).", "",
              "| scenario | retained He per shot (m⁻²) | shots to 1e18 (off) | shots to 1e19 (off) | shots to 1e18 (on) | shots to 1e19 (on) |",
              "|---|---|---|---|---|---|"]
        for scen, v in out["he_shot_counts_pure_W"].items():
            on18 = f"{v['shots_1e18'] * f:.3g}" if f else "–"
            on19 = f"{v['shots_1e19'] * f:.3g}" if f else "–"
            L.append(f"| {scen} | {v['retained_per_m2']:.3g} | {v['shots_1e18']:.3g} | {v['shots_1e19']:.3g} | {on18} | {on19} |")
        L.append("")
    sbs = out.get("side_by_side")
    if sbs:
        L += ["## W armor and Li film side by side", "",
              f"W rows: this task (Prototype and Demo from the converged inter-shot state of the pure-W multi-shot baseline; Experimental first shot). "
              f"Li rows: {out['li_doc_source']}; multi-shot minima from runs/li_multi/summary.json (6 Oct 2026). The alloy option changes only the W rows that name a grade.", "",
              "| scenario | W peak surface T, pure / ODS proxy 0.8 (K) | W melting (any grade) | W evaporation per shot, pure / ODS 0.8 (nm) | W sputtering per shot (nm, grade-independent) | "
              "W shots to consume 1 mm (evap.+sputter) | W shots to He 1e18 / 1e19 m⁻² (all grades) | W shots to recrystallization onset at the surface: pure / W-K / ODS (k 0.8–1.0) | "
              "W/steel interface max vs 925 K (pure-W baseline) | Li minimum film, single / multi-shot (µm) | Li evaporated per shot (kg/m²) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for scen, d in sbs.items():
            band = d["W_rx_onset_shots_ODS_band"]
            L.append(f"| {scen} ({d['W_state']}) | {d['W_peak_surface_K_pure']:.0f} / {d['W_peak_surface_K_ODS_p0.8']:.0f} | "
                     f"{'no' if (d['W_melt_um_max'] or 0) == 0 else f'{d['W_melt_um_max']:.2f} µm'} | "
                     f"{d['W_evap_nm_per_shot_pure']:.1e} / {d['W_evap_nm_per_shot_ODS_p0.8']:.1e} | "
                     f"{d['W_sputter_nm_per_shot']:.1e} | {fmt_n(d['W_shots_1mm_evap_plus_sputter_pure'])} | "
                     f"{d['W_He_shots_1e18']:.0f} / {d['W_He_shots_1e19']:.0f} | {fmt_n(d['W_rx_onset_shots_pure'])} / {fmt_n(d['W_rx_onset_shots_WK'])} / "
                     f"{fmt_n(band[0]) if band else '–'}–{fmt_n(band[1]) if band else '–'} | {d['W_interface_vs_925K']} | "
                     f"{d['Li_min_film_um_single']:.0f} / {'– (single shots only)' if d['Li_min_film_um_multishot'] is None else f'{d['Li_min_film_um_multishot']:.1f}'} | "
                     f"{d['Li_evap_kg_m2_per_shot']:.1e} |")
        L.append("")
    L += ["## Notes", ""] + [f"- {n}" for n in out["notes"]]
    (outdir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(outdir, res, rows):
    fd = outdir / "figures"
    fd.mkdir(exist_ok=True)
    styles = {"W_pure": ("k", "-"), "W_K": ("tab:blue", "--"), "W_K_k0.9": ("tab:blue", ":"), "W_K_k1.1": ("tab:blue", "-."),
              "W_ODS_La2O3_p1.0": ("tab:red", "-"), "W_ODS_La2O3_p0.9": ("tab:red", "--"), "W_ODS_La2O3_p0.8": ("tab:red", ":")}
    for name in ("Demo", "Prototype", "Experimental"):
        for start in ("cold", "hot"):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
            any_ = False
            T_init = None
            for label, (c, ls) in styles.items():
                j = res.get((label, name, start))
                if j is None:
                    continue
                any_, T_init = True, j["T_init_K"]
                h = np.loadtxt(outdir / "single" / label / f"{name}_{start}_history.csv", delimiter=",", skiprows=1)
                ax[0].semilogx(h[:, 0] * 1e6, h[:, 1], c=c, ls=ls, lw=1.2, label=f"{label} (peak {j['peak_surface_T_K']:.0f} K)")
                p = np.loadtxt(outdir / "single" / label / f"{name}_{start}_maxT_profile.csv", delimiter=",", skiprows=1)
                ax[1].semilogx(p[:, 0], p[:, 1], c=c, ls=ls, lw=1.2, label=label)
            if not any_:
                plt.close(fig)
                continue
            for key, c in (("W_pure", "k"), ("W_K", "tab:blue"), ("W_ODS_La2O3", "tab:red")):
                T = alloys.GRADES[key].T_rx.value
                ax[1].axhline(T, c=c, lw=0.6, alpha=0.6)
                ax[1].text(2e-3, T + 15, f"T_rx {key} {T:.0f} K", fontsize=7, color=c)
            ax[0].axhline(3695, ls="--", c="k", lw=0.8)
            ax[0].set(xlim=(1e-2, 50), xlabel="time after ignition (µs)", ylabel="W surface temperature (K)",
                      title=f"{name}, {start} start (T_init {T_init:.0f} K)")
            ax[0].legend(fontsize=7)
            ax[1].set(xlim=(1e-3, 200), xlabel="depth (µm)", ylabel="maximum temperature in the shot (K)", title="max-T depth profile")
            ax[1].legend(fontsize=7)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(fd / f"fig_alloy_{name}_{start}.{ext}", dpi=150)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = list(styles)
    for i, (name, start, mk) in enumerate((("Demo", "cold", "o"), ("Demo", "hot", "s"), ("Prototype", "cold", "^"), ("Prototype", "hot", "v"))):
        xs, ys, lo, hi = [], [], [], []
        for x, label in enumerate(labels):
            r = next((r for r in rows if r["variant"] == label and r["scenario"] == name and r["start"] == start), None)
            if r is None or not np.isfinite(r["shots_to_onset_surface"]):
                continue
            n0 = r["shots_to_onset_surface"]
            xs.append(x + 0.1 * i - 0.15)
            ys.append(n0)
            lo.append(n0 - min(r["shots_to_onset_surface_Q322"], r["shots_to_onset_surface_Q350"], r["shots_to_onset_surface_Trx_m100"]))
            hi.append(max(r["shots_to_onset_surface_Q322"], r["shots_to_onset_surface_Q350"], r["shots_to_onset_surface_Trx_p100"]) - n0)
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, capsize=3, label=f"{name}, {start} start")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="shots to 1 h Arrhenius-equivalent time at T_rx (surface)",
           title="Recrystallization onset indicator (bars: Q 322–350 kJ/mol and T_rx ±100 K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"fig_alloy_shots_to_onset.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "single"
    outdir = Path(next((a.split("=", 1)[1] for a in args if a.startswith("--out=")), ROOT / "results" / "alloys"))
    outdir.mkdir(parents=True, exist_ok=True)
    kv = dict(a.split("=", 1) for a in args[1:] if "=" in a and not a.startswith("--"))
    if cmd == "single":
        run_single(outdir, kv.get("variants", ",".join(VARIANTS)).split(","), kv.get("scen", ",".join(SCENARIOS)).split(","),
                   kv.get("starts", "cold,hot").split(","))
    elif cmd == "summary":
        hf = next((float(a.split("=", 1)[1]) for a in args if a.startswith("--he_factor=")), None)
        summary(outdir, hf)
    else:
        raise SystemExit(__doc__)
