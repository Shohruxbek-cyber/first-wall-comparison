"""Final numbers and figure for the damaged-conductivity multi-shot Demo runs.

Reads runs/w_damage/multi/{kirr0.5_conv,Heskin_conv}/Demo_per_shot.csv + Demo.json, the
undamaged 600-shot baseline (runs/w_multi/baseline), the single-shot worst case and, if present,
the extra-shot histories (runs/w_damage/multi/<run>_extra_shot, from run_extra_shot.py).

For each run: stopping shot, convergence, final pre-shot rise; last-shot peak surface, pre-shot
and max interface T vs the 925 K limit and vs the undamaged baseline; melt depth at the first and
last shot; W evaporated per shot, per full-power year (10 Hz, 3.15e8 shots) and years to lose
1 mm; an exponential fit of the last 200 shots when the pre-shot T has not settled; and whether
the two runs have the same pre-shot and interface temperatures.

Usage: python scripts/finalize_multishot.py [--out=DIR] [--figcopy=DIR]
  default DIR = runs/w_damage/multishot_final, figure copy in docs/figures/damage
Writes DIR/multishot_results.json, DIR/multishot_results.md, DIR/fig5_multishot_peaks.{png,pdf}.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs" / "w_damage"
LIMIT = 925.0
W_TM = 3695.0
SHOTS_PER_FPY = 3.15e8  # 10 Hz x 3.15e7 s
EVAP_LABEL = ("evaporation only, at full availability; melt-layer loss (splashing, melt motion) is not "
              "modelled and could be larger")
RUNS = {"kirr0.5_conv": "uniform k × 0.5", "Heskin_conv": "He-bubble skin (5 µm, × 0.25) + bulk × 0.5"}
BASELINE = ROOT / "runs" / "w_multi" / "baseline"
# undamaged max interface T, read from the baseline run (was the constant 934.09 K of the earlier spectra)
BASE_IF = json.loads((BASELINE / "Demo.json").read_text(encoding="utf-8"))["max_interface_T_K"]
WORST = D / "single" / "E_worst_Saito_f0.45_bulk0.5_hot" / "Demo.json"


def arg(name, default):
    v = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return Path(v[0]) if v else default


def read_csv(p):
    return np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)


def evap(nm_per_shot):
    per_fpy = nm_per_shot * SHOTS_PER_FPY
    return {"nm_per_shot": nm_per_shot, "nm_per_full_power_year": per_fpy,
            "mm_per_full_power_year": per_fpy * 1e-6, "years_to_lose_1mm": 1e6 / per_fpy if per_fpy > 0 else None,
            "label": EVAP_LABEL}


def approach(n, T_inf, A, tau):
    return T_inf - A * np.exp(-n / tau)


def fit_last(shots, T, n_fit=200):
    """Exponential approach T = T_inf - A exp(-n/tau) fitted to the last n_fit shots."""
    x, y = shots[-n_fit:], T[-n_fit:]
    p0 = (y[-1] + 10.0, max(y[-1] - y[0], 1.0) * np.exp(x[0] / 150.0), 150.0)
    p, cov = curve_fit(approach, x, y, p0=p0, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    return {"T_inf_K": float(p[0]), "T_inf_1sigma_K": float(err[0]), "tau_shots": float(p[2]),
            "rms_residual_K": float(np.sqrt(np.mean((approach(x, *p) - y) ** 2))), "fitted_shots": [int(x[0]), int(x[-1])],
            "label": "EXTRAPOLATION: exponential approach fitted to the last 200 shots, not a computed steady state"}


def analyse(tag, label):
    r = read_csv(D / "multi" / tag / "Demo_per_shot.csv")
    m = json.loads((D / "multi" / tag / "Demo.json").read_text(encoding="utf-8"))
    n = len(r)
    res = {"run": tag, "label": label, "stopped_at_shot": n, "max_shots": m["max_n_shots"],
           "converged": bool(m["converged"]),
           "ending": "met the convergence criterion" if m["converged"] else
                     ("hit the 600-shot cap" if n >= m["max_n_shots"] else "still running / stopped early"),
           "final_preshot_rise_K_per_shot": m["preshot_rise_per_shot_last10pct_K"],
           "last_shot_peak_surface_T_K": float(r[-1, 2]), "last_shot_preshot_T_K": float(r[-1, 1]),
           "max_interface_T_K": float(r[:, 3].max()), "last_shot_interface_T_K": float(r[-1, 3]),
           "interface_minus_925K": float(r[:, 3].max() - LIMIT), "interface_minus_undamaged_K": float(r[:, 3].max() - BASE_IF),
           "melt_depth_first_shot_um": float(r[0, 6]), "melt_depth_last_shot_um": float(r[-1, 6]),
           "evaporation_last_shot": evap(float(r[-1, 7])), "cum_evaporated_nm": float(r[-1, 5]),
           "energy_rel_residual": m["energy_rel_residual"]}
    if not m["converged"] and n >= 200:
        res["extrapolated_preshot"] = fit_last(r[:, 0], r[:, 1])
        res["extrapolated_interface"] = fit_last(r[:, 0], r[:, 3])
    ex = D / "multi" / f"{tag}_extra_shot" / "Demo_extra_shot.json"
    if ex.exists():
        e = json.loads(ex.read_text(encoding="utf-8"))
        res["extra_shot"] = {k: e[k] for k in ("shot_number", "preshot_surface_T_K", "peak_surface_T_K", "max_melt_depth_um",
                                               "melt_duration_us", "melt_start_us", "melt_end_us", "peak_interface_T_K",
                                               "ablated_nm_this_shot")}
    return res, r


def main():
    out = arg("out", D / "multishot_final")
    figcopy = arg("figcopy", ROOT / "docs" / "figures" / "damage")
    out.mkdir(parents=True, exist_ok=True)
    results, data = {}, {}
    for tag, label in RUNS.items():
        if (D / "multi" / tag / "Demo_per_shot.csv").exists():
            results[tag], data[tag] = analyse(tag, label)

    b = read_csv(BASELINE / "Demo_per_shot.csv")
    bm = json.loads((BASELINE / "Demo.json").read_text(encoding="utf-8"))
    results["undamaged_baseline"] = {
        "stopped_at_shot": len(b), "final_preshot_rise_K_per_shot": bm["preshot_rise_per_shot_last10pct_K"],
        "last_shot_preshot_T_K": float(b[-1, 1]), "last_shot_peak_surface_T_K": float(b[-1, 2]),
        "max_interface_T_K": float(b[:, 3].max()), "cum_evaporated_nm": float(b[-1, 5]),
        "evaporation_mean_over_run": evap(float(b[-1, 5]) / len(b)),
        "evaporation_last_shot": evap(float(b[-1, 5] - b[-2, 5]))}
    if WORST.exists():
        w = json.loads(WORST.read_text(encoding="utf-8"))
        results["single_shot_worst_case_hot_start"] = {"case": WORST.parent.name, "peak_surface_T_K": w["peak_surface_T_K"],
                                                       "melt_depth_um": w["melt_depth_um"],
                                                       "evaporation": evap(float(w["ablated_nm_per_shot"]))}

    # do the two damaged runs have the same deep temperatures?
    if len(data) == 2:
        a, c = data["kirr0.5_conv"], data["Heskin_conv"]
        n = min(len(a), len(c))
        results["kirr_vs_Heskin_same_shots"] = {
            "compared_shots": n, "max_abs_diff_preshot_surface_K": float(np.abs(a[:n, 1] - c[:n, 1]).max()),
            "max_abs_diff_peak_interface_K": float(np.abs(a[:n, 3] - c[:n, 3]).max()),
            "max_abs_diff_preshot_back_K": float(np.abs(a[:n, 4] - c[:n, 4]).max())}

    (out / "multishot_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out / "multishot_results.md").write_text(to_md(results), encoding="utf-8")
    make_fig5(out, figcopy, b)
    print(to_md(results))


def f1(x):
    return f"{x:.1f}"


def to_md(R):
    L = ["# Damaged-conductivity multi-shot Demo runs: final numbers", "",
         f"W evaporation: {EVAP_LABEL}. Full-power year = 3.15e8 shots at 10 Hz.", ""]
    L += ["| quantity | " + " | ".join(R[t]["label"] for t in RUNS if t in R) + " |", "|---|" + "---|" * sum(t in R for t in RUNS)]

    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(R[t]) for t in RUNS if t in R) + " |")
    row("stopped at shot", lambda x: f"{x['stopped_at_shot']} ({x['ending']})")
    row("pre-shot rise over last 10 % (K/shot)", lambda x: f"{x['final_preshot_rise_K_per_shot']:.3f}")
    row("last-shot peak surface T (K)", lambda x: f"{x['last_shot_peak_surface_T_K']:.0f}")
    row("last-shot pre-shot surface T (K)", lambda x: f1(x["last_shot_preshot_T_K"]))
    row("max interface T (K)", lambda x: f"{f1(x['max_interface_T_K'])} ({x['interface_minus_925K']:+.1f} vs 925 K; "
                                        f"{x['interface_minus_undamaged_K']:+.1f} vs undamaged {BASE_IF:.2f} K)")
    row("melt depth, first / last shot (µm)", lambda x: f"{x['melt_depth_first_shot_um']:.3f} / {x['melt_depth_last_shot_um']:.3f}")
    row("W evaporated, last shot (nm/shot)", lambda x: f"{x['evaporation_last_shot']['nm_per_shot']:.2e}")
    row("W evaporated per full-power year (nm)", lambda x: f"{x['evaporation_last_shot']['nm_per_full_power_year']:.3g}")
    row("years to lose 1 mm (evaporation only)", lambda x: f"{x['evaporation_last_shot']['years_to_lose_1mm']:.2f}")
    row("extrapolated steady pre-shot T (K)", lambda x: (f"{x['extrapolated_preshot']['T_inf_K']:.1f} ± {x['extrapolated_preshot']['T_inf_1sigma_K']:.2f}"
                                                        f" (τ = {x['extrapolated_preshot']['tau_shots']:.0f} shots)") if "extrapolated_preshot" in x else "not needed")
    row("extrapolated steady interface T (K)", lambda x: (f"{x['extrapolated_interface']['T_inf_K']:.1f} ± {x['extrapolated_interface']['T_inf_1sigma_K']:.2f}")
        if "extrapolated_interface" in x else "not needed")
    row("extra shot: max melt depth (µm)", lambda x: f"{x['extra_shot']['max_melt_depth_um']:.3f}" if "extra_shot" in x else "not run")
    row("extra shot: melt duration (µs)", lambda x: f"{x['extra_shot']['melt_duration_us']:.2f}" if "extra_shot" in x else "not run")
    L.append("")
    L.append("Extrapolated values are EXTRAPOLATIONS (exponential approach fitted to the last 200 shots), not computed steady states.")
    L.append("")
    b = R["undamaged_baseline"]
    L.append(f"Undamaged baseline ({b['stopped_at_shot']} shots): pre-shot {f1(b['last_shot_preshot_T_K'])} K, max interface "
             f"{b['max_interface_T_K']:.2f} K, evaporation {b['cum_evaporated_nm']:.3g} nm total = "
             f"{b['evaporation_mean_over_run']['nm_per_shot']:.2e} nm/shot mean, {b['evaporation_last_shot']['nm_per_shot']:.2e} nm/shot "
             f"at the last shot -> {b['evaporation_last_shot']['nm_per_full_power_year']:.3g} nm per full-power year, "
             f"{b['evaporation_last_shot']['years_to_lose_1mm']:.3g} years to lose 1 mm.")
    if "single_shot_worst_case_hot_start" in R:
        w = R["single_shot_worst_case_hot_start"]
        L.append(f"Single-shot worst case ({w['case']}): {w['evaporation']['nm_per_shot']:.2e} nm/shot -> "
                 f"{w['evaporation']['nm_per_full_power_year']:.3g} nm per full-power year, "
                 f"{w['evaporation']['years_to_lose_1mm']:.2f} years to lose 1 mm.")
    if "kirr_vs_Heskin_same_shots" in R:
        c = R["kirr_vs_Heskin_same_shots"]
        L.append(f"kirr0.5 vs Heskin over the first {c['compared_shots']} shots: max |Δ| pre-shot surface "
                 f"{c['max_abs_diff_preshot_surface_K']:.3g} K, peak interface {c['max_abs_diff_peak_interface_K']:.3g} K, "
                 f"pre-shot back face {c['max_abs_diff_preshot_back_K']:.3g} K.")
    return "\n".join(L) + "\n"


def make_fig5(out, figcopy, base):
    series = [(base, "undamaged (600 shots, earlier run)", "C0")]
    for tag, label in RUNS.items():
        p = D / "multi" / tag / "Demo_per_shot.csv"
        if p.exists():
            series.append((read_csv(p), label, "C3" if tag.startswith("kirr") else "C1"))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.4))
    ax = ax.ravel()
    for a, lab, c in series:
        ax[0].plot(a[:, 0], a[:, 2], c=c, label=lab)
        ax[1].plot(a[:, 0], a[:, 1], c=c, label=lab)
        ax[2].plot(a[:, 0], a[:, 3], c=c, label=lab)
        if a.shape[1] > 6:
            ax[3].plot(a[:, 0], a[:, 6], c=c, label=lab)
        else:  # the baseline CSV has no melt column; it never melts (peak < 3695 K)
            ax[3].plot(a[:, 0], np.zeros(len(a)), c=c, label=lab + " (no melt)")
    ax[0].axhline(W_TM, ls="--", c="k", lw=1)
    ax[0].text(5, W_TM + 8, "W melting point 3695 K", fontsize=9)
    ax[2].axhline(LIMIT, ls="--", c="k", lw=1)
    ax[2].text(5, LIMIT + 3, "ODS-RAFM limit 925 K", fontsize=9)
    ax[0].set(ylabel="peak W surface T (K)", title="(a) peak surface temperature")
    ax[1].set(ylabel="W surface T just before the shot (K)", title="(b) pre-shot surface temperature")
    ax[2].set(ylabel="peak W/steel interface T (K)", title="(c) W/steel interface")
    ax[3].set(ylabel="melt depth per shot (µm)", title="(d) melt depth")
    for i, x in enumerate(ax):
        x.set_xlabel("shot number (Demo, 10 Hz)")
        x.legend(fontsize=8, loc="center right" if i == 3 else "lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig5_multishot_peaks.{ext}", dpi=150, bbox_inches="tight")
    if figcopy is not None:
        figcopy.mkdir(parents=True, exist_ok=True)
        fig.savefig(figcopy / "fig5_multishot_peaks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
