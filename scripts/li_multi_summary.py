"""Summary of the multi-shot Li runs (added 2026-10-06).

Reads runs/li_multi/<Scenario>_d<delta>um/<Scenario>.json and <Scenario>_per_shot.csv (written by
scripts/run_li_multi.py) and writes
  runs/li_multi/summary.json, runs/li_multi/summary.md,
  runs/li_multi/fig_li_multishot_<Scenario>.{png,pdf}   (pre-shot steel front T, per-shot peak
                                                          steel T, Li evaporated per shot; one
                                                          curve per film thickness)
The multi-shot minimum film thickness uses the same log interpolation between the last failing
and the first passing grid thickness as the single-shot scan (scripts/make_summary.py,
crossing_delta), applied to the maximum peak steel temperature over the run.

Usage: python scripts/li_multi_summary.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs" / "li_multi"
LIMIT = 925.0


def crossing_delta(rows, limit=LIMIT):
    """As scripts/make_summary.py: thickness where the peak steel T crosses the limit."""
    rows = sorted(rows, key=lambda r: r["delta_um"])
    for a, b in zip(rows[:-1], rows[1:]):
        if (not a["pass"]) and b["pass"]:
            Ta, Tb = a["peak_K"], b["peak_K"]
            if a["status"] != "ok" or Ta <= Tb:
                return b["delta_um"], (a["delta_um"], b["delta_um"])
            f = (Ta - limit) / (Ta - Tb)
            return float(np.exp(np.log(a["delta_um"]) + f * np.log(b["delta_um"] / a["delta_um"]))), (a["delta_um"], b["delta_um"])
    if rows and rows[0]["pass"]:
        return rows[0]["delta_um"], (None, rows[0]["delta_um"])
    return None, (None, None)


runs = {}
for p in sorted(D.glob("*_d*um")):
    m = re.match(r"(Experimental|Prototype|Demo)_d([0-9.]+)um$", p.name)
    if not m or not (p / f"{m.group(1)}.json").exists():
        continue
    j = json.loads((p / f"{m.group(1)}.json").read_text(encoding="utf-8"))
    r = np.loadtxt(p / f"{m.group(1)}_per_shot.csv", delimiter=",", skiprows=1, ndmin=2)
    if "ledger_cumulative_J_m2" not in j and (p / f"{m.group(1)}_checkpoint.npz").exists():
        z = np.load(p / f"{m.group(1)}_checkpoint.npz", allow_pickle=False)
        j["ledger_cumulative_J_m2"] = json.loads(str(z["ledger"]))
    runs.setdefault(m.group(1), []).append((j, r))

summary = {}
lines = ["# Multi-shot Li film runs: summary", "",
         "Film replaced at full thickness and T_Li at the start of every shot after the first; steel carried forward; "
         "coolant fixed at T_c. Peak steel T is the maximum over the steel cells during the shot; 'pass' means the "
         "maximum over the whole run stays below 925 K.", ""]
for scen, lst in runs.items():
    lst.sort(key=lambda t: t[0]["delta_um"])
    rows = []
    for j, r in lst:
        above = r[:, 2] > LIMIT
        L = j.get("ledger_cumulative_J_m2", {})
        n_sh = max(int(j["n_shots"]), 1)
        dep = j["deposition_per_shot_J_m2"]
        steel_load = None
        if L:
            conducted = L["through_interface"][0] / n_sh  # Li -> steel across the interface, per shot (run average)
            steel_load = {"xray_direct": dep["xray_in_Fe"], "ion_direct": dep["ion_in_Fe"], "conducted_from_Li_avg": conducted,
                          "to_coolant_avg": L["to_coolant"] / n_sh, "vaporized_avg": L["vaporized_latent"] / n_sh + L["vaporized_sensible"] / n_sh,
                          "removed_with_film_avg": L["removed_with_film"] / n_sh}
            tot = dep["xray_in_Fe"] + dep["ion_in_Fe"] + conducted
            steel_load["xray_share_of_steel_load"] = dep["xray_in_Fe"] / tot if tot else None
        rows.append({"delta_um": j["delta_um"], "n_shots": int(j["n_shots"]), "max_n_shots": j["max_n_shots"], "status": j["status"],
                     "steel_load_J_m2_per_shot": steel_load,
                     "converged": j["converged"], "preshot_front_rise_K_per_shot": j["preshot_front_rise_per_shot_last10pct_K"],
                     "first_shot_peak_K": float(r[0, 2]), "last_shot_peak_K": float(r[-1, 2]), "peak_K": float(r[:, 2].max()),
                     "first_shot_above_925K": int(r[np.argmax(above), 0]) if above.any() else None,
                     "pass": bool((not above.any()) and j["status"] == "ok"),
                     "last_preshot_front_K": float(r[-1, 1]), "last_preshot_back_K": float(r[-1, 4]),
                     "fit_peak_steel": j.get("fit_peak_steel"), "fit_preshot_front": j.get("fit_preshot_front"),
                     "vaporized_first_shot_kg_m2": float(r[0, 5]), "vaporized_last_shot_kg_m2": float(r[-1, 5]),
                     "vaporized_max_shot_kg_m2": float(r[:, 5].max()),
                     "removed_with_film_total_J_m2": j["removed_with_film_total_J_m2"],
                     "energy_rel_residual": j["energy_rel_residual"], "max_abs_rel_residual_per_shot": j["max_abs_energy_rel_residual_per_shot"],
                     "deposition_per_shot_J_m2": j["deposition_per_shot_J_m2"], "wall_s": j["wall_s_this_session"]})
    dmin, bracket = crossing_delta(rows)
    summary[scen] = {"rows": rows, "min_delta_um_multishot": dmin, "grid_bracket_um": bracket}
    lines += [f"## {scen}", "",
              "| δ (µm) | shots | converged (rise K/shot) | peak steel, shot 1 | max over run | last shot | first shot > 925 K | "
              "pre-shot steel front, last | fit asymptote of peak steel (τ shots) | Li evaporated, shot 1 → last (kg/m²) | residual |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for q in rows:
        fit = q["fit_peak_steel"] or {}
        fit_s = f"{fit['T_inf_K']:.0f} K ({fit['tau_shots']:.0f})" if "T_inf_K" in fit else "—"
        lines.append(f"| {q['delta_um']:.3g} | {q['n_shots']} | {q['converged']} ({q['preshot_front_rise_K_per_shot']:.3f}) | "
                     f"{q['first_shot_peak_K']:.0f} K | {q['peak_K']:.0f} K | {q['last_shot_peak_K']:.0f} K | "
                     f"{q['first_shot_above_925K'] if q['first_shot_above_925K'] else 'never'} | {q['last_preshot_front_K']:.0f} K | {fit_s} | "
                     f"{q['vaporized_first_shot_kg_m2']:.2e} → {q['vaporized_last_shot_kg_m2']:.2e} | {q['energy_rel_residual']:.1e} |")
    lines += ["", f"Multi-shot minimum film thickness ({scen}): {dmin if dmin is None else f'{dmin:.1f} µm'} (bracket {bracket})", ""]
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, r in lst:
        lab = f"{j['delta_um']:.3g} µm"
        ax[0].plot(r[:, 0], r[:, 1], label=lab)
        ax[1].plot(r[:, 0], r[:, 2], label=lab)
        ax[2].plot(r[:, 0], r[:, 5], label=lab)
    ax[0].set(xlabel="shot number", ylabel="temperature (K)", title=f"{scen}: steel front face just before each shot")
    ax[1].axhline(LIMIT, ls="--", c="k", lw=0.8)
    ax[1].text(1, LIMIT + 12, "ODS-RAFM limit 925 K", fontsize=8)
    ax[1].set(xlabel="shot number", ylabel="temperature (K)", title="peak steel temperature in each shot")
    ax[2].set(xlabel="shot number", ylabel="Li evaporated per shot (kg/m$^2$)", title="Li evaporated per shot")
    for a in ax:
        a.legend(fontsize=8, title="film thickness")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(D / f"fig_li_multishot_{scen}.{ext}", dpi=150)
    plt.close(fig)

(D / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
(D / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
