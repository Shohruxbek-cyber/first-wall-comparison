"""Collect production outputs into summary tables and tornado plots.

Reads runs/w_single/*/, runs/w_multi/baseline/, runs/li_scan/*/, runs/source_term/he_shot_count.json
Writes runs/summary/summary.json, tornado_W_Demo.png/pdf, tornado_Li_Demo.png/pdf, li_scan.png/pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import DEFAULT_ION_SPECTRA  # noqa: E402

OUT = ROOT / "runs" / "summary"
OUT.mkdir(parents=True, exist_ok=True)
W_ARMOR_NM = 1e6  # 1 mm
LIMIT = 925.0


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def crossing_delta(rows, limit=LIMIT):
    """Film thickness where the substrate peak crosses the limit (log-interpolated between
    the last failing and first passing grid points); None if never passes."""
    rows = sorted(rows, key=lambda r: r["delta_um"])
    for a, b in zip(rows[:-1], rows[1:]):
        if (not a["substrate_pass"]) and b["substrate_pass"]:
            Ta, Tb = a["peak_substrate_K"], b["peak_substrate_K"]
            if a["status"] != "ok" or Ta <= Tb:  # failed run (substrate beyond property range)
                return b["delta_um"], (a["delta_um"], b["delta_um"])
            f = (Ta - limit) / (Ta - Tb)
            return float(np.exp(np.log(a["delta_um"]) + f * np.log(b["delta_um"] / a["delta_um"]))), (a["delta_um"], b["delta_um"])
    if rows and rows[0]["substrate_pass"]:
        return rows[0]["delta_um"], (None, rows[0]["delta_um"])
    return None, (None, None)


summary = {"W": {}, "Li": {}, "tornado": {}}

# ---------------- tungsten ----------------------------------------------------------------
he = load(ROOT / "runs/source_term/he_shot_count.json")
for scen in ("Experimental", "Prototype", "Demo"):
    base = load(ROOT / f"runs/w_single/baseline/{scen}.json")
    abl = base["ablated_nm_per_shot"]
    row = {"peak_surface_T_K_single": base["peak_surface_T_K"], "melt_depth_um": base["melt_depth_um"],
           "ablated_nm_per_shot": abl, "shots_to_ablate_1mm_evaporation_only": (W_ARMOR_NM / abl) if abl > 0 else float("inf")}
    h = he[f"{scen} ({DEFAULT_ION_SPECTRA})"]
    row["He_shots_to_1e18_to_1e19"] = [h["burn+debris 4He"]["shots_1e18"], h["burn 4He"]["shots_1e19"]]
    m = ROOT / f"runs/w_multi/baseline/{scen}.json"
    if m.exists():
        mm = load(m)
        row.update({"multishot_n": mm["n_shots"], "multishot_last_preshot_K": mm["last_preshot_K"],
                    "multishot_last_peak_K": mm["last_shot_peak_K"],
                    "preshot_rise_per_shot_K": mm["preshot_rise_per_shot_last10pct_K"],
                    "max_interface_T_K": mm["max_interface_T_K"]})
    summary["W"][scen] = row

# W tornado (Demo): peak surface T and ablation per shot
wv = {}
for d in sorted((ROOT / "runs/w_single").iterdir()):
    f = d / "Demo.json"
    if d.is_dir() and f.exists() and d.name != "baseline":
        wv[d.name] = load(f)
b = summary["W"]["Demo"]
torn = {k: {"peak_T_K": v["peak_surface_T_K"], "ablated_nm": v["ablated_nm_per_shot"]} for k, v in wv.items()}
summary["tornado"]["W_Demo"] = {"baseline": {"peak_T_K": b["peak_surface_T_K_single"], "ablated_nm": b["ablated_nm_per_shot"]}, "variants": torn}

# ---------------- lithium -------------------------------------------------------------------
cases = {}
for d in sorted((ROOT / "runs/li_scan").iterdir()):
    if d.is_dir():
        for f in d.glob("*.json"):
            if f.name.startswith("vapour"):
                continue
            j = load(f)
            cases[(j["scenario"], d.name)] = j
for scen in ("Experimental", "Prototype", "Demo"):
    base = cases.get((scen, "T600K"))
    if not base:
        continue
    dmin, bracket = crossing_delta(base["rows"])
    thick = [r for r in base["rows"] if r["delta_um"] >= 100]
    vap = float(np.median([r["vaporized_kg_m2"] for r in thick]))
    sens = {}
    for (s2, tag), j in cases.items():
        if s2 == scen and tag != "T600K":
            dm, _ = crossing_delta(j["rows"])
            sens[tag] = {"min_delta_um": dm,
                         "vaporized_kg_m2_thick": float(np.median([r["vaporized_kg_m2"] for r in j["rows"] if r["delta_um"] >= 100]))}
    dvals = [dmin] + [v["min_delta_um"] for v in sens.values() if v["min_delta_um"]]
    vvals = [vap] + [v["vaporized_kg_m2_thick"] for v in sens.values()]
    summary["Li"][scen] = {"baseline_T_li_K": 600, "min_delta_um": dmin, "grid_bracket_um": bracket,
                           "min_delta_range_um": [min(dvals), max(dvals)] if dmin else None,
                           "vaporized_kg_m2_thick_film": vap, "vaporized_range_kg_m2": [min(vvals), max(vvals)],
                           "flow_speed_m_s": base["flow_speed_for_full_replacement_m_s"], "sensitivity": sens,
                           "rows": base["rows"]}

# ---------------- plots ----------------------------------------------------------------------
NICE = {"alpha_evap-0.1": "evaporation coeff. α = 0.1", "xray_tau-1e-10": "x-ray pulse 0.1 ns",
        "xray_tau-1e-08": "x-ray pulse 10 ns", "xray_tau-5e-08": "x-ray pulse 50 ns",
        "normalize_ions-false": "raw (unnormalized) ion spectra", "k_scale-0.85": "W conductivity −15 %",
        "k_scale-1.15": "W conductivity +15 %", "w_lv-raffray_slope": "W L_v from vapor-pressure slope",
        "depth_mode-csda": "CSDA ion depth", "stretch-false": "no H-isotope depth stretch",
        "T500K": "film at 500 K", "T800K": "film at 800 K", "T600K_alpha_evap-0.1": "evaporation coeff. α = 0.1",
        "T600K_lv-handbook_147kJmol": "Li L_v handbook (147 kJ/mol)", "T600K_xray_tau-1e-10": "x-ray pulse 0.1 ns",
        "T600K_xray_tau-5e-08": "x-ray pulse 50 ns"}


def tornado(ax, base, variants, label, log=False):
    items = sorted(variants.items(), key=lambda kv: abs(np.log(kv[1] / base) if log else kv[1] - base))
    for i, (k, v) in enumerate(items):
        x = (v - base) if not log else np.log10(v / base)
        ax.barh(i, x, color="C3" if v > base else "C0")
        ax.text(x, i, f" {NICE.get(k, k)} " if x >= 0 else f" {NICE.get(k, k)} ", va="center",
                ha="left" if x >= 0 else "right", fontsize=9)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel(label)


if torn:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    tb = summary["tornado"]["W_Demo"]["baseline"]
    tornado(ax[0], tb["peak_T_K"], {k: v["peak_T_K"] for k, v in torn.items()}, "change in peak W surface T (K), Demo single shot")
    ab = {k: v["ablated_nm"] for k, v in torn.items() if v["ablated_nm"] > 0}
    if tb["ablated_nm"] > 0 and ab:
        tornado(ax[1], tb["ablated_nm"], ab, "log10(ablation / baseline), Demo single shot", log=True)
    ax[0].set_title(f"W Demo: baseline peak {tb['peak_T_K']:.0f} K")
    ax[1].set_title(f"W Demo: baseline evaporative ablation {tb['ablated_nm']:.2e} nm/shot")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tornado_W_Demo.{ext}", dpi=150)

if "Demo" in summary["Li"]:
    L = summary["Li"]["Demo"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    tornado(ax[0], L["vaporized_kg_m2_thick_film"], {k: v["vaporized_kg_m2_thick"] for k, v in L["sensitivity"].items()},
            "log10(Li vaporized / baseline), Demo, thick film", log=True)
    ax[0].set_title(f"Li Demo: baseline {L['vaporized_kg_m2_thick_film']:.2e} kg/m$^2$ per shot")
    tornado(ax[1], L["min_delta_um"], {k: v["min_delta_um"] for k, v in L["sensitivity"].items() if v["min_delta_um"]},
            "log10(minimum film thickness / baseline)", log=True)
    ax[1].set_title(f"Li Demo: baseline minimum film {L['min_delta_um']:.1f} $\\mu$m")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"tornado_Li_Demo.{ext}", dpi=150)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for scen, c in zip(("Experimental", "Prototype", "Demo"), ("C0", "C1", "C3")):
        if scen not in summary["Li"]:
            continue
        rows = summary["Li"][scen]["rows"]
        d = [r["delta_um"] for r in rows]
        ax[0].semilogx(d, [r["absorbed_frac_in_Li"] for r in rows], "o-", c=c, ms=3, label=scen)
        ax[1].semilogx(d, [r["peak_substrate_K"] if r["status"] == "ok" else np.nan for r in rows], "o-", c=c, ms=3, label=scen)
        ax[2].loglog(d, [r["vaporized_kg_m2"] if r["status"] == "ok" else np.nan for r in rows], "o-", c=c, ms=3, label=scen)
    ax[1].axhline(LIMIT, ls="--", c="k", lw=0.8)
    ax[0].set(xlabel=r"Li film thickness ($\mu$m)", ylabel="fraction of threat energy absorbed in Li", ylim=(0, 1))
    ax[1].set(xlabel=r"Li film thickness ($\mu$m)", ylabel="peak substrate temperature (K)", title="substrate limit 925 K (dashed)")
    ax[2].set(xlabel=r"Li film thickness ($\mu$m)", ylabel="Li vaporized per shot (kg/m$^2$)")
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("Li film on steel, single shot at T_Li = 600 K (no vapour shielding)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"li_scan.{ext}", dpi=150)

(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "tornado"}, indent=1, default=str)[:6000])
