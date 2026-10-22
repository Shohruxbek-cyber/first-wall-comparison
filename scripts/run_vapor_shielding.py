"""Vapor-shielding runs (revision of 20 Oct 2026; model: Section 2.5 of the paper).

Single shots with the reduced, time-coupled vapor column of ife_wall.vapor:
  * Li film, all three scenarios, the 13 production thicknesses, T_Li = 600 K, alpha = 1, with
    (f_rr, s) = (0,1) (0.5,1) (1,1) (0,2) (0.5,2) (1,2) and immediate return (tau_rr = 0);
  * Li Demo with a delayed, Xe-like return: f_rr = 1, tau_rr = 300 us, s = 1;
  * W Demo from 600 K with (f_rr, s) = (0,1) and (1,1);
  * regression with the option off: Li Demo at 31.6 um and W Demo against the stored
    production outputs (runs/li_scan/T600K/Demo.json, runs/w_single/baseline/Demo.json).
Each Li shot follows scripts/run_li_scan.py exactly (50 us at dt0 = 1e-13 s, then to 5 ms at
dt0 = 1e-9 s); each W shot follows scripts/run_w_single.py (50 us, then to the next shot).

Usage: python scripts/run_vapor_shielding.py [--workers=6] [--only=regression] [--out=results/vapor_shielding]
Output: <out>/li_<scenario>_<case>.json, w_Demo_<case>.json, regression.json, summary.{json,md}, figures.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DELTAS = np.geomspace(1e-6, 1e-3, 13)  # as scripts/run_li_scan.py
LIMIT = 925.0  # K, steel limit (as run_li_scan.py)
NA = 6.02214076e23
LI_CASES = [(0.0, 1.0, 0.0), (0.5, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 2.0, 0.0), (0.5, 2.0, 0.0), (1.0, 2.0, 0.0)]
LI_DELAYED = (1.0, 1.0, 3e-4)
W_CASES = [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]


def case_tag(c):
    f_rr, s, tau = c
    return f"frr{f_rr:g}_s{s:g}" + (f"_tau{tau * 1e6:g}us" if tau else "")


def git_hash():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "no-git"


def li_shot(task):
    """One Li single shot; returns the row (same fields as run_li_scan.py plus the vapor record)."""
    from ife_wall.library import M_LI
    from ife_wall.scenarios import SCENARIOS, Options, li_case
    name, i_delta, case = task
    scen = SCENARIOS[name]
    delta = float(DELTAS[i_delta])
    if case is None:
        opt = Options()
    else:
        f_rr, sf, tau = case
        opt = Options(vapor_shielding=True, vapor_f_rr=f_rr, vapor_stopping_factor=sf, vapor_tau_rr=tau)
    t0 = time.time()
    s = li_case(scen, delta, 600.0, opt)
    rho_li = float(s.materials[0].rho(np.array(600.0)))
    m_li0 = float(s.m[s.layer == 0].sum())
    pk = {"Ts": 0.0, "Tsub": 0.0}

    def rec(sv):
        T = sv.T
        pk["Ts"] = max(pk["Ts"], sv.surface_temperature(T)) if np.any(sv.layer == 0) else pk["Ts"]
        pk["Tsub"] = max(pk["Tsub"], float(T[sv.layer == 1].max()))

    status = "ok"
    try:
        s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec)
        s.run(5e-3, dt0=1e-9, dT_target=5.0, record=rec)
    except RuntimeError as e:
        status = f"stopped at t={s.t:.3e} s: {e}"
    mv = s.mass_evaporated
    row = {"scenario": name, "delta_um": delta * 1e6, "case": None if case is None else case_tag(case),
           "peak_Li_surface_K": pk["Ts"], "peak_substrate_K": pk["Tsub"],
           "substrate_pass": pk["Tsub"] < LIMIT and status == "ok",
           "vaporized_kg_m2": mv, "vaporized_um": mv / rho_li * 1e6,
           "burn_through": float(s.m[s.layer == 0].sum()) < 0.01 * m_li0, "status": status,
           "energy_rel_residual": None, "wall_s": None}
    if status == "ok":
        s.ledger.stored_final = s.stored_energy()
        row["energy_rel_residual"] = float(s.ledger.residual()[1])
    if case is not None:
        v = s.sources[1].finalize(s.t)
        n_atoms = mv / M_LI * NA if mv > 0 else 0.0
        v["stopped_energy_per_vapor_atom_eV"] = v["ion_energy_in_vapor_J_m2"] / n_atoms / 1.602176634e-19 if n_atoms else None
        row["vapor"] = v
    row["wall_s"] = time.time() - t0
    return row


def w_shot(case):
    """W Demo single shot from 600 K (as run_w_single.single, without the diagnostics recorder)."""
    from ife_wall.scenarios import SCENARIOS, Options, w_case
    scen = SCENARIOS["Demo"]
    if case is None:
        opt = Options()
    else:
        f_rr, sf, tau = case
        opt = Options(vapor_shielding=True, vapor_f_rr=f_rr, vapor_stopping_factor=sf, vapor_tau_rr=tau)
    t0 = time.time()
    s = w_case(scen, opt)
    peak = {"Ts": 0.0, "t": 0.0}

    def rec(sv):
        Ts = sv.surface_temperature()
        if Ts > peak["Ts"]:
            peak.update(Ts=Ts, t=sv.t)

    s.run(50e-6, dt0=1e-13, dT_target=5.0, record=rec)
    out = {"case": None if case is None else case_tag(case), "peak_surface_T_K": peak["Ts"], "t_peak_us": peak["t"] * 1e6,
           "evaporated_kg_m2_to_50us": s.mass_evaporated, "ablated_nm_per_shot": s.mass_evaporated / 19250.0 * 1e9}
    s.ledger.stored_final = s.stored_energy()
    out["energy_rel_residual_50us"] = float(s.ledger.residual()[1])
    s.run(1.0 / scen.rep_Hz, dt0=1e-9, dT_target=5.0, record=rec)
    out["pre_shot2_surface_T_K"] = s.surface_temperature()
    if case is not None:
        out["vapor"] = s.sources[1].finalize(s.t)
        # static check: share of the ion energy that the final W column would stop if it stood in front from t = 0
        from ife_wall.vapor import VaporShieldedSource
        w = VaporShieldedSource(s.sources[1].base, fixed_column=max(s.mass_evaporated, 1e-300))
        w.fractions(s)
        E = s.sources[1].base.fluence
        out["static_share_of_final_column"] = float(E @ w.vapor_share / E.sum())
    out["wall_s"] = time.time() - t0
    return out


def run_task(task):
    kind = task[0]
    if kind == "li":
        return ("li", task[1:], li_shot(task[1:]))
    return ("w", task[1:], w_shot(task[1]))


def crossing_delta(rows, limit=LIMIT):
    """As scripts/make_summary.py: log-interpolated thickness where the substrate peak crosses the limit."""
    rows = sorted(rows, key=lambda r: r["delta_um"])
    for a, b in zip(rows[:-1], rows[1:]):
        if (not a["substrate_pass"]) and b["substrate_pass"]:
            Ta, Tb = a["peak_substrate_K"], b["peak_substrate_K"]
            if a["status"] != "ok" or Ta <= Tb:
                return b["delta_um"], (a["delta_um"], b["delta_um"])
            f = (Ta - limit) / (Ta - Tb)
            return float(np.exp(np.log(a["delta_um"]) + f * np.log(b["delta_um"] / a["delta_um"]))), (a["delta_um"], b["delta_um"])
    if rows and rows[0]["substrate_pass"]:
        return rows[0]["delta_um"], (None, rows[0]["delta_um"])
    return None, (None, None)


def thick_film(rows):
    """'Thick film' evaporated mass: median over delta >= 100 um (as Section 3.9 of the manuscript)."""
    v = [r["vaporized_kg_m2"] for r in rows if r["delta_um"] >= 100 and r["status"] == "ok"]
    return float(np.median(v)) if v else None


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    out = Path(args.get("out", ROOT / "results" / "vapor_shielding"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    workers = int(args.get("workers", 6))
    only = args.get("only")
    if only == "figure":  # redraw the figure from the stored per-case files (no solver run)
        base_rows = {n: json.loads((ROOT / f"runs/li_scan/T600K/{n}.json").read_text(encoding="utf-8"))["rows"]
                     for n in ("Experimental", "Prototype", "Demo")}
        results = []
        for c in LI_CASES + [LI_DELAYED]:
            p = out / f"li_Demo_{case_tag(c)}.json"
            if p.exists():
                results += [("li", ("Demo", None, c), r) for r in json.loads(p.read_text(encoding="utf-8"))["rows"]]
        make_figure(out, results, base_rows)
        return
    tasks = [("li", "Demo", 6, None), ("w", None)]  # regression first
    if only != "regression":
        for c in LI_CASES:
            for name in ("Experimental", "Prototype", "Demo"):
                tasks += [("li", name, i, c) for i in range(len(DELTAS))]
        tasks += [("li", "Demo", i, LI_DELAYED) for i in range(len(DELTAS))]
        tasks += [("w", c) for c in W_CASES]
    t_start = time.time()
    results = []
    with Pool(workers) as pool:
        for k, res in enumerate(pool.imap_unordered(run_task, tasks), 1):
            results.append(res)
            kind, key, r = res
            label = f"{key[0]} {r.get('delta_um', 0):.3g} um {r.get('case')}" if kind == "li" else f"W Demo {r.get('case')}"
            print(f"[{k}/{len(tasks)}] {label}: {r.get('status', 'ok')}, {r.get('wall_s', 0):.0f} s", flush=True)
    meta = {"git": git_hash(), "run_date": time.strftime("%Y-%m-%d %H:%M"), "command": "python scripts/run_vapor_shielding.py "
            + " ".join(sys.argv[1:]), "wall_s_total": time.time() - t_start, "workers": workers}

    # ---- regression: option off against the stored production outputs ----
    prod_li = json.loads((ROOT / "runs/li_scan/T600K/Demo.json").read_text(encoding="utf-8"))["rows"][6]
    prod_w = json.loads((ROOT / "runs/w_single/baseline/Demo.json").read_text(encoding="utf-8"))
    li_off = next(r for kind, key, r in results if kind == "li" and key[2] is None)
    w_off = next(r for kind, key, r in results if kind == "w" and key[0] is None)
    reg = {"li_Demo_31.6um": {f: {"production": prod_li[f], "rerun": float(li_off[f]), "identical": bool(prod_li[f] == li_off[f])}
                              for f in ("peak_Li_surface_K", "peak_substrate_K", "vaporized_kg_m2")},
           "w_Demo": {"production": prod_w["peak_surface_T_K"], "rerun": float(w_off["peak_surface_T_K"]),
                      "identical": bool(prod_w["peak_surface_T_K"] == w_off["peak_surface_T_K"])}}
    reg["all_identical"] = all(v["identical"] for v in reg["li_Demo_31.6um"].values()) and reg["w_Demo"]["identical"]
    (out / "regression.json").write_text(json.dumps({"meta": meta, **reg}, indent=1), encoding="utf-8")
    print("regression (option off reproduces production):", reg["all_identical"], flush=True)
    if only == "regression":
        return

    # ---- per-case files and summary ----
    base_rows = {n: json.loads((ROOT / f"runs/li_scan/T600K/{n}.json").read_text(encoding="utf-8"))["rows"]
                 for n in ("Experimental", "Prototype", "Demo")}
    summary = {"meta": meta, "li": {}, "w": {}}
    for name in ("Experimental", "Prototype", "Demo"):
        base_min = crossing_delta(base_rows[name])[0]
        base_thick = thick_film(base_rows[name])
        summary["li"][name] = {"baseline": {"min_delta_um": base_min, "thick_film_kg_m2": base_thick}}
        for c in LI_CASES + ([LI_DELAYED] if name == "Demo" else []):
            rows = sorted([r for kind, key, r in results if kind == "li" and key[0] == name and key[2] == c],
                          key=lambda r: r["delta_um"])
            if not rows:
                continue
            tag = case_tag(c)
            (out / f"li_{name}_{tag}.json").write_text(json.dumps({"scenario": name, "case": tag, "f_rr": c[0],
                                                                  "stopping_factor": c[1], "tau_rr_s": c[2], "meta": meta,
                                                                  "rows": rows}, indent=1), encoding="utf-8")
            mn, bracket = crossing_delta(rows)
            thick = thick_film(rows)
            ok = [r for r in rows if r["status"] == "ok" and r["delta_um"] >= 100]
            summary["li"][name][tag] = {
                "min_delta_um": mn, "bracket_um": bracket, "thick_film_kg_m2": thick,
                "thick_film_ratio_to_baseline": thick / base_thick if thick and base_thick else None,
                "fraction_of_ion_energy_in_vapor_thick": float(np.median([r["vapor"]["fraction_of_ion_energy_in_vapor"] for r in ok])) if ok else None,
                "eV_per_vapor_atom_thick": float(np.median([r["vapor"]["stopped_energy_per_vapor_atom_eV"] for r in ok
                                                            if r["vapor"]["stopped_energy_per_vapor_atom_eV"] is not None])) if ok else None,
                "peak_Li_surface_K_thick": float(np.median([r["peak_Li_surface_K"] for r in ok])) if ok else None,
                "peak_substrate_K_by_delta": {f"{r['delta_um']:.3g}": r["peak_substrate_K"] for r in rows},
                "max_energy_rel_residual": max(abs(r["energy_rel_residual"]) for r in rows if r["energy_rel_residual"] is not None)}
    for kind, key, r in results:
        if kind == "w" and key[0] is not None:
            summary["w"][r["case"]] = r
            (out / f"w_Demo_{r['case']}.json").write_text(json.dumps({"meta": meta, **r}, indent=1), encoding="utf-8")
    summary["w"]["baseline_peak_K"] = prod_w["peak_surface_T_K"]
    summary["regression_all_identical"] = reg["all_identical"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_markdown(out, summary)
    make_figure(out, results, base_rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=1)[:4000])


def write_markdown(out, S):
    L = ["# Vapor-shielding runs: summary", "",
         f"Generated by `{S['meta']['command']}` at commit {S['meta']['git']} on {S['meta']['run_date']}. "
         "Single shots, T_Li = 600 K, alpha = 1; immediate return unless a time constant is given. "
         "Baseline = the production scan without shielding (runs/li_scan/T600K).", ""]
    for name, d in S["li"].items():
        b = d["baseline"]
        L += [f"## Li, {name}", "",
              "| Case | Minimum film (um) | Thick-film evaporation (kg/m2) | Ratio to baseline | Ion energy stopped in vapor | Stopped energy per vapor atom (eV) | Peak Li surface, thick film (K) | Max ledger residual |",
              "|---|---|---|---|---|---|---|---|",
              f"| no shielding (production) | {b['min_delta_um']:.3g} | {b['thick_film_kg_m2']:.4g} | 1 | 0 | – | – | – |" if b["min_delta_um"] else
              f"| no shielding (production) | – | {b['thick_film_kg_m2']:.4g} | 1 | 0 | – | – | – |"]
        for tag, c in d.items():
            if tag == "baseline":
                continue
            mn = f"{c['min_delta_um']:.3g}" if c["min_delta_um"] else "–"
            L.append(f"| {tag} | {mn} | {c['thick_film_kg_m2']:.4g} | {c['thick_film_ratio_to_baseline']:.3f} | "
                     f"{c['fraction_of_ion_energy_in_vapor_thick']:.3f} | {c['eV_per_vapor_atom_thick']:.3g} | "
                     f"{c['peak_Li_surface_K_thick']:.0f} | {c['max_energy_rel_residual']:.1e} |")
        L.append("")
    L += ["## W, Demo single shot from 600 K", "", "| Case | Peak surface (K) | Baseline (K) | Evaporated to 50 us (kg/m2) | Final column | Static share of the final column |",
          "|---|---|---|---|---|---|"]
    for tag, r in S["w"].items():
        if tag == "baseline_peak_K":
            continue
        L.append(f"| {tag} | {r['peak_surface_T_K']:.6f} | {S['w']['baseline_peak_K']:.6f} | {r['evaporated_kg_m2_to_50us']:.3e} | "
                 f"{r['vapor']['column_kg_m2']:.3e} | {r['static_share_of_final_column']:.2e} |")
    L += ["", f"Regression with the option off (Li Demo 31.6 um, W Demo): identical to production = {S['regression_all_identical']}.", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def make_figure(out, results, base_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    b = sorted(base_rows["Demo"], key=lambda r: r["delta_um"])
    ok = [r for r in b if r["status"] == "ok"]
    d_min = min(r["delta_um"] for r in ok)  # plot every case from the thinnest film the baseline completes
    ax[0].loglog([r["delta_um"] for r in ok], [r["vaporized_kg_m2"] for r in ok], "k-o", ms=3, label="no shielding")
    ax[1].semilogx([r["delta_um"] for r in ok], [r["peak_substrate_K"] for r in ok], "k-o", ms=3, label="no shielding")
    styles = {(0.0, 1.0, 0.0): ("C0", "-"), (0.5, 1.0, 0.0): ("C1", "-"), (1.0, 1.0, 0.0): ("C3", "-"),
              (0.0, 2.0, 0.0): ("C0", "--"), (0.5, 2.0, 0.0): ("C1", "--"), (1.0, 2.0, 0.0): ("C3", "--"),
              LI_DELAYED: ("C2", ":")}
    for c, (col, ls) in styles.items():
        rows = sorted([r for kind, key, r in results if kind == "li" and key[0] == "Demo" and key[2] == c
                       and r["status"] == "ok" and r["delta_um"] >= d_min * (1 - 1e-9)], key=lambda r: r["delta_um"])
        if not rows:
            continue
        lab = f"f_rr = {c[0]:g}, s = {c[1]:g}" + (f", tau = {c[2] * 1e6:g} us" if c[2] else "")
        ax[0].loglog([r["delta_um"] for r in rows], [r["vaporized_kg_m2"] for r in rows], color=col, ls=ls, marker=".", label=lab)
        ax[1].semilogx([r["delta_um"] for r in rows], [r["peak_substrate_K"] for r in rows], color=col, ls=ls, marker=".", label=lab)
    ax[1].axhline(LIMIT, color="k", lw=0.8, ls="--")
    ax[0].set(xlabel="Li film thickness (um)", ylabel="Li evaporated per shot (kg/m2)", title="Demo, single shot")
    ax[1].set(xlabel="Li film thickness (um)", ylabel="peak steel temperature (K)", title="Demo, single shot", ylim=(600, 1300))
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_vapor_shielding_Demo.{ext}", dpi=150)


if __name__ == "__main__":
    main()
