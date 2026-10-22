"""Degraded near-surface-layer sweep (16 Oct 2026): a real surface need not have bulk properties.

A W surface layer of thickness d (skin_um) with the SOLID-phase conductivity multiplied by f (skin_factor)
over undamaged bulk W (k_irr = 1): the existing options of 29 Sep 2026 (`scenarios.Options`, `_w_material`,
`w_case`), reused unchanged.  Density and specific heat stay at bulk; molten W keeps the undamaged
conductivity.  Scaling k by f scales the effusivity sqrt(k rho c) by sqrt(f).

Subcommands (all write only under results/skin_sweep/ unless --out is given):
  python scripts/run_skin_sweep.py regression            compare results/skin_sweep/regression_check with
                                                         results/baseline_pureW/w_single (keys + CSVs, bit-identical?)
  python scripts/run_skin_sweep.py launch [--jobs=N] [--dry]
                                                         run the grid d = 1, 3, 10 um x f = 1.0, 0.75, 0.5, 0.25 for the
                                                         three scenarios (cold start) and Demo from 921.1 K (hot start)
                                                         through the unchanged production script scripts/run_w_single.py,
                                                         N processes at a time; commands and git hashes are logged
  python scripts/run_skin_sweep.py summary               summary.md / summary.json and the figure
                                                         fig_skin_sweep_peak_vs_f.{pdf,png} (+ JPEG copy for the manuscript)
The multi-shot check uses the unchanged production driver, e.g.
  python scripts/run_w_multi.py Demo 100 skin_um=10 skin_factor=0.25 --out=results/skin_sweep/multi_Demo_d10_f0.25
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_w_single import git_hash  # noqa: E402
from ife_wall.scenarios import converged_preshot  # noqa: E402

D_UM = (1.0, 3.0, 10.0)
F = (1.0, 0.75, 0.5, 0.25)
SCEN = ("Experimental", "Prototype", "Demo")
HOT = {"Demo": converged_preshot("Demo", legacy=921.1)}  # converged pre-shot surface temperature of the pure-W 10 Hz run
T_MELT = 3695.0
T_RX = {"1300": 1300.0, "1523": 1523.0}  # GA2020 band lower edge (Section 5.1); 1 h anneal onset of pure W (Section 5.7)
BASELINE = ROOT / "results" / "baseline_pureW" / "w_single"
REG_KEYS = ("peak_surface_T_K", "t_peak_us", "melt_depth_um", "ablated_nm_per_shot", "energy_rel_residual",
            "pre_shot2_surface_T_K", "energy_rel_residual_full_cycle", "diagnostics", "ledger_J_m2")
PY = sys.executable


def tag(d, f):
    return f"d{d:g}_f{f:g}"


# ------------------------------------------------------------------------------------------ regression
def regression(outdir):
    rc = outdir / "regression_check"
    rep = {"reference": str(BASELINE.relative_to(ROOT)), "candidate": str(rc.relative_to(ROOT)), "keys": list(REG_KEYS),
           "csv_files": ["history", "maxT_profile", "He_layer"], "scenarios": {}}
    ok = True
    for name in SCEN:
        a_f, b_f = rc / f"{name}.json", BASELINE / f"{name}.json"
        if not a_f.exists() or not b_f.exists():
            rep["scenarios"][name] = "missing"
            ok = False
            continue
        a, b = json.loads(a_f.read_text(encoding="utf-8")), json.loads(b_f.read_text(encoding="utf-8"))
        per = {k: (a.get(k) == b[k]) for k in REG_KEYS if k in b}
        for suffix in ("history", "maxT_profile", "He_layer"):
            per[f"{suffix}_csv"] = (rc / f"{name}_{suffix}.csv").read_bytes() == (BASELINE / f"{name}_{suffix}.csv").read_bytes()
        per["newton"] = a.get("newton_to_50us")
        rep["scenarios"][name] = per
        ok &= all(v for k, v in per.items() if k != "newton")
    rep["bit_identical"] = bool(ok)
    rep["git"] = git_hash()
    (outdir / "regression_check" / "REGRESSION.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return ok


# ------------------------------------------------------------------------------------------ launch
def jobs_list(outdir):
    jobs = []
    for d in D_UM:
        for f in F:
            jobs.append({"label": f"cold/{tag(d, f)}", "scenarios": list(SCEN), "args": [f"skin_um={d:g}", f"skin_factor={f:g}"],
                         "out": outdir / "single" / "cold" / tag(d, f)})
    jobs.append({"label": "Demo_hot/ref_noskin", "scenarios": ["Demo"], "args": [f"T_init={HOT['Demo']}"],
                 "out": outdir / "single" / "Demo_hot" / "ref_noskin"})
    for d in D_UM:
        for f in F:
            jobs.append({"label": f"Demo_hot/{tag(d, f)}", "scenarios": ["Demo"],
                         "args": [f"T_init={HOT['Demo']}", f"skin_um={d:g}", f"skin_factor={f:g}"],
                         "out": outdir / "single" / "Demo_hot" / tag(d, f)})
    return jobs


def launch(outdir, n_jobs, dry):
    jobs = jobs_list(outdir)
    env = dict(os.environ, PYTHONUTF8="1")
    running, log = [], []
    gh = git_hash()
    logf = outdir / "single" / "commands.json"
    (outdir / "single").mkdir(parents=True, exist_ok=True)
    for j in jobs:
        cmd = [PY, str(ROOT / "scripts" / "run_w_single.py")] + j["scenarios"] + j["args"] + [f"--out={j['out']}"]
        shown = "python scripts/run_w_single.py " + " ".join(j["scenarios"] + j["args"]) + f" --out={j['out'].relative_to(ROOT).as_posix()}"
        if all((j["out"] / f"{s}.json").exists() for s in j["scenarios"]):
            print("skip (exists):", shown, flush=True)
            continue
        if dry:
            print(shown)
            continue
        j["out"].mkdir(parents=True, exist_ok=True)
        while len(running) >= n_jobs:
            running = _reap(running, log, logf, gh)
            time.sleep(2)
        fh = open(j["out"] / "log.txt", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        running.append({"p": p, "fh": fh, "label": j["label"], "command": shown, "start": time.strftime("%Y-%m-%d %H:%M:%S")})
        print("started:", shown, flush=True)
    while running:
        running = _reap(running, log, logf, gh)
        time.sleep(2)
    print(f"{len(log)} runs finished; log in {logf}")


def _reap(running, log, logf, gh):
    still = []
    for r in running:
        if r["p"].poll() is None:
            still.append(r)
            continue
        r["fh"].close()
        log.append({"label": r["label"], "command": r["command"], "git": gh, "start": r["start"],
                    "end": time.strftime("%Y-%m-%d %H:%M:%S"), "returncode": r["p"].returncode})
        print(f"finished ({r['p'].returncode}): {r['label']}", flush=True)
        old = json.loads(logf.read_text(encoding="utf-8")) if logf.exists() else []
        logf.write_text(json.dumps(old + [log[-1]], indent=1), encoding="utf-8")
    return still


# ------------------------------------------------------------------------------------------ summary
def _crossing(prof_csv, thr):
    """Deepest grid point whose maximum temperature in the shot reached thr (the ShotRecorder rule), um."""
    p = np.loadtxt(prof_csv, delimiter=",", skiprows=1)
    above = np.nonzero(p[:, 1] >= thr)[0]
    return float(p[above[-1], 0]) if len(above) else 0.0


def _sput():
    j = json.loads((ROOT / "runs" / "sputtering" / "sputtering.json").read_text(encoding="utf-8"))
    return {s: j["scenarios"][s]["sourced_nm_per_shot"] for s in SCEN}


def _row(jf, start, d, f, sput, label):
    j = json.loads(jf.read_text(encoding="utf-8"))
    name = j["scenario"]
    prof = jf.with_name(f"{name}_maxT_profile.csv")
    dg = j["diagnostics"]
    nw50, nwfc = j.get("newton_to_50us", {}), j.get("newton_full_cycle", {})
    ev = j["ablated_nm_per_shot"]
    row = {"label": label, "scenario": name, "start": start, "T_init_K": j["T_init_K"], "d_um": d, "f": f, "sqrt_f": float(np.sqrt(f)),
           "peak_surface_T_K": j["peak_surface_T_K"], "t_peak_us": j["t_peak_us"],
           "margin_to_melting_K": T_MELT - j["peak_surface_T_K"], "melt_depth_um": j["melt_depth_um"],
           "margin_to_1300K_K": 1300.0 - j["peak_surface_T_K"], "margin_to_1523K_K": 1523.0 - j["peak_surface_T_K"],
           "depth_above_1300K_um": _crossing(prof, 1300.0), "depth_above_1523K_um": _crossing(prof, 1523.0),
           "depth_above_1800K_um": _crossing(prof, 1800.0),
           "evaporation_nm_per_shot": ev, "sputtering_nm_per_shot": sput[name],
           "erosion_limited_shots_1mm": 1e6 / (ev + sput[name]),
           "interface_peak_in_shot_K": dg["depth_table"]["interface"]["peak_K"], "interface_pre_shot_K": dg["depth_table"]["interface"]["pre_shot_K"],
           "pre_shot2_surface_T_K": j.get("pre_shot2_surface_T_K"),
           "newton_steps_to_50us": nw50.get("steps_accepted"), "newton_rejected_to_50us": nw50.get("steps_rejected"),
           "newton_iters_total_to_50us": nw50.get("iters_total_accepted"), "newton_iters_max_step_to_50us": nw50.get("iters_max_per_step"),
           "newton_last_update_K_to_50us": nw50.get("last_step_final_update_K"),
           "newton_steps_full_cycle": nwfc.get("steps_accepted"), "newton_iters_total_full_cycle": nwfc.get("iters_total_accepted"),
           "newton_iters_max_step_full_cycle": nwfc.get("iters_max_per_step"), "newton_last_update_K_full_cycle": nwfc.get("last_step_final_update_K"),
           "energy_rel_residual_50us": j["energy_rel_residual"], "energy_rel_residual_full_cycle": j.get("energy_rel_residual_full_cycle"),
           "above_6000K_extrapolated": j["above_6000K_extrapolated"], "wall_s": j["wall_s"], "git": j.get("git"), "run_date": j.get("run_date"),
           "json": str(jf.relative_to(ROOT).as_posix())}
    return row


def load_rows(outdir):
    sput = _sput()
    rows = []
    rc = outdir / "regression_check"
    for name in SCEN:  # cold references without a skin: the regression-check run (bit-identical to the 27 Sep baseline)
        jf = rc / f"{name}.json"
        if jf.exists():
            rows.append(_row(jf, "cold", 0.0, 1.0, sput, "ref_noskin"))
    jf = outdir / "single" / "Demo_hot" / "ref_noskin" / "Demo.json"
    if jf.exists():
        rows.append(_row(jf, "hot", 0.0, 1.0, sput, "ref_noskin"))
    for d in D_UM:
        for f in F:
            for name in SCEN:
                jf = outdir / "single" / "cold" / tag(d, f) / f"{name}.json"
                if jf.exists():
                    rows.append(_row(jf, "cold", d, f, sput, tag(d, f)))
            jf = outdir / "single" / "Demo_hot" / tag(d, f) / "Demo.json"
            if jf.exists():
                rows.append(_row(jf, "hot", d, f, sput, tag(d, f)))
    return rows


def multi_summary(outdir):
    out = {}
    for jf in sorted(outdir.glob("multi_*/*.json")):
        j = json.loads(jf.read_text(encoding="utf-8"))
        csv = jf.with_name(f"{j['scenario']}_per_shot.csv")
        r = np.loadtxt(csv, delimiter=",", skiprows=1)
        base_csv = ROOT / "runs" / "w_multi" / "baseline" / f"{j['scenario']}_per_shot.csv"
        b = np.loadtxt(base_csv, delimiter=",", skiprows=1)
        n = int(r[-1, 0])
        cmp_shot = min(n, int(b[-1, 0]))
        out[jf.parent.name] = {k: j.get(k) for k in ("scenario", "n_shots", "converged", "preshot_rise_per_shot_last10pct_K", "last_preshot_K",
                                                   "first_shot_peak_K", "last_shot_peak_K", "max_interface_T_K", "last_shot_interface_T_K",
                                                   "max_melt_depth_um", "last_shot_melt_depth_um", "cum_ablated_nm", "last_shot_ablated_nm",
                                                   "energy_rel_residual", "newton_this_session", "options", "git", "wall_s_this_session")}
        out[jf.parent.name]["at_shot"] = cmp_shot
        out[jf.parent.name]["this_run_at_shot"] = {"preshot_surface_K": float(r[cmp_shot - 1, 1]), "peak_surface_K": float(r[cmp_shot - 1, 2]),
                                                   "peak_interface_K": float(r[cmp_shot - 1, 3]), "preshot_back_K": float(r[cmp_shot - 1, 4])}
        out[jf.parent.name]["baseline_at_shot"] = {"preshot_surface_K": float(b[cmp_shot - 1, 1]), "peak_surface_K": float(b[cmp_shot - 1, 2]),
                                                   "peak_interface_K": float(b[cmp_shot - 1, 3]), "preshot_back_K": float(b[cmp_shot - 1, 4]),
                                                   "source": str(base_csv.relative_to(ROOT).as_posix())}
        out[jf.parent.name]["interface_max_over_run_minus_925K"] = float(r[:, 3].max() - 925.0)
        out[jf.parent.name]["baseline_interface_max_to_same_shot_minus_925K"] = float(b[:cmp_shot, 3].max() - 925.0)
    return out


def fmt(x, nd=1):
    if x is None:
        return "–"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and (abs(x) >= 1e4 or (abs(x) < 1e-2 and x != 0)):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def write_md(outdir, out):
    rows, multi, cmds = out["single"], out["multi"], out["commands"]
    L = [f"# Degraded near-surface layer: sweep summary ({out['date']}, commit {out['git']})", "",
         "Motivation (article, Section 4.5): a real tungsten surface need not have bulk properties; depending on the surface treatment it can "
         "retain a defective layer, and that is where the short pulse is absorbed. The solver assumed bulk Tolias 2017 "
         "properties everywhere. Here a surface layer of thickness d has its solid-phase conductivity multiplied by f relative to bulk at the same "
         "temperature (density and specific heat at bulk; molten W undamaged), over undamaged bulk W. Effusivity scales as sqrt(f). "
         "**The factors are HYPOTHETICAL**: no measured effusivity of a machined, ground, polished or etched W surface was obtained; "
         "f = 0.75, 0.5, 0.25 bound the effect.", "",
         f"Solver options reused unchanged from 29 Sep 2026 (`skin_um`, `skin_factor`; `k_irr` = 1). Regression of the defaults against "
         f"results/baseline_pureW: **{'bit-identical' if out['regression']['bit_identical'] else 'DIFFERENCES'}** "
         f"(keys {', '.join(out['regression']['keys'])}; history, max-T profile and He-layer CSVs; `regression_check/REGRESSION.json`).", "",
         "Columns: margin to melting = 3695 K − peak; margins to the recrystallization onsets = onset − peak (negative: exceeded), onsets 1300 K "
         "(lower edge of the GA2020 band, Section 5.1) and 1523 K (one-hour anneal onset of pure W, Section 5.7); depth above = deepest point of the "
         "max-T depth grid (1 nm–200 µm, 240 log-spaced points) that reached the onset in the shot; evaporation from the run; sputtering from "
         "runs/sputtering (Eckstein yields; grade- and property-independent); erosion-limited shots = 1 mm/(evaporation + sputtering); interface = "
         "last W cell (1 µm from the W/steel interface), maximum in the shot cycle; Newton counts to 50 µs after ignition (first integration part, "
         "where the peak occurs) and over the full shot period; last update = largest |ΔT| of the final Newton iteration of the last accepted step; "
         "residual = energy-ledger residual (Section 3.11).", ""]
    for start, title in (("cold", "Single shots from the scenario initial temperature (cold start: 300 K Experimental, 600 K Prototype and Demo)"),
                         ("hot", "Demo single shots from the converged 10 Hz inter-shot state (hot start: uniform 921.1 K)")):
        L += [f"## {title}", "",
              "| scenario | d (µm) | f | √f | peak T (K) | t_peak (µs) | margin to melting (K) | melt (µm) | margin to 1300 K | margin to 1523 K | "
              "depth > 1300 K (µm) | depth > 1523 K (µm) | evap. (nm/shot) | sputter (nm/shot) | erosion-limited shots | interface max in shot (K) | "
              "pre-shot T next shot (K) | steps to 50 µs (rejected) | Newton iters total / max per step, to 50 µs | last update (K) | steps full cycle | "
              "iters total / max, full cycle | last update full cycle (K) | residual 50 µs | residual full cycle | wall (s) | git |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["start"] != start:
                continue
            L.append(f"| {r['scenario']} | {r['d_um']:g}{' (no skin)' if r['d_um'] == 0 else ''} | {r['f']:g} | {r['sqrt_f']:.3f} | "
                     f"{r['peak_surface_T_K']:.1f} | {r['t_peak_us']:.3f} | {r['margin_to_melting_K']:+.1f} | {r['melt_depth_um']:.3f} | "
                     f"{r['margin_to_1300K_K']:+.0f} | {r['margin_to_1523K_K']:+.0f} | {r['depth_above_1300K_um']:.2f} | {r['depth_above_1523K_um']:.2f} | "
                     f"{r['evaporation_nm_per_shot']:.2e} | {r['sputtering_nm_per_shot']:.2e} | {r['erosion_limited_shots_1mm']:.2e} | "
                     f"{r['interface_peak_in_shot_K']:.2f} | {fmt(r['pre_shot2_surface_T_K'], 2)} | "
                     f"{fmt(r['newton_steps_to_50us'])} ({fmt(r['newton_rejected_to_50us'])}) | {fmt(r['newton_iters_total_to_50us'])} / {fmt(r['newton_iters_max_step_to_50us'])} | "
                     f"{fmt(r['newton_last_update_K_to_50us'])} | {fmt(r['newton_steps_full_cycle'])} | "
                     f"{fmt(r['newton_iters_total_full_cycle'])} / {fmt(r['newton_iters_max_step_full_cycle'])} | {fmt(r['newton_last_update_K_full_cycle'])} | "
                     f"{r['energy_rel_residual_50us']:.1e} | {fmt(r['energy_rel_residual_full_cycle'])} | {r['wall_s']:.0f} | {r['git']} |")
        L.append("")
    L += ["## Range of the headline numbers across f = 1.0 → 0.25 (per d)", "",
          "| scenario | start | d (µm) | peak T at f = 1.0 → 0.25 (K) | Δpeak (K) | margin to melting at f = 0.25 (K) | melt at f = 0.25 (µm) | "
          "depth > 1300 K, f = 1.0 → 0.25 (µm) | depth > 1523 K, f = 1.0 → 0.25 (µm) | evaporation f = 1.0 → 0.25 (nm/shot) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["ranges"]:
        L.append(f"| {r['scenario']} | {r['start']} | {r['d_um']:g} | {r['peak_f1']:.0f} → {r['peak_f025']:.0f} | {r['dpeak']:+.0f} | {r['margin_f025']:+.0f} | "
                 f"{r['melt_f025']:.3f} | {r['d1300_f1']:.1f} → {r['d1300_f025']:.1f} | {r['d1523_f1']:.1f} → {r['d1523_f025']:.1f} | "
                 f"{r['evap_f1']:.1e} → {r['evap_f025']:.1e} |")
    L += ["", "## Indicators that do not depend on the thermal run (unchanged)", "",
          "- Helium damage-fluence count (Section 5.5): Demo 9 shots to 10¹⁸ m⁻² and 128 to 10¹⁹ m⁻² with the retention model, 8 and 83 at 100 % "
          "retention; Prototype 26 and 264; Experimental no count (single-shot design). Set by the ion spectrum, not by the W properties.",
          "- Mitteau design rule (Section 5.5): 3.0×10⁴, 1.0×10³ and 9.5 cycles; arithmetic on published heat flux factors, no model output.",
          "- Sputtering (Table 9): 2.0×10⁻⁵, 2.9×10⁻⁵ and 9.1×10⁻⁵ nm per shot; property-independent. The erosion-limited count above changes only through the evaporation.",
          "- The document has no failure criterion; these are indicators (Section 5.5).", ""]
    if multi:
        L += ["## Multi-shot check (production driver `run_w_multi.py`, unchanged solver options on the command line)", "",
              "| run | scenario | shots | converged | pre-shot rise last 10 % (K/shot) | first / last peak (K) | pre-shot surface at the last shot (K) | "
              "max interface (K) | interface at the compared shot: this run / pure-W baseline (K) | pre-shot surface at that shot: this run / baseline (K) | "
              "peak at that shot: this run / baseline (K) | max melt (µm) | evaporation last shot (nm) | residual | Newton iters total / max per step | wall (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, m in multi.items():
            nw = m.get("newton_this_session") or {}
            L.append(f"| {k} | {m['scenario']} | {m['n_shots']} | {m['converged']} | {fmt(m['preshot_rise_per_shot_last10pct_K'], 3)} | "
                     f"{m['first_shot_peak_K']:.0f} / {m['last_shot_peak_K']:.0f} | {m['last_preshot_K']:.1f} | {m['max_interface_T_K']:.2f} | "
                     f"shot {m['at_shot']}: {m['this_run_at_shot']['peak_interface_K']:.2f} / {m['baseline_at_shot']['peak_interface_K']:.2f} | "
                     f"{m['this_run_at_shot']['preshot_surface_K']:.2f} / {m['baseline_at_shot']['preshot_surface_K']:.2f} | "
                     f"{m['this_run_at_shot']['peak_surface_K']:.0f} / {m['baseline_at_shot']['peak_surface_K']:.0f} | {m['max_melt_depth_um']:.3f} | "
                     f"{m['last_shot_ablated_nm']:.2e} | {m['energy_rel_residual']:.1e} | {fmt(nw.get('iters_total_accepted'))} / {fmt(nw.get('iters_max_per_step'))} | "
                     f"{fmt(m['wall_s_this_session'], 0)} |")
        L.append("")
    L += ["## Commands and git hashes", "", "| run | command | git | start | end | return code |", "|---|---|---|---|---|---|"]
    for c in cmds:
        L.append(f"| {c['label']} | `{c['command']}` | {c['git']} | {c['start']} | {c['end']} | {c['returncode']} |")
    for k, m in multi.items():
        o = m["options"]
        L.append(f"| {k} | `python scripts/run_w_multi.py {m['scenario']} {m['n_shots']} skin_um={o['skin_um']:g} skin_factor={o['skin_factor']:g} "
                 f"--out=results/skin_sweep/{k}` | {m.get('git') or 'see multi_*/COMMAND.txt'} | – | – | – |")
    L += ["", "Reference runs: cold no-skin = `regression_check/` (run with the working tree that became commit 0be5c47, before it was "
          "committed, so its JSON files record the previous commit f728ed4); the 27 Sep baseline values are in results/baseline_pureW/w_single.", "",
          "## Notes", "",
          "- f = 1.0 with a skin is the mesh-only check: the three-layer mesh (skin cells from 1 nm growing by ×1.05, bulk starting at the last skin cell) "
          "differs from the two-layer production mesh; its difference from the no-skin reference is the mesh effect.",
          "- The hot start sets the whole wall, steel included, to 921.1 K, so its interface column is not the converged interface state; the "
          "multi-shot check is the interface test (Section 5.2).",
          "- `skin_status` in the JSON files is the 29 Sep label of the option (`annealed skin` for f = 1.0, `helium-bubble skin` otherwise); here "
          "every factor is a hypothetical surface-state factor.",
          "- No failure criterion exists in the document; `erosion-limited shots` is 1 mm divided by the per-shot loss (evaporation + sputtering)."]
    (outdir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def ranges(rows):
    out = []
    for (scen, start, d) in sorted({(r["scenario"], r["start"], r["d_um"]) for r in rows if r["d_um"] > 0},
                                   key=lambda x: (SCEN.index(x[0]), x[1], x[2])):
        sel = {r["f"]: r for r in rows if r["scenario"] == scen and r["start"] == start and r["d_um"] == d}
        if 1.0 not in sel or 0.25 not in sel:
            continue
        a, b = sel[1.0], sel[0.25]
        out.append({"scenario": scen, "start": start, "d_um": d, "peak_f1": a["peak_surface_T_K"], "peak_f025": b["peak_surface_T_K"],
                    "dpeak": b["peak_surface_T_K"] - a["peak_surface_T_K"], "margin_f025": b["margin_to_melting_K"], "melt_f025": b["melt_depth_um"],
                    "d1300_f1": a["depth_above_1300K_um"], "d1300_f025": b["depth_above_1300K_um"],
                    "d1523_f1": a["depth_above_1523K_um"], "d1523_f025": b["depth_above_1523K_um"],
                    "evap_f1": a["evaporation_nm_per_shot"], "evap_f025": b["evaporation_nm_per_shot"]})
    return out


def figure(outdir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Experimental", "cold", "Experimental, 20 MJ, R = 5 m, from 300 K"), ("Prototype", "cold", "Prototype, 50 MJ, R = 6.5 m, from 600 K"),
              ("Demo", "cold", "Demo, 154 MJ, R = 6.5 m, from 600 K"), ("Demo", "hot", "Demo, hot start from the converged 921.1 K state")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    styles = {1.0: ("tab:blue", "o"), 3.0: ("tab:orange", "s"), 10.0: ("tab:red", "^")}
    for ax, (scen, start, title) in zip(axes, panels):
        for d, (c, mk) in styles.items():
            pts = sorted([(r["f"], r["peak_surface_T_K"]) for r in rows if r["scenario"] == scen and r["start"] == start and r["d_um"] == d])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], c=c, marker=mk, lw=1.3, label=f"d = {d:g} µm")
        ref = next((r for r in rows if r["scenario"] == scen and r["start"] == start and r["d_um"] == 0), None)
        if ref:
            ax.plot([1.0], [ref["peak_surface_T_K"]], "k*", ms=11, zorder=5, label=f"no layer (baseline): {ref['peak_surface_T_K']:.0f} K")
        ax.axvline(1.0, c="0.6", lw=0.8, ls=":")
        if scen == "Demo":
            ax.axhline(T_MELT, c="k", ls="--", lw=0.9)
            ax.text(0.62, T_MELT - 42, "W melting point 3695 K", fontsize=7.5, ha="center")
        ax.set_xlim(0.2, 1.05)
        ax.invert_xaxis()
        ax.set_xticks(F)
        ax.set_xticklabels([f"{f:g}\n(√f = {np.sqrt(f):.2f})" for f in F], fontsize=7.5)
        ax.set_xlabel("conductivity factor f of the surface layer (–)")
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("peak W surface temperature (K)")
    fig.suptitle("Single shot: peak W surface temperature against the conductivity factor of a degraded surface layer of thickness d "
                 "(solid-phase k × f, ρ and c at bulk; HYPOTHETICAL factors)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_skin_sweep_peak_vs_f.{ext}", dpi=170)
    plt.close(fig)
    try:
        from PIL import Image
        jpg = ROOT / "docs" / "figures" / "full" / "skin_sweep_peak_vs_f.jpg"
        Image.open(outdir / "fig_skin_sweep_peak_vs_f.png").convert("RGB").save(jpg, quality=85, optimize=True)
        print("manuscript copy:", jpg)
    except Exception as e:  # noqa: BLE001
        print("JPEG copy not written:", e)


def summary(outdir):
    rows = load_rows(outdir)
    rc = outdir / "regression_check" / "REGRESSION.json"
    reg = json.loads(rc.read_text(encoding="utf-8")) if rc.exists() else {"bit_identical": False, "keys": list(REG_KEYS)}
    cmdf = outdir / "single" / "commands.json"
    cmds = json.loads(cmdf.read_text(encoding="utf-8")) if cmdf.exists() else []
    out = {"git": git_hash(), "date": time.strftime("%Y-%m-%d %H:%M"), "regression": reg, "grid": {"d_um": list(D_UM), "f": list(F), "sqrt_f": [float(np.sqrt(f)) for f in F]},
           "factor_status": "HYPOTHETICAL: no measured effusivity of a treated W surface was obtained",
           "single": rows, "ranges": ranges(rows), "multi": multi_summary(outdir), "commands": cmds,
           "unchanged_indicators": {"He_shots_Demo": "9 / 128 (retention model); 8 / 83 at 100 % retention (Section 5.5)",
                                    "He_shots_Prototype": "26 / 264 (retention model)", "Mitteau_cycles": "3.0e4, 1.0e3, 9.5 (Section 5.5)",
                                    "sputtering_nm_per_shot": _sput()}}
    (outdir / "summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    write_md(outdir, out)
    figure(outdir, rows)
    print(f"summary: {len(rows)} single runs, {len(out['multi'])} multi runs -> {outdir / 'summary.md'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "summary"
    outdir = Path(next((a.split("=", 1)[1] for a in args if a.startswith("--out=")), ROOT / "results" / "skin_sweep"))
    outdir.mkdir(parents=True, exist_ok=True)
    if cmd == "regression":
        sys.exit(0 if regression(outdir) else 1)
    elif cmd == "launch":
        n = int(next((a.split("=", 1)[1] for a in args if a.startswith("--jobs=")), 4))
        launch(outdir, n, "--dry" in args)
    elif cmd == "summary":
        summary(outdir)
    else:
        raise SystemExit(__doc__)
