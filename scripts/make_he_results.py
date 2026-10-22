"""Figures, tables and JSON/CSV for the He-retention results (reads runs/he_retention/*; writes runs/he_retention/results/).

  python scripts/make_he_results.py [sources|runs|sens|all]

sources : per-shot He and dpa profiles, He/dpa (section 5.1/5.2)
runs    : retained He vs shots, release fraction, profiles, shots to thresholds, comparison with 100 % retention
sens    : tornado chart from runs/he_retention/sens/Demo_<case>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ife_wall.he import analysis as an  # noqa: E402
from ife_wall.he import constants as C  # noqa: E402
from ife_wall.he import damage as dm  # noqa: E402
from ife_wall.he.grid import make_grid  # noqa: E402
from ife_wall.he.sources import ShotSources  # noqa: E402

OUT = ROOT / "runs" / "he_retention" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SCEN = ["Experimental", "Prototype", "Demo"]
MULTI = ["Prototype", "Demo"]   # repeated-shot scenarios; Experimental = single shot only (no repetition rate)
COL = an.SCEN_COLOR


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 120, "savefig.bbox": "tight", "axes.titlesize": 13})
    return plt


# ------------------------------------------------------------------------------------------ sources
def sources():
    plt = style()
    edges = make_grid()
    res = {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for name in SCEN:
        S = ShotSources(name, edges)
        x = S.x * 1e6
        he = S.he_profile() / dm.N_AT * 1e6          # appm per shot
        dpa = S.dpa_profile()
        m5 = S.x < 5e-6
        he_int5 = (S.he_profile()[m5] * S.dx[m5]).sum()          # He/m^2 within 5 um
        dpa_avg5 = (dpa[m5] * S.dx[m5]).sum() / S.dx[m5].sum()
        he_avg5_appm = he_int5 / S.dx[m5].sum() / dm.N_AT * 1e6
        res[name] = dict(
            He_incident_burn=S.he_per_shot_incident("burn"), He_incident_debris=S.he_per_shot_incident("debris"),
            He_incident_total=S.he_per_shot_incident(), He_implanted_total=S.he_per_shot_implanted(),
            He_implanted_burn=S.he_per_shot_implanted("burn"),
            peak_dpa_per_shot=float(dpa.max()), depth_of_peak_dpa_nm=float(x[np.argmax(dpa)] * 1e3),
            peak_dpa_per_shot_below_10nm=float(dpa[S.x > 10e-9].max()),
            mean_dpa_0_5um=float(dpa_avg5), mean_He_0_5um_appm=float(he_avg5_appm),
            He_over_dpa_0_5um_appm_per_dpa=float(he_avg5_appm / dpa_avg5),
            He_over_dpa_total_appm_per_dpa=float(S.he_per_shot_implanted() / ((dpa * S.dx).sum()) / dm.N_AT * 1e6),
            peak_He_appm_per_shot=float(he.max()), depth_of_peak_He_um=float(x[np.argmax(he)]),
            displacements_per_m2=S.displacements_per_m2(),
            dpa_arc_over_nrt=None)
        # arc-dpa total relative to NRT
        S2 = ShotSources(name, edges, dpa_kind="arc")
        res[name]["dpa_arc_over_nrt"] = S2.displacements_per_m2() / S.displacements_per_m2()
        he_b, he_d = S.he_profile("burn") / dm.N_AT * 1e6, S.he_profile("debris") / dm.N_AT * 1e6
        sel = x > 1e-3
        ax[0].loglog(x[sel], he[sel], color=COL[name], lw=2, label=name)
        if name == "Demo":
            ax[0].loglog(x[sel], he_b[sel], color=COL[name], lw=1, ls="--", label="Demo, burn He only")
            ax[0].loglog(x[sel], he_d[sel], color=COL[name], lw=1, ls=":", label="Demo, debris He only")
        ax[1].loglog(x[sel], dpa[sel], color=COL[name], lw=2, label=name)
        ax[2].loglog(x[sel], he[sel] / np.maximum(dpa[sel], 1e-300), color=COL[name], lw=2, label=name)
        res[name]["profile"] = dict(x_um=x.tolist(), He_appm_per_shot=he.tolist(), dpa_per_shot=dpa.tolist())
    ax[0].set(xlabel="depth ($\\mu$m)", ylabel="implanted He per shot (appm)", title="He implanted per shot", xlim=(1e-3, 100), ylim=(1e-8, 10))
    ax[0].legend(fontsize=9)
    ax[1].set(xlabel="depth ($\\mu$m)", ylabel="dpa per shot (NRT, $E_d$ = 90 eV)", title="Displacement damage per shot", xlim=(1e-3, 100), ylim=(1e-9, 1e-3))
    ax[1].legend(fontsize=9)
    nd = C.NEUTRON_HE_PER_DPA
    ax[2].axhspan(0.14, 0.9, color="0.8", alpha=0.7)
    ax[2].text(2e-3, 1.3, "neutrons, magnetic-fusion first wall: 0.14-0.9", fontsize=9)
    ax[2].set(xlabel="depth ($\\mu$m)", ylabel="He / dpa (appm per dpa)", title="He per dpa from the target ions", xlim=(1e-3, 100), ylim=(1e-1, 1e9))
    ax[2].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sources_per_shot.pdf")
    (OUT / "sources.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    import csv
    with open(OUT / "sources_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = [k for k in res["Demo"] if k != "profile"]
        w.writerow(["scenario"] + keys)
        for n in SCEN:
            w.writerow([n] + [res[n][k] for k in keys])
    for n in SCEN:
        print(n, {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in res[n].items() if k != "profile"})
    return res



# ------------------------------------------------------------------------------------------ Figure 1
def fig1_picture():
    plt = style()
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Ellipse
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.3))
    rng = np.random.default_rng(4)
    titles = ["1  He injection", "2  Vacancy production", "3  Diffusion", "4  Clusters and bubbles"]
    subs = ["target He ions stop\n0.2-5 $\\mu$m below the surface", "ions knock W atoms out of place:\nvacancies + self-interstitials (dpa)",
            "He, vacancies and interstitials\nmove (much faster when hot)", "He fills vacancies; bubbles grow;\nenough He near the surface: blisters"]
    nx, ny = 9, 7
    for k, ax in enumerate(axs):
        ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-1.2, ny + 1.6); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(titles[k], fontsize=14, fontweight="bold")
        ax.text(nx / 2 - 0.5, -1.0, subs[k], ha="center", va="top", fontsize=10.5)
        vac = {(3, 3), (4, 2), (5, 4)} if k >= 1 else set()
        if k == 3:
            vac = {(4, 4)}
        for i in range(nx):
            for j in range(ny):
                if (i, j) in vac:
                    if k == 3:
                        continue
                    ax.add_patch(Circle((i, j), 0.36, fill=False, ls="--", ec="C3", lw=1.5))
                else:
                    ax.add_patch(Circle((i, j), 0.36, fc="0.78", ec="0.45", lw=0.8))
        ax.plot([-0.5, nx - 0.5], [ny - 0.4, ny - 0.4], color="k", lw=2)       # surface
        ax.text(-0.4, ny + 0.2, "surface", fontsize=10)
        if k == 0:
            for x_, y_ in [(2, 4.6), (4, 3.2), (6, 4.0), (7, 2.6), (3, 3.0)]:
                ax.add_patch(Circle((x_ + 0.5, y_), 0.17, fc="C3", ec="k", lw=0.6))
            for x_ in (2.5, 4.5, 6.5, 7.5):
                ax.add_patch(FancyArrowPatch((x_, ny + 1.5), (x_, ny - 0.5), arrowstyle="-|>", mutation_scale=16, color="C3", lw=1.6))
            ax.text(0.0, ny + 1.25, "He$^{2+}$ ions", color="C3", fontsize=11, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
        if k == 1:
            ax.add_patch(FancyArrowPatch((1.0, ny + 1.4), (3.0, 3.4), arrowstyle="-|>", mutation_scale=16, color="C0", lw=2))
            ax.text(0.0, ny + 1.2, "D, T, He ion", color="C0", fontsize=11, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
            for x_, y_ in [(3.5, 3.5), (4.4, 2.6), (5.4, 4.4)]:
                ax.add_patch(Circle((x_, y_), 0.12, fc="C1", ec="k", lw=0.6))
            ax.text(4.9, 0.6, "vacancy (empty site)", color="C3", fontsize=10, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9)); ax.text(4.9, -0.1, "self-interstitial", color="C1", fontsize=10, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
        if k == 2:
            for x_, y_ in [(3.3, 3.2), (5.4, 4.4), (4.2, 1.6)]:
                ax.add_patch(Circle((x_, y_), 0.17, fc="C3", ec="k", lw=0.6))
            ax.add_patch(FancyArrowPatch((3.3, 3.5), (3.9, 2.6), arrowstyle="-|>", mutation_scale=14, color="C3", lw=1.6))
            ax.add_patch(FancyArrowPatch((5.4, 4.2), (4.7, 3.4), arrowstyle="-|>", mutation_scale=14, color="C3", lw=1.6))
            ax.add_patch(FancyArrowPatch((4.2, 1.9), (4.2, 2.5), arrowstyle="-|>", mutation_scale=14, color="C3", lw=1.6))
            ax.text(6.0, 5.4, "He", color="C3", fontsize=11, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
            ax.text(0.0, 5.3, "He escapes through\nthe surface", fontsize=10, color="0.25", bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
        if k == 3:
            ax.add_patch(Circle((4, 4), 0.62, fc="white", ec="C3", lw=2))
            for dx, dy in [(-0.25, 0.2), (0.25, 0.2), (0, -0.3), (0.0, 0.25), (-0.3, -0.15), (0.3, -0.15)]:
                ax.add_patch(Circle((4 + dx, 4 + dy), 0.12, fc="C3", ec="k", lw=0.5))
            ax.text(4.8, 2.3, "He$_k$V, k = 1...7", color="C3", fontsize=10.5, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
            ax.add_patch(Ellipse((2.0, ny - 0.25), 3.2, 1.0, fc="0.9", ec="k", lw=1.2, zorder=0))
            ax.text(2.0, ny + 0.75, "blister", fontsize=10.5, ha="center", bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_four_steps.pdf")
    fig.savefig(OUT / "fig1_four_steps.png", dpi=150)


# ------------------------------------------------------------------------------------------ runs
def profile_from_checkpoint(path, params=None):
    from ife_wall.he.fast import FastHe
    z = np.load(Path(path) / "checkpoint.npz", allow_pickle=True)
    fm = FastHe(make_grid(), params or C.baseline())
    return int(z["shot_no"]), fm.profiles(z["y"])


def threshold_table(base, src):
    """shots to each threshold: old (100 % retention) and new (baseline run)."""
    out = {}
    for sc in MULTI:   # Experimental is a single-shot design: no shot counts are derived for it
        if sc not in base:
            continue
        t = base[sc]["t"]
        d = {}
        for name, thr in an.THRESHOLDS.items():
            new = an.shots_to(thr, t["shot"], t["He_retained"])
            new["label"] = "computed" if new["kind"] == "computed" else "extrapolated lower bound"
            old_all = thr / src[sc]["He_implanted_total"]
            old_burn = thr / src[sc]["He_implanted_burn"]
            d[name] = dict(threshold=thr, old_all_He=old_all, old_burn_only=old_burn, new=new,
                           time_new=an.fmt_time(an.time_s(new["value"], sc)),
                           time_new_lower=an.fmt_time(an.time_s(new["lower"], sc)),
                           time_new_upper=an.fmt_time(an.time_s(new["upper"], sc)) if new["upper"] is not None else None,
                           time_old_all=an.fmt_time(an.time_s(old_all, sc)), time_old_burn=an.fmt_time(an.time_s(old_burn, sc)))
        out[sc] = d
    return out


def runs():
    plt = style()
    src = json.loads((OUT / "sources.json").read_text(encoding="utf-8")) if (OUT / "sources.json").exists() else sources()
    base = {}
    for sc in SCEN:
        p = an.RUNS / f"{sc}_baseline"
        if (p / "totals.csv").exists():
            base[sc] = an.load(p)
    thr = threshold_table(base, src)
    (OUT / "thresholds.json").write_text(json.dumps(thr, indent=1, default=float), encoding="utf-8")
    # ---- retained He vs shots
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, sc in zip(axs, MULTI):
        if sc not in base:
            continue
        t = base[sc]["t"]
        n = t["shot"]
        ax.loglog(n, t["He_in"], color="0.55", lw=1.5, label="100 % retention (old assumption)")
        ax.loglog(n, t["He_retained"], color=COL[sc], lw=2.5, label="total retained")
        ax.loglog(n, t["He_trapped"] + t["He_clusters"], color=COL[sc], lw=1.5, ls="--", label="trapped in vacancies/clusters")
        ax.loglog(n, np.maximum(t["He_mobile"], 1e5), color=COL[sc], lw=1.5, ls=":", label="mobile")
        N = n[-1]
        w = max(25, int(0.1 * len(n)))
        inc = (t["He_retained"][-1] - t["He_retained"][-1 - w]) / (n[-1] - n[-1 - w])
        nn = np.geomspace(N, 1e6, 50)
        ax.loglog(nn, t["He_retained"][-1] + inc * (nn - N), color=COL[sc], lw=2, ls="-.", alpha=0.6,
                  label="extrapolated lower bound (last slope kept)")
        for name, v in an.THRESHOLDS.items():
            ax.axhline(v, color="k", lw=0.6, ls=":")
        ax.axhspan(1e18, 1e19, color="C3", alpha=0.07)
        ax.axhspan(1e21, 1e22, color="C3", alpha=0.07)
        ax.text(1.2, 1.3e19, "pulsed damage threshold", fontsize=8.5)
        ax.text(1.2, 1.3e22, "blistering range", fontsize=8.5)
        ax.axvline(N, color="0.4", lw=0.8)
        ax.text(N * 0.9, 2e10, "computed | extrapolated lower bound", fontsize=8.5, ha="right", rotation=90, va="bottom")
        ax.set(xlabel="shot number", xlim=(1, 1e6), ylim=(1e10, 1e23), title=f"{sc} ({an.REP_HZ[sc]:g} Hz)")
        ax.title.set_fontsize(11)
    axs[0].set_ylabel("He retained in the armour (He/m$^2$)")
    axs[0].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_retained_vs_shots.pdf")
    # ---- release fraction
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for sc in MULTI:
        if sc not in base:
            continue
        t = base[sc]["t"]
        n = t["shot"]
        axs[0].semilogx(n, 100 * t["He_surface"] / t["He_in"], color=COL[sc], lw=2.5, label=sc)
        w = 10
        idx = np.arange(w, len(n))
        dR = t["He_retained"][idx] - t["He_retained"][idx - w]
        dI = t["He_in"][idx] - t["He_in"][idx - w]
        axs[1].semilogx(n[idx], 100 * (1 - dR / dI), color=COL[sc], lw=2.5, label=sc)
    axs[0].set(xlabel="shot number", ylabel="released / implanted (%)", title="Cumulative release fraction", ylim=(0, 100))
    axs[1].set(xlabel="shot number", ylabel="He lost from each new shot's He (%)", title="Release fraction of recent shots (10-shot window)", ylim=(0, 100))
    for ax in axs:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_release_fraction.pdf")
    # ---- depth profiles
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    peaks = {}
    for ax, sc in zip(axs, SCEN):
        if sc not in base:
            continue
        prof = dict(base[sc]["prof"])
        if sc == "Experimental":      # single-shot design: only the profile after the one shot is used
            prof = {k: v for k, v in prof.items() if k == 1}
        else:
            try:
                k_last, p_last = profile_from_checkpoint(an.RUNS / f"{sc}_baseline")
                prof[k_last] = {k: v for k, v in p_last.items()}
            except Exception as e:
                print("checkpoint profile failed", sc, e)
        if not prof:
            continue
        want = [1, 10, 100, 1000]
        shots_av = sorted(prof)
        pick = [k for k in want if k in prof]
        if shots_av[-1] not in pick and shots_av[-1] > (pick[-1] if pick else 0):
            pick.append(shots_av[-1])
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(pick)))
        peaks[sc] = {}
        for k, c in zip(pick, cmap):
            pr = prof[k]
            x = pr["x"] * 1e6
            tot = pr["He_trapped"] + pr["He"]
            lab = "after the single shot" if sc == "Experimental" else f"after {k} shot" + ("s" if k > 1 else "")
            ax.loglog(x, np.maximum(tot, 1e-6), color=c, lw=2, label=lab)
            peaks[sc][k] = dict(peak_He_appm=float(tot.max()), peak_He_at_percent=float(tot.max() / 1e4), depth_um=float(x[np.argmax(tot)]))
        ax.axhline(1e4, color="C3", lw=1, ls="--")
        ax.text(2e-3, 1.3e4, "1 at.%", color="C3", fontsize=9)
        ax.set(xlabel="depth ($\\mu$m)", title=sc, xlim=(1e-3, 1e3), ylim=(1e-3, 1e5))
        ax.legend(fontsize=9, loc="lower left")
    axs[0].set_ylabel("He (mobile + trapped), appm")
    fig.tight_layout()
    fig.savefig(OUT / "fig_depth_profiles.pdf")
    (OUT / "peak_He.json").write_text(json.dumps(peaks, indent=1), encoding="utf-8")
    import csv
    with open(OUT / "thresholds_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "threshold", "old_100pct_all_He_arithmetic", "old_100pct_burn_only_arithmetic", "new_value", "new_label", "new_lower_bound", "time_new"])
        for sc, d in thr.items():
            for name, r in d.items():
                w.writerow([sc, r["threshold"], f"{r['old_all_He']:.4g}", f"{r['old_burn_only']:.4g}", f"{r['new']['value']:.4g}",
                            r["new"]["label"], f"{r['new']['lower']:.4g}", r["time_new"]])
    for sc, d in thr.items():
        print(sc, {n: (round(r["new"]["value"]), r["new"]["kind"], r["time_new"], f"old {r['old_burn_only']:.0f}-{r['old_all_He']:.0f}") for n, r in d.items()})
    print("peaks", json.dumps(peaks)[:700])
    return thr


def sens():
    plt = style()
    d = an.RUNS / "sens"
    base = an.load(an.RUNS / "Demo_baseline")
    rows = {}
    for p in sorted(d.glob("Demo_*")):
        case = p.name[5:]
        if not (p / "totals.csv").exists():
            continue
        rows[case] = an.load(p)
    if not rows:
        return
    NS_LIST = (300, 100)
    Ns = 300
    nb = len(base["t"]["shot"])
    summ = {}
    for case, r in rows.items():
        t = r["t"]
        Nc = next((N for N in NS_LIST if len(t["shot"]) >= N and nb >= N), None)
        if Nc is None:
            continue
        Rb_c = base["t"]["He_retained"][Nc - 1]
        summ[case] = dict(shots=len(t["shot"]), N_compare=Nc, retained_at_N=float(t["He_retained"][Nc - 1]), baseline_at_N=float(Rb_c),
                          ratio_to_baseline=float(t["He_retained"][Nc - 1] / Rb_c),
                          retained_fraction_at_N=float(t["He_retained"][Nc - 1] / t["He_in"][Nc - 1]),
                          last_increment_fraction=float((t["He_retained"][-1] - t["He_retained"][-26]) / (t["He_in"][-1] - t["He_in"][-26])) if len(t["shot"]) > 30 else None,
                          shots_to={n: an.shots_to(v, t["shot"], t["He_retained"]) for n, v in an.THRESHOLDS.items()},
                          V_free_per_m2=float(t["V_free"][-1]))
    Rb = base["t"]["He_retained"][Ns - 1]
    summ["_N_s"] = Ns
    summ["_baseline"] = dict(retained_at_Ns=float(Rb), retained_fraction=float(Rb / base["t"]["He_in"][Ns - 1]),
                             shots_to={n: an.shots_to(v, base["t"]["shot"], base["t"]["He_retained"]) for n, v in an.THRESHOLDS.items()})
    (OUT / "sens_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    groups = [("He migration energy $E_m$ (0.06 / 0.28 eV)", "Em_He_low", "Em_He_high"),
              ("He-V binding energies (low / high set)", "Eb_low", "Eb_high"),
              ("Vacancy production ($\\xi$ = 0.21 / arc-dpa)", "xi_low", "dpa_arc"),
              ("Surface conductance (x 0.1)", "surface_0p1", None),
              ("Base temperature (-50 / +50 K)", "T_m50", "T_p50"),
              ("Rising pre-shot temperature (first shots)", "T_rising", None),
              ("Level 2: self-trapping + trap mutation (on)", "level2", None),
              ("Flat 2.4 eV for k > 4", "Eb_flat", None),
              ("Attempt frequency (1e12 / 1.3e13 s$^{-1}$)", "nu_low", "nu_high"),
              ("Dislocation density (2e10 / 1e14 m$^{-2}$)", "rho_d_low", "rho_d_high"),
              ("Vacancy migration energy (1.5 / 2.0 eV)", "Em_V_low", "Em_V_high"),
              ("V-I radius 0.465 nm", "riv_high", None),
              ("SIA migration (0.165 eV)", "Em_I_high", None),
              ("No I + He-V annihilation", "no_ann", None),
              ("No neutron source", "no_neutron", None)]
    items = []
    for label, lo, hi in groups:
        vals = [(summ[c]["ratio_to_baseline"] - 1) * 100 if c and c in summ else None for c in (lo, hi)]
        if any(v is not None for v in vals):
            items.append((label, vals, max(abs(v) for v in vals if v is not None)))
    items.sort(key=lambda a: a[2])
    fig, ax = plt.subplots(figsize=(10, 0.46 * len(items) + 1.6))
    for i, (label, vals, _) in enumerate(items):
        for v, col in zip(vals, ("C0", "C3")):
            if v is not None:
                ax.barh(i, v, color=col, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([a[0] for a in items], fontsize=10)
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="change of retained He at 300 shots (%)")
    ax.text(0.99, 0.02, "blue: first (low) variant, red: second (high) variant", transform=ax.transAxes, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tornado.pdf")
    for c, v in summ.items():
        if not c.startswith("_"):
            print(f"{c:14s} N={v['N_compare']} ratio {v['ratio_to_baseline']:.3f}  frac {v['retained_fraction_at_N']:.3f}  shots {v['shots']}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fig1", "all"):
        fig1_picture()
    if what in ("sources", "all"):
        sources()
    if what in ("runs", "all"):
        runs()
    if what in ("sens", "all"):
        sens()
