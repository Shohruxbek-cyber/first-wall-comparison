"""Our dpa per shot against Sharafat et al., J. Nucl. Mater. 347 (2005) 217, Table 5 (HAPL, 154 MJ NRL target, R = 6.5 m).

Run from the repository root:   PYTHONUTF8=1 python scripts/he_compare_sharafat.py
Reads:   references/sharafat2005_engineered_W.txt (Table 5, Sec. 6.1; a text copy of the paper, not
         distributed: without it the provenance check is skipped), data/srim/trim*/VACANCY.txt,
         runs/he_retention/results/sources.json (tie-out only), the He model (src/ife_wall/he).
Writes:  runs/he_retention/results/sharafat_comparison.json (every number printed below) and prints the tables.

Nothing here is tuned: our side is ShotSources("Demo") with the baseline settings (NRT, Ed = 90 eV), plus the
Ed = 25 eV NRT-type and arc-dpa variants of the same code.  Their side is read from the paper and re-derived.
Items marked QUOTED are literature values that are NOT computed here; items marked INFERRED are our reading of
their numbers (the paper does not say).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ife_wall import srim  # noqa: E402
from ife_wall.he import damage as dm  # noqa: E402
from ife_wall.he.grid import make_grid  # noqa: E402
from ife_wall.he.sources import HE_SPECIES, ShotSources  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

OUT = ROOT / "runs" / "he_retention" / "results" / "sharafat_comparison.json"
REF_TXT = ROOT / "references" / "sharafat2005_engineered_W.txt"
H_SPECIES = ("Protons", "Deuterons", "Tritons")
L5 = 5e-6                      # m, their penetration depth
N_AT = dm.N_AT                 # 6.306e28 m^-3, the value they imply too (7.382e22 / 1.17e-6)
EV, MEV_J = dm.EV, 1.602176634e-13
FULL_CASCADE_QUOTED = 2.0      # QUOTED (Wielunska 2020, 20 MeV W in W; Stoller 2013): NOT computed here
REP_HZ_SHARAFAT = 5.0          # their repetition rate (Table 3 / Sec. 4.4 text)

# --------------------------------------------------------------------------------------------- their Table 5
# printed rows, in printed order (labels, values and units line up one to one in the PDF, page 237)
T5 = dict(He_per_shot=1.96e20, area_m2=531.0, depth_m=5e-6, volume_m3=2.655e-3, He_per_m3=7.382e22, frac_appm=1.17,
          vac_per_He=308.0, vac_per_H=60.0, dpa_per_shot=4.3e-4, on_time_us=2.9, inst_dpa_rate=148.0,
          inst_He_per_dpa_appm=2721.0, qss_dpa_rate=2.15e-3, qss_He_rate=5.85e-6)
TEXT_226, TEXT_2725 = 226.0, 2725.0   # Sec. 6.1 text: "(226 dpa/s)" and "about 2725 He/dpa"


def check_against_text():
    """The numbers above must be present in the local copy of the paper (provenance check)."""
    txt = REF_TXT.read_text(encoding="utf-8", errors="replace")
    i0, i1 = txt.index("Table 5 Tungsten damage parameters"), txt.index("sites, which serve as bubble nucleation")
    block = txt[i0:i1]
    toks = ["1.96", "531", "2.655", "7.382", "1.17", "308", "60", "4.3", "2.9", "148", "2721", "2.15", "5.85"]
    missing = [t for t in toks if t not in block]
    sec = txt[txt.index("6.1. Ion implantation damage"):txt.index("6.2. Helium transport")]
    return dict(table5_tokens_missing=missing, text_226_present="226 dpa/s" in sec, text_2725_present="2725 He/dpa" in sec,
                SRIM_range_only="SRIM-2004" in sec, Ed_stated=bool(re.search(r"displacement energy|threshold|\bE_?d\b|Kinchin|NRT", sec, re.I)))


def rederive():
    t = T5
    he_m2 = t["He_per_shot"] / t["area_m2"]
    vol = t["area_m2"] * t["depth_m"]
    he_m3 = t["He_per_shot"] / vol
    frac = he_m3 / N_AT
    rows = []

    def add(name, formula, comp, printed):
        rows.append(dict(name=name, formula=formula, computed=comp, printed=printed,
                         rel_diff=None if printed is None else comp / printed - 1))
    add("He per m2 per shot", "He / area", he_m2, None)
    add("implanted volume (m3)", "area x 5 um", vol, t["volume_m3"])
    add("He per m3", "He / volume", he_m3, t["He_per_m3"])
    add("fractional He (appm)", "He per m3 / N_at, N_at = 6.306e28", frac * 1e6, t["frac_appm"])
    add("dpa from He only", "frac x 308", frac * t["vac_per_He"], t["dpa_per_shot"])
    add("dpa from He + 1 H per He", "frac x (308 + 60)", frac * (t["vac_per_He"] + t["vac_per_H"]), t["dpa_per_shot"])
    implied = t["dpa_per_shot"] / (t["frac_appm"] * 1e-6)
    add("instantaneous damage rate (dpa/s)", "dpa per shot / 2.9 us", t["dpa_per_shot"] / (t["on_time_us"] * 1e-6), t["inst_dpa_rate"])
    add("instantaneous He/dpa (appm/dpa)", "1.17 appm / 4.3e-4", t["frac_appm"] / t["dpa_per_shot"], t["inst_He_per_dpa_appm"])
    add("quasi-steady dpa rate (dpa/s)", "dpa per shot x 5 Hz", t["dpa_per_shot"] * REP_HZ_SHARAFAT, t["qss_dpa_rate"])
    add("quasi-steady He rate (He/at/s)", "frac x 5 Hz", t["frac_appm"] * 1e-6 * REP_HZ_SHARAFAT, t["qss_He_rate"])
    # rounding band of the implied H per He: dpa 4.25-4.35e-4, frac 1.165-1.175 appm
    lo = 4.25e-4 / 1.175e-6
    hi = 4.35e-4 / 1.165e-6
    extra = dict(He_per_m2=he_m2, He_per_m3=he_m3, frac_appm=frac * 1e6, implied_vac_per_He_total=implied,
                 implied_H_per_He=(implied - t["vac_per_He"]) / t["vac_per_H"],
                 implied_H_per_He_band=((lo - t["vac_per_He"]) / t["vac_per_H"], (hi - t["vac_per_He"]) / t["vac_per_H"]),
                 he_only_dpa=frac * t["vac_per_He"], dpa_averaged_over_5um=True,
                 text_226_implied_on_time_us=t["dpa_per_shot"] / TEXT_226 * 1e6, text_vs_table_dpa_rate=(TEXT_226, t["inst_dpa_rate"]),
                 text_vs_table_He_per_dpa=(TEXT_2725, t["inst_He_per_dpa_appm"]))
    return rows, extra


# --------------------------------------------------------------------------------------------- our side
def mean_first(S, prof, L=L5):
    """mean of a per-cell profile over the first L metres (partial cell weighted, exact at L)."""
    cum = np.concatenate([[0.0], np.cumsum(prof * S.dx)])
    return float(np.interp(L, S.edges, cum)) / L


def frac_below(S, prof, L=L5):
    cum = np.concatenate([[0.0], np.cumsum(prof * S.dx)])
    return 1.0 - float(np.interp(L, S.edges, cum)) / cum[-1]


def analyse(S):
    """All our numbers for one ShotSources object."""
    prof = S.dpa_profile()
    D = S.displacements_per_m2()
    rows = []
    for key in sorted({(s.group, s.species) for s in S.sub}, key=lambda k: (k[0] != "burn", k[1])):
        idx = [i for i, s in enumerate(S.sub) if (s.group, s.species) == key]
        n = sum(S.sub[i].n for i in idx)
        d = sum(float((S.dpa_prof[i] * S.dx).sum() * N_AT) for i in idx)
        eE = sum(S.sub[i].n * S.sub[i].E for i in idx) / n / MEV_J
        rows.append(dict(group=key[0], species=key[1], ions_per_m2=n, disp_per_m2=d, fraction=d / D, disp_per_ion=d / n,
                         mean_E_MeV=eE))

    def agg(pred):
        r = [x for x in rows if pred(x)]
        n, d = sum(x["ions_per_m2"] for x in r), sum(x["disp_per_m2"] for x in r)
        return dict(ions_per_m2=n, disp_per_m2=d, fraction=d / D, disp_per_ion=d / n if n else None)
    cls = dict(He=agg(lambda x: x["species"] in HE_SPECIES),
               He_burn=agg(lambda x: x["species"] in HE_SPECIES and x["group"] == "burn"),
               He_debris=agg(lambda x: x["species"] in HE_SPECIES and x["group"] == "debris"),
               H_isotopes=agg(lambda x: x["species"] in H_SPECIES),
               H_burn=agg(lambda x: x["species"] in H_SPECIES and x["group"] == "burn"),
               H_debris=agg(lambda x: x["species"] in H_SPECIES and x["group"] == "debris"),
               C=agg(lambda x: x["species"] == "12C"), Au=agg(lambda x: x["species"] == "Au"))
    k = int(np.argmax(prof))
    xs = [0.5e-9, 10e-9, 100e-9, 1e-6, 3e-6, 5e-6]
    return dict(
        disp_per_m2_total=D, total_over_Nat_5um=D / (N_AT * L5), mean_first_5um_exact=mean_first(S, prof),
        mean_first_5um_cells_below_5um=float((prof[S.x < L5] * S.dx[S.x < L5]).sum() / S.dx[S.x < L5].sum()),
        depth_covered_by_cell_rule_um=float(S.dx[S.x < L5].sum() * 1e6),
        frac_displacements_deeper_than_5um=frac_below(S, prof),
        peak_dpa=float(prof[k]), peak_cell_index=k, peak_cell_edges_nm=(float(S.edges[k] * 1e9), float(S.edges[k + 1] * 1e9)),
        peak_cell_center_nm=float(S.x[k] * 1e9),
        profile_samples={f"{x * 1e9:g} nm": float(np.interp(x, S.x, prof)) for x in xs},
        mean_first_5um_by_class={c: v["disp_per_m2"] / (N_AT * L5) for c, v in cls.items()},
        He_per_m2_incident=S.he_per_shot_incident(), He_per_m2_incident_burn=S.he_per_shot_incident("burn"),
        He_per_m2_implanted=S.he_per_shot_implanted(), by_species=rows, by_class=cls)


# --------------------------------------------------------------------------------------------- TRIM and single energy
def trim_total(run):
    """Printed 'Total Target Vacancies' and the depth-integrated total from the same VACANCY.txt."""
    txt = (ROOT / "data" / "srim" / run / "VACANCY.txt").read_text(errors="replace")
    printed = float([ln for ln in txt.splitlines() if "Total Target Vacancies" in ln][0].split("=")[1].split("/")[0])
    rows = []
    for ln in txt.splitlines():
        m = re.match(r"^\s*([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s*$", ln)
        if m:
            rows.append([float(v) for v in m.groups()])
    r = np.array(rows)
    dz = r[1, 0] - r[0, 0]                                   # Angstrom
    if "Knock-Ons" in txt:                                   # full cascade: col 3 = W vacancies (col 2 = knock-ons)
        depth_sum = float(r[:, 2].sum() * dz)
    else:                                                    # quick K-P: col 2 (by ions) + col 3 (by recoils)
        depth_sum = float((r[:, 1] + r[:, 2]).sum() * dz)
    return printed, depth_sum


def v_single(species, E0_eV, model):
    """Displacements per ion of fixed energy E0 (no backscatter correction): int sigma_d(E) / S(E) dE (as in verify_he_4)."""
    tab = srim.SrimTable(srim.SPECIES_CODE[species], "W")
    E = np.geomspace(300 * EV, E0_eV * EV, 800)
    return float(np.trapezoid(model(E) / (tab.stopping(E) * dm.M_W_KG), E))


def main():
    res = {"definitions": dict(N_at=N_AT, L_5um=L5, quoted_full_cascade_over_quickKP=FULL_CASCADE_QUOTED,
                               note_quoted="QUOTED from Wielunska 2020 (20 MeV W in W) and Stoller 2013 (not read here): not computed")}
    # ============================================================ (1) their numbers
    res["text_check"] = (check_against_text() if REF_TXT.exists() else
                         {"skipped": "the text copy of the paper is not distributed (copyright)"})
    rows, extra = rederive()
    res["sharafat_table5_printed"] = T5
    res["sharafat_rederived"] = rows
    res["sharafat_derived"] = extra
    print("=" * 100, "\n(1) SHARAFAT TABLE 5: re-derivation (labels, values, units line up 1:1 as printed; arithmetic below)")
    for r in rows:
        pr = "" if r["printed"] is None else f"printed {r['printed']:.4g}  diff {100 * r['rel_diff']:+.1f}%"
        print(f"  {r['name']:38s} {r['formula']:38s} -> {r['computed']:.4g}  {pr}")
    print(f"  implied vacancies per He in the printed dpa: {extra['implied_vac_per_He_total']:.1f}  -> implied H per He = {extra['implied_H_per_He']:.2f} "
          f"(rounding band {extra['implied_H_per_He_band'][0]:.2f} - {extra['implied_H_per_He_band'][1]:.2f})  [INFERRED: paper does not state it]")
    print(f"  Sec. 6.1 text says {TEXT_226:.0f} dpa/s and {TEXT_2725:.0f} He/dpa; table says {T5['inst_dpa_rate']:.0f} and {T5['inst_He_per_dpa_appm']:.0f}; "
          f"226 dpa/s would need an on-time of {extra['text_226_implied_on_time_us']:.2f} us, not 2.9 us")
    print(f"  text check: {res['text_check']}")

    # ============================================================ our sources (3 damage models)
    edges = make_grid()
    st = srim.SrimStopping(mode="projected", multilayer_trim=True)
    models = {"nrt_Ed90": dict(dpa_kind="nrt"), "nrt_Ed25": dict(dpa_kind="nrt", Ed=25.0), "arc_dpa": dict(dpa_kind="arc")}
    ours, srcs = {}, {}
    for name, kw in models.items():
        srcs[name] = ShotSources("Demo", edges, stopping=st, **kw)
        ours[name] = analyse(srcs[name])
    res["ours"] = ours
    B = ours["nrt_Ed90"]
    # tie-out with the stored results (same code path, must agree to rounding)
    stored = json.loads((ROOT / "runs/he_retention/results/sources.json").read_text(encoding="utf-8"))["Demo"]
    res["tieout_vs_sources_json"] = dict(
        disp_per_m2=(B["disp_per_m2_total"], stored["displacements_per_m2"]),
        mean_dpa_0_5um=(B["mean_first_5um_cells_below_5um"], stored["mean_dpa_0_5um"]),
        peak_dpa=(B["peak_dpa"], stored["peak_dpa_per_shot"]), He_incident_total=(B["He_per_m2_incident"], stored["He_incident_total"]))

    # ============================================================ (2) our quantity on their definition
    print("=" * 100, "\n(2) OUR dpa per shot (Demo, NRT Ed = 90 eV) on their definition")
    print(f"  tie-out with sources.json: " + ", ".join(f"{k}: {v[0]:.6e} vs {v[1]:.6e}" for k, v in res["tieout_vs_sources_json"].items()))
    for name in models:
        o = ours[name]
        print(f"  [{name:9s}] total displacements {o['disp_per_m2_total']:.4e} /m2 -> /(N_at x 5 um) = {o['total_over_Nat_5um']:.4e} dpa;  "
              f"mean first 5 um = {o['mean_first_5um_exact']:.4e} (cell rule: {o['mean_first_5um_cells_below_5um']:.4e} over {o['depth_covered_by_cell_rule_um']:.3f} um);  "
              f"{100 * o['frac_displacements_deeper_than_5um']:.1f}% of displacements deeper than 5 um")
        print(f"              PEAK {o['peak_dpa']:.4e} dpa in cell {o['peak_cell_index']} = [{o['peak_cell_edges_nm'][0]:g}, {o['peak_cell_edges_nm'][1]:g}] nm "
              f"(centre {o['peak_cell_center_nm']:g} nm); profile {o['profile_samples']}")
    obs = {}
    for name in models:
        o = ours[name]
        obs[name] = dict(total_over_Nat_5um=T5["dpa_per_shot"] / o["total_over_Nat_5um"], mean5_exact=T5["dpa_per_shot"] / o["mean_first_5um_exact"],
                         vs_peak=T5["dpa_per_shot"] / o["peak_dpa"], own_peak_over_own_mean5=o["peak_dpa"] / o["mean_first_5um_exact"])
    res["observed_ratios_their_over_ours"] = obs
    print("  ratio (their 4.3e-4) / ours:  " + "; ".join(f"{k}: total/(N_at 5um) {v['total_over_Nat_5um']:.2f}, mean5 {v['mean5_exact']:.2f}, vs OUR PEAK {v['vs_peak']:.2f}" for k, v in obs.items()))
    print(f"  (our peak / our 5 um mean = {obs['nrt_Ed90']['own_peak_over_own_mean5']:.2f}; their 4.3e-4 is a 5 um average by construction, so avg-vs-peak is NOT the cause)")

    # ============================================================ (3) decomposition by ion group
    print("=" * 100, "\n(3) OUR displacements by ion group (Demo, per m2 of wall, per shot; baseline NRT Ed = 90 eV; per-ion values for the other models)")
    print(f"  {'group':7s} {'species':10s} {'<E> MeV':>8s} {'ions/m2':>10s} {'disp/m2':>10s} {'fraction':>9s} {'disp/ion 90':>12s} {'disp/ion 25':>12s} {'disp/ion arc':>13s}")
    for r90, r25, ra in zip(B["by_species"], ours["nrt_Ed25"]["by_species"], ours["arc_dpa"]["by_species"]):
        print(f"  {r90['group']:7s} {r90['species']:10s} {r90['mean_E_MeV']:8.3g} {r90['ions_per_m2']:10.3e} {r90['disp_per_m2']:10.3e} {r90['fraction']:9.4f} "
              f"{r90['disp_per_ion']:12.4g} {r25['disp_per_ion']:12.4g} {ra['disp_per_ion']:13.4g}")
    tot_n = sum(r["ions_per_m2"] for r in B["by_species"])
    print(f"  all ions: {tot_n:.3e} /m2, total displacements {B['disp_per_m2_total']:.4e} /m2; H-isotope ions per He ion = "
          f"{B['by_class']['H_isotopes']['ions_per_m2'] / B['by_class']['He']['ions_per_m2']:.2f}")

    # ============================================================ (4) per-ion displacements
    print("=" * 100, "\n(4) displacements per ion: spectrum averages (ours) and TRIM quick Kinchin-Pease (Ed = 25 eV, from the repo files)")
    per_ion = {}
    for name in models:
        c = ours[name]["by_class"]
        per_ion[name] = dict(He=c["He"]["disp_per_ion"], He_burn=c["He_burn"]["disp_per_ion"], He_debris=c["He_debris"]["disp_per_ion"],
                             H_isotopes=c["H_isotopes"]["disp_per_ion"])
        print(f"  [{name:9s}] per He ion {per_ion[name]['He']:.4g} (burn {per_ion[name]['He_burn']:.4g}, debris {per_ion[name]['He_debris']:.4g});  per H-isotope ion {per_ion[name]['H_isotopes']:.4g}"
              f"   | theirs: 308 per He, 60 per H  ->  308 / ours = {308 / per_ion[name]['He']:.2f},  60 / ours = {60 / per_ion[name]['H_isotopes']:.2f}")
    res["per_ion_spectrum_average"] = per_ion
    Eavg = {k: [r["mean_E_MeV"] for r in B["by_species"] if r["species"] == "4He" and r["group"] == k][0] for k in ("burn", "debris")}
    res["He_mean_energy_MeV"] = Eavg
    print(f"  mean He energy: burn {Eavg['burn']:.3g} MeV, debris {Eavg['debris']:.3g} MeV (number-weighted, 4He)")

    runs = [("trim_supplementary/S1_he4_1keV_W", "4He", 1e3), ("trim_supplementary/S2_he4_10keV_W", "4He", 1e4), ("trim/B4", "4He", 1e5),
            ("trim/B5", "4He", 2e6), ("trim/B1", "Deuterons", 1e3), ("trim/B2", "Deuterons", 1e4), ("trim/B3", "Deuterons", 1e5)]
    single = []
    print(f"  {'run':36s} {'ion':9s} {'E keV':>8s} {'TRIM KP printed':>15s} {'TRIM depth sum':>15s} {'ours Ed25':>10s} {'/TRIM':>6s} {'ours Ed90':>10s} {'ours arc':>9s}")
    for run, sp, E0 in runs:
        printed, dsum = trim_total(run)
        v25 = v_single(sp, E0, dm.displacement_model(sp, "nrt", Ed=25.0))
        v90 = v_single(sp, E0, dm.displacement_model(sp, "nrt"))
        va = v_single(sp, E0, dm.displacement_model(sp, "arc"))
        single.append(dict(run=run, species=sp, E_keV=E0 / 1e3, trim_kp_printed=printed, trim_kp_depth_sum=dsum, ours_Ed25=v25,
                           ours_Ed25_over_trim=v25 / dsum, ours_Ed90=v90, ours_arc=va))
        print(f"  {run:36s} {sp:9s} {E0 / 1e3:8g} {printed:15.0f} {dsum:15.3f} {v25:10.3f} {v25 / dsum:6.2f} {v90:10.3f} {va:9.3f}")
    res["single_energy_vs_trim"] = single
    v35 = {nm: v_single("4He", 3.5e6, dm.displacement_model("4He", "nrt", Ed=25.0) if nm == "Ed25" else dm.displacement_model("4He", "nrt"))
           for nm in ("Ed25", "Ed90")}
    v2 = {nm: v_single("4He", 2e6, dm.displacement_model("4He", "nrt", Ed=25.0) if nm == "Ed25" else dm.displacement_model("4He", "nrt"))
          for nm in ("Ed25", "Ed90")}
    res["He_single_energy_ours"] = dict(E3p5MeV=v35, E2MeV=v2)
    print(f"  single He, ours: 3.5 MeV (DT alpha birth) Ed25 {v35['Ed25']:.1f}, Ed90 {v35['Ed90']:.1f};  2 MeV Ed25 {v2['Ed25']:.1f}, Ed90 {v2['Ed90']:.1f}   "
          f"-> 2 x Ed25(3.5 MeV) = {2 * v35['Ed25']:.0f}, 2 x TRIM KP (2 MeV) = {2 * 147}  vs their 308")
    kp_p, kp_d = trim_total("trim/B1")
    fc_p, fc_d = trim_total("trim_supplementary/B1_fullcascade")
    n_ions = 5000
    res["full_cascade_local_datapoint"] = dict(run="D 1 keV in W, 5000 ions, Ed = 25 eV", KP_printed=kp_p, KP_depth_sum=kp_d, FC_printed=fc_p, FC_depth_sum=fc_d,
                                               ratio_FC_over_KP=fc_d / kp_d, poisson_rel_err_estimate=float(np.hypot(1 / np.sqrt(kp_d * n_ions), 1 / np.sqrt(fc_d * n_ions))))
    f = res["full_cascade_local_datapoint"]
    print(f"  FULL CASCADE vs KP in the repo: D 1 keV in W, printed totals {kp_p:.0f} (KP) and {fc_p:.0f} (FC) /ion (rounded to integers); "
          f"depth-integrated {kp_d:.3f} (KP) and {fc_d:.3f} (FC): ratio {f['ratio_FC_over_KP']:.2f} (+-{100 * f['poisson_rel_err_estimate']:.0f}% Poisson estimate).")
    print(f"  This is the ONLY full-cascade run in the repo (D at 1 keV, near threshold); '~2x at MeV energies' is QUOTED (Wielunska 2020: 20 MeV W in W), not computed.")

    # ============================================================ (5) budget
    # Like for like = their definition: ALL displacements of the shot / (N_at x 5 um).  Same-Ed ordering:
    #   R(Ed 90) = [D(Ed 25) / D(Ed 90)]  x  R(Ed 25),
    #   R(Ed 25) = (He count ratio) x (their vacancies per He / our per He at Ed 25) x (ion-share ratio at Ed 25),
    # and their vacancies per He / our per He at Ed 25 = (full cascade, QUOTED 2) x (residual).
    v_S, h_S, n_S = T5["vac_per_He"], T5["vac_per_H"], T5["He_per_shot"] / T5["area_m2"]
    nHe = ours["nrt_Ed90"]["by_class"]["He"]["ions_per_m2"]
    v90, v25 = per_ion["nrt_Ed90"]["He"], per_ion["nrt_Ed25"]["He"]
    D90, D25 = ours["nrt_Ed90"]["disp_per_m2_total"], ours["nrt_Ed25"]["disp_per_m2_total"]
    share_ours90 = D90 / ours["nrt_Ed90"]["by_class"]["He"]["disp_per_m2"]
    share_ours25 = D25 / ours["nrt_Ed25"]["by_class"]["He"]["disp_per_m2"]
    share_S = 1.0 + h_S / v_S                                  # INFERRED: one H per He in their dpa
    R_obs = T5["dpa_per_shot"] / B["total_over_Nat_5um"]       # with the printed (rounded) 4.3e-4
    R_model = (n_S * (v_S + h_S) / (N_AT * L5)) / B["total_over_Nat_5um"]   # with their numbers re-derived (4.31e-4)
    F_Ed = D25 / D90
    F_count = n_S / nHe
    F_FC = FULL_CASCADE_QUOTED
    F_res = v_S / (v25 * F_FC)
    F_share = share_S / share_ours25
    prod = F_Ed * F_count * F_FC * F_res * F_share
    eHH = ours["nrt_Ed25"]["by_class"]
    budget = dict(
        observed_ratio_printed_4p3e4=R_obs, observed_ratio_rederived_4p31e4=R_model, product_of_factors=prod,
        observed_ratio_vs_mean_first_5um_exact=obs["nrt_Ed90"]["mean5_exact"],
        observed_ratio_vs_mean_first_5um_cell_rule=T5["dpa_per_shot"] / B["mean_first_5um_cells_below_5um"],
        factors=[
            dict(name="Ed 90 -> 25 eV, our whole model, all ions (4.71e19 / 1.12e19 displacements)", value=F_Ed,
                 status="computed; our Ed = 25 code reproduces TRIM K-P to about 4% at 2 MeV and 0.5% at 100 keV He"),
            dict(name="He ions per m2 (3.69e17 / 1.235e17 incident)", value=F_count, status="UNEXPLAINED; implausible for one target (see 6)"),
            dict(name="full cascade / quick K-P (their model not stated)", value=F_FC,
                 status="QUOTED (Wielunska 2020 for 20 MeV W in W, Stoller 2013), not computed; the one local point (D 1 keV) gives %.2f" % f["ratio_FC_over_KP"]),
            dict(name="residual vacancies per He: 308 / (our 62.0 at Ed 25 x 2)", value=F_res, status="UNEXPLAINED (their per-He model and energy are not stated)"),
            dict(name="ion share at Ed 25: (1 + 60/308) / (all-ion / He displacements, ours)", value=F_share,
                 status="computed for ours; INFERRED for theirs (1 H per He)"),
        ],
        parts=dict(F_Ed_all_ions=F_Ed, F_count=F_count, F_FC_quoted=F_FC, F_res=F_res, F_share=F_share,
                   F_vac_per_He_Ed90=v_S / v90, F_Ed_per_He=v25 / v90, v_S_He=v_S, v_ours_Ed90=v90, v_ours_Ed25=v25,
                   share_theirs=share_S, share_ours_Ed90=share_ours90, share_ours_Ed25=share_ours25,
                   F_res_if_local_FC_1p59=v_S / (v25 * f["ratio_FC_over_KP"]),
                   H_per_ion=dict(theirs=h_S, ours_Ed90=per_ion["nrt_Ed90"]["H_isotopes"], ours_Ed25=per_ion["nrt_Ed25"]["H_isotopes"],
                                  residual_after_FC2_at_Ed25=h_S / (per_ion["nrt_Ed25"]["H_isotopes"] * F_FC)),
                   share_ours_Ed25_is_1_plus=dict(H_over_He=eHH["H_isotopes"]["disp_per_m2"] / eHH["He"]["disp_per_m2"],
                                                  C_over_He=eHH["C"]["disp_per_m2"] / eHH["He"]["disp_per_m2"],
                                                  Au_over_He=eHH["Au"]["disp_per_m2"] / eHH["He"]["disp_per_m2"]),
                   net_unexplained_after_Ed_and_FC=F_count * F_res * F_share,
                   He_channel_unexplained=F_count * F_res),
        depth_note=dict(deeper_than_5um_fraction=B["frac_displacements_deeper_than_5um"],
                        text="their volume is area x 5 um and all their vacancies are put in it, so the like-for-like denominator is total displacements/(N_at x 5 um); "
                             "our 'mean of the first 5 um' leaves out the displacements deeper than 5 um (3.6%)"))
    he_only_S = extra["he_only_dpa"]
    c90, c25 = ours["nrt_Ed90"]["by_class"], ours["nrt_Ed25"]["by_class"]
    budget["partial_ratios"] = dict(
        He_only_vs_He_only_Ed90=he_only_S / (c90["He"]["disp_per_m2"] / (N_AT * L5)),
        He_only_vs_He_only_Ed25=he_only_S / (c25["He"]["disp_per_m2"] / (N_AT * L5)),
        HeH_vs_HeH_Ed90=T5["dpa_per_shot"] / ((c90["He"]["disp_per_m2"] + c90["H_isotopes"]["disp_per_m2"]) / (N_AT * L5)),
        all_ions_Ed90=obs["nrt_Ed90"]["total_over_Nat_5um"], all_ions_Ed25=obs["nrt_Ed25"]["total_over_Nat_5um"], all_ions_arc=obs["arc_dpa"]["total_over_Nat_5um"],
        their_avg_over_our_peak_Ed90=obs["nrt_Ed90"]["vs_peak"], our_peak_over_our_mean5_Ed90=obs["nrt_Ed90"]["own_peak_over_own_mean5"])
    # a single-energy-alpha reading of 308 (hypothesis, not shown by the paper)
    budget["hypothesis_single_energy_alpha"] = dict(
        text="308 is within ~5-10% of 2 x (K-P, Ed = 25 eV) for ONE He of 2-3.5 MeV; the spectrum average is lower because the debris He (mean 0.23 MeV) make ~40 vacancies each (Ed 25)",
        two_x_trim_kp_2MeV=2 * 147.0, two_x_ours_Ed25_2MeV=2 * v2["Ed25"], two_x_ours_Ed25_3p5MeV=2 * v35["Ed25"],
        ratio_308_over_these=[308 / (2 * 147.0), 308 / (2 * v2["Ed25"]), 308 / (2 * v35["Ed25"])],
        ours_Ed25_3p5MeV_over_spectrum_average=v35["Ed25"] / v25, residual_to_compare=F_res)
    res["budget"] = budget
    print("=" * 100, "\n(5) MULTIPLICATIVE BUDGET of (their 4.3e-4 dpa) / (our dpa), like for like: all displacements / (N_at x 5 um)")
    print(f"  observed: 4.3e-4 / {B['total_over_Nat_5um']:.4e} = {R_obs:.3f}   (their re-derived 4.31e-4: {R_model:.3f};  vs our mean of the first 5 um: {obs['nrt_Ed90']['mean5_exact']:.3f} exact, "
          f"{budget['observed_ratio_vs_mean_first_5um_cell_rule']:.3f} with the cell rule of the report)")
    for x in budget["factors"]:
        print(f"    x {x['value']:8.4f}   {x['name']}   [{x['status']}]")
    print(f"    product of the factors = {prod:.3f} = ratio with their re-derived dpa 4.31e-4 ({R_model:.3f}); the printed 4.3e-4 gives {R_obs:.3f} (rounding, {100 * (R_obs / prod - 1):+.2f}%)")
    sp_ = budget["parts"]["share_ours_Ed25_is_1_plus"]
    print(f"    ion share: theirs 1 + 60/308 = {share_S:.3f}; ours all-ion/He displacements = {share_ours25:.3f} at Ed 25 (1 + H {sp_['H_over_He']:.2f} + C {sp_['C_over_He']:.2f} + Au {sp_['Au_over_He']:.2f}), {share_ours90:.3f} at Ed 90")
    print(f"    He channel unexplained (count x residual) = {F_count * F_res:.2f};  net after Ed, quoted full cascade and our extra ions = {budget['parts']['net_unexplained_after_Ed_and_FC']:.2f}")
    print(f"    their per-He / ours at Ed 90 = {v_S / v90:.2f} (= {v25 / v90:.3f} Ed x {F_FC:g} full cascade x {F_res:.3f} residual); per H at Ed 25 after x2: residual {budget['parts']['H_per_ion']['residual_after_FC2_at_Ed25']:.2f}")
    pr = budget["partial_ratios"]
    print(f"    other like-for-like ratios: He only vs He only {pr['He_only_vs_He_only_Ed90']:.1f} (Ed 90) / {pr['He_only_vs_He_only_Ed25']:.1f} (Ed 25);  He+H vs He+H (Ed 90) {pr['HeH_vs_HeH_Ed90']:.1f};  "
          f"all ions: Ed 90 {pr['all_ions_Ed90']:.2f}, Ed 25 {pr['all_ions_Ed25']:.2f}, arc-dpa {pr['all_ions_arc']:.1f}")
    hy = budget["hypothesis_single_energy_alpha"]
    print(f"    hypothesis (not shown by the paper): 308 vs 2 x TRIM-KP(2 MeV) = {hy['two_x_trim_kp_2MeV']:.0f}, 2 x ours Ed25 (2 MeV) = {hy['two_x_ours_Ed25_2MeV']:.0f}, (3.5 MeV) = {hy['two_x_ours_Ed25_3p5MeV']:.0f}; "
          f"ours Ed25 single 3.5 MeV / spectrum average = {hy['ours_Ed25_3p5MeV_over_spectrum_average']:.2f} vs residual {F_res:.2f}")

    # ============================================================ (6) He count
    scen = SCENARIOS["Demo"]
    area = 4 * np.pi * scen.R ** 2
    N_DT = scen.yield_J / (17.6 * MEV_J)
    N_He_aries = nHe * area
    N_He_burn_aries = ours["nrt_Ed90"]["by_class"]["He_burn"]["ions_per_m2"] * area
    E_He_total_J = sum(s.n * s.E for s in srcs["nrt_Ed90"].sub if s.species in HE_SPECIES) * area
    raw = json.loads((ROOT / "runs/source_term/he_shot_count.json").read_text())["Demo (raw)"]["all He"]["incident_per_m2"]
    month_shots = 30 * 86400 * REP_HZ_SHARAFAT
    he6 = dict(
        He_per_m2_theirs=n_S, He_per_m2_ours_incident_total=nHe, He_per_m2_ours_burn=ours["nrt_Ed90"]["by_class"]["He_burn"]["ions_per_m2"],
        ratio_total=n_S / nHe, ratio_burn=n_S / ours["nrt_Ed90"]["by_class"]["He_burn"]["ions_per_m2"],
        ratio_vs_raw_table_count=n_S / raw, area_m2_4piR2=area,
        He_per_target_ARIES_normalized=N_He_aries, He_per_target_burn_ARIES=N_He_burn_aries, He_per_target_theirs=T5["He_per_shot"],
        N_DT_reactions_154MJ=N_DT, He_per_DT_ARIES=N_He_aries / N_DT, burn_He_per_DT_ARIES=N_He_burn_aries / N_DT, He_per_DT_theirs=T5["He_per_shot"] / N_DT,
        alpha_energy_available_MJ=N_DT * 3.5 * MEV_J / 1e6, theirs_if_all_3p5MeV_MJ=T5["He_per_shot"] * 3.5 * MEV_J / 1e6,
        ARIES_He_ion_energy_MJ=E_He_total_J / 1e6, ARIES_mean_He_energy_MeV=E_He_total_J / N_He_aries / MEV_J,
        theirs_mean_energy_if_ARIES_energy_MeV=E_He_total_J / T5["He_per_shot"] / MEV_J,
        text_sec63_He_per_m2_per_shot=1e16, table5_over_text_sec63=n_S / 1e16,
        month_of_ETF_shots_30d_5Hz=month_shots,
        He_per_W_atom_after_month_in_5um=dict(table5=month_shots * T5["frac_appm"] * 1e-6, ours=month_shots * nHe / (N_AT * L5), text_1e16=month_shots * 1e16 / (N_AT * L5)))
    res["He_count"] = he6
    print("=" * 100, "\n(6) He count: is 3.69e17 He/m2 (1.96e20 per target) plausible for ONE 154 MJ target?")
    print(f"  theirs {n_S:.3e} He/m2 vs ours {nHe:.3e} incident total (x{he6['ratio_total']:.2f}), burn only {he6['He_per_m2_ours_burn']:.3e} (x{he6['ratio_burn']:.2f}); raw-table count {raw:.3e} (x{he6['ratio_vs_raw_table_count']:.2f})")
    print(f"  per target (4 pi R^2 = {area:.1f} m2): ARIES He {N_He_aries:.3e} (burn {N_He_burn_aries:.3e}); theirs {T5['He_per_shot']:.3e}; DT reactions at 154 MJ / 17.6 MeV = {N_DT:.3e}")
    print(f"  He per DT reaction: ARIES {he6['He_per_DT_ARIES']:.2f} (burn only {he6['burn_He_per_DT_ARIES']:.2f}), theirs {he6['He_per_DT_theirs']:.2f}  (one alpha per reaction is the physical ceiling, plus a little DD 3He)")
    print(f"  energy: all DT alphas at birth carry {he6['alpha_energy_available_MJ']:.1f} MJ; 1.96e20 He at 3.5 MeV would be {he6['theirs_if_all_3p5MeV_MJ']:.0f} MJ; ARIES He ions carry {he6['ARIES_He_ion_energy_MJ']:.1f} MJ "
          f"(mean {he6['ARIES_mean_He_energy_MeV']:.2f} MeV), so 1.96e20 He would have a mean of {he6['theirs_mean_energy_if_ARIES_energy_MeV']:.2f} MeV")
    print(f"  their own text: Sec. 6.3 gives ~1e16 He/m2 per shot (Table 5 is x{he6['table5_over_text_sec63']:.0f} more); 'a month of ETF operation = number density of W' "
          f"({month_shots:.3e} shots at 5 Hz, 30 d): He per W atom in 5 um: Table 5 {he6['He_per_W_atom_after_month_in_5um']['table5']:.1f}, ours {he6['He_per_W_atom_after_month_in_5um']['ours']:.1f}, 1e16/m2 {he6['He_per_W_atom_after_month_in_5um']['text_1e16']:.2f}")
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("=" * 100, f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
