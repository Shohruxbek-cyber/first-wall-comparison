"""Verification test 4: per-shot sources (He and displacements) against independent calculations.  Pass: < 0.5 %.

He:  integrated He/m^2 per shot from ShotSources (sub-binned spectra, depth profiles on the model grid) vs
     (i) a separate re-implementation that reads the raw spectrum table (data/aries_nrl154_dd_spectra.csv) and applies
         the construction of the spectra: since 22 Oct 2026 one alpha per fusion in the table's burn/debris ratio
         (independent_he_conserving); before, the per-species energy normalisation (ACCOUNTING_MJ), and (ii) the existing script output
         runs/source_term/he_shot_count.json (burn and burn + debris; Demo values in the report: 5.58e16, 1.23e17).
     Retained (implanted) = incident x (1 - TRIM backscatter number fraction).
dpa: displacements per m^2 per shot from the depth-resolved profiles (straggling blur, binning) vs a direct path
     integral per sub-bin with 2000 energy points and no blur.  Informational checks against TRIM Kinchin-Pease
     vacancy totals (data/srim/trim*/VACANCY.txt, Ed = 25 eV) are reported too (not pass/fail).
"""
import json

import numpy as np

from _he_common import ROOT, record, style, VER
from ife_wall import srim
from ife_wall.he import damage as dm
from ife_wall.he.grid import make_grid
from ife_wall.he.sources import ShotSources
from ife_wall.ions import ACCOUNTING_MJ, DEFAULT_ION_SPECTRA
from ife_wall.scenarios import SCENARIOS

KEV_J = 1.602176634e-16


def independent_he(scen):
    if DEFAULT_ION_SPECTRA == "normalized":
        return independent_he_normalized(scen)
    return independent_he_conserving(scen)


def independent_he_conserving(scen):
    """Independent re-implementation for the conserving spectra (22 Oct 2026): 4He burn + debris = one alpha per
    D-T fusion (154 MJ / 17.6 MeV, scaled by Y/154 MJ), split in the ratio of the table counts with the burn-alpha
    bins 0.5 MeV wide below 4 MeV (log midpoints elsewhere); 3He at its table count (log-midpoint bins)."""
    rows = [r.split(",") for r in (ROOT / "data" / "aries_nrl154_dd_spectra.csv").read_text().splitlines()[2:] if r]
    sc = scen.yield_J / 154e6 / (4 * np.pi * scen.R ** 2)
    n_fusions = 154e6 / (17.6 * 1.602176634e-13)
    count = {}
    for g in ("burn", "debris"):
        for s in ("4He", "3He"):
            E = np.array([float(r[2]) for r in rows if r[0] == g and r[1] == s])
            n = np.array([float(r[3]) for r in rows if r[0] == g and r[1] == s])
            keep = n > 1.0
            E, n = E[keep], n[keep]
            edges = list(np.sqrt(E[:-1] * E[1:]))
            lo, hi = [E[0] ** 2 / edges[0]] + edges, edges + [E[-1] ** 2 / edges[-1]]
            if (g, s) == ("burn", "4He"):
                for i in range(8):  # 0-0.5, ..., 3.5-4 MeV
                    lo[i], hi[i] = 500.0 * i, 500.0 * (i + 1)
                lo[8] = 4000.0
            count[(g, s)] = float((n * (np.array(hi) - np.array(lo))).sum())
    he4 = count[("burn", "4He")] + count[("debris", "4He")]
    return {g: (n_fusions * count[(g, "4He")] / he4 + count[(g, "3He")]) * sc for g in ("burn", "debris")}


def independent_he_normalized(scen):
    rows = [r.split(",") for r in (ROOT / "data" / "aries_nrl154_dd_spectra.csv").read_text().splitlines()[2:] if r]
    sc = scen.yield_J / 154e6 / (4 * np.pi * scen.R ** 2)
    out = {"burn": 0.0, "debris": 0.0}
    for (g, s), E_ref in ACCOUNTING_MJ.items():
        if s not in ("4He", "3He"):
            continue
        E = np.array([float(r[2]) for r in rows if r[0] == g and r[1] == s])
        n = np.array([float(r[3]) for r in rows if r[0] == g and r[1] == s])
        keep = n > 1.0
        E, n = E[keep], n[keep]
        lE = np.log(E)
        mid = np.exp(0.5 * (lE[1:] + lE[:-1]))
        lo = np.r_[E[0] ** 2 / mid[0], mid]
        hi = np.r_[mid, E[-1] ** 2 / mid[-1]]
        num = n * (hi - lo)
        scale = E_ref * 1e6 / (num * E * KEV_J).sum()
        out[g] += (num * scale).sum() * sc
    return out


def main():
    edges = make_grid()
    ref = json.loads((ROOT / "runs" / "source_term" / "he_shot_count.json").read_text())
    res, worst_he, worst_dpa = {}, 0.0, 0.0
    st = srim.SrimStopping(mode="projected", multilayer_trim=True)
    for name, scen in SCENARIOS.items():
        S = ShotSources(name, edges, stopping=st)
        ind = independent_he(scen)
        burn, tot = S.he_per_shot_incident("burn"), S.he_per_shot_incident()
        r = ref[f"{name} ({DEFAULT_ION_SPECTRA})"]
        e1 = abs(burn - ind["burn"]) / ind["burn"]
        e2 = abs(tot - (ind["burn"] + ind["debris"])) / tot
        e3 = abs(burn - r["burn 4He"]["incident_per_m2"]) / burn
        e4 = abs(tot - r["all He"]["incident_per_m2"]) / tot
        # retained: implanted integral on the grid vs sum over sub-bins of n (1 - eta_N), and vs the existing script
        impl = S.he_per_shot_implanted()
        impl_ref = sum(s.n * (1 - st.reflected_number_fraction(s.species, "W", s.E)) for s in S.sub if s.species in ("4He", "3He"))
        e5 = abs(impl - impl_ref) / impl_ref
        e6 = abs(impl - r["all He"]["retained_per_m2"]) / impl
        # displacements: direct path integral per sub-bin
        models = {sp: dm.displacement_model(sp, "nrt") for sp in {s.species for s in S.sub}}
        tot_nd = 0.0
        for s in S.sub:
            E = np.geomspace(s.E, 300 * dm.EV, 2001)
            rr = st.range(s.species, "W", E) / dm.RHO_SRIM
            mid = np.sqrt(E[:-1] * E[1:])
            nd = (dm.N_AT * models[s.species](mid) * (rr[:-1] - rr[1:])).sum()
            tot_nd += s.n * nd * (1 - st.reflected_energy_fraction(s.species, "W", s.E))
        e7 = abs(S.displacements_per_m2() - tot_nd) / tot_nd
        worst_he = max(worst_he, e1, e2, e3, e4, e5, e6)
        worst_dpa = max(worst_dpa, e7)
        res[name] = dict(He_incident_burn=burn, He_incident_total=tot, He_implanted=impl,
                         He_ref_burn_independent=ind["burn"], He_ref_total_independent=ind["burn"] + ind["debris"],
                         err_burn_vs_independent=e1, err_total_vs_independent=e2, err_burn_vs_he_shot_count=e3,
                         err_total_vs_he_shot_count=e4, err_implanted_vs_subbins=e5, err_implanted_vs_he_shot_count=e6,
                         displacements_per_m2=S.displacements_per_m2(), displacements_direct=tot_nd, err_dpa=e7,
                         peak_dpa_per_shot=float(S.dpa_profile().max()),
                         peak_dpa_per_shot_deeper_than_10nm=float(S.dpa_profile()[S.x > 10e-9].max()))
        print(name, {k: (round(v, 6) if isinstance(v, float) and abs(v) < 1 else v) for k, v in res[name].items()}, flush=True)
    # informational: TRIM Kinchin-Pease vacancies (Ed = 25 eV)
    trim = {}
    for run, sp, E0 in [("trim/B5", "4He", 2000), ("trim/B4", "4He", 100), ("trim/B3", "Deuterons", 100),
                        ("trim/B2", "Deuterons", 10), ("trim_supplementary/S2_he4_10keV_W", "4He", 10)]:
        txt = (ROOT / "data" / "srim" / run / "VACANCY.txt").read_text(errors="replace")
        v = float([l for l in txt.splitlines() if "Total Target Vacancies" in l][0].split("=")[1].split("/")[0])
        m25 = dm.DisplacementModel(sp, lambda Td: dm.nu_nrt(Td, 25.0), 25.0)
        tab = srim.SrimTable(srim.SPECIES_CODE[sp], "W")
        E = np.geomspace(300 * dm.EV, E0 * 1e3 * dm.EV, 800)
        S_ = tab.stopping(E) * dm.M_W_KG
        mine = float(np.trapezoid(m25(E) / S_, E))
        trim[run] = dict(ion=sp, E_keV=E0, trim_KP_vacancies=v, this_model_Ed25=mine, ratio=mine / v)
    plt = style()
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.loglog([t["trim_KP_vacancies"] for t in trim.values()], [t["this_model_Ed25"] for t in trim.values()], "o", ms=8)
    lim = [5, 200]
    ax.plot(lim, lim, "k--", lw=1)
    for k, t in trim.items():
        ax.annotate(f"{t['ion']} {t['E_keV']:g} keV", (t["trim_KP_vacancies"], t["this_model_Ed25"]), fontsize=9, xytext=(5, -10), textcoords="offset points")
    ax.set(xlabel="TRIM quick Kinchin-Pease vacancies per ion (Ed = 25 eV)", ylabel="this model, Ed = 25 eV (per ion)")
    fig.savefig(str(VER / "test4_dpa_vs_trim.pdf"))
    ok = worst_he < 0.005 and worst_dpa < 0.005
    record("test4_sources", ok, worst_He_error=worst_he, worst_dpa_error=worst_dpa, threshold=0.005, per_scenario=res, trim_check=trim)


if __name__ == "__main__":
    main()
