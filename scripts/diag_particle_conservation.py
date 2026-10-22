"""Particle conservation of the ion spectra (revision of 22 Oct 2026).

For each ion species, the number of ions per shot over the whole chamber (whole target, 154 MJ) against what the
target can produce:
  * fusions: 154 MJ / 17.6 MeV = 5.461e19; one alpha per D-T fusion;
  * D and T: the fuel left after burn, from the target of Raffray (2005) Fig. 1 (DT ice 0.150-0.169 cm, DT in
    20 mg/cm^3 CH foam 0.169-0.195 cm, DT vapor 0.3 mg/cm^3 inside 0.150 cm) and the solid-DT density of the
    Peterson et al. (2002) target (2.976 mg between 1.875 and 2.1125 mm: 0.2506 g/cm^3);
  * C and H: the CH foam (20 mg/cm^3) and the 1 um CH layer (density 1.05 g/cm^3: ASSUMPTION, polystyrene);
  * Au: the 300 A gold layer (density 19.3 g/cm^3: ASSUMPTION, bulk gold).
Four constructions of the bins (ife_wall.ions): the table with log-midpoint edges ("raw"), the table with the
0.5 MeV edges of the burn alphas, the per-species energy normalization used before 22 Oct 2026 ("normalized"),
and the number- and energy-conserving production spectra ("conserving").  Also the helium per shot at the wall
of each scenario (burn only; burn and debris), and the cause of the helium excess step by step.

Usage: python scripts/diag_particle_conservation.py [--out=results/revision/particle_conservation]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import ions  # noqa: E402
from ife_wall.ions import KEV_J, MEV, N_FUSIONS_REF, YIELD_REF  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

U = 1.66053906660e-27  # kg
M_D, M_T, M_C, M_H, M_AU = 2.01410178, 3.01604928, 12.0, 1.00782503, 196.966570  # u (atomic masses)
# target of Raffray (2005) Fig. 1 (cm) and densities (g/cm^3)
R_VAP, R_ICE, R_FOAM = 0.150, 0.169, 0.195
RHO_VAP, RHO_FOAM = 0.3e-3, 20e-3
RHO_DT = 2.976e-3 / (4 / 3 * math.pi * (0.21125 ** 3 - 0.1875 ** 3))  # Peterson et al. (2002) target
RHO_CH_SOLID, CH_LAYER_CM = 1.05, 1e-4      # ASSUMPTION (polystyrene), 1 um
RHO_AU, AU_LAYER_CM = 19.3, 300e-8          # ASSUMPTION (bulk gold), 300 A


def shell(r0, r1):
    return 4 / 3 * math.pi * (r1 ** 3 - r0 ** 3)


def inventory():
    v_foam = shell(R_ICE, R_FOAM)
    f_solid = RHO_FOAM / RHO_CH_SOLID                    # volume fraction of CH in the wetted foam
    m_dt = (shell(R_VAP, R_ICE) * RHO_DT + v_foam * (1 - f_solid) * RHO_DT
            + 4 / 3 * math.pi * R_VAP ** 3 * RHO_VAP)     # g
    pairs = m_dt * 1e-3 / ((M_D + M_T) * U)
    m_ch = v_foam * RHO_FOAM + 4 * math.pi * R_FOAM ** 2 * CH_LAYER_CM * RHO_CH_SOLID   # g
    n_ch = m_ch * 1e-3 / ((M_C + M_H) * U)
    n_au = 4 * math.pi * (R_FOAM + CH_LAYER_CM) ** 2 * AU_LAYER_CM * RHO_AU * 1e-3 / (M_AU * U)
    return {"DT_mass_mg": m_dt * 1e3, "DT_density_g_cm3": RHO_DT, "D_initial": pairs, "T_initial": pairs,
            "fusions": N_FUSIONS_REF, "burn_up_fraction": N_FUSIONS_REF / pairs,
            "D_left_after_burn": pairs - N_FUSIONS_REF, "T_left_after_burn": pairs - N_FUSIONS_REF,
            "CH_mass_mg": m_ch * 1e3, "C_atoms": n_ch, "H_atoms": n_ch, "Au_atoms": n_au,
            "foam_solid_volume_fraction": f_solid}


def constructions():
    t_raw = ions.ion_spectra_tables("raw")
    t_cons = ions.ion_spectra_tables("conserving")
    out = {"raw (log-midpoint edges)": {k: (d["number"].sum(), d["energy"].sum()) for k, d in t_raw.items()},
           "table with the 0.5 MeV burn-alpha edges": {k: (d["num_table"].sum(), (d["num_table"] * d["E"]).sum() * KEV_J)
                                                       for k, d in t_cons.items()},
           "normalized (before 22 Oct 2026)": {k: (d["number"].sum(), d["energy"].sum())
                                               for k, d in ions.ion_spectra_tables("normalized").items()},
           "conserving (production)": {k: (d["number"].sum(), d["energy"].sum()) for k, d in t_cons.items()}}
    fit = {f"{g} {s}": {"number_factor": d["number_factor"], "tilt_exponent": d["tilt"], "energy_stretch": d["stretch"]}
           for (g, s), d in t_cons.items()}
    return out, fit


def wall_helium(spectra):
    res = {}
    for name, sc in SCENARIOS.items():
        f = sc.yield_J / YIELD_REF / (4 * math.pi * sc.R ** 2)
        b = ions.load_ion_bins(spectra=spectra)
        res[name] = {"burn_He_per_m2": f * sum(x.number for x in b if x.group == "burn" and x.species in ("4He", "3He")),
                     "burn_and_debris_He_per_m2": f * sum(x.number for x in b if x.species in ("4He", "3He")),
                     "burn_4He_per_m2": f * sum(x.number for x in b if x.group == "burn" and x.species == "4He"),
                     "burn_and_debris_4He_per_m2": f * sum(x.number for x in b if x.species == "4He")}
    return res


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "particle_conservation"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    cons, fit = constructions()
    nf = N_FUSIONS_REF
    tab = {}
    for c, d in cons.items():
        tab[c] = {f"{g} {s}": {"number": N, "per_fusion": N / nf, "energy_MJ": E / 1e6, "mean_keV": E / N / KEV_J}
                  for (g, s), (N, E) in d.items()}
        he4 = d[("burn", "4He")][0] + d[("debris", "4He")][0]
        he3 = d[("burn", "3He")][0] + d[("debris", "3He")][0]
        tab[c]["4He per fusion"] = he4 / nf
        tab[c]["all He per fusion"] = (he4 + he3) / nf
        tab[c]["D total"] = d[("burn", "Deuterons")][0] + d[("debris", "Deuterons")][0]
        tab[c]["T total"] = d[("burn", "Tritons")][0] + d[("debris", "Tritons")][0]
        tab[c]["ion energy MJ"] = sum(E for _, E in d.values()) / 1e6
    raw, lin, nor = (cons["raw (log-midpoint edges)"], cons["table with the 0.5 MeV burn-alpha edges"],
                     cons["normalized (before 22 Oct 2026)"])
    b4 = ("burn", "4He")
    t_raw4 = ions.ion_spectra_tables("raw")[b4]
    cause = {
        "1_log_midpoint_edges_on_the_0.5_MeV_burn_alpha_bins": {
            "burn_4He_number_log_midpoint": raw[b4][0], "burn_4He_number_0.5MeV_edges": lin[b4][0],
            "burn_4He_energy_MJ_log_midpoint": raw[b4][1] / 1e6, "burn_4He_energy_MJ_0.5MeV_edges": lin[b4][1] / 1e6,
            "edges_of_the_3.54_MeV_bin_keV_log_midpoint": [float(t_raw4["lo"][7]), float(t_raw4["hi"][7])]},
        "2_energy_normalization_scaled_numbers": {
            f"{g} {s}": nor[(g, s)][0] / raw[(g, s)][0] for (g, s) in raw},
        "3_the_table_itself": {"4He_per_fusion_0.5MeV_edges": tab["table with the 0.5 MeV burn-alpha edges"]["4He per fusion"],
                               "4He_per_fusion_log_midpoint": tab["raw (log-midpoint edges)"]["4He per fusion"],
                               "neutrons_per_fusion_log_midpoint": None},
    }
    # neutrons of the same table (not an ion; a check of the table's yield): log-midpoint edges
    rows = ions._table_rows()
    E, n = ions._species_table(rows, "burn", "Neutrons")
    lo, hi = ions._logmid_edges(E)
    cause["3_the_table_itself"]["neutrons_per_fusion_log_midpoint"] = float((n * (hi - lo)).sum() / nf)
    cause["3_the_table_itself"]["neutron_energy_MJ_log_midpoint"] = float((n * (hi - lo) * E).sum() * KEV_J / 1e6)
    checks = {
        "4He_per_fusion": {c: tab[c]["4He per fusion"] for c in tab},
        "D_against_fuel_left": {c: tab[c]["D total"] / inv["D_left_after_burn"] for c in tab},
        "T_against_fuel_left": {c: tab[c]["T total"] / inv["T_left_after_burn"] for c in tab},
        "C_against_target_C": {c: tab[c]["debris 12C"]["number"] / inv["C_atoms"] for c in tab},
        "H_against_target_H": {c: tab[c]["debris Protons"]["number"] / inv["H_atoms"] for c in tab},
        "Au_against_target_Au": {c: tab[c]["debris Au"]["number"] / inv["Au_atoms"] for c in tab},
    }
    res = {"note": __doc__.split("Usage:")[0].strip(), "fusions": nf, "Q_DT_MeV": ions.Q_DT_MEV,
           "target_inventory": inv, "constructions": tab, "conserving_fit": fit, "checks": checks,
           "cause_of_the_helium_excess": cause,
           "helium_at_the_wall_per_shot": {"normalized": wall_helium("normalized"), "conserving": wall_helium("conserving")}}
    (out / "particle_conservation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = ["species,construction,number,per_fusion,energy_MJ,mean_keV"]
    for c, d in tab.items():
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{k},{c},{v['number']:.6e},{v['per_fusion']:.6f},{v['energy_MJ']:.6f},{v['mean_keV']:.3f}")
    (out / "particle_conservation.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"fusions {nf:.4e}; DT {inv['DT_mass_mg']:.3f} mg, burn-up {inv['burn_up_fraction']:.4f}, "
          f"D or T left {inv['D_left_after_burn']:.4e}")
    for k, v in checks.items():
        print(k, {c.split(" ")[0]: round(x, 4) for c, x in v.items()})
    print(json.dumps(cause, indent=1))
    for c in ("normalized", "conserving"):
        print(c, {s: {k: f"{v:.3e}" for k, v in d.items()} for s, d in res["helium_at_the_wall_per_shot"][c].items()})


if __name__ == "__main__":
    main()
