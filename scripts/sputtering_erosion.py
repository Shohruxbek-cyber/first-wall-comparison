"""W erosion by physical sputtering from the per-shot threat ions (added 2026-09-27).

Yields: Eckstein (2007) fits, Eq. (2), Table 8 (data/eckstein2007_table8_W.csv), normal
incidence (ions travel radially from the target, so incidence on the spherical wall is normal).
Spectra: ARIES NRL 154 MJ ion tables (production construction of ife_wall.ions), scaled by yield, per unit wall
area; integrated over each energy bin with the same number- and energy-conserving within-bin
shape used for the thermal source (ions.bin_shape).

Notes:
  * Eckstein yields are per INCIDENT ion from TRIM.SP simulations, which already include
    reflected ions, so no backscatter correction is applied (it would double count).  The TRIM
    backscatter correction is used only for He retention (scripts/he_shot_count.py).
  * The fits were made to calculated yields plotted up to ~20 keV for H and He (Eckstein 2007,
    p. 38); most threat ions are above that, so their yields are EXTRAPOLATED with the fit
    formula.  The share of sputtering from ions above 20 keV is reported.
  * C and Au on W are NOT in Eckstein's Table 8 -> MISSING.  They are bracketed with labelled
    ESTIMATED proxies from the same table: C between 4He->W and N->W (M = 4 and 14 around
    M = 12); Au by W->W (M = 184 vs 197).  Totals are given with and without them.
  * Gross erosion: no redeposition, no surface roughness, no temperature dependence, no
    chemical or enhanced (e.g. He-induced fuzz) erosion.
Output: runs/sputtering/sputtering.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import KEV_J, bin_shape, load_ion_bins  # noqa: E402

EV = 1.602176634e-19
NA = 6.02214076e23
M_W, RHO_W = 0.18384, 19250.0  # kg/mol (IUPAC), kg/m^3 (Tolias, ~300 K)
PAR = {r["ion"]: {k: float(v) for k, v in r.items() if k not in ("ion", "target")}
       for r in csv.DictReader(l for l in open(ROOT / "data" / "eckstein2007_table8_W.csv", encoding="utf-8")
                               if not l.startswith("#"))}
SPECIES = {"Protons": "H", "Deuterons": "D", "Tritons": "T", "3He": "3He", "4He": "4He"}


def s_n_krc(eps):
    return 0.5 * np.log(1 + 1.2288 * eps) / (eps + 0.1728 * np.sqrt(eps) + 0.008 * eps ** 0.1504)


def yield_eckstein(ion, E_eV):
    p = PAR[ion]
    E = np.asarray(E_eV, float)
    x = np.clip(E / p["Eth_eV"] - 1.0, 0.0, None)
    xm = x ** p["mu"]
    return np.where(x > 0, p["q"] * s_n_krc(E / p["eps_L_eV"]) * xm / (p["lambda"] + xm), 0.0)


def check_eps_L():
    """Recompute eps_L from Eqs. (4)-(5) as a transcription check."""
    aB, e2 = 0.0529177, 1.439964  # nm, eV nm
    Z = {"H": 1, "D": 1, "T": 1, "3He": 2, "4He": 2, "N": 7, "W": 74}
    M = {"H": 1.008, "D": 2.014, "T": 3.016, "3He": 3.016, "4He": 4.003, "N": 14.007, "W": 183.84}
    out = {}
    for ion in PAR:
        aL = (9 * np.pi ** 2 / 128) ** (1 / 3) * aB / np.sqrt(Z[ion] ** (2 / 3) + 74 ** (2 / 3))
        epsL = (M[ion] + M["W"]) / M["W"] * Z[ion] * 74 * e2 / aL
        out[ion] = epsL / PAR[ion]["eps_L_eV"]
    return out


def sputtered_per_bin(b, ion_par_key):
    x = np.geomspace(b.E_lo, b.E_hi, 300)
    n = bin_shape(b, x)  # ions per J (whole target)
    Y = yield_eckstein(ion_par_key, x / EV)
    return float(np.trapezoid(n * Y, x)), float(np.trapezoid(n * Y * (x / KEV_J > 20.0), x))


bins = load_ion_bins(True)
scen = {"Experimental": (20e6, 5.0, None), "Prototype": (50e6, 6.5, 1.0), "Demo": (154e6, 6.5, 10.0)}
res = {"eps_L_recomputed_over_table": check_eps_L(), "scenarios": {}}
for name, (Y, R, rep) in scen.items():
    sc = Y / 154e6 / (4 * np.pi * R ** 2)
    per = {}
    above20 = 0.0
    for sp, key in SPECIES.items():
        tot = hi = 0.0
        for b in bins:
            if b.species == sp:
                a, h = sputtered_per_bin(b, key)
                tot += a
                hi += h
        per[sp] = tot * sc
        above20 += hi * sc
    # ESTIMATED proxies for C and Au
    prox = {}
    for sp, keys in (("12C", ("4He", "N")), ("Au", ("W",))):
        vals = []
        for key in keys:
            vals.append(sum(sputtered_per_bin(b, key)[0] for b in bins if b.species == sp) * sc)
        prox[sp] = [min(vals), max(vals)]
    atoms = sum(per.values())
    nm = lambda a: a * M_W / NA / RHO_W * 1e9  # noqa: E731
    lo = atoms + prox["12C"][0] + prox["Au"][0]
    hi_ = atoms + prox["12C"][1] + prox["Au"][1]
    r = {"sputtered_atoms_m2_per_shot_by_species": per,
         "sourced_total_atoms_m2": atoms, "sourced_nm_per_shot": nm(atoms),
         "share_from_ions_above_20keV_extrapolated": above20 / atoms if atoms else None,
         "ESTIMATED_proxy_atoms_m2": prox,
         "with_proxies_nm_per_shot_range": [nm(lo), nm(hi_)],
         "shots_to_erode_1mm_sourced": 1e6 / nm(atoms),
         "shots_to_erode_1mm_with_proxies_range": [1e6 / nm(hi_), 1e6 / nm(lo)]}
    if rep:
        r["full_power_years_to_erode_1mm_sourced"] = r["shots_to_erode_1mm_sourced"] / rep / 3.156e7
        r["full_power_years_with_proxies_range"] = [x / rep / 3.156e7 for x in r["shots_to_erode_1mm_with_proxies_range"]]
    res["scenarios"][name] = r
    print(f"{name}: sourced {nm(atoms):.3e} nm/shot ({atoms:.3e} atoms/m2; "
          f"{100 * r['share_from_ions_above_20keV_extrapolated']:.0f}% from >20 keV), with C/Au proxies "
          f"{nm(lo):.3e}-{nm(hi_):.3e} nm/shot; shots to 1 mm: {r['shots_to_erode_1mm_sourced']:.3e}"
          + (f" ({r['full_power_years_to_erode_1mm_sourced']:.2f} fpy)" if rep else ""))
    print("   by species (atoms/m2):", {k: f"{v:.2e}" for k, v in per.items()}, "proxies:", {k: [f'{x:.2e}' for x in v] for k, v in prox.items()})
print("eps_L check (recomputed/table):", {k: round(v, 4) for k, v in res["eps_L_recomputed_over_table"].items()})
d = ROOT / "runs" / "sputtering"
d.mkdir(parents=True, exist_ok=True)
(d / "sputtering.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
