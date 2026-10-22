"""Ionization of the lithium vapor column if it kept the ion energy it stops (revision of
20 Oct 2026).  An equilibrium (LTE) Saha estimate, single ionization
only, ground-state statistical weights (Li I g = 2, Li II g = 1; NIST ASD, Li I ionization energy
5.3917 eV), for the Demo thick-film cases of the reduced vapor-shielding model.

Energy per atom E: the ion energy stopped in the vapor per evaporated atom
(results/vapor_shielding/summary.json, eV_per_vapor_atom_thick).  Two partitions:
  (a) all of E thermal:      E = 3/2 kT                    (upper bound on T, ionization not paid)
  (b) ionization included:   E = 3/2 kT (1 + x) + chi x    (heavy particles and electrons at T)
Density: the thick-film column (thick_film_kg_m2) spread over the distance the vapor expands at
its sound speed in 2 us and in 40 us (the debris-ion window and the end of the ion pulse).

Usage: python scripts/diag_saha.py [--out=results/vapor_shielding/saha]
"""
import json
import math
import sys
from pathlib import Path

from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
K_B, H, M_E, EV = 1.380649e-23, 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19
R_GAS, M_LI = 8.314462618, 6.94e-3
CHI = 5.391714996 * EV
T_SURF = 2224.0  # K, thick-film Demo surface peak (Table 5)
TIMES = (2e-6, 40e-6)


def saha_x(T, n):
    """Ionized fraction x = n_i / n_total from x^2 / (1 - x) = S(T) / n."""
    S = 2.0 * (1.0 / 2.0) * (2 * math.pi * M_E * K_B * T / H ** 2) ** 1.5 * math.exp(-CHI / (K_B * T))
    a = S / n
    return (-a + math.sqrt(a * a + 4 * a)) / 2.0


def thermal_only(E, n):
    T = 2 * E / (3 * K_B)
    return T, saha_x(T, n)


def with_ionization(E, n):
    f = lambda T: 1.5 * K_B * T * (1 + saha_x(T, n)) + CHI * saha_x(T, n) - E  # noqa: E731
    T = brentq(f, 100.0, 2 * E / (3 * K_B) + 1.0, xtol=1e-6)
    return T, saha_x(T, n)


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "vapor_shielding" / "saha"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((ROOT / "results" / "vapor_shielding" / "summary.json").read_text(encoding="utf-8"))["li"]["Demo"]
    c_s = math.sqrt(5.0 / 3.0 * R_GAS * T_SURF / M_LI)
    res = {"sound_speed_m_s": c_s, "assumptions": __doc__.split("\n\n")[0], "cases": {}}
    for case, d in summ.items():
        if not case.startswith("frr"):
            continue
        E = d["eV_per_vapor_atom_thick"] * EV
        N = d["thick_film_kg_m2"] / M_LI * 6.02214076e23  # atoms per m^2
        rows = []
        for t in TIMES:
            n = N / (c_s * t)
            Ta, xa = thermal_only(E, n)
            Tb, xb = with_ionization(E, n)
            rows.append({"spread_time_s": t, "n_m3": n, "thermal_only": {"kT_eV": K_B * Ta / EV, "x": xa},
                         "with_ionization": {"kT_eV": K_B * Tb / EV, "x": xb}})
        res["cases"][case] = {"eV_per_atom": d["eV_per_vapor_atom_thick"], "column_kg_m2": d["thick_film_kg_m2"], "rows": rows}
    (out / "saha.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for case, v in res["cases"].items():
        print(case, round(v["eV_per_atom"], 3), [(f"{r['n_m3']:.2e}", round(r["thermal_only"]["x"], 4), round(r["with_ionization"]["x"], 4))
                                                  for r in v["rows"]])
