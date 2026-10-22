"""Compare our source term against Garoz et al. (2016) Table 2 (arXiv:2405.01541, p. 4) and
GA2020 Table I.  Garoz prints 1-2 significant figures, so each value is compared against its
rounding interval [v - u/2, v + u/2], where u = one unit of the last printed digit.

Writes runs/source_term/garoz_comparison.{json,md}.
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import IonSpectrumSource, KEV_J, load_ion_bins  # noqa: E402
from ife_wall.xray import rebin_spectrum  # noqa: E402

SCEN = {"Exp.": (20e6, 5.0), "Proto.": (50e6, 6.5), "Demo": (154e6, 6.5)}
SP = {"H": "Protons", "D": "Deuterons", "T": "Tritons", "He": "4He"}
# Garoz2016 Table 2, as printed (strings keep the printed precision)
G = {
    ("burn", "H"): dict(pct="0.4", E="2.55", w="500", F=("0.02", "0.03", "0.1")),
    ("burn", "D"): dict(pct="3.1", E="3.46", w="600", F=("0.2", "0.3", "0.90")),
    ("burn", "T"): dict(pct="2.8", E="2.66", w="600", F=("0.2", "0.3", "0.8")),
    ("burn", "He"): dict(pct="6.4", E="3.81", w="600", F=("0.4", "0.6", "1.9")),
    ("burn", "total"): dict(pct="12.7", E="3.40", w="600", F=("0.8", "1.2", "3.7")),
    ("debris", "H"): dict(pct="0.1", E="0.09", w="1500", F=("0.00", "0.01", "0.02")),
    ("debris", "D"): dict(pct="5.8", E="0.14", w="2000", F=("0.4", "0.5", "1.7")),
    ("debris", "T"): dict(pct="7.2", E="0.19", w="2000", F=("0.5", "0.7", "2.1")),
    ("debris", "He"): dict(pct="0.9", E="0.23", w="1500", F=("0.05", "0.1", "0.25")),
    ("debris", "total"): dict(pct="14.4", E="0.17", w="2000", F=("0.9", "1.3", "4.0")),
    ("all", "total"): dict(pct="27.1", E="0.29", w="3000", F=("1.7", "2.5", "7.7")),
}
X_G = dict(pct="1.4", E="0.007", w="0.1", F=("0.1", "0.1", "0.4"))


def interval(s):
    d = Decimal(s)
    u = Decimal(1).scaleb(d.as_tuple().exponent)
    return float(d - u / 2), float(d + u / 2)


def verdict(x, s):
    lo, hi = interval(s)
    if lo <= x <= hi:
        return "within rounding"
    v = float(s)
    return f"{(x / v - 1) * 100:+.1f}%" if v else "n/a"


bins = load_ion_bins(normalize=True)


def select(group, key):
    if key == "total":
        return [b for b in bins if group == "all" or b.group == group]
    return [b for b in bins if b.group == group and b.species == SP[key]]


rows, md = [], ["| Group | Species | Quantity | Garoz | This work | Verdict |", "|---|---|---|---|---|---|"]
for (g, k), ref in G.items():
    sb = select(g, k)
    E_tot = sum(b.energy for b in sb)
    N_tot = sum(b.number for b in sb)
    ours = {"pct": 100 * E_tot / 154e6, "E": E_tot / N_tot / KEV_J / 1e3}
    for name, (Y, R) in SCEN.items():
        ours[f"F_{name}"] = E_tot * Y / 154e6 / (4 * np.pi * R ** 2) / 1e4  # J/cm^2
    # pulse width at the wall (Demo): t10-t90 of energy arrival, ns
    src = IonSpectrumSource(sb, 154e6, 6.5, ["W"], stopping=None)
    t = np.linspace(src.t_first, src.t_last, 4000)
    Ec = np.array([src.energy(0, tt) for tt in t])
    t10, t50, t90 = (t[np.searchsorted(Ec, f * Ec[-1])] for f in (0.1, 0.5, 0.9))
    ours["w"] = (t90 - t10) * 1e9
    ours["t50_Demo_us"] = t50 * 1e6
    for q, lab in (("pct", "energy share (%)"), ("E", "mean energy (MeV, number-weighted)"), ("w", "pulse width (ns); ours = t10-t90 at Demo")):
        rows.append(dict(group=g, species=k, quantity=lab, garoz=ref[q], ours=ours[q], verdict=verdict(ours[q], ref[q])))
    for i, name in enumerate(SCEN):
        rows.append(dict(group=g, species=k, quantity=f"fluence {name} (J/cm2)", garoz=ref["F"][i], ours=ours[f"F_{name}"],
                         verdict=verdict(ours[f"F_{name}"], ref["F"][i])))
    rows.append(dict(group=g, species=k, quantity="median arrival Demo (us)", garoz="-", ours=ours["t50_Demo_us"], verdict="-"))

# x-rays
Ec, eb = rebin_spectrum(400)
n_ph = eb / (Ec * KEV_J)
xr = {"pct": 100 * 2.1 / 154, "E_num": eb.sum() / n_ph.sum() / KEV_J / 1e3, "E_en": np.sum(eb * Ec) / eb.sum() / 1e3}
rows.append(dict(group="x-ray", species="-", quantity="mean energy (MeV), number-weighted", garoz=X_G["E"], ours=xr["E_num"], verdict=verdict(xr["E_num"], X_G["E"])))
rows.append(dict(group="x-ray", species="-", quantity="mean energy (MeV), energy-weighted", garoz=X_G["E"], ours=xr["E_en"], verdict=verdict(xr["E_en"], X_G["E"])))
for i, (name, (Y, R)) in enumerate(SCEN.items()):
    F = 2.1e6 * Y / 154e6 / (4 * np.pi * R ** 2) / 1e4
    rows.append(dict(group="x-ray", species="-", quantity=f"fluence {name} (J/cm2)", garoz=X_G["F"][i], ours=F, verdict=verdict(F, X_G["F"][i])))

for r in rows:
    md.append(f"| {r['group']} | {r['species']} | {r['quantity']} | {r['garoz']} | {r['ours']:.4g} | {r['verdict']} |")
out = ROOT / "runs" / "source_term"
out.mkdir(parents=True, exist_ok=True)
(out / "garoz_comparison.json").write_text(json.dumps(rows, indent=1))
(out / "garoz_comparison.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
