"""Benchmark of the attenuation step of the vapor model against the BUCKY xenon attenuation of
Raffray (2005, Table 3; 154 MJ NRL target, R = 6.5 m): reduction of the initial burn-ion and
debris-ion deposition by 10 and 20 mTorr of Xe (at 300 K) over the 6.5 m path, 1 % / 2 % (burn) and
29 % / 48 % (debris).  Blanchard & Martin (2005, p. 193) give about 20 % of the ion energy at
10 mTorr.  This tests the stopping in a static gas column only, not the growth of a vapor column.

Needs SRIM-2012 stopping tables for p, d, t, 3He, 4He, 12C and Au in Xe gas in data/srim/
(inputs and instructions in data/srim_xe_inputs/); without them the script stops with a message.

Usage: python scripts/xe_benchmark.py [--out=results/revision/xe_benchmark]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import IonSpectrumSource, csda_fractions_layered, load_ion_bins  # noqa: E402
from ife_wall.srim import SrimStopping  # noqa: E402

KB, AMU, M_XE, R, T_GAS, TORR = 1.380649e-23, 1.66053906660e-27, 131.29, 6.5, 300.0, 133.322368
NEEDED = ["p", "d", "t", "he3", "he4", "c12", "au"]

if __name__ == "__main__":
    missing = [n for n in NEEDED if not (ROOT / "data" / "srim" / f"{n}_in_Xe.txt").exists()]
    if missing:
        raise SystemExit(f"missing SRIM tables in data/srim/: {', '.join(n + '_in_Xe.txt' for n in missing)} "
                         "(see data/srim_xe_inputs/README.md)")
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), ROOT / "results" / "revision" / "xe_benchmark"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    bins = load_ion_bins(True)
    res = {"reference": {"burn_10mTorr": 0.01, "burn_20mTorr": 0.02, "debris_10mTorr": 0.29, "debris_20mTorr": 0.48,
                         "all_ions_10mTorr_BM2005": 0.20}, "cases": {}}
    for mode in ("csda", "projected"):
        st = SrimStopping(mode)
        st.TARGET = dict(SrimStopping.TARGET, Xe="Xe")
        src = IonSpectrumSource(bins, 154e6, R, ["Xe"], st)
        E = src.fluence
        groups = np.array([b.group for b in src.bins])
        for mtorr in (10.0, 20.0):
            n = mtorr * 1e-3 * TORR / (KB * T_GAS)
            M = n * M_XE * AMU * R  # kg/m^2 of Xe between the target and the wall
            fr = np.array([csda_fractions_layered(b.E_mean, b.species, np.array([M]), np.array([0]), ["Xe"], st)[0]
                           for b in src.bins])
            res["cases"][f"{mode}_{mtorr:g}mTorr"] = {
                "column_kg_m2": M,
                "burn_fraction_stopped": float(E[groups == "burn"] @ fr[groups == "burn"] / E[groups == "burn"].sum()),
                "debris_fraction_stopped": float(E[groups == "debris"] @ fr[groups == "debris"] / E[groups == "debris"].sum()),
                "all_ions_fraction_stopped": float(E @ fr / E.sum())}
    (out / "xe_benchmark.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
