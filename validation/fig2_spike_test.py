"""Check (6 Oct 2026): are the narrow spikes in the blue model profiles of Fig. 2 grid artefacts?

Recomputes the 'projected' model profile exactly as validation/srim_checks.py does (uniform cells
over the TRIM target thickness, csda_fractions_layered, plotted as fraction per cell / cell width),
at the plotting grid of 2000 cells and at 4x and 16x finer grids. Read-only except for its own
output, runs/validation/fig2_spike_test.json. For every case, finds the spikes on the 2000-cell grid
(local maxima above 1.3x the median of the surrounding +-5 % of the plotted depth range) and follows
each spike to the finer grids.

Result (6 Oct 2026): the narrow spikes (B1, its full-cascade twin, S1, S2) stay at the same depth on
all three grids while their height grows almost in proportion to the number of cells. They sit at
the energies where consecutive rows of the SRIM range tables print the same projected range (whole
angstroms at low energy, e.g. d in W: 11 A at both 110 and 120 eV), so the energy lost between those
rows is placed at one depth.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.ions import csda_fractions_layered  # noqa: E402
from ife_wall.srim import DATA, EV, SPECIES_CODE, SrimStopping  # noqa: E402

CODE2SP = {v: k for k, v in SPECIES_CODE.items()}
OUT = ROOT / "runs" / "validation"
summ = []
for f in (DATA / "trim" / "summary.csv", DATA / "trim_supplementary" / "summary.csv"):
    summ += list(csv.DictReader(open(f, newline="")))
st = SrimStopping(mode="projected")
report = []
for r in summ:
    E0 = float(r["E_keV"]) * 1e3 * EV
    rho = float(r["density_g_cm3"]) * 1e3
    thick = float(r["thickness_um"]) * 1e-6
    prof = {}
    for nc in (2000, 8000, 32000):
        m = np.full(nc, thick * rho / nc)
        fr = np.asarray(csda_fractions_layered(E0, CODE2SP[r["ion"]], m, np.zeros(nc, int), [r["target"]], st))
        xc = (np.arange(nc) + 0.5) * thick / nc
        prof[nc] = (xc, fr / (thick / nc))  # 1/m, as in the figure
    x0, y0 = prof[2000]
    xmax = x0[np.nonzero(y0)[0][-1]] if np.any(y0 > 0) else thick
    win = max(3, int(0.05 * np.searchsorted(x0, xmax)))
    spikes = []
    for i in range(1, len(y0) - 1):
        if y0[i] > 0 and y0[i] >= y0[i - 1] and y0[i] >= y0[i + 1] and x0[i] <= xmax:
            lo, hi = max(0, i - win), min(len(y0), i + win + 1)
            med = np.median(y0[lo:hi][y0[lo:hi] > 0])
            if y0[i] > 1.3 * med:
                spikes.append(i)
    rows = []
    for i in spikes:
        xs = x0[i]
        row = {"depth_um": float(xs * 1e6)}
        for nc, (x, y) in prof.items():
            j = np.searchsorted(x, xs)
            lo, hi = max(0, j - int(len(x) * 0.002) - 2), min(len(x), j + int(len(x) * 0.002) + 3)
            k = lo + int(np.argmax(y[lo:hi]))
            wl, wh = max(0, k - int(0.05 * len(x))), min(len(x), k + int(0.05 * len(x)) + 1)
            med = np.median(y[wl:wh][y[wl:wh] > 0])
            row[f"peak_{nc}"] = float(y[k] * 1e-6)  # 1/um
            row[f"x_{nc}_um"] = float(x[k] * 1e6)
            row[f"over_median_{nc}"] = float(y[k] / med)
            half = med + 0.5 * (y[k] - med)
            a = k
            while a > 0 and y[a - 1] > half:
                a -= 1
            b = k
            while b < len(y) - 1 and y[b + 1] > half:
                b += 1
            row[f"width_{nc}_um"] = float((b - a + 1) * thick / len(x) * 1e6)
        rows.append(row)
    report.append({"run": r["run"], "ion": r["ion"], "target": r["target"], "E_keV": float(r["E_keV"]),
                   "n_spikes_2000": len(rows), "spikes": rows[:6]})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fig2_spike_test.json").write_text(json.dumps(report, indent=1))
for c in report:
    print(f"{c['run']:16s} {c['ion']:4s} {c['E_keV']:8g} keV -> {c['target']:2s}: {c['n_spikes_2000']} spikes on the 2000-cell grid")
    for s in c["spikes"][:4]:
        print("    x %.4g um | peak (1/um) 2000: %.4g  8000: %.4g  32000: %.4g | over local median: %.2f %.2f %.2f" % (
            s["depth_um"], s["peak_2000"], s["peak_8000"], s["peak_32000"],
            s["over_median_2000"], s["over_median_8000"], s["over_median_32000"]))
