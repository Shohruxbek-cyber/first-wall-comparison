"""Validation: Raffray (2005) J. Nucl. Mater. 347:178, Fig. 7 -- 3 mm W slab, NRL 154 MJ
direct-drive target, R = 6.5 m, no gas, coolant 500 C, h = 10 kW/m^2K, single shot.

Our inputs are the project baseline, not Raffray's (his code RACLETTE, his deposition):
  * x-rays: digitized Raffray Fig. 2 spectrum, 2.1 MJ (Table 1); arrival R/c; pulse 0.5 ns
    ("reach the chamber wall within about 20 ns over a time spread of sub-ns", p. 181)
  * ions: ARIES NRL 154 MJ tables (production construction of ife_wall.ions), time of flight to 6.5 m
  * ion deposition: SRIM baseline (TRIM profiles / projected model), x-ray: XCOM+Henke
  * W: Tolias2017 + Raffray Table 2 vapour law, alpha = 1; initial T = 500 C
Raffray Fig. 7 values (read from the figure, +-25 C): x-ray spike to ~1655 C at t~0;
surface peak ~2915 C at 1.84 us; 1 um ~2770 C; 5 um ~2030 C; 10 um ~1525 C at 3.4 us;
100 um ~515 C.  Text: "maximum W temperature is about 2900 C".

Variants: H-isotope depth stretch off (g = 1); CSDA depth; x-ray photoelectric only;
x-ray pulse 0.1 ns and 1 ns.
Usage: python validation/raffray2005.py [variant ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402
from ife_wall.ions import IonSpectrumSource, load_ion_bins  # noqa: E402
from ife_wall.library import tungsten  # noqa: E402
from ife_wall.source_term import SquarePulse  # noqa: E402
from ife_wall.thermal import BackBC, Layer, Mesh, ThermalSolver  # noqa: E402
from ife_wall.xray import XraySource  # noqa: E402

Y, R = 154e6, 6.5
AREA = 4 * np.pi * R ** 2
T0 = 500.0 + 273.15
C_LIGHT = 2.99792458e8
DEPTHS = {"1um": 1e-6, "5um": 5e-6, "10um": 10e-6, "100um": 100e-6}
TARGETS = {"xray_spike_C": 1655, "Ts_peak_C": 2915, "t_peak_us": 1.84, "1um_C": 2770, "5um_C": 2030,
           "10um_C": 1525, "100um_C": 515, "text_max_C": 2900}


def run(variant="baseline", t_end=10e-6):
    mode = "csda" if variant == "csda" else "projected"
    stop = srim.SrimStopping(mode=mode)
    if variant == "no_stretch":
        saved = srim.G_ANCHORS
        srim.G_ANCHORS = ((50.0, 1.0), (500.0, 1.0), (5000.0, 1.0))
    xmode = "pe" if variant == "xray_pe" else "pe+incoh"
    tau = {"xray_0.1ns": 0.1e-9, "xray_1ns": 1e-9}.get(variant, 0.5e-9)
    W = tungsten()
    mesh = Mesh.build([Layer(W, 3e-3, 1e-9, 1.05)], T0)
    t_x = R / C_LIGHT
    xs = XraySource(2.1e6 / AREA, SquarePulse(t_x, tau), ["W"], n_bins=100, mode=xmode)
    ions = IonSpectrumSource(load_ion_bins(True), Y, R, ["W"], stop)
    s = ThermalSolver(mesh, T0, back=BackBC("convective", h=1e4, T_coolant=T0), sources=[xs, ions],
                      alpha_evap=1.0, evaporation=True)
    hist = []

    def rec(sv):
        T = sv.T
        x = sv.x_centers()
        hist.append([sv.t, sv.surface_temperature(T)] + [float(np.interp(d, x, T)) for d in DEPTHS.values()])

    t0 = time.time()
    s.run(t_end, dt0=1e-13, dT_target=5.0, record=rec)
    if variant == "no_stretch":
        srim.G_ANCHORS = saved
    h = np.array(hist)
    t = h[:, 0]
    after_x = (t > t_x) & (t < t_x + 60e-9)
    res = {"variant": variant, "Ts_peak_C": float(h[:, 1].max() - 273.15),
           "t_peak_us": float(t[h[:, 1].argmax()] * 1e6),
           "xray_spike_C": float(h[after_x, 1].max() - 273.15),
           **{f"{k}_C": float(h[:, 2 + i].max() - 273.15) for i, k in enumerate(DEPTHS)},
           "evaporated_nm": s.mass_evaporated / 19250 * 1e9,
           "energy_rel_residual": float(s.ledger.residual()[1]),
           "incident_J_m2": s.ledger.incident, "deposited_J_m2": s.ledger.deposited,
           "not_absorbed_J_m2": s.ledger.passed_through, "wall_s": time.time() - t0}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    return res, h


if __name__ == "__main__":
    variants = sys.argv[1:] or ["baseline"]
    out = ROOT / "runs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    f = out / "raffray2005.json"
    if f.exists():
        allres = json.loads(f.read_text(encoding="utf-8"))
    for v in variants:
        res, h = run(v)
        allres[v] = res
        np.savetxt(out / f"raffray2005_{v}_history.csv", h, delimiter=",",
                   header="t_s,Ts_K," + ",".join(f"T_{k}_K" for k in DEPTHS), comments="")
    allres["targets_fig7"] = TARGETS
    f.write_text(json.dumps(allres, indent=1), encoding="utf-8")
