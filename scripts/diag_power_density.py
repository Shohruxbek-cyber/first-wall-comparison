"""Power density on the wall per shot and averaged over one second (revision of 20 Oct 2026),
from the source term of the production runs: x-rays 2.1 MJ x Y/154 MJ in
a 20 ns square pulse at R/c; ions from the ARIES spectra (production construction of ife_wall.ions), arriving by
time of flight.

For each scenario: the incident energy per shot (x-rays, ions), the power density during the
x-ray pulse, the peak of the ion arrival power, the mean ion power over the 10-90 % arrival
window, and the incident power averaged over one second (repetition rate x energy per shot;
Experimental: one shot in that second, for comparison).  No solver run is involved.

Usage: python scripts/diag_power_density.py [--out=results/revision/power_density]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402
from ife_wall.ions import IonSpectrumSource, load_ion_bins  # noqa: E402
from ife_wall.neutron import E_W_REF  # noqa: E402
from ife_wall.scenarios import SCENARIOS, XRAY_REF_J, YIELD_REF_J, Options  # noqa: E402

NEUTRON_REF_J = 109e6  # neutron energy of the NRL 154 MJ target (Raffray 2005, Table 1)

if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "power_density"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    tau_x = Options().xray_tau
    res = {"xray_pulse_s": tau_x, "scenarios": {}}
    for name, sc in SCENARIOS.items():
        area = 4 * np.pi * sc.R ** 2
        F_x = XRAY_REF_J * sc.yield_J / YIELD_REF_J / area
        ions = IonSpectrumSource(load_ion_bins(True), sc.yield_J, sc.R, ["W", "Fe"], srim.SrimStopping("projected"))
        F_i = ions.total_fluence()
        t = np.geomspace(ions.t_first, ions.t_last, 200001)
        E = np.array([ions.energy(ions.t_first, ti) for ti in t[::200]])  # cumulative energy on a coarser grid
        tc = t[::200]
        P = np.diff(E) / np.diff(tc)
        frac = E / E[-1]
        t10, t90 = np.interp(0.1, frac, tc), np.interp(0.9, frac, tc)
        rep = sc.rep_Hz if sc.rep_Hz else 1.0
        res["scenarios"][name] = {
            "yield_MJ": sc.yield_J / 1e6, "R_m": sc.R, "rep_Hz": sc.rep_Hz,
            "xray_energy_J_m2": F_x, "ion_energy_J_m2": F_i, "total_energy_J_m2": F_x + F_i,
            "xray_power_during_pulse_MW_m2": F_x / tau_x / 1e6,
            "ion_peak_power_MW_m2": float(P.max()) / 1e6, "t_ion_peak_us": float(tc[1:][np.argmax(P)]) * 1e6,
            "ion_10_90_window_us": [float(t10) * 1e6, float(t90) * 1e6],
            "ion_mean_power_10_90_MW_m2": 0.8 * F_i / (t90 - t10) / 1e6,
            "average_over_1s_MW_m2": rep * (F_x + F_i) / 1e6,
            "average_note": "repetition rate x energy per shot" if sc.rep_Hz else "one shot in that second (single-shot scenario)",
            # neutrons (revision of 20 Oct 2026): 109 MJ of the 154 MJ target [Raffray 2005, Table 1],
            # scaled by yield; volumetric heating of tungsten from Sharafat et al. (2005), as in ife_wall.neutron
            "neutron_energy_J_m2": NEUTRON_REF_J * sc.yield_J / YIELD_REF_J / area,
            "neutron_wall_load_MW_m2": rep * NEUTRON_REF_J * sc.yield_J / YIELD_REF_J / area / 1e6,
            "neutron_heating_W_J_m3_per_shot": E_W_REF * sc.yield_J / YIELD_REF_J * (6.5 / sc.R) ** 2,
            "neutron_heating_W_MW_m3_average": rep * E_W_REF * sc.yield_J / YIELD_REF_J * (6.5 / sc.R) ** 2 / 1e6}
    (out / "power_density.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
