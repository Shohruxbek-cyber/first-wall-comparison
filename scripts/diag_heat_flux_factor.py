"""Heat-flux factor of the ion load from the paper's own ion power history (revision of 21 Oct 2026), beside the 44.4 MW m^-2 s^1/2 of Gonzalez-Arrabal et al. (2020), with the design-rule cycle count of
Mitteau et al. (2024), N = exp(100 / F_HF), for each.

The ion energy arriving at the tungsten wall is that of the production source term (ife_wall.ions.IonSpectrumSource,
ARIES spectra in the production construction of ife_wall.ions, time of flight), and the absorbed part removes the
TRIM backscattered energy of each
component (srim.SrimStopping, as in the deposition).  Two forms:
  * window form: F = E_abs(t10..t90) / sqrt(t90 - t10), with t10 and t90 the times by which 10 % and 90 % of the
    absorbed ion energy has arrived;
  * peak form:   F = q_max sqrt(t90 - t10), with q_max the peak absorbed ion power density.
The x-rays (20 ns) are not included, as in the ion fluences of Gonzalez-Arrabal et al.  The absorbed energy is
evaluated on the time grid of Table 2 (scripts/diag_power_density.py: 1001 logarithmic times from the first to the
last ion arrival), so that the peak power is the same interval average as the Table 2 peak.

Usage: python scripts/diag_heat_flux_factor.py [--out=results/revision/heat_flux_factor]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall import srim  # noqa: E402
from ife_wall.ions import IonSpectrumSource, load_ion_bins  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

FHF_GA = {"Experimental": 9.7, "Prototype": 14.4, "Demo": 44.4}  # MJ m^-2 s^-1/2 = MW m^-2 s^1/2, GA2020 Table I


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "heat_flux_factor"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    st = srim.SrimStopping(mode="projected")
    bins = load_ion_bins(True)
    res = {"note": __doc__.split("Usage:")[0].strip(), "scenarios": {}}
    for name in ("Experimental", "Prototype", "Demo"):
        sc = SCENARIOS[name]
        src = IonSpectrumSource(bins, sc.yield_J, sc.R, ["W"], stopping=st)
        t = np.geomspace(src.t_first, src.t_last, 200001)[::200]  # the Table 2 grid
        keep = np.array([1.0 - st.reflected_energy_fraction(b.species, "W", b.E_mean) for b in src.bins])
        E_cum = np.array([float(keep @ src.energy_components(src.t_first, tt)) for tt in t])
        E_tot = E_cum[-1]
        t10, t90 = float(np.interp(0.1 * E_tot, E_cum, t)), float(np.interp(0.9 * E_tot, E_cum, t))
        dt = t90 - t10
        q = np.diff(E_cum) / np.diff(t)
        F_win = 0.8 * E_tot / math.sqrt(dt) / 1e6
        F_peak = float(q.max()) * math.sqrt(dt) / 1e6
        res["scenarios"][name] = {
            "incident_ion_J_m2": src.total_fluence(), "absorbed_ion_J_m2": E_tot,
            "t10_us": t10 * 1e6, "t90_us": t90 * 1e6, "window_us": dt * 1e6,
            "peak_absorbed_power_W_m2": float(q.max()), "t_peak_us": float(t[1:][int(np.argmax(q))] * 1e6),
            "F_HF_window_MW_m2_s05": F_win, "F_HF_peak_MW_m2_s05": F_peak, "F_HF_GA2020": FHF_GA[name],
            "cycles_window": math.exp(100.0 / F_win), "cycles_peak": math.exp(100.0 / F_peak),
            "cycles_GA2020": math.exp(100.0 / FHF_GA[name]),
        }
        r = res["scenarios"][name]
        print(f"{name}: absorbed ions {E_tot:.4g} J/m2 of {r['incident_ion_J_m2']:.4g}; t10-t90 {r['t10_us']:.3f}-{r['t90_us']:.3f} us "
              f"({r['window_us']:.3f} us); F window {F_win:.1f}, peak {F_peak:.1f}, GA {FHF_GA[name]} MW m-2 s1/2; "
              f"cycles {r['cycles_window']:.3g} / {r['cycles_peak']:.3g} / {r['cycles_GA2020']:.3g}")
    (out / "heat_flux_factor.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
