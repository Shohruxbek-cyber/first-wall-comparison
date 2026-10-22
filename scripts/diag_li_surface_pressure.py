"""Surface pressure state of the Demo lithium film (revision of 21 Oct 2026), from the recorded
extra cycles of the repeated-shot runs (scripts/li_extra_cycle.py) and the paper's own lithium vapor-pressure law
(Davison 1968, Eq. 9, as in ife_wall.library: log10 p[Pa] = 10.015 - 8064.5/T):

  * saturation pressure at the peak surface temperature;
  * recoil pressure of evaporation into vacuum, p_r = (1 + gamma) n1 k T1 = (1 + 5/3) x 0.31 x 0.67 p_sat
    = 0.554 p_sat, from the state at the sonic point that Anisimov (1968) derives (T1 = 0.67 T0, n1 = 0.31 n0,
    gamma = 5/3), and its impulse over the 10-90 % debris-ion arrival (1.18-2.43 us, Appendix E) and the shot;
  * the hottest liquid below the surface, its depth, and the boiling point at the recoil pressure (the same law
    inverted);
  * the limit of superheat estimated as the van der Waals spinodal at low pressure, T_sp = 27/32 T_c, for the
    critical-temperature estimates tabulated by Zaghloul and Hassanein (arXiv:2001.03557, Table 1 and their own
    3495 K, corrected to 3084 K in the same version).
The film model is static: these are the loads on the film, not its response.

Usage: python scripts/diag_li_surface_pressure.py [--out=results/revision/li_surface_pressure]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ("Demo_d1000um", "Demo_d100um", "Demo_d56.2um")
RECOIL = (1 + 5 / 3) * 0.31 * 0.67
DEBRIS_WINDOW_S = (1.18e-6, 2.43e-6)
TC_ESTIMATES_K = {"Dillon et al. 1966": 3223.0, "Ohse et al. 1985 (a)": 3503.0, "Ohse et al. 1985 (b)": 3344.0,
                  "Fortov et al. 1990": 3223.0, "Hess 1992": 4176.0, "Gathers 1994": 3741.2, "Likalter et al. 2002": 3350.0,
                  "Zaghloul and Hassanein": 3495.0, "Zaghloul and Hassanein, corrected": 3084.0}


def psat(T):
    return 10.0 ** (10.015 - 8064.5 / np.asarray(T, float))


def t_boil(p):
    return 8064.5 / (10.015 - np.log10(p))


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "li_surface_pressure"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    res = {"note": __doc__.split("Usage:")[0].strip(), "recoil_factor": RECOIL,
           "spinodal_K": {k: 27 / 32 * v for k, v in TC_ESTIMATES_K.items()}, "cases": {}}
    for case in CASES:
        h = np.genfromtxt(ROOT / "results" / "revision" / "li_extra_cycle" / case / "Demo_extra_cycle_history.csv",
                          delimiter=",", names=True)
        t, Ts, Tl, dl = h["t"], h["T_surface"], h["T_li_max"], h["li_max_depth_um"]
        ok = np.isfinite(Ts)
        i, j = int(np.nanargmax(Ts)), int(np.nanargmax(Tl))
        pr = RECOIL * psat(Ts[ok])
        w = (t[ok] >= DEBRIS_WINDOW_S[0]) & (t[ok] <= DEBRIS_WINDOW_S[1])
        p_r = RECOIL * float(psat(Ts[i]))
        res["cases"][case] = {
            "peak_surface_K": float(Ts[i]), "t_peak_surface_us": float(t[i] * 1e6), "p_sat_at_peak_Pa": float(psat(Ts[i])),
            "recoil_at_peak_Pa": p_r, "impulse_debris_window_Pa_s": float(np.trapezoid(pr[w], t[ok][w])),
            "impulse_shot_Pa_s": float(np.trapezoid(pr, t[ok])),
            "peak_liquid_K": float(Tl[j]), "depth_of_peak_liquid_um": float(dl[j]), "t_peak_liquid_us": float(t[j] * 1e6),
            "boiling_point_at_recoil_K": float(t_boil(p_r)),
            "superheat_over_boiling_at_recoil_K": float(Tl[j] - t_boil(p_r)),
            "margin_to_lowest_spinodal_K": float(min(res["spinodal_K"].values()) - Tl[j])}
        print(case, {k: round(v, 4) for k, v in res["cases"][case].items()})
    print("spinodal estimates (K):", {k: round(v) for k, v in res["spinodal_K"].items()})
    (out / "li_surface_pressure.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
