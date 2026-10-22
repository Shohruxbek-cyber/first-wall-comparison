"""How far a free lithium film travels along the wall before viscous drag stops it (revision of 21 Oct 2026), as an injection-spacing estimate for film renewal.

A film of thickness delta and mean speed U on a no-slip wall, with a free surface and no driving force, has the
half-parabolic (Nusselt) profile once the momentum has diffused across it (time ~ delta^2/nu); the wall shear is
3 mu U / delta, so U decays as exp(-t/tau) with tau = delta^2 / (3 nu), and the film stops within
L = U tau = U delta^2 / (3 nu).  Properties at the 600 K film temperature: density from ife_wall.library.lithium
(Davison 1968, Eq. 1), viscosity from Davison 1968, Eq. (10): log10 mu [N s m^-2] = -3.080 + 57.63/T - 5.172e-4 T.
Speeds: 15 m/s (the IFMIF/EVEDA free-surface flow, Wakai et al. 2016) and the pi R f speeds of Table 5.

Usage: python scripts/diag_film_travel.py [--out=results/revision/film_travel]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.library import lithium  # noqa: E402
from ife_wall.scenarios import SCENARIOS  # noqa: E402

T_LI = 600.0
FILMS_UM_NORMALIZED = {"Prototype": (12.2,), "Demo": (44.79, 46.0, 57.35, 58.0, 180.0, 181.5)}  # before 22 Oct 2026


def films_um():
    """The repeated-shot minima, the direct-run films and the helium-exclusion film of the current results
    (scripts/_films.py); the fixed list above for the spectra used before 22 Oct 2026."""
    from ife_wall.ions import DEFAULT_ION_SPECTRA
    if DEFAULT_ION_SPECTRA == "normalized":
        return FILMS_UM_NORMALIZED
    sys.path.insert(0, str(ROOT / "scripts"))
    from _films import current_films, direct_film_um
    f = current_films()
    p = f["Prototype"]["925 K, repeated shots"]
    d = f["Demo"]
    m925, m873 = d["925 K, repeated shots (with the direct runs)"], d["873 K, repeated shots (with the direct runs)"]
    he = d["helium, 10^6 shots to 10^18 m^-2"]
    return {"Prototype": (round(p, 1),),
            "Demo": (round(m925, 2), d["925 K, film of the direct run"], round(m873, 2), d["873 K, film of the direct run"],
                     direct_film_um(he), round(he, 1))}


def mu_li(T):
    return 10.0 ** (-3.080 + 57.63 / T - 5.172e-4 * T)


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "results" / "revision" / "film_travel"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    rho = float(lithium().rho(np.array(T_LI)))
    mu = mu_li(T_LI)
    nu = mu / rho
    res = {"note": __doc__.split("Usage:")[0].strip(), "T_K": T_LI, "rho_kg_m3": rho, "mu_Pa_s": mu, "nu_m2_s": nu, "cases": []}
    for name, films in films_um().items():
        sc = SCENARIOS[name]
        U_piRf = math.pi * sc.R * sc.rep_Hz
        for d_um in films:
            d = d_um * 1e-6
            tau = d * d / (3 * nu)
            for U in (15.0, U_piRf):
                res["cases"].append({"scenario": name, "delta_um": d_um, "U_m_s": U, "momentum_diffusion_time_s": d * d / nu,
                                     "decay_time_s": tau, "travel_m": U * tau, "shot_period_s": 1 / sc.rep_Hz,
                                     "path_piR_m": math.pi * sc.R})
    (out / "film_travel.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"rho {rho:.1f} kg/m3, mu {mu:.4e} Pa s, nu {nu:.4e} m2/s")
    for c in res["cases"]:
        print(f"{c['scenario']:9s} {c['delta_um']:6.1f} um  U {c['U_m_s']:6.1f} m/s: delta^2/nu {c['momentum_diffusion_time_s'] * 1e3:.2f} ms, "
              f"tau {c['decay_time_s'] * 1e3:.2f} ms, travel {c['travel_m'] * 100:.2f} cm (path pi R {c['path_piR_m']:.1f} m)")


if __name__ == "__main__":
    main()
