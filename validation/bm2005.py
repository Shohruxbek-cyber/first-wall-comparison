"""Benchmark: Blanchard & Martin (2005), J. Nucl. Mater. 347:192 -- HAPL baseline W-coated steel wall.

Their case: 150 MJ, R = 7.0 m, no gas; 250 um W on 3 mm steel; coolant 400 C,
h = 1e4 W/m^2K, initial 400 C; 5 Hz; free surface adiabatic.  Loads (Table 1, p. 193;
depths p. 195): x-rays 0.01 Y (0 ns, 1 ns, uniform 2 um), neutrons 0.70 Y (160 ns, 20 ns,
uniform over W + steel), burn ions 0.12 Y (200 ns, 800 ns, uniform 10 um), debris ions
0.17 Y (1000 ns, 2750 ns, uniform 1 um).  Properties: BM2005 Appendix A.
Targets: peak W 2554 C; 1393 C at 10 um; 849 C at 25 um (p. 196, Fig. 2).

Variants: (a) our Tolias W instead of their ITER-MPH W; (b) Table 2 fluences (x-ray 0.32,
burn 2.9, debris 4.0 J/cm^2) instead of Table 1 fractions; (c) evaporation on.
Usage: python validation/bm2005.py [n_shots]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.library import ferritic_steel, tungsten, tungsten_bm2005  # noqa: E402
from ife_wall.source_term import (PeriodicSource, SquarePulse, UniformDepthSource,  # noqa: E402
                                  VolumetricUniformSource)
from ife_wall.thermal import BackBC, Layer, Mesh, ThermalSolver  # noqa: E402

Y, R = 150e6, 7.0
AREA = 4 * np.pi * R ** 2
T0 = 400.0 + 273.15
PERIOD = 0.2


def build(w_mat, fluences, n_shots, evaporation=False):
    steel = ferritic_steel()
    mesh = Mesh.build([Layer(w_mat, 250e-6, 5e-9, 1.06, dx_first_back=0.5e-6),
                       Layer(steel, 3e-3, 0.5e-6, 1.1)], T0)
    rho0 = float(w_mat.rho(np.array(T0)))
    Fx, Fn, Fb, Fd = fluences
    srcs = [UniformDepthSource(Fx, SquarePulse(0.0, 1e-9), 2e-6, rho0),
            VolumetricUniformSource(Fn, SquarePulse(160e-9, 20e-9)),
            UniformDepthSource(Fb, SquarePulse(200e-9, 800e-9), 10e-6, rho0),
            UniformDepthSource(Fd, SquarePulse(1000e-9, 2750e-9), 1e-6, rho0)]
    srcs = [PeriodicSource(s, PERIOD, n_shots) for s in srcs]
    return ThermalSolver(mesh, T0, back=BackBC("convective", h=1e4, T_coolant=T0), sources=srcs,
                         evaporation=evaporation, alpha_evap=1.0)


def run(label, w_mat, fluences, n_shots, evaporation=False):
    t_start = time.time()
    s = build(w_mat, fluences, n_shots, evaporation)
    per_shot = []
    for k in range(n_shots):
        pk = {"Ts": 0.0, "T10": 0.0, "T25": 0.0, "Tif": 0.0}

        def rec(sv):
            T = sv.T
            x = sv.x_centers()
            pk["Ts"] = max(pk["Ts"], sv.surface_temperature(T))
            pk["T10"] = max(pk["T10"], float(np.interp(10e-6, x, T)))
            pk["T25"] = max(pk["T25"], float(np.interp(25e-6, x, T)))
            i_if = int(np.argmax(sv.layer == 1)) - 1
            pk["Tif"] = max(pk["Tif"], float(T[i_if]))

        s.run(k * PERIOD + 50e-6, dt0=1e-12, dT_target=5.0, record=rec)  # pulse + 50 us
        s.run((k + 1) * PERIOD, dt0=1e-9, dT_target=5.0, record=rec)
        pk["T_preshot_next"] = s.surface_temperature()
        per_shot.append({kk: v - 273.15 for kk, v in pk.items()})  # deg C
    res, rel = s.ledger.residual()
    out = {"label": label, "per_shot_C": per_shot, "energy_rel_residual": rel,
           "evaporated_nm": s.mass_evaporated / 19250 * 1e9, "wall_time_s": time.time() - t_start}
    print(f"{label}: shot1 Ts={per_shot[0]['Ts']:.0f} C, T10={per_shot[0]['T10']:.0f}, T25={per_shot[0]['T25']:.0f}; "
          f"last-shot Ts={per_shot[-1]['Ts']:.0f}, pre-shot={per_shot[-1]['T_preshot_next']:.0f}; "
          f"closure {rel:.1e}; {out['wall_time_s']:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    table1 = [f * Y / AREA for f in (0.01, 0.70, 0.12, 0.17)]
    table2 = [3.2e3, 0.70 * Y / AREA, 2.9e4, 4.0e4]
    results = [run("BM props, Table-1 loads (baseline)", tungsten_bm2005(), table1, n),
               run("Tolias W, Table-1 loads", tungsten(), table1, 1),
               run("BM props, Table-2 fluences", tungsten_bm2005(), table2, 1),
               run("Tolias W, Table-1 loads, evaporation on (alpha=1)", tungsten(), table1, 1, evaporation=True)]
    results.append({"targets_C": {"Ts": 2554, "T10": 1393, "T25": 849}})
    out = ROOT / "runs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bm2005.json").write_text(json.dumps(results, indent=2))
    print("saved", out / "bm2005.json")
