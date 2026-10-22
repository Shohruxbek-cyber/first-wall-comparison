"""Helium model against the high-temperature implantation and flash-anneal experiments of
Hashimoto et al., Fusion Sci. Technol. 47 (2005) 881 (abstract): 3He implanted into tungsten at
850 C up to 1e19 m^-2; a single 2000 C flash anneal did not change the retention, whereas the same
dose given as 1000 cycles of 1e16 m^-2, each followed by a flash, lowered the NRA helium yield by
95 % (single crystal) and 70 % (polycrystal).  Revision of 20 Oct 2026 (appendix on the helium model).

Model set-up (unchanged package, as part (b) of scripts/verify_he_6_benchmarks.py): the production
1 mm grid with a closed back face, 1.3 MeV 3He (the energy of the abstracts of Gilliam et al. 2005,
same group) as one sub-bin with its NRT Frenkel pairs (xi = 1), no neutrons, rtol 1e-3.  Single
crystal: dislocation density 2e10 m^-2; polycrystal: 5e12 m^-2 (the only microstructural
difference the model can express).  ASSUMED (not in the abstract): the single-step implantation
at a constant rate over t_imp (200, 2000 or 20000 s; 2000 s followed by a 600 s hold), the flash
as a linear ramp from 850 to 2000 C in 4 s and back to 850 C 18 s after its start, then 60 s at
850 C; in the cyclic case each 1e16 m^-2 packet in the first 2 s of an 18 s cycle (ramp from 2 to
6 s, back at 18 s), run for N_CYCLES cycles, by which the retained helium has reached its plateau.
Reported: helium in the whole 1 mm and within 5 um of the surface (NRA probes the near surface).

Usage: python scripts/he_validation_hashimoto.py [--out=runs/he_retention/validation_hashimoto]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.he import constants as C  # noqa: E402
from ife_wall.he import kernels as kn  # noqa: E402
from ife_wall.he.fast import FastHe  # noqa: E402
from ife_wall.he.grid import make_grid  # noqa: E402
from ife_wall.he.model import N_W, NC  # noqa: E402
from ife_wall.he.sources import ShotSources, SubBin  # noqa: E402

T_IMP, T_FLASH = 1123.15, 2273.15
E_HE3 = 1.3e3 * 1.602176634e-16  # 1.3 MeV in J
RHO_D = {"single crystal": 2e10, "polycrystal": C.RHO_D}
N_CYCLES = 60
TOL = dict(rtol=1e-3, atol=1e-9)


def model(rho_d):
    edges = make_grid()
    fm = FastHe(edges, C.baseline(sink_V=C.Z_V * rho_d, sink_I=C.Z_I * rho_d))
    S = ShotSources(None, edges, sub_bins=[SubBin("imp", "3He", E_HE3, E_HE3, E_HE3, 1e16, 1e-7, 1e-6)], n_slices=2)
    he = S.he_prof.sum(0) / N_W * 1e6   # appm per 1e16 m^-2
    fp = S.dpa_prof.sum(0) * 1e6        # appm of Frenkel pairs per 1e16 m^-2 (NRT, xi = 1)
    return fm, S, he, fp


def helium(fm, y, depth_um=5.0):
    u = y[kn.H:].reshape(fm.n, fm.S)
    w = fm.dx * 1e-6 * NC
    trapped = (u[:, 3:3 + fm.K] * np.arange(1, fm.K + 1)[None]).sum(1) * w
    mobile = u[:, 0] * w
    near = fm.x * 1e6 < depth_um
    return {"whole_m2": float((trapped + mobile).sum()), "trapped_m2": float(trapped.sum()), "mobile_m2": float(mobile.sum()),
            "first_5um_m2": float((trapped + mobile)[near].sum()), "implanted_m2": float(y[3]) * 1e-6 * NC,
            "released_m2": float(y[0]) * 1e-6 * NC}


def single_step(rho_d, t_imp, hold=0.0, dose=1e19):
    fm, S, he, fp = model(rho_d)
    k = dose / 1e16
    ns = 24
    te = np.linspace(0.0, t_imp, ns + 1)
    fm.set_sources(te, np.tile(he * k / t_imp, (ns, 1)), np.tile(fp * k / t_imp, (ns, 1)))
    ta = t_imp + hold
    fm.set_cycle(np.array([0.0, ta, ta + 4.0, ta + 18.0, ta + 78.0]),
                 np.tile(np.array([T_IMP, T_IMP, T_FLASH, T_IMP, T_IMP])[:, None], (1, fm.n)))
    y = fm.integrate(np.zeros(fm.N), 0.0, t_imp, **TOL)
    if hold > 0:
        y = fm.integrate(y, t_imp, ta, **TOL)
    before = helium(fm, y)
    for a, b in ((ta, ta + 4.0), (ta + 4.0, ta + 18.0), (ta + 18.0, ta + 78.0)):
        y = fm.integrate(y, a, b, **TOL)
    after = helium(fm, y)
    return {"t_implant_s": t_imp, "hold_s": hold, "before_flash": before, "after_flash": after,
            "after_over_before_whole": after["whole_m2"] / before["whole_m2"],
            "after_over_before_first_5um": after["first_5um_m2"] / before["first_5um_m2"],
            "mean_He_depth_um": float((S.he_profile() * S.x * S.dx).sum() / (S.he_profile() * S.dx).sum() * 1e6),
            "frenkel_pairs_per_He": float((fp * S.dx).sum() / (he * S.dx).sum())}


def cyclic(rho_d, n=N_CYCLES):
    fm, S, he, fp = model(rho_d)
    t_imp, t_up, t_cyc = 2.0, 4.0, 18.0
    ns = 24
    te = np.linspace(0.0, t_imp, ns + 1)
    fm.set_sources(te, np.tile(he / t_imp, (ns, 1)), np.tile(fp / t_imp, (ns, 1)))
    fm.set_cycle(np.array([0.0, t_imp, t_imp + 1e-3, t_imp + t_up, t_cyc]),
                 np.tile(np.array([T_IMP, T_IMP, T_IMP, T_FLASH, T_IMP])[:, None], (1, fm.n)))
    y = np.zeros(fm.N)
    hist = []
    for c in range(n):
        for a, b in ((0.0, t_imp), (t_imp, t_imp + 1e-3), (t_imp + 1e-3, t_imp + t_up), (t_imp + t_up, t_cyc)):
            y = fm.integrate(y, a, b, **TOL)
        if (c + 1) % 10 == 0:
            hist.append({"cycle": c + 1, **helium(fm, y)})
    return hist


if __name__ == "__main__":
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    ROOT / "runs" / "he_retention" / "validation_hashimoto"))
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"assumptions": __doc__.split("Usage:")[0].strip(), "single_step": {}, "cyclic": {}}
    for label, rho in RHO_D.items():
        res["single_step"][label] = [single_step(rho, 2000.0)]
        if label == "polycrystal":
            res["single_step"][label] += [single_step(rho, 200.0), single_step(rho, 20000.0), single_step(rho, 2000.0, hold=600.0)]
        (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for label, rho in RHO_D.items():
        res["cyclic"][label] = cyclic(rho)
        last = res["cyclic"][label][-1]
        base = res["single_step"][label][0]["before_flash"]
        res["cyclic"][label + " / single step"] = {"whole": last["whole_m2"] / base["whole_m2"],
                                                    "first_5um": last["first_5um_m2"] / base["first_5um_m2"]}
        (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    res["wall_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for label, v in res["single_step"].items():
        for r in v:
            print(label, r["t_implant_s"], r["hold_s"], "retained/implanted",
                  round(r["before_flash"]["whole_m2"] / r["before_flash"]["implanted_m2"], 4),
                  "after/before", f"{r['after_over_before_whole']:.3e}", f"{r['after_over_before_first_5um']:.3e}")
    for k, v in res["cyclic"].items():
        print(k, v if isinstance(v, dict) else [(h["cycle"], f"{h['whole_m2']:.4e}", f"{h['first_5um_m2']:.3e}") for h in v])
