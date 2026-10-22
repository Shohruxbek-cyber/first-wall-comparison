"""A1 diagnostic for the near-surface-layer sweep (16 Oct 2026): thermal diffusion lengths in W and Li
against the deposition depths of each load component.  Reads only existing repository data; writes
results/skin_sweep/diagnostic.{json,md}.

  L = sqrt(alpha t), alpha = k / (rho c_p) from the production property functions (library.tungsten():
  Tolias 2017 solid fits; library.lithium(): Davison 1968), at the temperatures and durations listed below.
  Deposition depths: x-ray attenuation length 1/(mu/rho * rho) at the 7.09 keV number-mean photon energy of the
  digitized spectrum (data/xcom_Z74.csv, photoelectric + incoherent, log-log interpolation; the same data as
  the solver uses above 1 keV), and the energy-weighted mean / x50 / x90 depths of every component from the
  static W deposition calculation runs/deposition/fractions.json (27 Sep 2026; it took the TRIM-profile path
  for keV light ions, Section 3.8).

Usage: python scripts/skin_sweep_diagnostic.py [--out=DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.library import lithium, tungsten  # noqa: E402
from ife_wall.scenarios import converged_preshot  # noqa: E402

RHO_W_REF = 19250.0  # kg/m^3, the density used for ablation depths (Section 3.9)
E_XRAY_MEAN_KEV = 7.09  # number-weighted mean photon energy of the digitized spectrum (Table 8)

# Durations (s) and where each comes from (checked 16 Oct 2026 against the stored run outputs)
DURATIONS = [
    ("x-ray pulse", 20e-9, "20 ns square pulse, baseline (Section 3.6)"),
    ("first ions to the wall", 0.105e-6, "first ion arrival at R = 6.5 m (Section 3.7; runs/source_term)"),
    ("burn ions from about 0.4 us (Sec. 5.1 wording)", 0.4e-6, "Section 5.1: 'the burn ions from about 0.4 us'; burn t10-t90 = 0.22-0.77 us at 6.5 m (runs/source_term/source_term_checks.json)"),
    ("debris-ion peak, 1.5-2.0 us", 2.0e-6, "peak surface temperature at 1.51, 1.94, 1.97 us (Section 5.1); debris energy-median arrival 1.63 us (Section 3.7)"),
    ("10-90 % arrival window of all ions, Demo", 1.847e-6, "t10-t90 = 1847 ns (runs/source_term/garoz_comparison.md, 'all / total')"),
    ("all ions arrived", 38.6e-6, "last arrival 38.6 us at R = 6.5 m (Section 3.7)"),
    ("end of the first integration part", 50e-6, "50 us (Section 3.2)"),
    ("Li runs, end of integration", 5e-3, "5 ms (Section 3.2, Table 3)"),
    ("Demo shot period", 0.1, "10 Hz"),
    ("Prototype shot period", 1.0, "1 Hz"),
]
T_W = (300.0, 600.0, converged_preshot("Demo", legacy=921.1), 1500.0, 3000.0)
T_LI = (600.0,)


def alpha(mat, T):
    T = np.array(float(T))
    return float(mat.k(T) / (mat.rho(T) * mat.cp(T)))


def xcom_mu_rho(E_keV):
    """photoelectric + incoherent mass attenuation (cm^2/g) from data/xcom_Z74.csv, log-log interpolation."""
    rows = []
    for line in (ROOT / "data" / "xcom_Z74.csv").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("E_MeV"):
            continue
        p = line.split(",")
        rows.append((float(p[0]) * 1e3, float(p[2]) + float(p[3])))  # keV, incoherent + photoelectric
    E = np.array([r[0] for r in rows])
    mu = np.array([r[1] for r in rows])
    # drop duplicated edge rows below the target by keeping the last occurrence (no edge between 6 and 8 keV for W)
    i = np.searchsorted(E, E_keV)
    E0, E1, m0, m1 = E[i - 1], E[i], mu[i - 1], mu[i]
    return float(np.exp(np.log(m0) + (np.log(E_keV) - np.log(E0)) / (np.log(E1) - np.log(E0)) * (np.log(m1) - np.log(m0)))), (E0, m0, E1, m1)


def main(outdir):
    W, Li = tungsten(), lithium()
    out = {"convention": "L = sqrt(alpha t), alpha = k/(rho c_p); solid-W fits of Tolias 2017 (library.tungsten), Li of Davison 1968",
           "alpha_W_m2_s": {f"{T:g}": alpha(W, T) for T in T_W}, "alpha_Li_m2_s": {f"{T:g}": alpha(Li, T) for T in T_LI},
           "k_rho_cp_W": {f"{T:g}": {"k_W_mK": float(W.k(np.array(T))), "rho_kg_m3": float(W.rho(np.array(T))), "cp_J_kgK": float(W.cp(np.array(T)))} for T in T_W},
           "durations": [], "deposition": {}}
    for name, t, src in DURATIONS:
        row = {"component": name, "t_s": t, "source": src,
               "L_W_um": {f"{T:g}": np.sqrt(alpha(W, T) * t) * 1e6 for T in T_W},
               "L_Li_um": {f"{T:g}": np.sqrt(alpha(Li, T) * t) * 1e6 for T in T_LI}}
        out["durations"].append(row)
    mu, (E0, m0, E1, m1) = xcom_mu_rho(E_XRAY_MEAN_KEV)
    out["deposition"]["xray_attenuation_length_um_at_7.09keV"] = 1e6 / (mu * 1e-1 * RHO_W_REF)  # cm^2/g -> m^2/kg: x0.1
    out["deposition"]["xray_mu_rho_cm2_g_at_7.09keV"] = mu
    out["deposition"]["xray_interpolation_rows"] = {"E_keV": [E0, E1], "mu_rho_pe_plus_incoh_cm2_g": [m0, m1]}
    fr = json.loads((ROOT / "runs" / "deposition" / "fractions.json").read_text(encoding="utf-8"))["W"]
    out["deposition"]["fraction_of_threat_energy_within_um"] = {"0.01": fr["within_10nm"], "0.1": fr["within_100nm"], "1": fr["within_1000nm"],
                                                              "10": fr["within_10000nm"], "100": fr["within_100000nm"]}
    out["deposition"]["per_component_um"] = {k: {"mean": v["mean_um"], "x50": v["x50_um"], "x90": v["x90_um"]} for k, v in fr["per_component_depth"].items()}
    out["deposition"]["source"] = "runs/deposition/fractions.json (27 Sep 2026, Demo loads on W; the static calculation used the TRIM-profile path for keV light ions, Section 3.8)"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "diagnostic.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = ["# A1 diagnostic: diffusion lengths against deposition depths (16 Oct 2026)", "",
         f"Convention: {out['convention']}.", "",
         "## Tungsten: alpha = k/(rho c_p)", "", "| T (K) | k (W/m/K) | rho (kg/m3) | c_p (J/kg/K) | alpha (m2/s) |", "|---|---|---|---|---|"]
    for T in T_W:
        p = out["k_rho_cp_W"][f"{T:g}"]
        L.append(f"| {T:g} | {p['k_W_mK']:.1f} | {p['rho_kg_m3']:.0f} | {p['cp_J_kgK']:.1f} | {out['alpha_W_m2_s'][f'{T:g}']:.3e} |")
    L += ["", "## Diffusion length sqrt(alpha t) in W (um) by load component and temperature", "",
          "| component | t | " + " | ".join(f"{T:g} K" for T in T_W) + " | source |", "|---|---|" + "---|" * len(T_W) + "---|"]
    for r in out["durations"]:
        L.append(f"| {r['component']} | {r['t_s']:.3g} s | " + " | ".join(f"{r['L_W_um'][f'{T:g}']:.3g}" for T in T_W) + f" | {r['source']} |")
    L += ["", "## Liquid lithium at 600 K (alpha = %.3e m2/s)" % out["alpha_Li_m2_s"]["600"], "",
          "| component | t | L_Li (um) |", "|---|---|---|"]
    for r in out["durations"]:
        L.append(f"| {r['component']} | {r['t_s']:.3g} s | {r['L_Li_um']['600']:.4g} |")
    dep = out["deposition"]
    L += ["", "## Deposition depths in W", "",
          f"- X-ray attenuation length at the 7.09 keV number-mean photon energy: {dep['xray_attenuation_length_um_at_7.09keV']:.2f} um "
          f"(mu/rho = {dep['xray_mu_rho_cm2_g_at_7.09keV']:.1f} cm2/g, photoelectric + incoherent, log-log between the XCOM rows at 6 and 8 keV; rho = 19 250 kg/m3).",
          "- Fraction of the threat energy (x-rays + ions, Demo, W) deposited within a depth (runs/deposition/fractions.json): "
          + ", ".join(f"{float(k):g} um: {v * 100:.1f} %" for k, v in dep["fraction_of_threat_energy_within_um"].items()) + ".",
          "", "| component | energy-weighted mean depth (um) | x50 (um) | x90 (um) |", "|---|---|---|---|"]
    for k, v in dep["per_component_um"].items():
        L.append(f"| {k} | {v['mean']:.3g} | {v['x50']:.3g} | {v['x90']:.3g} |")
    L += ["", f"Source: {dep['source']}."]
    (outdir / "diagnostic.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    od = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    main(Path(od) if od else ROOT / "results" / "skin_sweep")
