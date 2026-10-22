"""Named model cases (baseline + the sensitivity cases of Table I.5 of the paper).  Each case is a dict of
overrides for constants.baseline(...) ('params') and for FastSimulation(...) ('sim')."""
from __future__ import annotations

from . import constants as C

CASES = {
    "baseline": dict(),
    # He migration energy (D0 kept: only the 0.13 eV / 2.95e-8 pair is a consistent source pair)
    "Em_He_low": dict(params=dict(Em_He=0.06)),
    "Em_He_high": dict(params=dict(Em_He=0.28)),
    # He-V binding energies
    "Eb_low": dict(params=dict(Eb=C.EB_LOW.copy())),
    "Eb_high": dict(params=dict(Eb=C.EB_HIGH.copy())),
    "Eb_flat": dict(params=dict(Eb=C.EB_FLAT.copy())),
    # vacancy production
    "xi_low": dict(sim=dict(xi=C.XI_ARC_10KEV)),
    "dpa_arc": dict(sim=dict(dpa_kind="arc")),
    # surface boundary condition (conductance of the free surface reduced to 10 %)
    "surface_0p1": dict(params=dict(surface_factor=0.1)),
    # base temperature (whole cycle shifted)
    "T_m50": dict(sim=dict(T_offset=-50.0)),
    "T_p50": dict(sim=dict(T_offset=+50.0)),
    # Level 2
    "level2": dict(params=dict(level2="L2")),
    # further constants
    "nu_low": dict(params=dict(nu=C.NU_LOW)),
    "nu_high": dict(params=dict(nu=C.NU_HIGH)),
    "riv_high": dict(params=dict(r_iv=C.R_IV_HIGH)),
    "rho_d_low": dict(params=dict(sink_V=C.Z_V * C.RHO_D_LOW, sink_I=C.Z_I * C.RHO_D_LOW)),
    "rho_d_high": dict(params=dict(sink_V=C.Z_V * C.RHO_D_HIGH, sink_I=C.Z_I * C.RHO_D_HIGH)),
    "Em_V_low": dict(params=dict(Em_V=1.5)),
    "Em_V_high": dict(params=dict(Em_V=2.0)),
    "Em_I_high": dict(params=dict(Em_I=0.165, D0_I=9.98e-11)),
    "no_ann": dict(params=dict(ann_hev=False)),
    "no_neutron": dict(sim=dict(neutron=False)),
    # displacements counted per unit path length instead of per unit projected depth (revision of 20 Oct 2026)
    "dpa_path": dict(sim=dict(path_element="path")),
}


def build(name, **extra):
    """-> (Params, FastSimulation kwargs) for a case name."""
    c = CASES[name]
    p = dict(c.get("params", {}))
    if p.get("level2") == "L2":
        p["level2"] = C.level2_params()
    params = C.baseline(**p)
    sim = dict(c.get("sim", {}))
    sim.update(extra)
    return params, sim
