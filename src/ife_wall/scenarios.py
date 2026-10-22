"""HiPER scenario definitions for the production runs (values and sources: Tables 1 and F.1 of the paper).

Tungsten armor case (Garoz2016 p. 5): 1 mm W on 1 cm steel (ODS-RAFM in Garoz; BM2005
ferritic-steel properties used as a labelled substitute), LiPb coolant h = 1e4 W/m^2K.
  Experimental: 20 MJ, R = 5 m, initial 300 K, back adiabatic (author decision), single shot.
  Prototype:    50 MJ, R = 6.5 m, 1 Hz, initial = coolant = 600 K.
  Demo:        154 MJ, R = 6.5 m, 10 Hz, initial = coolant = 600 K.
Threat: x-rays 2.1 MJ x Y/154 MJ (digitized Raffray Fig. 2), arrival R/c, 20 ns square pulse
(baseline; sensitivity 0.1, 10, 50 ns); ions ARIES NRL tables (number- and energy-conserving bins since 22 Oct 2026, ife_wall.ions), scaled
by Y/154 MJ (ESTIMATED for 20 and 50 MJ), time of flight; SRIM deposition baseline.
No vapour shielding (conservative upper bound for damage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import alloys, srim
from .ions import DEFAULT_ION_SPECTRA, IonSpectrumSource, load_ion_bins
from .library import ferritic_steel, lithium, tungsten
from .source_term import SourceTerm, SquarePulse
from .thermal import BackBC, Layer, Mesh, ThermalSolver
from .xray import XraySource

C_LIGHT = 2.99792458e8
XRAY_REF_J, YIELD_REF_J = 2.1e6, 154e6


@dataclass
class Scenario:
    name: str
    yield_J: float
    R: float
    rep_Hz: float | None
    T_init: float
    back: str  # "adiabatic" | "convective"
    T_coolant: float = 600.0
    h: float = 1e4


SCENARIOS = {
    "Experimental": Scenario("Experimental", 20e6, 5.0, None, 300.0, "adiabatic", 300.0),
    "Prototype": Scenario("Prototype", 50e6, 6.5, 1.0, 600.0, "convective", 600.0),
    "Demo": Scenario("Demo", 154e6, 6.5, 10.0, 600.0, "convective", 600.0),
}


@dataclass
class Options:
    xray_tau: float = 20e-9
    xray_mode: str = "pe+incoh"
    depth_mode: str = "projected"  # or "csda"
    stretch: bool = True  # H-isotope depth stretch (ESTIMATED) on/off
    normalize_ions: bool = True  # False = the raw tables (overrides ion_spectra)
    # --- ion spectra (revision of 22 Oct 2026, F1; ife_wall.ions.ION_SPECTRA): "conserving" (production) keeps the
    # number and energy of every species; "normalized" reproduces every result before 22 Oct 2026 bit for bit ---
    ion_spectra: str = DEFAULT_ION_SPECTRA
    alpha_evap: float = 1.0
    w_lv: str = "raffray_table"
    k_scale: float = 1.0  # multiplies W thermal conductivity (property sensitivity)
    h: float | None = None  # coolant h override
    w_thickness: float = 1e-3
    steel_thickness: float = 1e-2
    dx0: float = 1e-9
    # --- damaged-conductivity rerun (2026-09-29); defaults reproduce all earlier results ---
    k_irr: float = 1.0  # factor on SOLID W conductivity in the bulk armor (radiation damage)
    skin_um: float = 0.0  # thickness of a separate W surface layer (0 = none)
    skin_factor: float = 1.0  # factor on SOLID W conductivity in that surface layer
    T_init: float | None = None  # override of the scenario's uniform initial temperature (hot start)
    trim_multilayer: bool = False  # opt-in fix: TRIM light-ion profiles also with W on steel
    # --- tungsten grade (alloy study, 2026-10-13); the default reproduces every earlier result ---
    w_material: str = "W_pure"  # key of ife_wall.alloys.GRADES
    alloy_k_proxy: float | None = None  # explicit, labelled conductivity factor for grades whose k is not sourced
    # --- reduced vapor shielding (revision of 20 Oct 2026; ife_wall.vapor); off reproduces every earlier result ---
    vapor_shielding: bool = False
    vapor_f_rr: float = 0.0  # fraction of the energy stopped in the vapor returned to the surface
    vapor_stopping_factor: float = 1.0  # stopping of the vapor relative to the cold tables (ionization; ASSUMPTION)
    vapor_tau_rr: float = 0.0  # time constant of the return (s); 0 = immediate
    # --- neutron heating (ife_wall.neutron; Sharafat et al. 2005 value for W); off reproduces every earlier result ---
    neutron_heating: bool = False
    neutron_steel_factor: float = 1.0  # steel heating per unit volume relative to tungsten (ASSUMPTION; no source)
    # --- grid growth factors (convergence study of 20 Oct 2026); the defaults are the production values ---
    grid_growth: float = 1.05  # geometric growth of the W or Li cells from the surface
    grid_growth_steel: float = 1.15  # growth of the steel cells
    # --- steel conductivity sensitivity (revision of 21 Oct 2026); None keeps the BM2005 33 W/(m K) ---
    steel_k: float | None = None
    extra: dict = field(default_factory=dict)


class Periodic(SourceTerm):
    """Repeat any source every `period` s for n shots (shot k shifted by k*period)."""

    def __init__(self, base, period, n):
        self.base, self.period, self.n = base, period, n

    def _ks(self, t0, t1):
        k0 = max(int(np.floor(t0 / self.period)), 0)
        k1 = min(int(np.floor(t1 / self.period)), self.n - 1)
        return range(k0, k1 + 1)

    def energy(self, t0, t1):
        return sum(self.base.energy(t0 - k * self.period, t1 - k * self.period) for k in self._ks(t0, t1))

    def energy_components(self, t0, t1):
        return sum(self.base.energy_components(t0 - k * self.period, t1 - k * self.period) for k in self._ks(t0, t1))

    def fractions(self, solver):
        return self.base.fractions(solver)

    def dt_limit(self, t):
        k = int(np.floor(t / self.period))
        return self.base.dt_limit(t - k * self.period) if 0 <= k < self.n else np.inf

    def next_event(self, t):
        # compare in absolute time: shifting by k*period loses ~1e-16 s of precision, which
        # could otherwise return an "event" equal to t (zero-length step; bug found 2026-09-27)
        k = max(int(np.floor(t / self.period)), 0)
        tol = 1e-12 * max(abs(t), 1e-9)
        best = np.inf
        for kk in (k, k + 1):
            if kk < self.n:
                tl = t - kk * self.period
                for _ in range(4):
                    e = self.base.next_event(tl)
                    if not np.isfinite(e) or e + kk * self.period > t + tol:
                        break
                    tl = e + tol  # skip an event that is (numerically) already reached
                if np.isfinite(e) and e + kk * self.period > t + tol:
                    best = min(best, e + kk * self.period)
        return best


def _w_material(opt: Options, solid_factor: float = 1.0):
    """Tolias W.  k_scale (old option) multiplies the whole k(T) curve.  solid_factor
    multiplies SOLID conductivity only: full factor below Tm - dTm/2, none above Tm + dTm/2
    (molten W has no lattice defects), linear blend across the mushy interval.  The wrapper
    replaces the tabulated interpolant set in Material.__post_init__, which is what the solver
    calls (ThermalSolver._conductances -> mat.k)."""
    W = tungsten(lv_option=opt.w_lv)
    f_alloy = alloys.conductivity_factor(opt)  # 1.0 for W_pure: no wrapper, the baseline is untouched
    if f_alloy != 1.0:
        k_a = W.k
        W.k = lambda T, _k=k_a, _s=f_alloy: _s * _k(T)
    if opt.k_scale != 1.0:
        k0 = W.k
        W.k = lambda T, _k=k0, _s=opt.k_scale: _s * _k(T)
    if solid_factor != 1.0:
        k1 = W.k
        lo, hi = W.Tm - W.dTm / 2, W.Tm + W.dTm / 2

        def k_damaged(T, _k=k1, _f=solid_factor, _lo=lo, _hi=hi):
            T = np.asarray(T, float)
            w = np.clip((T - _lo) / (_hi - _lo), 0.0, 1.0)  # 0 solid ... 1 liquid
            return _k(T) * (_f + (1.0 - _f) * w)
        W.k = k_damaged
    return W


def _vapor_shielded(ions, opt: Options):
    """The ion source behind the reduced vapor column (ife_wall.vapor; used only when
    opt.vapor_shielding is True)."""
    from .vapor import VaporShieldedSource
    return VaporShieldedSource(ions, f_rr=opt.vapor_f_rr, stopping_factor=opt.vapor_stopping_factor,
                               tau_rr=opt.vapor_tau_rr)


def interface_index(solver):
    """Index of the last W cell before the steel substrate (the last layer).  Identical to the
    old `argmax(layer == 1) - 1` for the two-layer (W, steel) mesh; correct with a W skin too."""
    return int(np.argmax(solver.layer == solver.layer.max())) - 1


def w_case(scen: Scenario, opt: Options = Options(), n_shots: int = 1):
    """Build the ThermalSolver for the W-armor case."""
    if not opt.stretch:
        srim.G_ANCHORS = ((50.0, 1.0), (500.0, 1.0), (5000.0, 1.0))
    else:
        srim.G_ANCHORS = ((50.0, 1.31), (500.0, 1.04), (5000.0, 1.0))
    T0 = scen.T_init if opt.T_init is None else float(opt.T_init)
    steel = ferritic_steel() if opt.steel_k is None else ferritic_steel(k=opt.steel_k)
    if opt.skin_um > 0:
        # two W layers: skin (skin_factor) + bulk (k_irr), separate Material objects
        skin_t = opt.skin_um * 1e-6
        skin = _w_material(opt, opt.skin_factor)
        bulk = _w_material(opt, opt.k_irr)
        from .thermal import _geometric_cells
        last_skin_dx = float(_geometric_cells(skin_t, opt.dx0, opt.grid_growth, np.inf)[-1])
        layers = [Layer(skin, skin_t, opt.dx0, opt.grid_growth),
                  Layer(bulk, opt.w_thickness - skin_t, last_skin_dx, opt.grid_growth, dx_first_back=2e-6),
                  Layer(steel, opt.steel_thickness, 2e-6, opt.grid_growth_steel)]
        els = ["W", "W", "Fe"]
    else:
        W = _w_material(opt, opt.k_irr)
        layers = [Layer(W, opt.w_thickness, opt.dx0, opt.grid_growth, dx_first_back=2e-6),
                  Layer(steel, opt.steel_thickness, 2e-6, opt.grid_growth_steel)]
        els = ["W", "Fe"]
    mesh = Mesh.build(layers, T0)
    area = 4 * np.pi * scen.R ** 2
    xs = XraySource(XRAY_REF_J * scen.yield_J / YIELD_REF_J / area,
                    SquarePulse(scen.R / C_LIGHT, opt.xray_tau), els, mode=opt.xray_mode)
    ions = IonSpectrumSource(load_ion_bins(opt.normalize_ions, spectra=opt.ion_spectra), scen.yield_J, scen.R, els,
                             srim.SrimStopping(mode=opt.depth_mode, multilayer_trim=opt.trim_multilayer))
    if opt.vapor_shielding:
        if n_shots > 1:
            raise ValueError("vapor shielding is implemented for single shots only")
        ions = _vapor_shielded(ions, opt)
    sources = [xs, ions]
    if opt.neutron_heating:
        from .neutron import neutron_source
        sources.append(neutron_source(scen, els, T0, mesh.materials, opt.neutron_steel_factor))
    if n_shots > 1:
        sources = [Periodic(s, 1.0 / scen.rep_Hz, n_shots) for s in sources]
    h = opt.h if opt.h is not None else scen.h
    back = BackBC("convective", h=h, T_coolant=scen.T_coolant) if scen.back == "convective" else BackBC()
    return ThermalSolver(mesh, T0, back=back, sources=sources, alpha_evap=opt.alpha_evap, evaporation=True)


def li_case(scen: Scenario, delta: float, T_li: float, opt: Options = Options(), li_lv="davison_slope"):
    """Li film of thickness delta on 1 cm steel.  ASSUMPTION: film and substrate start uniform
    at the film temperature T_li, with the coolant (if any) also at T_li, so each shot hits a
    fresh film on a substrate in equilibrium with it (complete film replacement)."""
    srim.G_ANCHORS = ((50.0, 1.31), (500.0, 1.04), (5000.0, 1.0))
    Li = lithium(lv_option=li_lv)
    steel = ferritic_steel() if opt.steel_k is None else ferritic_steel(k=opt.steel_k)
    dx0 = min(opt.dx0 * 10, delta / 100)
    mesh = Mesh.build([Layer(Li, delta, dx0, opt.grid_growth, dx_first_back=min(delta / 50, 2e-6)),
                       Layer(steel, opt.steel_thickness, min(delta / 50, 1e-6), opt.grid_growth_steel)], T_li)
    area = 4 * np.pi * scen.R ** 2
    xs = XraySource(XRAY_REF_J * scen.yield_J / YIELD_REF_J / area,
                    SquarePulse(scen.R / C_LIGHT, opt.xray_tau), ["Li", "Fe"], mode=opt.xray_mode)
    ions = IonSpectrumSource(load_ion_bins(opt.normalize_ions, spectra=opt.ion_spectra), scen.yield_J, scen.R, ["Li", "Fe"],
                             srim.SrimStopping(mode=opt.depth_mode))
    if opt.vapor_shielding:
        ions = _vapor_shielded(ions, opt)
    sources = [xs, ions]
    if opt.neutron_heating:
        from .neutron import neutron_source
        sources.append(neutron_source(scen, ["Li", "Fe"], T_li, mesh.materials, opt.neutron_steel_factor))
    T0 = T_li
    h = opt.h if opt.h is not None else scen.h
    back = BackBC("convective", h=h, T_coolant=T_li) if scen.back == "convective" else BackBC()
    return ThermalSolver(mesh, T0, back=back, sources=sources, alpha_evap=opt.alpha_evap, evaporation=True)


# ---- converged pre-shot state of the repeated-shot tungsten runs (revision of 22 Oct 2026) ----------------------
# The uniform hot start of the single-shot studies and the start of the helium-model cycles.  For the normalized
# spectra (every result before 22 Oct 2026) each script keeps the value it used then (`legacy`); otherwise the value
# is read from results/revision/converged_state/converged_state.json (scripts/diag_converged_state.py, from the final
# checkpoints of runs/w_multi/baseline: Demo after 600 shots, Prototype after 100), rounded to 0.1 K as before.
CONVERGED_STATE = Path(__file__).resolve().parents[2] / "results" / "revision" / "converged_state" / "converged_state.json"
if not CONVERGED_STATE.exists():  # a fresh copy of the release: the stored output of the tungsten_repeated stage
    CONVERGED_STATE = (Path(__file__).resolve().parents[2] / "results" / "tungsten_repeated" / "revision"
                       / "converged_state" / "converged_state.json")


def _converged(name, key, legacy, spectra):
    from .ions import DEFAULT_ION_SPECTRA
    sp = DEFAULT_ION_SPECTRA if spectra is None else spectra
    if sp == "normalized":
        if legacy is None:
            raise ValueError(f"no legacy {key} for {name}")
        return legacy
    import json
    d = json.loads(CONVERGED_STATE.read_text(encoding="utf-8"))
    if d["ion_spectra"] != sp:
        raise RuntimeError(f"{CONVERGED_STATE} was made with the {d['ion_spectra']} spectra, not {sp}")
    return round(float(d["scenarios"][name][key]), 1)


def converged_preshot(name, legacy=None, spectra=None):
    """Converged pre-shot tungsten surface temperature (K) of the scenario's repeated-shot run."""
    return _converged(name, "surface_K", legacy, spectra)


def converged_back_face(name, legacy=None, spectra=None):
    """True back-face temperature of the steel (K) in the same converged pre-shot state."""
    return _converged(name, "back_face_K", legacy, spectra)
