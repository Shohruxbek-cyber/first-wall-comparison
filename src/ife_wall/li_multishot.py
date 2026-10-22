"""Multi-shot lithium film: the film is replaced every shot, the steel is carried forward.

Added 2026-10-06.  NEW CODE PATH ONLY: `li_case`
(scenarios.py), `ThermalSolver` and `EnergyLedger` are unchanged; this module wraps a solver
built by `li_case`.

Modelling choice, not a computed result
---------------------------------------
* At the start of every shot after the first, the Li layer is rebuilt on its original mass
  grid (full thickness at T_li) at the uniform temperature T_li, while the steel enthalpy
  profile is carried forward unchanged.  This represents complete film replacement between
  shots (the assumption of the single-shot runs) without resetting the steel.
* The net enthalpy removed by the replacement (old film out, fresh film in) is booked in the
  ledger as `removed_with_film`, so the energy balance still closes.
* A real flowing film also carries heat away along the flow, which a 1D model cannot
  represent.  Because a thin film equilibrates with the steel front face within the cycle
  (delta^2/alpha ~ 40 us for 31.6 um), the replacement acts as a front-face heat sink of
  m_Li c_p (T_front - T_li) per cycle.  Relative to a stagnant film it is therefore
  non-conservative for the steel temperature; relative to a real flowing film the sign is
  unknown.
* The coolant sits at a fixed T_coolant with fixed h, as in the W case, not at T_li; the
  steel starts uniform at T_coolant.  With the baseline T_li = 600 K = T_coolant this equals
  the single-shot initial state.
* Each shot is integrated with the solver clock reset to 0 and the single-shot sources, so
  every shot follows exactly the single-shot time stepping (no `Periodic` wrapper is needed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenarios import Options, Scenario, li_case
from .thermal import BackBC, EnergyLedger, ThermalSolver


@dataclass
class LiLedger(EnergyLedger):
    """EnergyLedger with the enthalpy removed by film replacement (J/m^2, net of the fresh film)."""

    removed_with_film: float = 0.0

    def residual(self):
        d_stored = self.stored_final - self.stored_initial
        res = (self.deposited + self.front_flux_in - self.to_coolant - self.vaporized
               - self.removed_with_film - d_stored)
        scale = max(abs(self.deposited) + abs(self.front_flux_in), abs(d_stored), 1e-300)
        return res, res / scale


def steel_index(solver: ThermalSolver) -> int:
    """Index of the first steel cell (layer 1), i.e. the steel front face."""
    return int(np.argmax(solver.layer == 1))


def reset_film(solver: ThermalSolver, T_li: float) -> float:
    """Replace the Li layer (layer 0) by a fresh film: original mass grid `m_target0`, uniform
    T_li.  The steel cells are untouched.  Returns the net enthalpy removed (J/m^2), which is
    also added to `solver.ledger.removed_with_film`.  Clears the deposition-fraction cache."""
    s = solver.layer == 0
    E_old = float(np.sum(solver.m[s] * solver.H[s]))
    m_new = solver.m_target0.copy()
    H_new = solver.materials[0].h_of_T(np.full(len(m_new), float(T_li)))
    E_new = float(np.sum(m_new * H_new))
    rest = ~s
    solver.m = np.r_[m_new, solver.m[rest]]
    solver.H = np.r_[H_new, solver.H[rest]]
    solver.layer = np.r_[np.zeros(len(m_new), dtype=solver.layer.dtype), solver.layer[rest]]
    solver._frac_cache = None
    removed = E_old - E_new
    solver.ledger.removed_with_film += removed
    return removed


def renew_film(solver: ThermalSolver, T_li: float, fraction: float) -> tuple[float, float]:
    """Partial renewal (revision of 22 Oct 2026): `fraction` of the film left after the
    shot is replaced by fresh lithium at T_li, and the evaporated mass is made up with fresh lithium; the retained
    lithium keeps its enthalpy (the film temperature is carried forward).  The film is rebuilt on its original mass
    grid `m_target0` at the uniform mixed enthalpy: a thin film is close to isothermal at the end of the period
    (delta^2/alpha ~ 40 us for 31.6 um against 0.1 s).  fraction = 1 is `reset_film`.  Returns the net enthalpy
    removed (J/m^2, booked in `solver.ledger.removed_with_film` as for `reset_film`) and the film temperature after
    the renewal (K).  ASSUMPTION: uniform mixing; the 1D model has no flow along the wall."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"renewal fraction {fraction} not in (0, 1]")
    s = solver.layer == 0
    mat = solver.materials[0]
    E_old = float(np.sum(solver.m[s] * solver.H[s]))
    M_old = float(solver.m[s].sum())
    m_new = solver.m_target0.copy()
    M0 = float(m_new.sum())
    h_fresh = float(mat.h_of_T(np.array([float(T_li)]))[0])
    kept = (1.0 - fraction) * min(M_old, M0)
    h_mix = (kept * (E_old / M_old) + (M0 - kept) * h_fresh) / M0 if M_old > 0 else h_fresh
    H_new = np.full(len(m_new), h_mix)
    E_new = float(np.sum(m_new * H_new))
    rest = ~s
    solver.m = np.r_[m_new, solver.m[rest]]
    solver.H = np.r_[H_new, solver.H[rest]]
    solver.layer = np.r_[np.zeros(len(m_new), dtype=solver.layer.dtype), solver.layer[rest]]
    solver._frac_cache = None
    removed = E_old - E_new
    solver.ledger.removed_with_film += removed
    return removed, float(mat.T_of_h(np.array([h_mix]))[0])


def deposition_split(solver: ThermalSolver):
    """Static (t = 0 mesh) energy per shot deposited in the Li and in the steel by each source,
    J/m^2: {'xray_in_Li', 'xray_in_Fe', 'ion_in_Li', 'ion_in_Fe'}.  Same arithmetic as
    scripts/run_li_scan.py's absorbed fraction."""
    out = {}
    li, fe = solver.layer == 0, solver.layer == 1
    for src in solver.sources:
        fr = np.asarray(src.fractions(solver))
        if fr.ndim == 2:
            e = np.array([F for F in src.fluence])
            out["ion_in_Li"] = float(e @ fr[:, li].sum(axis=1))
            out["ion_in_Fe"] = float(e @ fr[:, fe].sum(axis=1))
        elif type(src).__name__ == "NeutronHeatingSource":  # optional source (ife_wall.neutron)
            out["neutron_in_Li"] = float(src.fluence * fr[li].sum())
            out["neutron_in_Fe"] = float(src.fluence * fr[fe].sum())
        else:
            out["xray_in_Li"] = float(src.fluence * fr[li].sum())
            out["xray_in_Fe"] = float(src.fluence * fr[fe].sum())
    return out


def li_multishot_case(scen: Scenario, delta: float, T_li: float, opt: Options = Options(),
                      li_lv: str = "davison_slope", T_coolant: float = 600.0) -> ThermalSolver:
    """Solver for the multi-shot Li film: `li_case` plus a fixed coolant temperature, the steel
    initially at T_coolant, and a `LiLedger`.  With T_coolant == T_li the state is identical
    to `li_case`'s."""
    s = li_case(scen, delta, T_li, opt, li_lv=li_lv)  # with vapor shielding, the driver resets the column every shot
    if scen.back == "convective":
        h = opt.h if opt.h is not None else scen.h
        s.back = BackBC("convective", h=h, T_coolant=float(T_coolant))
    if float(T_coolant) != float(T_li):
        fe = s.layer == 1
        s.H[fe] = s.materials[1].h_of_T(np.full(int(fe.sum()), float(T_coolant)))
    s.ledger = LiLedger(through_interface=[0.0] * (len(s.materials) - 1))
    s.ledger.stored_initial = s.stored_energy()
    return s
