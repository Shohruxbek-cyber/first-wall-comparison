"""Reduced, time-coupled vapor shielding of the ion source (revision of 20 October 2026).

Off by default: `scenarios.w_case` and `scenarios.li_case` wrap the ion source only when
`Options.vapor_shielding` is True, so with the option off every earlier result is unchanged.
Model and assumptions: Section 2.5 of the paper.

* Column.  The areal mass evaporated since the start of the shot, sigma_v(t) = M_e(t), stands
  between the wall and the incoming ions.  In 1D planar geometry at normal incidence the column
  keeps its areal mass while it expands (a few cm during the ion pulse, << R), so no expansion
  model is needed.  Single shots only: the column is not reset between shots.
* Stopping.  The vapor is one cell of the wall's front material in front of the mesh, and every
  ion component is deposited on that extended mesh with the production depth model (SRIM
  tables per unit areal mass: cold, neutral vapor).  Ionized vapor is represented by a stopping
  factor s that multiplies the column (an assumption; no sourced value for H or He ions in Li).
* Re-radiation.  A fraction f_rr of the energy stopped in the vapor returns to the surface cell,
  at once (tau_rr = 0) or relaxing from a store with time constant tau_rr; the rest leaves the
  wall's energy budget for the shot.  The returned energy is an extra source component whose
  depth profile is the surface cell, so the solver's ledger books it as deposited.
"""
from __future__ import annotations

import numpy as np

from .source_term import SourceTerm


class _ExtendedMesh:
    """What SourceTerm.fractions reads from the solver (m, layer), with one vapor cell of areal
    mass M of the front-layer material in front of the first wall cell."""

    def __init__(self, M, m, layer):
        self.m = np.r_[M, m]
        self.layer = np.r_[layer[0], layer]


class VaporShieldedSource(SourceTerm):
    """Wraps a multi-component ion source (ions.IonSpectrumSource) with the vapor column."""

    def __init__(self, base, f_rr=0.0, stopping_factor=1.0, tau_rr=0.0, fixed_column=None):
        if not 0.0 <= f_rr <= 1.0:
            raise ValueError("f_rr must lie in [0, 1]")
        if stopping_factor <= 0.0:
            raise ValueError("stopping_factor must be positive")
        if tau_rr < 0.0:
            raise ValueError("tau_rr must be >= 0")
        self.base = base
        self.f_rr, self.stopping_factor, self.tau_rr = float(f_rr), float(stopping_factor), float(tau_rr)
        self.fixed_column = fixed_column  # kg/m^2 held fixed (limiting-case tests); None: the evaporated mass
        self.bins = base.bins
        self.fluence = np.r_[base.fluence, 0.0]  # last component: returned radiation (no incident fluence)
        self.t_first, self.t_last = base.t_first, base.t_last
        self.vapor_share = np.zeros(len(base.bins))  # fraction of each component's energy stopped in the vapor
        self.column = 0.0  # kg/m^2 at the last fractions() call, before the stopping factor
        self.max_column = 0.0
        # bookkeeping over accepted steps (J/m^2); a step counts as accepted when the next call
        # starts at its end time, and a retried step (same start time) is discarded
        self.E_incident = 0.0
        self.E_vapor = 0.0
        self.E_reradiated = 0.0
        self.E_store = 0.0
        self._pending = None
        self.mass_offset = 0.0  # evaporated mass at the start of the current shot (multi-shot runs)

    def start_shot(self, solver):
        """Repeated shots (scripts/run_li_multi.py): the vapor of the previous shot is assumed to
        have cleared before the next one, so the column restarts from zero.  The last step of
        the previous shot is committed first."""
        self._settle(np.inf)
        self.mass_offset = float(solver.mass_evaporated)
        self.column = 0.0

    # ---- SourceTerm interface --------------------------------------------------------------
    def total_fluence(self):
        return self.base.total_fluence()

    def energy(self, t0, t1):
        return float(self.energy_components(t0, t1).sum())

    def energy_components(self, t0, t1):
        """Energy [J/m^2] of each ion component arriving in [t0, t1], then the returned radiation."""
        self._settle(t0)
        ev = self.base.energy_components(t0, t1)
        E_v = float(ev @ self.vapor_share)
        if self.tau_rr == 0.0:
            E_rr, store = self.f_rr * E_v, 0.0
        else:
            # store fed at a constant rate f_rr E_v / dt over the step, relaxing with tau_rr
            decay = np.exp(-(t1 - t0) / self.tau_rr)
            gain = self.f_rr * E_v
            store = self.E_store * decay + gain * self.tau_rr / (t1 - t0) * (1.0 - decay)
            E_rr = self.E_store + gain - store
        self._pending = (t0, t1, float(ev.sum()), E_v, E_rr, store)
        return np.r_[ev, E_rr]

    def fractions(self, solver):
        """(n_components + 1, n_cells): the wall fractions of every ion component behind the
        column, and the surface-cell profile of the returned radiation."""
        M = self.fixed_column if self.fixed_column is not None else solver.mass_evaporated - self.mass_offset
        self.column = float(M)
        self.max_column = max(self.max_column, self.column)
        if M <= 0.0:
            wall = np.asarray(self.base.fractions(solver))
            self.vapor_share = np.zeros(len(self.bins))
        else:
            ext = _ExtendedMesh(self.stopping_factor * M, np.asarray(solver.m), np.asarray(solver.layer))
            fr = np.asarray(self.base.fractions(ext))
            self.vapor_share = fr[:, 0].copy()
            wall = fr[:, 1:]
        rr = np.zeros((1, len(solver.m)))
        rr[0, 0] = 1.0
        return np.vstack([wall, rr])

    def dt_limit(self, t):
        return self.base.dt_limit(t)

    def next_event(self, t):
        return self.base.next_event(t)

    # ---- bookkeeping -------------------------------------------------------------------------
    def _settle(self, t0):
        """Commit the pending step if the solver has moved past it; drop it if it is being retried."""
        p = self._pending
        if p is None:
            return
        if t0 >= p[1]:  # the solver sets t = t1 exactly on acceptance; a retry restarts at t0 < t1
            self.E_incident += p[2]
            self.E_vapor += p[3]
            self.E_reradiated += p[4]
            self.E_store = p[5]
        self._pending = None

    def finalize(self, t):
        """Commit the last accepted step at the end of a run (call with solver.t); return the record."""
        self._settle(t)
        return self.summary()

    def summary(self):
        return {"f_rr": self.f_rr, "stopping_factor": self.stopping_factor, "tau_rr_s": self.tau_rr,
                "column_kg_m2": self.column, "max_column_kg_m2": self.max_column,
                "ion_energy_incident_J_m2": self.E_incident, "ion_energy_in_vapor_J_m2": self.E_vapor,
                "reradiated_to_wall_J_m2": self.E_reradiated, "stored_in_vapor_J_m2": self.E_store,
                "lost_from_wall_J_m2": self.E_vapor - self.E_reradiated - self.E_store,
                "fraction_of_ion_energy_in_vapor": self.E_vapor / self.E_incident if self.E_incident else 0.0}
