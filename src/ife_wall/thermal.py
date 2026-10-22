"""1D transient conduction with melting and surface vaporization (enthalpy method).

Formulation
-----------
* Lagrangian (mass) coordinate.  Each cell i carries a fixed areal mass m_i [kg/m^2]
  (reduced only by evaporation at the surface).  Cell thickness dx_i = m_i / rho(T_i), so
  temperature-dependent density needs no mesh motion and deposition data given in mass
  units (XCOM mu/rho, STAR stopping powers) apply directly.
* Unknown: specific enthalpy H_i [J/kg].  Backward Euler:
      m_i^{n+1} (H_i^{n+1} - H_i^n)/dt = F_{i-1/2} - F_{i+1/2} + S_i - [i=0] j L_v
  where F are conductive fluxes (face conductance from the harmonic combination of
  half-cells), S_i the deposited power [W/m^2] and j the HKL mass flux [kg/m^2 s].
  Solved by Newton iteration on H with T = T(H) from the material's enthalpy table.
* Surface vaporization: j = alpha p_sat(T_0) sqrt(M / 2 pi R T_0), evaluated at the
  surface-cell temperature (cells there are ~1 nm).  Evaporated mass leaves with its
  enthalpy H_0^n plus the latent heat L_v(T_0).  When the surface cell has lost half its
  target mass, the front layer is conservatively remapped onto its original mass grid
  measured from the new surface ("rezoning"), preserving mass and enthalpy exactly.
* Boundaries: front = adiabatic + optional imposed flux or Dirichlet temperature (tests);
  back = convection to coolant, or adiabatic.

Energy bookkeeping is accumulated independently of the solve and checked against the
change in stored enthalpy (see `EnergyLedger`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.linalg import solve_banded

from .materials import Material


# --------------------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------------------
@dataclass
class Layer:
    material: Material
    thickness: float  # m, at the initial temperature
    dx_first: float  # m, first cell at the plasma-facing side of the layer
    growth: float = 1.1  # geometric growth ratio
    dx_max: float = np.inf  # m
    dx_first_back: Optional[float] = None  # m; if set, cells also refine towards the back face


def _geometric_cells(L, dx0, r, dx_max, dx0_back=None):
    """Cell widths filling length L: geometric growth from dx0 (and optionally from dx0_back)."""
    if dx0_back is None:
        cells, x = [], 0.0
        while x < L - 1e-15 * L:
            d = min(dx0 * r ** len(cells), dx_max, L - x)
            cells.append(d)
            x += d
        if len(cells) > 1 and cells[-1] < 0.5 * cells[-2]:  # merge a sliver into its neighbour
            sliver = cells.pop()  # (not `cells[-2] += cells.pop()`: the index would shift)
            cells[-1] += sliver
        return np.array(cells)
    front = _geometric_cells(L / 2, dx0, r, dx_max)
    back = _geometric_cells(L / 2, dx0_back, r, dx_max)[::-1]
    return np.concatenate([front, back])


@dataclass
class Mesh:
    m: np.ndarray  # kg/m^2 per cell
    layer: np.ndarray  # layer index per cell
    materials: List[Material]

    @classmethod
    def build(cls, layers: Sequence[Layer], T_init: float):
        m, idx = [], []
        for j, lay in enumerate(layers):
            dx = _geometric_cells(lay.thickness, lay.dx_first, lay.growth, lay.dx_max, lay.dx_first_back)
            m.append(dx * float(lay.material.rho(np.array(T_init))))
            idx.append(np.full(len(dx), j))
        return cls(np.concatenate(m), np.concatenate(idx), [l.material for l in layers])


# --------------------------------------------------------------------------------------
# Energy ledger
# --------------------------------------------------------------------------------------
@dataclass
class EnergyLedger:
    """All in J/m^2, accumulated over the run."""

    incident: float = 0.0  # energy of the threat arriving at the surface
    deposited: float = 0.0  # absorbed in the mesh (incident minus pass-through)
    passed_through: float = 0.0  # left through the back of the domain without absorption
    front_flux_in: float = 0.0  # imposed surface flux / Dirichlet exchange (tests)
    to_coolant: float = 0.0  # removed through the back boundary
    vaporized_latent: float = 0.0
    vaporized_sensible: float = 0.0  # enthalpy carried by the evaporated mass
    through_interface: List[float] = field(default_factory=list)  # conducted across layer interfaces
    stored_initial: float = 0.0
    stored_final: float = 0.0

    @property
    def vaporized(self):
        return self.vaporized_latent + self.vaporized_sensible

    def residual(self):
        """(inputs - outputs - change in storage), and relative to the energy throughput."""
        d_stored = self.stored_final - self.stored_initial
        res = self.deposited + self.front_flux_in - self.to_coolant - self.vaporized - d_stored
        scale = max(abs(self.deposited) + abs(self.front_flux_in), abs(d_stored), 1e-300)
        return res, res / scale


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------
@dataclass
class BackBC:
    kind: str = "adiabatic"  # "adiabatic" | "convective"
    h: float = 0.0  # W/(m^2 K)
    T_coolant: float = 300.0  # K


@dataclass
class FrontBC:
    kind: str = "adiabatic"  # "adiabatic" | "flux" | "dirichlet"
    flux: Callable[[float], float] = lambda t: 0.0  # W/m^2 into the wall
    T_wall: float = 0.0


class SourceTerm:
    """Interface for volumetric heating.  Subclasses (x-ray, ions) implement:

    energy(t0, t1)      -> J/m^2 arriving at the surface during [t0, t1] (exact integral)
    fractions(solver)   -> per-cell fraction of that power absorbed (sums to <= 1)
    dt_limit(t)         -> largest acceptable time step at t (s)
    next_event(t)       -> time of the next abrupt change after t (or inf)
    """

    def energy(self, t0, t1): return 0.0
    def fractions(self, solver): return np.zeros(len(solver.m))
    def dt_limit(self, t): return np.inf
    def next_event(self, t): return np.inf


class ThermalSolver:
    def __init__(self, mesh: Mesh, T_init, back: BackBC = BackBC(), front: FrontBC = FrontBC(),
                 sources: Sequence[SourceTerm] = (), alpha_evap: float = 1.0,
                 evaporation: bool = True, rezone_fraction: float = 0.5):
        self.materials = mesh.materials
        self.m = mesh.m.astype(float).copy()
        self.layer = mesh.layer.copy()
        self.m_target0 = self.m[self.layer == 0].copy()  # original front-layer mass grid
        T_init = np.broadcast_to(np.asarray(T_init, dtype=float), self.m.shape).copy()
        self.H = self._by_layer(lambda mat, s: mat.h_of_T(T_init[s]))
        self.back, self.front = back, front
        self.sources = list(sources)
        self.alpha_evap = alpha_evap
        self.evaporation = evaporation and self.materials[0].vapor is not None
        self.rezone_fraction = rezone_fraction
        self.t = 0.0
        self.mass_evaporated = 0.0  # kg/m^2
        self.n_rezones = 0
        self.ledger = EnergyLedger(through_interface=[0.0] * (len(self.materials) - 1))
        self.ledger.stored_initial = self.stored_energy()
        self._frac_cache = None
        # Newton bookkeeping (16 Oct 2026, skin sweep): counters only, read after the run; they
        # touch no floating-point operation of the solve, so the numerics are unchanged.
        self.n_steps_accepted = 0
        self.n_steps_rejected = 0
        self.newton_iters_total = 0  # iterations summed over the accepted steps
        self.newton_iters_rejected = 0  # iterations spent in rejected attempts
        self.newton_iters_max_step = 0  # largest count in one accepted step (cap: max_iter = 60)
        self.newton_last_update_K = float("nan")  # largest |dT| update at the last iteration of the last accepted step

    def newton_stats(self):
        """Snapshot of the Newton counters (cumulative since the solver was built)."""
        return {"steps_accepted": self.n_steps_accepted, "steps_rejected": self.n_steps_rejected,
                "iters_total_accepted": self.newton_iters_total, "iters_in_rejected_attempts": self.newton_iters_rejected,
                "iters_max_per_step": self.newton_iters_max_step, "last_step_final_update_K": self.newton_last_update_K,
                "tol_T_K": 1e-6, "max_iter": 60}

    def _reject(self, n_it):
        self.n_steps_rejected += 1
        self.newton_iters_rejected += n_it

    # ---- property helpers -----------------------------------------------------------
    def _by_layer(self, fn):
        out = np.empty(len(self.m))
        for j, mat in enumerate(self.materials):
            s = self.layer == j
            if np.any(s):
                out[s] = fn(mat, s)
        return out

    def T_of(self, H):
        return self._by_layer(lambda mat, s: mat.T_of_h(H[s]))

    @property
    def T(self):
        return self.T_of(self.H)

    def dx(self, T=None):
        T = self.T if T is None else T
        return self.m / self._by_layer(lambda mat, s: mat.rho(T[s]))

    def x_centers(self):
        dx = self.dx()
        return np.cumsum(dx) - dx / 2

    def stored_energy(self):
        return float(np.sum(self.m * self.H))

    def liquid_fraction(self):
        return self._by_layer(lambda mat, s: mat.liquid_fraction(self.H[s]))

    def melt_depth(self):
        """Contiguous molten thickness from the surface (cells with liquid fraction >= 0.5), m."""
        f = self.liquid_fraction()
        dx = self.dx()
        n = np.argmax(f < 0.5) if np.any(f < 0.5) else len(f)
        return float(np.sum(dx[:n]))

    def surface_temperature(self, T=None):
        """Extrapolate from the first cell centre to the surface using the surface flux."""
        T = self.T if T is None else T
        mat = self.materials[self.layer[0]]
        k0 = float(mat.k(T[:1])[0])
        dx0 = self.dx(T)[0]
        if self.front.kind == "dirichlet":
            return self.front.T_wall
        q = self.front.flux(self.t) if self.front.kind == "flux" else 0.0
        if self.evaporation:
            q -= float(mat.vapor.hkl_flux(T[0], self.alpha_evap) * mat.vapor.L_v(T[0]))
        return float(T[0] + q * dx0 / (2 * k0))

    def _conductances(self, T):
        k = self._by_layer(lambda mat, s: mat.k(T[s]))
        dx = self.dx(T)
        r_half = dx / (2 * k)
        G = 1.0 / (r_half[:-1] + r_half[1:])  # between i and i+1
        G_back = 0.0
        if self.back.kind == "convective":
            G_back = 1.0 / (r_half[-1] + 1.0 / self.back.h)
        G_front = 1.0 / r_half[0] if self.front.kind == "dirichlet" else 0.0
        return G, G_back, G_front

    # ---- sources --------------------------------------------------------------------
    def _source_fractions(self):
        # Recompute when the cell list changes or the surface has receded by >1% of a surface cell.
        c = self._frac_cache
        if (c is None or c[0] != len(self.m)
                or abs(self.mass_evaporated - c[1]) > 0.01 * self.m_target0[0]):
            self._frac_cache = (len(self.m), self.mass_evaporated,
                                [np.asarray(s.fractions(self)) for s in self.sources])
        return self._frac_cache[2]

    def _source(self, t0, t1):
        """Average deposited power per cell over [t0, t1] (W/m^2), and incident/deposited energy."""
        dt = t1 - t0
        S = np.zeros(len(self.m))
        E_in = E_dep = 0.0
        for src, fr in zip(self.sources, self._source_fractions()):
            if fr.ndim == 2:  # multi-component source: one depth profile per component
                ev = src.energy_components(t0, t1)
                if ev.any():
                    S += (ev @ fr) / dt
                    E_in += float(ev.sum())
                    E_dep += float(ev @ fr.sum(axis=1))
                continue
            e = src.energy(t0, t1)
            if e:
                S += (e / dt) * fr
                E_in += e
                E_dep += e * fr.sum()
        return S, E_in, E_dep

    # ---- one implicit step ----------------------------------------------------------
    def step(self, dt, max_iter=60, tol_T=1e-6, dT_reject=np.inf):
        """Advance by dt.  Returns (accepted, max |dT|).  State changes only if the Newton
        iteration converged and the largest cell temperature change is <= dT_reject."""
        t1 = self.t + dt
        H0 = self.H.copy()
        H = H0.copy()
        S, E_in, E_dep = self._source(self.t, t1)
        # imposed surface flux: midpoint rule over the step (tests use constant flux)
        q_front = self.front.flux(self.t + 0.5 * dt) if self.front.kind == "flux" else 0.0
        vap = self.materials[self.layer[0]].vapor if self.evaporation else None
        m0_old = self.m[0]
        n = len(H)
        n_it, last_upd = 0, np.inf  # Newton bookkeeping only
        for it in range(max_iter):
            try:
                T = self.T_of(H)
            except ValueError:  # iterate left the property table: reject, retry with smaller dt
                self._reject(n_it)
                return False, np.inf
            dTdH = self._by_layer(lambda mat, s: mat.dTdh(H[s]))
            G, G_back, G_front = self._conductances(T)
            m = self.m.copy()
            j = djdT = 0.0
            if vap is not None:
                j = float(vap.hkl_flux(T[0], self.alpha_evap))
                djdT = vap.hkl_flux_dT(T[0], self.alpha_evap)
                m[0] = m0_old - j * dt
                if m[0] <= 0:
                    self._reject(n_it)
                    return False, np.inf
            flux = np.zeros(n)  # net conductive inflow
            flux[:-1] -= G * (T[:-1] - T[1:])
            flux[1:] += G * (T[:-1] - T[1:])
            flux[-1] -= G_back * (T[-1] - self.back.T_coolant)
            flux[0] += G_front * (self.front.T_wall - T[0]) + q_front
            R = m * (H - H0) / dt - flux - S
            Lv = 0.0
            if vap is not None:
                Lv = float(vap.L_v(T[0]))
                R[0] += j * Lv
            # Jacobian dR/dH (conductance temperature dependence lagged)
            diag = m / dt + np.r_[G, 0.0] * dTdH + np.r_[0.0, G] * dTdH
            diag[-1] += G_back * dTdH[-1]
            diag[0] += G_front * dTdH[0]
            if vap is not None:
                diag[0] += (djdT * Lv) * dTdH[0] - (djdT * dt) * (H[0] - H0[0]) / dt * dTdH[0]
            upper = -G * dTdH[1:]
            lower = -G * dTdH[:-1]
            ab = np.zeros((3, n))
            ab[0, 1:], ab[1], ab[2, :-1] = upper, diag, lower
            dH = solve_banded((1, 1), ab, -R)
            # damp very large updates (e.g. first iteration crossing a phase change)
            dT_est = np.abs(dH * dTdH)
            lim = 200.0
            if dT_est.max() > lim:
                dH *= lim / dT_est.max()
            H = H + dH
            n_it = it + 1
            last_upd = float(np.abs(dH * dTdH).max())  # the same expression as before; stored for the counters
            if last_upd < tol_T and it > 0:
                break
        else:
            self._reject(n_it)
            return False, np.inf
        # accept
        try:
            T_new = self.T_of(H)
        except ValueError:
            self._reject(n_it)
            return False, np.inf
        dT_max = float(np.abs(T_new - self.T_of(H0)).max())
        if dT_max > dT_reject:
            self._reject(n_it)
            return False, dT_max
        G, G_back, G_front = self._conductances(T_new)
        if vap is not None:
            j = float(vap.hkl_flux(T_new[0], self.alpha_evap))
            Lv = float(vap.L_v(T_new[0]))
            self.m[0] = m0_old - j * dt
            self.mass_evaporated += j * dt
            self.ledger.vaporized_latent += j * Lv * dt
            self.ledger.vaporized_sensible += j * H0[0] * dt
        L = self.ledger
        L.incident += E_in
        L.deposited += E_dep
        L.passed_through += E_in - E_dep
        L.front_flux_in += (q_front + G_front * (self.front.T_wall - T_new[0])) * dt
        L.to_coolant += G_back * (T_new[-1] - self.back.T_coolant) * dt
        for k_if in range(len(L.through_interface)):
            i = int(np.argmax(self.layer == k_if + 1)) - 1
            L.through_interface[k_if] += G[i] * (T_new[i] - T_new[i + 1]) * dt
        self.H = H
        self.t = t1
        self.n_steps_accepted += 1
        self.newton_iters_total += n_it
        self.newton_iters_max_step = max(self.newton_iters_max_step, n_it)
        self.newton_last_update_K = last_upd
        if vap is not None and self.m[0] < self.rezone_fraction * self.m_target0[0]:
            self._rezone_front_layer()
        return True, dT_max

    # ---- conservative remap of the front layer after evaporation ------------------------
    def _rezone_front_layer(self):
        s = self.layer == 0
        m_old, H_old = self.m[s], self.H[s]
        M = m_old.sum()
        b_old = np.r_[0.0, np.cumsum(m_old)]
        E_old = np.r_[0.0, np.cumsum(m_old * H_old)]
        b_tgt = np.r_[0.0, np.cumsum(self.m_target0)]
        b_new = b_tgt[b_tgt < M * (1 - 1e-12)]
        if M - b_new[-1] < 0.5 * (b_tgt[len(b_new)] - b_new[-1]) and len(b_new) > 1:
            b_new = b_new[:-1]  # avoid a sliver cell at the back of the layer
        b_new = np.r_[b_new, M]
        E_new = np.interp(b_new, b_old, E_old)  # piecewise-constant remap of enthalpy
        m_new = np.diff(b_new)
        H_new = np.diff(E_new) / m_new
        rest = ~s
        self.m = np.r_[m_new, self.m[rest]]
        self.H = np.r_[H_new, self.H[rest]]
        self.layer = np.r_[np.zeros(len(m_new), dtype=int), self.layer[rest]]
        self._frac_cache = None
        self.n_rezones += 1

    # ---- driver -----------------------------------------------------------------------
    def run(self, t_end, dt0=1e-12, dt_max=np.inf, dT_target=5.0, growth=1.3, dt_rel=0.02,
            dt_min=1e-18, record: Optional[Callable[["ThermalSolver"], None]] = None,
            dt_limit: Optional[Callable[[float], float]] = None):
        """Adaptive stepping to t_end.  dt is limited by
          * dT_target: largest temperature change of any cell in one step;
          * dt_rel: fraction of the time elapsed since the last event (run start or a source
            switching on/off), which controls the first-order backward-Euler error for the
            sqrt(t)-type transients that follow every event;
          * the sources' dt_limit (e.g. 0.1 x pulse duration) and the optional dt_limit(t);
          * never stepping over a source event."""
        dt = dt0
        n_rej = 0
        t_event = self.t
        while self.t < t_end * (1 - 1e-14):
            lim = min([dt_max] + [s.dt_limit(self.t) for s in self.sources]
                      + ([dt_limit(self.t)] if dt_limit else []))
            lim = min(lim, max(dt_rel * (self.t - t_event), dt0))
            nxt = min([s.next_event(self.t) for s in self.sources] + [t_end])
            if nxt - self.t <= 1e-13 * max(self.t, 1e-9) and nxt < t_end:
                nxt = t_end  # event numerically already reached: do not take a zero-length step
            dt = min(dt, lim, nxt - self.t)
            hit_event = nxt < t_end and self.t + dt >= nxt * (1 - 1e-12)
            ok, dT = self.step(dt, dT_reject=2 * dT_target)
            if not ok:
                dt *= 0.25
                n_rej += 1
                if dt < dt_min:
                    raise RuntimeError(f"time step underflow at t={self.t:.3e} s")
                continue
            if record:
                record(self)
            if hit_event:
                t_event = self.t
                dt = dt0
            dt *= min(growth, max(0.5, 0.9 * dT_target / max(dT, 1e-12)))
        self.ledger.stored_final = self.stored_energy()
        return n_rej
