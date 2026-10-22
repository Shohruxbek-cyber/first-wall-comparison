"""Shot diagnostics for the damaged-conductivity study (2026-09-29).

A `ShotRecorder` is updated after every accepted solver step (record callback) and gives:
  * maximum-temperature depth profile on a fixed grid (1 nm .. 200 um, log-spaced), with the
    depths where the peak temperature crosses 1300 K and 1800 K (W recrystallization range,
    GA2020 Sec. III.A);
  * helium-layer thermal history at 0.5, 1, 2, 4, 5 um: time above 1000 K (He fuzz threshold)
    and above 900 K (H outgassing) -- the thresholds cited in the article, Section 4.4 -- plus peak
    and time-averaged temperature over the cycle;
  * depth table: time-averaged, pre-shot and peak temperature at the surface, 5, 50,
    500 um and the W/steel interface;
  * maximum near-surface temperature gradient (K/um, top 20 um) and a fully-constrained
    elastic stress proxy  sigma = E alpha dT / (1 - nu)  with dT = peak surface T - pre-shot
    surface T and E, alpha, nu from the ITER-MPH polynomials printed in Blanchard & Martin
    2005, App. A (T in deg C), evaluated at the pre-shot temperature.  "Elastic upper-bound
    proxy, no plasticity, no fracture prediction."

Depths are measured from the current surface.  Interpolation onto the fixed grid makes the
profile independent of rezoning of the front cells.

Alloy study (2026-10-13): an optional `RxSpec` adds recrystallization bookkeeping -- cumulative
time above given thresholds and an Arrhenius-equivalent time -- at the surface and on the depth
grid.  Without it (the default) nothing in this module changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenarios import interface_index

R_GAS = 8.314462618

HE_DEPTHS_UM = (0.5, 1.0, 2.0, 4.0, 5.0)
DEPTH_TABLE_DEPTHS = {"surface": 0.0, "5um": 5e-6, "50um": 50e-6, "500um": 500e-6}
GRID = np.geomspace(1e-9, 200e-6, 240)
T_FUZZ, T_H = 1000.0, 900.0


@dataclass(frozen=True)
class RxSpec:
    """Recrystallization bookkeeping (diagnostic only; it never changes the temperatures).
    thresholds: temperatures (K) whose cumulative time-above is accumulated at the surface and
      at every point of GRID (temperature linear in time between accepted steps, as _time_above).
    Q: activation energies (J/mol) for the Arrhenius-equivalent time at T_ref,
      t_eq = integral exp[-Q/R (1/T - 1/T_ref)] dt  (trapezoid in the integrand),
      i.e. the time at T_ref giving the same thermally activated progress.  This assumes an
      isokinetic (additive) process with a single activation energy and no incubation; it is the
      JMAK/Arrhenius form of Shah et al. 2021 (FED 173, 112827) reduced to its rate factor.
    T_ref: the reference temperature (K), normally the sourced 1 h recrystallization onset."""
    thresholds: tuple = ()
    Q: tuple = ()
    T_ref: float | None = None


def _time_above_vec(dt, T0, T1, thr):
    """Vectorized _time_above for arrays T0, T1 over one step of length dt."""
    both = (T0 >= thr) & (T1 >= thr)
    none = (T0 < thr) & (T1 < thr)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (thr - T0) / (T1 - T0)
    part = np.where(T1 > T0, 1.0 - f, f)
    return dt * np.where(both, 1.0, np.where(none, 0.0, part))


def w_elastic(T_K):
    """ITER-MPH W polynomials as printed in BM2005 Appendix A (T in deg C).
    Returns E [Pa], nu [-], alpha [1/K]."""
    C = float(T_K) - 273.15
    E = (398.0 - 0.00231 * C - 2.72e-5 * C ** 2) * 1e9
    nu = 0.279 + 1.09e-5 * C
    alpha = (3.922 + 5.835e-5 * C + 5.705e-11 * C ** 2 - 2.046e-14 * C ** 3) * 1e-6
    return E, nu, alpha


def _time_above(t0, t1, T0, T1, thr):
    """Time within [t0, t1] with T above thr, T linear in between."""
    if T0 >= thr and T1 >= thr:
        return t1 - t0
    if T0 < thr and T1 < thr:
        return 0.0
    f = (thr - T0) / (T1 - T0)  # crossing fraction
    return (t1 - t0) * ((1 - f) if T1 > T0 else f)


class ShotRecorder:
    def __init__(self, solver, t_start, cycle, rx: RxSpec | None = None):
        self.t_start, self.cycle = t_start, cycle
        self.rx = rx
        if rx is not None:
            # column 0 = surface, columns 1.. = GRID
            self.rx_above = np.zeros((len(rx.thresholds), GRID.size + 1))
            self.rx_teq = np.zeros((len(rx.Q), GRID.size + 1))
            self.rx_prev = None
        self.maxT = np.full(GRID.size, -np.inf)
        self.he_above_fuzz = np.zeros(len(HE_DEPTHS_UM))
        self.he_above_H = np.zeros(len(HE_DEPTHS_UM))
        self.he_peak = np.full(len(HE_DEPTHS_UM), -np.inf)
        self.he_int = np.zeros(len(HE_DEPTHS_UM))
        self.b_int = {k: 0.0 for k in list(DEPTH_TABLE_DEPTHS) + ["interface"]}
        self.b_peak = {k: -np.inf for k in self.b_int}
        self.max_grad = 0.0
        self.t_prev = None
        self.he_prev = None
        self.b_prev = None
        s0 = self._sample(solver)
        self.b_pre = dict(s0[1])  # pre-shot temperatures
        self.Ts_pre = s0[1]["surface"]
        self._accept(solver.t, *s0)

    @staticmethod
    def _profile(sv):
        T = sv.T
        x = sv.x_centers()
        return np.r_[0.0, x], np.r_[sv.surface_temperature(T), T], T, x

    def _sample(self, sv):
        xs, Ts, T, x = self._profile(sv)
        grid_T = np.interp(GRID, xs, Ts)
        he = np.interp(np.array(HE_DEPTHS_UM) * 1e-6, xs, Ts)
        b = {k: float(np.interp(d, xs, Ts)) for k, d in DEPTH_TABLE_DEPTHS.items()}
        b["interface"] = float(T[interface_index(sv)])
        near = x < 20e-6
        g = np.abs(np.diff(T[near]) / np.diff(x[near])) * 1e-6 if near.sum() > 1 else np.array([0.0])
        return grid_T, b, he, float(g.max())

    def _accept(self, t, grid_T, b, he, g):
        self.maxT = np.maximum(self.maxT, grid_T)
        self.he_peak = np.maximum(self.he_peak, he)
        self.max_grad = max(self.max_grad, g)
        for k, v in b.items():
            self.b_peak[k] = max(self.b_peak[k], v)
        if self.t_prev is not None:
            dt = t - self.t_prev
            for i in range(len(he)):
                self.he_above_fuzz[i] += _time_above(self.t_prev, t, self.he_prev[i], he[i], T_FUZZ)
                self.he_above_H[i] += _time_above(self.t_prev, t, self.he_prev[i], he[i], T_H)
                self.he_int[i] += 0.5 * (self.he_prev[i] + he[i]) * dt
            for k in b:
                self.b_int[k] += 0.5 * (self.b_prev[k] + b[k]) * dt
            if self.rx is not None:
                prof = np.r_[b["surface"], grid_T]
                for i, thr in enumerate(self.rx.thresholds):
                    self.rx_above[i] += _time_above_vec(dt, self.rx_prev, prof, thr)
                for j, Q in enumerate(self.rx.Q):
                    a = Q / R_GAS
                    r0 = np.exp(-a * (1.0 / self.rx_prev - 1.0 / self.rx.T_ref))
                    r1 = np.exp(-a * (1.0 / prof - 1.0 / self.rx.T_ref))
                    self.rx_teq[j] += 0.5 * (r0 + r1) * dt
        self.t_prev, self.he_prev, self.b_prev = t, he, b
        if self.rx is not None:
            self.rx_prev = np.r_[b["surface"], grid_T]

    def update(self, sv):
        if sv.t >= self.t_start:
            self._accept(sv.t, *self._sample(sv))

    def results(self):
        span = (self.t_prev - self.t_start) if self.t_prev else 0.0
        cyc_ok = self.cycle is not None and span >= 0.999 * self.cycle
        avg = (lambda v: v / span) if cyc_ok else (lambda v: None)
        Ts_peak = self.b_peak["surface"]
        E, nu, alpha = w_elastic(self.Ts_pre)
        sigma = E * alpha * (Ts_peak - self.Ts_pre) / (1 - nu)

        def crossing(thr):
            above = np.nonzero(self.maxT >= thr)[0]
            return float(GRID[above[-1]] * 1e6) if len(above) else 0.0

        out = {
            "d_1300_um": crossing(1300.0), "d_1800_um": crossing(1800.0),
            "He_layer": [{"depth_um": d, "time_above_1000K_us": fa * 1e6, "time_above_900K_us": fh * 1e6,
                          "peak_K": float(p), "time_avg_K": avg(i)}
                         for d, fa, fh, p, i in zip(HE_DEPTHS_UM, self.he_above_fuzz, self.he_above_H, self.he_peak, self.he_int)],
            "depth_table": {k: {"pre_shot_K": self.b_pre[k], "peak_K": self.b_peak[k], "time_avg_K": avg(self.b_int[k])}
                         for k in self.b_int},
            "cycle_s": self.cycle if cyc_ok else None,
            "max_near_surface_gradient_K_per_um": self.max_grad,
            "elastic_stress_proxy_MPa": sigma / 1e6,
            "elastic_stress_proxy_note": "fully constrained E*alpha*dT/(1-nu), dT = peak - pre-shot surface T; "
                                         "E, alpha, nu: ITER-MPH polynomials (BM2005 App. A) at the pre-shot T; "
                                         "elastic upper-bound proxy, no plasticity, no fracture prediction",
        }
        if self.rx is not None:
            out["recrystallization"] = self._rx_results()
        return out

    def _rx_results(self):
        rx = self.rx
        he_d = np.array(HE_DEPTHS_UM) * 1e-6

        def at_he(row):  # interpolate a grid row (columns 1..) at the He depths
            return np.interp(he_d, GRID, row[1:])

        def deepest(row):
            above = np.nonzero(row[1:] > 0.0)[0]
            return float(GRID[above[-1]] * 1e6) if len(above) else 0.0

        return {
            "T_ref_K": rx.T_ref, "Q_J_per_mol": list(rx.Q), "thresholds_K": list(rx.thresholds),
            "surface": {"time_above_s": {f"{thr:g}": float(self.rx_above[i, 0]) for i, thr in enumerate(rx.thresholds)},
                        "t_eq_s": {f"{Q:g}": float(self.rx_teq[j, 0]) for j, Q in enumerate(rx.Q)}},
            "He_layer": [{"depth_um": d,
                          "time_above_s": {f"{thr:g}": float(at_he(self.rx_above[i])[n]) for i, thr in enumerate(rx.thresholds)},
                          "t_eq_s": {f"{Q:g}": float(at_he(self.rx_teq[j])[n]) for j, Q in enumerate(rx.Q)}}
                         for n, d in enumerate(HE_DEPTHS_UM)],
            "depth_any_time_above_um": {f"{thr:g}": deepest(self.rx_above[i]) for i, thr in enumerate(rx.thresholds)},
            "note": "diagnostic only: time above threshold and Arrhenius-equivalent time at T_ref "
                    "(isokinetic, single Q, no incubation); accumulated over this recorder's span",
        }

    def profile_rows(self):
        return np.c_[GRID * 1e6, self.maxT]

    def rx_profile_rows(self):
        """depth_um (0 = surface), time above each threshold (s), t_eq for each Q (s)."""
        if self.rx is None:
            return None
        return np.c_[np.r_[0.0, GRID * 1e6], self.rx_above.T, self.rx_teq.T]

    def rx_profile_header(self):
        return ",".join(["depth_um"] + [f"time_above_{thr:g}K_s" for thr in self.rx.thresholds]
                        + [f"t_eq_Q{Q / 1e3:g}kJ_s" for Q in self.rx.Q])
