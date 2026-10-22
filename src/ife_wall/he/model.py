"""Reaction-diffusion model of He, vacancies and self-interstitials in W (Level 1), 1D in depth.

Unknowns per cell (concentrations in appm = 1e-6 of the W atom density; depth in um):
    He        mobile interstitial He
    V         free vacancy (= He_0 V)
    I         W self-interstitial
    HV_k      vacancy holding k He atoms, k = 1..K
(+ optional Level-2 species, see Params.level2)

Equations (cell i, widths dx_i; Appendix I of the paper):
    dHe/dt  = div(D_He grad He) + g_He - sum_{k=0}^{K-1} a_k He V_k + sum_{k=1}^{K} b_k HV_k
              + sum_k k e_k HV_k I - sink_He
    dV_k/dt = a_{k-1} He V_{k-1} - a_k He V_k - b_k V_k + b_{k+1} V_{k+1} - e_k V_k I   (k >= 1)
    dV/dt   = div(D_V grad V) + g_V - a_0 He V + b_1 V_1 - c V I - sink_V
    dI/dt   = div(D_I grad I) + g_I - c V I - sum_k e_k V_k I - sink_I
with  D_s = D0_s exp(-Em_s/kT),   a_0 = 4 pi r_c (D_He + D_V),  a_k = 4 pi r_c D_He (k>=1),
      b_k = nu exp(-(Eb_k + Em_He)/kT),   c = 4 pi r_iv (D_I + D_V),   e_k = 4 pi r_iv D_I (k>=1).
Free surface: concentration 0 for He, V, I (perfect sink; optional reduced conductance).
Back face of the armour: no flux.  g_He, g_V = g_I are the per-shot sources (sources.py).
Ledger variables are integrated with the state so that He, vacancy and interstitial counts are
conserved to round-off (verify_he_3_conservation.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

KB = 8.617333262e-5  # eV/K
N_W = 6.306e28       # W atoms / m^3
NC = N_W * 1e-6      # m^-3 per appm

L_NAMES = ["He_surface", "He_sink", "V_surface", "V_sink", "I_surface", "I_sink", "recomb", "ann_HeV",
           "He_in", "V_prod", "I_prod"]


@dataclass
class Params:
    """Every physical constant used by the model.  Values for the production runs are set in
    constants.py (with citations); tests set their own."""
    K: int
    D0_He: float            # m^2/s
    Em_He: float            # eV
    D0_V: float
    Em_V: float
    D0_I: float
    Em_I: float
    r_c: float              # He-V capture radius, m
    r_iv: float             # V-I recombination radius, m
    nu: float               # attempt frequency, 1/s
    Eb: np.ndarray          # binding energy of the k-th He to HV_{k-1}, k = 1..K (eV)
    sink_He: float = 0.0    # sink strengths, m^-2 (loss rate = D * S * C)
    sink_V: float = 0.0
    sink_I: float = 0.0
    surface_factor: float = 1.0   # 1 = perfect sink (C = 0 at x = 0); <1 reduces the surface conductance
    ann_hev: bool = True    # I + HV_k -> k He (vacancy annihilated)
    trap_full: bool = False  # (unused in Level 1) He + HV_K is blocked
    level2: Optional[dict] = None   # Level-2 (sensitivity): keys J, EbC, rCl, nutm, Etm, nuHeHe, rHe1
    extra: dict = field(default_factory=dict)


class HeModel:
    def __init__(self, edges_m, p: Params):
        self.p = p
        self.edges = np.asarray(edges_m, float)
        self.n = len(self.edges) - 1
        self.dx = np.diff(self.edges) * 1e6                    # um
        self.x = 0.5 * (self.edges[1:] + self.edges[:-1])      # m
        self.K = p.K
        self.S = 3 + p.K
        self.N0 = self.n * self.S
        self.NL = len(L_NAMES)
        self.N = self.N0 + self.NL
        self.vk = np.array([1] + [2 + k for k in range(1, p.K + 1)])   # species index of V_0..V_K
        self.Eb = np.asarray(p.Eb, float)
        assert len(self.Eb) == p.K
        # sources (appm/s per cell), set per time segment
        self.g_He = np.zeros(self.n)
        self.g_V = np.zeros(self.n)
        self.T_fn = None                                        # callable t -> T per cell
        self._build_pattern()

    # ------------------------------------------------------------------ rates
    def rates(self, T):
        p = self.p
        kT = KB * T
        D = {"He": p.D0_He * np.exp(-p.Em_He / kT) * 1e12,
             "V": p.D0_V * np.exp(-p.Em_V / kT) * 1e12,
             "I": p.D0_I * np.exp(-p.Em_I / kT) * 1e12}        # um^2/s
        DHe, DV, DI = D["He"], D["V"], D["I"]
        a = np.empty((self.n, self.K))
        a[:, 0] = 4 * np.pi * p.r_c * (DHe + DV) * 1e-12 * NC
        if self.K > 1:
            a[:, 1:] = (4 * np.pi * p.r_c * DHe * 1e-12 * NC)[:, None]
        b = p.nu * np.exp(-(self.Eb[None, :] + p.Em_He) / kT[:, None])
        c = 4 * np.pi * p.r_iv * (DI + DV) * 1e-12 * NC
        e = (4 * np.pi * p.r_iv * DI * 1e-12 * NC) * (1.0 if p.ann_hev else 0.0)
        return D, a, b, c, e

    def _conductances(self, D):
        dx = self.dx
        Gc = 1.0 / (dx[:-1] / (2 * D[:-1]) + dx[1:] / (2 * D[1:]))
        Gs = p_surf = self.p.surface_factor * 2 * D[0] / dx[0]       # um/s: D/(dx0/2)
        return Gc, Gs

    # ------------------------------------------------------------------ sparse pattern
    def _build_pattern(self):
        n, S = self.n, self.S
        ci = np.arange(n)
        # local blocks: every (r, c) pair in each cell
        rr, cc = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        self._loc_rows = (ci[:, None, None] * S + rr[None]).ravel()
        self._loc_cols = (ci[:, None, None] * S + cc[None]).ravel()
        # diffusion off-diagonals for mobile species 0,1,2
        mob = np.array([0, 1, 2])
        i = np.arange(n - 1)
        self._lo_rows = ((i + 1)[:, None] * S + mob[None]).ravel()
        self._lo_cols = (i[:, None] * S + mob[None]).ravel()
        self._up_rows = (i[:, None] * S + mob[None]).ravel()
        self._up_cols = ((i + 1)[:, None] * S + mob[None]).ravel()
        # ledger rows: dense rows over relevant species
        L = self.N0
        rows, cols = [], []
        # surface / sink rows
        self._led_entries = {}

    # ------------------------------------------------------------------ right-hand side
    def _unpack(self, y):
        u = y[: self.N0].reshape(self.n, self.S)
        return u

    def rhs(self, t, y):
        n, S, K = self.n, self.S, self.K
        p = self.p
        T = self.T_fn(t)
        D, a, b, c, e = self.rates(T)
        u = y[: self.N0].reshape(n, S)
        He, V, I = u[:, 0], u[:, 1], u[:, 2]
        HV = u[:, 3:]
        Vk = u[:, self.vk]                                   # (n, K+1)
        f = np.zeros((n, S))
        # diffusion
        for s_idx, key in ((0, "He"), (1, "V"), (2, "I")):
            Gc, Gs = self._conductances(D[key])
            C = u[:, s_idx]
            flux = Gc * (C[:-1] - C[1:])                    # appm um/s through face i+1/2
            df = np.zeros(n)
            df[:-1] -= flux
            df[1:] += flux
            df[0] -= Gs * C[0]
            f[:, s_idx] += df / self.dx
        # trapping, detrapping
        r_t = a * He[:, None] * Vk[:, :K]                    # (n,K)  He + V_k -> V_{k+1}
        r_d = b * HV                                         # (n,K)  V_k -> V_{k-1} + He
        rec = c * V * I
        ann = e[:, None] * HV * I[:, None]
        f[:, 0] += -r_t.sum(1) + r_d.sum(1) + (ann * np.arange(1, K + 1)[None]).sum(1) + self.g_He
        dVk = np.zeros((n, K + 1))
        dVk[:, :K] -= r_t
        dVk[:, 1:] += r_t
        dVk[:, 1:] -= r_d
        dVk[:, :K] += r_d
        dVk[:, 1:] -= ann
        dVk[:, 0] -= rec
        f[:, self.vk] += dVk
        f[:, 2] += -rec - ann.sum(1)
        f[:, 1] += self.g_V
        f[:, 2] += self.g_V                                  # Frenkel pairs: g_I = g_V
        # bulk sinks
        sH, sV, sI = D["He"] * p.sink_He * 1e-12, D["V"] * p.sink_V * 1e-12, D["I"] * p.sink_I * 1e-12
        f[:, 0] -= sH * He
        f[:, 1] -= sV * V
        f[:, 2] -= sI * I
        out = np.zeros(self.N)
        out[: self.N0] = f.ravel()
        # ledger (appm um)
        N0, dx = self.N0, self.dx
        _, GsHe = self._conductances(D["He"]); _, GsV = self._conductances(D["V"]); _, GsI = self._conductances(D["I"])
        out[N0 + 0] = GsHe * He[0]
        out[N0 + 1] = (dx * sH * He).sum()
        out[N0 + 2] = GsV * V[0]
        out[N0 + 3] = (dx * sV * V).sum()
        out[N0 + 4] = GsI * I[0]
        out[N0 + 5] = (dx * sI * I).sum()
        out[N0 + 6] = (dx * rec).sum()
        out[N0 + 7] = (dx[:, None] * ann).sum()
        out[N0 + 8] = (dx * self.g_He).sum()
        out[N0 + 9] = (dx * self.g_V).sum()
        out[N0 + 10] = (dx * self.g_V).sum()
        return out

    # ------------------------------------------------------------------ Jacobian
    def jac(self, t, y):
        n, S, K = self.n, self.S, self.K
        p = self.p
        T = self.T_fn(t)
        D, a, b, c, e = self.rates(T)
        u = y[: self.N0].reshape(n, S)
        He, V, I = u[:, 0], u[:, 1], u[:, 2]
        HV = u[:, 3:]
        Vk = u[:, self.vk]
        J = np.zeros((n, S, S))
        vk = self.vk
        # trapping He + V_k -> V_{k+1}
        for k in range(K):
            dHe_, dV_ = a[:, k] * Vk[:, k], a[:, k] * He
            J[:, 0, 0] -= dHe_; J[:, 0, vk[k]] -= dV_
            J[:, vk[k], 0] -= dHe_; J[:, vk[k], vk[k]] -= dV_
            J[:, vk[k + 1], 0] += dHe_; J[:, vk[k + 1], vk[k]] += dV_
        # detrapping V_k -> V_{k-1} + He
        for k in range(1, K + 1):
            d = b[:, k - 1]
            J[:, vk[k], vk[k]] -= d
            J[:, vk[k - 1], vk[k]] += d
            J[:, 0, vk[k]] += d
        # recombination V + I
        J[:, 1, 1] -= c * I; J[:, 1, 2] -= c * V
        J[:, 2, 1] -= c * I; J[:, 2, 2] -= c * V
        # I + V_k -> k He
        if p.ann_hev:
            for k in range(1, K + 1):
                dHV, dI = e * I, e * HV[:, k - 1]
                J[:, vk[k], vk[k]] -= dHV; J[:, vk[k], 2] -= dI
                J[:, 2, vk[k]] -= dHV; J[:, 2, 2] -= dI
                J[:, 0, vk[k]] += k * dHV; J[:, 0, 2] += k * dI
        sH, sV, sI = D["He"] * p.sink_He * 1e-12, D["V"] * p.sink_V * 1e-12, D["I"] * p.sink_I * 1e-12
        J[:, 0, 0] -= sH; J[:, 1, 1] -= sV; J[:, 2, 2] -= sI
        # diffusion: diagonal + off-diagonals for mobile species
        lo = np.zeros((n - 1, 3)); up = np.zeros((n - 1, 3))
        Gs_all = []
        for m, key in enumerate(("He", "V", "I")):
            Gc, Gs = self._conductances(D[key])
            diag = np.zeros(n)
            diag[:-1] -= Gc / self.dx[:-1]
            diag[1:] -= Gc / self.dx[1:]
            diag[0] -= Gs / self.dx[0]
            J[:, m, m] += diag
            lo[:, m] = Gc / self.dx[1:]       # d f_{i+1} / d C_i
            up[:, m] = Gc / self.dx[:-1]      # d f_i / d C_{i+1}
            Gs_all.append(Gs)
        data = np.concatenate([J.ravel(), lo.ravel(), up.ravel()])
        rows = np.concatenate([self._loc_rows, self._lo_rows, self._up_rows])
        cols = np.concatenate([self._loc_cols, self._lo_cols, self._up_cols])
        # ledger rows
        N0, dx = self.N0, self.dx
        ci = np.arange(n) * S
        lr, lc, ld = [], [], []

        def add(row, col, val):
            lr.append(np.full(len(np.atleast_1d(col)), N0 + row)); lc.append(np.atleast_1d(col)); ld.append(np.atleast_1d(val))
        add(0, 0, Gs_all[0]); add(1, ci + 0, dx * sH)
        add(2, 1, Gs_all[1]); add(3, ci + 1, dx * sV)
        add(4, 2, Gs_all[2]); add(5, ci + 2, dx * sI)
        add(6, ci + 1, dx * c * I); add(6, ci + 2, dx * c * V)
        if p.ann_hev:
            for k in range(1, K + 1):
                add(7, ci + vk[k], dx * e * I); add(7, ci + 2, dx * e * HV[:, k - 1])
        rows = np.concatenate([rows] + lr)
        cols = np.concatenate([cols] + lc)
        data = np.concatenate([data] + ld)
        return sp.csc_matrix((data, (rows, cols)), shape=(self.N, self.N))

    # ------------------------------------------------------------------ helpers
    def y0(self):
        return np.zeros(self.N)

    def totals(self, y):
        """dict of areal quantities in He/m^2 (He), vacancies/m^2, etc. from state y."""
        n, S, K = self.n, self.S, self.K
        u = y[: self.N0].reshape(n, S)
        w = self.dx * 1e-6 * NC                       # appm -> per m^2 : appm * dx[um]*1e-6 m * Nc
        He_mobile = float((u[:, 0] * w).sum())
        He_trapped = float(((u[:, 3:3 + K] * np.arange(1, K + 1)[None]).sum(1) * w).sum())
        V_free = float((u[:, 1] * w).sum())
        V_He = float((u[:, 3:3 + K].sum(1) * w).sum())
        I_tot = float((u[:, 2] * w).sum())
        led = {nm: float(y[self.N0 + i]) * 1e-6 * NC for i, nm in enumerate(L_NAMES)}  # appm um -> per m^2
        return dict(He_mobile=He_mobile, He_trapped=He_trapped, V_free=V_free, V_He=V_He, I=I_tot, **led)

    def conservation(self, y):
        t = self.totals(y)
        He_err = t["He_mobile"] + t["He_trapped"] + t["He_surface"] + t["He_sink"] - t["He_in"]
        V_err = t["V_free"] + t["V_He"] + t["V_surface"] + t["V_sink"] + t["recomb"] + t["ann_HeV"] - t["V_prod"]
        I_err = t["I"] + t["I_surface"] + t["I_sink"] + t["recomb"] + t["ann_HeV"] - t["I_prod"]
        return dict(He=(He_err, t["He_in"]), V=(V_err, t["V_prod"]), I=(I_err, t["I_prod"]))

    # ------------------------------------------------------------------ integration
    def integrate(self, y, t0, t1, T_fn=None, method="Radau", rtol=1e-6, atol=1e-10, atol_led=1e-3, **kw):
        if T_fn is not None:
            self.T_fn = T_fn
        at = np.full(self.N, atol)
        at[self.N0:] = atol_led
        sol = solve_ivp(self.rhs, (t0, t1), y, method=method, jac=self.jac, rtol=rtol, atol=at, **kw)
        if not sol.success:
            raise RuntimeError(sol.message)
        self.last_nfev, self.last_njev, self.last_nlu = sol.nfev, sol.njev, sol.nlu
        return sol.y[:, -1], sol


class CycleTemperature:
    """T(cell, t) for one shot cycle from a recorded thermal history (scripts/record_he_thermal.py)."""

    def __init__(self, npz_path, x_cells, T_offset=0.0):
        z = np.load(npz_path, allow_pickle=False)
        self.t = z["t"]
        zz = z["z"]
        Tm = z["T"].astype(float)
        self.T = np.array([np.interp(x_cells, zz, Tm[k]) for k in range(len(self.t))]) + T_offset
        self.P = float(self.t[-1])
        self.meta = z["meta"].item() if "meta" in z else ""

    def __call__(self, tau):
        tau = min(max(tau, self.t[0]), self.t[-1])
        k = int(np.searchsorted(self.t, tau)) - 1
        k = min(max(k, 0), len(self.t) - 2)
        f = (tau - self.t[k]) / (self.t[k + 1] - self.t[k])
        return self.T[k] * (1 - f) + self.T[k + 1] * f
