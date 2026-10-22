"""Compiled (numba) model + LSODA driver.  Same equations as model.HeModel (which is the NumPy
reference used for cross-checks); see kernels.py for the state layout."""
from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp

from . import kernels as kn
from .model import Params, CycleTemperature, N_W, NC


class FastHe:
    def __init__(self, edges_m, p: Params):
        self.p = p
        self.edges = np.asarray(edges_m, float)
        self.n = len(self.edges) - 1
        self.dx = np.diff(self.edges) * 1e6
        self.x = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.K = p.K
        l2 = p.level2 or {}
        self.J = int(l2.get("J", 0))
        self.S = 3 + self.K + self.J
        self.N = kn.H + self.n * self.S
        self.mu = max(self.S, kn.H + 2)   # same-species coupling to the next cell is S away
        self.ab = np.zeros((2 * self.mu + 1, self.N))
        prm = np.zeros(kn.NPAR)
        prm[kn.P_D0He], prm[kn.P_EmHe] = p.D0_He, p.Em_He
        prm[kn.P_D0V], prm[kn.P_EmV] = p.D0_V, p.Em_V
        prm[kn.P_D0I], prm[kn.P_EmI] = p.D0_I, p.Em_I
        prm[kn.P_rc], prm[kn.P_riv], prm[kn.P_nu] = p.r_c, p.r_iv, p.nu
        prm[kn.P_SHe], prm[kn.P_SV], prm[kn.P_SI] = p.sink_He, p.sink_V, p.sink_I
        prm[kn.P_surf] = p.surface_factor
        prm[kn.P_ann] = 1.0 if p.ann_hev else 0.0
        prm[kn.P_NC] = NC
        prm[kn.P_L2] = 1.0 if self.J else 0.0
        prm[kn.P_nuHeHe] = l2.get("nuHeHe", 0.0)
        prm[kn.P_rHe1] = l2.get("rHe1", 0.0)
        self.prm = prm
        self.Eb = np.asarray(p.Eb, float)
        J = max(self.J, 1)
        self.EbC = np.asarray(l2.get("EbC", np.zeros(J)), float)
        self.rCl = np.asarray(l2.get("rCl", np.zeros(J)), float)
        self.nutm = np.asarray(l2.get("nutm", np.zeros(J)), float)
        self.Etm = np.asarray(l2.get("Etm", np.zeros(J)), float)
        self.tT = np.array([0.0, 1e30]); self.Tc = np.full((2, self.n), 900.0)
        self.tN = np.array([0.0, 1.0]); self.GHe = np.zeros((2, self.n)); self.GV = np.zeros((2, self.n))
        self.bgHe = np.zeros(self.n); self.bgV = np.zeros(self.n)

    # ---- callbacks for the integrator
    def rhs(self, t, y):
        out = np.empty(self.N)
        kn.rhs_nb(t, y, self.prm, self.Eb, self.dx, self.tT, self.Tc, self.tN, self.GHe, self.GV, self.bgHe, self.bgV,
                  self.K, self.J, self.EbC, self.rCl, self.nutm, self.Etm, out)
        return out

    def jac(self, t, y):
        kn.jac_nb(t, y, self.prm, self.Eb, self.dx, self.tT, self.Tc, self.K, self.J, self.EbC, self.rCl,
                  self.nutm, self.Etm, self.mu, self.ab)
        return self.ab

    def dense_jac(self, t, y):
        ab = self.jac(t, y)
        N = self.N
        Jd = np.zeros((N, N))
        for j in range(N):
            for r in range(max(0, j - self.mu), min(N, j + self.mu + 1)):
                Jd[r, j] = ab[self.mu + r - j, j]
        return Jd

    def set_cycle(self, tT, Tc):
        self.tT, self.Tc = np.asarray(tT, float), np.asarray(Tc, float)

    def set_sources(self, te, G_He, G_V):
        """te: slice edges (s); G_He, G_V: slice-average rates (appm/s) per cell, shape (ns, n).
        Builds a continuous piecewise-linear rate through the slice mid-points (zero at the outer
        edges) and rescales every cell so that the time integral equals the slice-integral exactly."""
        te = np.asarray(te, float)
        mids = 0.5 * (te[1:] + te[:-1])
        dts = np.diff(te)
        tN = np.r_[te[0], mids, te[-1]]
        def nodes(G):
            Gn = np.vstack([np.zeros((1, self.n)), G, np.zeros((1, self.n))])
            tr = (0.5 * (Gn[1:] + Gn[:-1]) * np.diff(tN)[:, None]).sum(0)
            ex = (G * dts[:, None]).sum(0)
            sc = np.divide(ex, tr, out=np.ones_like(ex), where=tr > 0)
            return Gn * sc[None]
        self.tN, self.GHe, self.GV = tN, nodes(G_He), nodes(G_V)

    # ---- integration
    def integrate(self, y, t0, t1, rtol=1e-6, atol=1e-10, atol_led=1e-9, method="LSODA", **kw):
        at = np.full(self.N, atol)
        at[:kn.H] = atol_led
        kw.setdefault("first_step", (t1 - t0) * 1e-6)
        sol = solve_ivp(self.rhs, (t0, t1), y, method=method, jac=self.jac, lband=self.mu, uband=self.mu,
                        rtol=rtol, atol=at, **kw)
        if not sol.success:
            raise RuntimeError(sol.message)
        self.last = (sol.nfev, sol.njev, sol.nlu)
        return sol.y[:, -1]

    # ---- diagnostics
    def totals(self, y):
        n, S, K = self.n, self.S, self.K
        u = y[kn.H:].reshape(n, S)
        w = self.dx * 1e-6 * NC
        ks = np.arange(1, K + 1)[None]
        He_mobile = float((u[:, 0] * w).sum())
        He_trapped = float(((u[:, 3:3 + K] * ks).sum(1) * w).sum())
        He_cl = 0.0
        if self.J:
            sz = np.arange(2, self.J + 2)[None]
            He_cl = float(((u[:, 3 + K:3 + K + self.J] * sz).sum(1) * w).sum())
        return dict(He_mobile=He_mobile, He_trapped=He_trapped, He_clusters=He_cl,
                    V_free=float((u[:, 1] * w).sum()), V_He=float((u[:, 3:3 + K].sum(1) * w).sum()),
                    I=float((u[:, 2] * w).sum()),
                    He_surface=float(y[0]) * 1e-6 * NC, V_surface=float(y[1]) * 1e-6 * NC, I_surface=float(y[2]) * 1e-6 * NC,
                    He_in=float(y[3]) * 1e-6 * NC, V_prod=float(y[4]) * 1e-6 * NC, I_prod=float(y[5]) * 1e-6 * NC)

    def profiles(self, y):
        n, S, K = self.n, self.S, self.K
        u = y[kn.H:].reshape(n, S)
        d = dict(x=self.x, He=u[:, 0], V=u[:, 1], I=u[:, 2], HV=u[:, 3:3 + K],
                 He_trapped=(u[:, 3:3 + K] * np.arange(1, K + 1)[None]).sum(1))
        if self.J:
            d["He_cl"] = (u[:, 3 + K:3 + K + self.J] * np.arange(2, self.J + 2)[None]).sum(1)
            d["C"] = u[:, 3 + K:3 + K + self.J]
        return d

    def pack_from_numpy(self, y_np, N0_np):
        """Convert a state of the NumPy HeModel (ledger at the end) to this layout (L1 only)."""
        z = np.zeros(self.N)
        z[kn.H:] = y_np[:N0_np]
        return z
