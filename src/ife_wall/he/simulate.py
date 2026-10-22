"""Shot-by-shot driver: sources + thermal cycle + HeModel."""
from __future__ import annotations

import time

import numpy as np

from .model import HeModel, Params, CycleTemperature, NC, N_W
from .sources import ShotSources


class Simulation:
    """n identical thermal cycles of period P, each with the per-shot sources.

    cycle: CycleTemperature (recorded T(x,t) of one shot cycle) or a callable tau -> T per cell.
    xi:    vacancy survival fraction applied to the NRT dpa (vacancy / SIA production = xi * dpa * N_W).
    """

    def __init__(self, scen_name, params: Params, cycle_path=None, edges=None, xi=1.0, dpa_kind="nrt",
                 Ed=None, n_slices=40, n_sub=6, P=None, T_const=None, neutron=None, sources=None, T_offset=0.0,
                 he_scale=1.0):
        from .grid import make_grid
        self.edges = make_grid() if edges is None else edges
        self.model = HeModel(self.edges, params)
        self.src = sources or ShotSources(scen_name, self.edges, n_sub=n_sub, n_slices=n_slices,
                                          dpa_kind=dpa_kind, Ed=Ed)
        if T_const is not None:
            self.cycle = None
            self.model.T_fn = lambda t: np.full(self.model.n, float(T_const))
            self.P = P
        else:
            self.cycle = CycleTemperature(cycle_path, self.model.x, T_offset)
            self.model.T_fn = self.cycle
            self.P = self.cycle.P if P is None else P
        self.xi = xi
        self.he_scale = he_scale
        self.neutron = neutron or {}
        s = self.src
        self.te = np.r_[0.0, s.t_edges, self.P]                  # segment edges inside one cycle
        n = self.model.n
        gHe = np.zeros((len(self.te) - 1, n))
        gV = np.zeros((len(self.te) - 1, n))
        gHe[1:-1] = s.G_He / N_W * 1e6 * he_scale                # He/m^3/s -> appm/s
        gV[1:-1] = xi * s.G_dpa * 1e6                            # dpa/s -> appm/s
        # optional neutron contribution (uniform in depth and time over the cycle)
        gV += xi * self.neutron.get("dpa_per_shot", 0.0) * 1e6 / self.P
        gHe += self.neutron.get("He_appm_per_shot", 0.0) / self.P
        self.gHe, self.gV = gHe, gV
        self.y = self.model.y0()
        self.shot_no = 0
        self.history = []

    def shot(self, method="Radau", rtol=1e-6, atol=1e-10):
        m = self.model
        y = self.y
        nfev = 0
        for j in range(len(self.te) - 1):
            m.g_He, m.g_V = self.gHe[j], self.gV[j]
            y, sol = m.integrate(y, self.te[j], self.te[j + 1], method=method, rtol=rtol, atol=atol)
            nfev += sol.nfev
        self.y = y
        self.shot_no += 1
        return nfev

    def run(self, n_shots, record=None, verbose=0, **kw):
        """record: set of shot numbers at which to store the totals (always stores the last shot)."""
        t0 = time.time()
        out = []
        for k in range(n_shots):
            self.shot(**kw)
            if record is None or self.shot_no in record or k == n_shots - 1:
                out.append(self.snapshot())
            if verbose and self.shot_no % verbose == 0:
                tt = self.model.totals(self.y)
                print(f"shot {self.shot_no}: He retained {tt['He_mobile'] + tt['He_trapped']:.3e}/m2 "
                      f"({time.time() - t0:.0f}s)", flush=True)
        return out

    def snapshot(self):
        t = self.model.totals(self.y)
        t["shot"] = self.shot_no
        t["He_retained"] = t["He_mobile"] + t["He_trapped"]
        return t

    def profiles(self):
        """depth profiles (appm) of mobile He, trapped He, free V, V-He total, I"""
        m = self.model
        u = self.y[: m.N0].reshape(m.n, m.S)
        K = m.K
        return dict(x=m.x, He=u[:, 0], V=u[:, 1], I=u[:, 2], HV=u[:, 3:3 + K],
                    He_trapped=(u[:, 3:3 + K] * np.arange(1, K + 1)[None]).sum(1))


# ==========================================================================================
# Compiled driver (production)
# ==========================================================================================
from .fast import FastHe  # noqa: E402


class FastSimulation:
    """n identical thermal cycles with the per-shot sources, integrated with FastHe (numba + LSODA).

    Each shot is integrated in three pieces: [0, t_first] (x-ray heating only), the ion arrival
    window, and the cooling tail up to the next shot.  A cycle record from record_he_thermal.py
    gives T(x, t); `T_offset` shifts it (sensitivity), `T_const` replaces it (tests).
    """

    def __init__(self, scen_name, params, cycle_path=None, edges=None, xi=1.0, dpa_kind="nrt", Ed=None,
                 T_offset=0.0, T_const=None, P=None, neutron=True, sources=None, n_slices=40, n_sub=6,
                 he_scale=1.0, rtol=1e-3, atol=1e-9, neutron_scale=1.0, path_element="projected"):
        from .grid import make_grid
        from . import constants as C
        self.edges = make_grid() if edges is None else edges
        self.fm = FastHe(self.edges, params)
        n = self.fm.n
        self.src = sources or ShotSources(scen_name, self.edges, n_sub=n_sub, n_slices=n_slices,
                                          dpa_kind=dpa_kind, Ed=Ed, path_element=path_element)
        if T_const is not None:
            self.fm.set_cycle([0.0, 1e30], np.full((2, n), float(T_const)))
            self.P = P
        else:
            cyc = CycleTemperature(cycle_path, self.fm.x, T_offset)
            self.fm.set_cycle(cyc.t, cyc.T)
            self.P = cyc.P if P is None else P
        s = self.src
        self.te = s.t_edges
        self.xi, self.rtol, self.atol = xi, rtol, atol
        self.fm.set_sources(self.te, s.G_He / N_W * 1e6 * he_scale, xi * s.G_dpa * 1e6)
        if neutron:
            dpa_n = C.NEUTRON_DPA_PER_SHOT * neutron_scale
            self.fm.bgV[:] = xi * dpa_n * 1e6 / self.P
            self.fm.bgHe[:] = dpa_n * C.NEUTRON_HE_PER_DPA / self.P     # appm/s
        self.y = np.zeros(self.fm.N)
        self.shot_no = 0

    def shot(self):
        fm, y = self.fm, self.y
        te0, te1 = self.te[0], self.te[-1]
        for a, b in ((0.0, te0), (te0, te1), (te1, self.P)):
            y = fm.integrate(y, a, b, rtol=self.rtol, atol=self.atol)
        self.y = y
        self.shot_no += 1

    def snapshot(self):
        t = self.fm.totals(self.y)
        t["shot"] = self.shot_no
        t["He_retained"] = t["He_mobile"] + t["He_trapped"] + t["He_clusters"]
        t["He_cons_err"] = t["He_retained"] + t["He_surface"] - t["He_in"]
        return t

    def run(self, n_shots, record=None, profile_at=(), verbose=0, time_limit=None, on_snapshot=None):
        """Advance n_shots.  record: iterable of shot numbers to store totals at (default: log-spaced
        plus every 10th); profile_at: shot numbers at which depth profiles are stored."""
        t0 = time.time()
        rec = set(record) if record is not None else None
        prof = set(profile_at)
        snaps, profs = [], {}
        for _ in range(n_shots):
            self.shot()
            k = self.shot_no
            if rec is None or k in rec or k == n_shots:
                sn = self.snapshot()
                snaps.append(sn)
                if on_snapshot:
                    on_snapshot(sn)
            if k in prof:
                profs[k] = self.fm.profiles(self.y)
            if verbose and k % verbose == 0:
                sn = self.snapshot()
                print(f"shot {k}: He retained {sn['He_retained']:.3e}/m2, released {sn['He_surface']:.3e}, "
                      f"cons err {sn['He_cons_err']:.1e}  ({time.time() - t0:.0f}s)", flush=True)
            if time_limit and time.time() - t0 > time_limit:
                break
        return snaps, profs
