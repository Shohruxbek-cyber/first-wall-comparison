"""SRIM-2012 stopping/range tables and TRIM depth profiles (data/srim/, see its README.md).

Production ion-stopping provider (replaces the PRELIMINARY NistScaledStopping).

Depth model (chosen 2026-09-27):
  * Energy loss along the path follows SRIM total stopping S = S_e + S_n (SR Module tables).
  * "projected" mode (baseline): energy is placed at depth by the projected range,
        x(E) = Rp_eff(E0) - Rp_eff(E),   Rp_eff(E) = f(E) * Rp_table(E),
    where f(E) = TRIM mean range / table projected range from the Part-B / supplementary
    TRIM runs.  f departs from 1 only for keV H isotopes and He in W (2.1 at 1 keV, 1.4-1.5
    at 10 keV); elsewhere TRIM and the tables agree within ~4 % and f = 1.
  * "csda" mode (sensitivity): depth = CSDA path length (integral of dE/S).
  * Backscattered energy (TRIM, W targets) is removed from the deposit: the ion's deposited
    energy is (1 - eta_E(E0)) E0.  Li backscatter <= 0.14 % is neglected.
Lengths are handled in areal mass [kg/m^2] (length x SRIM default density), so Li tables at
0.534 g/cm^3 apply to liquid Li at any density (README: lengths scale as 0.534/rho).

ASSUMPTIONS:
  * f and eta_E for p and t in W are taken from the d runs at equal velocity (E * m_d/m);
    for 3He from the 4He runs at equal velocity.
  * No TRIM runs for C in W or any ion in Fe: f = 1, eta_E = 0 there.
  * Below the lowest TRIM energy (1 keV) f and eta_E are held at their 1 keV values; above
    the highest they go to 1 and 0.
  * Straggling is not represented (each bin's profile ends at Rp_eff); checked against the
    TRIM profiles in validation/srim_profiles.py.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "srim"
EV = 1.602176634e-19
LEN = {"A": 1e-10, "nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0, "km": 1e3}
EN = {"eV": EV, "keV": 1e3 * EV, "MeV": 1e6 * EV, "GeV": 1e9 * EV}
MEV_MG_CM2 = 1e6 * EV / 1e-2  # 1 MeV/(mg/cm^2) in J m^2/kg

SPECIES_CODE = {"Protons": "p", "Deuterons": "d", "Tritons": "t", "3He": "he3", "4He": "he4", "12C": "c12", "Au": "au"}
MASS_U = {"p": 1.008, "d": 2.014, "t": 3.016, "he3": 3.016, "he4": 4.003, "c12": 12.0, "au": 196.97}  # as used in SRIM


def parse_sr_table(path):
    """Parse an SR Module output file.  Returns dict with SI arrays (E [J], Se/Sn [J m^2/kg],
    Rp/straggling [m]) and the target density [kg/m^3] used by SRIM."""
    txt = Path(path).read_text(errors="replace")
    rho = float(re.search(r"Density =\s*([\d.E+-]+) g/cm3", txt).group(1)) * 1e3
    if "MeV/(mg/cm2)" not in txt:
        raise ValueError(f"{path}: stopping units are not MeV/(mg/cm2)")
    row = re.compile(r"^\s*([\d.]+)\s+(eV|keV|MeV|GeV)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.]+)\s+(A|nm|um|mm|m|km)"
                     r"\s+([\d.]+)\s+(A|nm|um|mm|m|km)\s+([\d.]+)\s+(A|nm|um|mm|m|km)\s*$")
    E, Se, Sn, Rp, dRp, lat = [], [], [], [], [], []
    for line in txt.splitlines():
        m = row.match(line)
        if m:
            E.append(float(m.group(1)) * EN[m.group(2)])
            Se.append(float(m.group(3)) * MEV_MG_CM2)
            Sn.append(float(m.group(4)) * MEV_MG_CM2)
            Rp.append(float(m.group(5)) * LEN[m.group(6)])
            dRp.append(float(m.group(7)) * LEN[m.group(8)])
            lat.append(float(m.group(9)) * LEN[m.group(10)])
    d = {k: np.array(v) for k, v in dict(E=E, Se=Se, Sn=Sn, Rp=Rp, dRp=dRp, lat=lat).items()}
    if len(d["E"]) < 50 or np.any(np.diff(d["E"]) <= 0):
        raise ValueError(f"{path}: table parse failed ({len(d['E'])} rows)")
    d["rho"] = rho
    return d


class SrimTable:
    """One ion in one target.  Ranges in areal mass [kg/m^2]."""

    def __init__(self, ion, target):
        self.ion, self.target = ion, target
        d = parse_sr_table(DATA / f"{ion}_in_{target}.txt")
        self.E, self.Se, self.Sn, self.rho = d["E"], d["Se"], d["Sn"], d["rho"]
        self.S = self.Se + self.Sn
        self.Rp = d["Rp"] * self.rho  # projected range, kg/m^2
        self.dRp = d["dRp"] * self.rho
        # CSDA range: below the first row S ~ sqrt(E) (velocity-proportional) -> R0 = 2 E0 / S0
        R0 = 2 * self.E[0] / self.S[0]
        lnE = np.log(self.E)
        integrand = self.E / self.S  # dR = dE / S = (E/S) dlnE
        self.Rcsda = R0 + np.r_[0.0, np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(lnE))]
        self._lnE = lnE

    # log-log interpolation; below the table, power-law continuation
    @staticmethod
    def _interp_up(E, lnE, y, low_exp):
        E = np.asarray(E, float)
        lnE_c = np.log(np.maximum(E, 1e-300))
        out = np.exp(np.interp(lnE_c, lnE, np.log(y)))
        below = E < np.exp(lnE[0])
        out = np.where(below, y[0] * (np.maximum(E, 0) / np.exp(lnE[0])) ** low_exp, out)
        return np.where(E > 0, out, 0.0)

    @staticmethod
    def _interp_down(R, R_tab, lnE, low_exp):
        R = np.asarray(R, float)
        lnE_out = np.interp(np.log(np.maximum(R, 1e-300)), np.log(R_tab), lnE)
        E = np.exp(lnE_out)
        below = R < R_tab[0]
        E = np.where(below, np.exp(lnE[0]) * (np.maximum(R, 0) / R_tab[0]) ** (1 / low_exp), E)
        return np.where(R > 0, E, 0.0)

    def csda_range(self, E):
        return self._interp_up(E, self._lnE, self.Rcsda, 0.5)

    def csda_energy(self, R):
        return self._interp_down(R, self.Rcsda, self._lnE, 0.5)

    def proj_range(self, E):
        return self._interp_up(E, self._lnE, self.Rp, 0.5)

    def stopping(self, E):
        return self._interp_up(E, self._lnE, self.S, 0.5)


def _read_trim_summary():
    rows = []
    for f in (DATA / "trim" / "summary.csv", DATA / "trim_supplementary" / "summary.csv"):
        with open(f, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


@lru_cache(maxsize=None)
def trim_points(ion, target):
    """(E_keV, f = TRIM/table range, eta_E backscattered energy fraction, eta_N number fraction)
    from the TRIM runs for this ion/target (quick-KP runs; B1_fullcascade excluded)."""
    pts = [(float(r["E_keV"]), float(r["trim_over_table_range"]), float(r["backscattered_energy_fraction"]),
            float(r["backscatter_fraction"]))
           for r in _read_trim_summary() if r["ion"] == ion and r["target"] == target and "fullcascade" not in r["run"]]
    return sorted(pts)


def _trim_curve(ion, target, E_J, col, above_value):
    """Interpolate a TRIM-derived quantity in log E, with equal-velocity mapping for p, t
    (from d) and he3 (from he4)."""
    ref = {"p": "d", "t": "d", "he3": "he4"}.get(ion, ion)
    pts = trim_points(ref, target)
    if not pts:
        return None
    E_ref_keV = np.asarray(E_J, float) * MASS_U[ref] / MASS_U[ion] / (1e3 * EV)
    Ek = np.array([p[0] for p in pts])
    v = np.array([p[col] for p in pts])
    out = np.interp(np.log(np.maximum(E_ref_keV, 1e-300)), np.log(Ek), v)  # held flat below Ek[0]
    top = E_ref_keV > Ek[-1]
    # above the highest TRIM energy relax to the far-field value over one decade
    frac = np.clip(np.log10(np.maximum(E_ref_keV, 1e-300) / Ek[-1]), 0, 1)
    return np.where(top, v[-1] + (above_value - v[-1]) * frac, out)


class SrimStopping:
    """Stopping provider for csda_fractions_layered: `range`/`energy_at_range` are the
    depth-mapping range functions (projected-effective or CSDA)."""

    TARGET = {"W": "W", "Li": "Li", "Fe": "Fe"}

    def __init__(self, mode="projected", reflection=True, multilayer_trim=False):
        if mode not in ("projected", "csda"):
            raise ValueError(mode)
        self.mode, self.reflection = mode, reflection
        self.multilayer_trim = multilayer_trim  # see srim_fractions (bug fix, opt-in, 2026-09-29)
        self._tab = {}
        self._inv = {}

    def table(self, species, element):
        key = (SPECIES_CODE[species], self.TARGET[element])
        if key not in self._tab:
            self._tab[key] = SrimTable(*key)
        return self._tab[key]

    def f_range(self, species, element, E):
        ion, tgt = SPECIES_CODE[species], self.TARGET[element]
        f = _trim_curve(ion, tgt, E, 1, 1.0) if tgt == "W" else None
        return np.ones_like(np.asarray(E, float)) if f is None else f

    def fractions(self, E0, species, m_cells, layer_of_cell, layer_elements):
        """Per-cell deposited fraction for one ion (baseline dispatch in srim_fractions)."""
        return srim_fractions(E0, species, m_cells, layer_of_cell, layer_elements, self)

    def reflected_energy_fraction(self, species, element, E0):
        ion, tgt = SPECIES_CODE[species], self.TARGET[element]
        if not self.reflection or tgt != "W":
            return 0.0
        eta = _trim_curve(ion, tgt, E0, 2, 0.0)
        return 0.0 if eta is None else float(eta)

    def reflected_number_fraction(self, species, element, E0):
        ion, tgt = SPECIES_CODE[species], self.TARGET[element]
        if tgt != "W":
            return 0.0
        eta = _trim_curve(ion, tgt, E0, 3, 0.0)
        return 0.0 if eta is None else float(eta)

    def _grid(self, species, element):
        """Monotone tabulation of the effective projected range for inversion."""
        key = (species, element)
        if key not in self._inv:
            tab = self.table(species, element)
            E = np.geomspace(1.0 * EV, tab.E[-1], 3000)
            R = tab.proj_range(E) * self.f_range(species, element, E)
            R = np.maximum.accumulate(R) * (1 + 1e-12 * np.arange(len(R)))  # enforce strict monotonicity
            self._inv[key] = (E, R)
        return self._inv[key]

    def range(self, species, element, E):
        tab = self.table(species, element)
        if self.mode == "csda":
            return tab.csda_range(E)
        Eg, Rg = self._grid(species, element)
        return np.interp(np.asarray(E, float), Eg, Rg)

    def energy_at_range(self, species, element, R):
        tab = self.table(species, element)
        if self.mode == "csda":
            return tab.csda_energy(R)
        Eg, Rg = self._grid(species, element)
        R = np.asarray(R, float)
        return np.where(R > 0, np.interp(R, Rg, Eg), 0.0)


def trim_profile(run_dir):
    """Deposited energy per unit depth from a TRIM run: IONIZ (ions + recoils) + PHONON
    (ions + recoils).  Returns depth [m] (bin centres), dE/dx [J/m per ion], bin width [m]."""
    def cols(fname):
        rows = []
        for line in (Path(run_dir) / fname).read_text(errors="replace").splitlines():
            p = line.split()
            if len(p) == 3:
                try:
                    rows.append([float(x) for x in p])
                except ValueError:
                    pass
        return np.array(rows)

    ion = cols("IONIZ.txt")
    ph = cols("PHONON.txt")
    if len(ion) != len(ph) or not np.allclose(ion[:, 0], ph[:, 0]):
        raise ValueError(f"{run_dir}: IONIZ/PHONON depth grids differ")
    x_hi = ion[:, 0] * 1e-10  # TRIM lists the bin's deeper edge
    w = np.diff(np.r_[0.0, x_hi])
    q = (ion[:, 1] + ion[:, 2] + ph[:, 1] + ph[:, 2]) * EV / 1e-10  # eV/(A ion) -> J/(m ion)
    return x_hi - w / 2, q, w


# ==========================================================================================
# TRIM-profile deposition for light ions in W (author decision 2026-09-27)
# ==========================================================================================
# validation/srim_checks.py showed the table-based projected model is 24-31 % too shallow
# (x50) for d and He at 1-100 keV in W, so there the deposit follows the TRIM profiles.
TRIM_RUNS_W = {  # ion -> [(E_keV, run directory)], quick-KP runs
    "d": [(1.0, "trim/B1"), (10.0, "trim/B2"), (100.0, "trim/B3")],
    "he4": [(1.0, "trim_supplementary/S1_he4_1keV_W"), (10.0, "trim_supplementary/S2_he4_10keV_W"),
            (100.0, "trim/B4"), (2000.0, "trim/B5")],
}
U_GRID = np.linspace(0.0, 12.0, 1201)  # depth in units of the 50 %-energy depth


@lru_cache(maxsize=None)
def _trim_shape(run):
    """(x50 [m, at SRIM density], cumulative deposit fraction C(u) on U_GRID, normalised to 1)."""
    x, q, w = trim_profile(DATA / run)
    c = np.r_[0.0, np.cumsum(q * w)]
    edges = np.r_[0.0, x + w / 2]
    c /= c[-1]
    x50 = float(np.interp(0.5, c, edges))
    return x50, np.interp(U_GRID * x50, edges, c, right=1.0)


def trim_interp(ion_ref, E_keV_ref, exclude=None):
    """x50 [m] and C(u) for the reference ion at E_keV_ref, log-E interpolation between TRIM
    runs (optionally excluding one run, for leave-one-out tests).  Returns None outside the
    covered range [E_min, E_max] except below E_min, where the E_min shape is used with
    x50 extrapolated by the power law of the first interval."""
    pts = [(E, r) for E, r in TRIM_RUNS_W.get(ion_ref, []) if r != exclude]
    if not pts or E_keV_ref > pts[-1][0]:
        return None
    Es = np.array([p[0] for p in pts])
    x50s = np.array([_trim_shape(p[1])[0] for p in pts])
    if E_keV_ref <= Es[0]:
        k = np.log(x50s[1] / x50s[0]) / np.log(Es[1] / Es[0])
        return x50s[0] * (E_keV_ref / Es[0]) ** k, _trim_shape(pts[0][1])[1]
    i = int(np.searchsorted(Es, E_keV_ref)) - 1
    i = min(max(i, 0), len(Es) - 2)
    a = np.log(E_keV_ref / Es[i]) / np.log(Es[i + 1] / Es[i])
    x50 = np.exp((1 - a) * np.log(x50s[i]) + a * np.log(x50s[i + 1]))
    C = (1 - a) * _trim_shape(pts[i][1])[1] + a * _trim_shape(pts[i + 1][1])[1]
    return x50, C


def _trim_fractions_W(E0, species, m_cells, rho_srim_W, stopping, exclude=None):
    """Per-cell deposited fraction for an ion entering a W layer, from interpolated TRIM
    profiles.  p, t use d at equal velocity; he3 uses he4 (x50 scaled by the mass ratio).
    Returns None if the energy is above the TRIM coverage."""
    ion = SPECIES_CODE[species]
    ref = {"p": "d", "t": "d", "d": "d", "he3": "he4", "he4": "he4"}.get(ion)
    if ref is None:
        return None
    E_ref_keV = E0 * MASS_U[ref] / MASS_U[ion] / (1e3 * EV)
    r = trim_interp(ref, E_ref_keV, exclude)
    if r is None:
        return None
    x50, C = r
    x50 *= MASS_U[ion] / MASS_U[ref]  # equal-velocity range scaling (ASSUMPTION for p, t, he3)
    xi50 = x50 * rho_srim_W  # kg/m^2
    b = np.r_[0.0, np.cumsum(m_cells)] / xi50
    cum = np.interp(b, U_GRID, C, right=1.0)
    frac = np.diff(cum)
    return frac * (1.0 - stopping.reflected_energy_fraction(species, "W", E0))


def srim_fractions(E0, species, m_cells, layer_of_cell, layer_elements, stopping, exclude=None):
    """Baseline deposit for one ion: TRIM profiles for H isotopes / He in a W front layer up to
    the highest TRIM energy (d: 100 keV, he4: 2 MeV, at equal velocity); otherwise the
    SRIM-table model (projected or CSDA per `stopping.mode`) via csda_fractions_layered."""
    from .ions import csda_fractions_layered
    if stopping.mode != "projected" or layer_elements[layer_of_cell[0]] != "W":
        return csda_fractions_layered(E0, species, m_cells, layer_of_cell, layer_elements, stopping)
    # Cells treated as the W medium for the TRIM/stretch corrections:
    #  * default (multilayer_trim=False): only when the WHOLE mesh is one layer.  This is the
    #    behaviour behind all results up to 2026-09-27; with W on steel the corrections are
    #    silently skipped (bug found 2026-09-29).  Kept as default so existing
    #    results reproduce exactly.
    #  * multilayer_trim=True (fix, opt-in): the leading block of W cells (one or more W
    #    layers, e.g. skin + bulk), provided the ion stops inside it.
    if getattr(stopping, "multilayer_trim", False):
        is_w = np.array([layer_elements[j] == "W" for j in layer_of_cell])
        nW = int(np.argmin(is_w)) if not is_w.all() else len(m_cells)
    else:
        nW = len(m_cells) if np.all(layer_of_cell == layer_of_cell[0]) else 0
    if nW:
        tab = stopping.table(species, "W")
        f = _trim_fractions_W(E0, species, m_cells[:nW], tab.rho, stopping, exclude)
        eta = stopping.reflected_energy_fraction(species, "W", E0)
        if f is not None and (nW == len(m_cells) or f.sum() >= (1.0 - eta) * (1 - 1e-6)):
            out = np.zeros(len(m_cells))
            out[:nW] = f
            return out
        if SPECIES_CODE[species] in ("p", "d", "t"):
            # H isotopes above the d TRIM coverage (100 keV): stretch the table model's depth by g
            g = h_isotope_depth_stretch(species, E0)
            m_mod = np.array(m_cells, float)
            m_mod[:nW] /= g
            return csda_fractions_layered(E0, species, m_mod, layer_of_cell, layer_elements, stopping)
    return csda_fractions_layered(E0, species, m_cells, layer_of_cell, layer_elements, stopping)


# ESTIMATED (pending TRIM runs of d at 300 keV, 1 MeV, 3 MeV in W): depth stretch applied to
# the SRIM-table projected model for H isotopes in W above the d TRIM coverage.  Anchors, in
# energy per nucleon (equal velocity): 50 keV/u -> 1.31 (d 100 keV, TRIM x50 / model x50,
# validation/srim_checks.py); 500 keV/u -> 1.04 (He-4 2 MeV, same ratio); >= 5 MeV/u -> 1.0.
G_ANCHORS = ((50.0, 1.31), (500.0, 1.04), (5000.0, 1.0))


def h_isotope_depth_stretch(species, E0):
    e_u = E0 / (1e3 * EV) / MASS_U[SPECIES_CODE[species]]  # keV per amu
    x = np.log([a[0] for a in G_ANCHORS])
    y = [a[1] for a in G_ANCHORS]
    return float(np.interp(np.log(max(e_u, 1e-30)), x, y))
