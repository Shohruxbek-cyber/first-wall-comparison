"""Ion energy deposition in the continuous-slowing-down approximation (CSDA).

Stopping data: NIST PSTAR (protons) and ASTAR (alphas), data/pstar_*.csv, data/astar_*.csv,
total (electronic + nuclear) stopping power in MeV cm^2/g and CSDA range in g/cm^2.

Deposition is computed cell by cell from the residual CSDA range:
    R(E_out) = R(E_in) - m_cell      (mass units)
so the energy left in a cell is E_in - E_out and the sum over cells telescopes exactly to
the incident energy (minus anything that exits the back of the domain).

Known limitation: CSDA path length is not penetration depth.  NIST
gives detour factors (projected/CSDA range) as low as ~0.1 for keV ions in W, so CSDA
depth profiles are too deep for low-energy ions.  A projected-range variant is a planned
sensitivity case.

Below the 1 keV lower limit of the NIST tables, R(E) is extended as R(1 keV) sqrt(E/1 keV),
i.e. electronic stopping proportional to velocity (Lindhard-Scharff regime).  This is an
ASSUMPTION; it affects only the lowest debris-ion bins.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .source_term import PulsedSource

DATA = Path(__file__).resolve().parents[2] / "data"
MEV = 1.602176634e-13  # J


class StoppingTable:
    """CSDA range table in SI: E [J per ion], R [kg/m^2]."""

    def __init__(self, csv_path):
        d = np.genfromtxt(csv_path, delimiter=",", skip_header=2)
        self.E = d[:, 0] * MEV
        self.S = d[:, 3] * MEV * 0.1  # MeV cm^2/g -> J m^2/kg
        self.R = d[:, 4] * 10.0  # g/cm^2 -> kg/m^2
        self.detour = d[:, 6]
        self.source = str(csv_path)
        self._lnE, self._lnR = np.log(self.E), np.log(self.R)

    @classmethod
    def nist(cls, prog, matno):
        return cls(DATA / f"{prog.lower()}_{matno}.csv")

    def range(self, E):
        E = np.asarray(E, float)
        R = np.exp(np.interp(np.log(np.maximum(E, 1e-300)), self._lnE, self._lnR))
        low = E < self.E[0]
        R = np.where(low, self.R[0] * np.sqrt(np.maximum(E, 0) / self.E[0]), R)
        return np.where(E > 0, R, 0.0)

    def energy_at_range(self, R):
        R = np.asarray(R, float)
        E = np.exp(np.interp(np.log(np.maximum(R, 1e-300)), self._lnR, self._lnE))
        low = R < self.R[0]
        E = np.where(low, self.E[0] * (np.maximum(R, 0) / self.R[0]) ** 2, E)
        return np.where(R > 0, E, 0.0)


def csda_cell_energy(E0, m_cells, tables_per_cell):
    """Energy [J] deposited per cell by one ion of energy E0 [J] entering cell 0.

    tables_per_cell: sequence giving the StoppingTable for each cell (layered media)."""
    out = np.zeros(len(m_cells))
    E = E0
    for i, (m, tab) in enumerate(zip(m_cells, tables_per_cell)):
        if E <= 0:
            break
        E_out = float(tab.energy_at_range(tab.range(E) - m)) if tab.range(E) > m else 0.0
        out[i] = E - E_out
        E = E_out
    return out


class MonoIonSource(PulsedSource):
    """Monoenergetic ions (energy E_ion [J]) with total energy fluence `fluence` [J/m^2]."""

    def __init__(self, fluence, shape, E_ion, tables_by_layer, dt_fraction=0.1):
        super().__init__(fluence, shape, dt_fraction)
        self.E_ion, self.tables = E_ion, tables_by_layer

    def fractions(self, solver):
        tabs = [self.tables[j] for j in solver.layer]
        return csda_cell_energy(self.E_ion, solver.m, tabs) / self.E_ion


# ==========================================================================================
# Threat ion spectra (ARIES NRL 154 MJ tables), time of flight, and spectral sources
# ==========================================================================================
from dataclasses import dataclass  # noqa: E402

from .source_term import TabulatedPulse, SourceTerm  # noqa: E402

C_LIGHT = 2.99792458e8  # m/s
AMU_MEV = 931.49410242  # MeV
KEV_J = 1.602176634e-16

# ion mass [u] and charge; SRIM/NIST naming
ION_MASS_U = {"Protons": 1.00728, "Deuterons": 2.01355, "Tritons": 3.01550, "3He": 3.01493, "4He": 4.00151,
              "12C": 11.99671, "Au": 196.96657}
# published energies per species (MJ), ARIES-IFE accounting page, archived 2005-05-06
ACCOUNTING_MJ = {
    ("burn", "Protons"): 0.54, ("burn", "Deuterons"): 4.78, ("burn", "Tritons"): 4.32, ("burn", "3He"): 0.0024,
    ("burn", "4He"): 9.86,
    ("debris", "Protons"): 0.11, ("debris", "Deuterons"): 8.87, ("debris", "Tritons"): 11.1, ("debris", "3He"): 0.002,
    ("debris", "4He"): 1.33, ("debris", "12C"): 0.56, ("debris", "Au"): 0.15,
}
YIELD_REF = 154e6  # J


@dataclass
class IonBin:
    group: str
    species: str
    E_lo: float  # J per ion
    E_hi: float
    E_mean: float  # tabulated mid-bin energy, J (used for the bin's depth profile)
    energy: float  # J (whole target)
    number: float  # ions (whole target)
    lnE_tab: np.ndarray = None  # species table (ln E [J]) for the within-bin spectral shape
    lnn_tab: np.ndarray = None  # ln dN/dE at those points (arbitrary normalisation)


# Ion spectra (revision of 22 Oct 2026).  Three constructions of the bins:
#   "conserving" (production): number AND energy of every species fixed.  4He burn + debris = one alpha per D-T
#       fusion (154 MJ / 17.6 MeV), split in the table's burn/debris ratio; every other species keeps its table
#       number (D and T agree with the fuel left after burn within 3 %).  The energy is then set to the accounting:
#       burn ions (born at fixed nuclear energies) keep their table energies and the bin weights get a power-law
#       tilt, num_i (E_i/E_ref)^b, so no ion is moved above the birth or knock-on energies that bound the table;
#       debris ions (the expanding target plasma) keep their bin weights and the energy axis is stretched by the
#       factor a.  Burn 4He bins are 0.5 MeV wide (LINEAR_BINS_KEV).
#   "conserving_tilt", "conserving_stretch": the same numbers with the tilt, or the stretch, for both groups
#       (sensitivity of the tail-sensitive results only).
#   "normalized": the construction used before 22 Oct 2026 (log-midpoint edges; each species' NUMBER scaled so that
#       its energy matches the accounting; 4He then 1.20 per fusion).  Reproduces every earlier result bit for bit.
#   "raw": the table integral as read (log-midpoint edges, no scaling; normalize=False / normalize_ions=false).
# The default is "conserving"; the environment variable IFE_WALL_ION_SPECTRA overrides it, so that scripts that pass
# normalize=True reproduce the earlier results with IFE_WALL_ION_SPECTRA=normalized.
ION_SPECTRA = ("conserving", "conserving_tilt", "conserving_stretch", "normalized", "raw")
ENERGY_FIT = {"conserving": {"burn": "tilt", "debris": "stretch"}, "conserving_tilt": {"burn": "tilt", "debris": "tilt"},
              "conserving_stretch": {"burn": "stretch", "debris": "stretch"}}
DEFAULT_ION_SPECTRA = os.environ.get("IFE_WALL_ION_SPECTRA", "conserving")
if DEFAULT_ION_SPECTRA not in ION_SPECTRA:
    raise ValueError(f"IFE_WALL_ION_SPECTRA={DEFAULT_ION_SPECTRA!r}; expected one of {ION_SPECTRA}")
Q_DT_MEV = 17.6  # MeV per D-T fusion (F1: fusions from 154 MJ at 17.6 MeV each)
N_FUSIONS_REF = YIELD_REF / (Q_DT_MEV * MEV)  # 5.461e19
# burn 4He: the first eight mid-bin energies (228.7 ... 3542.8 keV) lie one in each 0.5 MeV interval from 0 to
# 4 MeV, with spacings of 434-527 keV: the bins are 0.5 MeV wide and the tabulated energies are bin means.  The
# log-midpoint rule made the 3.54 MeV bin 1.52 MeV wide (3.31-4.83 MeV).  (width keV, number of rows)
LINEAR_BINS_KEV = {("burn", "4He"): (500.0, 8)}
FIRST_EDGE_KEV = 0.5  # lower edge used for the within-bin shape of a bin that starts at 0 (log grids)


def _table_rows():
    return [r.split(",") for r in (DATA / "aries_nrl154_dd_spectra.csv").read_text().splitlines()[2:] if r]


def _species_table(rows, g, s):
    d = np.array([(float(r[2]), float(r[3])) for r in rows if r[0] == g and r[1] == s])
    E, n = d[:, 0], d[:, 1]  # keV, #/keV
    keep = n > 1.0  # rows with 1.0 #/keV are table padding ("1.00000E+00")
    return E[keep], n[keep]


def _logmid_edges(E):
    lE = np.log(E)
    mid = np.exp(0.5 * (lE[1:] + lE[:-1]))
    return np.r_[E[0] ** 2 / mid[0], mid], np.r_[mid, E[-1] ** 2 / mid[-1]]


def ion_bin_edges_keV(g, s, E, spectra="conserving"):
    """Bin edges (keV) for the table's mid-bin energies E: log midpoints; for "conserving" the linear 0.5 MeV grid
    of LINEAR_BINS_KEV (the bin after it starts at the top of that grid).  The first edge may be 0."""
    lo, hi = _logmid_edges(E)
    if spectra.startswith("conserving") and (g, s) in LINEAR_BINS_KEV:
        w, k = LINEAR_BINS_KEV[(g, s)]
        e = w * np.arange(k + 1)
        if not np.all((E[:k] > e[:-1]) & (E[:k] < e[1:])):
            raise ValueError(f"{g} {s}: the first {k} mid-bin energies are not one per {w} keV interval")
        lo[:k], hi[:k] = e[:-1], e[1:]
        lo[k] = e[k]
    return lo, hi


def _power_tilt(num, E, mean):
    """Exponent b such that the weights num (E/E_ref)^b have the mean energy `mean`; returns (b, weights)."""
    from scipy.optimize import brentq
    lr = np.log(E) - np.average(np.log(E), weights=num)

    def w(b):
        x = b * lr
        return num * np.exp(x - x.max())

    b = brentq(lambda b: (w(b) * E).sum() / w(b).sum() - mean, -60.0, 60.0, xtol=1e-14, rtol=1e-15, maxiter=500)
    return b, w(b)


def ion_spectra_tables(spectra=None):
    """Per species: table energies E (keV), dN/dE (#/keV), edges (keV), and the bin numbers and energies (J) of the
    chosen construction (whole target, energies with the stretch a applied), with the number factor f, the tilt
    exponent b and the stretch a."""
    spectra = DEFAULT_ION_SPECTRA if spectra is None else spectra
    if spectra not in ION_SPECTRA:
        raise ValueError(f"spectra={spectra!r}; expected one of {ION_SPECTRA}")
    rows = _table_rows()
    out = {}
    for (g, s), E_ref in ACCOUNTING_MJ.items():
        E, n = _species_table(rows, g, s)
        lo, hi = ion_bin_edges_keV(g, s, E, spectra)
        out[(g, s)] = dict(E=E, n=n, lo=lo, hi=hi, num_table=n * (hi - lo), E_acc=E_ref * 1e6)
    if spectra in ENERGY_FIT:
        he = out[("burn", "4He")]["num_table"].sum() + out[("debris", "4He")]["num_table"].sum()
        f_he = N_FUSIONS_REF / he
        for (g, s), d in out.items():
            f = f_he if s == "4He" else 1.0
            N_t = f * d["num_table"].sum()
            if ENERGY_FIT[spectra][g] == "tilt":
                b, w = _power_tilt(d["num_table"], d["E"], d["E_acc"] / N_t / KEV_J)
                number, a = w * (N_t / w.sum()), 1.0
            else:
                b, number = 0.0, f * d["num_table"]
                a = d["E_acc"] / (number * d["E"] * KEV_J).sum()
            d.update(number=number, energy=number * d["E"] * a * KEV_J, number_factor=f, tilt=b, stretch=a)
    else:
        for (g, s), d in out.items():
            en = d["num_table"] * d["E"] * KEV_J
            scale = d["E_acc"] / en.sum() if spectra == "normalized" else 1.0
            d.update(number=d["num_table"] * scale, energy=en * scale, number_factor=scale, tilt=0.0, stretch=1.0)
    return out


def load_ion_bins(normalize=True, spectra=None):
    """Ion bins of the ARIES NRL 154 MJ tables (whole target).  spectra: "conserving", "normalized" or "raw" (see
    ION_SPECTRA above); None = DEFAULT_ION_SPECTRA.  normalize=False (the earlier switch) selects "raw"."""
    if not normalize:
        spectra = "raw"
    spectra = DEFAULT_ION_SPECTRA if spectra is None else spectra
    if spectra not in ION_SPECTRA:
        raise ValueError(f"spectra={spectra!r}; expected one of {ION_SPECTRA}")
    if spectra in ("normalized", "raw"):  # the construction used before 22 Oct 2026, unchanged (bit for bit)
        rows = _table_rows()
        bins = []
        for (g, s), E_ref in ACCOUNTING_MJ.items():
            d = np.array([(float(r[2]), float(r[3])) for r in rows if r[0] == g and r[1] == s])
            E, n = d[:, 0], d[:, 1]  # keV, #/keV
            keep = n > 1.0  # rows with 1.0 #/keV are table padding ("1.00000E+00")
            E, n = E[keep], n[keep]
            lE = np.log(E)
            mid = np.exp(0.5 * (lE[1:] + lE[:-1]))
            lo = np.r_[E[0] ** 2 / mid[0], mid]
            hi = np.r_[mid, E[-1] ** 2 / mid[-1]]
            num = n * (hi - lo)
            en = num * E * KEV_J
            scale = E_ref * 1e6 / en.sum() if spectra == "normalized" else 1.0
            lnE_tab, lnn_tab = np.log(E * KEV_J), np.log(n)
            for a, b, e, N, Eb in zip(lo, hi, E, num, en):
                bins.append(IonBin(g, s, a * KEV_J, b * KEV_J, e * KEV_J, Eb * scale, N * scale, lnE_tab, lnn_tab))
        return bins
    bins = []
    for (g, s), d in ion_spectra_tables(spectra).items():
        k = d["stretch"] * KEV_J
        lnE_tab, lnn_tab = np.log(d["E"] * k), np.log(d["n"])
        lo = np.where(d["lo"] > 0, d["lo"], FIRST_EDGE_KEV)
        for a, b, e, N, Eb in zip(lo, d["hi"], d["E"], d["number"], d["energy"]):
            bins.append(IonBin(g, s, a * k, b * k, e * k, Eb, N, lnE_tab, lnn_tab))
    return bins


def speed(E_J, mass_u):
    mc2 = mass_u * AMU_MEV * MEV
    gamma = 1.0 + E_J / mc2
    return C_LIGHT * np.sqrt(1.0 - 1.0 / gamma ** 2)


def arrival_pulse(b: IonBin, R, n_sub=24):
    """Power-vs-time shape for one bin: ions emitted at t = 0 from the target point, free
    streaming to radius R.  Within the bin dN/dE = bin_shape() (log-log interpolation of the
    table, tilted so the bin's number and energy both match the table; flat-in-bin would put
    artificial power jumps at bin edges).  The power arriving is dN/dE(E) E |dE/dt| on a
    sub-grid between t(E_hi) and t(E_lo); the pulse is normalised and carries the bin energy."""
    m = ION_MASS_U[b.species]
    E = np.geomspace(b.E_hi, b.E_lo, n_sub)
    t = R / speed(E, m)
    dEdt = np.abs(np.gradient(E, t))
    nE = bin_shape(b, E)
    return TabulatedPulse(t, nE * E * dEdt)


def bin_shape(b: IonBin, E, n_quad=400):
    """Within-bin dN/dE [ions/J] at energies E.

    Shape: log-log interpolation of the species table, multiplied by an exponential tilt
    exp(c ln(E/E_mid)) and an amplitude A.  c and A are solved so that the bin reproduces the
    table exactly in BOTH particle number (b.number) and energy (b.energy), i.e. its mean
    energy equals the tabulated mid-bin energy (author request 2026-09-27)."""
    if b.lnE_tab is None:
        return np.full_like(np.asarray(E, float), b.number / (b.E_hi - b.E_lo))
    if not hasattr(b, "_tilt"):
        x = np.geomspace(b.E_lo, b.E_hi, n_quad)
        base = np.exp(np.interp(np.log(x), b.lnE_tab, b.lnn_tab))
        target = b.energy / b.number  # = tabulated mid-bin energy
        lx = np.log(x / b.E_mean)
        # A bin wider than about e^3 (the first burn 4He bin of the conserving spectra, 0.5-500 keV) overflows at the
        # ends of the bracket; its tilt factor is then scaled by exp(-max) (cancels in the mean and in A).  Every
        # narrower bin takes the unshifted arithmetic, so the earlier spectra are reproduced bit for bit.
        with np.errstate(over="ignore", invalid="ignore"):
            wide = not all(np.isfinite(np.trapezoid(base * np.exp(c * lx) * x, x)) for c in (-200.0, 200.0))

        def shift(c):
            return float((c * lx).max()) if wide else 0.0

        def mean_E(c):
            w = base * np.exp(c * lx - shift(c))
            return np.trapezoid(w * x, x) / np.trapezoid(w, x)

        from scipy.optimize import brentq
        c = brentq(lambda c: mean_E(c) - target, -200.0, 200.0, xtol=1e-12)
        w = base * np.exp(c * lx - shift(c))
        A = b.number / np.trapezoid(w, x)
        object.__setattr__(b, "_tilt", (c, A, shift(c)))
    c, A, sh = b._tilt
    E = np.asarray(E, float)
    return A * np.exp(np.interp(np.log(E), b.lnE_tab, b.lnn_tab)) * np.exp(c * np.log(E / b.E_mean) - sh)


# ---------------- stopping providers ----------------------------------------------------
class NistScaledStopping:
    """RETIRED for production (2026-09-27): superseded by srim.SrimStopping.  Kept only for
    the SRIM-vs-NIST cross-check.  Constructing it outside that check raises unless
    cross_check=True.
    NIST PSTAR/ASTAR with equal-velocity scaling
        S_ion(E) = S_ref(E m_ref/m_ion),  R_ion(E) = (m_ion/m_ref) R_ref(E m_ref/m_ion)
    for D, T (from protons) and 3He (from alphas).  Valid only for W and Fe (no Li in STAR).
    C and Au are not available -> KeyError."""

    REF = {"Protons": ("PSTAR", "Protons"), "Deuterons": ("PSTAR", "Protons"), "Tritons": ("PSTAR", "Protons"),
           "3He": ("ASTAR", "4He"), "4He": ("ASTAR", "4He")}
    MATNO = {"W": "074", "Fe": "026"}

    def __init__(self, cross_check=False):
        if not cross_check:
            raise RuntimeError("NistScaledStopping is retired for production; use ife_wall.srim.SrimStopping")
        self._tab = {}

    def table(self, species, element):
        prog, ref = self.REF[species]
        key = (prog, element)
        if key not in self._tab:
            self._tab[key] = StoppingTable.nist(prog, self.MATNO[element])
        return self._tab[key], ION_MASS_U[species] / ION_MASS_U[ref]

    def range(self, species, element, E):
        tab, r = self.table(species, element)
        return r * tab.range(np.asarray(E) / r)

    def energy_at_range(self, species, element, Rg):
        tab, r = self.table(species, element)
        return r * tab.energy_at_range(np.asarray(Rg) / r)


def csda_fractions_layered(E0, species, m_cells, layer_of_cell, layer_elements, stopping):
    """Vectorised CSDA: fraction of E0 deposited in each cell of a layered mesh."""
    out = np.zeros(len(m_cells))
    E_in = E0
    for j, el in enumerate(layer_elements):
        sel = np.nonzero(layer_of_cell == j)[0]
        if len(sel) == 0 or E_in <= 0:
            continue
        R0 = float(stopping.range(species, el, E_in))
        Rb = R0 - np.r_[0.0, np.cumsum(m_cells[sel])]
        Eb = np.where(Rb > 0, stopping.energy_at_range(species, el, np.maximum(Rb, 1e-300)), 0.0)
        Eb[0] = E_in
        out[sel] = Eb[:-1] - Eb[1:]
        E_in = float(Eb[-1])
    # energy carried away by backscattered ions (TRIM; W only) never reaches the mesh; the
    # solver books it with the pass-through energy (incident - deposited)
    if hasattr(stopping, "reflected_energy_fraction") and len(layer_elements):
        out *= 1.0 - stopping.reflected_energy_fraction(species, layer_elements[layer_of_cell[0]], E0)
    return out / E0


class IonSpectrumSource(SourceTerm):
    """Sum over (species, bin) components; each has its own TOF pulse and depth profile."""

    def __init__(self, bins, yield_J, R, layer_elements, stopping):
        scale = yield_J / YIELD_REF / (4 * np.pi * R ** 2)  # whole-target J -> J/m^2 at wall
        self.bins = [b for b in bins if b.energy > 0]
        self.fluence = np.array([b.energy * scale for b in self.bins])
        self.pulses = [arrival_pulse(b, R) for b in self.bins]
        self.layer_elements, self.stopping = layer_elements, stopping
        self._events = np.unique(np.concatenate([[p.t[0], p.t[-1]] for p in self.pulses]))
        self.t_first, self.t_last = self._events[0], self._events[-1]
        self._frac = None

    def total_fluence(self):
        return float(self.fluence.sum())

    def energy(self, t0, t1):
        if t1 <= self.t_first or t0 >= self.t_last:
            return 0.0
        return float(sum(F * (p.cumulative(t1) - p.cumulative(t0)) for F, p in zip(self.fluence, self.pulses)))

    def energy_components(self, t0, t1):
        """Energy [J/m^2] of each (species, bin) component arriving in [t0, t1]."""
        if t1 <= self.t_first or t0 >= self.t_last:
            return np.zeros(len(self.bins))
        return np.array([F * (p.cumulative(t1) - p.cumulative(t0)) for F, p in zip(self.fluence, self.pulses)])

    def fractions(self, solver):
        """(n_components, n_cells) matrix of depth fractions (the solver weights rows by energy_components)."""
        fn = getattr(self.stopping, "fractions", None)
        if fn is None:
            fn = lambda E0, sp, m, lay, els: csda_fractions_layered(E0, sp, m, lay, els, self.stopping)
        self._frac = np.array([fn(b.E_mean, b.species, solver.m, solver.layer, self.layer_elements)
                               for b in self.bins])
        return self._frac

    def dt_limit(self, t):
        return (self.t_last - self.t_first) / 200 if self.t_first <= t < self.t_last else np.inf

    def next_event(self, t):
        # Only the first and last arrival are true on/off events.  Bin edges inside the pulse
        # are continuous in power and are resolved by dt_limit; treating them as events would
        # reset the step size ~250 times per shot (profiling 2026-09-27).
        ev = [e for e in (self.t_first, self.t_last) if e > t * (1 + 1e-14) + 1e-300]
        return float(min(ev)) if ev else np.inf
