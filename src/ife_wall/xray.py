"""Photon (x-ray) energy deposition.

Spectrum: DIGITIZED from Raffray (2005) Fig. 2 (NRL 154 MJ target), rebinned onto >= 50
logarithmic bins and normalised to the published x-ray energy (Raffray Table 1: 2.1 MJ)
scaled to the scenario yield.

Attenuation: Beer law per energy bin, q(x) = F_E mu exp(-mu x).  mu/rho for each element:
  * E >= 1 keV: NIST XCOM, photoelectric + incoherent ("pe+incoh"), or
    photoelectric only ("pe", sensitivity case);
  * E < 1 keV: CXRO/Henke photoabsorption (XCOM starts at 1 keV).
Counting incoherent scattering as locally absorbed over-estimates deposition (the scattered
photon carries most of the energy away); photoelectric-only under-estimates slightly.  The
two bracket the answer.  Fluorescence and photo-electron transport are neglected (they
redistribute energy over ~0.1-1 um).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .source_term import PulsedSource

DATA = Path(__file__).resolve().parents[2] / "data"
KEV = 1.602176634e-16  # J


def load_digitized_spectrum():
    d = np.genfromtxt(DATA / "raffray2005_fig2_photon_spectrum_DIGITIZED.csv", delimiter=",", comments="#")
    return d[:, 0], d[:, 1]  # keV, J/keV


def rebin_spectrum(n_bins=100, total_energy=None):
    """Return (E_center_keV, energy_per_bin_J) on log bins; normalised to total_energy [J] if given."""
    E, dEdE = load_digitized_spectrum()
    edges = np.logspace(np.log10(E[0]), np.log10(E[-1]), n_bins + 1)
    fine = np.logspace(np.log10(E[0]), np.log10(E[-1]), 20000)
    y = np.exp(np.interp(np.log(fine), np.log(E), np.log(dEdE)))
    cum = np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(fine))]
    e_bin = np.diff(np.interp(edges, fine, cum))
    centers = np.sqrt(edges[1:] * edges[:-1])
    if total_energy is not None:
        e_bin *= total_energy / e_bin.sum()
    return centers, e_bin


class MuRho:
    """mu/rho(E) [m^2/kg] for one element, E in keV."""

    def __init__(self, z, henke_symbol, mode="pe+incoh"):
        x = np.genfromtxt(DATA / f"xcom_Z{z}.csv", delimiter=",", skip_header=2, usecols=range(8))
        E = x[:, 0] * 1e3  # keV
        mu = x[:, 3] + (x[:, 2] if mode == "pe+incoh" else 0.0)  # cm^2/g
        # absorption edges appear twice in XCOM (below/above): nudge the lower copy down
        for i in range(1, len(E)):
            if E[i] <= E[i - 1]:
                E[i - 1] = E[i] * (1 - 1e-6)
        h = np.genfromtxt(DATA / f"henke_{henke_symbol}.csv", delimiter=",", skip_header=2)
        hE, hmu = h[:, 0] * 1e-3, h[:, 3]
        sel = hE < E[0]
        self.lnE = np.log(np.r_[hE[sel], E])
        self.lnmu = np.log(np.r_[hmu[sel], mu] * 0.1)  # cm^2/g -> m^2/kg
        self.mode = mode

    def __call__(self, E_keV):
        return np.exp(np.interp(np.log(E_keV), self.lnE, self.lnmu))


ELEMENTS = {"W": (74, "w"), "Li": (3, "li"), "Fe": (26, "fe")}


class XraySource(PulsedSource):
    """Spectral Beer-law deposition on the current (layered) mesh.

    layer_elements: element symbol for each layer index (e.g. ["W", "Fe"]); steel is
    represented by Fe for photon attenuation (ASSUMPTION; Cr/W alloying ignored)."""

    def __init__(self, fluence, shape, layer_elements, n_bins=100, mode="pe+incoh", dt_fraction=0.1):
        super().__init__(fluence, shape, dt_fraction)
        self.E, e_bin = rebin_spectrum(n_bins)
        self.w = e_bin / e_bin.sum()  # energy weight of each bin
        mu = {el: MuRho(*ELEMENTS[el], mode=mode) for el in set(layer_elements)}
        self.mu_layer = np.array([mu[el](self.E) for el in layer_elements])  # (n_layers, n_bins)

    def fractions(self, solver):
        mu_cells = self.mu_layer[solver.layer]  # (n_cells, n_bins)
        tau = np.vstack([np.zeros(len(self.E)), np.cumsum(mu_cells * solver.m[:, None], axis=0)])
        att = np.exp(-tau)
        return (att[:-1] - att[1:]) @ self.w
