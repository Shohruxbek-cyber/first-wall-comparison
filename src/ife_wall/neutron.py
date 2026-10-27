"""Volumetric neutron heating of the wall (revision of 20 Oct 2026); off by default.

`scenarios.w_case` and `scenarios.li_case` add this source only when `Options.neutron_heating`
is True, so with the option off every earlier result is unchanged.

Source of the magnitude: Sharafat et al. (2005), Sec. 4, for the NRL 154 MJ target in a 6.5 m
chamber: "125 ns later neutrons deposit about 800 MW/cm3 for over 5 ns" in solid tungsten
armor, i.e. about 800 MW/cm^3 x 5 ns = 4 J/cm^3 per shot.  That value is scaled with the neutron
fluence at the wall, Y/154 MJ x (6.5 m/R)^2, and deposited uniformly per unit mass in the tungsten
(per unit volume at the initial density).  No value for the steel was found: the steel receives a
stated fraction of the tungsten value per unit volume (Options.neutron_steel_factor), and the
lithium film none (it is at most 1 mm thick).  The heating arrives as a square pulse of 5 ns at
the time of flight of 14.1 MeV neutrons to the wall.
"""
from __future__ import annotations

import numpy as np

from .source_term import PulsedSource, SquarePulse

E_W_REF = 800e6 * 1e6 * 5e-9  # J/m^3 per shot in W: 800 MW/cm^3 for 5 ns (Sharafat et al. 2005)
YIELD_REF, R_REF = 154e6, 6.5
MN_MEV, EN_MEV, C = 939.565, 14.1, 2.99792458e8


def fluence_scale(scen):
    """Neutron fluence at the wall of a scenario relative to the reference target and chamber (154 MJ at 6.5 m),
    Y/154 MJ x (6.5 m/R)^2: exactly 1 for Demo."""
    return scen.yield_J / YIELD_REF * (R_REF / scen.R) ** 2


def neutron_arrival(R):
    """Time of flight of 14.1 MeV neutrons over R [s]."""
    gamma = 1.0 + EN_MEV / MN_MEV
    return R / (C * np.sqrt(1.0 - 1.0 / gamma ** 2))


class NeutronHeatingSource(PulsedSource):
    """Energy density e_layer [J/m^3 per shot] per layer, deposited uniformly per unit mass
    (e_layer / rho_layer at the initial temperature) in each cell of that layer."""

    def __init__(self, e_by_layer, rho_by_layer, t_start, duration=5e-9):
        super().__init__(0.0, SquarePulse(t_start, duration))
        self.h = np.asarray(e_by_layer, float) / np.asarray(rho_by_layer, float)  # J/kg per shot, per layer

    def fractions(self, solver):
        e = self.h[np.asarray(solver.layer)] * np.asarray(solver.m)
        self.fluence = float(e.sum())  # J/m^2 per shot on the current mesh
        return e / self.fluence if self.fluence > 0 else np.zeros(len(e))


def neutron_source(scen, mesh_layers_elements, T_init, materials, steel_factor):
    """The source for a case: tungsten layers at the scaled Sharafat value, steel ('Fe') at
    steel_factor times it per unit volume, other layers (Li) at zero."""
    e_W = E_W_REF * scen.yield_J / YIELD_REF * (R_REF / scen.R) ** 2
    e = [e_W if el == "W" else (steel_factor * e_W if el == "Fe" else 0.0) for el in mesh_layers_elements]
    rho = [float(mat.rho(np.array(float(T_init)))) for mat in materials]
    return NeutronHeatingSource(e, rho, neutron_arrival(scen.R))
