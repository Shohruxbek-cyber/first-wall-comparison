"""Tungsten grades for the alloy study (13 October 2026): a registry with provenance.

Every property carries a value, a unit, a source key (see REFS), the material condition it
was measured in, and a confidence label:
  "full text"  read in the paper (open-access copy in references/)
  "abstract"   taken from the abstract only (OpenAlex/Crossref record)
  "cited via"  reported second-hand by another paper that was read
  "assumption" a modelling assumption of this study, stated as such
A value of None means NOT SOURCED (TODO_SOURCE): the solver refuses to use it unless an
explicit, labelled proxy is given (Options.alloy_k_proxy).

What the solver uses: only the thermal properties.  An alloy enters the heat-conduction
model as a multiplier on the Tolias 2017 conductivity of pure W (and, if ever sourced, on
density and specific heat).  Recrystallization temperatures, DBTT and plasma-exposure
results are DIAGNOSTIC or literature-only quantities: they do not change the computed
temperatures.  No mechanism for potassium bubbles or oxide particles is modelled.

Temperatures are stored in kelvin; sources quote degrees Celsius (K = deg C + 273.15).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

C2K = 273.15
R_GAS = 8.314462618

# ------------------------------------------------------------------------------------------
# References (every entry was verified on 13 Oct 2026: DOI resolved through Crossref/OpenAlex,
# and "full text" entries were read from an open-access copy saved under references/)
# ------------------------------------------------------------------------------------------
REFS = {
    "Nogami2021": "Nogami, S., Hasegawa, A., Fukuda, M., Rieth, M., Reiser, J., Pintsuk, G. (2021). Mechanical properties of "
                  "tungsten: recent research on modified tungsten materials in Japan. J. Nucl. Mater. 543, 152506. "
                  "doi:10.1016/j.jnucmat.2020.152506 [full text: KIT postprint]",
    "Terentyev2022": "Terentyev, D., Rieth, M., Pintsuk, G., et al. (2022). Recent progress in the assessment of irradiation effects "
                     "for in-vessel fusion materials: tungsten and copper alloys. Nucl. Fusion 62, 026045. doi:10.1088/1741-4326/ac4062 "
                     "[full text: IOP open access]",
    "Ma2022a": "Ma, X., Feng, F., Zhang, X., et al. (2022). Effect of potassium bubbles on the thermal shock fatigue behavior of "
               "large-volume potassium-doped tungsten alloy. Nucl. Fusion 62, 126062. doi:10.1088/1741-4326/ac9a5b [full text: IOP open access]",
    "Ma2022b": "Ma, X., Zhang, X., Wang, T., Gao, Y., Yuan, Y., Cheng, L., Zhu, J., Lv, W., Lang, S., Ge, C., Yan, Q. (2023). "
               "Decreased surface blistering and deuterium retention in potassium-doped tungsten exposed to deuterium plasma "
               "following ion irradiation. Nucl. Fusion 63, 026013 (online 21 Nov 2022). doi:10.1088/1741-4326/aca48c "
               "[full text read online 14 Oct 2026 (IOP, CC BY)]",
    "Zhu2014": "Zhu, D.H., Chen, J.L., Zhou, Z.J., Yan, R., Ding, R. (2014). Influences of dispersed lanthanum oxide additive on "
               "the properties of tungsten-based plasma-facing material. Fusion Sci. Technol. 66, 337-342. doi:10.13182/FST13-738 "
               "[abstract only; closed access. W-1%La2O3 by resistance sintering under ultrahigh pressure; thermal conductivity "
               "compared with pure W but the number is not in the abstract]",
    "Ma2021": "Ma, X., Zhang, X., Wang, T., et al. (2021). Large-scale potassium-doped tungsten alloy with superior recrystallization "
              "resistance, ductility and strength induced by potassium bubbles. J. Nucl. Mater. 559, 153450. "
              "doi:10.1016/j.jnucmat.2021.153450 [cited via Ma2022a; not read]",
    "Linsmeier2017": "Linsmeier, Ch., et al. (2017). Development of advanced high heat flux and plasma-facing materials. "
                     "Nucl. Fusion 57, 092007. doi:10.1088/1741-4326/aa6f71 [full text, first part read online]",
    "Rieth2013": "Rieth, M., Dudarev, S.L., Gonzalez de Vicente, S.M., et al. (2013). Recent progress in research on tungsten "
                 "materials for nuclear fusion applications in Europe. J. Nucl. Mater. 432, 482-500. doi:10.1016/j.jnucmat.2012.08.018 "
                 "[metadata verified; text not accessible]",
    "Rieth2019": "Rieth, M., Doerner, R., Hasegawa, A., Ueda, Y., Wirtz, M. (2019). Behavior of tungsten under irradiation and plasma "
                 "interaction. J. Nucl. Mater. 519, 334-368. doi:10.1016/j.jnucmat.2019.03.035 [metadata verified; text not accessible]",
    "Shi2024": "Shi, J., et al. (2024). Effects of minor rhenium additions on the thermal properties and recrystallization temperature "
               "of tungsten alloy. Nucl. Mater. Energy, 101609. doi:10.1016/j.nme.2024.101609 [abstract]",
    "Lied2020": "Lied, P., et al. (2020). Comparison of K-doped and pure cold-rolled tungsten sheets: tensile properties and "
                "brittle-to-ductile transition temperatures. J. Nucl. Mater. 544, 152664. doi:10.1016/j.jnucmat.2020.152664 [abstract]",
    "Bonnekoh2020": "Bonnekoh, C., et al. (2020). The brittle-to-ductile transition in cold-rolled tungsten sheets: on the loss of "
                    "room-temperature ductility after annealing and the phenomenon of 45 deg embrittlement. Int. J. Refract. Met. Hard "
                    "Mater., 105347. doi:10.1016/j.ijrmhm.2020.105347 [abstract]",
    "Shah2020": "Shah, V., van Dommelen, J.A.W., Altstadt, E., Das, A., Geers, M.G.D. (2020). Brittle-ductile transition temperature "
                "of recrystallized tungsten following exposure to fusion relevant cyclic high heat load. J. Nucl. Mater. 541, 152416. "
                "doi:10.1016/j.jnucmat.2020.152416 [abstract]",
    "Shah2021": "Shah, V., van Dommelen, J.A.W., Heijkoop, S.E.S., Oude Vrielink, M.A., Geers, M.G.D. (2021). A numerical model for "
                "the recrystallization kinetics of tungsten monoblocks under cyclic heat loads. Fusion Eng. Des. 173, 112827. "
                "doi:10.1016/j.fusengdes.2021.112827 [abstract]",
    "Ciucani2019": "Ciucani, U.M., Thum, A., Devos, C., Pantleon, W. (2019). Recovery and recrystallization kinetics of differently "
                   "rolled, thin tungsten plates in the temperature range from 1325 to 1400 deg C. Nucl. Mater. Energy 20, 100701. "
                   "doi:10.1016/j.nme.2019.100701 [abstract]",
    "Tanure2019": "Tanure, L., et al. (2019). Evolution of microstructure, texture and grain boundary character distribution of "
                  "potassium doped tungsten fibers annealed at variable temperatures. J. Phys.: Conf. Ser. 1270, 012038. "
                  "doi:10.1088/1742-6596/1270/1/012038 [abstract]",
    "Du2023": "Du, J., et al. (2023). Properties of potassium doped and tantalum containing tungsten after heavy ion irradiation. "
              "Crystals 13, 951. doi:10.3390/cryst13060951 [abstract]",
    "Ding2016": "Ding, X.-Y., Luo, L.-M., Chen, H.-Y., et al. (2016). Chemical synthesis and oxide dispersion properties of "
                "strengthened tungsten via spark plasma sintering. Materials 9, 879. doi:10.3390/ma9110879 [full text: MDPI]",
    "Ding2020": "Ding, X., Xu, Q., Zhu, X., et al. (2020). Microstructure evolution and effect on deuterium retention in oxide "
                "dispersion strengthened tungsten during He+ irradiation. Nucl. Eng. Technol. 52, 2860-2866. "
                "doi:10.1016/j.net.2020.05.022 [abstract]",
    "Sun2022": "Sun, Y., Wang, S., Li, C., et al. (2022). Dependence of deuterium retention and surface blistering on deuterium "
               "plasma exposure temperature and fluence in lanthanum oxide doped tungsten. Nucl. Mater. Energy 32, 101217. "
               "doi:10.1016/j.nme.2022.101217 [abstract]",
    "Yao2021": "Yao, G., Liu, X., Zhao, Z., et al. (2021). Excellent performance of W-Y2O3 composite via powder process improvement "
               "and Y2O3 refinement. Mater. Des. 212, 110249. doi:10.1016/j.matdes.2021.110249 [abstract]",
    "Zhang2021": "Zhang, X., et al. (2021). Surface blistering and deuterium retention behaviors in pure and ZrC-doped tungsten "
                 "exposed to deuterium plasma. Nucl. Fusion. doi:10.1088/1741-4326/abdfda [abstract]",
    "Nogami2026": "Nogami, S., Matsuta, N., Miura, K., Nakabayashi, S., Takida, T., Hasegawa, A. (2026). Modifying tungsten for "
                  "expanded operation temperature range of plasma facing components. Plasma Fusion Res. 21, 1405047. "
                  "doi:10.1585/pfr.21.1405047 [abstract (J-STAGE): K-doping and La2O3 particles, or Re/Ta solid solution, widen the "
                  "window between DBTT and recrystallization; solid-solution alloying lowers thermal diffusivity and conductivity; "
                  "large dispersed particles such as La2O3 can reduce them too and introduce anisotropy]",
    "Fukuda2018": "Fukuda, M., Hasegawa, A., Nogami, S. (2018). Thermal properties of pure tungsten and its alloys for fusion "
                  "applications. Fusion Eng. Des. 132, 1-6. doi:10.1016/j.fusengdes.2018.04.117 [cited via Nogami2021 and Ma2022a; not read]",
    "Pintsuk2010": "Pintsuk, G., Uytdenhouwen, I. (2010). Thermo-mechanical and thermal shock characterization of potassium doped "
                   "tungsten. Int. J. Refract. Met. Hard Mater. 28(6), 661-668. doi:10.1016/j.ijrmhm.2010.03.001 "
                   "[metadata verified via Crossref 13 Oct 2026; result cited via Ma2022a; text not read]",
    "Huang2019": "Huang, B., et al. (2019). Design of highly thermal-shock resistant tungsten alloys with nanoscaled intra- and "
                 "inter-type K bubbles. J. Alloys Compd. 782, 149-159. doi:10.1016/j.jallcom.2018.12.168 [metadata verified via OpenAlex 13 Oct 2026; result cited via Ma2022a; text not read]",
    "Tolias2017": "Tolias, P. (2017). Analytical expressions for thermophysical properties of solid and liquid tungsten relevant for "
                  "fusion applications. Nucl. Mater. Energy 13, 42-57 (arXiv:1703.06302v4). doi:10.1016/j.nme.2017.08.002 (journal DOI as printed in Reza 2022 ref. 68) [full text]",
    "Raffray2005": "Raffray, A.R., and the HAPL Team (2005). Threats, design limits and design windows for laser IFE dry wall "
                   "chambers. J. Nucl. Mater. 347, 178-191. doi:10.1016/j.jnucmat.2005.08.015 [full text; "
                   "Table 2: W vapour pressure and heat of vaporization used by library.w_vapor]",
    "Repo": "This repository, data/README.md: every photon (XCOM, Henke) and ion (ARIES spectra, SRIM/TRIM) input with its "
            "source; elemental-W tables are used for every grade [repository data, not a publication]",
    "GA2020": "Gonzalez-Arrabal, R., Rivera, A., Perlado, J.M. (2020). Limitations for tungsten as plasma facing material in the "
              "diverse scenarios of the European inertial confinement fusion facility HiPER. Matter Radiat. Extremes 5, 055201. "
              "doi:10.1063/5.0010954 [full text]",
}


@dataclass(frozen=True)
class Prop:
    value: object  # float | tuple | str | None (None = TODO_SOURCE)
    unit: str
    source: str  # key(s) in REFS, comma separated
    condition: str = ""
    confidence: str = "full text"
    note: str = ""

    @property
    def sourced(self) -> bool:
        return self.value is not None and bool(self.source)


@dataclass
class Grade:
    key: str
    name: str
    dopant: Prop
    k_factor: Prop  # multiplier on Tolias 2017 k(T); used by the solver
    rho_factor: Prop
    cp_factor: Prop
    T_rx: Prop  # recrystallization onset used for the diagnostics, K
    T_rx_other: tuple = ()  # other sourced values, for the report
    dbtt: dict = field(default_factory=dict)  # label -> Prop, K
    thermal_shock: Prop = Prop(None, "GW/m^2", "", "", "none")
    plasma: tuple = ()  # literature-only Props
    he_threshold_factor: Prop = Prop(None, "-", "", "", "none",
                                     "alloy-specific pulsed damage-fluence threshold relative to the GA2020 range 1e18-1e19 m^-2: not sourced")
    Tm: Prop = Prop(None, "K", "", "", "none", "TODO_SOURCE")
    L_f: Prop = Prop(None, "J/mol", "", "", "none", "TODO_SOURCE")
    vapor: Prop = Prop(None, "-", "", "", "none", "TODO_SOURCE")
    deposition: Prop = Prop(None, "-", "", "", "none", "TODO_SOURCE")
    emissivity: Prop = Prop(None, "-", "", "radiative cooling is not modelled (front face adiabatic except evaporation)", "none",
                            "not needed by the model; no value sourced")

    def diffusivity_factor(self):
        """Derived: k_factor / (rho_factor * cp_factor), None if any of them is unsourced."""
        vals = (self.k_factor.value, self.rho_factor.value, self.cp_factor.value)
        if any(v is None or not isinstance(v, (int, float)) for v in vals):
            return None
        return vals[0] / (vals[1] * vals[2])
    runnable: bool = True
    notes: str = ""


def _K(deg_c):
    return deg_c + C2K


# ------------------------------------------------------------------------------------------
# Activation energy for recrystallization (used by the Arrhenius-equivalent time diagnostic)
# ------------------------------------------------------------------------------------------
Q_RX = Prop((322e3, 350e3), "J/mol", "Shah2021",
            "JMAK recrystallization model fitted to differently processed tungsten monoblocks under cyclic heat loads",
            "abstract", "range quoted in the abstract; the mid value 336 kJ/mol is used, the ends as sensitivity")

# ------------------------------------------------------------------------------------------
# Grades
# ------------------------------------------------------------------------------------------
GRADES = {
    "W_pure": Grade(
        key="W_pure", name="pure W (baseline: Tolias 2017 properties)",
        dopant=Prop(0.0, "ppm", "Tolias2017", "pure polycrystalline W; no commercial grade chosen", "full text"),
        k_factor=Prop(1.0, "-", "Tolias2017", "the baseline fits themselves", "full text"),
        rho_factor=Prop(1.0, "-", "Tolias2017", "", "full text"),
        cp_factor=Prop(1.0, "-", "Tolias2017", "", "full text"),
        T_rx=Prop(_K(1250), "K", "Nogami2021", "hot-rolled plate, high deformation ratio (H); 1 h isochronal anneal; hardness drop",
                  "full text", "1200-1300 deg C by KAM mapping in the same study"),
        T_rx_other=(Prop(_K(1400), "K", "Nogami2021", "swaged rod (H), 1 h anneal", "full text"),
                    Prop(_K(1300), "K", "Linsmeier2017", "'recrystallization at ca. 1300 deg C' (rolled/forged pure W)", "full text"),
                    Prop((1300.0, 1800.0), "K", "GA2020", "range used in the full-results document (GA2020 Sec. III.A citing its refs 40, 41)", "full text")),
        dbtt={
            "tensile": Prop(_K(150), "K", "Nogami2021", "hot-rolled plate (H), tensile, total elongation -> 0", "full text"),
            "bending_5pct": Prop(_K(150), "K", "Terentyev2022", "W bar (IGP), L-T, 5 % flexural strain criterion; +/-25 deg C", "full text"),
            "fracture_toughness": Prop(_K(400), "K", "Terentyev2022", "non-irradiated W, T-L orientation", "full text"),
            "charpy": Prop(_K(550), "K", "Nogami2021", "KLST Charpy, plate (H), L-S", "full text",
                           "Charpy DBTT exceeds tensile DBTT by several hundred K (strain rate, notch)"),
            "charpy_rod": Prop(_K(800), "K", "Linsmeier2017", "pure W rod, Charpy; +/-50 deg C", "full text"),
            "recrystallized_tensile_shift": Prop((100.0, 300.0), "K", "Nogami2021", "after 1 h at 1500 or 2300 deg C", "full text"),
            "recrystallized_small_punch": Prop(_K(400), "K", "Shah2020", "recrystallized monoblock: brittle below 400 deg C", "abstract"),
            "irradiated_bending": Prop(_K(550), "K", "Terentyev2022", "W bar after ~1 dpa at 1100 deg C; > 600 deg C after 1 dpa at 600 deg C", "full text"),
            "irradiated_tensile_shift": Prop(600.0, "K", "Nogami2021", "increase of tensile DBTT by neutron irradiation (conditions in the paper)", "full text"),
        },
        thermal_shock=Prop((None, 0.37), "GW/m^2", "Huang2019", "SPS pure W: cracking below 0.37 GW/m^2 (5 ms, 100 cycles, RT)", "cited via"),
        plasma=(Prop((1e18, 1e19), "m^-2", "GA2020", "pulsed damage-fluence threshold (Renk 2012; Dutta 2014)", "full text"),),
        Tm=Prop(3695.0, "K", "Tolias2017", "", "full text"),
        L_f=Prop(52.3e3, "J/mol", "Tolias2017", "", "full text"),
        vapor=Prop("Raffray 2005 Table 2 p_sat(T) and heat of vaporization (library.w_vapor)", "-", "Raffray2005", "", "full text"),
        deposition=Prop("elemental W: XCOM/Henke photon attenuation, SRIM/TRIM ion ranges", "-", "Repo", "data/README.md", "full text"),
        notes="The baseline of the whole study; every published number was computed with it.",
    ),
    "W_K": Grade(
        key="W_K", name="K-doped W (potassium-bubble dispersion strengthened)",
        dopant=Prop((30.0, 92.0), "ppm K", "Nogami2021,Shi2024,Du2023,Ma2022a",
                    "~30 ppm (Nogami plates), 40-50 ppm (Shi, W-Re-K), 82 ppm (Du), ~92 ppm (Ma plate)", "full text",
                    "the range of the sources read"),
        k_factor=Prop(1.0, "-", "Nogami2021,Ma2022a,Fukuda2018",
                      "rolled plates: 'K-doping showed almost no effect on thermal conductivity' (Nogami, Fig. 5 citing Fukuda 2018); "
                      "'almost the same ... below 1 %' vs ITER-W, laser flash, RT to high T (Ma 2022a, Fig. 5)", "full text",
                      "measured difference below 1 %; a swaged K-doped rod with residual pores was lower (Nogami)"),
        rho_factor=Prop(1.0, "-", "Ma2022a", "relative density ~99.5 % (Archimedes)", "assumption",
                        "ppm-level doping: density and specific heat taken as pure W"),
        cp_factor=Prop(1.0, "-", "", "", "assumption", "no measurement found; taken as pure W"),
        T_rx=Prop(_K(1350), "K", "Nogami2021", "hot-rolled plate (H), ~30 ppm K; 1 h isochronal anneal; hardness drop", "full text",
                  "+100 deg C over the pure W plate of the same route; 1300-1400 deg C by KAM"),
        T_rx_other=(Prop(_K(1600), "K", "Nogami2021", "swaged K-doped rod (H), 1 h anneal (+200 deg C over the pure W rod)", "full text"),
                    Prop(_K(1588), "K", "Shi2024", "W-0.5%Re-K (40-50 ppm K): 'elevated RCT of approximately 1588 deg C'", "abstract"),
                    Prop((_K(1900), _K(2100)), "K", "Tanure2019", "K-doped W fibres: primary recrystallization up to 1900 deg C, abnormal grain growth 1900-2100 deg C (annealing time not in the abstract)", "abstract")),
        dbtt={
            "tensile": Prop(_K(50), "K", "Nogami2021", "plate (H), tensile", "full text", "100 deg C below the pure W plate"),
            "bending_5pct": Prop(_K(75), "K", "Terentyev2022", "K-doped rolled plate 7 mm (Tohoku); +/-25 deg C", "full text"),
            "bending_5pct_KIT": Prop(_K(150), "K", "Terentyev2022", "K-doped cold-rolled plate 1 mm (KIT), L-S; +/-50 deg C", "full text"),
            "charpy": Prop(_K(350), "K", "Nogami2021", "KLST Charpy, plate (H)", "full text", "200 deg C below the pure W plate"),
            "charpy_rod": Prop(_K(1000), "K", "Linsmeier2017", "WVM rod, Charpy, 'around 1000 deg C'", "full text"),
            "thin_foil": Prop(_K(-100), "K", "Lied2020", "50 um cold-rolled K-doped foil: BDTT below -100 deg C", "abstract"),
            "recrystallized_tensile_shift": Prop((100.0, 300.0), "K", "Nogami2021", "after 1 h at 1500 or 2300 deg C", "full text"),
            "irradiated_bending": Prop(_K(325), "K", "Terentyev2022", "7 mm plate after ~1 dpa at 600 or 1100 deg C; +/-50 deg C", "full text"),
            "irradiated_tensile_shift": Prop(500.0, "K", "Nogami2021", "increase of tensile DBTT by neutron irradiation", "full text"),
        },
        thermal_shock=Prop((0.44, 0.55), "GW/m^2", "Ma2022a", "rolled W-K plate, 1 ms pulses, 100 shots at RT: cracking threshold", "full text",
                           "> 0.95 GW/m^2 at base temperatures up to 600 deg C for forged K-doped rods (Pintsuk2010, cited via Ma2022a)"),
        plasma=(Prop("D plasma (60 eV, 500 K, 5.76e25 D/m^2): undamaged K-doped W retained more D (13.6 vs 11.8e20 m^-2) and "
                     "blistered more than pure W; after Fe-ion pre-damage (0.05, 0.5 dpa) its retention fell by 35-39 % against "
                     "17-23 % in pure W, so the pre-damaged K-doped W retained less; no significant blistering advantage (abstract)",
                     "-", "Ma2022b", "15 mm hot-rolled K-doped plate vs pure W", "full text",
                     "the paper is inconsistent on blistering: its abstract reports no significant advantage, its conclusion (b) claims one"),),
        Tm=Prop("pure W", "K", "", "", "assumption", "ppm-level K: melting point taken as pure W; the potassium in the bubbles is not modelled"),
        L_f=Prop("pure W", "J/mol", "", "", "assumption", "taken as pure W"),
        vapor=Prop("pure W", "-", "", "", "assumption", "vapour pressure and heat of vaporization of pure W (Raffray 2005); no alloy value sourced"),
        deposition=Prop("elemental W", "-", "", "", "assumption", "ppm-level K does not change the photon or ion stopping tables"),
        notes="Thermal properties equal to pure W within the measured 1 %: the heat-conduction results are those of pure W; "
              "the alloy differs in the recrystallization and DBTT diagnostics only.",
    ),
    "W_ODS_La2O3": Grade(
        key="W_ODS_La2O3", name="W-1 wt% La2O3 (oxide dispersion strengthened; Plansee grade WL10)",
        dopant=Prop(1.0, "wt% La2O3", "Nogami2021,Linsmeier2017", "W-3%Re-1%La2O3 plate (Nogami); WL10 (Linsmeier)", "full text"),
        k_factor=Prop(None, "-", "Ding2016,Nogami2026,Zhu2014", "qualitative statements only, no number", "none",
                      "TODO_SOURCE: no measured k(T) of W-1%La2O3 was found in an accessible source. Qualitative: sintered "
                      "W-1%La2O3 had a room-temperature conductivity below the >140 W/m/K of W-Nd2O3/CeO2 (Ding2016 citing Liu 2012); "
                      "'large dispersed particles such as La2O3 can reduce thermal conductivity and introduce anisotropy' "
                      "(Nogami2026, abstract). Zhu 2014 measured the conductivity of sintered W-1%La2O3 "
                      "against pure W but the number is not in the abstract (closed access). Run only with an explicit proxy "
                      "factor, reported as a sensitivity band."),
        rho_factor=Prop(1.0, "-", "", "", "assumption", "1 wt% oxide: density taken as pure W (rule-of-mixtures change < 1 %)"),
        cp_factor=Prop(1.0, "-", "", "", "assumption", "taken as pure W"),
        T_rx=Prop(_K(1550), "K", "Nogami2021", "W-3%Re-1%La2O3 hot-rolled plate (L); 1 h isochronal anneal; hardness drop", "full text",
                  "CONTAINS 3 % Re: the W-3%Re (L) plate without La2O3 gave 1400 deg C, so the La2O3 contribution is about +150 deg C; "
                  "no value for W-1%La2O3 without Re was found. Linsmeier 2017: La2O3 'suppression of recrystallization' (no number)."),
        T_rx_other=(),
        dbtt={
            "tensile": Prop(_K(50), "K", "Nogami2021", "W-3%Re-1%La2O3 plate (L), tensile", "full text", "contains Re"),
            "charpy": Prop(_K(550), "K", "Nogami2021", "W-3%Re-1%La2O3 plate (L), KLST Charpy", "full text", "contains Re; low absorbed energy"),
            "charpy_WL10": Prop(_K(950), "K", "Linsmeier2017", "WL10, Charpy, estimated; +/-50 deg C (pure W 800 +/- 50)", "full text",
                                "the same review reports another test in which WL10 had a LOWER DBTT than pure W"),
            "tensile_other_ODS": Prop((_K(500), _K(550)), "K", "Ding2016", "W-1%CeO2 by SPS (not La2O3)", "full text", "different oxide; for orientation only"),
        },
        plasma=(Prop("larger D blisters and higher D retention than pure W at 500 K", "-", "Sun2022",
                     "W-1%La2O3, 40 eV D plasma, 400-600 K, 3.6e24-1.4e25 D/m^2", "abstract"),
                Prop("loops (8.5 nm at 1e20 He/m^2), no He bubbles observed in irradiations up to 5e21 He/m^2; D retention rises after He pre-irradiation",
                     "-", "Ding2020", "W-1%La2O3, 5 keV He+, 600 deg C", "abstract")),
        Tm=Prop("pure W matrix", "K", "", "", "assumption",
                "the oxide phase (1 wt%) is not modelled: its melting or decomposition below the W melting point is not represented (TODO_SOURCE)"),
        L_f=Prop("pure W", "J/mol", "", "", "assumption", "taken as pure W"),
        vapor=Prop("pure W", "-", "", "", "assumption", "evaporation or decomposition of the oxide particles is not represented (TODO_SOURCE)"),
        deposition=Prop("elemental W", "-", "", "", "assumption", "1 wt% La2O3 changes the mean atomic number and the density by about 1 %: neglected"),
        notes="No pure W-1%La2O3 thermal data found: the simulation uses the pure-W properties scaled by a PROXY factor "
              "(0.8-1.0) and reports the band; the recrystallization temperature is that of a Re-containing plate.",
    ),
    "W_ODS_Y2O3": Grade(
        key="W_ODS_Y2O3", name="W-Y2O3 (oxide dispersion strengthened)",
        dopant=Prop(None, "wt% Y2O3", "", "", "none", "1-2 wt% in the literature read (Ding2016 citing Liu 2012; Yao2021 content not in the abstract)"),
        k_factor=Prop(None, "-", "", "", "none", "TODO_SOURCE: sintered W-1%Y2O3 below 140 W/m/K at RT (Ding2016 citing Liu 2012); no k(T)"),
        rho_factor=Prop(None, "-", "", "", "none"), cp_factor=Prop(None, "-", "", "", "none"),
        T_rx=Prop(None, "K", "", "", "none", "TODO_SOURCE"),
        dbtt={},
        plasma=(Prop("improved helium-ion irradiation resistance (qualitative)", "-", "Yao2021", "W-Y2O3 by wet chemistry + SPS", "abstract"),),
        runnable=False,
        notes="Literature-only entry: not run (no sourced thermal data).",
    ),
    "W_3Re": Grade(
        key="W_3Re", name="W-3 wt% Re (solid-solution alloy; context only)",
        dopant=Prop(3.0, "wt% Re", "Nogami2021", "", "full text"),
        k_factor=Prop(None, "-", "", "", "none", "TODO_SOURCE: 'thermal conductivity decreased with increase in amount of Re, especially "
                                               "at low temperatures' (Nogami2021 Fig. 5 citing Fukuda2018); no number read"),
        rho_factor=Prop(None, "-", "", "", "none"), cp_factor=Prop(None, "-", "", "", "none"),
        T_rx=Prop(_K(1500), "K", "Nogami2021", "W-3%Re plate (H), 1 h anneal", "full text"),
        dbtt={"tensile": Prop(_K(100), "K", "Nogami2021", "plate (H)", "full text"),
              "charpy": Prop(_K(450), "K", "Nogami2021", "plate (H), KLST", "full text"),
              "irradiated_tensile_shift": Prop(400.0, "K", "Nogami2021", "below 400 deg C after neutron irradiation", "full text")},
        runnable=False,
        notes="Not run: conductivity reduction not quantified in the sources read; Re also transmutes and hardens under neutrons (Nogami2021).",
    ),
}

RUNNABLE = [k for k, g in GRADES.items() if g.runnable]


# ------------------------------------------------------------------------------------------
# What the solver reads
# ------------------------------------------------------------------------------------------
def conductivity_factor(opt) -> float:
    """Multiplier on the pure-W k(T) for Options.w_material.  Exactly 1.0 for W_pure (no wrapper is
    applied by the caller).  Unsourced grades need Options.alloy_k_proxy; a warning records it."""
    key = getattr(opt, "w_material", "W_pure")
    if key not in GRADES:
        raise KeyError(f"unknown tungsten grade {key!r}; known: {list(GRADES)}")
    g = GRADES[key]
    if not g.runnable:
        raise ValueError(f"{key} is a literature-only entry (no sourced thermal data): {g.notes}")
    proxy = getattr(opt, "alloy_k_proxy", None)
    if g.k_factor.value is not None:
        if proxy is not None and float(proxy) != float(g.k_factor.value):
            warnings.warn(f"{key}: alloy_k_proxy={proxy} overrides the sourced conductivity factor {g.k_factor.value} (sensitivity run)")
            return float(proxy)
        return float(g.k_factor.value)
    if proxy is None:
        raise ValueError(f"{key}: thermal conductivity is not sourced ({g.k_factor.note}); pass alloy_k_proxy=<factor> to run it as a labelled proxy")
    warnings.warn(f"{key}: conductivity not sourced; running with the PROXY factor {proxy} on pure-W k(T)")
    return float(proxy)


def thermal_status(opt) -> str:
    key = getattr(opt, "w_material", "W_pure")
    g = GRADES[key]
    proxy = getattr(opt, "alloy_k_proxy", None)
    if g.k_factor.value is not None and proxy is None:
        return f"SOURCED k factor {g.k_factor.value} ({g.k_factor.source})"
    return f"PROXY k factor {proxy} (not sourced: {g.k_factor.source or 'no source'})"


def recrystallization_thresholds(key: str, delta_K: float = 100.0):
    """(T_rx - delta, T_rx, T_rx + delta) in K, or None if T_rx is not sourced."""
    T = GRADES[key].T_rx.value
    if T is None:
        return None
    return (T - delta_K, T, T + delta_K)


def q_rx_mid() -> float:
    lo, hi = Q_RX.value
    return 0.5 * (lo + hi)


def dbtt_regime(T_K: float, key: str) -> dict:
    """Compare a (pre-shot) temperature with the grade's DBTT values; an indicator, not a fracture model."""
    out = {}
    for label, p in GRADES[key].dbtt.items():
        v = p.value
        if v is None or isinstance(v, str):
            continue
        if isinstance(v, tuple):
            out[label] = "shift (range)"
            continue
        if "shift" in label:
            continue
        out[label] = ("ductile (above DBTT)" if T_K > v + 50 else "brittle (below DBTT)" if T_K < v - 50 else "within +/-50 K of DBTT")
    return out


def property_rows():
    """Every property of every grade, for the report: list of dicts."""
    rows = []
    for key, g in GRADES.items():
        for name, p in (("dopant", g.dopant), ("k_factor", g.k_factor), ("rho_factor", g.rho_factor), ("cp_factor", g.cp_factor),
                        ("T_rx", g.T_rx), ("thermal_shock", g.thermal_shock), ("he_threshold_factor", g.he_threshold_factor),
                        ("Tm", g.Tm), ("L_f", g.L_f), ("vapor", g.vapor), ("deposition", g.deposition), ("emissivity", g.emissivity)):
            rows.append({"grade": key, "property": name, "value": p.value, "unit": p.unit, "source": p.source,
                         "condition": p.condition, "confidence": p.confidence, "note": p.note, "sourced": p.sourced})
        a = g.diffusivity_factor()
        rows.append({"grade": key, "property": "alpha_factor (derived)", "value": a, "unit": "-", "source": g.k_factor.source if a is not None else "",
                     "condition": "k_factor / (rho_factor * cp_factor)", "confidence": g.k_factor.confidence if a is not None else "none",
                     "note": "derived; rho and cp factors are assumptions for the alloys" if a is not None else "TODO_SOURCE (k not sourced)",
                     "sourced": a is not None and bool(g.k_factor.source)})
        for i, p in enumerate(g.T_rx_other):
            rows.append({"grade": key, "property": f"T_rx_other_{i}", "value": p.value, "unit": p.unit, "source": p.source,
                         "condition": p.condition, "confidence": p.confidence, "note": p.note, "sourced": p.sourced})
        for label, p in g.dbtt.items():
            rows.append({"grade": key, "property": f"dbtt_{label}", "value": p.value, "unit": p.unit, "source": p.source,
                         "condition": p.condition, "confidence": p.confidence, "note": p.note, "sourced": p.sourced})
        for i, p in enumerate(g.plasma):
            rows.append({"grade": key, "property": f"plasma_{i}", "value": p.value, "unit": p.unit, "source": p.source,
                         "condition": p.condition, "confidence": p.confidence, "note": p.note, "sourced": p.sourced})
    return rows


def unsourced():
    return [r for r in property_rows() if not r["sourced"]]
