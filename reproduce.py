"""Rerun the calculations behind the paper, in stages.

    python reproduce.py --list              list the stages
    python reproduce.py STAGE [STAGE ...]   run these stages
    python reproduce.py all                 run every stage, in the order of --list
    python reproduce.py --check STAGE ...   compare results/STAGE/ with the committed version (git)

Each stage runs its commands in a fresh work copy, work/STAGE/ (the run scripts write into the
folders runs/ and results/ of that copy, as listed in Supplementary Section S8), takes the small
outputs into results/STAGE/, and prints how long it took.  Outputs of other stages that a stage
reads are copied in from results/ first, so every stage can run on its own.  At most --jobs
calculations (default 4) run at the same time.  The third-party data tables must be installed
first (python fetch_data.py --check).
"""
import csv
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WORK = ROOT / "work"
CODE = ["src", "scripts", "validation", "tests", "data", "pyproject.toml"]
KEEP_EXT = {".json", ".csv", ".md", ".txt"}
MAX_BYTES = 2_000_000


def cmd(name, *args, w=1, after=()):
    return {"name": name, "args": list(args), "w": w, "after": list(after)}


def li_scan(scen, *opts):
    tag = "_".join(o.replace("=", "-") for o in opts)
    return cmd(f"li_{scen}_{tag}", "scripts/run_li_scan.py", scen, *opts)


W_SINGLE_VARIANTS = ["alpha_evap=0.1", "xray_tau=1e-10", "xray_tau=1e-08", "xray_tau=5e-08", "normalize_ions=false",
                     "k_scale=0.85", "k_scale=1.15", "w_lv=raffray_slope", "depth_mode=csda", "stretch=false"]
LI_VARIANTS = [["T_li=500"], ["T_li=800"], ["T_li=600", "alpha_evap=0.1"], ["T_li=600", "li_lv=handbook_147kJmol"],
               ["T_li=600", "xray_tau=1e-10"], ["T_li=600", "xray_tau=5e-08"]]
LI_MULTI = [("Demo", 5, "Demo_d17.8um"), ("Demo", 6, "Demo_d31.6um"), ("Demo", 7, "Demo_d56.2um"), ("Demo", 8, "Demo_d100um"),
            ("Demo", 12, "Demo_d1000um"), ("Prototype", 4, "Prototype_d10um"), ("Prototype", 5, "Prototype_d17.8um"),
            ("Prototype", 6, "Prototype_d31.6um"), ("Prototype", 7, "Prototype_d56.2um")]
LI_DEMO_RUNS = [f"li_multi_{t}" for s, _, t in LI_MULTI if s == "Demo"]
RENEWAL = [(i, t, f) for f in ("0.03", "0.3") for i, t in ((6, "d31.6um"), (7, "d56.2um"), (8, "d100um"))]
HE_BATCH = ["Demo:Em_He_low:600", "Demo:Em_He_high:300", "Demo:Eb_low:600", "Demo:Eb_high:600", "Demo:xi_low:300",
            "Demo:dpa_arc:300", "Demo:surface_0p1:300", "Demo:T_m50:300", "Demo:T_p50:300", "Demo:level2:400"]
VERIFY_HE = ["verify_he_1_diffusion", "verify_he_2_trapping", "verify_he_3_conservation", "verify_he_3b_numpy_crosscheck",
             "verify_he_4_sources", "verify_he_5_convergence", "verify_he_6_benchmarks", "verify_he_7_level2"]


def copy_verify_checkpoint(work):
    """The neutron-heated Demo runs continue the 600-shot run of the verification stage (Table C.2 row)."""
    src = work / "runs" / "w_multi" / "verify_2026-10-05" / "Demo_checkpoint.npz"
    for f in ("0.5", "1"):
        dst = work / "results" / "neutron" / f"w_multi_Demo_steel{f}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / "Demo_checkpoint.npz")


def stage_list():
    S = []

    S.append(dict(name="verification", title="Verification",
                  produces="Supplementary Tables S1, S2; Supplementary Fig. S1",
                  contains="analytical and conservation tests (pytest) and their curves, the grid and time-step "
                           "convergence study on a 600-shot Demo run, and the surface-cell check",
                  steps=[cmd("record_he_Prototype", "scripts/record_he_thermal.py", "Prototype", "converged",
                             "--Tback=converged"),
                         cmd("pytest", "-m", "pytest", "-q", "-p", "no:cacheprovider", after=["record_he_Prototype"]),
                         cmd("plot_verification", "scripts/plot_verification.py", after=["pytest"]),
                         cmd("demo600", "scripts/run_w_multi.py", "Demo", "600", "--out=runs/w_multi/verify_2026-10-05"),
                         cmd("convergence", "scripts/convergence_study.py", after=["demo600"]),
                         cmd("extrapolate", "scripts/convergence_extrapolate.py", after=["convergence"]),
                         cmd("T0_vs_Ts", "scripts/diag_T0_vs_Ts.py"),
                         cmd("li_dx0", "scripts/run_li_scan.py", "Demo", "dx0=1e-9")],
                  outputs=["runs/verification/results.json", "runs/verification/verification_curves.npz",
                           "runs/w_multi/verify_2026-10-05/*", "results/revision/convergence/*",
                           "results/revision/T0_vs_Ts/*", "runs/li_scan/T600K_dx0-1e-9/*"],
                  inputs=[("vapor_shielding", "runs/li_scan/vapour_shielding_bound_Demo.json"),
                          ("tungsten_repeated", "results/revision/converged_state/*"),
                          ("tungsten_repeated", "runs/w_multi/baseline/*"),
                          ("lithium_repeated", "results/revision/steel_behind_film/*")]))

    S.append(dict(name="benchmarks", title="Benchmarks",
                  produces="Tables 2, 3; Section 2.3; Supplementary Table S5; Supplementary Fig. S2",
                  contains="power densities, the source term against Garoz et al., SRIM against NIST, the depth model "
                           "against TRIM, the Blanchard-Martin and Raffray benchmarks, lithium ranges against "
                           "Hassanein and Morozov, ion arrival times, the particle count of the ion spectra and the "
                           "helium and alpha depths",
                  steps=[cmd("power_density", "scripts/diag_power_density.py"),
                         cmd("srim_checks", "validation/srim_checks.py"),
                         cmd("garoz", "scripts/check_source_vs_garoz.py"),
                         cmd("bm2005", "validation/bm2005.py", "10"),
                         cmd("raffray2005", "validation/raffray2005.py", "baseline", "no_stretch", "csda", "xray_pe",
                             "xray_0.1ns", "xray_1ns"),
                         cmd("hm2002", "scripts/benchmark_hm2002_li.py"),
                         cmd("source_term", "scripts/plot_source_term.py"),
                         cmd("alpha_depth", "scripts/diag_alpha_depth.py"),
                         cmd("fig2_spike", "validation/fig2_spike_test.py"),
                         cmd("benchmark_rises", "scripts/diag_benchmark_rises.py", after=["bm2005", "raffray2005"]),
                         cmd("particle_conservation", "scripts/diag_particle_conservation.py"),
                         cmd("he_energy_depth", "scripts/diag_he_energy_depth.py")],
                  outputs=["results/revision/power_density/*", "runs/validation/*", "runs/source_term/garoz_comparison.*",
                           "runs/source_term/source_term_checks.json", "results/benchmarks/hm2002/*",
                           "results/revision/alpha_depth/*", "results/revision/benchmark_rises/*",
                           "results/revision/particle_conservation/*", "results/revision/he_energy_depth/*"],
                  inputs=[]))

    S.append(dict(name="tungsten_single", title="Tungsten, single shots",
                  produces="Sections 4.1, 4.3, 4.4; Figs. 1, 3",
                  contains="single shots of the three scenarios, the ten sensitivity variants, the TRIM-profile case, "
                           "the pure-tungsten reference run used by later regression checks, sputtering, the "
                           "helium fluence per shot, the heat-flux factors and the steel-conductivity check",
                  steps=[cmd("w_single", "scripts/run_w_single.py")]
                        + [cmd("w_single_" + v.replace("=", "-"), "scripts/run_w_single.py", v) for v in W_SINGLE_VARIANTS]
                        + [cmd("trim", "scripts/run_w_single.py", "trim_multilayer=true",
                               "--out=results/revision/trim_multilayer_cold"),
                           cmd("baseline_pureW", "scripts/run_w_single.py", "--out=results/baseline_pureW/w_single"),
                           cmd("sputtering", "scripts/sputtering_erosion.py"),
                           cmd("he_shot_count", "scripts/he_shot_count.py"),
                           cmd("heat_flux_factor", "scripts/diag_heat_flux_factor.py"),
                           cmd("steel_k", "scripts/diag_steel_k.py", after=["w_single"])],
                  outputs=["runs/w_single/*/*", "results/revision/trim_multilayer_cold/*", "results/baseline_pureW/w_single/*",
                           "runs/sputtering/*", "runs/source_term/he_shot_count.json",
                           "results/revision/heat_flux_factor/*", "results/revision/steel_k29/*"],
                  inputs=[]))

    S.append(dict(name="tungsten_repeated", title="Tungsten, repeated shots",
                  produces="Table 4; Fig. 2; Supplementary Table S9",
                  contains="600 Demo shots and 100 Prototype shots to the quasi-steady state, the converged pre-shot "
                           "state that starts the hot single shots and the helium cycles, the runs with other coolant "
                           "coefficients and with a steel conductivity of 29 W/m/K, the heat balance and the "
                           "design-rule cycle counts",
                  steps=[cmd("demo600", "scripts/run_w_multi.py", "Demo", "600"),
                         cmd("proto100", "scripts/run_w_multi.py", "Prototype", "100"),
                         cmd("converged_state", "scripts/diag_converged_state.py", after=["demo600", "proto100"])]
                        + [cmd(f"{s}_h{h}", "scripts/run_w_multi.py", s, "100", f"h={h}")
                           for s in ("Demo", "Prototype") for h in ("5000", "20000")]
                        + [cmd("k29", "scripts/run_w_multi.py", "Demo", "600", "steel_k=29",
                               "--out=results/revision/steel_k29/w_multi_Demo"),
                           cmd("table_h1", "scripts/table_h1_design_rule.py")],
                  outputs=["runs/w_multi/baseline/*", "runs/w_multi/h-5000/*", "runs/w_multi/h-20000/*",
                           "results/revision/converged_state/*", "results/revision/steel_k29/w_multi_Demo/*",
                           "results/revision/table_h1_design_rule/*"],
                  inputs=[("tungsten_single", "runs/w_single/baseline/*")]))

    S.append(dict(name="damage_surface", title="Damaged conductivity and surface layer",
                  produces="Section 4.5; Fig. 4; Supplementary Table S12",
                  contains="single shots and 600-shot Demo runs with a reduced conductivity and a helium-damaged layer, "
                           "the surface-layer sweep with its regression check, and the erosion lifetimes",
                  steps=[cmd("damage_single", "scripts/run_damage_study.py", "--skip-multi", "--workers", "4", w=4),
                         cmd("kirr0.5", "scripts/run_w_multi.py", "Demo", "600", "k_irr=0.5", "--converge",
                             "--out=runs/w_damage/multi/kirr0.5_conv"),
                         cmd("Heskin", "scripts/run_w_multi.py", "Demo", "600", "k_irr=0.5", "skin_um=5", "skin_factor=0.25",
                             "--converge", "--out=runs/w_damage/multi/Heskin_conv"),
                         cmd("extra_kirr0.5", "scripts/run_extra_shot.py", "runs/w_damage/multi/kirr0.5_conv", "Demo",
                             "k_irr=0.5", after=["kirr0.5"]),
                         cmd("extra_Heskin", "scripts/run_extra_shot.py", "runs/w_damage/multi/Heskin_conv", "Demo",
                             "k_irr=0.5", "skin_um=5", "skin_factor=0.25", after=["Heskin"]),
                         cmd("plot_damage", "scripts/plot_damage_study.py",
                             after=["damage_single", "kirr0.5", "Heskin", "extra_kirr0.5", "extra_Heskin"]),
                         cmd("erosion", "scripts/diag_erosion_lifetime.py", after=["kirr0.5", "Heskin"]),
                         cmd("skin_diag", "scripts/skin_sweep_diagnostic.py"),
                         cmd("skin_regcheck", "scripts/run_w_single.py", "--out=results/skin_sweep/regression_check"),
                         cmd("skin_regression", "scripts/run_skin_sweep.py", "regression", after=["skin_regcheck"]),
                         cmd("skin_launch", "scripts/run_skin_sweep.py", "launch", "--jobs=4", w=4),
                         cmd("skin_multi", "scripts/run_w_multi.py", "Demo", "100", "skin_um=10", "skin_factor=0.25",
                             "--out=results/skin_sweep/multi_Demo_d10_f0.25"),
                         cmd("skin_summary", "scripts/run_skin_sweep.py", "summary",
                             after=["skin_regression", "skin_launch", "skin_multi", "skin_diag"])],
                  outputs=["runs/w_damage/*", "runs/w_damage/single/*/*", "runs/w_damage/multi/*_conv/*",
                           "runs/w_damage/multi/*_conv_extra_shot/*",
                           "results/skin_sweep/*", "results/skin_sweep/regression_check/*",
                           "results/skin_sweep/single/*", "results/skin_sweep/single/*/*/*",
                           "results/skin_sweep/multi_Demo_d10_f0.25/*", "results/revision/erosion/*"],
                  inputs=[("tungsten_single", "results/baseline_pureW/w_single/*"), ("tungsten_single", "runs/sputtering/*"),
                          ("tungsten_single", "runs/w_single/baseline/*"), ("tungsten_repeated", "runs/w_multi/baseline/*"),
                          ("tungsten_repeated", "results/revision/converged_state/*"),
                          ("lithium_single", "runs/deposition/*")]))

    S.append(dict(name="lithium_single", title="Lithium, single shots",
                  produces="Sections 5.1, 5.2; Figs. 5, 6; Supplementary Fig. S4",
                  contains="the absorbed fractions, film-thickness scans of the three scenarios with their sensitivity "
                           "variants, the summary tables and the energy-balance estimate",
                  steps=[cmd("deposition", "scripts/deposition_profiles.py")]
                        + [li_scan(s, "T_li=600") for s in ("Experimental", "Prototype", "Demo")]
                        + [li_scan(s, *v) for s in ("Prototype", "Demo") for v in LI_VARIANTS]
                        + [cmd("summary", "scripts/make_summary.py",
                               after=[f"li_{s}_T_li-600" for s in ("Experimental", "Prototype", "Demo")]
                                     + [f"li_{s}_" + "_".join(o.replace("=", "-") for o in v)
                                        for s in ("Prototype", "Demo") for v in LI_VARIANTS]),
                           cmd("hand_estimate", "scripts/diag_li_hand_estimate.py", after=["summary"])],
                  outputs=["runs/deposition/*", "runs/li_scan/T500K/*", "runs/li_scan/T600K/*", "runs/li_scan/T800K/*",
                           "runs/li_scan/T600K_alpha_evap-0.1/*", "runs/li_scan/T600K_lv-handbook_147kJmol/*",
                           "runs/li_scan/T600K_xray_tau-1e-10/*", "runs/li_scan/T600K_xray_tau-5e-08/*",
                           "runs/summary/*", "results/revision/li_hand_estimate/*"],
                  inputs=[("tungsten_single", "runs/w_single/*/*"), ("tungsten_single", "runs/source_term/he_shot_count.json"),
                          ("tungsten_repeated", "runs/w_multi/baseline/*")]))

    S.append(dict(name="lithium_repeated", title="Lithium, repeated shots",
                  produces="Table 5; Section 5.3; Fig. 7; Supplementary Table S6; Supplementary Fig. S3",
                  contains="600 Demo shots and 100 Prototype shots for nine film thicknesses, the minimum films and the "
                           "compatibility limit, helium and displacements in the steel and the helium-exclusion films, "
                           "the direct runs at the minimum and helium films, the bounding runs with the latent heat "
                           "returned, one recorded extra shot of each converged run (steel-face swing, heat-flux "
                           "factors) and the surface pressure",
                  steps=[cmd(f"li_multi_{t}", "scripts/run_li_multi.py", s, "600" if s == "Demo" else "100",
                             f"delta_index={i}", "--converge", f"--out=runs/li_multi/{t}") for s, i, t in LI_MULTI]
                        + [cmd("li_multi_summary", "scripts/li_multi_summary.py", after=[f"li_multi_{t}" for _, _, t in LI_MULTI]),
                           cmd("compatibility", "scripts/diag_li_compatibility.py", after=["li_multi_summary"]),
                           cmd("steel_behind_film", "scripts/steel_behind_film.py"),
                           cmd("steel_criteria", "scripts/diag_steel_criteria.py", after=["steel_behind_film"])]
                        + [cmd(f"direct_{c}_{lim}", "scripts/run_li_direct.py", c, lim, after=LI_DEMO_RUNS)
                           for c, lim in (("baseline", "925"), ("baseline", "873"), ("return_latent", "925"))]
                        + [cmd("direct_he_925", "scripts/run_li_direct.py", "he", "925",
                               after=LI_DEMO_RUNS + ["steel_criteria", "direct_baseline_925"])]
                        + [cmd(f"latent_{t}", "scripts/run_li_multi.py", "Demo", "600", f"delta_index={i}", "--converge",
                               "--return-latent", f"--out=results/revision/li_return_latent/Demo_{t}")
                           for i, t in ((7, "d56.2um"), (8, "d100um"))]
                        + [cmd("extra_cycles", "scripts/run_li_extra_cycles.py",
                               after=[f"li_multi_{t}" for _, _, t in LI_MULTI]
                                     + ["direct_baseline_925", "direct_baseline_873", "direct_he_925"]),
                           cmd("table_extra_cycles", "scripts/table_li_extra_cycles.py", after=["extra_cycles"]),
                           cmd("surface_pressure", "scripts/diag_li_surface_pressure.py", after=["extra_cycles"])],
                  outputs=[f"runs/li_multi/{t}/*" for _, _, t in LI_MULTI]
                          + ["runs/li_multi/summary.*", "results/revision/li_compatibility/*",
                             "results/revision/steel_behind_film/*", "results/revision/li_direct/Demo_d*um/*",
                             "results/revision/li_direct/notes/*", "results/revision/li_return_latent/*/*",
                             "results/revision/li_extra_cycle/*", "results/revision/li_extra_cycle/*/*",
                             "results/revision/li_surface_pressure/*"],
                  inputs=[("lithium_single", "runs/summary/*"), ("lithium_single", "runs/deposition/*"),
                          ("tungsten_single", "runs/source_term/he_shot_count.json")]))

    S.append(dict(name="tungsten_grades", title="Tungsten grades",
                  produces="Supplementary Tables S10, S11; Supplementary Figs. S5, S6",
                  contains="single shots of pure W, W-K and ODS W-La2O3 (with the conductivity band), 600 Demo and 100 "
                           "Prototype shots of the grades, the recrystallization and DBTT indicators",
                  steps=[cmd("alloy_single", "scripts/run_alloy_study.py", "single"),
                         cmd("Demo_ODS", "scripts/run_w_multi.py", "Demo", "600", "w_material=W_ODS_La2O3", "alloy_k_proxy=0.8",
                             "--converge", "--out=results/alloys/multi/Demo_W_ODS_La2O3_p0.8"),
                         cmd("Proto_WK", "scripts/run_w_multi.py", "Prototype", "100", "w_material=W_K", "--converge",
                             "--out=results/alloys/multi/Prototype_W_K"),
                         cmd("Proto_ODS", "scripts/run_w_multi.py", "Prototype", "100", "w_material=W_ODS_La2O3",
                             "alloy_k_proxy=0.8", "--converge", "--out=results/alloys/multi/Prototype_W_ODS_La2O3_p0.8"),
                         cmd("alloy_summary", "scripts/run_alloy_study.py", "summary",
                             after=["alloy_single", "Demo_ODS", "Proto_WK", "Proto_ODS"])],
                  outputs=["results/alloys/summary.*", "results/alloys/regression_W_pure.json", "results/alloys/single/*/*",
                           "results/alloys/multi/Demo_W_ODS_La2O3_p0.8/*", "results/alloys/multi/Prototype_W_K/*",
                           "results/alloys/multi/Prototype_W_ODS_La2O3_p0.8/*"],
                  inputs=[("tungsten_single", "results/baseline_pureW/w_single/*"), ("tungsten_single", "runs/sputtering/*"),
                          ("tungsten_single", "runs/source_term/he_shot_count.json"),
                          ("tungsten_repeated", "runs/w_multi/baseline/*"),
                          ("tungsten_repeated", "results/revision/converged_state/*"),
                          ("lithium_repeated", "runs/li_multi/summary.*")]))

    S.append(dict(name="vapor_shielding", title="Vapor shielding",
                  produces="Section 6; Fig. 8; Supplementary Table S4",
                  contains="the time-coupled shielding cases on the Demo and Prototype films, the static bound, the "
                           "tungsten column, the ionization estimate, larger stopping factors and two 600-shot runs "
                           "with shielding",
                  steps=[cmd("vapor", "scripts/run_vapor_shielding.py", "--workers=4", w=4),
                         cmd("bound", "scripts/vapour_shielding_bound.py"),
                         cmd("w_vapor", "scripts/diag_w_vapor.py"),
                         cmd("saha", "scripts/diag_saha.py", after=["vapor"]),
                         cmd("stopping", "scripts/diag_vapor_stopping.py"),
                         cmd("multi_31.6", "scripts/run_li_multi.py", "Demo", "600", "delta_index=6", "vapor_shielding=true",
                             "vapor_f_rr=0.0", "vapor_stopping_factor=1.0", "--converge",
                             "--out=results/vapor_shielding/multi/Demo_d31.6um_frr0_s1"),
                         cmd("multi_56.2", "scripts/run_li_multi.py", "Demo", "600", "delta_index=7", "vapor_shielding=true",
                             "vapor_f_rr=0.0", "vapor_stopping_factor=1.0", "--converge",
                             "--out=results/vapor_shielding/multi/Demo_d56.2um_frr0_s1")]
                        + [cmd(f"direct_vapor_{lim}", "scripts/run_li_direct.py", "vapor_frr0_s1", lim,
                               after=["multi_31.6", "multi_56.2"]) for lim in ("925", "873")],
                  outputs=["results/vapor_shielding/*", "results/vapor_shielding/*/*", "results/vapor_shielding/multi/*/*",
                           "runs/li_scan/vapour_shielding_bound_Demo.json",
                           "results/revision/li_direct_variants/vapor_frr0_s1/*/*"],
                  inputs=[("tungsten_single", "runs/w_single/baseline/*"), ("lithium_single", "runs/li_scan/T600K/*"),
                          ("lithium_single", "runs/summary/*"), ("damage_surface", "runs/w_damage/multi/*/*")]))

    neutron_li = [(i, t, f) for i, t in ((6, "d31.6um"), (7, "d56.2um"), (8, "d100um")) for f in ("0.5", "1")]
    S.append(dict(name="neutron_heating", title="Neutron heating and coolant position",
                  produces="Sections 4.2, 5.3, 7.5",
                  contains="the heat balance with neutron heating, the neutron-heated tungsten and lithium runs (Demo "
                           "continued from the 600-shot run of the verification stage), the coolant 3.5 mm behind the "
                           "surface, the minimum films with and without neutron heating, the drift left at the last "
                           "shot, the rounding of short steps, the neutron displacement rates and the direct runs at "
                           "the minimum films of the neutron-heated and 3.5 mm variants",
                  steps=[cmd("heat_balance", "scripts/diag_heat_balance_neutron.py"),
                         cmd("checkpoint", "copy_verify_checkpoint")]
                        + [cmd(f"w_Demo_steel{f}", "scripts/run_w_multi.py", "Demo", "1200", "neutron_heating=true",
                               f"neutron_steel_factor={f}", "--converge", "--resume",
                               f"--out=results/neutron/w_multi_Demo_steel{f}", after=["checkpoint"]) for f in ("0.5", "1")]
                        + [cmd(f"w_Proto_steel{f}", "scripts/run_w_multi.py", "Prototype", "100", "neutron_heating=true",
                               f"neutron_steel_factor={f}", "--converge", f"--out=results/neutron/w_multi_Prototype_steel{f}")
                           for f in ("0.5", "1")]
                        + [cmd(f"li_{t}_steel{f}", "scripts/run_li_multi.py", "Demo", "600", f"delta_index={i}",
                               "neutron_heating=true", f"neutron_steel_factor={f}", "--converge",
                               f"--out=results/neutron/li_multi_Demo_{t}_steel{f}") for i, t, f in neutron_li]
                        + [cmd("cool_w_single", "scripts/run_w_single.py", "steel_thickness=0.0025",
                               "--out=results/cooling_3p5mm/w_single"),
                           cmd("cool_w_Demo", "scripts/run_w_multi.py", "Demo", "600", "steel_thickness=0.0025", "--converge",
                               "--out=results/cooling_3p5mm/w_multi_Demo"),
                           cmd("cool_w_Proto", "scripts/run_w_multi.py", "Prototype", "100", "steel_thickness=0.0025",
                               "--converge", "--out=results/cooling_3p5mm/w_multi_Prototype"),
                           cmd("cool_li_Demo", "scripts/run_li_scan.py", "Demo", "T_li=600", "steel_thickness=0.0035"),
                           cmd("cool_li_Proto", "scripts/run_li_scan.py", "Prototype", "T_li=600", "steel_thickness=0.0035")]
                        + [cmd(f"cool_{t}", "scripts/run_li_multi.py", s, "600" if s == "Demo" else "100", f"delta_index={i}",
                               "steel_thickness=0.0035", "--converge", f"--out=results/cooling_3p5mm/li_multi_{t}")
                           for s, i, t in (("Demo", 6, "Demo_d31.6um"), ("Demo", 7, "Demo_d56.2um"),
                                           ("Prototype", 4, "Prototype_d10um"), ("Prototype", 5, "Prototype_d17.8um"))]
                        + [cmd("cooling_summary", "scripts/diag_cooling_summary.py",
                               after=["cool_w_single", "cool_w_Demo", "cool_w_Proto", "cool_li_Demo", "cool_li_Proto",
                                      "cool_Demo_d31.6um", "cool_Demo_d56.2um", "cool_Prototype_d10um", "cool_Prototype_d17.8um"]),
                           cmd("li_variants", "scripts/diag_li_multi_variants.py",
                               after=[f"li_{t}_steel{f}" for _, t, f in neutron_li]
                                     + ["cool_Demo_d31.6um", "cool_Demo_d56.2um", "cool_Prototype_d10um", "cool_Prototype_d17.8um"]),
                           cmd("drift", "scripts/diag_quasi_steady_drift.py",
                               after=["w_Demo_steel0.5", "w_Demo_steel1", "cool_w_Demo"]),
                           cmd("clock_rounding", "scripts/diag_clock_rounding.py",
                               after=["w_Demo_steel0.5", "w_Demo_steel1", "w_Proto_steel0.5", "w_Proto_steel1"]),
                           cmd("neutron_dpa", "scripts/diag_neutron_dpa.py")]
                        + [cmd(f"direct_neutron{f}_{lim}", "scripts/run_li_direct.py", f"neutron_steel{f}", lim,
                               after=[f"li_{t}_steel{f}" for _, t, g in neutron_li if g == f])
                           for f in ("0.5", "1") for lim in ("925", "873")]
                        + [cmd(f"direct_cooling_{lim}", "scripts/run_li_direct.py", "steel_3.5mm", lim,
                               after=["cool_Demo_d31.6um", "cool_Demo_d56.2um"]) for lim in ("925", "873")],
                  outputs=["results/neutron/heat_balance/*", "results/neutron/*/*", "results/cooling_3p5mm/*",
                           "results/cooling_3p5mm/w_single/*", "results/cooling_3p5mm/w_multi_Demo/*",
                           "results/cooling_3p5mm/w_multi_Prototype/*", "results/cooling_3p5mm/li_multi_*/*",
                           "runs/li_scan/T600K_steel_thickness-0.0035/*", "results/revision/li_multi_variants/*",
                           "results/revision/quasi_steady_drift/*", "results/revision/clock_rounding/*",
                           "results/revision/neutron_dpa/*", "results/revision/li_direct_variants/neutron_steel0.5/*/*",
                           "results/revision/li_direct_variants/neutron_steel1/*/*",
                           "results/revision/li_direct_variants/steel_3.5mm/*/*"],
                  inputs=[("tungsten_single", "runs/w_single/baseline/*"), ("tungsten_repeated", "runs/w_multi/baseline/*"),
                          ("lithium_repeated", "runs/li_multi/*/*"), ("lithium_repeated", "runs/li_multi/summary.*"),
                          ("vapor_shielding", "results/vapor_shielding/multi/*/*")],
                  raw=[("verification", "runs/w_multi/verify_2026-10-05/Demo_checkpoint.npz",
                        cmd("demo600", "scripts/run_w_multi.py", "Demo", "600", "--out=runs/w_multi/verify_2026-10-05"))]))

    li_in = [("lithium_single", "runs/summary/*"), ("lithium_repeated", "runs/li_multi/*/*"),
             ("lithium_repeated", "runs/li_multi/summary.*"), ("lithium_repeated", "results/revision/li_compatibility/*"),
             ("lithium_repeated", "results/revision/steel_behind_film/*"),
             ("lithium_repeated", "results/revision/li_direct/Demo_d*um/*"),
             ("lithium_repeated", "results/revision/li_return_latent/*/*"),
             ("neutron_heating", "results/neutron/*/*"), ("neutron_heating", "results/cooling_3p5mm/li_multi_*/*"),
             ("neutron_heating", "results/revision/li_direct_variants/*/*/*"),
             ("vapor_shielding", "results/vapor_shielding/multi/*/*"),
             ("vapor_shielding", "results/revision/li_direct_variants/*/*/*")]
    S.append(dict(name="lithium_films", title="Lithium, minimum films and partial renewal",
                  produces="Table 5; Section 5.4",
                  contains="the minimum films from the grid and direct runs of every variant, helium and displacements "
                           "in the steel at the chosen films, the viscous stopping distance, and six 600-shot Demo runs "
                           "with 3 % and 30 % of the film renewed per shot",
                  steps=[cmd("minima_direct", "scripts/diag_li_minima_direct.py"),
                         cmd("steel_at_films", "scripts/steel_at_films.py", "--auto", after=["minima_direct"]),
                         cmd("film_travel", "scripts/diag_film_travel.py", after=["minima_direct"])]
                        + [cmd(f"renew_{t}_r{f}", "scripts/run_li_multi.py", "Demo", "600", f"delta_index={i}",
                               f"renew={f}", "--converge", f"--out=results/revision/li_renewal/Demo_{t}_r{f}")
                           for i, t, f in RENEWAL]
                        + [cmd("renewal_summary", "scripts/li_renewal_summary.py",
                               after=[f"renew_{t}_r{f}" for _, t, f in RENEWAL])],
                  outputs=["results/revision/li_direct/minima_direct.json",
                           "results/revision/steel_behind_film/steel_at_films.json", "results/revision/film_travel/*",
                           "results/revision/li_renewal/*", "results/revision/li_renewal/*/*"],
                  inputs=li_in))

    S.append(dict(name="helium", title="Helium retention model",
                  produces="Table 4; Supplementary Tables S16–S18",
                  contains="the temperature cycles from the converged pre-shot state, 1450 Demo, 450 Prototype and 500 "
                           "Experimental shots of the baseline model, the path-length count, ten sensitivity cases, the "
                           "rising-temperature case, the verification tests, the comparison with experiment, stopping "
                           "depths, the time above the temperature thresholds and the Experimental counts",
                  steps=[cmd("rec_Demo", "scripts/record_he_thermal.py", "Demo", "converged", "--Tback=converged"),
                         cmd("rec_Proto", "scripts/record_he_thermal.py", "Prototype", "converged", "--Tback=converged"),
                         cmd("rec_Exp", "scripts/record_he_thermal.py", "Experimental", "cold"),
                         cmd("ret_Demo", "scripts/run_he_retention.py", "Demo", "--shots", "1450", "--case", "baseline",
                             after=["rec_Demo"]),
                         cmd("ret_Proto", "scripts/run_he_retention.py", "Prototype", "--shots", "450", "--case", "baseline",
                             after=["rec_Proto"]),
                         cmd("ret_Exp", "scripts/run_he_retention.py", "Experimental", "--shots", "500", "--case", "baseline",
                             after=["rec_Exp"]),
                         cmd("dpa_Demo", "scripts/run_he_retention.py", "Demo", "--shots", "150", "--case", "dpa_path", "--out",
                             "runs/he_retention/Demo_dpa_path", after=["rec_Demo"]),
                         cmd("dpa_Proto", "scripts/run_he_retention.py", "Prototype", "--shots", "350", "--case", "dpa_path",
                             "--out", "runs/he_retention/Prototype_dpa_path", after=["rec_Proto"]),
                         cmd("case_thresholds", "scripts/he_case_thresholds.py", "runs/he_retention/Demo_dpa_path",
                             "runs/he_retention/Prototype_dpa_path", after=["dpa_Demo", "dpa_Proto"]),
                         cmd("batch", "scripts/run_he_batch.py", "--jobs", "3", *HE_BATCH, w=3, after=["rec_Demo"]),
                         cmd("rising_T", "scripts/run_he_rising_T.py", "300", after=["rec_Demo"]),
                         cmd("hashimoto", "scripts/he_validation_hashimoto.py")]
                        + [cmd(v, f"scripts/{v}.py", *(["250"] if v == "verify_he_6_benchmarks" else []),
                               after=(["batch", "rec_Proto"] if k == 0 else [VERIFY_HE[k - 1]]))
                           for k, v in enumerate(VERIFY_HE)]
                        + [cmd("he_results", "scripts/make_he_results.py",
                               after=["ret_Demo", "ret_Proto", "ret_Exp", "batch", "rising_T", VERIFY_HE[-1]]),
                           cmd("sharafat", "scripts/he_compare_sharafat.py", after=["he_results"]),
                           cmd("depths", "scripts/diag_he_depths.py"),
                           cmd("time_above", "scripts/he_time_above.py", after=["rec_Demo"]),
                           cmd("experimental_helium", "scripts/diag_experimental_helium.py", after=["ret_Exp", "he_results"])],
                  outputs=["runs/he_retention/*_baseline/*", "runs/he_retention/*_dpa_path/*", "runs/he_retention/results/*",
                           "runs/he_retention/sens/*/*", "runs/he_retention/verification/*",
                           "runs/he_retention/validation_hashimoto/*", "results/revision/he_depths/*",
                           "results/revision/experimental_helium/*"],
                  inputs=[("tungsten_repeated", "runs/w_multi/baseline/*"),
                          ("tungsten_repeated", "results/revision/converged_state/*"),
                          ("tungsten_single", "runs/source_term/he_shot_count.json")]))

    S.append(dict(name="figures", title="Figures",
                  produces="Figs. 1–8; Supplementary Figs. S1–S6",
                  contains="the fourteen figures of the paper, drawn from the outputs of the stages above",
                  steps=[cmd("figures", "scripts/make_article_figures.py")],
                  outputs=["docs/figures/article/*.png"],
                  inputs=["*"]))

    S.append(dict(name="article", title="Article",
                  produces="Tables 1, A.1, B.1, B.2, C.1; Supplementary Tables S3, S7, S8, S13–S15, S19",
                  contains="the PDF built from the paper source in paper/ and the figures of the figures stage",
                  steps=[cmd("article", "scripts/make_article.py")],
                  outputs=["docs/article/*.pdf"],
                  inputs=["figures"]))
    return S


STAGES = stage_list()
BY_NAME = {s["name"]: s for s in STAGES}


# ------------------------------------------------------------------------------------------ paths
def to_results(legacy):
    """Path of a work-copy output inside results/<stage>/: the runs/ and results/ prefixes are dropped,
    and the figures and the article PDF go to the top of their stage folder."""
    p = legacy.replace("\\", "/")
    for pre in ("docs/figures/article/", "docs/article/", "runs/", "results/"):
        if p.startswith(pre):
            return p[len(pre):]
    return p


def _match(path, glob):
    pp, gg = path.split("/"), glob.split("/")
    return len(pp) == len(gg) and all(re.fullmatch(re.escape(g).replace(r"\*", "[^/]*"), x) for x, g in zip(pp, gg))


def from_results(stage, rel):
    """Inverse of to_results for a file of results/<stage>/ (None if no output pattern of the stage matches)."""
    for g in BY_NAME[stage]["outputs"]:
        for pre in ("runs/", "results/", "docs/figures/article/", "docs/article/"):
            if g.startswith(pre) and _match(pre + rel, g):
                return pre + rel
    return None


def keep(path, size):
    """The outputs that go into results/: small JSON, CSV, Markdown and text files (not logs), plus the small
    binary files that the figures need (the verification curves), the figure PNGs and the article PDF."""
    name = path.split("/")[-1]
    ext = os.path.splitext(name)[1].lower()
    if path.startswith("docs/figures/article/"):
        return ext == ".png"
    if path.startswith("docs/article/"):
        return ext == ".pdf" and not name.startswith("_")
    if name == "verification_curves.npz":
        return True
    if ext not in KEEP_EXT or size > MAX_BYTES:
        return False
    if name.startswith("log") or name.endswith(("_log.txt", "_err.txt", ".log")) or name in ("COMMAND.txt", "git_hash.txt"):
        return False
    if "_history" in name:
        return path.startswith(("runs/w_single/baseline/", "results/baseline_pureW/", "results/alloys/single/",
                                "results/skin_sweep/regression_check/"))
    return True


def normalize(text, roots):
    """Absolute folder paths of the work copy (or of the copy that made an output) become paths relative to
    the repository root, in every spelling a file may use (plain, JSON-escaped, forward slashes)."""
    for r in roots:
        r = str(r).rstrip("\\/")
        for v in (r + "\\", r.replace("\\", "\\\\") + "\\\\", r.replace("\\", "/") + "/"):
            text = text.replace(v, "")
    return text


def collect(stage, work):
    st = BY_NAME[stage]
    out_dir = RESULTS / stage
    if out_dir.exists():
        shutil.rmtree(out_dir)
    done = set()
    for g in st["outputs"]:
        for p in sorted(work.glob(g)):
            rel = p.relative_to(work).as_posix()
            if not p.is_file() or rel in done or not keep(rel, p.stat().st_size):
                continue
            done.add(rel)
            dst = out_dir / to_results(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix.lower() in KEEP_EXT:
                text = normalize(p.read_text(encoding="utf-8", errors="replace"), [work.resolve(), work])
                dst.write_text(text, encoding="utf-8", newline="\n")
            else:
                shutil.copy2(p, dst)
    return len(done)


# ------------------------------------------------------------------------------------------- work copy
def prepare(stage):
    work = WORK / stage
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for c in CODE:
        s = ROOT / c
        if s.is_dir():
            shutil.copytree(s, work / c, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        elif s.exists():
            shutil.copy2(s, work / c)
    for d in ("runs", "results", "docs/figures/article", "docs/figures/full", "docs/figures/damage", "docs/article"):
        (work / d).mkdir(parents=True, exist_ok=True)
    st = BY_NAME[stage]
    staged = 0

    def put(src_stage, rel, legacy):
        nonlocal staged
        dst = work / legacy
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RESULTS / src_stage / rel, dst)
        staged += 1

    if st["inputs"] == ["*"]:
        for s in STAGES:
            if s["name"] in ("figures", "article"):
                continue
            base = RESULTS / s["name"]
            for p in sorted(base.rglob("*")) if base.exists() else []:
                rel = p.relative_to(base).as_posix()
                legacy = from_results(s["name"], rel) if p.is_file() else None
                if legacy:
                    put(s["name"], rel, legacy)
    elif st["inputs"] == ["figures"]:
        for p in sorted((RESULTS / "figures").glob("*.png")):
            put("figures", p.name, "docs/figures/article/" + p.name)
    else:
        for src, g in st["inputs"]:
            pre = next(x for x in ("runs/", "results/") if g.startswith(x))
            base = RESULTS / src
            for p in sorted(base.glob(g[len(pre):])) if base.exists() else []:
                if p.is_file():
                    put(src, p.relative_to(base).as_posix(), pre + p.relative_to(base).as_posix())
    if stage == "article":
        paper = ROOT / "paper"
        for p in sorted(paper.glob("*.md")) + sorted(paper.glob("references.json")):
            if p.name != "README.md":
                shutil.copy2(p, work / "docs" / "article" / p.name)
    st["_prereq"] = []
    for src, rel, fallback in st.get("raw", []):
        p = WORK / src / rel
        dst = work / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            shutil.copy2(p, dst)
        else:
            st["_prereq"].append(fallback)
    return work, staged


def run_stage(stage, jobs):
    st = BY_NAME[stage]
    t0 = time.time()
    print(f"== {stage}: {st['title']}", flush=True)
    if stage == "article" and not [p for p in (ROOT / "paper").glob("*.md") if p.name != "README.md"]:
        print("   the paper source is not in this release yet (see paper/README.md); nothing to do")
        return 0
    missing = missing_data() if stage != "article" else []
    if missing:
        print("   missing third-party data tables:\n     " + "\n     ".join(missing[:8]) +
              ("\n     ..." if len(missing) > 8 else "") + "\n   install them first: python fetch_data.py --help")
        return 2
    work, staged = prepare(stage)
    print(f"   work copy {work.relative_to(ROOT)}; {staged} input files from results/", flush=True)
    pre = [x["name"] for x in st["_prereq"]]
    steps = [dict(c) for c in st["_prereq"]] + [dict(c, after=c["after"] + (pre if c["name"] == "checkpoint" else []))
                                                 for c in st["steps"]]
    if pre:
        print(f"   {', '.join(pre)}: the raw file it provides is not in work/ from an earlier stage; rerun here", flush=True)
    failed = schedule(steps, work, jobs)
    n = collect(stage, work)
    dt = time.time() - t0
    (RESULTS / stage).mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, "elapsed_s": round(dt, 1), "jobs": jobs, "failed": failed,
           "python": platform.python_version(), "platform": platform.platform(terse=True),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (WORK / stage / "STAGE_RUN.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"   {n} files written to results/{stage}/; {stage} took {fmt(dt)}" +
          (f"; FAILED: {', '.join(failed)} (logs in work/{stage}/_logs/)" if failed else ""), flush=True)
    return 1 if failed else 0


def schedule(steps, work, jobs):
    logs = work / "_logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1")
    pending, running, done, failed = list(steps), {}, set(), []
    while pending or running:
        busy = sum(r["w"] for r in running.values())
        for c in list(pending):
            if all(a in done for a in c["after"]) and (busy + c["w"] <= jobs or not running):
                if c["args"] and c["args"][0] == "copy_verify_checkpoint":
                    try:
                        copy_verify_checkpoint(work)
                        done.add(c["name"])
                    except OSError as e:
                        print(f"   {c['name']} failed: {e}")
                        failed.append(c["name"])
                        done.add(c["name"])
                    pending.remove(c)
                    continue
                log = open(logs / f"{c['name']}.log", "w", encoding="utf-8")
                p = subprocess.Popen([sys.executable] + c["args"], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[c["name"]] = {"p": p, "w": c["w"], "t0": time.time(), "log": log}
                pending.remove(c)
                busy += c["w"]
        for name, r in list(running.items()):
            rc = r["p"].poll()
            if rc is not None:
                r["log"].close()
                print(f"   {'ok  ' if rc == 0 else 'FAIL'} {name} ({fmt(time.time() - r['t0'])})", flush=True)
                if rc != 0:
                    failed.append(name)
                done.add(name)
                del running[name]
        if pending and not running and not any(all(a in done for a in c["after"]) for c in pending):
            failed += [c["name"] for c in pending]
            break
        time.sleep(2)
    return failed


def fmt(s):
    s = int(round(s))
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def missing_data():
    """The third-party tables that are not installed (fetch_data.py lists and checks them)."""
    sys.path.insert(0, str(ROOT))
    import fetch_data
    return fetch_data.check(verbose=False)[0]


# ------------------------------------------------------------------------------------------- check
IGNORE = {"wall_s", "wall_s_total", "wall_s_this_session", "run_date", "git", "git_hash", "date", "generated", "timestamp",
          "elapsed_s", "runtime_s", "started", "finished", "start", "end", "wall_time_s", "wall_min", "host", "commit",
          "newton_this_session"}


def _num(x):
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cmp_json(a, b, path, out, rtol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append((path + "/" + k, "only in " + ("committed" if k not in a else "new"), None))
            else:
                cmp_json(a[k], b[k], path + "/" + k, out, rtol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}", None))
        for i, (x, y) in enumerate(zip(a, b)):
            cmp_json(x, y, f"{path}[{i}]", out, rtol)
    else:
        fa, fb = _num(a), _num(b)
        if fa is not None and fb is not None:
            if not (fa == fb or (math.isnan(fa) and math.isnan(fb))):
                rel = abs(fa - fb) / max(abs(fa), abs(fb), 1e-300)
                if rel > rtol:
                    out.append((path, f"{a} vs {b}", rel))
        elif a != b:
            out.append((path, f"{str(a)[:50]!r} vs {str(b)[:50]!r}", None))


def cmp_table(ta, tb, rtol):
    ra = list(csv.reader(io.StringIO(ta)))
    rb = list(csv.reader(io.StringIO(tb)))
    if len(ra) != len(rb):
        return [("rows", f"{len(ra)} vs {len(rb)}", None)]
    n, worst = 0, 0.0
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            fu, fv = _num(u), _num(v)
            if fu is not None and fv is not None:
                rel = abs(fu - fv) / max(abs(fu), abs(fv), 1e-300) if fu != fv else 0.0
                if rel > rtol:
                    n, worst = n + 1, max(worst, rel)
            elif u != v:
                n += 1
    return [("cells", f"{n} cells differ", worst)] if n else []


MASK = re.compile(r"\b20\d\d-\d\d-\d\d[ T]\d\d:\d\d(:\d\d)?\b|commit ([0-9a-f]{7,40}|no-git)|\b\d+(\.\d+)? ?(s|min|h) wall\b")


def check(stage, rtol, ref):
    base = RESULTS / stage
    files = sorted(p.relative_to(ROOT).as_posix() for p in base.rglob("*") if p.is_file())
    if ref:
        old = {p.relative_to(Path(ref)).as_posix() for p in (Path(ref) / "results" / stage).rglob("*") if p.is_file()}
    else:
        r = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", f"results/{stage}"], cwd=ROOT,
                           capture_output=True, text=True)
        old = set(r.stdout.split())

    def committed(rel):
        if ref:
            return (Path(ref) / rel).read_bytes()
        return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True).stdout

    same = close = 0
    diffs, new = [], [f for f in files if f not in old]
    for rel in files:
        if rel not in old:
            continue
        a, b = (ROOT / rel).read_bytes(), committed(rel)
        if a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"):
            same += 1
            continue
        ext = os.path.splitext(rel)[1].lower()
        ta, tb = a.decode("utf-8", "replace"), b.decode("utf-8", "replace")
        out = []
        if ext == ".json":
            cmp_json(json.loads(ta), json.loads(tb), "", out, rtol)
        elif ext == ".csv":
            out = cmp_table(ta, tb, rtol)
        elif ext in (".md", ".txt"):
            if MASK.sub("#", ta).splitlines() != MASK.sub("#", tb).splitlines():
                out = [("text", "lines differ", None)]
        else:
            out = [("binary", "content differs", None)]
        if out:
            diffs.append((rel, out))
        else:
            close += 1
    print(f"== check {stage}: {len(files)} files; {same} identical, {close} equal within {rtol:g}, "
          f"{len(diffs)} differ, {len(new)} new, {len(old - set(files))} missing")
    for rel, out in diffs:
        worst = max([o[2] or 0 for o in out], default=0)
        print(f"   {rel}: {len(out)} differences, largest relative {worst:.2g}; e.g. " +
              "; ".join(f"{o[0]}: {o[1]}" for o in out[:3]))
    for rel in sorted(old - set(files)):
        print(f"   missing: {rel}")
    return 1 if diffs else 0


def main(argv):
    jobs = 4
    rtol = 1e-9
    ref = None
    args = []
    it = iter(argv)
    for a in it:
        if a == "--jobs":
            jobs = int(next(it))
        elif a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        elif a.startswith("--rtol="):
            rtol = float(a.split("=", 1)[1])
        elif a.startswith("--ref="):
            ref = a.split("=", 1)[1]
        else:
            args.append(a)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "--list":
        for k, s in enumerate(STAGES, 1):
            print(f"{k:2d}  {s['name']:18s} {s['produces']}")
        return 0
    if args[0] == "--check":
        names = [s["name"] for s in STAGES] if args[1:] == ["all"] else args[1:]
        return max([check(n, rtol, ref) for n in names] or [0])
    names = [s["name"] for s in STAGES] if args == ["all"] else args
    for n in names:
        if n not in BY_NAME:
            print(f"unknown stage {n!r}; python reproduce.py --list")
            return 2
    rc = 0
    t0 = time.time()
    for n in names:
        rc = max(rc, run_stage(n, jobs))
    if len(names) > 1:
        print(f"== all requested stages took {fmt(time.time() - t0)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
