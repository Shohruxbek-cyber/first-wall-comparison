# Degraded near-surface layer: sweep summary (2026-10-22 17:19, commit 0694a93)

Motivation (article, Section 4.5): a real tungsten surface need not have bulk properties; depending on the surface treatment it can retain a defective layer, and that is where the short pulse is absorbed. The solver assumed bulk Tolias 2017 properties everywhere. Here a surface layer of thickness d has its solid-phase conductivity multiplied by f relative to bulk at the same temperature (density and specific heat at bulk; molten W undamaged), over undamaged bulk W. Effusivity scales as sqrt(f). **The factors are HYPOTHETICAL**: no measured effusivity of a machined, ground, polished or etched W surface was obtained; f = 0.75, 0.5, 0.25 bound the effect.

Solver options reused unchanged from 29 Sep 2026 (`skin_um`, `skin_factor`; `k_irr` = 1). Regression of the defaults against results/baseline_pureW: **bit-identical** (keys peak_surface_T_K, t_peak_us, melt_depth_um, ablated_nm_per_shot, energy_rel_residual, pre_shot2_surface_T_K, energy_rel_residual_full_cycle, diagnostics, ledger_J_m2; history, max-T profile and He-layer CSVs; `regression_check/REGRESSION.json`).

Columns: margin to melting = 3695 K − peak; margins to the recrystallization onsets = onset − peak (negative: exceeded), onsets 1300 K (lower edge of the GA2020 band, Section 5.1) and 1523 K (one-hour anneal onset of pure W, Section 5.7); depth above = deepest point of the max-T depth grid (1 nm–200 µm, 240 log-spaced points) that reached the onset in the shot; evaporation from the run; sputtering from runs/sputtering (Eckstein yields; grade- and property-independent); erosion-limited shots = 1 mm/(evaporation + sputtering); interface = last W cell (1 µm from the W/steel interface), maximum in the shot cycle; Newton counts to 50 µs after ignition (first integration part, where the peak occurs) and over the full shot period; last update = largest |ΔT| of the final Newton iteration of the last accepted step; residual = energy-ledger residual (Section 3.11).

## Single shots from the scenario initial temperature (cold start: 300 K Experimental, 600 K Prototype and Demo)

| scenario | d (µm) | f | √f | peak T (K) | t_peak (µs) | margin to melting (K) | melt (µm) | margin to 1300 K | margin to 1523 K | depth > 1300 K (µm) | depth > 1523 K (µm) | evap. (nm/shot) | sputter (nm/shot) | erosion-limited shots | interface max in shot (K) | pre-shot T next shot (K) | steps to 50 µs (rejected) | Newton iters total / max per step, to 50 µs | last update (K) | steps full cycle | iters total / max, full cycle | last update full cycle (K) | residual 50 µs | residual full cycle | wall (s) | git |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Experimental | 0 (no skin) | 1 | 1.000 | 868.2 | 1.650 | +2826.8 | 0.000 | +432 | +655 | 0.00 | 0.00 | 8.83e-44 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2486 (4) | 6572 / 5 | 2.70e-10 | – | – / – | – | 2.2e-14 | – | 5 | a9d4d2e |
| Prototype | 0 (no skin) | 1 | 1.000 | 1370.8 | 2.136 | +2324.2 | 0.000 | -71 | +152 | 1.34 | 0.00 | 7.48e-25 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2573 (4) | 6833 / 4 | 1.62e-08 | 3857 | 10142 / 4 | 1.32e-08 | 2.3e-12 | 4.17e-12 | 10 | a9d4d2e |
| Demo | 0 (no skin) | 1 | 1.000 | 3026.9 | 2.150 | +668.1 | 0.000 | -1727 | -1504 | 17.23 | 12.05 | 3.38e-07 | 1.09e-04 | 9.11e+09 | 617.43 | 610.13 | 3305 (5) | 9994 / 4 | 2.85e-07 | 4473 | 13152 / 4 | 3.38e-11 | 1.8e-12 | 2.31e-12 | 12 | a9d4d2e |
| Experimental | 1 | 1 | 1.000 | 868.2 | 1.650 | +2826.8 | 0.000 | +432 | +655 | 0.00 | 0.00 | 8.83e-44 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2486 (4) | 6569 / 5 | 6.52e-10 | – | – / – | – | 2.4e-14 | – | 6 | 0694a93 |
| Prototype | 1 | 1 | 1.000 | 1370.8 | 2.139 | +2324.2 | 0.000 | -71 | +152 | 1.34 | 0.00 | 7.48e-25 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2573 (4) | 6834 / 4 | 1.43e-08 | 3857 | 10160 / 4 | 1.32e-08 | 2.4e-12 | 4.34e-12 | 11 | 0694a93 |
| Demo | 1 | 1 | 1.000 | 3026.9 | 2.150 | +668.1 | 0.000 | -1727 | -1504 | 17.23 | 12.05 | 3.38e-07 | 1.09e-04 | 9.11e+09 | 617.43 | 610.13 | 3304 (5) | 9993 / 4 | 2.68e-07 | 4472 | 13151 / 4 | 3.38e-11 | 1.9e-12 | 2.32e-12 | 14 | 0694a93 |
| Experimental | 1 | 0.75 | 0.866 | 880.5 | 1.621 | +2814.5 | 0.000 | +419 | +642 | 0.00 | 0.00 | 4.64e-43 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2489 (3) | 6597 / 5 | 6.46e-09 | – | – / – | – | 2.4e-14 | – | 7 | 0694a93 |
| Prototype | 1 | 0.75 | 0.866 | 1386.7 | 2.097 | +2308.3 | 0.000 | -87 | +136 | 1.34 | 0.00 | 1.76e-24 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2581 (4) | 6867 / 4 | 1.12e-10 | 3865 | 10195 / 4 | 1.32e-08 | 2.4e-12 | 4.24e-12 | 11 | 0694a93 |
| Demo | 1 | 0.75 | 0.866 | 3084.6 | 2.131 | +610.4 | 0.000 | -1785 | -1562 | 17.23 | 12.05 | 6.29e-07 | 1.09e-04 | 9.09e+09 | 617.43 | 610.13 | 3336 (7) | 10111 / 4 | 1.87e-07 | 4504 | 13272 / 4 | 3.39e-11 | 1.8e-12 | 2.35e-12 | 15 | 0694a93 |
| Experimental | 1 | 0.5 | 0.707 | 905.7 | 1.618 | +2789.3 | 0.000 | +394 | +617 | 0.00 | 0.00 | 1.20e-41 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2510 (8) | 6693 / 5 | 8.06e-10 | – | – / – | – | 2.2e-14 | – | 7 | 0694a93 |
| Prototype | 1 | 0.5 | 0.707 | 1419.1 | 2.105 | +2275.9 | 0.000 | -119 | +104 | 1.34 | 0.00 | 9.49e-24 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2597 (5) | 6954 / 5 | 4.23e-09 | 3881 | 10277 / 5 | 1.32e-08 | 2.5e-12 | 4.43e-12 | 12 | 0694a93 |
| Demo | 1 | 0.5 | 0.707 | 3202.3 | 2.102 | +492.7 | 0.000 | -1902 | -1679 | 17.23 | 12.05 | 2.08e-06 | 1.09e-04 | 8.97e+09 | 617.43 | 610.13 | 3393 (7) | 10357 / 4 | 7.29e-09 | 4561 | 13519 / 4 | 3.38e-11 | 2.0e-12 | 2.49e-12 | 15 | 0694a93 |
| Experimental | 1 | 0.25 | 0.500 | 985.1 | 1.427 | +2709.9 | 0.000 | +315 | +538 | 0.00 | 0.00 | 9.75e-38 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2540 (7) | 6832 / 5 | 2.38e-08 | – | – / – | – | 2.5e-14 | – | 7 | 0694a93 |
| Prototype | 1 | 0.25 | 0.500 | 1522.1 | 1.854 | +2172.9 | 0.000 | -222 | +1 | 1.27 | 0.00 | 1.11e-21 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2644 (6) | 7177 / 5 | 1.31e-10 | 3928 | 10501 / 5 | 1.32e-08 | 2.7e-12 | 4.60e-12 | 11 | 0694a93 |
| Demo | 1 | 0.25 | 0.500 | 3562.4 | 2.089 | +132.6 | 0.000 | -2262 | -2039 | 17.23 | 12.05 | 4.91e-05 | 1.09e-04 | 6.31e+09 | 617.43 | 610.13 | 3547 (5) | 11006 / 5 | 9.49e-09 | 4715 | 14171 / 5 | 3.39e-11 | 2.1e-12 | 2.61e-12 | 15 | 0694a93 |
| Experimental | 3 | 1 | 1.000 | 868.2 | 1.650 | +2826.8 | 0.000 | +432 | +655 | 0.00 | 0.00 | 8.82e-44 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2486 (4) | 6576 / 5 | 2.87e-10 | – | – / – | – | 2.3e-14 | – | 6 | 0694a93 |
| Prototype | 3 | 1 | 1.000 | 1370.8 | 2.135 | +2324.2 | 0.000 | -71 | +152 | 1.34 | 0.00 | 7.48e-25 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2573 (4) | 6838 / 4 | 1.65e-08 | 3857 | 10161 / 4 | 1.32e-08 | 2.4e-12 | 4.38e-12 | 10 | 0694a93 |
| Demo | 3 | 1 | 1.000 | 3026.8 | 2.153 | +668.2 | 0.000 | -1727 | -1504 | 17.23 | 12.05 | 3.38e-07 | 1.09e-04 | 9.11e+09 | 617.43 | 610.13 | 3304 (5) | 10015 / 4 | 3.56e-07 | 4472 | 13177 / 4 | 3.38e-11 | 1.9e-12 | 2.33e-12 | 14 | 0694a93 |
| Experimental | 3 | 0.75 | 0.866 | 909.8 | 1.621 | +2785.2 | 0.000 | +390 | +613 | 0.00 | 0.00 | 2.08e-41 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2508 (6) | 6676 / 5 | 1.41e-08 | – | – / – | – | 2.4e-14 | – | 7 | 0694a93 |
| Prototype | 3 | 0.75 | 0.866 | 1424.9 | 2.088 | +2270.1 | 0.000 | -125 | +98 | 1.64 | 0.00 | 1.31e-23 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2596 (4) | 6933 / 5 | 4.60e-10 | 3880 | 10262 / 5 | 1.32e-08 | 2.4e-12 | 4.32e-12 | 12 | 0694a93 |
| Demo | 3 | 0.75 | 0.866 | 3208.8 | 2.120 | +486.2 | 0.000 | -1909 | -1686 | 17.23 | 12.05 | 2.27e-06 | 1.09e-04 | 8.96e+09 | 617.43 | 610.13 | 3384 (6) | 10323 / 4 | 9.67e-09 | 4552 | 13484 / 4 | 3.38e-11 | 1.9e-12 | 2.40e-12 | 15 | 0694a93 |
| Experimental | 3 | 0.5 | 0.707 | 995.1 | 1.437 | +2699.9 | 0.000 | +305 | +528 | 0.00 | 0.00 | 3.01e-37 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2538 (5) | 6797 / 5 | 1.10e-08 | – | – / – | – | 2.6e-14 | – | 7 | 0694a93 |
| Prototype | 3 | 0.5 | 0.707 | 1537.8 | 1.865 | +2157.2 | 0.000 | -238 | -15 | 2.02 | 0.32 | 2.23e-21 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2643 (5) | 7142 / 5 | 5.59e-09 | 3927 | 10467 / 5 | 1.32e-08 | 2.7e-12 | 4.49e-12 | 11 | 0694a93 |
| Demo | 3 | 0.5 | 0.707 | 3563.9 | 2.102 | +131.1 | 0.000 | -2264 | -2041 | 17.23 | 12.05 | 5.11e-05 | 1.09e-04 | 6.23e+09 | 617.43 | 610.13 | 3528 (6) | 10896 / 4 | 9.27e-08 | 4696 | 14060 / 4 | 3.38e-11 | 2.1e-12 | 2.65e-12 | 16 | 0694a93 |
| Experimental | 3 | 0.25 | 0.500 | 1237.1 | 1.594 | +2457.9 | 0.000 | +63 | +286 | 0.00 | 0.00 | 1.35e-28 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2633 (6) | 7176 / 5 | 2.24e-09 | – | – / – | – | 2.4e-14 | – | 7 | 0694a93 |
| Prototype | 3 | 0.25 | 0.500 | 1852.1 | 2.069 | +1842.9 | 0.000 | -552 | -329 | 2.35 | 1.41 | 1.62e-16 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2766 (5) | 7639 / 5 | 6.12e-09 | 4050 | 10967 / 5 | 1.32e-08 | 3.2e-12 | 5.23e-12 | 12 | 0694a93 |
| Demo | 3 | 0.25 | 0.500 | 3878.5 | 2.173 | -183.5 | 0.896 | -2579 | -2356 | 16.38 | 11.45 | 8.09e-04 | 1.09e-04 | 1.09e+09 | 617.43 | 610.13 | 3810 (25) | 12725 / 27 | 2.80e-07 | 4978 | 15888 / 27 | 3.38e-11 | 2.5e-12 | 2.98e-12 | 17 | 0694a93 |
| Experimental | 10 | 1 | 1.000 | 868.2 | 1.650 | +2826.8 | 0.000 | +432 | +655 | 0.00 | 0.00 | 8.87e-44 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2486 (4) | 6571 / 5 | 3.60e-10 | – | – / – | – | 2.4e-14 | – | 6 | 0694a93 |
| Prototype | 10 | 1 | 1.000 | 1370.8 | 2.136 | +2324.2 | 0.000 | -71 | +152 | 1.34 | 0.00 | 7.50e-25 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2573 (4) | 6834 / 4 | 1.87e-08 | 3857 | 10145 / 4 | 1.32e-08 | 2.2e-12 | 4.28e-12 | 11 | 0694a93 |
| Demo | 10 | 1 | 1.000 | 3027.0 | 2.148 | +668.0 | 0.000 | -1727 | -1504 | 17.23 | 12.05 | 3.38e-07 | 1.09e-04 | 9.11e+09 | 617.43 | 610.13 | 3306 (6) | 10000 / 4 | 3.22e-07 | 4474 | 13151 / 4 | 3.39e-11 | 1.8e-12 | 2.32e-12 | 14 | 0694a93 |
| Experimental | 10 | 0.75 | 0.866 | 948.3 | 1.643 | +2746.7 | 0.000 | +352 | +575 | 0.00 | 0.00 | 1.76e-39 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2514 (5) | 6689 / 5 | 2.39e-08 | – | – / – | – | 2.3e-14 | – | 7 | 0694a93 |
| Prototype | 10 | 0.75 | 0.866 | 1477.3 | 2.126 | +2217.7 | 0.000 | -177 | +46 | 2.35 | 0.00 | 1.54e-22 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2611 (4) | 6996 / 4 | 2.60e-09 | 3895 | 10313 / 4 | 1.32e-08 | 2.5e-12 | 4.25e-12 | 12 | 0694a93 |
| Demo | 10 | 0.75 | 0.866 | 3340.7 | 2.171 | +354.3 | 0.000 | -2041 | -1818 | 16.38 | 11.45 | 7.82e-06 | 1.09e-04 | 8.53e+09 | 617.43 | 610.13 | 3424 (5) | 10419 / 4 | 9.31e-10 | 4592 | 13571 / 4 | 3.38e-11 | 2.0e-12 | 2.52e-12 | 16 | 0694a93 |
| Experimental | 10 | 0.5 | 0.707 | 1086.8 | 1.662 | +2608.2 | 0.000 | +213 | +436 | 0.00 | 0.00 | 1.49e-33 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2560 (4) | 6873 / 5 | 6.40e-09 | – | – / – | – | 2.3e-14 | – | 7 | 0694a93 |
| Prototype | 10 | 0.5 | 0.707 | 1661.4 | 2.142 | +2033.6 | 0.000 | -361 | -138 | 3.36 | 1.34 | 3.01e-19 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2682 (5) | 7271 / 4 | 1.65e-10 | 3966 | 10586 / 4 | 1.32e-08 | 2.7e-12 | 4.67e-12 | 12 | 0694a93 |
| Demo | 10 | 0.5 | 0.707 | 3716.6 | 2.254 | -21.6 | 0.275 | -2417 | -2194 | 15.56 | 9.83 | 2.71e-04 | 1.09e-04 | 2.63e+09 | 617.43 | 610.13 | 3624 (10) | 11475 / 12 | 4.21e-07 | 4792 | 14634 / 12 | 3.38e-11 | 2.2e-12 | 2.71e-12 | 15 | 0694a93 |
| Experimental | 10 | 0.25 | 0.500 | 1394.3 | 1.695 | +2300.7 | 0.000 | -94 | +129 | 0.69 | 0.00 | 1.29e-24 | 2.40e-05 | 4.17e+10 | 300.00 | – | 2669 (6) | 7297 / 5 | 3.15e-09 | – | – / – | – | 2.3e-14 | – | 8 | 0694a93 |
| Prototype | 10 | 0.25 | 0.500 | 2067.4 | 2.200 | +1627.6 | 0.000 | -767 | -544 | 4.34 | 2.74 | 4.56e-14 | 3.55e-05 | 2.82e+10 | 605.68 | 601.12 | 2829 (4) | 7845 / 4 | 2.24e-08 | 4113 | 11176 / 4 | 1.32e-08 | 3.3e-12 | 5.37e-12 | 12 | 0694a93 |
| Demo | 10 | 0.25 | 0.500 | 4013.2 | 2.212 | -318.2 | 1.478 | -2713 | -2490 | 11.45 | 8.87 | 2.02e-03 | 1.09e-04 | 4.71e+08 | 617.43 | 610.13 | 3925 (34) | 13165 / 16 | 2.36e-07 | 5093 | 16344 / 16 | 3.38e-11 | 2.7e-12 | 3.25e-12 | 18 | 0694a93 |

## Demo single shots from the converged 10 Hz inter-shot state (hot start: uniform 921.1 K)

| scenario | d (µm) | f | √f | peak T (K) | t_peak (µs) | margin to melting (K) | melt (µm) | margin to 1300 K | margin to 1523 K | depth > 1300 K (µm) | depth > 1523 K (µm) | evap. (nm/shot) | sputter (nm/shot) | erosion-limited shots | interface max in shot (K) | pre-shot T next shot (K) | steps to 50 µs (rejected) | Newton iters total / max per step, to 50 µs | last update (K) | steps full cycle | iters total / max, full cycle | last update full cycle (K) | residual 50 µs | residual full cycle | wall (s) | git |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Demo | 0 (no skin) | 1 | 1.000 | 3317.5 | 2.159 | +377.5 | 0.000 | -2017 | -1794 | 33.48 | 19.09 | 6.86e-06 | 1.09e-04 | 8.60e+09 | 936.45 | 930.17 | 3273 (4) | 9737 / 4 | 3.86e-08 | 4441 | 12890 / 4 | 1.55e-11 | 3.7e-12 | 2.14e-12 | 11 | 0694a93 |
| Demo | 1 | 1 | 1.000 | 3317.5 | 2.159 | +377.5 | 0.000 | -2017 | -1794 | 33.48 | 19.09 | 6.86e-06 | 1.09e-04 | 8.60e+09 | 936.45 | 930.17 | 3275 (5) | 9745 / 4 | 3.67e-08 | 4443 | 12902 / 4 | 1.55e-11 | 3.6e-12 | 2.13e-12 | 13 | 0694a93 |
| Demo | 1 | 0.75 | 0.866 | 3376.0 | 2.135 | +319.0 | 0.000 | -2076 | -1853 | 33.48 | 19.09 | 1.15e-05 | 1.09e-04 | 8.27e+09 | 936.45 | 930.17 | 3306 (6) | 9878 / 4 | 1.25e-07 | 4474 | 13033 / 4 | 1.55e-11 | 3.7e-12 | 2.17e-12 | 14 | 0694a93 |
| Demo | 1 | 0.5 | 0.707 | 3495.5 | 2.111 | +199.5 | 0.000 | -2196 | -1973 | 33.48 | 19.09 | 3.14e-05 | 1.09e-04 | 7.10e+09 | 936.45 | 930.17 | 3361 (6) | 10098 / 4 | 2.17e-08 | 4529 | 13255 / 4 | 1.54e-11 | 3.7e-12 | 2.11e-12 | 14 | 0694a93 |
| Demo | 1 | 0.25 | 0.500 | 3713.0 | 2.185 | -18.0 | 0.233 | -2413 | -2190 | 33.48 | 19.09 | 2.52e-04 | 1.09e-04 | 2.77e+09 | 936.45 | 930.17 | 3508 (8) | 11087 / 22 | 1.90e-07 | 4676 | 14248 / 22 | 1.55e-11 | 4.4e-12 | 2.50e-12 | 15 | 0694a93 |
| Demo | 3 | 1 | 1.000 | 3317.4 | 2.154 | +377.6 | 0.000 | -2017 | -1794 | 33.48 | 19.09 | 6.86e-06 | 1.09e-04 | 8.60e+09 | 936.45 | 930.17 | 3279 (7) | 9763 / 4 | 6.65e-08 | 4447 | 12918 / 4 | 1.55e-11 | 3.6e-12 | 2.02e-12 | 13 | 0694a93 |
| Demo | 3 | 0.75 | 0.866 | 3499.2 | 2.125 | +195.8 | 0.000 | -2199 | -1976 | 33.48 | 19.09 | 3.31e-05 | 1.09e-04 | 7.02e+09 | 936.45 | 930.17 | 3349 (4) | 10049 / 4 | 1.06e-10 | 4517 | 13203 / 4 | 1.54e-11 | 3.6e-12 | 2.10e-12 | 14 | 0694a93 |
| Demo | 3 | 0.5 | 0.707 | 3719.1 | 2.185 | -24.1 | 0.288 | -2419 | -2196 | 33.48 | 19.09 | 2.68e-04 | 1.09e-04 | 2.65e+09 | 936.45 | 930.17 | 3486 (8) | 10887 / 24 | 5.42e-09 | 4654 | 14041 / 24 | 1.56e-11 | 4.0e-12 | 2.31e-12 | 15 | 0694a93 |
| Demo | 3 | 0.25 | 0.500 | 3970.7 | 2.167 | -275.7 | 1.207 | -2671 | -2448 | 33.48 | 19.09 | 1.46e-03 | 1.09e-04 | 6.38e+08 | 936.45 | 930.17 | 3780 (39) | 12539 / 21 | 9.37e-09 | 4948 | 15701 / 21 | 1.55e-11 | 4.7e-12 | 2.67e-12 | 17 | 0694a93 |
| Demo | 10 | 1 | 1.000 | 3317.6 | 2.157 | +377.4 | 0.000 | -2018 | -1795 | 33.48 | 19.09 | 6.87e-06 | 1.09e-04 | 8.60e+09 | 936.45 | 930.17 | 3279 (7) | 9760 / 4 | 4.28e-08 | 4447 | 12908 / 4 | 1.55e-11 | 3.5e-12 | 2.13e-12 | 12 | 0694a93 |
| Demo | 10 | 0.75 | 0.866 | 3622.0 | 2.182 | +73.0 | 0.000 | -2322 | -2099 | 33.48 | 19.09 | 8.84e-05 | 1.09e-04 | 5.06e+09 | 936.45 | 930.17 | 3392 (6) | 10195 / 4 | 1.11e-07 | 4560 | 13345 / 4 | 1.55e-11 | 3.8e-12 | 2.26e-12 | 14 | 0694a93 |
| Demo | 10 | 0.5 | 0.707 | 3824.6 | 2.229 | -129.6 | 0.766 | -2525 | -2302 | 31.81 | 18.14 | 6.20e-04 | 1.09e-04 | 1.37e+09 | 936.45 | 930.17 | 3511 (4) | 11046 / 19 | 2.65e-09 | 4679 | 14205 / 19 | 1.54e-11 | 4.1e-12 | 2.45e-12 | 15 | 0694a93 |
| Demo | 10 | 0.25 | 0.500 | 4101.9 | 2.196 | -406.9 | 1.799 | -2802 | -2579 | 30.23 | 14.79 | 3.41e-03 | 1.09e-04 | 2.84e+08 | 936.45 | 930.17 | 3906 (46) | 13229 / 21 | 4.69e-08 | 5074 | 16401 / 21 | 1.54e-11 | 5.0e-12 | 2.79e-12 | 17 | 0694a93 |

## Range of the headline numbers across f = 1.0 → 0.25 (per d)

| scenario | start | d (µm) | peak T at f = 1.0 → 0.25 (K) | Δpeak (K) | margin to melting at f = 0.25 (K) | melt at f = 0.25 (µm) | depth > 1300 K, f = 1.0 → 0.25 (µm) | depth > 1523 K, f = 1.0 → 0.25 (µm) | evaporation f = 1.0 → 0.25 (nm/shot) |
|---|---|---|---|---|---|---|---|---|---|
| Experimental | cold | 1 | 868 → 985 | +117 | +2710 | 0.000 | 0.0 → 0.0 | 0.0 → 0.0 | 8.8e-44 → 9.7e-38 |
| Experimental | cold | 3 | 868 → 1237 | +369 | +2458 | 0.000 | 0.0 → 0.0 | 0.0 → 0.0 | 8.8e-44 → 1.3e-28 |
| Experimental | cold | 10 | 868 → 1394 | +526 | +2301 | 0.000 | 0.0 → 0.7 | 0.0 → 0.0 | 8.9e-44 → 1.3e-24 |
| Prototype | cold | 1 | 1371 → 1522 | +151 | +2173 | 0.000 | 1.3 → 1.3 | 0.0 → 0.0 | 7.5e-25 → 1.1e-21 |
| Prototype | cold | 3 | 1371 → 1852 | +481 | +1843 | 0.000 | 1.3 → 2.4 | 0.0 → 1.4 | 7.5e-25 → 1.6e-16 |
| Prototype | cold | 10 | 1371 → 2067 | +697 | +1628 | 0.000 | 1.3 → 4.3 | 0.0 → 2.7 | 7.5e-25 → 4.6e-14 |
| Demo | cold | 1 | 3027 → 3562 | +536 | +133 | 0.000 | 17.2 → 17.2 | 12.1 → 12.1 | 3.4e-07 → 4.9e-05 |
| Demo | cold | 3 | 3027 → 3879 | +852 | -184 | 0.896 | 17.2 → 16.4 | 12.1 → 11.5 | 3.4e-07 → 8.1e-04 |
| Demo | cold | 10 | 3027 → 4013 | +986 | -318 | 1.478 | 17.2 → 11.5 | 12.1 → 8.9 | 3.4e-07 → 2.0e-03 |
| Demo | hot | 1 | 3317 → 3713 | +396 | -18 | 0.233 | 33.5 → 33.5 | 19.1 → 19.1 | 6.9e-06 → 2.5e-04 |
| Demo | hot | 3 | 3317 → 3971 | +653 | -276 | 1.207 | 33.5 → 33.5 | 19.1 → 19.1 | 6.9e-06 → 1.5e-03 |
| Demo | hot | 10 | 3318 → 4102 | +784 | -407 | 1.799 | 33.5 → 30.2 | 19.1 → 14.8 | 6.9e-06 → 3.4e-03 |

## Indicators that do not depend on the thermal run (unchanged)

- Helium damage-fluence count (Section 5.5): Demo 9 shots to 10¹⁸ m⁻² and 128 to 10¹⁹ m⁻² with the retention model, 8 and 83 at 100 % retention; Prototype 26 and 264; Experimental no count (single-shot design). Set by the ion spectrum, not by the W properties.
- Mitteau design rule (Section 5.5): 3.0×10⁴, 1.0×10³ and 9.5 cycles; arithmetic on published heat flux factors, no model output.
- Sputtering (Table 9): 2.0×10⁻⁵, 2.9×10⁻⁵ and 9.1×10⁻⁵ nm per shot; property-independent. The erosion-limited count above changes only through the evaporation.
- The document has no failure criterion; these are indicators (Section 5.5).

## Multi-shot check (production driver `run_w_multi.py`, unchanged solver options on the command line)

| run | scenario | shots | converged | pre-shot rise last 10 % (K/shot) | first / last peak (K) | pre-shot surface at the last shot (K) | max interface (K) | interface at the compared shot: this run / pure-W baseline (K) | pre-shot surface at that shot: this run / baseline (K) | peak at that shot: this run / baseline (K) | max melt (µm) | evaporation last shot (nm) | residual | Newton iters total / max per step | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multi_Demo_d10_f0.25 | Demo | 100 | False | 0.999 | 4013 / 4066 | 793.2 | 807.05 | shot 100: 807.05 / 807.06 | 793.16 / 793.16 | 4066 / 3203 | 1.716 | 2.76e-03 | 8.3e-09 | 1644482 / 34 | 2918 |

## Commands and git hashes

| run | command | git | start | end | return code |
|---|---|---|---|---|---|
| cold/d1_f1 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=1 skin_factor=1 --out=results/skin_sweep/single/cold/d1_f1` | 0694a93 | 2026-10-22 08:49:19 | 2026-10-22 08:49:55 | 0 |
| cold/d1_f0.75 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=1 skin_factor=0.75 --out=results/skin_sweep/single/cold/d1_f0.75` | 0694a93 | 2026-10-22 08:49:57 | 2026-10-22 08:50:35 | 0 |
| cold/d1_f0.5 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=1 skin_factor=0.5 --out=results/skin_sweep/single/cold/d1_f0.5` | 0694a93 | 2026-10-22 08:50:38 | 2026-10-22 08:51:16 | 0 |
| cold/d1_f0.25 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=1 skin_factor=0.25 --out=results/skin_sweep/single/cold/d1_f0.25` | 0694a93 | 2026-10-22 08:51:18 | 2026-10-22 08:51:56 | 0 |
| cold/d3_f1 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=3 skin_factor=1 --out=results/skin_sweep/single/cold/d3_f1` | 0694a93 | 2026-10-22 08:51:58 | 2026-10-22 08:52:34 | 0 |
| cold/d3_f0.75 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=3 skin_factor=0.75 --out=results/skin_sweep/single/cold/d3_f0.75` | 0694a93 | 2026-10-22 08:52:36 | 2026-10-22 08:53:14 | 0 |
| cold/d3_f0.5 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=3 skin_factor=0.5 --out=results/skin_sweep/single/cold/d3_f0.5` | 0694a93 | 2026-10-22 08:53:16 | 2026-10-22 08:53:56 | 0 |
| cold/d3_f0.25 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=3 skin_factor=0.25 --out=results/skin_sweep/single/cold/d3_f0.25` | 0694a93 | 2026-10-22 08:53:58 | 2026-10-22 08:54:40 | 0 |
| cold/d10_f1 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=10 skin_factor=1 --out=results/skin_sweep/single/cold/d10_f1` | 0694a93 | 2026-10-22 08:54:42 | 2026-10-22 08:55:18 | 0 |
| cold/d10_f0.75 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=10 skin_factor=0.75 --out=results/skin_sweep/single/cold/d10_f0.75` | 0694a93 | 2026-10-22 08:55:20 | 2026-10-22 08:56:00 | 0 |
| cold/d10_f0.5 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=10 skin_factor=0.5 --out=results/skin_sweep/single/cold/d10_f0.5` | 0694a93 | 2026-10-22 08:56:02 | 2026-10-22 08:56:42 | 0 |
| cold/d10_f0.25 | `python scripts/run_w_single.py Experimental Prototype Demo skin_um=10 skin_factor=0.25 --out=results/skin_sweep/single/cold/d10_f0.25` | 0694a93 | 2026-10-22 08:56:44 | 2026-10-22 08:57:28 | 0 |
| Demo_hot/ref_noskin | `python scripts/run_w_single.py Demo T_init=920.9 --out=results/skin_sweep/single/Demo_hot/ref_noskin` | 0694a93 | 2026-10-22 08:57:30 | 2026-10-22 08:57:46 | 0 |
| Demo_hot/d1_f1 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=1 skin_factor=1 --out=results/skin_sweep/single/Demo_hot/d1_f1` | 0694a93 | 2026-10-22 08:57:48 | 2026-10-22 08:58:06 | 0 |
| Demo_hot/d1_f0.75 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=1 skin_factor=0.75 --out=results/skin_sweep/single/Demo_hot/d1_f0.75` | 0694a93 | 2026-10-22 08:58:08 | 2026-10-22 08:58:26 | 0 |
| Demo_hot/d1_f0.5 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=1 skin_factor=0.5 --out=results/skin_sweep/single/Demo_hot/d1_f0.5` | 0694a93 | 2026-10-22 08:58:28 | 2026-10-22 08:58:46 | 0 |
| Demo_hot/d1_f0.25 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=1 skin_factor=0.25 --out=results/skin_sweep/single/Demo_hot/d1_f0.25` | 0694a93 | 2026-10-22 08:58:48 | 2026-10-22 08:59:08 | 0 |
| Demo_hot/d3_f1 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=3 skin_factor=1 --out=results/skin_sweep/single/Demo_hot/d3_f1` | 0694a93 | 2026-10-22 08:59:10 | 2026-10-22 08:59:28 | 0 |
| Demo_hot/d3_f0.75 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=3 skin_factor=0.75 --out=results/skin_sweep/single/Demo_hot/d3_f0.75` | 0694a93 | 2026-10-22 08:59:30 | 2026-10-22 08:59:48 | 0 |
| Demo_hot/d3_f0.5 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=3 skin_factor=0.5 --out=results/skin_sweep/single/Demo_hot/d3_f0.5` | 0694a93 | 2026-10-22 08:59:50 | 2026-10-22 09:00:10 | 0 |
| Demo_hot/d3_f0.25 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=3 skin_factor=0.25 --out=results/skin_sweep/single/Demo_hot/d3_f0.25` | 0694a93 | 2026-10-22 09:00:12 | 2026-10-22 09:00:34 | 0 |
| Demo_hot/d10_f1 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=10 skin_factor=1 --out=results/skin_sweep/single/Demo_hot/d10_f1` | 0694a93 | 2026-10-22 09:00:36 | 2026-10-22 09:00:52 | 0 |
| Demo_hot/d10_f0.75 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=10 skin_factor=0.75 --out=results/skin_sweep/single/Demo_hot/d10_f0.75` | 0694a93 | 2026-10-22 09:00:54 | 2026-10-22 09:01:14 | 0 |
| Demo_hot/d10_f0.5 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=10 skin_factor=0.5 --out=results/skin_sweep/single/Demo_hot/d10_f0.5` | 0694a93 | 2026-10-22 09:01:16 | 2026-10-22 09:01:36 | 0 |
| Demo_hot/d10_f0.25 | `python scripts/run_w_single.py Demo T_init=920.9 skin_um=10 skin_factor=0.25 --out=results/skin_sweep/single/Demo_hot/d10_f0.25` | 0694a93 | 2026-10-22 09:01:38 | 2026-10-22 09:02:00 | 0 |
| multi_Demo_d10_f0.25 | `python scripts/run_w_multi.py Demo 100 skin_um=10 skin_factor=0.25 --out=results/skin_sweep/multi_Demo_d10_f0.25` | see multi_*/COMMAND.txt | – | – | – |

Reference runs: cold no-skin = `regression_check/` (run with the working tree that became commit 0be5c47, before it was committed, so its JSON files record the previous commit f728ed4); the 27 Sep baseline values are in results/baseline_pureW/w_single.

## Notes

- f = 1.0 with a skin is the mesh-only check: the three-layer mesh (skin cells from 1 nm growing by ×1.05, bulk starting at the last skin cell) differs from the two-layer production mesh; its difference from the no-skin reference is the mesh effect.
- The hot start sets the whole wall, steel included, to 921.1 K, so its interface column is not the converged interface state; the multi-shot check is the interface test (Section 5.2).
- `skin_status` in the JSON files is the 29 Sep label of the option (`annealed skin` for f = 1.0, `helium-bubble skin` otherwise); here every factor is a hypothetical surface-state factor.
- No failure criterion exists in the document; `erosion-limited shots` is 1 mm divided by the per-shot loss (evaporation + sputtering).
