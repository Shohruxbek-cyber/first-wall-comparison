# Alloy study: summary (2026-10-22 16:38, commit 0694a93)

W_pure regression against results/baseline_pureW: **bit-identical** (keys: peak_surface_T_K, t_peak_us, melt_depth_um, ablated_nm_per_shot, energy_rel_residual, pre_shot2_surface_T_K, energy_rel_residual_full_cycle, diagnostics, ledger_J_m2; history, max-T profile and He-layer CSVs).

## Single shots, cold start

| variant | scenario | T_init (K) | k factor | status | peak surface (K) | melt (µm) | evap. (nm/shot) | d(1300 K) (µm) | T_rx (K) | time > T_rx, surface (µs) [−100 K / +100 K] | depth > T_rx (µm) | t_eq/shot at T_rx (s) | shots to 1 h onset, surface [Q 322 / 350 kJ; T_rx −100 / +100 K] | shots to onset, 1 µm | rx depth after 1e5 / 1e6 / 1e7 shots (µm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W_K | Demo | 600.0 | 1.00 | SOURCED k factor 1.0 | 3026.9 | 0.000 | 3.38e-07 | 17.23 | 1623 | 6.103 [7.352 / 5.188] | 10.34 | 8.70e-02 | 4.14e+04 [6.55e+04 / 2.61e+04; 8075 / 1.76e+05] | 8.66e+04 | 1.09 / 3.19 / 5.32 |
| W_K | Experimental | 300.0 | 1.00 | SOURCED k factor 1.0 | 868.2 | 0.000 | 8.83e-44 | 0.00 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 1.59e-16 | 2.26e+19 [9.00e+18 / 5.66e+19; 4.40e+18 / 9.58e+19] | 1.82e+20 | 0.00 / 0.00 / 0.00 |
| W_K | Prototype | 600.0 | 1.00 | SOURCED k factor 1.0 | 1370.8 | 0.000 | 7.48e-25 | 1.34 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 7.00e-09 | 5.15e+11 [4.17e+11 / 6.34e+11; 1.00e+11 / 2.18e+12] | 1.44e+12 | 0.00 / 0.00 / 0.00 |
| W_K_k0.9 | Demo | 600.0 | 0.90 | SOURCED k factor 1.0 | 3142.7 | 0.000 | 1.16e-06 | 17.23 | 1623 | 6.692 [8.079 / 5.667] | 10.34 | 1.44e-01 | 2.50e+04 [4.03e+04 / 1.54e+04; 4867 / 1.06e+05] | 5.33e+04 | 1.56 / 3.54 / 5.60 |
| W_K_k0.9 | Experimental | 300.0 | 0.90 | SOURCED k factor 1.0 | 897.9 | 0.000 | 4.22e-42 | 0.00 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 7.38e-16 | 4.88e+18 [2.07e+18 / 1.15e+19; 9.52e+17 / 2.07e+19] | 4.32e+19 | 0.00 / 0.00 / 0.00 |
| W_K_k0.9 | Prototype | 600.0 | 0.90 | SOURCED k factor 1.0 | 1410.4 | 0.000 | 5.93e-24 | 1.82 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 1.59e-08 | 2.26e+11 [1.90e+11 / 2.69e+11; 4.41e+10 / 9.58e+11] | 6.69e+11 | 0.00 / 0.00 / 0.00 |
| W_K_k1.1 | Demo | 600.0 | 1.10 | SOURCED k factor 1.0 | 2925.5 | 0.000 | 1.06e-07 | 17.23 | 1623 | 5.624 [6.760 / 4.795] | 10.34 | 5.41e-02 | 6.66e+04 [1.03e+05 / 4.28e+04; 1.30e+04 / 2.82e+05] | 1.37e+05 | 0.69 / 2.88 / 5.06 |
| W_K_k1.1 | Experimental | 300.0 | 1.10 | SOURCED k factor 1.0 | 842.6 | 0.000 | 2.51e-45 | 0.00 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 3.89e-17 | 9.26e+19 [3.48e+19 / 2.46e+20; 1.81e+19 / 3.93e+20] | 6.87e+20 | 0.00 / 0.00 / 0.00 |
| W_K_k1.1 | Prototype | 600.0 | 1.10 | SOURCED k factor 1.0 | 1336.6 | 0.000 | 1.13e-25 | 0.85 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 3.31e-09 | 1.09e+12 [8.56e+11 / 1.38e+12; 2.12e+11 / 4.62e+12] | 2.90e+12 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p0.8 | Demo | 600.0 | 0.80 | PROXY k factor 0.8 | 3277.0 | 0.000 | 4.34e-06 | 17.23 | 1823 | 5.405 [6.272 / 4.736] | 8.01 | 1.62e-02 | 2.22e+05 [3.27e+05 / 1.51e+05; 6.14e+04 / 7.04e+05] | 4.86e+05 | 0.00 / 1.56 / 3.54 |
| W_ODS_La2O3_p0.8 | Experimental | 300.0 | 0.80 | PROXY k factor 0.8 | 932.8 | 0.000 | 2.91e-40 | 0.00 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 2.57e-16 | 1.40e+19 [5.69e+18 / 3.44e+19; 3.86e+18 / 4.43e+19] | 1.38e+20 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p0.8 | Prototype | 600.0 | 0.80 | PROXY k factor 0.8 | 1456.9 | 0.000 | 5.86e-23 | 2.23 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 2.58e-09 | 1.40e+12 [1.08e+12 / 1.79e+12; 3.85e+11 / 4.42e+12] | 4.40e+12 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p0.9 | Demo | 600.0 | 0.90 | PROXY k factor 0.9 | 3142.7 | 0.000 | 1.16e-06 | 17.23 | 1823 | 4.901 [5.667 / 4.314] | 8.01 | 9.40e-03 | 3.83e+05 [5.52e+05 / 2.66e+05; 1.06e+05 / 1.21e+06] | 8.18e+05 | 0.00 / 1.15 / 3.19 |
| W_ODS_La2O3_p0.9 | Experimental | 300.0 | 0.90 | PROXY k factor 0.9 | 897.9 | 0.000 | 4.22e-42 | 0.00 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 4.81e-17 | 7.49e+19 [2.84e+19 / 1.97e+20; 2.07e+19 / 2.37e+20] | 6.63e+20 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p0.9 | Prototype | 600.0 | 0.90 | PROXY k factor 0.9 | 1410.4 | 0.000 | 5.93e-24 | 1.82 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 1.04e-09 | 3.47e+12 [2.60e+12 / 4.63e+12; 9.58e+11 / 1.10e+13] | 1.03e+13 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p1.0 | Demo | 600.0 | 1.00 | PROXY k factor 1.0 | 3026.9 | 0.000 | 3.38e-07 | 17.23 | 1823 | 4.505 [5.188 / 3.785] | 8.01 | 5.66e-03 | 6.36e+05 [8.98e+05 / 4.49e+05; 1.76e+05 / 2.01e+06] | 1.33e+06 | 0.00 / 0.73 / 2.74 |
| W_ODS_La2O3_p1.0 | Experimental | 300.0 | 1.00 | PROXY k factor 1.0 | 868.2 | 0.000 | 8.83e-44 | 0.00 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 1.04e-17 | 3.47e+20 [1.23e+20 / 9.73e+20; 9.58e+19 / 1.10e+21] | 2.79e+21 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p1.0 | Prototype | 600.0 | 1.00 | PROXY k factor 1.0 | 1370.8 | 0.000 | 7.48e-25 | 1.34 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 4.56e-10 | 7.90e+12 [5.72e+12 / 1.09e+13; 2.18e+12 / 2.50e+13] | 2.21e+13 | 0.00 / 0.00 / 0.00 |
| W_pure | Demo | 600.0 | 1.00 | SOURCED k factor 1.0 | 3026.9 | 0.000 | 3.38e-07 | 17.23 | 1523 | 7.352 [9.072 / 6.103] | 12.05 | 4.46e-01 | 8075 [1.37e+04 / 4761; 1252 / 4.14e+04] | 1.69e+04 | 2.60 / 4.81 / 6.87 |
| W_pure | Experimental | 300.0 | 1.00 | SOURCED k factor 1.0 | 868.2 | 0.000 | 8.83e-44 | 0.00 | 1523 | 0.000 [0.000 / 0.000] | 0.00 | 8.17e-16 | 4.40e+18 [1.88e+18 / 1.03e+19; 6.83e+17 / 2.26e+19] | 3.55e+19 | 0.00 / 0.00 / 0.00 |
| W_pure | Prototype | 600.0 | 1.00 | SOURCED k factor 1.0 | 1370.8 | 0.000 | 7.48e-25 | 1.34 | 1523 | 0.000 [0.000 / 0.000] | 0.00 | 3.59e-08 | 1.00e+11 [8.71e+10 / 1.16e+11; 1.56e+10 / 5.15e+11] | 2.81e+11 | 0.00 / 0.00 / 0.00 |

## Single shots, hot start

| variant | scenario | T_init (K) | k factor | status | peak surface (K) | melt (µm) | evap. (nm/shot) | d(1300 K) (µm) | T_rx (K) | time > T_rx, surface (µs) [−100 K / +100 K] | depth > T_rx (µm) | t_eq/shot at T_rx (s) | shots to 1 h onset, surface [Q 322 / 350 kJ; T_rx −100 / +100 K] | shots to onset, 1 µm | rx depth after 1e5 / 1e6 / 1e7 shots (µm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W_K | Demo | 920.9 | 1.00 | SOURCED k factor 1.0 | 3317.5 | 0.000 | 6.86e-06 | 33.48 | 1623 | 12.472 [16.886 / 9.661] | 16.38 | 3.17e-01 | 1.13e+04 [1.88e+04 / 6833; 2212 / 4.81e+04] | 2.05e+04 | 2.74 / 5.60 / 8.87 |
| W_K | Prototype | 609.5 | 1.00 | SOURCED k factor 1.0 | 1380.6 | 0.000 | 1.28e-24 | 1.49 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 8.68e-09 | 4.15e+11 [3.39e+11 / 5.07e+11; 8.09e+10 / 1.76e+12] | 1.14e+12 | 0.00 / 0.00 / 0.00 |
| W_K_k0.9 | Demo | 920.9 | 0.90 | SOURCED k factor 1.0 | 3428.9 | 0.000 | 1.84e-05 | 33.48 | 1623 | 13.779 [18.686 / 10.652] | 16.38 | 4.77e-01 | 7546 [1.27e+04 / 4474; 1472 / 3.20e+04] | 1.39e+04 | 3.19 / 5.90 / 9.34 |
| W_K_k0.9 | Prototype | 609.5 | 0.90 | SOURCED k factor 1.0 | 1420.2 | 0.000 | 9.82e-24 | 2.02 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 1.95e-08 | 1.84e+11 [1.56e+11 / 2.18e+11; 3.59e+10 / 7.81e+11] | 5.37e+11 | 0.00 / 0.00 / 0.00 |
| W_K_k1.1 | Demo | 920.9 | 1.10 | SOURCED k factor 1.0 | 3220.1 | 0.000 | 2.74e-06 | 33.48 | 1623 | 11.404 [15.414 / 8.853] | 16.38 | 2.17e-01 | 1.66e+04 [2.70e+04 / 1.01e+04; 3229 / 7.02e+04] | 2.93e+04 | 2.47 / 5.32 / 8.43 |
| W_K_k1.1 | Prototype | 609.5 | 1.10 | SOURCED k factor 1.0 | 1346.4 | 0.000 | 1.99e-25 | 1.04 | 1623 | 0.000 [0.000 / 0.000] | 0.00 | 4.15e-09 | 8.68e+11 [6.88e+11 / 1.09e+12; 1.69e+11 / 3.68e+12] | 2.28e+12 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p0.8 | Demo | 920.9 | 0.80 | PROXY k factor 0.8 | 3558.4 | 0.000 | 5.38e-05 | 33.48 | 1823 | 9.505 [11.892 / 7.792] | 12.05 | 4.84e-02 | 7.44e+04 [1.14e+05 / 4.86e+04; 2.06e+04 / 2.36e+05] | 1.41e+05 | 0.59 / 3.04 / 5.90 |
| W_ODS_La2O3_p0.8 | Prototype | 609.5 | 0.80 | PROXY k factor 0.8 | 1466.8 | 0.000 | 9.40e-23 | 2.47 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 3.13e-09 | 1.15e+12 [9.02e+11 / 1.47e+12; 3.18e+11 / 3.65e+12] | 3.58e+12 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p0.9 | Demo | 920.9 | 0.90 | PROXY k factor 0.9 | 3428.9 | 0.000 | 1.84e-05 | 33.48 | 1823 | 8.535 [10.652 / 7.013] | 12.05 | 3.11e-02 | 1.16e+05 [1.74e+05 / 7.70e+04; 3.20e+04 / 3.67e+05] | 2.14e+05 | 0.00 / 2.74 / 5.32 |
| W_ODS_La2O3_p0.9 | Prototype | 609.5 | 0.90 | PROXY k factor 0.9 | 1420.2 | 0.000 | 9.82e-24 | 2.02 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 1.27e-09 | 2.83e+12 [2.13e+12 / 3.74e+12; 7.81e+11 / 8.95e+12] | 8.24e+12 | 0.00 / 0.00 / 0.00 |
| W_ODS_La2O3_p1.0 | Demo | 920.9 | 1.00 | PROXY k factor 1.0 | 3317.5 | 0.000 | 6.86e-06 | 33.48 | 1823 | 7.760 [9.661 / 6.391] | 12.05 | 2.07e-02 | 1.74e+05 [2.57e+05 / 1.18e+05; 4.81e+04 / 5.51e+05] | 3.14e+05 | 0.00 / 2.35 / 5.06 |
| W_ODS_La2O3_p1.0 | Prototype | 609.5 | 1.00 | PROXY k factor 1.0 | 1380.6 | 0.000 | 1.28e-24 | 1.49 | 1823 | 0.000 [0.000 / 0.000] | 0.00 | 5.66e-10 | 6.37e+12 [4.65e+12 / 8.71e+12; 1.76e+12 / 2.02e+13] | 1.76e+13 | 0.00 / 0.00 / 0.00 |
| W_pure | Demo | 920.9 | 1.00 | SOURCED k factor 1.0 | 3317.5 | 0.000 | 6.86e-06 | 33.48 | 1523 | 16.886 [24.269 / 12.472] | 19.09 | 1.63e+00 | 2212 [3923 / 1245; 343 / 1.13e+04] | 3993 | 4.81 / 8.01 / 11.45 |
| W_pure | Prototype | 609.5 | 1.00 | SOURCED k factor 1.0 | 1380.6 | 0.000 | 1.28e-24 | 1.49 | 1523 | 0.000 [0.000 / 0.000] | 0.00 | 4.45e-08 | 8.09e+10 [7.08e+10 / 9.23e+10; 1.25e+10 / 4.15e+11] | 2.23e+11 | 0.00 / 0.00 / 0.00 |

## Inter-shot (pre-shot) temperature against the sourced DBTT values

| grade | scenario | start | pre-shot T (K) | bending_5pct | bending_5pct_KIT | charpy | charpy_WL10 | charpy_rod | fracture_toughness | irradiated_bending | recrystallized_small_punch | recrystallized_tensile_shift | tensile | tensile_other_ODS | thin_foil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W_K | Demo | cold | 600.0 | ductile (above DBTT) | ductile (above DBTT) | within +/-50 K of DBTT | – | brittle (below DBTT) | – | within +/-50 K of DBTT | – | shift (range) | ductile (above DBTT) | – | ductile (above DBTT) |
| W_K | Demo | hot | 920.9 | ductile (above DBTT) | ductile (above DBTT) | ductile (above DBTT) | – | brittle (below DBTT) | – | ductile (above DBTT) | – | shift (range) | ductile (above DBTT) | – | ductile (above DBTT) |
| W_K | Experimental | cold | 300.0 | within +/-50 K of DBTT | brittle (below DBTT) | brittle (below DBTT) | – | brittle (below DBTT) | – | brittle (below DBTT) | – | shift (range) | within +/-50 K of DBTT | – | ductile (above DBTT) |
| W_K | Prototype | cold | 600.0 | ductile (above DBTT) | ductile (above DBTT) | within +/-50 K of DBTT | – | brittle (below DBTT) | – | within +/-50 K of DBTT | – | shift (range) | ductile (above DBTT) | – | ductile (above DBTT) |
| W_K | Prototype | hot | 609.5 | ductile (above DBTT) | ductile (above DBTT) | within +/-50 K of DBTT | – | brittle (below DBTT) | – | within +/-50 K of DBTT | – | shift (range) | ductile (above DBTT) | – | ductile (above DBTT) |
| W_ODS_La2O3 | Demo | cold | 600.0 | – | – | brittle (below DBTT) | brittle (below DBTT) | – | – | – | – | – | ductile (above DBTT) | shift (range) | – |
| W_ODS_La2O3 | Demo | hot | 920.9 | – | – | ductile (above DBTT) | brittle (below DBTT) | – | – | – | – | – | ductile (above DBTT) | shift (range) | – |
| W_ODS_La2O3 | Experimental | cold | 300.0 | – | – | brittle (below DBTT) | brittle (below DBTT) | – | – | – | – | – | within +/-50 K of DBTT | shift (range) | – |
| W_ODS_La2O3 | Prototype | cold | 600.0 | – | – | brittle (below DBTT) | brittle (below DBTT) | – | – | – | – | – | ductile (above DBTT) | shift (range) | – |
| W_ODS_La2O3 | Prototype | hot | 609.5 | – | – | brittle (below DBTT) | brittle (below DBTT) | – | – | – | – | – | ductile (above DBTT) | shift (range) | – |
| W_pure | Demo | cold | 600.0 | ductile (above DBTT) | – | brittle (below DBTT) | – | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | shift (range) | ductile (above DBTT) | – | – |
| W_pure | Demo | hot | 920.9 | ductile (above DBTT) | – | ductile (above DBTT) | – | brittle (below DBTT) | ductile (above DBTT) | ductile (above DBTT) | ductile (above DBTT) | shift (range) | ductile (above DBTT) | – | – |
| W_pure | Experimental | cold | 300.0 | brittle (below DBTT) | – | brittle (below DBTT) | – | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | shift (range) | brittle (below DBTT) | – | – |
| W_pure | Prototype | cold | 600.0 | ductile (above DBTT) | – | brittle (below DBTT) | – | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | shift (range) | ductile (above DBTT) | – | – |
| W_pure | Prototype | hot | 609.5 | ductile (above DBTT) | – | brittle (below DBTT) | – | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | brittle (below DBTT) | shift (range) | ductile (above DBTT) | – | – |

## Multi-shot runs (production script `run_w_multi.py` with the alloy options)

| run | scenario | shots | converged | last pre-shot surface (K) | last peak (K) | max interface (K) | max melt (µm) | cum. evaporation (nm) | residual |
|---|---|---|---|---|---|---|---|---|---|
| Demo_W_ODS_La2O3_p0.8 | Demo | 550 | True | 919.8 | 3557 | 931 | 0.000 | 2.06e-02 | -6.1e-08 |
| Demo_W_ODS_La2O3_p0.8_uninterrupted | Demo | 550 | True | 920.1 | 3676 | 932 | 0.000 | 4.89e-02 | -6.4e-08 |
| Prototype_W_K | Prototype | 100 | True | 609.5 | 1381 | 615 | 0.000 | 1.20e-22 | -1.7e-07 |
| Prototype_W_ODS_La2O3_p0.8 | Prototype | 100 | True | 609.5 | 1467 | 615 | 0.000 | 8.92e-21 | -6.3e-08 |

## Helium damage-fluence indicator (existing machinery, `scripts/he_shot_count.py`, all He, conserving spectra)

Shots for the retained He fluence to reach the GA2020 pulsed-damage range 1e18–1e19 m⁻². The alloy option `he_threshold_factor` multiplies the threshold fluence; **no sourced value exists for W-K or ODS-W**, so the 'on' columns are only the scaling rule (factor not set: option off).

| scenario | retained He per shot (m⁻²) | shots to 1e18 (off) | shots to 1e19 (off) | shots to 1e18 (on) | shots to 1e19 (on) |
|---|---|---|---|---|---|
| Experimental | 2.23e+16 | 44.9 | 449 | – | – |
| Prototype | 3.3e+16 | 30.3 | 303 | – | – |
| Demo | 1.02e+17 | 9.85 | 98.5 | – | – |

## W armor and Li film side by side

W rows: this task (Prototype and Demo from the converged inter-shot state of the pure-W multi-shot baseline; Experimental first shot). Li rows: the lithium results of the paper (Table 5) (minimum film for steel < 925 K, single shot on 600 K steel; thick-film evaporation); multi-shot minima from runs/li_multi/summary.json (6 Oct 2026). The alloy option changes only the W rows that name a grade.

| scenario | W peak surface T, pure / ODS proxy 0.8 (K) | W melting (any grade) | W evaporation per shot, pure / ODS 0.8 (nm) | W sputtering per shot (nm, grade-independent) | W shots to consume 1 mm (evap.+sputter) | W shots to He 1e18 / 1e19 m⁻² (all grades) | W shots to recrystallization onset at the surface: pure / W-K / ODS (k 0.8–1.0) | W/steel interface max vs 925 K (pure-W baseline) | Li minimum film, single / multi-shot (µm) | Li evaporated per shot (kg/m²) |
|---|---|---|---|---|---|---|---|---|---|---|
| Experimental (first shot) | 868 / 933 | no | 8.8e-44 / 2.9e-40 | 2.4e-05 | 4.17e+10 | 45 / 449 | 4.40e+18 / 2.26e+19 / 1.40e+19–3.47e+20 | single shot | 7 / – (single shots only) | 6.8e-07 |
| Prototype (converged inter-shot state) | 1381 / 1467 | no | 1.3e-24 / 9.4e-23 | 3.6e-05 | 2.82e+10 | 30 / 303 | 8.09e+10 / 4.15e+11 / 1.15e+12–6.37e+12 | -310 K | 12 / 12.1 | 7.9e-06 |
| Demo (converged inter-shot state) | 3317 / 3558 | no | 6.9e-06 / 5.4e-05 | 1.1e-04 | 8.60e+09 | 10 / 98 | 2212 / 1.13e+04 / 7.44e+04–1.74e+05 | +9 K | 31 / 44.7 | 7.1e-04 |

## Notes

- thermal results of W_K equal those of W_pure by construction (sourced k factor 1.0)
- W_ODS_La2O3 conductivity is a PROXY band, not a measurement
- shots_to_onset = 3600 s / Arrhenius-equivalent time per shot at T_rx (isokinetic, single Q, no incubation)
- dbtt_regime compares the pre-shot (inter-shot minimum) temperature with sourced DBTT values; it is not a fracture model
