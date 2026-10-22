# Multi-shot Li film runs: summary

Film replaced at full thickness and T_Li at the start of every shot after the first; steel carried forward; coolant fixed at T_c. Peak steel T is the maximum over the steel cells during the shot; 'pass' means the maximum over the whole run stays below 925 K.

## Demo

| δ (µm) | shots | converged (rise K/shot) | peak steel, shot 1 | max over run | last shot | first shot > 925 K | pre-shot steel front, last | fit asymptote of peak steel (τ shots) | Li evaporated, shot 1 → last (kg/m²) | residual |
|---|---|---|---|---|---|---|---|---|---|---|
| 17.8 | 390 | True (0.047) | 1116 K | 1212 K | 1212 K | 1 | 795 K | 1214 K (107) | 7.57e-04 → 8.45e-04 | -6.3e-13 |
| 31.6 | 370 | True (0.046) | 907 K | 1006 K | 1006 K | 9 | 783 K | 1008 K (101) | 7.45e-04 → 7.86e-04 | 3.1e-13 |
| 56.2 | 320 | True (0.049) | 781 K | 871 K | 871 K | never | 759 K | 873 K (91) | 7.43e-04 → 7.58e-04 | 8.2e-15 |
| 100 | 260 | True (0.050) | 707 K | 796 K | 796 K | never | 728 K | 799 K (76) | 7.43e-04 → 7.48e-04 | 3.0e-13 |
| 1e+03 | 100 | True (0.012) | 646 K | 667 K | 667 K | never | 626 K | 667 K (32) | 7.43e-04 → 7.43e-04 | 3.3e-13 |

Multi-shot minimum film thickness (Demo): 44.7 µm (bracket (31.622776601683796, 56.23413251903491))

## Prototype

| δ (µm) | shots | converged (rise K/shot) | peak steel, shot 1 | max over run | last shot | first shot > 925 K | pre-shot steel front, last | fit asymptote of peak steel (τ shots) | Li evaporated, shot 1 → last (kg/m²) | residual |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 100 | True (0.000) | 964 K | 969 K | 969 K | 1 | 609 K | 969 K (12) | 1.09e-05 → 1.13e-05 | 3.8e-13 |
| 17.8 | 100 | True (0.000) | 829 K | 834 K | 834 K | never | 609 K | 834 K (11) | 9.64e-06 → 9.85e-06 | 4.2e-13 |
| 31.6 | 100 | True (0.000) | 737 K | 742 K | 742 K | never | 609 K | 742 K (11) | 9.11e-06 → 9.22e-06 | 4.3e-13 |
| 56.2 | 100 | True (0.000) | 680 K | 685 K | 685 K | never | 609 K | 685 K (11) | 9.09e-06 → 9.18e-06 | 6.0e-13 |

Multi-shot minimum film thickness (Prototype): 12.1 µm (bracket (10.0, 17.78279410038923))

