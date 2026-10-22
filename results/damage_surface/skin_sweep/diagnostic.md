# A1 diagnostic: diffusion lengths against deposition depths (16 Oct 2026)

Convention: L = sqrt(alpha t), alpha = k/(rho c_p); solid-W fits of Tolias 2017 (library.tungsten), Li of Davison 1968.

## Tungsten: alpha = k/(rho c_p)

| T (K) | k (W/m/K) | rho (kg/m3) | c_p (J/kg/K) | alpha (m2/s) |
|---|---|---|---|---|
| 300 | 179.9 | 19248 | 131.3 | 7.119e-05 |
| 600 | 137.3 | 19168 | 139.4 | 5.139e-05 |
| 920.9 | 122.2 | 19079 | 146.7 | 4.364e-05 |
| 1500 | 107.6 | 18908 | 158.6 | 3.589e-05 |
| 3000 | 92.1 | 18318 | 224.8 | 2.238e-05 |

## Diffusion length sqrt(alpha t) in W (um) by load component and temperature

| component | t | 300 K | 600 K | 920.9 K | 1500 K | 3000 K | source |
|---|---|---|---|---|---|---|---|
| x-ray pulse | 2e-08 s | 1.19 | 1.01 | 0.934 | 0.847 | 0.669 | 20 ns square pulse, baseline (Section 3.6) |
| first ions to the wall | 1.05e-07 s | 2.73 | 2.32 | 2.14 | 1.94 | 1.53 | first ion arrival at R = 6.5 m (Section 3.7; runs/source_term) |
| burn ions from about 0.4 us (Sec. 5.1 wording) | 4e-07 s | 5.34 | 4.53 | 4.18 | 3.79 | 2.99 | Section 5.1: 'the burn ions from about 0.4 us'; burn t10-t90 = 0.22-0.77 us at 6.5 m (runs/source_term/source_term_checks.json) |
| debris-ion peak, 1.5-2.0 us | 2e-06 s | 11.9 | 10.1 | 9.34 | 8.47 | 6.69 | peak surface temperature at 1.51, 1.94, 1.97 us (Section 5.1); debris energy-median arrival 1.63 us (Section 3.7) |
| 10-90 % arrival window of all ions, Demo | 1.85e-06 s | 11.5 | 9.74 | 8.98 | 8.14 | 6.43 | t10-t90 = 1847 ns (runs/source_term/garoz_comparison.md, 'all / total') |
| all ions arrived | 3.86e-05 s | 52.4 | 44.5 | 41 | 37.2 | 29.4 | last arrival 38.6 us at R = 6.5 m (Section 3.7) |
| end of the first integration part | 5e-05 s | 59.7 | 50.7 | 46.7 | 42.4 | 33.5 | 50 us (Section 3.2) |
| Li runs, end of integration | 0.005 s | 597 | 507 | 467 | 424 | 335 | 5 ms (Section 3.2, Table 3) |
| Demo shot period | 0.1 s | 2.67e+03 | 2.27e+03 | 2.09e+03 | 1.89e+03 | 1.5e+03 | 10 Hz |
| Prototype shot period | 1 s | 8.44e+03 | 7.17e+03 | 6.61e+03 | 5.99e+03 | 4.73e+03 | 1 Hz |

## Liquid lithium at 600 K (alpha = 2.343e-05 m2/s)

| component | t | L_Li (um) |
|---|---|---|
| x-ray pulse | 2e-08 s | 0.6845 |
| first ions to the wall | 1.05e-07 s | 1.568 |
| burn ions from about 0.4 us (Sec. 5.1 wording) | 4e-07 s | 3.061 |
| debris-ion peak, 1.5-2.0 us | 2e-06 s | 6.845 |
| 10-90 % arrival window of all ions, Demo | 1.85e-06 s | 6.578 |
| all ions arrived | 3.86e-05 s | 30.07 |
| end of the first integration part | 5e-05 s | 34.23 |
| Li runs, end of integration | 0.005 s | 342.3 |
| Demo shot period | 0.1 s | 1531 |
| Prototype shot period | 1 s | 4840 |

## Deposition depths in W

- X-ray attenuation length at the 7.09 keV number-mean photon energy: 2.31 um (mu/rho = 224.9 cm2/g, photoelectric + incoherent, log-log between the XCOM rows at 6 and 8 keV; rho = 19 250 kg/m3).
- Fraction of the threat energy (x-rays + ions, Demo, W) deposited within a depth (runs/deposition/fractions.json): 0.01 um: 0.8 %, 0.1 um: 8.5 %, 1 um: 54.7 %, 10 um: 82.4 %, 100 um: 97.2 %.

| component | energy-weighted mean depth (um) | x50 (um) | x90 (um) |
|---|---|---|---|
| x-rays | 18.4 | 3.19 | 47.1 |
| burn Protons | 18 | 11 | 25.9 |
| burn Deuterons | 48.6 | 31.9 | 124 |
| burn Tritons | 31 | 19.3 | 68.8 |
| burn 3He | 0.599 | 0.48 | 1.1 |
| burn 4He | 1.75 | 1.56 | 3.39 |
| debris Protons | 0.282 | 0.234 | 0.602 |
| debris Deuterons | 0.402 | 0.33 | 0.829 |
| debris Tritons | 0.519 | 0.414 | 1.11 |
| debris 3He | 0.062 | 0.05 | 0.13 |
| debris 4He | 0.398 | 0.335 | 0.81 |
| debris 12C | 0.168 | 0.151 | 0.339 |
| debris Au | 0.393 | 0.39 | 0.673 |

Source: runs/deposition/fractions.json (27 Sep 2026, Demo loads on W; the static calculation used the TRIM-profile path for keV light ions, Section 3.8).
