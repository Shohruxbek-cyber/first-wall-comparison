# SRIM-2012 stopping tables and TRIM runs

The model takes the stopping and range of the target ions from 21 SRIM stopping tables (7 ions in W, Li and Fe)
and the depth profiles of light ions in tungsten from 15 TRIM runs. **The tables and the TRIM output are not in
this repository.** Their inputs are: `<ion>_in_<target>.SR.IN` for each table and `TRIM.IN` in each run folder.
The checksums of every file used for the paper are in `../srim_SHA256SUMS.txt`, and those of the tables without
their calculation-date line in `../srim_SHA256SUMS_nodate.txt`.

## Making the files

1. Install SRIM-2012 (www.srim.org; the paper used the package `rsim.exe.e`, 13,373,598 bytes, SHA-256
   `38b0f645d320221caddd4933c05c3f482de893eb11941cc7db7b46c1e93715ff`). The tables of the paper were made with
   its SR Module, version SRIM-2012.01, and the TRIM runs with TRIM, version SRIM-2012.03, under Wine 9.0 on
   Ubuntu 24.04 with a period as decimal point.
2. Stopping tables: for each `<ion>_in_<target>.SR.IN`, copy it into the `SR Module` folder of SRIM as `SR.IN`
   and run `SRModule.exe`; it writes the table named on the third line of the input (for example `he4_in_W.txt`).
   Put the 21 tables here, next to their inputs.
3. TRIM runs: for each folder `trim/B1` … `trim/B12` and `trim_supplementary/*`, run TRIM in batch mode
   (`TRIMAUTO = 1`) with the `TRIM.IN` of that folder, and put the output files of the run (`IONIZ.txt`,
   `PHONON.txt`, `E2RECOIL.txt`, `RANGE.txt`, `BACKSCAT.txt`, `TDATA.txt`, `TRANSMIT.txt`, `RANGE_3D.txt`,
   `VACANCY.txt`, `NOVAC.txt`, `LATERAL.txt`, `TRIMOUT.txt`) into the folder.
4. `python fetch_data.py --check` (in the repository root) compares the files with the checksums. A table made
   on another day differs from the one used for the paper in its calculation-date line only; the check accepts
   that.

## Settings

- **Stopping tables.** Units MeV/(mg/cm²), compound correction 1, solid targets at SRIM's default densities:
  W 19.35 g/cm³ (183.85 amu), Li 0.534 g/cm³ (6.941 amu), Fe 7.8658 g/cm³ (55.847 amu). Ion masses: p 1.008,
  d 2.014, t 3.016, ³He 3.016, ⁴He 4.003, ¹²C 12.000, Au 196.97 amu. 140 energies for H and He ions and 105 for
  C and Au.
- **Liquid lithium.** The stopping in MeV/(mg/cm²) does not depend on density; ranges in µm scale as 0.534/ρ
  (×1.047 at 0.51 g/cm³). The model works in areal mass.
- **TRIM runs.** "Ion Distribution and Quick Calculation of Damage" (Kinchin–Pease), 5000 ions per run, normal
  incidence, random seed 4711, single-layer targets at SRIM's default density; displacement, lattice and
  surface energies W 25, 3 and 8.68 eV, Li 25, 3 and 1.67 eV. The target thickness was doubled and the run
  repeated whenever an ion was transmitted (B9: 3 → 6 µm; S1: 0.02 → 0.04 µm). `B1_fullcascade` repeats B1 with
  full damage cascades.

## The TRIM runs

`trim/summary.csv` and `trim_supplementary/summary.csv` (with `.json` copies) give one row per run, computed
from its output: backscattered and transmitted ions, mean range and straggling, the projected range of the
stopping table at the same energy and the ratio of the two, the depths holding 50 % and 90 % of the deposited
energy, and the energy check (ionization + phonons + backscattered + transmitted energy over the incident
energy). The model reads these two tables, and the depth profiles of the run folders.

| Run | Ion → target | Thickness (µm) | Backscattered | TRIM mean range (µm) | Table projected range (µm) | TRIM/table | 50 %-energy depth (µm) | Energy check |
|---|---|---|---|---|---|---|---|---|
| B1 | d 1 keV → W | 0.05 | 38.9 % | 0.01099 | 0.0052 | 2.11 | 0.00636 | 1.106 |
| B2 | d 10 keV → W | 0.3 | 20.5 % | 0.06256 | 0.0439 | 1.43 | 0.03863 | 1.025 |
| B3 | d 100 keV → W | 1.5 | 2.6 % | 0.4279 | 0.4152 | 1.03 | 0.2325 | 1.004 |
| B4 | ⁴He 100 keV → W | 1 | 5.2 % | 0.2041 | 0.1963 | 1.04 | 0.1141 | 1.004 |
| B5 | ⁴He 2 MeV → W | 10 | 0.0 % | 2.707 | 2.71 | 1.00 | 1.369 | 1.000 |
| B6 | Au 15 MeV → W | 3 | 0.0 % | 1.115 | 1.11 | 1.00 | 0.6576 | 0.983 |
| B7 | d 10 keV → Li | 1 | 0.1 % | 0.5449 | 0.5524 | 0.99 | 0.2447 | 1.020 |
| B8 | d 100 keV → Li | 5 | 0.0 % | 2.566 | 2.63 | 0.98 | 0.9969 | 1.003 |
| B9 | ⁴He 100 keV → Li | 6 | 0.0 % | 2.674 | 2.68 | 1.00 | 0.9459 | 1.007 |
| B10 | ⁴He 2 MeV → Li | 30 | 0.0 % | 22.64 | 22.55 | 1.00 | 11.53 | 1.000 |
| B11 | t 100 keV → Li | 5 | 0.0 % | 2.938 | 2.99 | 0.98 | 1.055 | 1.004 |
| B12 | ¹²C 500 keV → Li | 5 | 0.0 % | 3.844 | 3.85 | 1.00 | 1.292 | 1.003 |
| B1_fullcascade | d 1 keV → W | 0.05 | 40.1 % | 0.01118 | 0.0052 | 2.15 | 0.00628 | 1.059 |
| S1_he4_1keV_W | ⁴He 1 keV → W | 0.04 | 40.4 % | 0.00657 | 0.0032 | 2.05 | 0.00379 | 1.136 |
| S2_he4_10keV_W | ⁴He 10 keV → W | 0.12 | 26.0 % | 0.03284 | 0.0218 | 1.51 | 0.01925 | 1.028 |

No run transmitted an ion. For hydrogen and helium of a few keV in tungsten the TRIM ranges are 1.4–2.1 times the
projected ranges of the stopping tables, so the model scales the table depth of an ion by this ratio and
removes the backscattered energy (`src/ife_wall/srim.py`); at 100 keV and above, and in lithium, the two
agree within 4 %. At 1 keV the energy check overshoots by 11–14 % (6 % with full
cascades), a known limitation of TRIM for very slow light ions near a surface.

## Checks against NIST

Tungsten, SRIM over NIST PSTAR/ASTAR (`validation/srim_checks.py` repeats this check):

| Ion | E (MeV) | S_elec | S_nuc | R_proj |
|---|---|---|---|---|
| p | 0.01 | 1.005 | 1.087 | 0.954 |
| p | 0.1 | 1.020 | 1.027 | 0.942 |
| p | 1 | 1.031 | 0.999 | 0.982 |
| p | 3.5 | 1.032 | 0.999 | 0.971 |
| p | 10 | 1.020 | 1.001 | 0.976 |
| ⁴He | 0.01 | 0.893 | 1.008 | 1.039 |
| ⁴He | 0.1 | 1.027 | 0.975 | 1.018 |
| ⁴He | 1 | 0.985 | 0.959 | 1.002 |
| ⁴He | 3.5 | 1.033 | 0.966 | 0.989 |
| ⁴He | 10 | 1.034 | 0.972 | 0.976 |

The electronic stopping of d and t equals that of p at the same velocity within 0.1 % in W, Li and Fe.
