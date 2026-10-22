# Input data

The model reads the files below. Those marked "not in this repository" are third-party tables that are not
redistributed here; `python fetch_data.py --download` downloads and parses the NIST and CXRO tables, the SRIM
files are made with SRIM (see `srim/README.md`), and `python fetch_data.py --check` compares every table with
the checksums of the files used for the paper.

| File | Content | Source | In this repository |
|---|---|---|---|
| `aries_nrl154_dd_spectra.csv` | ion spectra of the NRL 154 MJ direct-drive target (debris and burn ions, particles per keV) | ARIES-IFE Reference Target Spectra, data courtesy of J. Perkins (LLNL), aries.ucsd.edu, archived by the Internet Archive on 6 May 2005; parsed by `scripts/parse_aries_spectra.py` | yes |
| `raffray2005_fig2_photon_spectrum_DIGITIZED.csv` | x-ray spectrum of the same target, digitized | A.R. Raffray, J. Nucl. Mater. 347 (2005) 178–191, Fig. 2 | yes |
| `raffray2005_fig7_DIGITIZED.json` | curves digitized for the Raffray benchmark (Tables 3 and E.2 of the paper) | A.R. Raffray (2005), Fig. 7 | yes |
| `eckstein2007_table8_W.csv` | fit parameters of the sputtering yield of tungsten | W. Eckstein, Sputtering yields, in: R. Behrisch, W. Eckstein (Eds.), Sputtering by Particle Bombardment, Topics in Applied Physics 110, Springer, 2007, Table 8 | yes |
| `xcom_Z3.csv`, `xcom_Z26.csv`, `xcom_Z74.csv` | photon cross sections of Li, Fe and W, 1 keV–100 GeV | M.J. Berger et al., XCOM: Photon Cross Sections Database, NIST Standard Reference Database 8, doi:10.18434/T48G6X | no: `fetch_data.py --download` |
| `pstar_026.csv`, `pstar_074.csv`, `astar_026.csv`, `astar_074.csv` | proton and alpha stopping powers and ranges in Fe and W (cross-check of SRIM) | M.J. Berger et al., ESTAR, PSTAR, ASTAR, NIST Standard Reference Database 124, doi:10.18434/T4NC7P | no: `fetch_data.py --download` |
| `henke_li.csv`, `henke_fe.csv`, `henke_w.csv` | atomic scattering factors f1, f2 and the photoabsorption derived from f2, 10 eV–30 keV | CXRO, Lawrence Berkeley National Laboratory; B.L. Henke, E.M. Gullikson, J.C. Davis, At. Data Nucl. Data Tables 54 (1993) 181–342 | no: `fetch_data.py --download` |
| `raw/` | the downloaded NIST and CXRO responses that `scripts/parse_data.py` reads | as above | no: `fetch_data.py --download` |
| `srim/*.SR.IN`, `srim/trim*/*/TRIM.IN` | inputs of the SRIM-2012 stopping tables and TRIM runs | this work | yes |
| `srim/trim/summary.*`, `srim/trim_supplementary/summary.*` | one row per TRIM run: backscattered fraction, range against the stopping table, energy check | this work, from the TRIM output | yes |
| `srim/*.txt`, `srim/trim*/*/*.txt` | SRIM-2012 stopping and range tables (21) and TRIM output (15 runs) | made with SRIM-2012 (J.F. Ziegler, J.P. Biersack; J.F. Ziegler et al., Nucl. Instrum. Methods Phys. Res. B 268 (2010) 1818–1823) from the inputs above | no: see `srim/README.md` |
| `srim_xe_inputs/*.SR.IN` | inputs of the xenon stopping tables of the xenon benchmark (future work) | this work | yes |

Checksums: `THIRD_PARTY_SHA256SUMS.txt` (NIST and CXRO), `srim_SHA256SUMS.txt` (every SRIM file, whole
file) and `srim_SHA256SUMS_nodate.txt` (the SRIM stopping tables without their calculation-date line).

The NIST tables are Standard Reference Data, copyrighted by the U.S. Secretary of Commerce; the CXRO pages are
copyright of the Regents of the University of California unless marked otherwise. Neither is redistributed
here. The material-property fits and scenario parameters taken from the literature are coded in `src/ife_wall/`,
each with its source; Table F.1 of the paper lists them.
