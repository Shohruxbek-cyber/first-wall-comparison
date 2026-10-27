# first-wall-comparison

Code and results for the paper *Comparative analysis of tungsten armor and a liquid-lithium film as first walls for
the HiPER laser-fusion chamber: thermal response, vapor shielding and helium transmission to the steel* (S. Bozorov,
2026; the paper and its supplementary material are on arXiv, link to follow). The Python package
`ife_wall` models a 1 mm tungsten armor and a thin flowing lithium film on steel under the x-rays and ions of the
HiPER scenarios: one-dimensional heat conduction with melting and evaporation, vapor shielding, neutron heating and
helium retention. `reproduce.py` reruns every calculation of the paper in stages.

## Requirements

Python 3.14.6 (64-bit; tested on Windows 11 only) and the pinned packages of `requirements.txt`:

```
python -m venv .venv
.venv\Scripts\activate             (Linux, macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
python fetch_data.py --download
python -m pytest
```

The NIST and CXRO tables are not distributed; `fetch_data.py --download` gets them (needs bash and curl). The SRIM-2012 tables are made
with SRIM from the inputs in `data/srim/` (see its README). `python fetch_data.py --check` compares both with the
files used for the paper; tests that need a missing table are skipped with a message. On Windows, set
`PYTHONUTF8=1` before running a script directly.

## Running

```
python reproduce.py all              rerun everything
python reproduce.py STAGE            one stage (python reproduce.py --list)
python reproduce.py --check STAGE    compare with the committed results
```

Each stage runs in `work/STAGE/` (at most 4 calculations at once; `--jobs N` changes this), writes its outputs to
`results/STAGE/` and prints how long it took. The times below are approximate, for four calculations at once on a
6-core laptop, estimated from the run times of the same calculations for the paper; all stages together take about
36 hours.

## Stages

| Stage | Paper | Contents | Time |
|---|---|---|---|
| `verification` | Supplementary Tables S1, S2; Supplementary Fig. S1 | solver tests, grid and time-step convergence | 1.7 h |
| `benchmarks` | Tables 2, 3; Section 2.3; Supplementary Table S5; Supplementary Fig. S2 | source term and its particle count, SRIM and TRIM checks, published benchmarks | 3 min |
| `tungsten_single` | Sections 4.1, 4.3, 4.4; Figs. 1, 3 | single shots, sensitivity variants, sputtering, heat-flux factors | 3 min |
| `tungsten_repeated` | Table 4; Fig. 2; Supplementary Table S9 | 600 Demo and 100 Prototype shots, converged pre-shot state, coolant and steel variants | 4 h |
| `damage_surface` | Section 4.5; Fig. 4; Supplementary Table S12 | reduced conductivity, damaged surface layer, erosion lifetimes | 5.5 h |
| `lithium_single` | Sections 5.1, 5.2; Figs. 5, 6; Supplementary Fig. S4 | film-thickness scans and variants | 13 min |
| `lithium_repeated` | Table 5; Section 5.3; Fig. 7; Supplementary Table S6; Supplementary Fig. S3 | repeated shots for nine films, helium in the steel, direct runs at the minimum films, steel-face pulse | 5.7 h |
| `tungsten_grades` | Supplementary Tables S10, S11; Supplementary Figs. S5, S6 | K-doped and ODS tungsten | 2.7 h |
| `vapor_shielding` | Section 6; Fig. 8; Supplementary Table S4 | shielding cases, the static bound, direct runs with shielding | 3.1 h |
| `neutron_heating` | Sections 4.2, 5.3, 7.5 | neutron heating; coolant 3.5 mm behind the surface; direct runs of both | 5.6 h |
| `lithium_films` | Table 5; Section 5.4 | minimum films from the direct runs, steel at those films, partial film renewal | 5.6 h |
| `helium` | Table 4; Supplementary Tables S16–S18 | helium-retention model, its tests and sensitivity | 1.8 h |
| `figures` | Figs. 1–8; Supplementary Figs. S1–S6 | all figures, drawn from the stage outputs | 1 min |
| `article` | Tables 1, A.1, B.1, B.2, C.1; Supplementary Tables S3, S7, S8, S13–S15, S19 | the PDF and its text tables; needs the paper source | – |

## Files

- `src/ife_wall/`: the model; `src/ife_wall/he/`: the helium-retention model.
- `scripts/`: `run_*.py` start the calculations, the other scripts derive the tables, checks and figures; each
  file's docstring says what it does.
- `validation/`: the published benchmarks and SRIM checks; `tests/`: the test suite.
- `data/`: input data and checksums (`data/README.md`).
- `results/`: the outputs of the stages. The full run folders (time histories, checkpoints) are not included in this
  archive because of their size; they are available from the author on request.
- `docs/`, `refs/`: notes and references of the tungsten-grade study; `paper/`: a note on the paper (on arXiv,
  link to follow).

## Code availability

The code (the Python package `ife_wall` with its run scripts, tests and validation scripts) and the outputs behind
every table and figure of the paper are archived at Zenodo, https://doi.org/10.5281/zenodo.23577155 (all versions),
and developed in this repository under the MIT licence. The third-party input data (NIST stopping-power and
attenuation tables, CXRO photoabsorption data and SRIM-2012 stopping and range tables) are not redistributed;
`fetch_data.py` retrieves or checks them. To cite the code, use `CITATION.cff`.

## License

MIT (`LICENSE`). The third-party data keep the terms of their sources (`data/README.md`).
