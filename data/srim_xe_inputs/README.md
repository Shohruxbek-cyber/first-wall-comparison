# SRIM inputs for the xenon benchmark (future work)

These seven SR Module inputs ask SRIM-2012 for the stopping and range of p, d, t, ³He, ⁴He, ¹²C and Au in xenon
gas, in the same format, units and energy ranges as the tables of `data/srim/` (stopping in MeV/(mg/cm²); gas
target at 0.005858 g/cm³, the ideal-gas density of Xe at 0 °C and 1 atm; only areal-mass quantities are used,
so the density does not matter). `scripts/xe_benchmark.py` needs the seven tables to compare the attenuation
step of the vapor model with the BUCKY xenon attenuation of Raffray (2005, Table 3). The tables were not made
for the paper, which lists this benchmark as future work.

To make them: for each file here, copy it into the `SR Module` folder of SRIM-2012 as `SR.IN` and run
`SRModule.exe`, which writes the table named on the third line (for example `p_in_Xe.txt`); put the seven tables
into `data/srim/` and run `python scripts/xe_benchmark.py`.

The comparison tests only the stopping of the target ions in a static gas column of known areal mass, not how
the lithium vapor column builds up during a shot.
