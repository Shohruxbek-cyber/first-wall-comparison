#!/usr/bin/env bash
# Re-download the raw photon and ion data tables used by ife_wall.
# Run from the project root:  bash scripts/fetch_data.sh && python scripts/parse_data.py
# Raw responses are kept in data/raw/ so parsed CSVs can be audited against them.
set -euo pipefail
RAW=data/raw
mkdir -p "$RAW"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$RAW/RETRIEVED_UTC.txt"

# NIST XCOM (Berger et al., NIST Standard Reference Database 8), default energy grid 1 keV - 100 GeV.
for z in 74 3 26; do
  curl -sS -m 120 -X POST "https://physics.nist.gov/cgi-bin/Xcom/xcom3_1" \
    --data "ZNum=$z&OutOpt=PIC&NumAdd=1&Energies=&WindowXmin=0.001&WindowXmax=100000&Output=on" \
    -o "$RAW/xcom_Z$z.html"
done

# NIST PSTAR / ASTAR (ICRU Report 49). Elemental Li is not in the STAR material list.
for prog in PSTAR ASTAR; do
  for m in 074 026; do
    curl -sS -m 120 -X POST "https://physics.nist.gov/cgi-bin/Star/ap_table.pl" \
      -F prog=$prog -F matno=$m -F GraphType=StopPwr -F Graph3=on -F ShowDefault=on \
      -o "$RAW/${prog,,}_$m.html"
  done
done

# CXRO / Henke atomic scattering factors (Henke, Gullikson & Davis 1993), 10 eV - 30 keV.
for el in w li fe; do
  curl -sS -m 120 "https://henke.lbl.gov/optical_constants/sf/$el.nff" -o "$RAW/henke_$el.nff"
done
echo "done: $(ls "$RAW" | wc -l) files in $RAW"
