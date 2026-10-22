"""Parse the archived ARIES-IFE direct-drive ion spectra (dd.shtml) into CSV.

Source: ARIES-IFE Reference Target Spectra, data courtesy of J. Perkins (LLNL),
archived by the Internet Archive:
http://web.archive.org/web/20050506140724/http://aries.ucsd.edu:80/ARIES/DOCS/ARIES-IFE/SPECTRA/dd.shtml
Tables are "Mid bin energy (keV)" vs "#/keV" (number of particles per keV, whole target).
"""
import re, csv, sys, html
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "references/aries_wayback/dd.shtml"
if not src.exists():
    sys.exit(f"{src} not found: save the archived page (URL above) there first")
text = html.unescape(re.sub(r"<[^>]*>", " ", src.read_text(errors="replace")))
lines = [l.strip() for l in text.splitlines() if l.strip()]

group, species, rows = "debris", None, []
species_re = re.compile(r"^(Deuterons|Tritons|Protons|3 He|4 He|12 C|Au|Gammas|Neutrons)$")
for l in lines:
    if l.startswith("Burn Product Spectrum"):
        group = "burn"
    m = species_re.match(l)
    if m:
        species = m.group(1).replace(" ", "")
        continue
    p = l.split()
    if species and len(p) == 2:
        try:
            e, n = float(p[0]), float(p[1])
        except ValueError:
            continue
        rows.append((group, species, e, n))

out = root / "data/aries_nrl154_dd_spectra.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["# ARIES-IFE NRL 154 MJ direct-drive target (Perkins). Archived 2005-05-06, web.archive.org. Units: E_mid keV, dN/dE particles/keV (whole target)"])
    w.writerow(["group", "species", "E_mid_keV", "dNdE_per_keV"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows -> {out}")
