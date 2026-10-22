"""Parse raw NIST XCOM / PSTAR / ASTAR HTML and Henke .nff files (data/raw/) into CSVs (data/).

Units written to CSV:
  XCOM:  E_MeV, cross sections in cm^2/g
  STAR:  T_MeV, stopping powers in MeV cm^2/g, CSDA and projected ranges in g/cm^2
  Henke: E_eV, f1, f2 (dimensionless), plus mu_rho_cm2_g derived from f2 via
         mu/rho = 2 r_e lambda f2 N_A / A   (Henke, Gullikson & Davis 1993, Eq. 1-2)
"""
import html
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data"
retrieved = (RAW / "RETRIEVED_UTC.txt").read_text().strip()

NUM = r"[-+]?\d\.\d+E[-+]\d+"


def html_text(p):
    return html.unescape(re.sub(r"<[^>]*>", " ", p.read_text(errors="replace")))


def parse_xcom(z):
    rows = []
    for line in html_text(RAW / f"xcom_Z{z}.html").splitlines():
        edge = re.match(r"\s*\d+\s+([KLMNOP])\s*(\d*)\s+", line.replace("\xa0", " "))
        vals = re.findall(NUM, line)
        if len(vals) == 8:
            rows.append([float(v) for v in vals] + [f"{edge.group(1)}{edge.group(2)}" if edge else ""])
    hdr = "E_MeV,coherent,incoherent,photoelectric,pair_nuclear,pair_electron,total_with_coh,total_without_coh,edge"
    out = OUT / f"xcom_Z{z}.csv"
    with out.open("w") as f:
        f.write(f"# NIST XCOM, Z={z}, retrieved {retrieved}. Cross sections in cm^2/g.\n{hdr}\n")
        for r in rows:
            f.write(",".join(f"{v:.4E}" for v in r[:8]) + f",{r[8]}\n")
    return out, len(rows)


def parse_star(prog, matno):
    rows = []
    for line in html_text(RAW / f"{prog.lower()}_{matno}.html").splitlines():
        vals = re.findall(r"[-+]?\d\.\d+E[-+]\d+|\b\d\.\d{4}\b", line)
        if len(vals) == 7 and "E" in vals[0]:
            rows.append(vals)
    out = OUT / f"{prog.lower()}_{matno}.csv"
    with out.open("w") as f:
        f.write(f"# NIST {prog} (ICRU 49), material {matno}, retrieved {retrieved}.\n")
        f.write("T_MeV,S_elec_MeVcm2_g,S_nuc_MeVcm2_g,S_tot_MeVcm2_g,R_csda_g_cm2,R_proj_g_cm2,detour\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    return out, len(rows)


RE = 2.8179403262e-13  # classical electron radius, cm (CODATA 2018)
NA = 6.02214076e23
HC_EV_CM = 1.23984198e-4  # h c in eV cm
A_MOLAR = {"w": 183.84, "li": 6.94, "fe": 55.845}  # g/mol, IUPAC standard atomic weights


def parse_henke(el):
    d = np.loadtxt(RAW / f"henke_{el}.nff", skiprows=1)
    e, f1, f2 = d[:, 0], d[:, 1], d[:, 2]
    lam = HC_EV_CM / e
    mu_rho = 2 * RE * lam * f2 * NA / A_MOLAR[el]
    out = OUT / f"henke_{el}.csv"
    with out.open("w") as f:
        f.write(f"# CXRO/Henke atomic scattering factors ({el}), retrieved {retrieved}. "
                f"mu_rho = 2 r_e lambda f2 N_A/A (photoabsorption only), A={A_MOLAR[el]} g/mol.\n")
        f.write("E_eV,f1,f2,mu_rho_cm2_g\n")
        for row in zip(e, f1, f2, mu_rho):
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")
    return out, len(e)


if __name__ == "__main__":
    for z in (74, 3, 26):
        print(*parse_xcom(z))
    for prog in ("PSTAR", "ASTAR"):
        for m in ("074", "026"):
            print(*parse_star(prog, m))
    for el in ("w", "li", "fe"):
        print(*parse_henke(el))
