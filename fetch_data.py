"""Install and check the third-party data tables that this repository does not contain.

    python fetch_data.py --check           list the tables that are missing, and check the present ones
                                           against the checksums of the files used for the paper
    python fetch_data.py --download        download the NIST and CXRO tables from the original web services
                                           (scripts/fetch_data.sh; needs bash and curl), parse them into
                                           data/*.csv (scripts/parse_data.py), then check them
    python fetch_data.py --from FOLDER     copy the tables from FOLDER, a folder laid out like data/ that
                                           holds them (for example the data archive of the paper), then
                                           check them

Not distributed here:
  * the NIST XCOM, PSTAR and ASTAR tables and the CXRO (Henke) scattering factors: data/*.csv and the raw
    responses in data/raw/ (--download makes them);
  * the SRIM-2012 stopping tables (data/srim/*.txt) and the TRIM output (data/srim/trim*/*/*.txt).  No web
    service makes these: run SRIM-2012 with the inputs in data/srim/ (see data/srim/README.md), or copy
    them with --from.

Checksums: data/THIRD_PARTY_SHA256SUMS.txt (NIST, CXRO), data/srim_SHA256SUMS.txt (SRIM, whole files) and
data/srim_SHA256SUMS_nodate.txt (SRIM stopping tables without their "Calc. date" line, so that a table
made again on another day still matches if every number agrees).  A downloaded NIST or CXRO table that
differs only in its retrieval date matches on its data rows.
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def data_rows(b):
    return b"\n".join(l for l in b.replace(b"\r\n", b"\n").split(b"\n") if l.strip() and not l.lstrip().startswith(b"#"))


def no_date(b):
    return b"".join(l for l in b.splitlines(keepends=True) if b"Calc. date" not in l)


def sums(name):
    """[(path relative to ROOT, columns...)] from a checksum file of data/."""
    out = []
    for line in (DATA / name).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if name.startswith("srim"):  # sha256sum format: "<hash> *srim/<file>"
            out.append(("data/" + parts[1].lstrip("*"), parts[0]))
        else:  # "<kind> <hash of file> <hash of data rows> <path>"
            out.append((parts[3], parts[0], parts[1], parts[2]))
    return out


def srim_files():
    nodate = dict((p, h) for p, h in sums("srim_SHA256SUMS_nodate.txt"))
    return [(p, h, nodate.get(p)) for p, h in sums("srim_SHA256SUMS.txt")]


def check(verbose=True):
    """Returns (missing required files, files that differ)."""
    missing, differ, same, close = [], [], 0, 0
    for rel, h, h_nodate in srim_files():
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        b = p.read_bytes()
        if sha(b) == h:
            same += 1
        elif h_nodate and sha(no_date(b)) == h_nodate:
            close += 1
        else:
            differ.append(rel)
    for rel, kind, h, h_rows in sums("THIRD_PARTY_SHA256SUMS.txt"):
        p = ROOT / rel
        if not p.exists():
            if kind == "required":
                missing.append(rel)
            continue
        b = p.read_bytes()
        if sha(b) == h:
            same += 1
        elif sha(data_rows(b)) == h_rows:
            close += 1
        else:
            differ.append(rel)
    if verbose:
        print(f"{same} files identical to those used for the paper, {close} identical apart from a date line, "
              f"{len(differ)} different, {len(missing)} missing")
        for rel in missing[:12]:
            print("   missing:", rel)
        if len(missing) > 12:
            print(f"   ... and {len(missing) - 12} more")
        for rel in differ:
            print("   different:", rel)
        if missing:
            nist = [m for m in missing if not m.startswith("data/srim/")]
            srim = [m for m in missing if m.startswith("data/srim/")]
            if nist:
                print("NIST/CXRO tables missing: python fetch_data.py --download (or --from FOLDER)")
            if srim:
                print("SRIM files missing: make them with SRIM-2012 from the inputs in data/srim/ "
                      "(data/srim/README.md), or python fetch_data.py --from FOLDER")
        if differ:
            print("A different file may change the results; the tables used for the paper are the ones whose "
                  "checksums are listed in data/.")
    return missing, differ


def copy_from(folder):
    folder = Path(folder).resolve()
    rels = [r for r, *_ in srim_files()] + [r for r, *_ in sums("THIRD_PARTY_SHA256SUMS.txt")]
    n = 0
    for rel in rels:
        src = folder / Path(rel).relative_to("data")
        dst = ROOT / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    print(f"copied {n} files from {folder}")


def download():
    bash = shutil.which("bash")
    if not bash or not shutil.which("curl"):
        raise SystemExit("--download needs bash and curl on the PATH (on Windows: Git Bash)")
    subprocess.run([bash, "scripts/fetch_data.sh"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/parse_data.py"], cwd=ROOT, check=True)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--download":
        download()
    elif argv[0] == "--from" and len(argv) > 1:
        copy_from(argv[1])
    elif argv[0] != "--check":
        print(__doc__)
        return 2
    missing, differ = check()
    return 1 if missing or differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
