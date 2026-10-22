"""One fully recorded extra cycle of every converged production repeated-shot lithium run (revision of 22 Oct 2026):
runs/li_multi/<Demo|Prototype>_d*um and the direct runs results/revision/li_direct/Demo_d*um (folders with the suffix
_direct), with scripts/li_extra_cycle.py into results/revision/li_extra_cycle/<name>/.  A run whose cycle is already
recorded is skipped, so a restart continues.  Exit code 1 if any cycle failed.

Usage: python scripts/run_li_extra_cycles.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "li_extra_cycle"


def runs():
    for d in sorted((ROOT / "runs" / "li_multi").glob("*_d*um")):
        scen = d.name.split("_")[0]
        if scen in ("Demo", "Prototype"):
            yield d, scen, d.name
    for d in sorted((ROOT / "results" / "revision" / "li_direct").glob("Demo_d*um")):
        yield d, "Demo", d.name + "_direct"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bad = []
    for d, scen, name in runs():
        j, ck = d / f"{scen}.json", d / f"{scen}_checkpoint.npz"
        if not (j.exists() and ck.exists()) or not json.loads(j.read_text(encoding="utf-8")).get("converged"):
            print(f"skip {name}: not a converged run with a checkpoint", flush=True)
            continue
        o = OUT / name
        if (o / f"{scen}_extra_cycle.json").exists():
            print(f"done already: {name}", flush=True)
            continue
        o.mkdir(parents=True, exist_ok=True)
        with open(OUT / f"{name}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, "scripts/li_extra_cycle.py", d.relative_to(ROOT).as_posix(), scen,
                                  f"--out={o.relative_to(ROOT).as_posix()}"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        print(f"{name}: rc {rc}", flush=True)
        if rc:
            bad.append(name)
    print("failed:", bad, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
