"""Run several He-retention jobs with limited concurrency (each job = scripts/run_he_retention.py).

  python scripts/run_he_batch.py --jobs 3 Demo:Em_He_low:600 Demo:Eb_low:600 ...
Output of job S:case:N goes to runs/he_retention/sens/<S>_<case>/ ; jobs whose meta.json exists are skipped.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--outdir", default="runs/he_retention/sens")
    ap.add_argument("spec", nargs="+")
    a = ap.parse_args()
    queue = []
    for s in a.spec:
        scen, case, n = s.split(":")[:3]
        extra = s.split(":")[3:]
        out = ROOT / a.outdir / f"{scen}_{case}"
        if (out / "meta.json").exists():
            print("skip", out)
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "run_he_retention.py"), scen, "--shots", n, "--case", case,
               "--out", str(out), "--resume"] + (["--set"] + extra if extra else [])
        queue.append((s, cmd, out))
    running = []
    log = ROOT / "runs" / "he_retention" / "logs"
    log.mkdir(parents=True, exist_ok=True)
    while queue or running:
        running = [(s, p) for s, p in running if p.poll() is None]
        while queue and len(running) < a.jobs:
            s, cmd, out = queue.pop(0)
            lf = open(log / f"batch_{s.replace(':', '_')}.log", "w")
            running.append((s, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)))
            print(time.strftime("%H:%M:%S"), "started", s, flush=True)
        time.sleep(5)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
