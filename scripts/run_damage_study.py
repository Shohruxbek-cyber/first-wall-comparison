"""Damaged-conductivity study (2026-09-29).  Outputs under runs/w_damage/.

Runs (each a subprocess, log in runs/logs/dmg_<case>.log):
  F  multi-shot Demo, 300 shots, uniform k_irr = 0.5 (started first); plus the helium-bubble
     skin case (skin 5 um, skin_factor 0.25 = 0.5 x f_He(0.5), bulk 0.5), 300 shots
  A  uniform sweep k_irr = 1.0, 0.85, 0.75, 0.6, 0.5, 0.4 for Demo and Prototype, from 600 K
     and from the scenario's converged pre-shot surface temperature (hot start: Demo 921.1 K,
     Prototype 609.3 K, from runs/w_multi/baseline)
  B  hot-start check = A's k_irr = 1.0 hot start vs the 600-shot baseline last-shot peak
  C  annealed skin (skin_factor 1, bulk 0.5), skin = d_1300 and d_1800 of the matching uniform
     0.5 run, then one self-consistency pass (rerun if the crossing depth moved > 10 %)
  D  helium-bubble skin, 5 um: skin_factor = f_He (bulk 1) and 0.5 f_He (bulk 0.5), hot start
  E  worst credible combination = bulk 0.5 + He skin 0.5 x 0.45 (Saito lower f_He), a bound
  G  mesh check: Demo uniform 0.5 from 600 K with dx0 = 0.5 nm
  T  TRIM multi-layer fix check: Demo uniform 0.5 hot start with trim_multilayer (bug report)
f_He: Saito et al. 2026 (arXiv:2608.15609), otherwise 0.5 and 0.25 (ESTIMATED).
Usage: python scripts/run_damage_study.py [--workers N] [--fhe 0.5,0.25] [--skip-multi]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "w_damage"
LOGS = ROOT / "runs" / "logs"
PY = sys.executable
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONWARNINGS": "ignore"}
sys.path.insert(0, str(ROOT / "src"))
from ife_wall.scenarios import converged_preshot  # noqa: E402
HOT = {n: converged_preshot(n, legacy=v) for n, v in {"Demo": 921.1, "Prototype": 609.3}.items()}
K_SWEEP = (1.0, 0.85, 0.75, 0.6, 0.5, 0.4)


def job(name, argv):
    t0 = time.time()
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"dmg_{name}.log", "w", encoding="utf-8") as log:
        r = subprocess.run([PY] + argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    return name, r.returncode, time.time() - t0


def single_job(case, scen, *opts):
    d = OUT / "single" / case
    return (f"{case}_{scen}", ["scripts/run_w_single.py", scen, *opts, f"--out={d}"])


def start_opts(scen, hot):
    return [f"T_init={HOT[scen]}"] if hot else []


def read(case, scen):
    p = OUT / "single" / case / f"{scen}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fhe", default="0.5,0.25")
    ap.add_argument("--skip-multi", action="store_true")
    a = ap.parse_args()
    fhe = [float(x) for x in a.fhe.split(",")]
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "f_He": fhe, "jobs": {}}
    sfile = OUT / "status.json"
    OUT.mkdir(parents=True, exist_ok=True)

    def record(name, rc, dt):
        status["jobs"][name] = {"returncode": rc, "minutes": round(dt / 60, 2)}
        sfile.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {'OK' if rc == 0 else 'FAILED rc=%d' % rc} ({dt / 60:.1f} min)", flush=True)

    jobs = []
    if not a.skip_multi:  # F first (long)
        jobs.append(("F_multi_Demo_kirr0.5", ["scripts/run_w_multi.py", "Demo", "300", "k_irr=0.5",
                                              f"--out={OUT / 'multi' / 'kirr0.5'}"]))
        jobs.append(("F_multi_Demo_Heskin", ["scripts/run_w_multi.py", "Demo", "300", "k_irr=0.5", "skin_um=5",
                                             f"skin_factor={0.5 * fhe[0]}", f"--out={OUT / 'multi' / 'Heskin'}"]))
    for scen in ("Demo", "Prototype"):  # A
        for hot in (False, True):
            for k in K_SWEEP:
                jobs.append(single_job(f"A_kirr{k}_{'hot' if hot else '600K'}", scen, f"k_irr={k}", *start_opts(scen, hot)))
    for f in fhe:  # D (+E)
        jobs.append(single_job(f"D_Heskin_f{f}_bulk1.0_hot", "Demo", "skin_um=5", f"skin_factor={f}", *start_opts("Demo", True)))
        jobs.append(single_job(f"D_Heskin_f{f}_bulk0.5_hot", "Demo", "skin_um=5", f"skin_factor={0.5 * f}", "k_irr=0.5",
                               *start_opts("Demo", True)))
    # D/E with Saito 2026 f_He (0.70 at 280 appm, 0.45 at 3100 appm He; arXiv:2608.15609 Eq. 2)
    hot = start_opts("Demo", True)
    jobs.append(single_job("D_Heskin_Saito_f0.7_bulk1.0_hot", "Demo", "skin_um=5", "skin_factor=0.7", *hot))
    jobs.append(single_job("D_Heskin_Saito_f0.7_bulk0.5_hot", "Demo", "skin_um=5", "skin_factor=0.35", "k_irr=0.5", *hot))
    jobs.append(single_job("D_Heskin_Saito_f0.45_bulk1.0_hot", "Demo", "skin_um=5", "skin_factor=0.45", *hot))
    jobs.append(single_job("E_worst_Saito_f0.45_bulk0.5_hot", "Demo", "skin_um=5", "skin_factor=0.225", "k_irr=0.5", *hot))
    jobs.append(single_job("G_mesh_dx0.5nm_kirr0.5_600K", "Demo", "k_irr=0.5", "dx0=5e-10"))
    jobs.append(single_job("T_trimfix_kirr0.5_hot", "Demo", "k_irr=0.5", "trim_multilayer=true", *start_opts("Demo", True)))
    jobs.append(single_job("T_trimfix_kirr1.0_hot", "Demo", "k_irr=1.0", "trim_multilayer=true", *start_opts("Demo", True)))

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, n, argv) for n, argv in jobs]
        pending_C = True
        for fut in as_completed(futs):
            record(*fut.result())
            # C: launch as soon as both uniform 0.5 Demo runs exist
            if pending_C and read("A_kirr0.5_600K", "Demo") and read("A_kirr0.5_hot", "Demo"):
                pending_C = False
                cjobs = []
                for start in ("600K", "hot"):
                    u = read(f"A_kirr0.5_{start}", "Demo")["diagnostics"]
                    for thr in ("1300", "1800"):
                        sk = u[f"d_{thr}_um"]
                        status[f"C_skin_{thr}_{start}_um"] = sk
                        cjobs.append(single_job(f"C_annealed_d{thr}_{start}", "Demo", "k_irr=0.5", f"skin_um={sk:.4f}",
                                                "skin_factor=1.0", *start_opts("Demo", start == "hot")))
                futs += [ex.submit(job, n, argv) for n, argv in cjobs]
                for f2 in as_completed(futs[-len(cjobs):]):
                    record(*f2.result())
                # self-consistency pass
                sc = []
                for start in ("600K", "hot"):
                    for thr in ("1300", "1800"):
                        r = read(f"C_annealed_d{thr}_{start}", "Demo")
                        old = status[f"C_skin_{thr}_{start}_um"]
                        new = r["diagnostics"][f"d_{thr}_um"] if r else None
                        status[f"C_selfconsistent_{thr}_{start}"] = {"skin_um_used": old, "crossing_um_in_skin_run": new,
                                                                     "moved_pct": None if new is None else 100 * (new / old - 1)}
                        if new is not None and abs(new / old - 1) > 0.10:
                            sc.append(single_job(f"C_annealed_d{thr}_{start}_pass2", "Demo", "k_irr=0.5", f"skin_um={new:.4f}",
                                                 "skin_factor=1.0", *start_opts("Demo", start == "hot")))
                sfile.write_text(json.dumps(status, indent=1), encoding="utf-8")
                for f3 in as_completed([ex.submit(job, n, argv) for n, argv in sc]):
                    record(*f3.result())
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sfile.write_text(json.dumps(status, indent=1), encoding="utf-8")
    failed = [k for k, v in status["jobs"].items() if v["returncode"] != 0]
    print("FAILED:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
