"""Direct repeated-shot run at an interpolated minimum film (revision of 22 Oct 2026).

The minimum film of the case is interpolated from its own finished runs (scripts/diag_li_minima_direct.py, rule of
Section 2.7), rounded to the nearest whole micrometer (as for 45.93 -> 46 um), and run with
scripts/run_li_multi.py Demo 600 delta_um=<film> <case options> --converge --out=<folder>.

Usage: python scripts/run_li_direct.py <case> <925|873> [--resume]
  case: baseline            the production runs runs/li_multi/Demo_d*um      -> results/revision/li_direct/
        return_latent       the baseline 925 K film with --return-latent      -> results/revision/li_return_latent/
        he                  the Demo helium-exclusion film (10^6 shots to 10^18 m^-2; the limit argument is ignored)
                                                                              -> results/revision/li_direct/
        neutron_steel0.5, neutron_steel1, vapor_frr0_s1, steel_3.5mm
                            the variant's own runs (diag_li_minima_direct.VARIANTS)
                                                                              -> results/revision/li_direct_variants/<case>/
  --resume  continue the run from its checkpoint (passed to run_li_multi.py); without it an existing output folder
            of the same film is removed first.
A limit not bracketed by the case's runs, or a 873 K film equal to the case's 925 K film, is recorded in
results/revision/li_direct/notes/<case>_<limit>.json and the script exits 0 without running.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _films import direct_film_um  # noqa: E402
from diag_li_minima_direct import VARIANTS, crossing, points  # noqa: E402

OPTS = {"neutron_steel0.5": ["neutron_heating=true", "neutron_steel_factor=0.5"],
        "neutron_steel1": ["neutron_heating=true", "neutron_steel_factor=1"],
        "vapor_frr0_s1": ["vapor_shielding=true", "vapor_f_rr=0.0", "vapor_stopping_factor=1.0"],
        "steel_3.5mm": ["steel_thickness=0.0035"]}


def film_for(case, limit):
    if case == "he":
        crit = json.loads((ROOT / "results/revision/steel_behind_film/steel_criteria.json").read_text(encoding="utf-8"))
        return crit["scenarios"]["Demo"]["film_um"]["1e6_shots_to_1e18"], "helium-exclusion film (steel_criteria.json)"
    own = sorted((ROOT / "runs" / "li_multi").glob("Demo_d*um")) if case in ("baseline", "return_latent") \
        else sorted(ROOT.glob(VARIANTS[case]))
    c = crossing(points(own), float(limit))
    return (None, "not bracketed by " + ", ".join(str(p.relative_to(ROOT)) for p in own)) if c is None \
        else (c["delta_um"], f"interpolated between {c['bracket_um']} um")


def note(case, limit, **kw):
    d = ROOT / "results" / "revision" / "li_direct" / "notes"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{case}_{limit}.json").write_text(json.dumps(kw, indent=1), encoding="utf-8")
    print(json.dumps(kw), flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    case, limit = args[0], args[1]
    resume = "--resume" in sys.argv[1:]
    d_int, how = film_for(case, limit)
    if d_int is None:
        note(case, limit, ran=False, reason=how)
        return 0
    d = direct_film_um(d_int)
    if case not in ("he",) and limit == "873":
        d925, _ = film_for(case, "925")
        if d925 is not None and direct_film_um(d925) == d:
            note(case, limit, ran=False, reason=f"873 K film {d_int:.2f} um rounds to the 925 K run's {d:g} um")
            return 0
    if case in ("baseline", "he"):
        out = ROOT / "results" / "revision" / "li_direct" / f"Demo_d{d:g}um"
    elif case == "return_latent":
        out = ROOT / "results" / "revision" / "li_return_latent" / f"Demo_d{d:g}um"
    else:
        out = ROOT / "results" / "revision" / "li_direct_variants" / case / f"Demo_d{d:g}um"
    done = out / "Demo.json"
    if resume and done.exists():
        r = json.loads(done.read_text(encoding="utf-8"))
        if r.get("status") == "ok" and (r.get("converged") or r.get("n_shots", 0) >= r.get("max_n_shots", 1e9)):
            print(f"already complete: {done.relative_to(ROOT)}", flush=True)
            return 0
    if out.exists() and not resume:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "direct_run_choice.json").write_text(json.dumps(
        {"case": case, "limit_K": None if case == "he" else float(limit), "interpolated_um": d_int, "how": how,
         "run_um": d}, indent=1), encoding="utf-8")
    cmd = [sys.executable, "scripts/run_li_multi.py", "Demo", "600", f"delta_um={d:g}", *OPTS.get(case, []), "--converge",
           f"--out={out.relative_to(ROOT).as_posix()}"] + (["--return-latent"] if case == "return_latent" else []) \
        + (["--resume"] if resume else [])
    print("run:", " ".join(cmd[1:]), f"({how}: {d_int:.3f} um)", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
