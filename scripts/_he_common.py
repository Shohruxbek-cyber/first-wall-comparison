"""Shared helpers for scripts/verify_he_*.py: path setup, result recording, figure style."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VER = ROOT / "runs" / "he_retention" / "verification"
VER.mkdir(parents=True, exist_ok=True)


def record(test, passed, **kw):
    """Merge one test result into runs/he_retention/verification/results.json and print PASS/FAIL."""
    f = VER / "results.json"
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    d[test] = dict(passed=bool(passed), **kw)
    f.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}  {test}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if not isinstance(v, (list, dict))), flush=True)
    return passed


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "legend.fontsize": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 120, "savefig.bbox": "tight"})
    return plt
