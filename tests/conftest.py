import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# A test that needs a file this repository does not contain is skipped with a message saying how to get it,
# instead of failing.  Any other missing file is still an error.
HINTS = [
    ("data/srim/", "an SRIM-2012 table or TRIM output, not distributed: see data/srim/README.md, then "
                   "python fetch_data.py --check"),
    ("data/raw/", "a raw NIST or CXRO response, not distributed: python fetch_data.py --download"),
    ("data/", "a NIST or CXRO table, not distributed: python fetch_data.py --download"),
    ("runs/he_retention/thermal/", "a recorded helium temperature cycle: "
                                   "python scripts/record_he_thermal.py Prototype converged --Tback=converged"),
]


def _hint(exc):
    name = getattr(exc, "filename", None)
    if not name:  # numpy.loadtxt puts the path only into the message: "<path> not found."
        msg = str(exc)
        name = msg[:-len(" not found.")] if msg.endswith(" not found.") else None
    if not name:
        return None
    p = Path(name)
    if not p.is_absolute():
        p = Path.cwd() / p
    try:
        rel = p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    for prefix, hint in HINTS:
        if rel.startswith(prefix):
            return f"needs {rel}, {hint}"
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except FileNotFoundError as e:
        hint = _hint(e)
        if hint is None:
            raise
        pytest.skip(hint)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except FileNotFoundError as e:
        hint = _hint(e)
        if hint is None:
            raise
        pytest.skip(hint)
