"""Pre-send check for the rendered documents (added 14 October 2026): a PDF built
for sending must not contain working-draft markers.

After a renderer has written its PDF, `check()` reads the PDF text back (whitespace-normalized, so
a marker split over a line break is still found) and looks for the draft markers below.  If any
survives, the build FAILS with exit code 2 and prints every marker with its page, unless the
renderer was run with `--draft`, in which case it prints a loud warning and lets the working copy
through.  A PDF built without `--draft` therefore cannot contain these markers.

Markers: the AI-disclosure placeholder (both spellings used so far), the acknowledgement line that
names a person pending permission, and the older editing markers.
"""
import sys

import pymupdf

MARKERS = ("AI DISCLOSURE", "AUTHOR TO INSERT", "AUTHOR SUPPLIES WORDING", "pending his permission",
           "TO FILL", "CITATION NEEDED", "AUTHOR TO CONFIRM", "EDITOR:")
BANNER = "#" * 78


def find_markers(pdf_path, extra=()):
    """[(marker, page number)] for every draft marker in the PDF text (MARKERS plus `extra`)."""
    doc = pymupdf.open(str(pdf_path))
    hits = []
    for i, page in enumerate(doc):
        text = " ".join(page.get_text().split())
        for m in MARKERS + tuple(extra):
            if m in text:
                hits.append((m, i + 1))
    doc.close()
    return hits


def check(pdf_path, label, argv=None, extra=()):
    """Return the exit code for the renderer: 0 if clean (or a declared draft), 2 if markers survive.
    `extra`: further markers for one document (the article adds its repository placeholders)."""
    argv = sys.argv[1:] if argv is None else argv
    hits = find_markers(pdf_path, extra)
    if not hits:
        print(f"pre-send check ({label}): no draft markers in the PDF")
        return 0
    lines = "\n  ".join(f"{m!r} on page {p}" for m, p in hits)
    if "--draft" in argv:
        print(f"{BANNER}\nDRAFT BUILD ({label}): {len(hits)} draft marker(s) in the PDF, NOT FOR SENDING:\n  {lines}\n{BANNER}")
        return 0
    print(f"{BANNER}\nPRE-SEND CHECK FAILED ({label}): {len(hits)} draft marker(s) survive in the PDF:\n  {lines}\n"
          f"Resolve them in the Markdown sources, or rebuild with --draft for a working copy.\n{BANNER}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    # standalone use: python scripts/presend_check.py <pdf> [--draft]
    sys.exit(check(sys.argv[1], sys.argv[1], sys.argv[2:]))
