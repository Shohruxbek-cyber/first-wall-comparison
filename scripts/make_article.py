"""Build the research-article draft (revision of 20 October 2026; Markdown -> A4 PDF).

Same renderer as scripts/make_revised_full_results.py (Python-Markdown, PyMuPDF Story), with
numbered citations: the text cites `[@key]` or `[@key1; @key2]`, keys resolve in
docs/article/references.json, numbers follow the order of first citation (main text, then
appendices), and the reference list is generated in that order (Elsevier numbered style).
An unknown key stops the build.  Earlier documents and the full-results draft are untouched.

Usage: python scripts/make_article.py [--draft]
Input:  docs/article/*.md in the order of PARTS, docs/article/references.json, the figures
        docs/figures/article/*.png of scripts/make_article_figures.py
Output: docs/article/Bozorov_IFE_first_wall_article.{md,pdf}, and the supplementary material
        docs/article/Bozorov_IFE_first_wall_supplementary_S1.{md,pdf} from docs/article/S1_helium_model.md
The pre-send check (scripts/presend_check.py) fails the build while a draft marker or a
repository placeholder survives, unless --draft is given.
"""
import json
import re
import sys
from pathlib import Path

import markdown
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "article"
STEM = "Bozorov_IFE_first_wall_article"
PARTS = ["00_front.md", "01_introduction.md", "02_methods.md", "03_validation.md", "04_results_tungsten.md",
         "05_results_lithium.md", "06_vapor_shielding.md", "07_discussion.md", "08_conclusions.md",
         "09_declarations.md", "REFERENCES", "A_appendices.md"]
EXTRA_MARKERS = ("[GitHub URL]", "[Zenodo DOI]", "[version]", "AUTHOR TO VERIFY", "[e-mail]")
SUPP, SUPP_STEM = "S1_helium_model.md", "Bozorov_IFE_first_wall_supplementary_S1"
CITE = re.compile(r"\[@([A-Za-z0-9_:\-]+(?:\s*;\s*@[A-Za-z0-9_:\-]+)*)\]")


def compress(nums):
    """[1, 2, 3, 5] -> '1–3, 5'."""
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}" if j == i else (f"{nums[i]}, {nums[j]}" if j == i + 1 else f"{nums[i]}–{nums[j]}"))
        i = j + 1
    return ", ".join(out)


def number_citations(texts, refs):
    order = {}
    for t in texts:
        for m in CITE.finditer(t):
            for key in (k.strip().lstrip("@") for k in m.group(1).split(";")):
                if key not in refs:
                    raise KeyError(f"citation key not in references.json: {key}")
                order.setdefault(key, len(order) + 1)

    def repl(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        return "[" + compress([order[k] for k in keys]) + "]"
    return [CITE.sub(repl, t) for t in texts], order


def reference_list(order, refs):
    lines = ["## References", ""]
    for key, n in sorted(order.items(), key=lambda kv: kv[1]):
        r = refs[key]
        doi = f" https://doi.org/{r['doi']}" if r.get("doi") else ""
        lines.append(f"<p class=\"ref\">[{n}] {r['text']}{doi}</p>")
    return "\n".join(lines) + "\n"


FIG_DIR = "figures/article/"  # scripts/make_article_figures.py: PNG at print size, 600 dpi


def figure_html(md_text, breaks=()):
    """Figure links -> centered <img> paragraphs, numbered fig1, fig2, ... in order.  The figures of
    FIG_DIR at their print width (the PDF engine caps wider ones at the text width), other images at
    the text width; a figure in `breaks` starts a new page.  Returns the text and {id: width in pt}."""
    from PIL import Image
    widths = {}

    def repl(m):
        src, k = m.group(2), f"fig{len(widths) + 1}"
        if src.startswith(FIG_DIR):
            with Image.open(ROOT / "docs" / src) as im:
                widths[k] = round(im.width * 72.0 / im.info.get("dpi", (600, 600))[0], 1)
        else:
            widths[k] = 480.0
        style = ' style="page-break-before: always"' if k in breaks else ""
        return f'<p class="fig"{style}><img id="{k}" src="{src}" width="{widths[k]:g}"/></p>'
    return re.sub(r"!\[([^\]]*)\]\(\.\./(figures/[^)]+)\)", repl, md_text), widths


def render_pdf(md_text, pdf_path, title, base_pt=9.4):
    """Markdown -> A4 PDF.  The engine shrinks an image that does not fit into the rest of a page, so
    dry layout passes first give every figure it would shrink a page break of its own.  Returns the
    page count and the figures that start a new page."""
    css = f"""
body {{ font-family: serif; font-size: {base_pt}pt; line-height: 1.3; }}
h1 {{ font-size: {base_pt + 5}pt; margin: 0 0 4pt 0; font-family: sans-serif; }}
h2 {{ font-size: {base_pt + 2}pt; margin: 10pt 0 3pt 0; font-family: sans-serif; }}
h3 {{ font-size: {base_pt + 0.6}pt; margin: 7pt 0 2pt 0; font-family: sans-serif; font-style: italic; }}
p {{ margin: 2.5pt 0; text-align: justify; }}
li {{ margin: 1pt 0; }}
p.ref {{ font-size: {base_pt - 1}pt; margin: 1pt 0; text-align: left; }}
table {{ border-collapse: collapse; margin: 4pt 0; }}
th, td {{ border: 0.5pt solid #888; padding: 1pt 3pt; font-size: {base_pt - 1.6}pt; vertical-align: top; font-family: sans-serif; }}
th {{ font-weight: bold; }}
code {{ font-family: monospace; font-size: {base_pt - 1.4}pt; }}
p.fig {{ text-align: center; margin: 5pt 0 2pt 0; }}
"""
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (54, 50, -54, -50)
    text_w = where.width - 2 * base_pt  # the engine's body margin is 1 em

    def layout(breaks, writer=None):
        """One pass over all pages (drawn only with a writer); returns the page count and the figures
        drawn narrower than their width."""
        md, widths = figure_html(md_text, breaks)
        story = pymupdf.Story(html=markdown.markdown(md, extensions=["tables"]), user_css=css,
                              archive=pymupdf.Archive(str(ROOT / "docs")))
        drawn, more, pages = {}, 1, 0

        def note(e):
            if e.id in widths:
                drawn[e.id] = max(drawn.get(e.id, 0.0), e.rect[2] - e.rect[0])
        while more:
            more, filled = story.place(where)
            story.element_positions(note)
            if writer is None:
                story.draw(None)
            else:
                dev = writer.begin_page(mediabox)
                story.draw(dev)
                writer.end_page()
            pages += 1
            if filled[3] - filled[1] < 1.0 or pages > 400:
                if writer is not None:
                    writer.close()  # see make_revised_full_results.py: an element that cannot be placed loops forever
                raise RuntimeError(f"page {pages}: nothing could be placed; reword or split the element at this page break")
        return pages, [k for k in widths if drawn.get(k, 0.0) < min(widths[k], text_w) - 1.0]

    breaks = set()
    while True:
        pages, shrunk = layout(breaks)
        if not shrunk:
            break
        if shrunk[0] in breaks:
            raise RuntimeError(f"{shrunk[0]} is shrunk even at the top of a page")
        breaks.add(shrunk[0])
    raw = pdf_path.with_name("_" + pdf_path.stem + ".raw.pdf")
    writer = pymupdf.DocumentWriter(str(raw), "compress")  # the 600-dpi figures: 0.26 GB uncompressed
    pages, shrunk = layout(breaks, writer)
    writer.close()
    del writer  # releases the file, so that the unlink below works on Windows
    if shrunk:
        raise RuntimeError(f"figures drawn narrower than their width: {shrunk}")
    doc = pymupdf.open(str(raw))
    for i, page in enumerate(doc):  # page numbers
        page.insert_text((mediabox.width / 2 - 8, mediabox.height - 28), str(i + 1), fontsize=8)
    doc.set_metadata({"title": title, "author": "Shohruxbek Bozorov", "subject": title,  # review S-14: subject = title
                      "creator": "ife_wall scripts/make_article.py"})
    doc.save(str(pdf_path), garbage=4, deflate=True, deflate_images=True, clean=True)
    doc.close()
    try:
        raw.unlink()
    except OSError:
        pass
    return pages, sorted(breaks, key=lambda k: int(k[3:]))


def word_count(md):
    text = re.sub(r"<[^>]+>", " ", md)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-.,×⁻⁰¹²³⁴⁵⁶⁷⁸⁹]*", text))


if __name__ == "__main__":
    refs = json.loads((ART / "references.json").read_text(encoding="utf-8"))
    names = [p for p in PARTS if p != "REFERENCES"]
    texts = [(ART / p).read_text(encoding="utf-8").rstrip() + "\n\n" for p in names]
    texts, order = number_citations(texts, refs)
    by_name = dict(zip(names, texts))
    md = "".join(reference_list(order, refs) + "\n" if p == "REFERENCES" else by_name[p] for p in PARTS)
    (ART / f"{STEM}.md").write_text(md, encoding="utf-8")
    title = re.search(r"^# (.+)$", by_name["00_front.md"], re.M).group(1)
    pdf = ART / f"{STEM}.pdf"
    n, moved = render_pdf(md, pdf, title)
    main = "".join(by_name[p] for p in names[1:names.index("09_declarations.md")])
    unused = sorted(set(refs) - set(order))
    print(f"article: {n} page(s) -> {pdf}; {len(order)} references cited; main text (Sections 1-8) "
          f"{word_count(main)} words; abstract {word_count(by_name['00_front.md'].split('## Abstract')[-1].split('**Keywords')[0])} words;"
          f" unused reference entries: {unused}; figures moved to a new page: {moved or 'none'}")
    # Supplementary Material S1 (the former Appendix I, moved out on 21 Oct 2026): its own
    # citation numbers and reference list, its own PDF
    s_text = (ART / SUPP).read_text(encoding="utf-8").rstrip() + "\n\n"
    (s_text,), s_order = number_citations([s_text], refs)
    s_md = s_text + reference_list(s_order, refs) + "\n"
    (ART / f"{SUPP_STEM}.md").write_text(s_md, encoding="utf-8")
    s_title = re.search(r"^# (.+)$", s_text, re.M).group(1)
    s_pdf = ART / f"{SUPP_STEM}.pdf"
    s_n, _ = render_pdf(s_md, s_pdf, s_title)
    print(f"supplementary material: {s_n} page(s) -> {s_pdf}; {len(s_order)} references cited; {word_count(s_text)} words")
    sys.path.insert(0, str(ROOT / "scripts"))
    from presend_check import check
    rc = check(pdf, "article", extra=EXTRA_MARKERS)
    sys.exit(max(rc, check(s_pdf, "supplementary material", extra=EXTRA_MARKERS)))
