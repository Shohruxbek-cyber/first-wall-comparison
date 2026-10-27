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
        docs/article/Bozorov_IFE_first_wall_supplementary.{md,pdf} from docs/article/S0_title.md ... S8_*.md
The pre-send check (scripts/presend_check.py) fails the build while a draft marker or a
repository placeholder survives, unless --draft is given.
"""
import json
import os
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
EXTRA_MARKERS = ("[GitHub URL]", "[Zenodo DOI]", "[version]", "AUTHOR TO VERIFY", "[e-mail]", "[awaiting ")
SUPP_PARTS = ["S0_title.md", "S1_numerics.md", "S2_source_term.md", "S3_benchmarks_lithium.md",
              "S4_literature_inputs.md", "S5_limitations.md", "S6_tungsten.md", "S7_helium_model.md",
              "S8_reproduction.md"]  # S0: the title page
SUPP_STEM = "Bozorov_IFE_first_wall_supplementary"
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


CAPTION = re.compile(r"<p><strong>((Table|Figure) ((?:[A-C]\.|S)?\d+))\.</strong>(.*?)</p>", re.S)
BLOCK = re.compile(r"(?P<h><h[23]>)|(?P<cap><p><strong>(?:Table|Figure) (?:[A-C]\.|S)?\d+\.</strong>.*?</p>)"
                   r"|(?P<fig><p class=\"fig\"[^>]*><img id=\"(?P<figid>fig\d+)\")|(?P<tab><table>.*?</table>)|(?P<p><p(?: [^>]*)?>)", re.S)
BREAK = ' style="page-break-before: always"'


def mark_floats(html, breaks=(), splits=None):
    """Captions, table rows, figures and headings get ids, so that a layout pass can see the page each one
    falls on.  An id in `breaks` starts a new page; a table k with rows in splits[k] continues before each
    of them under a '(continued)' caption with its header repeated.  Returns the HTML and the floats in
    document order: ("table", caption label, [(caption id, its end-marker id, [row ids]), ...]),
    ("figure", image id, caption end-marker id, caption label) and ("heading", id, id of the next block or None)."""
    splits = splits or {}
    floats, n = [], {"cap": 0, "tab": 0, "h": 0}
    state = {"cap": None, "fig": None, "head": None}

    def nid(kind):
        n[kind] += 1
        return f"{kind}{n[kind]}"

    def follow(marker):
        """The block after a heading: tell the heading which id marks it."""
        if state["head"] is not None:
            floats[state["head"]] = ("heading", floats[state["head"]][1], marker)
            state["head"] = None

    def repl(m):
        if m.group("h"):
            k = nid("h")
            follow(k)
            floats.append(("heading", k, None))
            state["head"] = len(floats) - 1
            return f'{m.group("h")[:-1]} id="{k}"{BREAK if k in breaks else ""}>'
        if m.group("cap"):
            c = CAPTION.match(m.group("cap"))
            k = nid("cap")
            follow(k)
            cls = "tcap" if c.group(2) == "Table" else "fcap"
            state["cap"] = (k, c.group(1)) if cls == "tcap" else None
            if cls == "fcap":
                if state["fig"] is None:
                    raise RuntimeError(f"figure caption without a figure: {c.group(1)}")
                floats.append(("figure", state["fig"], k + "e", c.group(1)))
                state["fig"] = None
            return (f'<p class="{cls}" id="{k}"{BREAK if k in breaks else ""}><strong>{c.group(1)}.</strong>'
                    f'{c.group(4)}<span id="{k}e"></span></p>')
        if m.group("fig"):
            follow(m.group("figid"))
            state["fig"] = m.group("figid")
            return m.group("fig")
        if m.group("tab"):
            t = nid("tab")
            if state["cap"] is None:
                raise RuntimeError("table without a caption above it")
            cap, label = state["cap"]
            state["cap"] = None
            head = re.search(r"<thead>.*?</thead>", m.group("tab"), re.S).group(0)
            rows = re.findall(r"<tr>.*?</tr>", re.search(r"<tbody>(.*?)</tbody>", m.group("tab"), re.S).group(1), re.S)
            cuts = sorted(splits.get(t, ()))
            segs, out, start = [], [], 0
            for j in cuts + [len(rows)]:
                if start > 0:
                    c = f"{t}c{start}"
                    out.append(f'<p class="tcap" id="{c}"{BREAK if c in breaks else ""}><strong>{label} (continued).</strong>'
                               f'<span id="{c}e"></span></p>')
                else:
                    c = cap
                body = [r.replace("<tr>", f'<tr id="{t}r{i}"' + (' class="last"' if i == j - 1 else "") + ">", 1)
                        for i, r in zip(range(start, j), rows[start:j])]
                out.append(f'<table id="{t}">{head}<tbody>{"".join(body)}</tbody></table>')
                segs.append((c, c + "e", [f"{t}r{i}" for i in range(start, j)]))
                start = j
            floats.append(("table", label, segs))
            return "\n".join(out)
        if state["head"] is not None:  # a plain paragraph after a heading: mark its start
            k = floats[state["head"]][1] + "n"
            follow(k)
            return m.group("p") + f'<span id="{k}"></span>'
        return m.group("p")
    html = BLOCK.sub(repl, html)
    # the blank line after a table sits on the paragraph after it: a bottom margin on the table itself makes the
    # engine start the document over when the table ends at the foot of a page
    html = re.sub(r"(</table>\s*)<p>", r'\1<p class="aftertab">', html)
    # an empty block after every table: without it the engine (PyMuPDF 1.26) can start the document over when a
    # table ends exactly at the foot of a page (seen in the supplement, 27 Oct 2026)
    html = html.replace("</table>", "</table><div></div>")
    return html, floats


MAX_DEFER = 3  # a float may move past at most three paragraphs, never past a heading or another float
FLOAT_LABEL = re.compile(r"\*\*((Table|Figure) (?:[A-C]\.|S)?\d+)\.\*\*")


def float_blocks(blocks):
    """{label: (first, last)}: the Markdown blocks of every float (a caption and its table, an image and its caption)."""
    out = {}
    for i, b in enumerate(blocks):
        m = FLOAT_LABEL.match(b)
        if m and m.group(2) == "Table" and i + 1 < len(blocks) and blocks[i + 1].startswith("|"):
            out[m.group(1)] = (i, i + 1)
        elif m and m.group(2) == "Figure" and i > 0 and blocks[i - 1].startswith('<p class="fig"'):
            out[m.group(1)] = (i - 1, i)
    return out


def movable(b):
    """A block a float may move past: a paragraph or an equation, not a heading, table, figure or reference."""
    return bool(b.strip()) and not FLOAT_LABEL.match(b) and not b.startswith(("#", "|", '<p class="fig"', '<p class="ref"'))


def landing(blocks, spans, label):
    """Block index after each of the next paragraphs a float may move past, in order (at most MAX_DEFER).  It
    passes paragraphs, equations, subsection headings and floats of the other kind (a table may pass a figure);
    it stops at a section heading, a float of its own kind and the references, so tables and figures each keep
    their order."""
    a, b = spans[label]
    starts = {v[0]: (k, v) for k, v in spans.items()}
    out, i = [], b + 1
    while i < len(blocks) and len(out) < MAX_DEFER:
        x = blocks[i]
        if i in starts:
            other, (s, e) = starts[i]
            if other.split()[0] == label.split()[0]:
                break
            i = e + 1
        elif x.startswith("### "):
            i += 1
        elif movable(x):
            i += 1
            out.append(i)
        else:
            break
    return out


def defer_limits(md):
    blocks = re.split(r"\n[ \t]*\n", md)
    spans = float_blocks(blocks)
    return {label: len(landing(blocks, spans, label)) for label in spans}


def defer_floats(md, defer):
    """Move each float of `defer` past that many of the paragraphs after it (the way a float waits for the next
    page in typesetting); the text before it then fills the page instead of a gap."""
    for label in sorted(defer, key=lambda k: -float_blocks(re.split(r"\n[ \t]*\n", md))[k][0]):
        k = defer[label]
        if not k:
            continue
        blocks = re.split(r"\n[ \t]*\n", md)
        spans = float_blocks(blocks)
        a, b = spans[label]
        to = landing(blocks, spans, label)[k - 1]
        blocks = blocks[:a] + blocks[b + 1:to] + blocks[a:b + 1] + blocks[to:]
        md = "\n\n".join(blocks)
    return md


def render_pdf(md_text, pdf_path, title, base_pt=9.4):
    """Markdown -> A4 PDF.  Dry layout passes first move every float that would not be whole on one page:
    a figure the engine would shrink, or whose caption would fall on the next page, and a table caption
    whose table would run onto the next page, start a new page; a heading separated from the block after
    it starts a new page; a table taller than a page continues under a '(continued)' caption.  Returns the
    page count, the elements that start a new page, and the tables continued on a later page."""
    blank = round(base_pt * 1.3, 1)  # one blank line before and after a float and before a heading
    css = f"""
body {{ font-family: serif; font-size: {base_pt}pt; line-height: 1.3; }}
h1 {{ font-size: {base_pt + 5}pt; margin: 0 0 4pt 0; font-family: sans-serif; }}
h2 {{ font-size: {base_pt + 2}pt; margin: {blank + 2}pt 0 3pt 0; font-family: sans-serif; }}
h3 {{ font-size: {base_pt + 0.6}pt; margin: {blank}pt 0 2pt 0; font-family: sans-serif; font-style: italic; }}
p {{ margin: 2.5pt 0; text-align: justify; }}
li {{ margin: 1pt 0; }}
p.ref {{ font-size: {base_pt - 1}pt; margin: 1pt 0; text-align: left; }}
p.tcap {{ font-size: {base_pt - 0.6}pt; margin: {blank}pt 0 3pt 0; text-align: justify; }}
p.fcap {{ font-size: {base_pt - 0.6}pt; margin: 3pt 0 {blank}pt 0; text-align: justify; }}
table {{ border-collapse: collapse; margin: 0; }}
p.aftertab {{ margin-top: {blank}pt; }}
th, td {{ border: none; padding: 1.2pt 3pt; font-size: 8pt; vertical-align: top; font-family: sans-serif; text-align: left; }}
th {{ font-weight: bold; border-top: 0.8pt solid #000; border-bottom: 0.5pt solid #000; }}
td {{ border-bottom: 0.25pt solid #c4c4c4; }}
tr.last td {{ border-bottom: 0.8pt solid #000; }}
code {{ font-family: monospace; font-size: {base_pt - 1.4}pt; }}
p.fig {{ text-align: center; margin: {blank}pt 0 2pt 0; }}
"""
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (54, 50, -54, -50)
    text_w = where.width - 2 * base_pt  # the engine's body margin is 1 em

    limits = defer_limits(figure_html(md_text)[0])
    links = []  # (page, href, rect) of every <a href> fragment, recorded in the drawn pass

    def layout(breaks, splits, defer, writer=None):
        """One pass over all pages (drawn only with a writer); returns the page count, the figures drawn
        narrower than their width, the first page of every id, and the floats."""
        md, widths = figure_html(md_text, breaks)
        md = defer_floats(md, defer)
        html, floats = mark_floats(markdown.markdown(md, extensions=["tables"]), breaks, splits)
        story = pymupdf.Story(html=html, user_css=css, archive=pymupdf.Archive(str(ROOT / "docs")))
        drawn, page_of, more, pages = {}, {}, 1, 0

        def note(e):
            if e.id in widths:
                drawn[e.id] = max(drawn.get(e.id, 0.0), e.rect[2] - e.rect[0])
            if writer is not None and getattr(e, "href", None) and e.rect[2] > e.rect[0]:
                links.append((pages, e.href, e.rect))
            if e.id:
                if page_of and e.id == next(iter(page_of)) and pages > page_of[e.id] + 1:
                    restarted.append(pages)
                page_of.setdefault(e.id, pages)
        restarted, thin = [], []
        while more:
            more, filled = story.place(where)
            if more and filled[3] - filled[1] < 20.0:
                thin.append(pages + 1)  # a page holding next to nothing (e.g. a forced break right after a full page)
            story.element_positions(note)
            if restarted:
                raise RuntimeError(f"the layout engine started the document over on page {pages + 1}")
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
        if thin:
            raise RuntimeError(f"nearly empty page(s) {thin}: a forced page break falls right after a full page")
        return pages, [k for k in widths if drawn.get(k, 0.0) < min(widths[k], text_w) - 1.0], page_of, floats

    def first_problem(shrunk, page_of, floats, breaks, splits, tall, defer):
        """The first float in document order that is not whole on one page, and the change that fixes it: move
        it past the next paragraph while it may, else start it on a new page.  A float or heading right after a
        heading takes that heading with it to the new page."""
        lead = {f[2]: f[1] for f in floats if f[0] == "heading" and f[2] is not None}

        def fix(label, x):
            return ("defer", label) if defer.get(label, 0) < limits.get(label, 0) else ("break", x)

        def anchor(x):
            while x in lead:
                x = lead[x]
            return x
        for f in floats:
            if f[0] == "figure":
                img, cap_end = f[1], f[2]
                if img in shrunk or page_of[img] != page_of[cap_end]:
                    if anchor(img) in breaks:
                        raise RuntimeError(f"{img} and its caption do not fit on one page")
                    return fix(f[3], anchor(img))
            elif f[0] == "heading":
                if f[2] is not None and page_of[f[1]] != page_of[f[2]]:
                    if anchor(f[1]) in breaks:
                        raise RuntimeError(f"heading {f[1]} cannot be kept with the block after it")
                    return ("break", anchor(f[1]))
            else:
                label, segs = f[1], f[2]
                t = segs[0][2][0].split("r")[0]
                for cap, cap_end, rows in segs:
                    on = [page_of[cap], page_of[cap_end]] + [page_of[r] for r in rows]
                    if len(set(on)) == 1:
                        continue
                    if t not in tall:
                        if anchor(cap) not in breaks:
                            return fix(label, anchor(cap))
                        return ("tall", t, anchor(cap))  # split even at the top of a page: continue it instead
                    later = [i for i, r in enumerate(rows) if page_of[r] > page_of[cap_end]]
                    top = anchor(cap) in breaks  # the caption already starts a page
                    # two rows under a caption, one under a "(continued)" caption or at the top of a page
                    if not later or later[0] < (1 if top or cap != segs[0][0] else 2):
                        if top:
                            raise RuntimeError(f"{label}: no room for a row under its caption")
                        return ("break", anchor(cap))
                    return ("split", t, int(rows[later[0]].split("r")[1]))
        return None

    breaks, splits, tall, defer = set(), {}, set(), {}
    while True:
        pages, shrunk, page_of, floats = layout(breaks, splits, defer)
        fix = first_problem(shrunk, page_of, floats, breaks, splits, tall, defer)
        if os.environ.get("LAYOUT_DEBUG"):
            print("  layout:", pages, "pages; fix", fix)
        if fix is None:
            break
        if fix[0] == "break":
            breaks.add(fix[1])
        elif fix[0] == "defer":
            defer[fix[1]] = defer.get(fix[1], 0) + 1
        elif fix[0] == "tall":
            tall.add(fix[1])
            breaks.discard(fix[2])
        else:  # the bottom rule of a continued part can push its last row over: then cut one row earlier
            splits.setdefault(fix[1], set()).discard(fix[2] + 1)
            splits[fix[1]].add(fix[2])
    raw = pdf_path.with_name("_" + pdf_path.stem + ".raw.pdf")
    writer = pymupdf.DocumentWriter(str(raw), "compress")  # the 600-dpi figures: 0.26 GB uncompressed
    pages, shrunk, page_of, floats = layout(breaks, splits, defer, writer)
    writer.close()
    del writer  # releases the file, so that the unlink below works on Windows
    if shrunk or first_problem(shrunk, page_of, floats, breaks, splits, tall, {k: MAX_DEFER for k in limits}) is not None:
        raise RuntimeError(f"a float is not whole on one page in the final layout: {shrunk}")
    doc = pymupdf.open(str(raw))
    for i, page in enumerate(doc):  # page numbers, in the embedded body font (a Base-14 font would not be embedded)
        page.insert_htmlbox(pymupdf.Rect(0, mediabox.height - 43, mediabox.width, mediabox.height - 20), str(i + 1),
                            css="* {font-family: serif; font-size: 8pt; text-align: center; margin: 0}")
    lines = {}  # the writer keeps no links: one clickable rectangle per link and line (the engine reports fragments)
    for pg, href, r in links:
        k = (pg, href, round(r[1], 1), round(r[3], 1))
        lines[k] = lines[k] | pymupdf.Rect(r) if k in lines else pymupdf.Rect(r)
    for (pg, href, _, _), r in lines.items():
        doc[pg].insert_link({"kind": pymupdf.LINK_URI, "from": r, "uri": href})
    doc.set_metadata({"title": title, "author": "Shohruxbek Bozorov", "subject": title,  # review S-14: subject = title
                      "creator": "ife_wall scripts/make_article.py"})
    doc.save(str(pdf_path), garbage=4, deflate=True, deflate_images=True, clean=True)
    doc.close()
    try:
        raw.unlink()
    except OSError:
        pass
    names = {}
    for f in floats:
        if f[0] == "table":
            names.update({seg[0]: f[1] for seg in f[2]})
            names[f[2][0][2][0].split("r")[0]] = f[1]
        elif f[0] == "figure":
            names[f[1]] = f[3]
    heads = dict(re.findall(r'<h[23] id="(h\d+)"[^>]*>(.*?)</h', mark_floats(markdown.markdown(
        defer_floats(figure_html(md_text)[0], defer), extensions=["tables"]))[0]))
    names.update({k: "heading " + re.sub(r"<[^>]+>", "", v)[:40] for k, v in heads.items()})
    moved = [names.get(k, k) for k in breaks] + [f"{k} after {v} paragraph(s)" for k, v in defer.items()]
    continued = [f"{names[t]} ({len(v)} continuation(s))" for t, v in splits.items()]
    return pages, moved, continued


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
    n, moved, continued = render_pdf(md, pdf, title)
    main = "".join(by_name[p] for p in names[1:names.index("09_declarations.md")])
    unused = sorted(set(refs) - set(order))
    print(f"article: {n} page(s) -> {pdf}; {len(order)} references cited; main text (Sections 1-8) "
          f"{word_count(main)} words; abstract {word_count(by_name['00_front.md'].split('## Abstract')[-1].split('**Keywords')[0])} words;"
          f" unused reference entries: {unused}; moved to a new page: {moved or 'none'}; continued: {continued or 'none'}")
    # Supplementary material (since 23 Oct 2026 every appendix but A-C, as Sections S1-S8): its own title page
    # naming the paper, its own citation numbers and reference list, its own PDF
    s_text = "".join((ART / p).read_text(encoding="utf-8").rstrip() + "\n\n" for p in SUPP_PARTS)
    s_text = s_text.replace("{{TITLE}}", title)
    (s_text,), s_order = number_citations([s_text], refs)
    s_md = s_text + reference_list(s_order, refs) + "\n"
    (ART / f"{SUPP_STEM}.md").write_text(s_md, encoding="utf-8")
    s_title = "Supplementary material: " + title
    s_pdf = ART / f"{SUPP_STEM}.pdf"
    s_n, s_moved, s_continued = render_pdf(s_md, s_pdf, s_title)
    print(f"supplementary material: {s_n} page(s) -> {s_pdf}; {len(s_order)} references cited; {word_count(s_text)} words;"
          f" moved to a new page: {s_moved or 'none'}; continued: {s_continued or 'none'}")
    sys.path.insert(0, str(ROOT / "scripts"))
    from presend_check import check
    rc = check(pdf, "article", extra=EXTRA_MARKERS)
    sys.exit(max(rc, check(s_pdf, "supplementary material", extra=EXTRA_MARKERS)))
