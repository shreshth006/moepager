#!/usr/bin/env python3
"""Assemble docs/report/sections/*.md into REPORT.md and REPORT.html (for PDF)."""
import re
import sys
from pathlib import Path

import markdown

here = Path(__file__).parent
sys.path.insert(0, str(here))
from references import REFS  # noqa: E402

parts = [p.read_text() for p in sorted((here / "sections").glob("*.md"))]
md = "\n\n".join(parts)

# Citations: [@a] or [@a; @b] -> IEEE numbers in order of first citation.
order: dict[str, int] = {}


def cite(m: re.Match) -> str:
    keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
    for k in keys:
        if k not in REFS:
            sys.exit(f"unknown citation key: {k}")
        order.setdefault(k, len(order) + 1)
    return "[" + ", ".join(str(order[k]) for k in keys) + "]"


md = re.sub(r"\[(@[\w-]+(?:;\s*@[\w-]+)*)\]", cite, md)
unused = sorted(set(REFS) - set(order))
if unused:
    sys.exit(f"references never cited: {', '.join(unused)}")
refs = "\n\n".join(f"[{n}] {REFS[k]}" for k, n in order.items())
md = md.replace("<!-- REFERENCES -->", f'<div class="refs" markdown="1">\n\n{refs}\n\n</div>')
# Python-Markdown nests lists only at 4-space indents; the sections use 2.
# Lines inside top-level ``` fences are left alone.
out, fenced = [], False
for line in md.split("\n"):
    if line.startswith("```"):
        fenced = not fenced
    elif line.startswith("  ") and not fenced:
        line = " " * (2 * (len(line) - len(line.lstrip(" ")))) + line.lstrip(" ")
    # Python-Markdown needs a blank line between a paragraph and a list.
    elif (
        re.match(r"(- |\d+\. )", line)
        and out
        and out[-1].strip()
        and not re.match(r"(\s|- |\d+\. |\|)", out[-1])
        and not fenced
    ):
        out.append("")
    out.append(line)
md = "\n".join(out)
(here / "REPORT.md").write_text(md)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "md_in_html"])

# --- Paper layout: numbered captions, and which blocks span both columns.
NARROW_FIGS = {"fig-sim-prose.png"}
fig_no = 0


def figure(m: re.Match) -> str:
    global fig_no
    fig_no += 1
    src, alt, cap = m.group("src"), m.group("alt"), m.group("cap")
    cls = "narrow" if Path(src).name in NARROW_FIGS else "span"
    return (
        f'<figure class="{cls}"><img src="{src}" alt="{alt}">'
        f'<figcaption><span class="fno">Fig. {fig_no}.</span> {cap}</figcaption></figure>'
    )


img = r'<img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+)" ?/?>'
body = re.sub(
    rf"<p>{img}\s*(?:</p>\s*<p>)?<em>Figure: (?P<cap>.*?)</em></p>", figure, body, flags=re.S
)
if "<img" in re.sub(r"<figure.*?</figure>", "", body, flags=re.S):
    sys.exit("figure without a '*Figure: ...*' caption")


def roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    r = ""
    for v, sym in vals:
        while n >= v:
            r, n = r + sym, n - v
    return r


tab_no = 0


def table(m: re.Match) -> str:
    global tab_no
    tab_no += 1
    cap, inner = m.group("cap"), m.group("inner")
    rows = re.findall(r"<tr>(.*?)</tr>", inner, flags=re.S)
    cols = len(re.findall(r"<t[hd][ >]", rows[0])) if rows else 0
    longest = max((len(re.sub(r"<[^>]+>", "", r).strip()) for r in rows), default=0)
    cls = "span" if cols >= 4 or longest > 70 else "narrow"
    return (
        f'<table class="{cls}"><caption><span class="tno">Table {roman(tab_no)}</span>'
        f'<span class="tcap">{cap}</span></caption>{inner}</table>'
    )


body = re.sub(
    r"<p>Table: (?P<cap>.*?)</p>\s*<table>(?P<inner>.*?)</table>", table, body, flags=re.S
)
if body.count("<table>"):
    sys.exit("table without a 'Table: ...' caption")


def pre(m: re.Match) -> str:
    text = re.sub(r"<[^>]+>", "", m.group(0))
    longest = max(len(line) for line in text.split("\n"))
    cls = "span" if longest > 56 else "narrow"
    if text.count("\n") > 24:
        cls += " long"  # may break across pages rather than leave a gap
    return m.group(0).replace("<pre>", f'<pre class="{cls}">', 1)


body = re.sub(r"<pre>.*?</pre>", pre, body, flags=re.S)
front, _, main = body.partition("<!-- BODY -->")

css = """
@page { size: A4; margin: 18mm 15mm 20mm 15mm;
        @bottom-center { content: counter(page); font: 9pt 'Nimbus Roman', 'Liberation Serif', serif; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Nimbus Roman', 'Liberation Serif', 'Times New Roman', serif; font-size: 10pt;
       line-height: 1.22; color: #000; background: #fff; margin: 0; }
.paper-title { text-align: center; margin: 0 0 14pt; }
.paper-title .title { font-size: 22pt; line-height: 1.15; margin: 0 6mm 10pt; }
.paper-title .authors { font-size: 11pt; margin: 0 0 4pt; }
.paper-title .meta { font-size: 8.5pt; margin: 0 0 2pt; }
main { column-count: 2; column-gap: 6mm; text-align: justify; hyphens: auto; }
p { margin: 0; text-indent: 1em; orphans: 2; widows: 2; }
li p, td p, figcaption, .abstract p, .refs p, caption { text-indent: 0; }
.abstract { font-size: 9pt; font-weight: bold; line-height: 1.2; }
.abstract p { margin: 0 0 3pt; }
.abstract ul, .abstract ol { margin: 0 0 3pt; }
.abstract code { font-weight: normal; }
.index-terms { font-size: 9pt; font-weight: bold; margin: 4pt 0 6pt !important; }
h1 { font-size: 10pt; font-weight: normal; font-variant: small-caps; letter-spacing: 0.03em;
     text-align: center; margin: 10pt 0 4pt; break-after: avoid; }
h2 { font-size: 10pt; font-weight: normal; font-style: italic; text-align: left;
     margin: 7pt 0 2pt; break-after: avoid; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 5pt 0 1pt; break-after: avoid; }
ul, ol { margin: 2pt 0 3pt; padding-left: 1.3em; }
li { margin: 0 0 1pt; }
li > ul, li > ol { margin: 1pt 0; }
code { font-family: 'Liberation Mono', 'Nimbus Mono PS', monospace; font-size: 0.82em;
       overflow-wrap: anywhere; hyphens: none; }
pre { font-size: 7.4pt; line-height: 1.25; margin: 5pt 0; padding: 3pt 0;
      border-top: 0.6pt solid #000; border-bottom: 0.6pt solid #000;
      white-space: pre-wrap; text-align: left; break-inside: avoid; }
pre code { font-size: 1em; }
pre.long { break-inside: auto; }
.paper-title p { text-indent: 0; }
.span { column-span: all; }
figure { margin: 6pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; display: block; margin: 0 auto 3pt; }
figure.span img { max-height: 92mm; }
figcaption { font-size: 8pt; text-align: justify; text-align-last: center; }
.fno { font-weight: normal; }
table { border-collapse: collapse; margin: 6pt auto 7pt; font-size: 7.6pt; line-height: 1.18;
        text-align: left; break-inside: avoid; border-top: 1pt solid #000; border-bottom: 1pt solid #000; }
table.span { width: 100%; font-size: 8pt; }
table.narrow { width: 100%; }
caption { caption-side: top; font-size: 8pt; text-align: center; padding-bottom: 4pt; }
.tno { display: block; font-variant: small-caps; letter-spacing: 0.04em; font-size: 8.5pt; }
.tcap { font-variant: small-caps; }
th { font-weight: normal; font-style: italic; border-bottom: 0.6pt solid #000; padding: 2pt 4pt;
     vertical-align: bottom; }
td { padding: 1.5pt 4pt; vertical-align: top; }
.refs { font-size: 8pt; line-height: 1.2; text-align: left; }
.refs p { padding-left: 2.6em; text-indent: -2.6em; margin: 0 0 2pt; }
a { color: inherit; text-decoration: none; }
"""
html = (
    "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
    f"<title>moepager</title><style>{css}</style></head>"
    f"<body>{front}<main>{main}</main></body></html>"
)
(here / "REPORT.html").write_text(html)
print(f"wrote REPORT.md, REPORT.html ({fig_no} figures, {tab_no} tables)")
