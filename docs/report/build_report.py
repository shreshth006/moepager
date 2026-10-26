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
md = md.replace(
    "<!-- REFERENCES -->", "\n\n".join(f"[{n}] {REFS[k]}" for k, n in order.items())
)
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
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Liberation Sans', Arial, sans-serif; font-size: 10.5pt; color: #111;
       line-height: 1.42; max-width: 180mm; margin: auto; }
h1 { font-size: 17pt; border-bottom: 2px solid #2a78d6; padding-bottom: 3px; margin-top: 22px;
     page-break-after: avoid; }
h2 { font-size: 13pt; margin-top: 16px; page-break-after: avoid; }
h3 { font-size: 11.5pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9c8c3; padding: 3px 6px; vertical-align: top; }
th { background: #eef3fb; }
img { max-width: 100%; display: block; margin: 8px auto; page-break-inside: avoid; }
code { font-family: 'Liberation Mono', monospace; font-size: 9pt; background: #f3f2ef; padding: 0 2px; }
pre { background: #f3f2ef; padding: 6px; font-size: 8.2pt; white-space: pre-wrap; page-break-inside: avoid; }
em { color: #444; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>moepager report</title><style>{css}</style></head><body>{body}</body></html>"
(here / "REPORT.html").write_text(html)
print("wrote REPORT.md, REPORT.html")
