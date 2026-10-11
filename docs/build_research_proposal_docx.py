#!/usr/bin/env python3
"""Build a Word .docx from RESEARCH_PROPOSAL.md with screenshot-matched styling.

Style targets (matching the user's Edge AI Research Proposal screenshots):
- Blue section / subsection headings
- Consolas green for inline code and fenced code blocks
- Yellow highlight + red text for revision-marked metrics still needing fill-in
- Bordered tables, embedded hardware + landmark figures
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
from docx.shared import Inches, Pt, RGBColor, Cm, Twips
from docx.enum.style import WD_STYLE_TYPE

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "RESEARCH_PROPOSAL.md"
OUT_PATH = ROOT / "RESEARCH_PROPOSAL.docx"
IMAGES = {
    "images/hardware_architecture_rpi5_appliance.jpg": ROOT
    / "images"
    / "hardware_architecture_rpi5_appliance.jpg",
    "images/facial_landmark_skeleton_liveness.jpg": ROOT
    / "images"
    / "facial_landmark_skeleton_liveness.jpg",
}

BLUE = RGBColor(0x05, 0x63, 0xC1)  # Word hyperlink / heading blue
DARK_BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREEN = RGBColor(0x00, 0x80, 0x00)
RED = RGBColor(0xFF, 0x00, 0x00)
BLACK = RGBColor(0x11, 0x11, 0x11)
ORANGE = RGBColor(0xC0, 0x50, 0x00)
TABLE_HEADER_BG = "D6EAF8"
CODE_BG = "F4F4F4"

REV_SPAN_RE = re.compile(
    r'<span\s+style="background-color:#FFFF00;color:#FF0000;">(.*?)</span>',
    re.DOTALL | re.IGNORECASE,
)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HTML_SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?(?:span|sub|sup|br\s*/?)>", re.IGNORECASE)
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")
FENCE_RE = re.compile(r"^```")
MATH_BLOCK_RE = re.compile(r"^\\\[|\\\]$")


def set_run_font(run, name="Times New Roman", size=11, bold=None, color=None, italic=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade_cell(cell, hex_color: str):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_borders(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "8")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "444444")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def add_page_number(section):
    """Footer style matching screenshots: Duy Anh Huynh | Edge AI Research Proposal • N"""
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = p.add_run("Duy Anh Huynh | Edge AI Research Proposal • ")
    set_run_font(run, "Times New Roman", 9, color=RGBColor(0x55, 0x55, 0x55))

    # PAGE field
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")

    r = p.add_run()
    r._r.append(fld_begin)
    r._r.append(instr)
    r._r.append(fld_sep)
    r._r.append(fld_end)
    set_run_font(r, "Times New Roman", 9, color=RGBColor(0x55, 0x55, 0x55))


def configure_styles(doc: Document):
    styles = doc.styles

    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = normal.paragraph_format
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.15

    # Match screenshot look: blue serif-ish section titles (use Georgia for H1/H2)
    heading_specs = {
        "Heading 1": (20, True, BLUE, Pt(0), Pt(14)),
        "Heading 2": (15, True, BLUE, Pt(18), Pt(8)),
        "Heading 3": (12.5, True, BLUE, Pt(12), Pt(6)),
        "Heading 4": (11, True, DARK_BLUE, Pt(10), Pt(4)),
    }
    for name, (size, bold, color, before, after) in heading_specs.items():
        st = styles[name]
        st.font.name = "Arial"
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.color.rgb = color
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        st.paragraph_format.space_before = before
        st.paragraph_format.space_after = after
        st.paragraph_format.keep_with_next = True

    # Code block style
    try:
        code_style = styles.add_style("CodeBlock", WD_STYLE_TYPE.PARAGRAPH)
    except ValueError:
        code_style = styles["CodeBlock"]
    code_style.font.name = "Consolas"
    code_style.font.size = Pt(9)
    code_style.font.color.rgb = GREEN
    code_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    code_style.paragraph_format.space_before = Pt(4)
    code_style.paragraph_format.space_after = Pt(4)
    code_style.paragraph_format.left_indent = Cm(0.3)


def strip_md_noise(text: str) -> str:
    text = text.replace("\\*", "*")
    text = text.replace("\\<", "<").replace("\\>", ">")
    text = html.unescape(text)
    return text


def add_inline_runs(paragraph, text: str, base_size=11):
    """Parse markdown/HTML inline spans into runs (revision, code, bold, italic, links, sub)."""
    text = strip_md_noise(text)

    # Tokenize revision spans first
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in REV_SPAN_RE.finditer(text):
        if m.start() > pos:
            parts.append(("plain", text[pos : m.start()]))
        parts.append(("rev", m.group(1)))
        pos = m.end()
    if pos < len(text):
        parts.append(("plain", text[pos:]))
    if not parts:
        parts = [("plain", text)]

    for kind, chunk in parts:
        if kind == "rev":
            # Still allow nested markdown inside revision span
            _emit_md_fragment(paragraph, chunk, base_size, revision=True)
        else:
            _emit_md_fragment(paragraph, chunk, base_size, revision=False)


def _emit_md_fragment(paragraph, text: str, base_size: int, revision: bool):
    # Replace <sub>..</sub> with unicode-ish markers then handle as plain with smaller font
    text = HTML_SUB_RE.sub(r"↓\1↑", text)
    text = HTML_TAG_RE.sub("", text)

    # Split on inline code, links, bold, italic in one pass via regex finditer of combined pattern
    token_re = re.compile(
        r"(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*)|↓[^↑]+↑)"
    )
    pos = 0
    for m in token_re.finditer(text):
        if m.start() > pos:
            _add_text_run(paragraph, text[pos : m.start()], base_size, revision)
        tok = m.group(0)
        if tok.startswith("`") and tok.endswith("`"):
            run = paragraph.add_run(tok[1:-1])
            set_run_font(run, "Consolas", max(9, base_size - 1), color=GREEN)
            if revision:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                run.font.color.rgb = RED
        elif tok.startswith("[") and "](" in tok:
            label, url = LINK_RE.match(tok).groups()
            run = paragraph.add_run(f"{label} ({url})")
            set_run_font(run, "Times New Roman", base_size, color=BLUE)
            if revision:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                run.font.color.rgb = RED
        elif tok.startswith("**") and tok.endswith("**"):
            run = paragraph.add_run(tok[2:-2])
            set_run_font(
                run,
                "Times New Roman",
                base_size,
                bold=True,
                color=RED if revision else BLACK,
            )
            if revision:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif tok.startswith("*") and tok.endswith("*"):
            run = paragraph.add_run(tok[1:-1])
            set_run_font(
                run,
                "Times New Roman",
                base_size,
                italic=True,
                color=RED if revision else BLACK,
            )
            if revision:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif tok.startswith("↓") and tok.endswith("↑"):
            run = paragraph.add_run(tok[1:-1])
            set_run_font(
                run,
                "Times New Roman",
                max(8, base_size - 2),
                color=RED if revision else BLACK,
            )
            if revision:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            # approximate subscript via vertAlign
            rPr = run._element.get_or_add_rPr()
            vert = OxmlElement("w:vertAlign")
            vert.set(qn("w:val"), "subscript")
            rPr.append(vert)
        else:
            _add_text_run(paragraph, tok, base_size, revision)
        pos = m.end()
    if pos < len(text):
        _add_text_run(paragraph, text[pos:], base_size, revision)


def _add_text_run(paragraph, text: str, base_size: int, revision: bool):
    if not text:
        return
    run = paragraph.add_run(text)
    set_run_font(
        run,
        "Times New Roman",
        base_size,
        color=RED if revision else BLACK,
    )
    if revision:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def add_heading_styled(doc, text: str, level: int):
    # Strip trailing markdown bold etc.
    clean = BOLD_RE.sub(r"\1", text.strip())
    clean = INLINE_CODE_RE.sub(r"\1", clean)
    clean = HTML_TAG_RE.sub("", clean)
    clean = strip_md_noise(clean)
    # Screenshot style: major section titles in ALL CAPS
    if level <= 2:
        clean = clean.upper()
    p = doc.add_heading(clean, level=level)
    # Re-apply blue on all runs (python-docx theme can override)
    color = BLUE if level <= 3 else DARK_BLUE
    font_name = "Georgia" if level <= 2 else "Arial"
    for run in p.runs:
        run.font.color.rgb = color
        run.font.name = font_name
        run.bold = True
        run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    return p


def add_table(doc, rows: list[list[str]]):
    if not rows:
        return
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    table.autofit = True

    for i, row in enumerate(rows):
        for j in range(cols):
            cell = table.rows[i].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            val = row[j] if j < len(row) else ""
            add_inline_runs(p, val, base_size=10)
            set_cell_borders(cell)
            if i == 0:
                shade_cell(cell, TABLE_HEADER_BG)
                for run in p.runs:
                    run.bold = True
                    if run.font.color.rgb != RED:
                        run.font.color.rgb = DARK_BLUE
    doc.add_paragraph()


def parse_table_row(line: str) -> list[str]:
    """Split a markdown table row, treating \\| as a literal pipe inside a cell."""
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    # Protect escaped pipes, then split
    placeholder = "\x00PIPE\x00"
    protected = inner.replace("\\|", placeholder)
    cells = [c.strip().replace(placeholder, "|") for c in protected.split("|")]
    return cells


def is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells if c)


def shade_paragraph(paragraph, hex_color: str):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    pPr.append(shd)


def add_code_block(doc, lines: list[str]):
    for line in lines:
        p = doc.add_paragraph(style="CodeBlock")
        run = p.add_run(line if line else " ")
        set_run_font(run, "Consolas", 9, color=GREEN)
        shade_paragraph(p, CODE_BG)


def add_hr(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "0563C1")
    pBdr.append(bottom)
    pPr.append(pBdr)


def convert_md_to_docx():
    md = MD_PATH.read_text(encoding="utf-8")
    lines = md.splitlines()

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    configure_styles(doc)
    add_page_number(section)

    i = 0
    in_code = False
    code_lines: list[str] = []
    in_math = False
    math_lines: list[str] = []

    while i < len(lines):
        line = lines[i]

        # Fenced code
        if FENCE_RE.match(line.strip()):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                add_code_block(doc, code_lines)
                code_lines = []
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Display math \[ ... \]
        if line.strip() == "\\[":
            in_math = True
            math_lines = []
            i += 1
            continue
        if in_math:
            if line.strip() == "\\]":
                in_math = False
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(" ".join(math_lines).strip())
                set_run_font(run, "Cambria Math", 11, italic=True, color=DARK_BLUE)
                i += 1
                continue
            math_lines.append(line.strip())
            i += 1
            continue

        # Blank
        if not line.strip():
            i += 1
            continue

        # Horizontal rule
        if re.fullmatch(r"-{3,}", line.strip()) or line.strip() == "***":
            add_hr(doc)
            i += 1
            continue

        # Headings
        hm = HEADING_RE.match(line)
        if hm:
            level = len(hm.group(1))
            add_heading_styled(doc, hm.group(2), level)
            i += 1
            continue

        # Images
        im = IMG_RE.search(line.strip())
        if im and line.strip().startswith("!["):
            alt, rel = im.group(1), im.group(2)
            path = IMAGES.get(rel) or (ROOT / rel)
            if path.exists():
                # Caption above
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cap.add_run(alt)
                set_run_font(run, "Arial", 10, bold=True, italic=True, color=DARK_BLUE)
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(str(path), width=Inches(5.8))
            else:
                p = doc.add_paragraph()
                add_inline_runs(p, f"[Missing image: {rel}]")
            i += 1
            continue

        # Tables
        if line.strip().startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = parse_table_row(lines[i])
                if not is_separator_row(cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            continue

        # Blockquote
        if line.startswith(">"):
            text = line.lstrip("> ").rstrip()
            # gather consecutive blockquote lines
            chunks = [text]
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                chunks.append(lines[i].lstrip("> ").rstrip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            shade_paragraph(p, "FFF8E1")
            add_inline_runs(p, " ".join(chunks))
            continue

        # Ordered list
        om = re.match(r"^(\d+)\.\s+(.*)$", line)
        if om:
            p = doc.add_paragraph(style="List Number")
            add_inline_runs(p, om.group(2))
            if REV_SPAN_RE.search(om.group(2)):
                shade_paragraph(p, "FCE4D6")
            i += 1
            continue

        # Unordered list
        if re.match(r"^[-*]\s+", line):
            body = re.sub(r"^[-*]\s+", "", line)
            p = doc.add_paragraph(style="List Bullet")
            add_inline_runs(p, body)
            if REV_SPAN_RE.search(body):
                shade_paragraph(p, "FCE4D6")
            i += 1
            continue

        # Sub-bullet indented
        if re.match(r"^\s{2,}[-*]\s+", line):
            body = re.sub(r"^\s*[-*]\s+", "", line)
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Cm(1.0)
            add_inline_runs(p, body)
            if REV_SPAN_RE.search(body):
                shade_paragraph(p, "FCE4D6")
            i += 1
            continue

        # Normal paragraph — merge soft-wrapped continuation? keep line-based for clarity
        # But merge consecutive non-empty non-special lines that are continuations of prose
        para_parts = [line.rstrip()]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if (
                HEADING_RE.match(nxt)
                or nxt.strip().startswith("|")
                or nxt.strip().startswith("![")
                or nxt.startswith(">")
                or FENCE_RE.match(nxt.strip())
                or re.fullmatch(r"-{3,}", nxt.strip())
                or re.match(r"^(\d+)\.\s+", nxt)
                or re.match(r"^[-*]\s+", nxt)
                or nxt.strip() in ("\\[", "\\]")
            ):
                break
            # don't merge if next looks like a short bold title line alone
            para_parts.append(nxt.rstrip())
            i += 1
        text = " ".join(para_parts)
        # Skip pure HTML comment leftovers
        if text.strip().startswith("<!--"):
            continue
        p = doc.add_paragraph()
        # Title-ish bold-only lines under H1
        if text.startswith("**") and text.endswith("**") and text.count("**") == 2:
            run = p.add_run(text[2:-2])
            set_run_font(run, "Georgia", 13, bold=True, color=DARK_BLUE)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            add_inline_runs(p, text)
            # Peach/orange shading like screenshots for revision-marked paragraphs
            if REV_SPAN_RE.search(text):
                shade_paragraph(p, "FCE4D6")

    doc.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    out = convert_md_to_docx()
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
