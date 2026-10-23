# -*- coding: utf-8 -*-
"""Build a bilingual review DOCX from the English and Chinese Markdown files."""
from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "chapter4_revised_en.md"
ZH = ROOT / "chapter4_revised_zh.md"
OUT = ROOT / "chapter4_revised_bilingual.docx"

SECTION_PAGEBREAK = {"Experimental protocol", "实验方案", "Validation of the final surrogate",
                     "最终代理模型验证", "Comparative performance under frozen specifications",
                     "冻结规格下的比较性能", "Geometry and diversity of accepted designs",
                     "合格设计的几何与多样性", "Targeted analysis and practical implications",
                     "针对性分析与实践含义"}


def set_font(run, size=10, bold=None, italic=None, color=None, east_asia="Microsoft YaHei"):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east_asia)


def add_rich_text(paragraph, text, size=10, color=None):
    parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
    for part in parts:
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        italic = part.startswith("*") and part.endswith("*") and not bold
        content = part[2:-2] if bold else part[1:-1] if italic else part
        run = paragraph.add_run(content)
        set_font(run, size=size, bold=bold or None, italic=italic or None, color=color)


def parse_md(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = []
    current = None
    buf = []

    def flush():
        nonlocal buf
        if current is not None and buf:
            blocks = []
            i = 0
            while i < len(buf):
                if not buf[i].strip():
                    i += 1
                    continue
                if buf[i].lstrip().startswith("|"):
                    table = []
                    while i < len(buf) and buf[i].lstrip().startswith("|"):
                        table.append(buf[i].strip())
                        i += 1
                    blocks.append(("table", table))
                    continue
                if buf[i].startswith("!["):
                    blocks.append(("image", buf[i].strip()))
                    i += 1
                    continue
                para = [buf[i].strip()]
                i += 1
                while i < len(buf) and buf[i].strip() and not buf[i].lstrip().startswith("|") \
                        and not buf[i].startswith("!["):
                    para.append(buf[i].strip())
                    i += 1
                blocks.append(("text", " ".join(para)))
            current["blocks"] = blocks
        buf = []

    for line in lines:
        m = re.match(r"^(#{1,2})\s+(.*)$", line)
        if m:
            flush()
            current = {"level": len(m.group(1)), "title": m.group(2).strip(), "blocks": []}
            sections.append(current)
        else:
            buf.append(line)
    flush()
    return sections


def strip_table_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def add_table(doc, lines):
    rows = [strip_table_row(x) for x in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for ri, row in enumerate(rows):
        for ci in range(ncols):
            cell = table.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            text = row[ci] if ci < len(row) else ""
            add_rich_text(p, text, size=7.2)
            if ri == 0:
                for run in p.runs:
                    run.bold = True
        if ri == 0:
            trPr = table.rows[0]._tr.get_or_add_trPr()
            tblHeader = OxmlElement("w:tblHeader")
            tblHeader.set(qn("w:val"), "true")
            trPr.append(tblHeader)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    set_font(run, size=8)


def add_label(doc, text, color):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run(text)
    set_font(run, size=7.5, bold=True, color=color)


def main():
    en = parse_md(EN)
    zh = parse_md(ZH)
    if len(en) != len(zh):
        raise RuntimeError(f"Section count differs: {len(en)} vs {len(zh)}")

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(1.8)
    sec.bottom_margin = Cm(1.8)
    sec.left_margin = Cm(1.8)
    sec.right_margin = Cm(1.8)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(10)
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    header = sec.header.paragraphs[0]
    header.text = "Chapter 4 revision | 第四章修订审阅稿"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        set_font(run, size=8, color=RGBColor(100, 100, 100))
    add_page_number(sec.footer.paragraphs[0])

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(80)
    run = p.add_run("Chapter 4\nResults and discussion")
    set_font(run, size=22, bold=True, color=RGBColor(31, 72, 110))
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p2.add_run("第四章\n结果与讨论")
    set_font(run, size=20, bold=True, color=RGBColor(31, 72, 110))
    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3.paragraph_format.space_before = Pt(24)
    run = p3.add_run("Bilingual review copy | 中英文对照审阅版\nBased on the frozen TST subset and existing ROSS records\n未重新训练、未新增候选生成、未新增 ROSS 计算")
    set_font(run, size=10, color=RGBColor(80, 80, 80))
    doc.add_page_break()

    for si, (se, sz) in enumerate(zip(en, zh)):
        if se["title"] != "Results and discussion" and se["title"] in SECTION_PAGEBREAK:
            doc.add_page_break()
        level = se["level"]
        if level == 1:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(se["title"])
            set_font(run, size=16, bold=True, color=RGBColor(31, 72, 110))
            pz = doc.add_paragraph()
            pz.paragraph_format.space_after = Pt(10)
            run = pz.add_run(sz["title"])
            set_font(run, size=15, bold=True, color=RGBColor(31, 72, 110))
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(se["title"])
            set_font(run, size=12.5, bold=True, color=RGBColor(31, 72, 110))
            pz = doc.add_paragraph()
            pz.paragraph_format.space_after = Pt(4)
            run = pz.add_run(sz["title"])
            set_font(run, size=12, bold=True, color=RGBColor(31, 72, 110))

        if len(se["blocks"]) != len(sz["blocks"]):
            raise RuntimeError(f"Block count differs in {se['title']}: {len(se['blocks'])} vs {len(sz['blocks'])}")

        for be, bz in zip(se["blocks"], sz["blocks"]):
            if be[0] != bz[0]:
                raise RuntimeError(f"Block type differs in {se['title']}: {be[0]} vs {bz[0]}")
            if be[0] == "table":
                add_table(doc, be[1])
                doc.add_paragraph().paragraph_format.space_after = Pt(2)
            elif be[0] == "image":
                m = re.search(r"\((.*?)\)", be[1])
                if m:
                    path = ROOT / m.group(1)
                    pimg = doc.add_paragraph()
                    pimg.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    pimg.add_run().add_picture(str(path), width=Cm(16.4))
            else:
                if be[1].startswith("*Note:*"):
                    add_label(doc, "Note / 注", RGBColor(100, 100, 100))
                    p = doc.add_paragraph()
                    p.paragraph_format.space_after = Pt(5)
                    add_rich_text(p, be[1], size=8.5)
                    pz = doc.add_paragraph()
                    pz.paragraph_format.space_after = Pt(8)
                    add_rich_text(pz, bz[1], size=8.5)
                else:
                    pe = doc.add_paragraph()
                    pe.paragraph_format.space_after = Pt(2)
                    pe.paragraph_format.line_spacing = 1.12
                    add_rich_text(pe, be[1], size=10)
                    pz = doc.add_paragraph()
                    pz.paragraph_format.space_after = Pt(8)
                    pz.paragraph_format.line_spacing = 1.18
                    add_rich_text(pz, bz[1], size=10)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
