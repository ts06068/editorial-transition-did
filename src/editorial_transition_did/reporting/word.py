from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from tqdm.auto import tqdm

from ..config import AnalysisPaths


def _configure_word_document(document: Document) -> None:
    # Landscape letter paper with one-inch margins leaves nine inches for tables.
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1


def _set_word_cell_margins(
    cell,
    vertical_twips: int = 15,
    horizontal_twips: int = 35,
) -> None:
    # python-docx lacks a cell-margin API; OOXML stores these values in twips.
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for edge in ("top", "start", "bottom", "end"):
        element = margins.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            margins.append(element)
        margin = vertical_twips if edge in {"top", "bottom"} else horizontal_twips
        element.set(qn("w:w"), str(margin))
        element.set(qn("w:type"), "dxa")


def _word_border_element(cell, edge: str):
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    element = borders.find(qn(f"w:{edge}"))
    if element is None:
        element = OxmlElement(f"w:{edge}")
        borders.append(element)
    return element


def _clear_word_cell_borders(cell) -> None:
    # Remove default borders before adding the required horizontal rules.
    for edge in ("top", "start", "bottom", "end", "insideH", "insideV"):
        element = _word_border_element(cell, edge)
        element.set(qn("w:val"), "nil")


def _update_word_cell_borders(
    cell,
    top: bool | None = None,
    bottom: bool | None = None,
) -> None:
    for edge, enabled in (("top", top), ("bottom", bottom)):
        if enabled is None:
            continue
        element = _word_border_element(cell, edge)
        element.set(qn("w:val"), "single" if enabled else "nil")
        if enabled:
            element.set(qn("w:sz"), "6")
            element.set(qn("w:space"), "0")
            element.set(qn("w:color"), "000000")


def _manuscript_word_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime | pd.Timestamp | np.datetime64):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return "NA"
    if isinstance(value, bool | np.bool_):
        return "Yes" if value else "No"
    return str(value)


@dataclass(frozen=True)
class ManuscriptWordRow:
    # Group rows carry section boundaries, not data.
    values: tuple[object, ...]
    kind: str = "data"


@dataclass
class ManuscriptWordBlock:
    # Each block is an independently headed table panel.
    headers: list[list[str]]
    rows: list[ManuscriptWordRow]
    panel_title: str | None = None
    header_merges: list[tuple[int, int, int, int]] = field(default_factory=list)
    header_underlines: list[tuple[int, int, int]] = field(default_factory=list)
    column_widths: list[float] = field(default_factory=list)
    alignments: list[str] = field(default_factory=list)
    font_size: float = 10.0


@dataclass
class ManuscriptWordTable:
    # Specs separate table content from OOXML rendering.
    output_path: Path
    title: str
    blocks: list[ManuscriptWordBlock]
    footnotes: list[str] = field(default_factory=list)


def _set_word_table_layout(table, widths: Sequence[float]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    properties = table._tbl.tblPr
    layout = properties.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        properties.append(layout)
    layout.set(qn("w:type"), "fixed")
    # Set both width representations; Word viewers disagree about which one wins.
    for column, width in zip(table.columns, widths, strict=False):
        column.width = Inches(width)
    for row in table.rows:
        for cell, width in zip(row.cells, widths, strict=False):
            cell.width = Inches(width)
            properties = cell._tc.get_or_add_tcPr()
            cell_width = properties.first_child_found_in("w:tcW")
            if cell_width is None:
                cell_width = OxmlElement("w:tcW")
                properties.append(cell_width)
            cell_width.set(qn("w:w"), str(int(width * 1440)))
            cell_width.set(qn("w:type"), "dxa")


def _set_manuscript_cell_text(
    cell,
    value: object,
    alignment: str,
    font_size: float,
    bold: bool = False,
    keep_with_next: bool = False,
) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    _set_word_cell_margins(cell)
    paragraph = cell.paragraphs[0]
    paragraph.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }.get(alignment, WD_ALIGN_PARAGRAPH.CENTER)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1
    paragraph.paragraph_format.keep_with_next = keep_with_next
    run = paragraph.add_run(_manuscript_word_text(value))
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(font_size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def _append_manuscript_title(document: Document, title: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    # Bold the table identifier, not the title.
    match = re.match(r"^((?:Supplemental )?Table [^.]+\.)\s*(.*)$", title)
    parts = (match.group(1), match.group(2)) if match else (title, "")
    for index, text_value in enumerate(parts):
        if not text_value:
            continue
        prefix = " " if index == 1 else ""
        run = paragraph.add_run(prefix + text_value)
        run.bold = index == 0
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def _append_panel_title(document: Document, title: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    match = re.match(r"^(Panel [A-Z]+\.)\s*(.*)$", title)
    parts = (match.group(1), match.group(2)) if match else (title, "")
    for index, text_value in enumerate(parts):
        if not text_value:
            continue
        prefix = " " if index == 1 else ""
        run = paragraph.add_run(prefix + text_value)
        run.bold = index == 0
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def _append_manuscript_block(document: Document, block: ManuscriptWordBlock) -> None:
    if block.panel_title:
        _append_panel_title(document, block.panel_title)
    column_count = max(len(row) for row in block.headers)
    # Reject rows that do not match the declared table geometry.
    if any(len(row.values) != column_count for row in block.rows):
        raise ValueError(f"Word table block has inconsistent row width: {block.panel_title}")
    header_count = len(block.headers)
    table = document.add_table(rows=header_count + len(block.rows), cols=column_count)
    widths = block.column_widths or [9.0 / column_count] * column_count
    # Preserve column proportions while fitting the printable width.
    if not math.isclose(sum(widths), 9.0, abs_tol=0.05):
        scale = 9.0 / sum(widths)
        widths = [width * scale for width in widths]
    _set_word_table_layout(table, widths)
    alignments = block.alignments or ["center"] * column_count

    for row_index, header_values in enumerate(block.headers):
        # Pad short header tiers and keep each tier on one page.
        padded = [*header_values, *([""] * (column_count - len(header_values)))]
        row = table.rows[row_index]
        row_properties = row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
        for column_index, value in enumerate(padded):
            cell = row.cells[column_index]
            _clear_word_cell_borders(cell)
            _set_manuscript_cell_text(
                cell,
                value,
                "center",
                block.font_size,
                keep_with_next=True,
            )

    for start_row, start_column, end_row, end_column in block.header_merges:
        # Word may retain text from merged cells; restore the anchor label.
        merged_cell = table.cell(start_row, start_column).merge(table.cell(end_row, end_column))
        _set_manuscript_cell_text(
            merged_cell,
            block.headers[start_row][start_column],
            "center",
            block.font_size,
            keep_with_next=True,
        )

    for body_index, body_row in enumerate(block.rows):
        table_row_index = header_count + body_index
        row = table.rows[table_row_index]
        row_properties = row._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
        is_group = body_row.kind == "group"
        for column_index, value in enumerate(body_row.values):
            cell = row.cells[column_index]
            _clear_word_cell_borders(cell)
            _set_manuscript_cell_text(
                cell,
                value,
                alignments[column_index],
                block.font_size,
                bold=is_group,
                keep_with_next=is_group,
            )

    # Manuscript tables use sparse horizontal rules only.
    for column_index in range(column_count):
        _update_word_cell_borders(table.cell(0, column_index), top=True)
        _update_word_cell_borders(table.cell(header_count - 1, column_index), bottom=True)
    for row_index, start_column, end_column in block.header_underlines:
        for column_index in range(start_column, end_column + 1):
            _update_word_cell_borders(table.cell(row_index, column_index), bottom=True)
    for body_index, body_row in enumerate(block.rows):
        if body_row.kind == "group" and body_index > 0:
            table_row_index = header_count + body_index
            for column_index in range(column_count):
                _update_word_cell_borders(table.cell(table_row_index, column_index), top=True)
    if block.rows:
        for column_index in range(column_count):
            _update_word_cell_borders(
                table.cell(header_count + len(block.rows) - 1, column_index),
                bottom=True,
            )


def _append_manuscript_footnotes(document: Document, footnotes: Sequence[str]) -> None:
    for index, footnote in enumerate(footnotes):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(4 if index == 0 else 0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1
        run = paragraph.add_run(footnote)
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def _append_manuscript_table(document: Document, spec: ManuscriptWordTable) -> None:
    _append_manuscript_title(document, spec.title)
    for block in spec.blocks:
        _append_manuscript_block(document, block)
    _append_manuscript_footnotes(document, spec.footnotes)


def _append_table_legends(
    document: Document,
    specs: Sequence[ManuscriptWordTable],
) -> None:
    heading = document.add_paragraph()
    heading.paragraph_format.space_after = Pt(8)
    run = heading.add_run("Table Legends")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for spec in specs:
        _append_manuscript_title(document, spec.title)


def _write_manuscript_bundle(
    specs: Sequence[ManuscriptWordTable],
    output_path: Path,
) -> None:
    document = Document()
    _configure_word_document(document)
    _append_table_legends(document, specs)
    # Put legends first and start every complete table on a new page.
    for spec in specs:
        document.add_page_break()
        _append_manuscript_table(document, spec)
    document.save(output_path)


def write_manuscript_word_tables(
    main_specs: Sequence[ManuscriptWordTable],
    supplemental_specs: Sequence[ManuscriptWordTable],
    paths: AnalysisPaths,
) -> list[Path]:
    outputs: list[Path] = []
    all_specs = [*main_specs, *supplemental_specs]
    # Write standalone tables for editing and bundles for review.
    for spec in tqdm(all_specs, desc="Writing manuscript-style Word tables", leave=False):
        document = Document()
        _configure_word_document(document)
        _append_manuscript_table(document, spec)
        document.save(spec.output_path)
        outputs.append(spec.output_path)

    main_bundle = paths.main_table_dir / "Main_Tables.docx"
    supplemental_bundle = paths.supp_table_dir / "Supplementary_Tables.docx"
    combined_bundle = paths.output_dir / "CLEAR_CARD_editorial_analysis_tables.docx"
    _write_manuscript_bundle(main_specs, main_bundle)
    _write_manuscript_bundle(supplemental_specs, supplemental_bundle)
    _write_manuscript_bundle(all_specs, combined_bundle)
    outputs.extend([main_bundle, supplemental_bundle, combined_bundle])
    return outputs
