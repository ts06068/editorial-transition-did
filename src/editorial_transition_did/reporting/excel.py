"""Excel table writers used by the editorial-transition analysis."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CATEGORY_ORDER, COHORTS, JOURNAL_DISPLAY, JOURNAL_ORDER


def _excel_formats(workbook) -> dict[str, object]:
    """Build the shared manuscript-style Excel formats."""
    # Centralized formats prevent visual drift between standalone and combined workbooks.
    base = {"font_name": "Times New Roman", "font_size": 10}
    return {
        "title": workbook.add_format(
            {**base, "font_size": 11, "bold": True, "align": "left", "bottom": 1}
        ),
        "header": workbook.add_format(
            {**base, "align": "center", "valign": "vcenter", "bottom": 1}
        ),
        "header_top": workbook.add_format(
            {**base, "align": "center", "valign": "vcenter", "top": 1, "bottom": 1}
        ),
        "group": workbook.add_format({**base, "bold": True, "top": 1}),
        "stub": workbook.add_format({**base, "indent": 1}),
        "cell": workbook.add_format({**base, "align": "center"}),
        "cell_left": workbook.add_format({**base, "align": "left"}),
        "cell_bottom": workbook.add_format({**base, "align": "center", "bottom": 1}),
        "footnote": workbook.add_format({**base, "font_size": 9, "text_wrap": True}),
        "number": workbook.add_format({**base, "align": "center", "num_format": "0.00"}),
        "integer": workbook.add_format({**base, "align": "center", "num_format": "0"}),
    }


def _format_n_percent(row: pd.Series) -> str:
    """Format a count and percentage for Table 1."""
    if not np.isfinite(row["percent"]):
        return "NA"
    return f"{int(row['n'])} ({row['percent']:.1f})"


def _citation_cell(row: pd.Series, panel: str) -> str:
    """Format one citation-summary cell for the requested panel."""
    if int(row["eligible_n"]) == 0:
        return "NA"
    if panel == "mean":
        sd = "-" if not np.isfinite(row["sd"]) else f"{row['sd']:.2f}"
        return f"{row['mean']:.2f} ({sd})"
    if panel == "median":
        return f"{row['median']:.2f} [{row['q1']:.2f}, {row['q3']:.2f}]"
    if panel == "eligible":
        return f"{int(row['eligible_n'])}/{int(row['total_n'])}"
    if panel == "zero":
        return f"{int(row['zero_n'])} ({row['zero_pct']:.1f})"
    raise ValueError(panel)


def _write_hierarchical_header(worksheet, row: int, formats: dict[str, object]) -> int:
    """Write the shared two-row category header and return the next row."""
    # NORA spans its aggregate plus the three mutually exclusive component categories.
    worksheet.merge_range(row, 0, row + 1, 0, "Publication cohort", formats["header_top"])
    worksheet.merge_range(row, 1, row + 1, 1, "Overall", formats["header_top"])
    worksheet.merge_range(row, 2, row + 1, 2, "ORA", formats["header_top"])
    worksheet.merge_range(row, 3, row, 6, "NORA", formats["header_top"])
    worksheet.merge_range(row, 7, row + 1, 7, "Non-citable items", formats["header_top"])
    for offset, label in enumerate(["Overall", "Meta-analysis", "Review", "GSS"], start=3):
        worksheet.write(row + 1, offset, label, formats["header"])
    return row + 2


def write_table_1(composition: pd.DataFrame, path: Path) -> None:
    """Write publication composition by journal and cohort."""
    title = "Table 1. Publication composition by journal and June-May publication cohort"
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Table 1")
        writer.sheets["Table 1"] = worksheet
        formats = _excel_formats(workbook)
        worksheet.merge_range(0, 0, 0, 7, title, formats["title"])
        row = _write_hierarchical_header(worksheet, 2, formats)
        # Fixed journal/cohort/category orders make successive workbook versions comparable.
        for journal in JOURNAL_ORDER:
            worksheet.write(row, 0, JOURNAL_DISPLAY[journal], formats["group"])
            for col in range(1, 8):
                worksheet.write_blank(row, col, None, formats["group"])
            row += 1
            journal_data = composition[composition["journal_code"].eq(journal)]
            for cohort_label, _, _, _ in COHORTS:
                worksheet.write(row, 0, cohort_label, formats["stub"])
                subset = journal_data[journal_data["cohort"].eq(cohort_label)].set_index("category")
                for col, category in enumerate(CATEGORY_ORDER, start=1):
                    worksheet.write(
                        row,
                        col,
                        _format_n_percent(subset.loc[category]),
                        formats["cell"],
                    )
                row += 1
        worksheet.set_column(0, 0, 34)
        worksheet.set_column(1, 7, 19)
        worksheet.set_row(2, 20)
        row += 1
        footnotes = [
            (
                "Data are n (%). Percentages use all items in the same journal and publication "
                "cohort as the denominator."
            ),
            "NORA includes meta-analysis, review, and guideline/statement/statistics (GSS).",
            (
                "*The latest descriptive cohort spans 13 months (June 2025 through June 2026) "
                "and is not used as a like-for-like annual comparison."
            ),
            (
                "Abbreviation: CIRC, Circulation; EHJ, European Heart Journal; JACC, Journal of "
                "the American College of Cardiology; ORA, original research article; NORA, "
                "non-original research article; GSS, guideline, statement, and statistics."
            ),
        ]
        for note in footnotes:
            worksheet.merge_range(row, 0, row, 7, note, formats["footnote"])
            row += 1
        worksheet.hide_gridlines(2)
        worksheet.freeze_panes(4, 1)


def write_table_2(citations: pd.DataFrame, path: Path) -> None:
    """Write observed 1-year and latest cumulative citation summaries."""
    title = (
        "Table 2. Observed 1-year and latest total cumulative citation count by journal and "
        "publication cohort"
    )
    panels = [
        ("Panel A. Mean (SD)", "mean"),
        ("Panel B. Median [IQR]", "median"),
        ("Panel C. Eligible/total articles", "eligible"),
        ("Panel D. Zero citations, n (%)", "zero"),
    ]
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Table 2")
        writer.sheets["Table 2"] = worksheet
        formats = _excel_formats(workbook)
        worksheet.merge_range(0, 0, 0, 7, title, formats["title"])
        row = 2
        # Stack summary definitions vertically so all panels retain one category hierarchy.
        for panel_title, panel_key in panels:
            worksheet.merge_range(row, 0, row, 7, panel_title, formats["group"])
            row = _write_hierarchical_header(worksheet, row + 1, formats)
            for journal in JOURNAL_ORDER:
                worksheet.write(row, 0, JOURNAL_DISPLAY[journal], formats["group"])
                for col in range(1, 8):
                    worksheet.write_blank(row, col, None, formats["group"])
                row += 1
                journal_data = citations[citations["journal_code"].eq(journal)]
                for cohort_label, _, _, _ in COHORTS:
                    subset = journal_data[journal_data["cohort"].eq(cohort_label)].set_index(
                        "category"
                    )
                    worksheet.write(
                        row,
                        0,
                        str(subset.loc["Overall", "display_cohort"]),
                        formats["stub"],
                    )
                    for col, category in enumerate(CATEGORY_ORDER, start=1):
                        worksheet.write(
                            row,
                            col,
                            _citation_cell(subset.loc[category], panel_key),
                            formats["cell"],
                        )
                    row += 1
            row += 2
        worksheet.set_column(0, 0, 21)
        worksheet.set_column(1, 7, 20)
        footnotes = [
            (
                "A 1-year citation count is observed only when the full 12-month follow-up is "
                "complete at the analysis cutoff (July 24, 2026)."
            ),
            (
                "For the June 2025-June 2026 cohort, values labeled total cumulative are all "
                "citations accrued by July 24, 2026; follow-up duration varies by publication "
                "date and these values are not directly comparable with 1-year citation counts."
            ),
            "NA denotes an item without a matching Scopus paper record, not zero citations.",
            (
                "NORA includes meta-analysis, review, and GSS. Citation counts are based on "
                "distinct citing Scopus EIDs with nonnegative citation lag."
            ),
        ]
        for note in footnotes:
            worksheet.merge_range(row, 0, row, 7, note, formats["footnote"])
            row += 1
        worksheet.hide_gridlines(2)


def write_flat_table(
    data: pd.DataFrame,
    path: Path,
    title: str,
    sheet_name: str,
    footnotes: Sequence[str] = (),
) -> None:
    """Write a flat dataframe with consistent manuscript formatting."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name[:31])
        writer.sheets[sheet_name[:31]] = worksheet
        formats = _excel_formats(workbook)
        last_col = max(len(data.columns) - 1, 0)
        worksheet.merge_range(0, 0, 0, last_col, title, formats["title"])
        for col, column in enumerate(data.columns):
            worksheet.write(2, col, str(column), formats["header_top"])
        for row_idx, values in enumerate(data.itertuples(index=False, name=None), start=3):
            for col_idx, value in enumerate(values):
                # Preserve numeric cells for downstream use while representing missingness explicitly.
                if pd.isna(value):
                    worksheet.write(row_idx, col_idx, "NA", formats["cell"])
                elif isinstance(value, int | np.integer):
                    worksheet.write_number(row_idx, col_idx, int(value), formats["integer"])
                elif isinstance(value, float | np.floating):
                    worksheet.write_number(row_idx, col_idx, float(value), formats["number"])
                else:
                    worksheet.write(row_idx, col_idx, str(value), formats["cell_left"])
        worksheet.autofilter(2, 0, max(2, len(data) + 2), last_col)
        worksheet.freeze_panes(3, 0)
        worksheet.set_column(0, last_col, 18)
        row = len(data) + 4
        for note in footnotes:
            worksheet.merge_range(row, 0, row, last_col, note, formats["footnote"])
            row += 1
        worksheet.hide_gridlines(2)


def write_table_5(forecast_summary: pd.DataFrame, path: Path) -> None:
    """Write predicted cumulative citation counts after the landmark."""
    output = forecast_summary.copy()
    # Assemble manuscript-ready compound cells without altering the tidy source columns.
    output["Predicted mean (SD)"] = output.apply(
        lambda row: f"{row['predicted_mean']:.2f} ({row['predicted_sd']:.2f})"
        if np.isfinite(row["predicted_mean"]) and np.isfinite(row["predicted_sd"])
        else "NA",
        axis=1,
    )
    output["Mean 80% bounds"] = output.apply(
        lambda row: f"{row['mean_lower_80_bound']:.2f}, {row['mean_upper_80_bound']:.2f}"
        if np.isfinite(row["mean_lower_80_bound"])
        else "NA",
        axis=1,
    )
    output["Observed mean (complete n)"] = output.apply(
        lambda row: f"{row['observed_mean_if_complete']:.2f} ({int(row['observed_complete_n'])})"
        if np.isfinite(row["observed_mean_if_complete"])
        else f"NA (0/{int(row['articles'])})",
        axis=1,
    )
    display = output[
        [
            "journal",
            "publication_cohort",
            "category",
            "target",
            "articles",
            "Predicted mean (SD)",
            "Mean 80% bounds",
            "Observed mean (complete n)",
        ]
    ].rename(
        columns={
            "journal": "Journal",
            "publication_cohort": "Publication cohort",
            "category": "Category",
            "target": "Citation target",
            "articles": "Articles",
        }
    )
    write_flat_table(
        display,
        path,
        "Table 5. Predicted cumulative citation counts for the first post-intervention cohort",
        "Table 5",
        footnotes=[
            (
                "Predictions are generated from the observed 1-year citation count and article "
                "metadata after final model refitting through May 2024 publications."
            ),
            (
                "Mean 80% bounds summarize article-level empirical residual bounds and are not "
                "confidence intervals for the journal-level mean."
            ),
            (
                "Observed values are displayed only where the corresponding follow-up horizon "
                "is complete at the analysis cutoff."
            ),
        ],
    )


def write_combined_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    """Write a collection of tidy result tables to one workbook."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        formats = _excel_formats(workbook)
        for sheet_name, frame in tables.items():
            # Excel limits worksheet names to 31 characters; preserve insertion order across sheets.
            safe_name = sheet_name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False, startrow=2)
            worksheet = writer.sheets[safe_name]
            last_col = max(len(frame.columns) - 1, 0)
            # Rewrite pandas' header row with the same manuscript style used by standalone tables.
            worksheet.merge_range(0, 0, 0, last_col, sheet_name, formats["title"])
            for col, column in enumerate(frame.columns):
                worksheet.write(2, col, str(column), formats["header_top"])
            worksheet.freeze_panes(3, 0)
            worksheet.autofilter(2, 0, max(2, len(frame) + 2), last_col)
            worksheet.set_column(0, last_col, 18)
            worksheet.hide_gridlines(2)
