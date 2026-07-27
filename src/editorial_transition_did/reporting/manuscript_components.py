from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CATEGORY_ORDER, JOURNAL_DISPLAY, JOURNAL_ORDER
from ..utils import format_p_value
from .excel import _citation_cell
from .word import (
    ManuscriptWordBlock,
    ManuscriptWordRow,
)


def _word_number(value: object, digits: int = 2) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if not np.isfinite(numeric) else f"{float(numeric):.{digits}f}"


def _word_percent(value: object, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "NA" if not np.isfinite(numeric) else f"{float(numeric) * 100:.{digits}f}%"


def _word_mean_sd(mean: object, sd: object, n: object | None = None) -> str:
    mean_value = pd.to_numeric(pd.Series([mean]), errors="coerce").iloc[0]
    sd_value = pd.to_numeric(pd.Series([sd]), errors="coerce").iloc[0]
    if not np.isfinite(mean_value):
        return "NA"
    sd_text = "-" if not np.isfinite(sd_value) else f"{float(sd_value):.2f}"
    output = f"{float(mean_value):.2f} ({sd_text})"
    if n is not None and pd.notna(n):
        output += f"; n={int(n)}"
    return output


def _word_estimate_ci(estimate: object, low: object, high: object) -> str:
    values = pd.to_numeric(pd.Series([estimate, low, high]), errors="coerce")
    if not np.isfinite(values).all():
        return "NA"
    return f"{values.iloc[0]:.2f} ({values.iloc[1]:.2f}, {values.iloc[2]:.2f})"


def _category_word_header() -> tuple[
    list[list[str]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int]],
]:
    headers = [
        [
            "Publication cohort",
            "Overall",
            "ORA",
            "NORA",
            "",
            "",
            "",
            "Non-citable items",
        ],
        ["", "", "", "Overall", "Meta-analysis", "Review", "GSS", ""],
    ]
    merges = [
        (0, 0, 1, 0),
        (0, 1, 1, 1),
        (0, 2, 1, 2),
        (0, 3, 0, 6),
        (0, 7, 1, 7),
    ]
    return headers, merges, [(0, 3, 6)]


def _did_word_blocks(
    frame: pd.DataFrame,
    include_specification: bool,
) -> list[ManuscriptWordBlock]:
    panel_a_rows: list[ManuscriptWordRow] = []
    panel_b_rows: list[ManuscriptWordRow] = []
    for _, row in frame.iterrows():
        specification = str(row.get("display_label", row.get("analysis", "")))
        prefix = (specification,) if include_specification else ()
        panel_a_rows.append(
            ManuscriptWordRow(
                prefix
                + (
                    str(row["category"]),
                    _word_mean_sd(row["jacc_pre_mean"], row["jacc_pre_sd"], row["jacc_pre_n"]),
                    _word_mean_sd(row["jacc_post_mean"], row["jacc_post_sd"], row["jacc_post_n"]),
                    _word_mean_sd(row["circ_pre_mean"], row["circ_pre_sd"], row["circ_pre_n"]),
                    _word_mean_sd(row["circ_post_mean"], row["circ_post_sd"], row["circ_post_n"]),
                    _word_mean_sd(row["ehj_pre_mean"], row["ehj_pre_sd"], row["ehj_pre_n"]),
                    _word_mean_sd(row["ehj_post_mean"], row["ehj_post_sd"], row["ehj_post_n"]),
                )
            )
        )
        panel_b_rows.append(
            ManuscriptWordRow(
                prefix
                + (
                    str(row["category"]),
                    _word_number(row["jacc_change"]),
                    _word_number(row["circ_change"]),
                    _word_number(row["ehj_change"]),
                    _word_number(row["control_change"]),
                    _word_estimate_ci(row["did"], row["ci_low"], row["ci_high"]),
                    format_p_value(float(row["p_value"])),
                )
            )
        )
    first_header = ["Specification"] if include_specification else []
    first_width = [2.0] if include_specification else []
    first_alignment = ["left"] if include_specification else []
    return [
        ManuscriptWordBlock(
            panel_title="Panel A. Journal-specific pre- and post-period citation summaries",
            headers=[
                first_header
                + [
                    "Category",
                    "JACC pre",
                    "JACC post",
                    "CIRC pre",
                    "CIRC post",
                    "EHJ pre",
                    "EHJ post",
                ]
            ],
            rows=panel_a_rows,
            column_widths=first_width + [0.9, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15],
            alignments=first_alignment + ["left"] + ["center"] * 6,
            font_size=8.5 if include_specification else 9.0,
        ),
        ManuscriptWordBlock(
            panel_title="Panel B. Difference-in-differences effect estimates",
            headers=[
                first_header
                + [
                    "Category",
                    "JACC change",
                    "CIRC change",
                    "EHJ change",
                    "Control change",
                    "Raw DiD (95% CI)",
                    "P value",
                ]
            ],
            rows=panel_b_rows,
            column_widths=first_width + [0.9, 0.85, 0.85, 0.85, 1.0, 1.55, 0.7],
            alignments=first_alignment + ["left"] + ["center"] * 6,
            font_size=8.5 if include_specification else 9.5,
        ),
    ]


def _period_composition_word_block(data: pd.DataFrame) -> ManuscriptWordBlock:
    rows: list[ManuscriptWordRow] = []
    for journal in [JOURNAL_DISPLAY[key] for key in JOURNAL_ORDER]:
        rows.append(ManuscriptWordRow((journal, "", ""), kind="group"))
        journal_data = data[data["journal"].eq(journal)].copy()
        journal_data["category_order"] = journal_data["category"].map(
            {category: index for index, category in enumerate(CATEGORY_ORDER)}
        )
        journal_data = journal_data.sort_values(["period", "category_order"])
        for _, row in journal_data.iterrows():
            value = (
                f"{int(row['n'])} ({row['percent']:.1f})" if np.isfinite(row["percent"]) else "NA"
            )
            rows.append(ManuscriptWordRow((str(row["period"]), str(row["category"]), value)))
    return ManuscriptWordBlock(
        headers=[["Publication period", "Article category", "Articles, n (%)"]],
        rows=rows,
        column_widths=[2.1, 3.6, 3.3],
        alignments=["center", "left", "center"],
    )


def _period_citation_word_block(data: pd.DataFrame) -> ManuscriptWordBlock:
    rows: list[ManuscriptWordRow] = []
    for journal in [JOURNAL_DISPLAY[key] for key in JOURNAL_ORDER]:
        rows.append(ManuscriptWordRow((journal, "", "", "", "", ""), kind="group"))
        journal_data = data[data["journal"].eq(journal)].copy()
        journal_data["category_order"] = journal_data["category"].map(
            {category: index for index, category in enumerate(CATEGORY_ORDER)}
        )
        journal_data = journal_data.sort_values(["period", "category_order"])
        for _, row in journal_data.iterrows():
            rows.append(
                ManuscriptWordRow(
                    (
                        str(row["period"]),
                        str(row["category"]),
                        f"{int(row['eligible_n'])}/{int(row['total_n'])}",
                        _citation_cell(row, "mean"),
                        _citation_cell(row, "median"),
                        _citation_cell(row, "zero"),
                    )
                )
            )
    return ManuscriptWordBlock(
        headers=[
            [
                "Publication period",
                "Article category",
                "Eligible/total",
                "Mean (SD)",
                "Median [IQR]",
                "Zero citations, n (%)",
            ]
        ],
        rows=rows,
        column_widths=[1.35, 1.35, 1.1, 1.45, 2.0, 1.75],
        alignments=["center", "left", "center", "center", "center", "center"],
        font_size=9.5,
    )


def _forecast_word_block(forecast_summary: pd.DataFrame) -> ManuscriptWordBlock:
    rows: list[ManuscriptWordRow] = []
    category_order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    horizon_order = {"C15": 0, "C18": 1, "C24": 2}
    for journal_code in JOURNAL_ORDER:
        journal = JOURNAL_DISPLAY[journal_code]
        rows.append(ManuscriptWordRow((journal, "", "", "", "", "", ""), kind="group"))
        subset = forecast_summary[forecast_summary["journal"].eq(journal)].copy()
        subset["category_order"] = subset["category"].map(category_order)
        subset["horizon_order"] = subset["target"].map(horizon_order)
        subset = subset.sort_values(["category_order", "horizon_order"])
        for _, row in subset.iterrows():
            observed = (
                f"{row['observed_mean_if_complete']:.2f} ({int(row['observed_complete_n'])})"
                if np.isfinite(row["observed_mean_if_complete"])
                else f"NA (0/{int(row['articles'])})"
            )
            rows.append(
                ManuscriptWordRow(
                    (
                        str(row["publication_cohort"]),
                        str(row["category"]),
                        str(row["target"]),
                        int(row["articles"]),
                        _word_mean_sd(row["predicted_mean"], row["predicted_sd"]),
                        (
                            f"{row['mean_lower_80_bound']:.2f}, {row['mean_upper_80_bound']:.2f}"
                            if np.isfinite(row["mean_lower_80_bound"])
                            else "NA"
                        ),
                        observed,
                    )
                )
            )
    return ManuscriptWordBlock(
        headers=[
            [
                "Publication cohort",
                "Category",
                "Citation target",
                "Articles",
                "Predicted mean (SD)",
                "Mean 80% bounds",
                "Observed mean (complete n)",
            ]
        ],
        rows=rows,
        column_widths=[1.35, 1.2, 1.05, 0.7, 1.45, 1.25, 2.0],
        alignments=["center", "left", "center", "center", "center", "center", "center"],
        font_size=9.0,
    )
