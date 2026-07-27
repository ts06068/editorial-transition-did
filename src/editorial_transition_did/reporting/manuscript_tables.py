from __future__ import annotations

import pandas as pd

from ..config import (
    CATEGORY_ORDER,
    COHORTS,
    HORIZONS,
    JOURNAL_DISPLAY,
    JOURNAL_ORDER,
    AnalysisPaths,
)
from ..utils import format_p_value
from .excel import _citation_cell, _format_n_percent
from .manuscript_components import (
    _category_word_header,
    _did_word_blocks,
    _forecast_word_block,
    _period_citation_word_block,
    _period_composition_word_block,
    _word_estimate_ci,
    _word_mean_sd,
    _word_number,
    _word_percent,
)
from .word import (
    ManuscriptWordBlock,
    ManuscriptWordRow,
    ManuscriptWordTable,
    _manuscript_word_text,
)


def build_manuscript_word_table_specs(
    composition: pd.DataFrame,
    citations: pd.DataFrame,
    primary_did: pd.DataFrame,
    metrics: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    period_summaries: dict[str, pd.DataFrame],
    robustness_raw: pd.DataFrame,
    robustness_log: pd.DataFrame,
    placebo: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    source_qa: pd.DataFrame,
    publisher_only: pd.DataFrame,
    validation: pd.DataFrame,
    paths: AnalysisPaths,
) -> tuple[list[ManuscriptWordTable], list[ManuscriptWordTable]]:
    # Declare table content here; keep statistics out of the OOXML renderer.
    category_headers, category_merges, category_underlines = _category_word_header()

    # Atomic counts beside totals make denominator errors visible.
    table_1_rows: list[ManuscriptWordRow] = []
    for journal in JOURNAL_ORDER:
        table_1_rows.append(
            ManuscriptWordRow((JOURNAL_DISPLAY[journal], "", "", "", "", "", "", ""), kind="group")
        )
        journal_data = composition[composition["journal_code"].eq(journal)]
        for cohort_label, _, _, _ in COHORTS:
            subset = journal_data[journal_data["cohort"].eq(cohort_label)].set_index("category")
            table_1_rows.append(
                ManuscriptWordRow(
                    (cohort_label,)
                    + tuple(_format_n_percent(subset.loc[category]) for category in CATEGORY_ORDER)
                )
            )
    table_1 = ManuscriptWordTable(
        output_path=paths.main_table_dir / "Table_1_publication_composition.docx",
        title="Table 1. Publication composition by journal and June-May publication cohort",
        blocks=[
            ManuscriptWordBlock(
                headers=category_headers,
                rows=table_1_rows,
                header_merges=category_merges,
                header_underlines=category_underlines,
                column_widths=[1.45, 1.0, 0.9, 1.0, 1.25, 1.0, 0.85, 1.55],
                alignments=["center"] * 8,
                font_size=9.5,
            )
        ],
        footnotes=[
            "Data are n (%). Percentages use all items in the same journal and publication cohort as the denominator.",
            "NORA includes meta-analysis, review, and guideline/statement/statistics (GSS).",
            "*The latest descriptive cohort spans 13 months (June 2025 through June 2026) and is not used as a like-for-like annual comparison.",
            "Abbreviation: CIRC, Circulation; EHJ, European Heart Journal; JACC, Journal of the American College of Cardiology; ORA, original research article; NORA, non-original research article; GSS, guideline, statement, and statistics.",
        ],
    )

    # Reuse one row hierarchy so all four panels stay aligned by category.
    table_2_blocks: list[ManuscriptWordBlock] = []
    for panel_title, panel_key in [
        ("Panel A. Mean (SD)", "mean"),
        ("Panel B. Median [IQR]", "median"),
        ("Panel C. Eligible/total articles", "eligible"),
        ("Panel D. Zero citations, n (%)", "zero"),
    ]:
        rows: list[ManuscriptWordRow] = []
        for journal in JOURNAL_ORDER:
            rows.append(
                ManuscriptWordRow(
                    (JOURNAL_DISPLAY[journal], "", "", "", "", "", "", ""), kind="group"
                )
            )
            journal_data = citations[citations["journal_code"].eq(journal)]
            for cohort_label, _, _, _ in COHORTS:
                subset = journal_data[journal_data["cohort"].eq(cohort_label)].set_index("category")
                rows.append(
                    ManuscriptWordRow(
                        (str(subset.loc["Overall", "display_cohort"]),)
                        + tuple(
                            _citation_cell(subset.loc[category], panel_key)
                            for category in CATEGORY_ORDER
                        )
                    )
                )
        table_2_blocks.append(
            ManuscriptWordBlock(
                panel_title=panel_title,
                headers=category_headers,
                rows=rows,
                header_merges=category_merges,
                header_underlines=category_underlines,
                column_widths=[1.55, 0.95, 0.85, 0.95, 1.2, 0.95, 0.8, 1.75],
                alignments=["center"] * 8,
                font_size=9.0,
            )
        )
    table_2 = ManuscriptWordTable(
        output_path=paths.main_table_dir / "Table_2_observed_1year_citations.docx",
        title="Table 2. Observed 1-year and latest total cumulative citation count by journal and publication cohort",
        blocks=table_2_blocks,
        footnotes=[
            "A 1-year citation count is observed only when the full 12-month follow-up is complete at the analysis cutoff (July 24, 2026).",
            "For the June 2025-June 2026 cohort, values labeled total cumulative are citations accrued by July 24, 2026; follow-up duration varies by publication date.",
            "NA denotes an item without a matching Scopus paper record, not zero citations. NORA includes meta-analysis, review, and GSS.",
        ],
    )

    # Use the same source-cell layout for primary and sensitivity estimates.
    table_3 = ManuscriptWordTable(
        output_path=paths.main_table_dir / "Table_3_primary_difference_in_differences.docx",
        title="Table 3. Primary difference-in-differences analysis of observed 1-year citations",
        blocks=_did_word_blocks(primary_did, include_specification=False),
        footnotes=[
            "Values in Panel A are mean (SD); n. The pre period is July 2023-June 2024 and the post period is July 2024-June 2025.",
            "The control change is the equally weighted mean of the CIRC and EHJ changes. Confidence intervals use 2,000 publication-month block-bootstrap repetitions with each month resampled jointly across journals.",
            "Abbreviation: DiD, difference-in-differences; CI, confidence interval.",
        ],
    )

    # Accuracy and calibration use the same rows but answer different questions.
    table_4_a_rows: list[ManuscriptWordRow] = []
    table_4_b_rows: list[ManuscriptWordRow] = []
    for _, row in metrics.iterrows():
        table_4_a_rows.append(
            ManuscriptWordRow(
                (
                    row["horizon"],
                    row["model"],
                    "Yes" if row["selected_by_rolling_mae"] else "No",
                    int(row["n"]),
                    _word_number(row["mae"]),
                    _word_number(row["rmse"]),
                    _word_number(row["median_absolute_error"]),
                    _word_number(row["r2"]),
                    _word_number(row["mean_bias"]),
                )
            )
        )
        table_4_b_rows.append(
            ManuscriptWordRow(
                (
                    row["horizon"],
                    row["model"],
                    _word_number(row["calibration_intercept"]),
                    _word_number(row["calibration_slope"]),
                    _word_percent(row["coverage_80"]),
                    _word_percent(row["coverage_95"]),
                )
            )
        )
    table_4 = ManuscriptWordTable(
        output_path=paths.main_table_dir / "Table_4_temporal_prediction_performance.docx",
        title="Table 4. Temporal test performance for citation increment prediction",
        blocks=[
            ManuscriptWordBlock(
                panel_title="Panel A. Prediction error and explained variation",
                headers=[
                    [
                        "Horizon",
                        "Model",
                        "Selected",
                        "Test n",
                        "MAE",
                        "RMSE",
                        "Median AE",
                        "R-squared",
                        "Mean bias",
                    ]
                ],
                rows=table_4_a_rows,
                column_widths=[0.85, 1.75, 0.7, 0.65, 0.7, 0.7, 0.85, 0.85, 0.95],
                alignments=["center", "left"] + ["center"] * 7,
                font_size=9.0,
            ),
            ManuscriptWordBlock(
                panel_title="Panel B. Calibration and empirical interval coverage",
                headers=[
                    [
                        "Horizon",
                        "Model",
                        "Calibration intercept",
                        "Calibration slope",
                        "80% coverage",
                        "95% coverage",
                    ]
                ],
                rows=table_4_b_rows,
                column_widths=[1.0, 2.4, 1.45, 1.4, 1.35, 1.4],
                alignments=["center", "left", "center", "center", "center", "center"],
            ),
        ],
        footnotes=[
            "The development period ends May 2022 and the temporal test period spans June 2022-May 2024.",
            "Prediction targets are citation increments during months 13-15, 13-18, and 13-24 after publication.",
            "The forecasting model at each horizon is selected by rolling temporal-validation MAE.",
            "Abbreviation: MAE, mean absolute error; RMSE, root mean squared error; AE, absolute error.",
        ],
    )

    # Main and supplemental forecasts require identical caveats.
    forecast_footnotes = [
        "Predictions use observed 1-year citation counts and article metadata after final model refitting through May 2024 publications.",
        "Mean 80% bounds summarize article-level empirical residual bounds and are not confidence intervals for the journal-level mean.",
        "Observed values are shown only where the corresponding follow-up horizon is complete at the analysis cutoff.",
    ]
    table_5 = ManuscriptWordTable(
        output_path=paths.main_table_dir / "Table_5_post_cohort_forecasts.docx",
        title="Table 5. Predicted cumulative citation counts for the first post-intervention cohort",
        blocks=[_forecast_word_block(forecast_summary)],
        footnotes=forecast_footnotes,
    )
    main_specs = [table_1, table_2, table_3, table_4, table_5]

    # Order supplements from descriptive results through provenance checks.
    supplemental_specs: list[ManuscriptWordTable] = []
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_1_quarterly_composition.docx",
            "Supplemental Table 1. Quarterly publication composition",
            [_period_composition_word_block(period_summaries["quarter_composition"])],
            [
                "Data are n (%); percentages use all items in the same journal-quarter as the denominator."
            ],
        )
    )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_2_monthly_composition.docx",
            "Supplemental Table 2. Monthly publication composition",
            [_period_composition_word_block(period_summaries["month_composition"])],
            [
                "Data are n (%); percentages use all items in the same journal-month as the denominator."
            ],
        )
    )
    citation_note = [
        "Only articles with complete 12-month follow-up at the July 24, 2026 analysis cutoff contribute to observed 1-year citation summaries.",
        "NA denotes no eligible matched Scopus record, not zero citations.",
    ]
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_3_quarterly_1year_citations.docx",
            "Supplemental Table 3. Quarterly observed 1-year citation summaries",
            [_period_citation_word_block(period_summaries["quarter_citations"])],
            citation_note,
        )
    )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_4_monthly_1year_citations.docx",
            "Supplemental Table 4. Monthly observed 1-year citation summaries",
            [_period_citation_word_block(period_summaries["month_citations"])],
            citation_note,
        )
    )

    # Retain raw journal cells so alternative controls remain auditable.
    raw_blocks = _did_word_blocks(robustness_raw, include_specification=True)
    for panel_letter, journal in zip(["C", "D", "E"], ["circ", "ehj", "jacc"], strict=False):
        journal_rows: list[ManuscriptWordRow] = []
        for _, row in robustness_raw.iterrows():
            journal_rows.append(
                ManuscriptWordRow(
                    (
                        str(row["display_label"]),
                        _word_mean_sd(
                            row[f"{journal}_pre_mean"],
                            row[f"{journal}_pre_sd"],
                            row[f"{journal}_pre_n"],
                        ),
                        _word_mean_sd(
                            row[f"{journal}_post_mean"],
                            row[f"{journal}_post_sd"],
                            row[f"{journal}_post_n"],
                        ),
                        _word_number(row[f"{journal}_change"]),
                    )
                )
            )
        raw_blocks.append(
            ManuscriptWordBlock(
                panel_title=f"Panel {panel_letter}. {JOURNAL_DISPLAY[journal]} cell summaries",
                headers=[["Specification", "Pre mean (SD); n", "Post mean (SD); n", "Change"]],
                rows=journal_rows,
                column_widths=[4.0, 1.9, 1.9, 1.2],
                alignments=["left", "center", "center", "center"],
                font_size=9.5,
            )
        )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_5A_raw_robustness.docx",
            "Supplemental Table 5A. Raw-scale robustness analyses",
            raw_blocks,
            [
                "DiD estimates compare JACC with the specified control journal(s). Confidence intervals use publication-month block bootstrap with each month resampled jointly across journals.",
                "The primary cutoff is July 2024. June and August are alternative cutoff definitions; the August analysis uses 11 complete months per period because July 2025 has incomplete 12-month follow-up at the analysis cutoff.",
                "The transition washout excludes June-August 2024. Article-type and leave-one-control-out analyses are prespecified sensitivity analyses.",
            ],
        )
    )

    # Report the fitted log coefficient and its exponentiated interpretation.
    log_rows: list[ManuscriptWordRow] = []
    for _, row in robustness_log.iterrows():
        log_rows.append(
            ManuscriptWordRow(
                (
                    str(row["display_label"]),
                    str(row["category"]),
                    str(row["controls"]),
                    int(row["n"]),
                    _word_estimate_ci(
                        row["log_coefficient"], row["log_ci_low"], row["log_ci_high"]
                    ),
                    (
                        f"{row['percent_change']:.1f}% "
                        f"({row['percent_ci_low']:.1f}%, {row['percent_ci_high']:.1f}%)"
                    ),
                    format_p_value(float(row["p_value"])),
                    str(row["covariance"]),
                )
            )
        )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_5B_log_robustness.docx",
            "Supplemental Table 5B. Log-scale robustness analyses",
            [
                ManuscriptWordBlock(
                    headers=[
                        [
                            "Specification",
                            "Category",
                            "Controls",
                            "N",
                            "Log coefficient (95% CI)",
                            "Relative change (95% CI)",
                            "P value",
                            "Covariance",
                        ]
                    ],
                    rows=log_rows,
                    column_widths=[2.1, 0.8, 0.8, 0.55, 1.45, 1.55, 0.65, 1.1],
                    alignments=[
                        "left",
                        "left",
                        "center",
                        "center",
                        "center",
                        "center",
                        "center",
                        "center",
                    ],
                    font_size=8.5,
                )
            ],
            [
                "Models use ordinary least squares of log(1 + 1-year citations) with journal and publication-month fixed effects and publication-month-clustered standard errors."
            ],
        )
    )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_6_placebo_analysis.docx",
            "Supplemental Table 6. Pre-intervention placebo difference-in-differences analysis",
            _did_word_blocks(placebo, include_specification=False),
            [
                "The placebo intervention date is July 2023; all periods precede the actual editorial-board change."
            ],
        )
    )

    # Group folds by horizon so each model choice remains traceable.
    fold_a_rows: list[ManuscriptWordRow] = []
    fold_b_rows: list[ManuscriptWordRow] = []
    for horizon in HORIZONS:
        fold_a_rows.append(ManuscriptWordRow((horizon,) + ("",) * 9, kind="group"))
        fold_b_rows.append(ManuscriptWordRow((horizon,) + ("",) * 6, kind="group"))
        subset = fold_metrics[fold_metrics["horizon"].eq(horizon)].sort_values(["fold", "model"])
        for _, row in subset.iterrows():
            validation_period = f"{pd.Timestamp(row['validation_start']):%Y-%m} to {pd.Timestamp(row['validation_end']):%Y-%m}"
            fold_a_rows.append(
                ManuscriptWordRow(
                    (
                        f"Fold {int(row['fold'])}",
                        f"{pd.Timestamp(row['training_end']):%Y-%m}",
                        validation_period,
                        str(row["model"]),
                        int(row["n"]),
                        _word_number(row["mae"]),
                        _word_number(row["rmse"]),
                        _word_number(row["median_absolute_error"]),
                        _word_number(row["r2"]),
                        _word_number(row["mean_bias"]),
                    )
                )
            )
            fold_b_rows.append(
                ManuscriptWordRow(
                    (
                        f"Fold {int(row['fold'])}",
                        str(row["model"]),
                        _word_number(row["calibration_intercept"]),
                        _word_number(row["calibration_slope"]),
                        _word_percent(row["coverage_80"]),
                        _word_percent(row["coverage_95"]),
                        validation_period,
                    )
                )
            )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_7_temporal_validation_folds.docx",
            "Supplemental Table 7. Rolling temporal validation performance",
            [
                ManuscriptWordBlock(
                    panel_title="Panel A. Prediction error by temporal-validation fold",
                    headers=[
                        [
                            "Horizon/fold",
                            "Training through",
                            "Validation period",
                            "Model",
                            "N",
                            "MAE",
                            "RMSE",
                            "Median AE",
                            "R-squared",
                            "Mean bias",
                        ]
                    ],
                    rows=fold_a_rows,
                    column_widths=[0.75, 0.8, 1.15, 1.45, 0.45, 0.65, 0.65, 0.75, 0.75, 0.7],
                    alignments=["center", "center", "center", "left"] + ["center"] * 6,
                    font_size=8.0,
                ),
                ManuscriptWordBlock(
                    panel_title="Panel B. Calibration and interval coverage by fold",
                    headers=[
                        [
                            "Horizon/fold",
                            "Model",
                            "Calibration intercept",
                            "Calibration slope",
                            "80% coverage",
                            "95% coverage",
                            "Validation period",
                        ]
                    ],
                    rows=fold_b_rows,
                    column_widths=[1.0, 1.8, 1.25, 1.2, 1.05, 1.05, 1.65],
                    alignments=["center", "left", "center", "center", "center", "center", "center"],
                    font_size=9.0,
                ),
            ],
            ["Each validation fold occurs strictly after its corresponding training period."],
        )
    )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_8_forecast_details.docx",
            "Supplemental Table 8. Detailed post-cohort citation forecasts",
            [_forecast_word_block(forecast_summary)],
            forecast_footnotes,
        )
    )

    qa_rows = [
        ManuscriptWordRow((str(row["check"]), _manuscript_word_text(row["value"])))
        for _, row in source_qa.iterrows()
    ]
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_9_data_quality.docx",
            "Supplemental Table 9. Data quality checks before analysis",
            [
                ManuscriptWordBlock(
                    headers=[["Data quality check", "Value"]],
                    rows=qa_rows,
                    column_widths=[6.5, 2.5],
                    alignments=["left", "center"],
                )
            ],
        )
    )

    # Preserve unmatched publisher records for provenance, not citation analysis.
    publisher_rows: list[ManuscriptWordRow] = []
    for journal in [JOURNAL_DISPLAY[key] for key in JOURNAL_ORDER]:
        publisher_rows.append(ManuscriptWordRow((journal, "", "", "", "", ""), kind="group"))
        subset = publisher_only[publisher_only["journal_display"].eq(journal)].sort_values(
            "publication_date"
        )
        for _, row in subset.iterrows():
            publisher_rows.append(
                ManuscriptWordRow(
                    (
                        pd.Timestamp(row["publication_date"]).strftime("%Y-%m-%d"),
                        row["doi_norm"] if pd.notna(row["doi_norm"]) else "NA",
                        row["title"],
                        row["category"],
                        row["issue_url"],
                        row["publication_date_source"],
                    )
                )
            )
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_10_publisher_only_items.docx",
            "Supplemental Table 10. Publisher-listed items without a matching Scopus paper record",
            [
                ManuscriptWordBlock(
                    headers=[
                        ["Publication date", "DOI", "Title", "Category", "Issue URL", "Date source"]
                    ],
                    rows=publisher_rows,
                    column_widths=[0.85, 1.3, 2.75, 0.95, 2.25, 0.9],
                    alignments=["center", "left", "left", "center", "left", "center"],
                    font_size=8.0,
                )
            ],
            [
                "These items contribute to composition counts but are excluded from citation, DiD, and prediction analyses because a missing Scopus record is not equivalent to zero citations."
            ],
        )
    )

    # Publish the invariants checked for this run.
    validation_rows = [
        ManuscriptWordRow(
            (
                str(row["check"]),
                "Passed" if bool(row["passed"]) else "Failed",
                str(row["detail"]),
            )
        )
        for _, row in validation.iterrows()
    ]
    supplemental_specs.append(
        ManuscriptWordTable(
            paths.supp_table_dir / "Supplemental_Table_11_output_validation.docx",
            "Supplemental Table 11. Automated output validation",
            [
                ManuscriptWordBlock(
                    headers=[["Validation check", "Result", "Detail"]],
                    rows=validation_rows,
                    column_widths=[4.2, 1.0, 3.8],
                    alignments=["left", "center", "left"],
                    font_size=9.5,
                )
            ],
        )
    )
    return main_specs, supplemental_specs
