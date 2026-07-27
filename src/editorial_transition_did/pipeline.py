from __future__ import annotations

import json

from tqdm.auto import tqdm

from .config import (
    ANALYSIS_CUTOFF,
    PUBLICATION_END,
    RECENT_START,
    AnalysisPaths,
)
from .data import prepare_analysis_data
from .did import format_table_3, run_placebo_did, run_primary_did, run_robustness
from .prediction import aggregate_forecasts, fit_prediction_models, format_table_4
from .reporting.descriptive_figures import plot_figure_1, plot_supplemental_composition
from .reporting.did_figures import plot_figure_2, plot_figure_3, plot_pretrend_difference
from .reporting.excel import (
    write_combined_workbook,
    write_flat_table,
    write_table_1,
    write_table_2,
    write_table_5,
)
from .reporting.manuscript_tables import build_manuscript_word_table_specs
from .reporting.plotting import configure_plot_style
from .reporting.prediction_figures import plot_figure_4, plot_figure_5
from .reporting.text import write_figure_legends, write_results_readme
from .reporting.word import write_manuscript_word_tables
from .summaries import build_citation_summary, build_composition_table, build_period_summaries
from .validation import build_manifest, validate_results


def run_all(
    paths: AnalysisPaths | None = None,
    force_refresh_db: bool = False,
    primary_bootstrap_repetitions: int = 2000,
    robustness_bootstrap_repetitions: int = 1500,
) -> dict[str, object]:
    paths = paths or AnalysisPaths.from_environment()
    paths.create_directories()
    configure_plot_style()

    stages = tqdm(total=8, desc="CLEAR-CARD analysis")
    data, source_qa = prepare_analysis_data(paths, force_refresh_db=force_refresh_db)
    stages.update(1)

    composition = build_composition_table(data)
    citations = build_citation_summary(data)
    period_summaries = build_period_summaries(data)
    write_table_1(composition, paths.main_table_dir / "Table_1_publication_composition.xlsx")
    write_table_2(citations, paths.main_table_dir / "Table_2_observed_1year_citations.xlsx")
    plot_figure_1(data, paths)
    plot_supplemental_composition(data, paths)
    stages.update(1)

    primary_did = run_primary_did(data, n_boot=primary_bootstrap_repetitions)
    formatted_table_3 = format_table_3(primary_did)
    write_flat_table(
        formatted_table_3,
        paths.main_table_dir / "Table_3_primary_difference_in_differences.xlsx",
        "Table 3. Primary difference-in-differences analysis of observed 1-year citations",
        "Table 3",
        footnotes=[
            "The primary post period is July 2024 through June 2025; the pre period is the preceding 12 months.",
            "The control change is the equally weighted mean of the CIRC and EHJ changes. Confidence intervals use 2,000 publication-month block-bootstrap repetitions with each month resampled jointly across journals.",
            "NA denotes a category with no articles in at least one journal-period cell, for which the prespecified equal-journal-weighted DiD is undefined.",
        ],
    )
    plot_figure_2(data, primary_did, paths)
    stages.update(1)

    robustness_raw, robustness_log = run_robustness(data, n_boot=robustness_bootstrap_repetitions)
    plot_figure_3(robustness_raw, robustness_log, paths)
    pretrend_monthly, pretrend_info = plot_pretrend_difference(data, paths)
    placebo = run_placebo_did(data, n_boot=robustness_bootstrap_repetitions)
    stages.update(1)

    metrics, fold_metrics, test_predictions, forecasts, final_models = fit_prediction_models(data)
    formatted_table_4 = format_table_4(metrics)
    write_flat_table(
        formatted_table_4,
        paths.main_table_dir / "Table_4_temporal_prediction_performance.xlsx",
        "Table 4. Temporal test performance for citation increment prediction",
        "Table 4",
        footnotes=[
            "The development period ends May 2022 and the nonrandom temporal test period spans June 2022 through May 2024.",
            "Prediction targets are incremental citations accrued during months 13-15, 13-18, and 13-24 after publication.",
            "The model used for forecasting at each horizon is selected by mean absolute error across rolling temporal validation folds before temporal-holdout evaluation.",
            "Prediction interval coverage is estimated from model-specific rolling temporal validation residuals.",
        ],
    )
    plot_figure_4(data, test_predictions, paths)
    stages.update(1)

    forecast_summary = aggregate_forecasts(forecasts)
    write_table_5(forecast_summary, paths.main_table_dir / "Table_5_post_cohort_forecasts.xlsx")
    forecast_monthly = plot_figure_5(forecasts, paths)
    stages.update(1)

    write_flat_table(
        period_summaries["quarter_composition"],
        paths.supp_table_dir / "Supplemental_Table_1_quarterly_composition.xlsx",
        "Supplemental Table 1. Quarterly publication composition",
        "ST1",
    )
    write_flat_table(
        period_summaries["month_composition"],
        paths.supp_table_dir / "Supplemental_Table_2_monthly_composition.xlsx",
        "Supplemental Table 2. Monthly publication composition",
        "ST2",
    )
    write_flat_table(
        period_summaries["quarter_citations"],
        paths.supp_table_dir / "Supplemental_Table_3_quarterly_1year_citations.xlsx",
        "Supplemental Table 3. Quarterly observed 1-year citation summaries",
        "ST3",
    )
    write_flat_table(
        period_summaries["month_citations"],
        paths.supp_table_dir / "Supplemental_Table_4_monthly_1year_citations.xlsx",
        "Supplemental Table 4. Monthly observed 1-year citation summaries",
        "ST4",
    )
    write_flat_table(
        robustness_raw,
        paths.supp_table_dir / "Supplemental_Table_5A_raw_robustness.xlsx",
        "Supplemental Table 5A. Raw-scale robustness analyses",
        "ST5A",
    )
    write_flat_table(
        robustness_log,
        paths.supp_table_dir / "Supplemental_Table_5B_log_robustness.xlsx",
        "Supplemental Table 5B. Log-scale robustness analyses",
        "ST5B",
    )
    write_flat_table(
        placebo,
        paths.supp_table_dir / "Supplemental_Table_6_placebo_analysis.xlsx",
        "Supplemental Table 6. Pre-intervention placebo difference-in-differences analysis",
        "ST6",
    )
    write_flat_table(
        fold_metrics,
        paths.supp_table_dir / "Supplemental_Table_7_temporal_validation_folds.xlsx",
        "Supplemental Table 7. Rolling temporal validation performance",
        "ST7",
    )
    write_flat_table(
        forecast_summary,
        paths.supp_table_dir / "Supplemental_Table_8_forecast_details.xlsx",
        "Supplemental Table 8. Detailed post-cohort citation forecasts",
        "ST8",
    )
    write_flat_table(
        source_qa,
        paths.supp_table_dir / "Supplemental_Table_9_data_quality.xlsx",
        "Supplemental Table 9. Data quality checks before analysis",
        "ST9",
    )
    publisher_only = data[data["db_paper_id"].isna()][
        [
            "journal_display",
            "publication_date",
            "doi_norm",
            "title",
            "category",
            "issue_url",
            "publication_date_source",
        ]
    ].copy()
    write_flat_table(
        publisher_only,
        paths.supp_table_dir / "Supplemental_Table_10_publisher_only_items.xlsx",
        "Supplemental Table 10. Publisher-listed items without a matching Scopus paper record",
        "ST10",
        footnotes=[
            "These items contribute to publication-composition counts but are excluded from citation, DiD, and prediction analyses because a missing Scopus record is not equivalent to zero citations.",
        ],
    )
    test_predictions.to_csv(
        paths.cache_dir / "temporal_test_predictions.csv.gz", index=False, compression="gzip"
    )
    forecasts.to_csv(
        paths.cache_dir / "article_level_forecasts.csv.gz", index=False, compression="gzip"
    )
    stages.update(1)

    validation = validate_results(data, composition, citations, primary_did, metrics)
    write_flat_table(
        validation,
        paths.supp_table_dir / "Supplemental_Table_11_output_validation.xlsx",
        "Supplemental Table 11. Automated output validation",
        "ST11",
    )
    combined_tables = {
        "Table 1 tidy": composition,
        "Table 2 tidy": citations,
        "Table 3 DiD": primary_did,
        "Table 4 prediction": metrics,
        "Table 5 forecasts": forecast_summary,
        "Robustness raw": robustness_raw,
        "Robustness log": robustness_log,
        "Placebo": placebo,
        "Pretrend monthly": pretrend_monthly,
        "Validation": validation,
    }
    write_combined_workbook(
        combined_tables, paths.output_dir / "CLEAR_CARD_editorial_analysis_tables.xlsx"
    )
    write_figure_legends(paths)
    write_results_readme(paths)
    main_word_specs, supplemental_word_specs = build_manuscript_word_table_specs(
        composition=composition,
        citations=citations,
        primary_did=primary_did,
        metrics=metrics,
        forecast_summary=forecast_summary,
        period_summaries=period_summaries,
        robustness_raw=robustness_raw,
        robustness_log=robustness_log,
        placebo=placebo,
        fold_metrics=fold_metrics,
        source_qa=source_qa,
        publisher_only=publisher_only,
        validation=validation,
        paths=paths,
    )
    word_tables = write_manuscript_word_tables(
        main_word_specs,
        supplemental_word_specs,
        paths,
    )

    summary = {
        "analysis_cutoff": ANALYSIS_CUTOFF.strftime("%Y-%m-%d"),
        "records": int(len(data)),
        "recent_records": int(
            data["publication_date"].between(RECENT_START, PUBLICATION_END).sum()
        ),
        "primary_overall_did": primary_did.loc[
            primary_did["category"].eq("Overall"),
            ["did", "ci_low", "ci_high", "p_value"],
        ]
        .iloc[0]
        .to_dict(),
        "pretrend": pretrend_info,
        "prediction_test_metrics": metrics.to_dict(orient="records"),
        "validation_passed": bool(validation["passed"].all()),
    }
    (paths.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    stages.update(1)
    stages.close()

    manifest = build_manifest(paths)
    write_flat_table(
        manifest,
        paths.output_dir / "output_manifest.xlsx",
        "Output manifest",
        "Manifest",
    )
    return {
        "data": data,
        "source_qa": source_qa,
        "composition": composition,
        "citations": citations,
        "primary_did": primary_did,
        "robustness_raw": robustness_raw,
        "robustness_log": robustness_log,
        "prediction_metrics": metrics,
        "fold_metrics": fold_metrics,
        "forecast_summary": forecast_summary,
        "forecast_monthly": forecast_monthly,
        "validation": validation,
        "manifest": manifest,
        "word_tables": word_tables,
        "models": final_models,
    }
