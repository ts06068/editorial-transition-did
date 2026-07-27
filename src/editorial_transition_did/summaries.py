"""Period assignment and descriptive table construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_CUTOFF,
    CATEGORY_ORDER,
    COHORTS,
    JOURNAL_DISPLAY,
    JOURNAL_ORDER,
    PUBLICATION_END,
    RECENT_START,
)
from .utils import category_mask


def assign_recent_cohort(data: pd.DataFrame) -> pd.Series:
    """Assign the prespecified June-May descriptive publication cohorts."""

    output = pd.Series(pd.NA, index=data.index, dtype="string")
    # Cohort boundaries follow the study year, not the calendar year.
    for label, start, end, _ in COHORTS:
        output.loc[data["publication_date"].between(start, end, inclusive="both")] = label
    return output


def add_period_columns(
    data: pd.DataFrame,
    frequency: str,
    date_column: str = "publication_date",
) -> pd.DataFrame:
    """Add stable labels and boundaries for yearly, quarterly, or monthly summaries."""

    output = data.copy()
    dates = pd.to_datetime(output[date_column], errors="coerce")
    if frequency == "Yearly":
        # Annual output must reuse the prespecified June-May cohort boundaries.
        temporary = pd.DataFrame({"publication_date": dates}, index=output.index)
        output["period_label"] = assign_recent_cohort(temporary)
        metadata = {
            label: {
                "start": start,
                "end": end,
                "date": start + (end - start) / 2,
            }
            for label, start, end, _ in COHORTS
        }
        output["period_start"] = output["period_label"].map(
            {label: values["start"] for label, values in metadata.items()}
        )
        output["period_end"] = output["period_label"].map(
            {label: values["end"] for label, values in metadata.items()}
        )
        output["period_date"] = output["period_label"].map(
            {label: values["date"] for label, values in metadata.items()}
        )
    elif frequency == "Quarterly":
        periods = dates.dt.to_period("Q")
        output["period_label"] = periods.astype("string")
        output["period_start"] = periods.dt.start_time
        output["period_end"] = periods.dt.end_time.dt.normalize()
        # Anchor quarters at their middle month for stable plotting coordinates.
        output["period_date"] = periods.dt.start_time + pd.DateOffset(months=1)
    elif frequency == "Monthly":
        periods = dates.dt.to_period("M")
        output["period_label"] = periods.astype("string")
        output["period_start"] = periods.dt.start_time
        output["period_end"] = periods.dt.end_time.dt.normalize()
        output["period_date"] = periods.dt.start_time
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    output["frequency"] = frequency
    return output[output["period_label"].notna()].copy()


def complete_period_mask(data: pd.DataFrame, months: int) -> pd.Series:
    """Identify periods whose full follow-up finishes by the analysis cutoff."""

    # Start follow-up at period end so partial periods cannot pass completeness checks.
    return data["period_end"].map(
        lambda value: (
            value + pd.DateOffset(months=months) <= ANALYSIS_CUTOFF if pd.notna(value) else False
        )
    )


def _summary_values(values: pd.Series) -> dict[str, float | int]:
    # Missing outcomes stay outside the denominator; never coerce them to zero.
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "eligible_n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "zero_n": 0,
            "zero_pct": np.nan,
        }
    return {
        "eligible_n": int(clean.size),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if clean.size > 1 else np.nan,
        "median": float(clean.median()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
        "zero_n": int(clean.eq(0).sum()),
        "zero_pct": float(clean.eq(0).mean() * 100),
    }


def build_composition_table(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize article-category counts by journal and recent cohort."""

    recent = data[data["publication_date"].between(RECENT_START, PUBLICATION_END)].copy()
    recent["cohort"] = assign_recent_cohort(recent)
    rows: list[dict[str, object]] = []
    for journal in JOURNAL_ORDER:
        journal_data = recent[recent["journal_code"].eq(journal)]
        for cohort_label, _, _, _ in COHORTS:
            subset = journal_data[journal_data["cohort"].eq(cohort_label)]
            denominator = len(subset)
            for category in CATEGORY_ORDER:
                # Overall and NORA overlap atomic rows; their percentages do not sum.
                count = int(category_mask(subset, category).sum())
                rows.append(
                    {
                        "journal_code": journal,
                        "journal": JOURNAL_DISPLAY[journal],
                        "cohort": cohort_label,
                        "category": category,
                        "n": count,
                        "denominator": denominator,
                        "percent": 100.0 * count / denominator if denominator else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_citation_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed citations by journal, cohort, and article category."""

    recent = data[data["publication_date"].between(RECENT_START, PUBLICATION_END)].copy()
    recent["cohort"] = assign_recent_cohort(recent)
    cohort_complete = {label: complete for label, _, _, complete in COHORTS}
    rows: list[dict[str, object]] = []
    for journal in JOURNAL_ORDER:
        journal_data = recent[recent["journal_code"].eq(journal)]
        for cohort_label, _, _, _ in COHORTS:
            cohort_data = journal_data[journal_data["cohort"].eq(cohort_label)]
            for category in CATEGORY_ORDER:
                subset = cohort_data[category_mask(cohort_data, category)]
                is_complete = cohort_complete[cohort_label]
                if is_complete:
                    outcome = subset["c12_obs"]
                    outcome_type = "1-year citation"
                    display_cohort = cohort_label
                else:
                    # The incomplete cohort reports snapshot totals, never partial C12.
                    outcome = pd.to_numeric(subset["citations_at_snapshot"], errors="coerce").where(
                        subset["db_paper_id"].notna()
                    )
                    outcome_type = "total cumulative"
                    display_cohort = f"{cohort_label.rstrip('*')} (total cumulative)"
                values = _summary_values(outcome)
                rows.append(
                    {
                        "journal_code": journal,
                        "journal": JOURNAL_DISPLAY[journal],
                        "cohort": cohort_label,
                        "display_cohort": display_cohort,
                        "category": category,
                        "cohort_complete": is_complete,
                        "outcome_type": outcome_type,
                        "total_n": int(len(subset)),
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def build_period_summaries(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the monthly and quarterly composition and citation supplements."""

    recent = data[data["publication_date"].between(RECENT_START, PUBLICATION_END)].copy()
    recent["month"] = recent["publication_date"].dt.to_period("M").astype(str)
    recent["quarter"] = recent["publication_date"].dt.to_period("Q").astype(str)
    outputs: dict[str, pd.DataFrame] = {}
    for period_name in ["month", "quarter"]:
        composition_rows: list[dict[str, object]] = []
        citation_rows: list[dict[str, object]] = []
        for (journal, period), group in recent.groupby(
            ["journal_code", period_name], observed=True
        ):
            for category in CATEGORY_ORDER:
                subset = group[category_mask(group, category)]
                denominator = len(group)
                count = len(subset)
                composition_rows.append(
                    {
                        "journal": JOURNAL_DISPLAY[journal],
                        "period": period,
                        "category": category,
                        "n": count,
                        "denominator": denominator,
                        "percent": 100.0 * count / denominator if denominator else np.nan,
                    }
                )
                citation_rows.append(
                    {
                        "journal": JOURNAL_DISPLAY[journal],
                        "period": period,
                        "category": category,
                        "total_n": count,
                        # c12_obs already excludes incomplete follow-up.
                        **_summary_values(subset["c12_obs"]),
                    }
                )
        outputs[f"{period_name}_composition"] = pd.DataFrame(composition_rows)
        outputs[f"{period_name}_citations"] = pd.DataFrame(citation_rows)
    return outputs
