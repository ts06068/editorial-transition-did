from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ..config import (
    ATOMIC_CATEGORIES,
    FREQUENCY_ORDER,
    JOURNAL_COLORS,
    JOURNAL_DISPLAY,
    JOURNAL_ORDER,
    PRIMARY_INTERVENTION,
    PUBLICATION_END,
    RECENT_START,
    AnalysisPaths,
)
from ..summaries import add_period_columns, complete_period_mask
from ..utils import category_mask
from .plotting import configure_plot_style, format_period_axis, save_figure


def plot_figure_1(data: pd.DataFrame, paths: AnalysisPaths) -> pd.DataFrame:
    configure_plot_style()
    # Restrict every panel to the prespecified reporting era before assigning periods.
    recent = data[data["publication_date"].between(RECENT_START, PUBLICATION_END)].copy()
    # Column order is a manuscript contract: broad groups first, then the non-citable group.
    panel_categories = ["Overall", "ORA", "NORA", "Non-citable items"]
    rows: list[pd.DataFrame] = []
    for frequency in FREQUENCY_ORDER:
        periodized = add_period_columns(recent, frequency)
        # Exclude partially observed periods so low follow-up cannot appear as low citation impact.
        periodized = periodized[
            complete_period_mask(periodized, 12) & periodized["c12_obs"].notna()
        ].copy()
        for category in panel_categories:
            subset = periodized[category_mask(periodized, category)]
            grouped = (
                subset.groupby(
                    [
                        "journal_code",
                        "frequency",
                        "period_label",
                        "period_date",
                        "period_end",
                    ],
                    observed=True,
                )["c12_obs"]
                .agg(mean="mean", sd="std", n="size")
                .reset_index()
            )
            grouped["category_group"] = category
            rows.append(grouped)
    summary = pd.concat(rows, ignore_index=True)

    # Rows encode time resolution and columns encode article category throughout Figure 1.
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.2), sharex=False)
    panel_index = 0
    for row_index, frequency in enumerate(FREQUENCY_ORDER):
        for column_index, category in enumerate(panel_categories):
            axis = axes[row_index, column_index]
            panel = summary[
                summary["frequency"].eq(frequency) & summary["category_group"].eq(category)
            ]
            for journal in JOURNAL_ORDER:
                journal_data = panel[panel["journal_code"].eq(journal)].sort_values("period_date")
                axis.plot(
                    journal_data["period_date"],
                    journal_data["mean"],
                    color=JOURNAL_COLORS[journal],
                    linewidth=1.5,
                    marker="s",
                    markersize=3,
                    label=JOURNAL_DISPLAY[journal],
                )
            # The transition band and intervention line are kept identical across all panels.
            axis.axvspan(
                pd.Timestamp("2024-06-01"),
                pd.Timestamp("2024-08-31"),
                color="0.90",
                zorder=0,
            )
            axis.axvline(PRIMARY_INTERVENTION, color="0.25", linestyle="--", linewidth=1.0)
            axis.set_title(
                f"[{chr(65 + panel_index)}] {frequency}: {category}",
                loc="left",
                fontsize=9,
            )
            if column_index == 0:
                axis.set_ylabel("Mean 1-year citations")
            axis.set_xlabel(
                "June-May cohort"
                if frequency == "Yearly"
                else f"Publication {frequency[:-2].lower()}"
            )
            format_period_axis(axis, frequency, panel)
            panel_index += 1
    axes[0, 0].legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    save_figure(fig, paths.main_figure_dir / "Figure_1_monthly_1year_citations")
    plt.close(fig)
    return summary


def plot_supplemental_composition(data: pd.DataFrame, paths: AnalysisPaths) -> pd.DataFrame:
    configure_plot_style()
    recent = data[data["publication_date"].between(RECENT_START, PUBLICATION_END)].copy()
    # Build one tidy summary first so the exported data and plotted values share a source.
    grouped_parts: list[pd.DataFrame] = []
    for frequency in FREQUENCY_ORDER:
        periodized = add_period_columns(recent, frequency)
        grouped_parts.append(
            periodized.groupby(
                ["journal_code", "frequency", "period_label", "period_date", "category"],
                observed=True,
            )
            .size()
            .rename("n")
            .reset_index()
        )
    grouped = pd.concat(grouped_parts, ignore_index=True)
    # This figure transposes the main layout: resolution by row and journal by column.
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharey=False)
    category_colors = {
        "ORA": "#2F5597",
        "Meta-analysis": "#70AD47",
        "Review": "#A5A5A5",
        "GSS": "#FFC000",
        "Non-citable items": "#ED7D31",
    }
    panel_index = 0
    for row_index, frequency in enumerate(FREQUENCY_ORDER):
        for column_index, journal in enumerate(JOURNAL_ORDER):
            axis = axes[row_index, column_index]
            panel = grouped[
                grouped["frequency"].eq(frequency) & grouped["journal_code"].eq(journal)
            ]
            # Reindexing preserves the same stack and color order even when a category is absent.
            pivot = (
                panel.pivot_table(
                    index=["period_date", "period_label"],
                    columns="category",
                    values="n",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reindex(columns=ATOMIC_CATEGORIES, fill_value=0)
                .sort_index()
            )
            period_dates = pivot.index.get_level_values("period_date")
            axis.stackplot(
                period_dates,
                [pivot[column].to_numpy() for column in ATOMIC_CATEGORIES],
                labels=ATOMIC_CATEGORIES,
                colors=[category_colors[column] for column in ATOMIC_CATEGORIES],
                alpha=0.85,
            )
            axis.axvline(PRIMARY_INTERVENTION, color="0.2", linestyle="--", linewidth=1)
            axis.set_title(
                f"[{chr(65 + panel_index)}] {frequency}: {JOURNAL_DISPLAY[journal]}",
                loc="left",
                fontsize=9,
            )
            if column_index == 0:
                axis.set_ylabel("Number of publications")
            axis.set_xlabel(
                "June-May cohort"
                if frequency == "Yearly"
                else f"Publication {frequency[:-2].lower()}"
            )
            format_period_axis(axis, frequency, panel)
            panel_index += 1
    axes[0, 2].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
    fig.tight_layout()
    save_figure(fig, paths.supp_figure_dir / "Supplemental_Figure_1_monthly_composition")
    plt.close(fig)
    return grouped
