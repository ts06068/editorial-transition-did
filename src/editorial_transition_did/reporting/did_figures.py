from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..config import (
    CATEGORY_ORDER,
    FREQUENCY_ORDER,
    JOURNAL_COLORS,
    JOURNAL_DISPLAY,
    JOURNAL_ORDER,
    PRIMARY_INTERVENTION,
    PRIMARY_WINDOW,
    AnalysisPaths,
)
from ..summaries import add_period_columns, complete_period_mask
from ..utils import category_mask, format_p_value
from .plotting import configure_plot_style, format_period_axis, save_figure


def plot_figure_2(data: pd.DataFrame, primary_did: pd.DataFrame, paths: AnalysisPaths) -> None:
    configure_plot_style()
    # Panel A shows observed levels; panel B shows the DiD estimand.
    overall = primary_did.loc[primary_did["category"].eq("Overall")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [0.9, 1.35]})

    axis = axes[0]
    x_values = np.array([0.0, 1.0])
    # Offset marks only; all estimates share the same two time points.
    horizontal_jitter = {"circ": -0.035, "ehj": 0.0, "jacc": 0.035}
    for journal in JOURNAL_ORDER:
        means = np.array([overall[f"{journal}_pre_mean"], overall[f"{journal}_post_mean"]])
        ses = np.array(
            [
                overall[f"{journal}_pre_sd"] / math.sqrt(overall[f"{journal}_pre_n"]),
                overall[f"{journal}_post_sd"] / math.sqrt(overall[f"{journal}_post_n"]),
            ]
        )
        axis.errorbar(
            x_values + horizontal_jitter[journal],
            means,
            yerr=1.96 * ses,
            color=JOURNAL_COLORS[journal],
            marker="s",
            markersize=5,
            linewidth=1.5,
            capsize=2,
            label=JOURNAL_DISPLAY[journal],
        )
    axis.set_xticks(x_values, ["Pre\n2023.7-2024.6", "Post\n2024.7-2025.6"])
    axis.set_xlim(-0.2, 1.2)
    axis.set_ylabel("Mean 1-year citations")
    axis.set_title("[A] Pre-post change", loc="left")
    axis.legend(loc="upper left")

    axis = axes[1]
    # Model execution order is not a presentation contract.
    forest = primary_did.set_index("category").loc[CATEGORY_ORDER].reset_index()
    y_values = np.arange(len(forest))[::-1]
    x_min, x_max = -15.0, 15.0
    for y_value, row in zip(y_values, forest.itertuples(index=False), strict=False):
        estimate = float(row.did)
        low = float(row.ci_low)
        high = float(row.ci_high)
        if not np.isfinite(estimate):
            axis.text(0, y_value, "NA", ha="center", va="center", color="0.45", fontsize=8)
            continue
        # Clip display coordinates and mark truncation at each boundary.
        clipped_estimate = float(np.clip(estimate, x_min, x_max))
        clipped_low = float(np.clip(low, x_min, x_max))
        clipped_high = float(np.clip(high, x_min, x_max))
        axis.plot([clipped_low, clipped_high], [y_value, y_value], color="0.35", linewidth=1)
        axis.plot(
            [clipped_low, clipped_low], [y_value - 0.06, y_value + 0.06], color="0.35", linewidth=1
        )
        axis.plot(
            [clipped_high, clipped_high],
            [y_value - 0.06, y_value + 0.06],
            color="0.35",
            linewidth=1,
        )
        marker = "s"
        if estimate < x_min:
            marker = "<"
        elif estimate > x_max:
            marker = ">"
        axis.scatter(
            [clipped_estimate],
            [y_value],
            marker=marker,
            s=30,
            color=JOURNAL_COLORS["jacc"],
            zorder=3,
        )
        if low < x_min:
            axis.scatter([x_min], [y_value], marker="<", s=22, color="0.35", zorder=2)
        if high > x_max:
            axis.scatter([x_max], [y_value], marker=">", s=22, color="0.35", zorder=2)
        if estimate < x_min or estimate > x_max:
            text_x = x_min + 0.8 if estimate < x_min else x_max - 0.8
            horizontal = "left" if estimate < x_min else "right"
            axis.text(
                text_x, y_value + 0.16, f"{estimate:.1f}", ha=horizontal, va="bottom", fontsize=7
            )
    axis.axvline(0, color="0.35", linestyle="--", linewidth=1)
    axis.set_xlim(x_min, x_max)
    axis.set_yticks(y_values, forest["category"])
    axis.set_xlabel("Raw difference-in-differences in 1-year citations")
    axis.set_title("[B] JACC vs equal-weight CIRC and EHJ", loc="left")
    fig.tight_layout()
    save_figure(fig, paths.main_figure_dir / "Figure_2_primary_difference_in_differences")
    plt.close(fig)


def plot_figure_3(raw: pd.DataFrame, log_results: pd.DataFrame, paths: AnalysisPaths) -> None:
    configure_plot_style()
    # Scale height with row count to keep labels readable.
    figure_height = max(5.2, 1.6 + 0.36 * max(len(raw), len(log_results)))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, figure_height))
    panels = [
        (
            axes[0],
            raw,
            "did",
            "ci_low",
            "ci_high",
            "Raw DiD in citations",
            "[A] Raw-scale analyses",
        ),
        (
            axes[1],
            log_results,
            "percent_change",
            "percent_ci_low",
            "percent_ci_high",
            "Relative change from log(1 + citation) OLS, %",
            "[B] Log-scale analyses",
        ),
    ]
    for axis, frame, estimate, low, high, xlabel, title in panels:
        # Reverse rows because Matplotlib places y=0 at the bottom.
        display = frame.iloc[::-1].reset_index(drop=True)
        y_values = np.arange(len(display))
        errors = np.vstack(
            [
                display[estimate].to_numpy() - display[low].to_numpy(),
                display[high].to_numpy() - display[estimate].to_numpy(),
            ]
        )
        axis.errorbar(
            display[estimate],
            y_values,
            xerr=errors,
            fmt="s",
            color=JOURNAL_COLORS["jacc"],
            ecolor="0.35",
            capsize=2,
            markersize=4,
            linewidth=1,
        )
        axis.axvline(0, color="0.35", linestyle="--", linewidth=1)
        axis.set_yticks(y_values, display["display_label"], fontsize=7)
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left")
    fig.tight_layout()
    save_figure(fig, paths.main_figure_dir / "Figure_3_robustness_analyses")
    plt.close(fig)


def plot_pretrend_difference(
    data: pd.DataFrame,
    paths: AnalysisPaths,
    category: str = "ORA",
) -> tuple[pd.DataFrame, dict[str, float]]:
    configure_plot_style()
    trend_start = PRIMARY_WINDOW.pre_start - pd.DateOffset(months=12)
    recent = data[
        data["publication_date"].between(trend_start, PRIMARY_WINDOW.post_end)
        & data["c12_obs"].notna()
    ].copy()
    recent = recent[category_mask(recent, category)].copy()
    summaries: list[pd.DataFrame] = []
    for frequency in FREQUENCY_ORDER:
        periodized = add_period_columns(recent, frequency)
        periodized = periodized[complete_period_mask(periodized, 12)].copy()
        grouped = (
            periodized.groupby(
                ["frequency", "period_label", "period_date", "period_end", "journal_code"],
                observed=True,
            )["c12_obs"]
            .mean()
            .unstack("journal_code")
            .reset_index()
        )
        # Average controls first; publication volume must not change their weights.
        grouped["control_mean"] = grouped[["circ", "ehj"]].mean(axis=1)
        grouped["jacc_minus_control"] = grouped["jacc"] - grouped["control_mean"]
        summaries.append(grouped)
    combined = pd.concat(summaries, ignore_index=True)
    monthly = combined[combined["frequency"].eq("Monthly")].sort_values("period_date").copy()
    pre = monthly[monthly["period_date"].lt(PRIMARY_INTERVENTION)].copy()
    # Fit the diagnostic slope on complete pre-intervention months only.
    pre["month_index"] = np.arange(len(pre))
    trend = stats.linregress(pre["month_index"], pre["jacc_minus_control"])
    trend_info = {
        "category": category,
        "monthly_slope": float(trend.slope),
        "slope_ci_low": float(trend.slope - 1.96 * trend.stderr),
        "slope_ci_high": float(trend.slope + 1.96 * trend.stderr),
        "p_value": float(trend.pvalue),
    }

    # Match the frequency order used by the descriptive figures.
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8), sharex=False)
    for panel_index, (axis, frequency) in enumerate(zip(axes, FREQUENCY_ORDER, strict=False)):
        panel = combined[combined["frequency"].eq(frequency)].sort_values("period_date")
        axis.plot(
            panel["period_date"],
            panel["jacc_minus_control"],
            color=JOURNAL_COLORS["jacc"],
            marker="s",
            markersize=3,
            linewidth=1.3,
        )
        axis.axhline(0, color="0.55", linewidth=0.8)
        axis.axvspan(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-08-31"), color="0.90")
        axis.axvline(PRIMARY_INTERVENTION, color="0.25", linestyle="--", linewidth=1)
        axis.set_title(f"[{chr(65 + panel_index)}] {frequency}", loc="left")
        axis.set_xlabel(
            "June-May cohort" if frequency == "Yearly" else f"Publication {frequency[:-2].lower()}"
        )
        axis.set_ylabel(f"JACC minus control\nmean {category} 1-year citations")
        format_period_axis(axis, frequency, panel)
        if frequency == "Monthly":
            axis.text(
                0.02,
                0.96,
                f"Pre-intervention monthly slope: {trend.slope:.2f} "
                f"(95% CI {trend_info['slope_ci_low']:.2f} to {trend_info['slope_ci_high']:.2f}; "
                f"p {format_p_value(trend.pvalue)})",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
    fig.tight_layout()
    save_figure(fig, paths.supp_figure_dir / "Supplemental_Figure_2_pretrend_difference")
    plt.close(fig)
    return combined, trend_info
