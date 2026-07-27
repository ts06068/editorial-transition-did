from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..config import (
    FREQUENCY_ORDER,
    HORIZONS,
    JOURNAL_COLORS,
    JOURNAL_DISPLAY,
    JOURNAL_ORDER,
    PRIMARY_INTERVENTION,
    AnalysisPaths,
)
from ..summaries import add_period_columns
from ..utils import format_p_value
from .plotting import configure_plot_style, format_period_axis, save_figure


def plot_figure_4(data: pd.DataFrame, test_predictions: pd.DataFrame, paths: AnalysisPaths) -> None:
    configure_plot_style()
    # Keep the temporal test era untouched for validation plots.
    scatter = data[
        data["publication_date"].between(pd.Timestamp("2022-06-01"), pd.Timestamp("2024-05-31"))
        & data["c12_obs"].notna()
        & data["increment_12m"].notna()
    ].copy()
    # Align journal panels above forecast-horizon panels.
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for index, (axis, journal) in enumerate(zip(axes[0], JOURNAL_ORDER, strict=False)):
        panel = scatter[scatter["journal_code"].eq(journal)]
        x_values = np.log1p(panel["c12_obs"].to_numpy(dtype=float))
        y_values = np.log1p(panel["increment_12m"].to_numpy(dtype=float))
        axis.scatter(
            x_values,
            y_values,
            s=8,
            alpha=0.28,
            color=JOURNAL_COLORS[journal],
            edgecolors="none",
        )
        rho, p_value = stats.spearmanr(panel["c12_obs"], panel["increment_12m"])
        axis.text(
            0.04,
            0.95,
            f"Spearman rho = {rho:.2f}\np {format_p_value(p_value)}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.set_title(f"[{chr(65 + index)}] {JOURNAL_DISPLAY[journal]}", loc="left")
        axis.set_xlabel("log(1 + 1-year citations)")
        if index == 0:
            axis.set_ylabel("log(1 + second-year citation increment)")

    model_tests = test_predictions[test_predictions["horizon"].isin(HORIZONS)].copy()
    for offset, (axis, horizon) in enumerate(zip(axes[1], HORIZONS, strict=False), start=3):
        panel = model_tests[model_tests["horizon"].eq(horizon)].copy()
        # Ties can reduce the number of valid quantile bins below ten.
        quantile_count = min(10, panel["predicted_increment"].nunique())
        panel["prediction_group"] = pd.qcut(
            panel["predicted_increment"], q=quantile_count, duplicates="drop"
        )
        calibration = (
            panel.groupby("prediction_group", observed=True)
            .agg(
                predicted=("predicted_increment", "mean"),
                observed=("observed_increment", "mean"),
                observed_se=("observed_increment", "sem"),
            )
            .reset_index(drop=True)
        )
        # Shared square limits preserve distance from the calibration diagonal.
        maximum = max(calibration["predicted"].max(), calibration["observed"].max()) * 1.05
        axis.plot([0, maximum], [0, maximum], color="0.55", linestyle="--", linewidth=1)
        axis.errorbar(
            calibration["predicted"],
            calibration["observed"],
            yerr=1.96 * calibration["observed_se"],
            color=JOURNAL_COLORS["jacc"],
            marker="s",
            markersize=4,
            linewidth=1.2,
            capsize=2,
        )
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, maximum)
        axis.set_title(f"[{chr(65 + offset)}] {horizon} after year 1", loc="left")
        axis.set_xlabel("Mean predicted citation increment")
        if offset == 3:
            axis.set_ylabel("Mean observed citation increment")
    fig.tight_layout()
    save_figure(fig, paths.main_figure_dir / "Figure_4_prediction_relationship_and_calibration")
    plt.close(fig)


def plot_figure_5(forecasts: pd.DataFrame, paths: AnalysisPaths) -> pd.DataFrame:
    configure_plot_style()
    # Return the same period-level frame used by the plot.
    period_rows: list[dict[str, object]] = []
    for frequency in FREQUENCY_ORDER:
        periodized = add_period_columns(forecasts, frequency)
        for keys, group in periodized.groupby(
            [
                "horizon",
                "journal_code",
                "frequency",
                "period_label",
                "period_date",
                "period_end",
            ],
            observed=True,
        ):
            horizon, journal, _, period_label, period_date, period_end = keys
            observed_values = group["observed_cumulative"].dropna()
            # Mark a period observed only when every article reaches the horizon.
            fully_observed = len(observed_values) == len(group)
            period_rows.append(
                {
                    "horizon": horizon,
                    "journal_code": journal,
                    "frequency": frequency,
                    "period_label": period_label,
                    "period_date": period_date,
                    "period_end": period_end,
                    "n": len(group),
                    "is_observed": bool(fully_observed),
                    "observed_mean": float(observed_values.mean()) if fully_observed else np.nan,
                    "forecast_mean": float(group["predicted_cumulative"].mean()),
                    "forecast_low80": float(group["cumulative_low80"].mean()),
                    "forecast_high80": float(group["cumulative_high80"].mean()),
                }
            )
    period_summary = pd.DataFrame(period_rows)

    # Keep frequency on rows and forecast horizon on columns.
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8.2), sharey=False)
    panel_index = 0
    for row_index, frequency in enumerate(FREQUENCY_ORDER):
        for column_index, horizon in enumerate(HORIZONS):
            axis = axes[row_index, column_index]
            panel = period_summary[
                period_summary["frequency"].eq(frequency) & period_summary["horizon"].eq(horizon)
            ]
            for journal in JOURNAL_ORDER:
                journal_data = panel[panel["journal_code"].eq(journal)].sort_values("period_date")
                observed = journal_data[journal_data["is_observed"]]
                forecast = journal_data[~journal_data["is_observed"]]
                axis.plot(
                    observed["period_date"],
                    observed["observed_mean"],
                    color=JOURNAL_COLORS[journal],
                    marker="s",
                    markersize=2.5,
                    linewidth=1.4,
                    label=JOURNAL_DISPLAY[journal],
                )
                if not forecast.empty:
                    # Join the forecast at the last observed mean; do not imply a gap.
                    bridge = journal_data[
                        journal_data["period_date"].isin(
                            [observed["period_date"].max() if not observed.empty else pd.NaT]
                        )
                    ].copy()
                    if not bridge.empty:
                        bridge["forecast_mean"] = bridge["observed_mean"]
                        forecast_line = pd.concat([bridge, forecast], ignore_index=True)
                    else:
                        forecast_line = forecast
                    axis.plot(
                        forecast_line["period_date"],
                        forecast_line["forecast_mean"],
                        color=JOURNAL_COLORS[journal],
                        linestyle="--",
                        marker="s",
                        markersize=2.5,
                        linewidth=1.4,
                    )
                    # Draw uncertainty only for model-based values.
                    axis.fill_between(
                        forecast["period_date"],
                        forecast["forecast_low80"],
                        forecast["forecast_high80"],
                        color=JOURNAL_COLORS[journal],
                        alpha=0.08,
                        linewidth=0,
                    )
            target_months = int(HORIZONS[horizon]["months"])
            axis.axvline(PRIMARY_INTERVENTION, color="0.25", linestyle=":", linewidth=1)
            axis.set_title(
                f"[{chr(65 + panel_index)}] {frequency}: C{target_months}",
                loc="left",
                fontsize=9,
            )
            axis.set_xlabel(
                "June-May cohort"
                if frequency == "Yearly"
                else f"Publication {frequency[:-2].lower()}"
            )
            if column_index == 0:
                axis.set_ylabel("Mean citation count")
            format_period_axis(axis, frequency, panel)
            panel_index += 1
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, paths.main_figure_dir / "Figure_5_observed_and_forecast_citations")
    plt.close(fig)
    return period_summary
