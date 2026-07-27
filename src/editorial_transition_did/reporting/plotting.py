from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import pandas as pd

from ..config import FIGURE_STYLE


def configure_plot_style() -> None:
    """Apply the manuscript figure style."""
    # A single rcParams source keeps typography and line weights consistent across exporters.
    mpl.rcParams.update(FIGURE_STYLE)


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    """Write a figure as vector PDF and high-resolution PNG."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Matching stems provide an editable manuscript asset and a review-friendly raster copy.
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)


def format_period_axis(axis: mpl.axes.Axes, frequency: str, data: pd.DataFrame) -> None:
    if frequency == "Yearly":
        # Annual cohorts are June-May, so use their explicit labels instead of calendar years.
        ticks = data[["period_date", "period_label"]].drop_duplicates().sort_values("period_date")
        labels = ticks["period_label"].replace(
            {
                "2022.6-2023.5": "2022.6-23.5",
                "2023.6-2024.5": "2023.6-24.5",
                "2024.6-2025.5": "2024.6-25.5",
                "2025.6-2026.6*": "2025.6-26.6",
            }
        )
        axis.set_xticks(ticks["period_date"], labels, fontsize=7)
    else:
        # Quarterly and monthly panels share sparse calendar-year ticks to avoid label crowding.
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
