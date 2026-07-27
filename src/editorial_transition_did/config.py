"""Study configuration and filesystem paths for the editorial-transition analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _configured_path(value: str | None, default: Path, project_root: Path) -> Path:
    """Resolve an optional environment path relative to the project root."""

    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class AnalysisPaths:
    """Filesystem locations used by a full analysis run.

    Keeping paths in an immutable value object makes data transformations testable
    without relying on the package installation directory or module-level globals.
    """

    project_root: Path
    input_xlsx: Path
    html_root: Path
    env_file: Path
    output_dir: Path

    @property
    def root(self) -> Path:
        """Compatibility alias for the project root used by reporting modules."""

        return self.project_root

    @classmethod
    def from_environment(cls, project_root: Path | str | None = None) -> AnalysisPaths:
        """Build paths from ``EDITORIAL_DID_*`` variables and stable defaults."""

        root_value = project_root or os.getenv("EDITORIAL_DID_PROJECT_ROOT") or Path.cwd()
        root = Path(root_value).expanduser().resolve()
        return cls(
            project_root=root,
            input_xlsx=_configured_path(
                os.getenv("EDITORIAL_DID_INPUT_PATH"),
                root / "article_final.xlsx",
                root,
            ),
            html_root=_configured_path(
                os.getenv("EDITORIAL_DID_HTML_ROOT"),
                root,
                root,
            ),
            env_file=_configured_path(
                os.getenv("EDITORIAL_DID_ENV_FILE"),
                root / ".env",
                root,
            ),
            output_dir=_configured_path(
                os.getenv("EDITORIAL_DID_OUTPUT_DIR"),
                root / "analysis_results",
                root,
            ),
        )

    @property
    def main_table_dir(self) -> Path:
        return self.output_dir / "main_tables"

    @property
    def main_figure_dir(self) -> Path:
        return self.output_dir / "main_figures"

    @property
    def supp_table_dir(self) -> Path:
        return self.output_dir / "supplementary_tables"

    @property
    def supp_figure_dir(self) -> Path:
        return self.output_dir / "supplementary_figures"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    def create_directories(self) -> None:
        """Create every output directory required by the analysis."""

        for path in (
            self.output_dir,
            self.main_table_dir,
            self.main_figure_dir,
            self.supp_table_dir,
            self.supp_figure_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


ANALYSIS_CUTOFF = pd.Timestamp("2026-07-24")
PUBLICATION_START = pd.Timestamp("2008-01-01")
PUBLICATION_END = pd.Timestamp("2026-06-30")
RECENT_START = pd.Timestamp("2022-06-01")
PRIMARY_INTERVENTION = pd.Timestamp("2024-07-01")

JOURNALS = {
    "circ": {
        "display": "CIRC",
        "db": "Circulation",
        "color": "#FF3333",
    },
    "ehj": {
        "display": "EHJ",
        "db": "European Heart Journal",
        "color": "#33CC33",
    },
    "jacc": {
        "display": "JACC",
        "db": "Journal of the American College of Cardiology",
        "color": "#3333FF",
    },
}
JOURNAL_ORDER = ["circ", "ehj", "jacc"]
JOURNAL_DISPLAY = {key: value["display"] for key, value in JOURNALS.items()}
JOURNAL_COLORS = {key: value["color"] for key, value in JOURNALS.items()}

CATEGORY_ORDER = [
    "Overall",
    "ORA",
    "NORA",
    "Meta-analysis",
    "Review",
    "GSS",
    "Non-citable items",
]
ATOMIC_CATEGORIES = [
    "ORA",
    "Meta-analysis",
    "Review",
    "GSS",
    "Non-citable items",
]
NORA_CATEGORIES = {"Meta-analysis", "Review", "GSS"}

COHORTS = [
    ("2022.6-2023.5", pd.Timestamp("2022-06-01"), pd.Timestamp("2023-05-31"), True),
    ("2023.6-2024.5", pd.Timestamp("2023-06-01"), pd.Timestamp("2024-05-31"), True),
    ("2024.6-2025.5", pd.Timestamp("2024-06-01"), pd.Timestamp("2025-05-31"), True),
    (
        "2025.6-2026.6*",
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2026-06-30"),
        False,
    ),
]

HORIZONS = {
    "3 months": {"months": 15, "increment": "increment_3m", "observed": "c15_obs"},
    "6 months": {"months": 18, "increment": "increment_6m", "observed": "c18_obs"},
    "12 months": {"months": 24, "increment": "increment_12m", "observed": "c24_obs"},
}

FREQUENCY_ORDER = ["Yearly", "Quarterly", "Monthly"]

PREDICTOR_COLUMNS = [
    "landmark_log",
    "publication_year_centered",
    "month_sin",
    "month_cos",
    "journal_code",
    "category",
]

FIGURE_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "legend.frameon": False,
}


@dataclass(frozen=True)
class DidWindow:
    """Calendar windows defining one difference-in-differences specification."""

    label: str
    cutoff: pd.Timestamp
    pre_start: pd.Timestamp
    pre_end: pd.Timestamp
    post_start: pd.Timestamp
    post_end: pd.Timestamp


PRIMARY_WINDOW = DidWindow(
    label="Primary: July 2024 official transition, 12-month windows",
    cutoff=PRIMARY_INTERVENTION,
    pre_start=pd.Timestamp("2023-07-01"),
    pre_end=pd.Timestamp("2024-06-30"),
    post_start=pd.Timestamp("2024-07-01"),
    post_end=pd.Timestamp("2025-06-30"),
)

PLACEBO_WINDOW = DidWindow(
    label="Placebo: July 2023 cutoff, 12-month windows",
    cutoff=pd.Timestamp("2023-07-01"),
    pre_start=pd.Timestamp("2022-07-01"),
    pre_end=pd.Timestamp("2023-06-30"),
    post_start=pd.Timestamp("2023-07-01"),
    post_end=pd.Timestamp("2024-06-30"),
)


def make_sensitivity_windows() -> list[DidWindow]:
    """Return the alternative intervention-window definitions."""

    return [
        DidWindow(
            label="Sensitivity: June 2024 cutoff, 12-month windows",
            cutoff=pd.Timestamp("2024-06-01"),
            pre_start=pd.Timestamp("2023-06-01"),
            pre_end=pd.Timestamp("2024-05-31"),
            post_start=pd.Timestamp("2024-06-01"),
            post_end=pd.Timestamp("2025-05-31"),
        ),
        DidWindow(
            label="Sensitivity: August 2024 cutoff, 11 complete-month windows",
            cutoff=pd.Timestamp("2024-08-01"),
            pre_start=pd.Timestamp("2023-09-01"),
            pre_end=pd.Timestamp("2024-07-31"),
            post_start=pd.Timestamp("2024-08-01"),
            post_end=pd.Timestamp("2025-06-30"),
        ),
        DidWindow(
            label="Transition washout (June-August 2024), 10-month windows",
            cutoff=pd.Timestamp("2024-09-01"),
            pre_start=pd.Timestamp("2023-08-01"),
            pre_end=pd.Timestamp("2024-05-31"),
            post_start=pd.Timestamp("2024-09-01"),
            post_end=pd.Timestamp("2025-06-30"),
        ),
    ]
