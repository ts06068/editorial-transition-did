"""Reproducible analysis of citation changes after the JACC editorial transition."""

from __future__ import annotations

import os
from pathlib import Path

from .config import (
    ANALYSIS_CUTOFF,
    JOURNAL_DISPLAY,
    PRIMARY_INTERVENTION,
    AnalysisPaths,
)

__all__ = [
    "ANALYSIS_CUTOFF",
    "JOURNAL_DISPLAY",
    "PRIMARY_INTERVENTION",
    "AnalysisPaths",
    "run_all",
]
__version__ = "0.1.0"


def run_all(
    paths: AnalysisPaths | None = None,
    force_refresh_db: bool = False,
    primary_bootstrap_repetitions: int = 2000,
    robustness_bootstrap_repetitions: int = 1500,
) -> dict[str, object]:
    """Run the full analysis while deferring heavy plotting/model imports."""

    resolved_paths = paths or AnalysisPaths.from_environment()
    resolved_paths.create_directories()
    matplotlib_cache = resolved_paths.project_root / ".matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    from .pipeline import run_all as _run_all

    return _run_all(
        paths=resolved_paths,
        force_refresh_db=force_refresh_db,
        primary_bootstrap_repetitions=primary_bootstrap_repetitions,
        robustness_bootstrap_repetitions=robustness_bootstrap_repetitions,
    )


def package_root() -> Path:
    """Return the installed source-package directory (mainly for diagnostics)."""

    return Path(__file__).resolve().parent
