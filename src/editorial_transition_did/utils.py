"""Small, side-effect-free helpers shared across analysis modules."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import NORA_CATEGORIES


def normalize_doi(value: object) -> str | None:
    """Normalize a DOI or DOI URL for deterministic record matching."""

    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    # DOI URLs, registry prefixes, and trailing citation punctuation all refer to
    # the same identifier and must collapse to one database join key.
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text).strip().rstrip(".,;")
    return text or None


def parse_issue_date(value: object, root: Path) -> pd.Timestamp | pd.NaT:
    """Recover a publication date from a locally archived publisher page."""

    if pd.isna(value):
        return pd.NaT
    path = root / Path(str(value).replace("\\", "/"))
    if not path.exists():
        return pd.NaT
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Patterns are ordered from structured metadata to publisher-specific markup,
    # so the most stable publication-date source wins when several are present.
    patterns = [
        r'<meta\s+name=["\']publish_date["\']\s+content=["\'](\d{4}-\d{2}-\d{2})["\']',
        r'["\']datePublished["\']\s*:\s*["\'](\d{4}-\d{2}-\d{2})["\']',
        r"toc__latest-issue__date[^>]*>\s*([^<]+?)\s*</span>",
        r"serial-publication-details.*?([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            parsed = pd.to_datetime(match.group(1).strip(), errors="coerce")
            if pd.notna(parsed):
                return pd.Timestamp(parsed).normalize()
    return pd.NaT


def category_mask(data: pd.DataFrame, category: str) -> pd.Series:
    """Select one reporting category, including the composite NORA group."""

    if category == "Overall":
        return pd.Series(True, index=data.index)
    if category == "NORA":
        # NORA pools the three non-original-research categories for reporting only.
        return data["category"].isin(NORA_CATEGORIES)
    return data["category"].eq(category)


def format_p_value(value: float) -> str:
    """Format a p value using the manuscript convention."""

    if not np.isfinite(value):
        return "NA"
    # Manuscript tables report very small values as a threshold, not rounded zero.
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"
