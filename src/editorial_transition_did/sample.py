"""Small, deterministic analysis used for smoke tests and demonstrations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import PRIMARY_WINDOW
from .did import calculate_raw_did
from .summaries import (
    build_citation_summary,
    build_composition_table,
    build_period_summaries,
)

SAMPLE_JOURNALS = {
    "circ": "CIRC",
    "ehj": "EHJ",
    "jacc": "JACC",
}


def make_sample_data() -> pd.DataFrame:
    """Return a deterministic, article-level three-journal sample.

    The values are deliberately simple. JACC increases by 6 citations from
    pre to post, while CIRC and EHJ increase by 2 and 1, respectively. The
    equal-journal-weighted difference-in-differences estimate is therefore
    ``6 - (2 + 1) / 2 = 4.5``.
    """

    dates = {
        "pre": [pd.Timestamp("2023-08-15"), pd.Timestamp("2024-02-15")],
        "post": [pd.Timestamp("2024-08-15"), pd.Timestamp("2025-02-15")],
    }
    categories = ["ORA", "Review"]
    citations = {
        "jacc": {"pre": [10, 12, 14, 16], "post": [16, 18, 20, 22]},
        "circ": {"pre": [8, 10, 12, 14], "post": [10, 12, 14, 16]},
        "ehj": {"pre": [6, 8, 10, 12], "post": [7, 9, 11, 13]},
    }

    rows: list[dict[str, object]] = []
    record_number = 0
    # Cross the three journals, two periods, two dates, and two article types.
    for journal_code, journal_display in SAMPLE_JOURNALS.items():
        for period in ("pre", "post"):
            value_index = 0
            for publication_date in dates[period]:
                for category in categories:
                    record_number += 1
                    c12 = citations[journal_code][period][value_index]
                    value_index += 1
                    record_id = f"sample-{journal_code}-{record_number:03d}"
                    # Match the production cohort schema closely enough to exercise summaries.
                    rows.append(
                        {
                            "analysis_record_id": record_id,
                            "paper_id": record_number,
                            "db_paper_id": record_number,
                            "eid": f"2-s2.0-{record_number:010d}",
                            "doi_norm": f"10.0000/{record_id}",
                            "title": f"Synthetic {journal_display} {category} article",
                            "journal_code": journal_code,
                            "journal_display": journal_display,
                            "publication_date": publication_date,
                            "publication_month": publication_date.to_period("M").to_timestamp(),
                            "publication_year": publication_date.year,
                            "publication_quarter": str(publication_date.to_period("Q")),
                            "category": category,
                            "is_jacc": int(journal_code == "jacc"),
                            "is_nora": category == "Review",
                            "complete_12m": True,
                            "c12_obs": float(c12),
                            "citations_at_snapshot": float(c12 + 2),
                        }
                    )

    return pd.DataFrame(rows).sort_values(
        ["publication_date", "journal_code", "category"], ignore_index=True
    )


def run_sample(
    output_dir: str | Path = Path("outputs") / "sample",
    bootstrap_repetitions: int = 200,
) -> dict[str, object]:
    """Run a database-free smoke analysis and write lightweight artefacts."""

    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be at least 1")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    data = make_sample_data()
    # Reuse production analysis functions so this run tests real integration boundaries.
    composition = build_composition_table(data)
    citations = build_citation_summary(data)
    period_summaries = build_period_summaries(data)
    primary_did = pd.DataFrame(
        [
            calculate_raw_did(
                data,
                PRIMARY_WINDOW,
                category="Overall",
                controls=("circ", "ehj"),
                n_boot=bootstrap_repetitions,
                # A fixed seed keeps local and CI smoke-test output reproducible.
                seed=20260724,
            )
        ]
    )

    tables = {
        "sample_composition.csv": composition,
        "sample_citations.csv": citations,
        "sample_monthly_composition.csv": period_summaries["month_composition"],
        "sample_monthly_citations.csv": period_summaries["month_citations"],
        "sample_quarterly_composition.csv": period_summaries["quarter_composition"],
        "sample_quarterly_citations.csv": period_summaries["quarter_citations"],
        "sample_primary_did.csv": primary_did,
    }
    for filename, table in tables.items():
        # CSV keeps smoke artefacts inspectable without spreadsheet dependencies.
        table.to_csv(destination / filename, index=False)

    did_row = primary_did.iloc[0]
    # The JSON summary gives CI a compact, machine-readable completion record.
    summary = {
        "mode": "sample",
        "records": int(len(data)),
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "primary_overall_did": {
            "estimate": float(did_row["did"]),
            "ci_low": float(did_row["ci_low"]),
            "ci_high": float(did_row["ci_high"]),
            "p_value": float(did_row["p_value"]),
        },
        "files": [*tables, "sample_summary.json"],
    }
    summary_path = destination / "sample_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "data": data,
        "composition": composition,
        "citations": citations,
        "period_summaries": period_summaries,
        "primary_did": primary_did,
        "summary": summary,
        "output_dir": destination,
    }
