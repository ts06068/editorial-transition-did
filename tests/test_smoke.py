"""Fast, private-data-free checks for the public analysis surface."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from editorial_transition_did.config import PRIMARY_WINDOW
from editorial_transition_did.did import calculate_raw_did
from editorial_transition_did.sample import make_sample_data, run_sample
from editorial_transition_did.summaries import add_period_columns
from editorial_transition_did.utils import normalize_doi

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://doi.org/10.1000/Example.1", "10.1000/example.1"),
        ("doi: 10.1000/EXAMPLE.2.;", "10.1000/example.2"),
        (None, None),
    ],
)
def test_normalize_doi(value: object, expected: str | None) -> None:
    assert normalize_doi(value) == expected


def test_monthly_period_assignment() -> None:
    frame = pd.DataFrame(
        {"publication_date": pd.to_datetime(["2024-01-31", "2024-02-01", "2025-07-15"])}
    )

    result = add_period_columns(frame, "Monthly")

    # February 2024 catches leap-year boundary bugs.
    assert result["period_label"].astype(str).tolist() == ["2024-01", "2024-02", "2025-07"]
    assert result["period_start"].dt.day.tolist() == [1, 1, 1]
    assert result["period_end"].dt.day.tolist() == [31, 29, 31]


def test_raw_did_matches_hand_calculation() -> None:
    # Hand calculation: 6 - mean(2, 1) = 4.5.
    result = calculate_raw_did(
        make_sample_data(),
        PRIMARY_WINDOW,
        category="Overall",
        controls=("circ", "ehj"),
        n_boot=64,
        seed=7,
    )

    assert result["jacc_change"] == pytest.approx(6.0)
    assert result["circ_change"] == pytest.approx(2.0)
    assert result["ehj_change"] == pytest.approx(1.0)
    assert result["control_change"] == pytest.approx(1.5)
    assert result["did"] == pytest.approx(4.5)


def test_sample_run_writes_lightweight_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "sample-output"

    results = run_sample(output_dir=output_dir, bootstrap_repetitions=32)

    expected_files = {
        "sample_composition.csv",
        "sample_citations.csv",
        "sample_monthly_composition.csv",
        "sample_monthly_citations.csv",
        "sample_quarterly_composition.csv",
        "sample_quarterly_citations.csv",
        "sample_primary_did.csv",
        "sample_summary.json",
    }
    # Filenames are part of the sample-mode contract.
    assert {path.name for path in output_dir.iterdir()} == expected_files
    assert all((output_dir / filename).stat().st_size > 0 for filename in expected_files)
    assert results["primary_did"].loc[0, "did"] == pytest.approx(4.5)

    summary = json.loads((output_dir / "sample_summary.json").read_text(encoding="utf-8"))
    assert summary["records"] == 24
    assert summary["primary_overall_did"]["estimate"] == pytest.approx(4.5)


def test_notebook_is_output_free_and_portable() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "analysis.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        # Notebook commits must not carry outputs or local state.
        assert "execution" not in cell.get("metadata", {})
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    # Use the installed package. No path surgery.
    assert "run_all(" in source
    assert "reload(" not in source
    assert "sys.path" not in source
    assert re.search(r"[A-Za-z]:[\\/]", source) is None
    assert "/Users/" not in source
