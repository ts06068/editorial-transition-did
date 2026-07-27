"""Final analysis invariants and generated-file manifest creation."""

from datetime import datetime

import pandas as pd

from .config import (
    ATOMIC_CATEGORIES,
    CATEGORY_ORDER,
    HORIZONS,
    JOURNAL_ORDER,
    PRIMARY_WINDOW,
    AnalysisPaths,
)
from .did import select_did_window


def validate_results(
    data: pd.DataFrame,
    composition: pd.DataFrame,
    citations: pd.DataFrame,
    primary_did: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the final cohort, estimates, and prediction outputs."""
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    # Cohort integrity: only the three prespecified journals and mutually
    # exclusive atomic article categories may reach the analysis frame.
    add(
        "JAMA excluded",
        set(data["journal_code"].dropna().unique()) == set(JOURNAL_ORDER),
        ", ".join(sorted(data["journal_code"].dropna().unique())),
    )
    add(
        "All records classified",
        data["category"].isin(ATOMIC_CATEGORIES).all(),
        f"unclassified={int((~data['category'].isin(ATOMIC_CATEGORIES)).sum())}",
    )
    add(
        "No duplicate analysis IDs",
        not data["analysis_record_id"].duplicated().any(),
        f"duplicates={int(data['analysis_record_id'].duplicated().sum())}",
    )
    # Citation increments are derived from cumulative snapshots, so a negative
    # value would reveal a source-order or interval-construction error.
    add(
        "No negative citation increments",
        not data[["increment_3m", "increment_6m", "increment_12m"]].lt(0).any().any(),
        "Cumulative counts are nondecreasing",
    )
    # The primary outcome is valid only when every included publication has the
    # complete 12-month follow-up required by the estimand.
    primary_data = select_did_window(data, PRIMARY_WINDOW)
    add(
        "Primary cohort has complete 1-year follow-up",
        primary_data["c12_obs"].notna().all(),
        f"eligible={primary_data['c12_obs'].notna().sum()}/{len(primary_data)}",
    )
    # The incomplete latest cohort uses explicitly labeled cumulative counts;
    # it must not be mistaken for the complete one-year outcome.
    latest_citations = citations[citations["cohort"].eq("2025.6-2026.6*")]
    latest_overall = latest_citations[latest_citations["category"].eq("Overall")]
    add(
        "Latest cohort reports separately labeled total cumulative citations",
        latest_citations["outcome_type"].eq("total cumulative").all()
        and latest_overall["eligible_n"].gt(0).all(),
        (
            f"eligible Scopus records={int(latest_overall['eligible_n'].sum())}/"
            f"{int(latest_overall['total_n'].sum())}"
        ),
    )
    # This accounting identity catches overlap or leakage in category assignment.
    atomic = composition[composition["category"].isin(ATOMIC_CATEGORIES)]
    overall = composition[composition["category"].eq("Overall")]
    atomic_sum = atomic.groupby(["journal_code", "cohort"], observed=True)["n"].sum()
    overall_n = overall.set_index(["journal_code", "cohort"])["n"]
    add(
        "Atomic category counts sum to overall counts",
        atomic_sum.equals(overall_n.reindex(atomic_sum.index)),
        f"cells={len(atomic_sum)}",
    )
    # Sparse subgroup estimates may be undefined, but the planned row set and
    # overall estimand must always be present.
    undefined_groups = ",".join(primary_did.loc[primary_did["did"].isna(), "category"].astype(str))
    add(
        "Primary DiD available for all planned groups",
        len(primary_did) == len(CATEGORY_ORDER)
        and primary_did.loc[primary_did["category"].eq("Overall"), "did"].notna().all(),
        (f"groups={len(primary_did)}; sparse undefined groups={undefined_groups or 'none'}"),
    )
    # Verify the designated forecasting model has evidence at every requested
    # citation horizon, independent of any additional benchmark model rows.
    model_rows = metrics[metrics["model"].eq("Poisson gradient boosting")]
    add(
        "Temporal prediction metrics available at all horizons",
        set(model_rows["horizon"]) == set(HORIZONS),
        f"horizons={','.join(model_rows['horizon'].astype(str))}",
    )
    output = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    # Fail closed: reporting continues only when every invariant is satisfied.
    if not output["passed"].all():
        failures = output.loc[~output["passed"], "check"].tolist()
        raise RuntimeError(f"Analysis validation failed: {failures}")
    return output


def build_manifest(paths: AnalysisPaths) -> pd.DataFrame:
    """Return metadata for every file currently present in the results tree."""
    files = [path for path in paths.output_dir.rglob("*") if path.is_file()]
    rows = []
    for path in sorted(files):
        try:
            # Prefer project-relative paths so manifests remain portable.
            relative_path = path.relative_to(paths.root)
        except ValueError:
            # Custom output directories can sit outside the configured root;
            # anchor those paths at the output directory's parent instead.
            relative_path = path.relative_to(paths.output_dir.parent)
        rows.append(
            {
                "file": str(relative_path),
                "size_bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return pd.DataFrame(rows)
