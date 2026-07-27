"""Difference-in-differences estimators and robustness specifications."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from tqdm.auto import tqdm

from .config import (
    CATEGORY_ORDER,
    JOURNAL_DISPLAY,
    PLACEBO_WINDOW,
    PRIMARY_WINDOW,
    DidWindow,
    make_sensitivity_windows,
)
from .utils import category_mask, format_p_value


def select_did_window(data: pd.DataFrame, window: DidWindow) -> pd.DataFrame:
    """Select complete 1-year outcomes in a pre/post DiD window."""

    # Require complete C12 follow-up before exposure assignment.
    selected = data[
        data["publication_date"].between(window.pre_start, window.post_end, inclusive="both")
        & data["c12_obs"].notna()
    ].copy()
    selected["period"] = np.select(
        [
            selected["publication_date"].between(
                window.pre_start, window.pre_end, inclusive="both"
            ),
            selected["publication_date"].between(
                window.post_start, window.post_end, inclusive="both"
            ),
        ],
        ["Pre", "Post"],
        default="Outside",
    )
    # Sensitivity windows may leave handover months deliberately unassigned.
    return selected[selected["period"].isin(["Pre", "Post"])].copy()


def _bootstrap_raw_did(
    selected: pd.DataFrame,
    controls: Sequence[str],
    n_boot: int,
    seed: int,
) -> np.ndarray:
    # Aggregate first so all journals in a sampled publication month move together.
    monthly = (
        selected.groupby(["journal_code", "period", "publication_month"], observed=True)["c12_obs"]
        .agg(total="sum", n="size")
        .reset_index()
    )
    # Matrix contract: column zero is treated; all remaining columns are controls.
    journals = ["jacc", *controls]
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for period in ["Pre", "Post"]:
        period_data = monthly[monthly["period"].eq(period)]
        totals = (
            period_data.pivot(index="publication_month", columns="journal_code", values="total")
            .reindex(columns=journals)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        counts = (
            period_data.pivot(index="publication_month", columns="journal_code", values="n")
            .reindex(columns=journals)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if totals.shape[0] == 0 or np.any(counts.sum(axis=0) == 0):
            return np.full(n_boot, np.nan)
        arrays[period] = (totals, counts)

    # Draw months jointly across journals to retain common shocks and covariance.
    rng = np.random.default_rng(seed)
    means: dict[str, np.ndarray] = {}
    for period, (totals, counts) in arrays.items():
        sampled = rng.integers(0, totals.shape[0], size=(n_boot, totals.shape[0]))
        sampled_totals = totals[sampled].sum(axis=1)
        sampled_counts = counts[sampled].sum(axis=1)
        means[period] = np.divide(
            sampled_totals,
            sampled_counts,
            out=np.full_like(sampled_totals, np.nan, dtype=float),
            where=sampled_counts > 0,
        )
    changes = means["Post"] - means["Pre"]
    # Weight control journals equally; article volume must not set the estimand.
    return changes[:, 0] - changes[:, 1:].mean(axis=1)


def calculate_raw_did(
    data: pd.DataFrame,
    window: DidWindow,
    category: str = "Overall",
    controls: Sequence[str] = ("circ", "ehj"),
    n_boot: int = 2000,
    seed: int = 20260724,
) -> dict[str, object]:
    """Calculate the prespecified raw four-mean DiD and block-bootstrap CI."""

    selected = select_did_window(data, window)
    selected = selected[category_mask(selected, category)].copy()
    selected = selected[selected["journal_code"].isin(["jacc", *controls])]
    cell_stats = (
        selected.groupby(["journal_code", "period"], observed=True)["c12_obs"]
        .agg(n="size", mean="mean", sd="std", se="sem")
        .reset_index()
    )
    lookup = cell_stats.set_index(["journal_code", "period"])

    def cell(journal: str, period: str, column: str) -> float:
        try:
            return float(lookup.loc[(journal, period), column])
        except KeyError:
            return np.nan

    changes = {
        journal: cell(journal, "Post", "mean") - cell(journal, "Pre", "mean")
        for journal in ["jacc", *controls]
    }
    # Estimand: JACC change minus the unweighted mean control change.
    control_change = float(np.mean([changes[journal] for journal in controls]))
    estimate = changes["jacc"] - control_change
    draws = _bootstrap_raw_did(selected, controls, n_boot=n_boot, seed=seed)
    finite_draws = draws[np.isfinite(draws)]
    if finite_draws.size:
        ci_low, ci_high = np.quantile(finite_draws, [0.025, 0.975])
        # The two-sided empirical p value uses the smaller bootstrap tail twice.
        p_value = min(
            1.0,
            2
            * min(
                float(np.mean(finite_draws <= 0)),
                float(np.mean(finite_draws >= 0)),
            ),
        )
    else:
        ci_low = ci_high = p_value = np.nan

    result: dict[str, object] = {
        "analysis": window.label,
        "category": category,
        "controls": "+".join(JOURNAL_DISPLAY[journal] for journal in controls),
        "pre_window": f"{window.pre_start:%Y-%m-%d} to {window.pre_end:%Y-%m-%d}",
        "post_window": f"{window.post_start:%Y-%m-%d} to {window.post_end:%Y-%m-%d}",
        "control_weighting": "Equal journal weighting",
        "control_change": control_change,
        "did": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
    }
    for journal in ["circ", "ehj", "jacc"]:
        pre_n = cell(journal, "Pre", "n")
        post_n = cell(journal, "Post", "n")
        result[f"{journal}_pre_n"] = int(pre_n) if np.isfinite(pre_n) else 0
        result[f"{journal}_pre_mean"] = cell(journal, "Pre", "mean")
        result[f"{journal}_pre_sd"] = cell(journal, "Pre", "sd")
        result[f"{journal}_post_n"] = int(post_n) if np.isfinite(post_n) else 0
        result[f"{journal}_post_mean"] = cell(journal, "Post", "mean")
        result[f"{journal}_post_sd"] = cell(journal, "Post", "sd")
        result[f"{journal}_change"] = changes.get(journal, np.nan)
    return result


def run_primary_did(data: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Run the primary raw DiD for every prespecified reporting category."""

    rows = []
    for index, category in enumerate(tqdm(CATEGORY_ORDER, desc="Primary raw DiD")):
        # Offset seeds keep strata reproducible without recycling bootstrap draws.
        rows.append(
            calculate_raw_did(
                data,
                PRIMARY_WINDOW,
                category=category,
                controls=("circ", "ehj"),
                n_boot=n_boot,
                seed=20260724 + index,
            )
        )
    return pd.DataFrame(rows)


def _mean_sd_text(mean: float, sd: float, n: int) -> str:
    if not np.isfinite(mean):
        return "NA"
    sd_text = "-" if not np.isfinite(sd) else f"{sd:.2f}"
    return f"{mean:.2f} ({sd_text}); n={n}"


def format_table_3(primary_did: pd.DataFrame) -> pd.DataFrame:
    """Format primary DiD results for the manuscript Excel table."""

    rows: list[dict[str, object]] = []
    for _, row in primary_did.iterrows():
        output: dict[str, object] = {"Category": row["category"]}
        for journal in ["jacc", "circ", "ehj"]:
            display = JOURNAL_DISPLAY[journal]
            output[f"{display} pre"] = _mean_sd_text(
                row[f"{journal}_pre_mean"],
                row[f"{journal}_pre_sd"],
                row[f"{journal}_pre_n"],
            )
            output[f"{display} post"] = _mean_sd_text(
                row[f"{journal}_post_mean"],
                row[f"{journal}_post_sd"],
                row[f"{journal}_post_n"],
            )
            output[f"{display} change"] = row[f"{journal}_change"]
        output["Equal-weight control change"] = row["control_change"]
        output["Raw DiD (95% CI)"] = (
            f"{row['did']:.2f} ({row['ci_low']:.2f}, {row['ci_high']:.2f})"
            if np.isfinite(row["did"])
            else "NA"
        )
        output["P value"] = format_p_value(row["p_value"])
        rows.append(output)
    return pd.DataFrame(rows)


def calculate_log_did(
    data: pd.DataFrame,
    window: DidWindow,
    category: str = "Overall",
    controls: Sequence[str] = ("circ", "ehj"),
) -> dict[str, object]:
    """Fit the log-scale fixed-effect DiD sensitivity model."""

    selected = select_did_window(data, window)
    selected = selected[category_mask(selected, category)].copy()
    selected = selected[selected["journal_code"].isin(["jacc", *controls])]
    selected["post"] = selected["period"].eq("Post").astype(int)
    # log1p keeps zero-citation papers and limits leverage from the citation tail.
    selected["log_c12"] = np.log1p(selected["c12_obs"])
    selected["month_cluster"] = selected["publication_month"].astype(str)
    # Month effects absorb common shocks; JACC-by-post is the differential change.
    model = smf.ols(
        "log_c12 ~ C(journal_code) + C(month_cluster) + is_jacc:post",
        data=selected,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": selected["month_cluster"], "use_correction": True},
        use_t=True,
    )
    term = "is_jacc:post"
    coefficient = float(model.params[term])
    ci_low, ci_high = model.conf_int().loc[term].astype(float)
    return {
        "analysis": window.label,
        "category": category,
        "controls": "+".join(JOURNAL_DISPLAY[journal] for journal in controls),
        "n": int(model.nobs),
        "log_coefficient": coefficient,
        "log_ci_low": float(ci_low),
        "log_ci_high": float(ci_high),
        "percent_change": float(np.expm1(coefficient) * 100),
        "percent_ci_low": float(np.expm1(ci_low) * 100),
        "percent_ci_high": float(np.expm1(ci_high) * 100),
        "p_value": float(model.pvalues[term]),
        "covariance": "Publication-month fixed effects and clustered SE",
    }


def run_robustness(
    data: pd.DataFrame,
    n_boot: int = 1500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run timing, article-type, control, and log-scale robustness analyses."""

    sensitivity_windows = make_sensitivity_windows()
    timing_specs = [
        ("Primary July cutoff", PRIMARY_WINDOW),
        ("June cutoff sensitivity", sensitivity_windows[0]),
        ("August cutoff sensitivity", sensitivity_windows[1]),
        ("June-August washout", sensitivity_windows[2]),
    ]
    raw_specs: list[tuple[str, DidWindow, str, Sequence[str]]] = []
    # Cross timing windows with strata, then isolate each control at the primary cutoff.
    for category in ["Overall", "ORA", "Review"]:
        raw_specs.extend(
            [
                (f"{label}, {category}", window, category, ("circ", "ehj"))
                for label, window in timing_specs
            ]
        )
    raw_specs.extend(
        [
            ("Primary July, Overall, CIRC only", PRIMARY_WINDOW, "Overall", ("circ",)),
            ("Primary July, Overall, EHJ only", PRIMARY_WINDOW, "Overall", ("ehj",)),
            ("Primary July, ORA, CIRC only", PRIMARY_WINDOW, "ORA", ("circ",)),
            ("Primary July, ORA, EHJ only", PRIMARY_WINDOW, "ORA", ("ehj",)),
        ]
    )
    raw_rows = []
    for index, (label, window, category, controls) in enumerate(
        tqdm(raw_specs, desc="Robustness raw DiD")
    ):
        result = calculate_raw_did(
            data,
            window,
            category=category,
            controls=controls,
            n_boot=n_boot,
            seed=20260800 + index,
        )
        result["display_label"] = label
        raw_rows.append(result)

    log_specs: list[tuple[str, DidWindow, str, Sequence[str]]] = []
    for category in ["Overall", "ORA", "Review"]:
        log_specs.extend(
            [
                (f"{label}, {category}", window, category, ("circ", "ehj"))
                for label, window in timing_specs
            ]
        )
    log_rows = []
    for label, window, category, controls in tqdm(log_specs, desc="Robustness log OLS"):
        result = calculate_log_did(data, window, category=category, controls=controls)
        result["display_label"] = label
        log_rows.append(result)
    return pd.DataFrame(raw_rows), pd.DataFrame(log_rows)


def run_placebo_did(data: pd.DataFrame, n_boot: int = 1500) -> pd.DataFrame:
    """Run the original July 2023 pre-intervention placebo specifications."""

    # A placebo signal means divergence predates treatment and weakens parallel trends.
    return pd.DataFrame(
        [
            calculate_raw_did(
                data,
                PLACEBO_WINDOW,
                category=category,
                controls=("circ", "ehj"),
                n_boot=n_boot,
                seed=20260900 + index,
            )
            for index, category in enumerate(["Overall", "ORA", "Review"])
        ]
    )
