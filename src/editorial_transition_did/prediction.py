from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

from .config import ATOMIC_CATEGORIES, CATEGORY_ORDER, HORIZONS, PREDICTOR_COLUMNS
from .utils import category_mask


def _prediction_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["landmark_log"] = np.log1p(output["c12_obs"].astype(float))
    output["publication_year_centered"] = output["publication_date"].dt.year - 2016
    month = output["publication_date"].dt.month.astype(float)
    output["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    output["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return output


def make_prediction_model(seed: int = 20260724) -> Pipeline:
    numeric = ["landmark_log", "publication_year_centered", "month_sin", "month_cos"]
    categorical = ["journal_code", "category"]
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    estimator = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _baseline_multiplier_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
) -> np.ndarray:
    train = train.copy()
    ratio = (train[outcome].astype(float) + 1.0) / (train["c12_obs"].astype(float) + 1.0)
    train["ratio"] = ratio
    group_ratio = train.groupby(["journal_code", "category"], observed=True)["ratio"].median()
    journal_ratio = train.groupby("journal_code", observed=True)["ratio"].median()
    global_ratio = float(train["ratio"].median())
    predictions = []
    for row in test[["journal_code", "category", "c12_obs"]].itertuples(index=False):
        multiplier = group_ratio.get((row.journal_code, row.category), np.nan)
        if not np.isfinite(multiplier):
            multiplier = journal_ratio.get(row.journal_code, global_ratio)
        predictions.append(max(0.0, (float(row.c12_obs) + 1.0) * float(multiplier) - 1.0))
    return np.asarray(predictions, dtype=float)


def _calibration_parameters(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    if len(observed) < 3 or np.nanstd(predicted) == 0:
        return np.nan, np.nan
    calibration = LinearRegression().fit(predicted.reshape(-1, 1), observed)
    return float(calibration.intercept_), float(calibration.coef_[0])


def prediction_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    interval80: tuple[np.ndarray, np.ndarray] | None = None,
    interval95: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    intercept, slope = _calibration_parameters(observed, predicted)
    metrics: dict[str, float | int] = {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "median_absolute_error": float(median_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
        "mean_bias": float(np.mean(predicted - observed)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "coverage_80": np.nan,
        "coverage_95": np.nan,
    }
    if interval80 is not None:
        metrics["coverage_80"] = float(
            np.mean((observed >= interval80[0]) & (observed <= interval80[1]))
        )
    if interval95 is not None:
        metrics["coverage_95"] = float(
            np.mean((observed >= interval95[0]) & (observed <= interval95[1]))
        )
    return metrics


def _residual_intervals(
    predicted: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, np.ndarray]:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return {
            "low80": np.full_like(predicted, np.nan),
            "high80": np.full_like(predicted, np.nan),
            "low95": np.full_like(predicted, np.nan),
            "high95": np.full_like(predicted, np.nan),
        }
    q025, q10, q90, q975 = np.quantile(residuals, [0.025, 0.10, 0.90, 0.975])
    return {
        "low80": np.maximum(0.0, predicted + q10),
        "high80": np.maximum(0.0, predicted + q90),
        "low95": np.maximum(0.0, predicted + q025),
        "high95": np.maximum(0.0, predicted + q975),
    }


def _eligible_prediction_data(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    frame = data[
        data["publication_date"].le(pd.Timestamp("2024-05-31"))
        & data["c12_obs"].notna()
        & data[outcome].notna()
        & data["category"].isin(ATOMIC_CATEGORIES)
    ].copy()
    frame = frame[frame[outcome].ge(0)]
    return _prediction_features(frame)


def fit_prediction_models(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    test_rows: list[pd.DataFrame] = []
    forecast_rows: list[pd.DataFrame] = []
    final_models: dict[str, object] = {}

    development_end = pd.Timestamp("2022-05-31")
    test_start = pd.Timestamp("2022-06-01")
    test_end = pd.Timestamp("2024-05-31")
    validation_starts = [pd.Timestamp(year, 6, 1) for year in range(2017, 2022)]

    for horizon_index, (horizon, definition) in enumerate(
        tqdm(HORIZONS.items(), desc="Temporal citation prediction")
    ):
        outcome = definition["increment"]
        eligible = _eligible_prediction_data(data, outcome)
        development = eligible[eligible["publication_date"].le(development_end)].copy()
        test = eligible[
            eligible["publication_date"].between(test_start, test_end, inclusive="both")
        ].copy()
        oof_parts: list[pd.DataFrame] = []
        selection_scores: list[dict[str, object]] = []
        for fold_index, validation_start in enumerate(validation_starts, start=1):
            validation_end = validation_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
            train_fold = development[development["publication_date"].lt(validation_start)]
            valid_fold = development[
                development["publication_date"].between(
                    validation_start, validation_end, inclusive="both"
                )
            ]
            if len(train_fold) < 1000 or len(valid_fold) < 100:
                continue
            model = make_prediction_model(seed=20260724 + horizon_index * 10 + fold_index)
            model.fit(train_fold[PREDICTOR_COLUMNS], train_fold[outcome])
            prediction = np.maximum(0.0, model.predict(valid_fold[PREDICTOR_COLUMNS]))
            baseline = _baseline_multiplier_predict(train_fold, valid_fold, outcome)
            observed = valid_fold[outcome].to_numpy(dtype=float)
            fold_model_metrics = prediction_metrics(observed, prediction)
            fold_baseline_metrics = prediction_metrics(observed, baseline)
            for model_name, values in [
                ("Poisson gradient boosting", fold_model_metrics),
                ("Historical multiplier baseline", fold_baseline_metrics),
            ]:
                selection_scores.append(
                    {"model": model_name, "mae": values["mae"], "rmse": values["rmse"]}
                )
                fold_rows.append(
                    {
                        "horizon": horizon,
                        "fold": fold_index,
                        "training_end": validation_start - pd.Timedelta(days=1),
                        "validation_start": validation_start,
                        "validation_end": validation_end,
                        "model": model_name,
                        **values,
                    }
                )
            part = valid_fold[["analysis_record_id"]].copy()
            part["observed"] = observed
            part["predicted"] = prediction
            part["residual"] = observed - prediction
            part["baseline_predicted"] = baseline
            part["baseline_residual"] = observed - baseline
            oof_parts.append(part)

        if not oof_parts:
            raise RuntimeError(
                f"No rolling temporal validation folds were available for {horizon}."
            )
        oof = pd.concat(oof_parts, ignore_index=True)
        selection = pd.DataFrame(selection_scores).groupby("model", observed=True)["mae"].mean()
        selected_model_name = str(selection.idxmin())
        model_residuals = oof["residual"].to_numpy(dtype=float)
        baseline_residuals = oof["baseline_residual"].to_numpy(dtype=float)

        development_model = make_prediction_model(seed=20260724 + horizon_index)
        development_model.fit(development[PREDICTOR_COLUMNS], development[outcome])
        test_prediction = np.maximum(0.0, development_model.predict(test[PREDICTOR_COLUMNS]))
        test_baseline = _baseline_multiplier_predict(development, test, outcome)
        model_intervals = _residual_intervals(test_prediction, model_residuals)
        baseline_intervals = _residual_intervals(test_baseline, baseline_residuals)
        observed = test[outcome].to_numpy(dtype=float)
        model_values = prediction_metrics(
            observed,
            test_prediction,
            interval80=(model_intervals["low80"], model_intervals["high80"]),
            interval95=(model_intervals["low95"], model_intervals["high95"]),
        )
        baseline_values = prediction_metrics(
            observed,
            test_baseline,
            interval80=(baseline_intervals["low80"], baseline_intervals["high80"]),
            interval95=(baseline_intervals["low95"], baseline_intervals["high95"]),
        )
        for model_name, values in [
            ("Poisson gradient boosting", model_values),
            ("Historical multiplier baseline", baseline_values),
        ]:
            metric_rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "development_period": "2008-01-01 to 2022-05-31",
                    "temporal_test_period": "2022-06-01 to 2024-05-31",
                    "selected_by_rolling_mae": model_name == selected_model_name,
                    **values,
                }
            )

        if selected_model_name == "Poisson gradient boosting":
            selected_test_prediction = test_prediction
            selected_test_intervals = model_intervals
            selected_residuals = model_residuals
        else:
            selected_test_prediction = test_baseline
            selected_test_intervals = baseline_intervals
            selected_residuals = baseline_residuals

        test_part = test[
            [
                "analysis_record_id",
                "doi_norm",
                "journal_code",
                "journal_display",
                "publication_date",
                "category",
                "c12_obs",
            ]
        ].copy()
        test_part["horizon"] = horizon
        test_part["selected_model"] = selected_model_name
        test_part["observed_increment"] = observed
        test_part["predicted_increment"] = selected_test_prediction
        test_part["gradient_boosting_increment"] = test_prediction
        test_part["baseline_increment"] = test_baseline
        for key, values in selected_test_intervals.items():
            test_part[f"increment_{key}"] = values
        test_rows.append(test_part)

        final_train = eligible.copy()
        final_model = make_prediction_model(seed=20260824 + horizon_index)
        final_model.fit(final_train[PREDICTOR_COLUMNS], final_train[outcome])
        final_models[horizon] = {
            "selected_model": selected_model_name,
            "poisson_gradient_boosting": final_model,
        }

        candidates = data[
            data["publication_date"].between(pd.Timestamp("2022-06-01"), pd.Timestamp("2025-06-30"))
            & data["c12_obs"].notna()
            & data["category"].isin(ATOMIC_CATEGORIES)
        ].copy()
        candidates = _prediction_features(candidates)
        gradient_prediction = np.maximum(0.0, final_model.predict(candidates[PREDICTOR_COLUMNS]))
        baseline_prediction = _baseline_multiplier_predict(final_train, candidates, outcome)
        predicted_increment = (
            gradient_prediction
            if selected_model_name == "Poisson gradient boosting"
            else baseline_prediction
        )
        forecast_intervals = _residual_intervals(predicted_increment, selected_residuals)
        forecast = candidates[
            [
                "analysis_record_id",
                "doi_norm",
                "journal_code",
                "journal_display",
                "publication_date",
                "publication_month",
                "category",
                "c12_obs",
            ]
        ].copy()
        forecast["horizon"] = horizon
        forecast["target_months"] = definition["months"]
        forecast["selected_model"] = selected_model_name
        forecast["observed_cumulative"] = candidates[definition["observed"]].to_numpy()
        forecast["predicted_increment"] = predicted_increment
        forecast["gradient_boosting_increment"] = gradient_prediction
        forecast["baseline_increment"] = baseline_prediction
        forecast["predicted_cumulative"] = forecast["c12_obs"] + predicted_increment
        for key, values in forecast_intervals.items():
            forecast[f"cumulative_{key}"] = forecast["c12_obs"].to_numpy(dtype=float) + values
        forecast_rows.append(forecast)

    metrics = pd.DataFrame(metric_rows)
    folds = pd.DataFrame(fold_rows)
    test_predictions = pd.concat(test_rows, ignore_index=True)
    forecasts = pd.concat(forecast_rows, ignore_index=True)
    return metrics, folds, test_predictions, forecasts, final_models


def format_table_4(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon",
        "model",
        "selected_by_rolling_mae",
        "n",
        "mae",
        "rmse",
        "median_absolute_error",
        "r2",
        "mean_bias",
        "calibration_intercept",
        "calibration_slope",
        "coverage_80",
        "coverage_95",
    ]
    output = metrics[columns].copy()
    output = output.rename(
        columns={
            "horizon": "Prediction horizon",
            "model": "Model",
            "selected_by_rolling_mae": "Selected",
            "n": "Test n",
            "mae": "MAE",
            "rmse": "RMSE",
            "median_absolute_error": "Median AE",
            "r2": "R-squared",
            "mean_bias": "Mean bias",
            "calibration_intercept": "Calibration intercept",
            "calibration_slope": "Calibration slope",
            "coverage_80": "80% coverage",
            "coverage_95": "95% coverage",
        }
    )
    return output


def aggregate_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    post = forecasts[
        forecasts["publication_date"].between(
            pd.Timestamp("2024-06-01"), pd.Timestamp("2025-05-31")
        )
    ].copy()
    rows: list[dict[str, object]] = []
    for (journal, horizon), group in post.groupby(["journal_display", "horizon"], observed=True):
        for category in CATEGORY_ORDER:
            subset = group if category == "Overall" else group[category_mask(group, category)]
            observed = subset["observed_cumulative"].dropna()
            rows.append(
                {
                    "journal": journal,
                    "publication_cohort": "2024.6-2025.5",
                    "category": category,
                    "target": f"C{int(subset['target_months'].iloc[0])}"
                    if len(subset)
                    else horizon,
                    "articles": int(len(subset)),
                    "predicted_mean": float(subset["predicted_cumulative"].mean())
                    if len(subset)
                    else np.nan,
                    "predicted_sd": float(subset["predicted_cumulative"].std())
                    if len(subset) > 1
                    else np.nan,
                    "mean_lower_80_bound": float(subset["cumulative_low80"].mean())
                    if len(subset)
                    else np.nan,
                    "mean_upper_80_bound": float(subset["cumulative_high80"].mean())
                    if len(subset)
                    else np.nan,
                    "observed_complete_n": int(len(observed)),
                    "observed_mean_if_complete": float(observed.mean())
                    if len(observed)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)
