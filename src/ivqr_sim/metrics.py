"""Core Monte Carlo performance metrics for raw estimator results."""

from __future__ import annotations

import numpy as np
import pandas as pd


RAW_RESULT_REQUIRED_COLUMNS = {
    "alpha_hat",
    "alpha_true",
    "failed",
    "converged",
    "cr_length",
    "cr_covers_true",
    "cr_empty",
    "runtime_seconds",
}


def validate_metric_input(df: pd.DataFrame) -> None:
    """Validate the minimal raw-result columns needed for metric computation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("metric input must be a pandas DataFrame")

    missing = sorted(RAW_RESULT_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"metric input is missing required columns: {missing}")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes"}
    false_values = {"false", "0", "no"}

    def convert(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
            return pd.NA
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in true_values:
                return True
            if normalized in false_values:
                return False
        return pd.NA

    return series.map(convert).astype("boolean")


def _mean_numeric(series: pd.Series) -> float:
    values = _to_numeric(series).dropna()
    if values.empty:
        return float(np.nan)
    return float(values.mean())


def _mean_bool(series: pd.Series) -> float:
    values = _to_bool(series).dropna()
    if values.empty:
        return float(np.nan)
    return float(values.astype(float).mean())


def estimation_errors(df: pd.DataFrame) -> pd.Series:
    """Return alpha_hat - alpha_true, recomputed from raw columns."""
    validate_metric_input(df)
    alpha_hat = _to_numeric(df["alpha_hat"])
    alpha_true = _to_numeric(df["alpha_true"])
    return alpha_hat - alpha_true


def bias(df: pd.DataFrame) -> float:
    errors = estimation_errors(df).dropna()
    if errors.empty:
        return float(np.nan)
    return float(errors.mean())


def median_bias(df: pd.DataFrame) -> float:
    errors = estimation_errors(df).dropna()
    if errors.empty:
        return float(np.nan)
    return float(errors.median())


def rmse(df: pd.DataFrame) -> float:
    errors = estimation_errors(df).dropna()
    if errors.empty:
        return float(np.nan)
    return float(np.sqrt((errors**2).mean()))


def mae(df: pd.DataFrame) -> float:
    errors = estimation_errors(df).dropna()
    if errors.empty:
        return float(np.nan)
    return float(errors.abs().mean())


def coverage(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    return _mean_bool(df["cr_covers_true"])


def average_cr_length(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    return _mean_numeric(df["cr_length"])


def failure_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    return _mean_bool(df["failed"])


def non_convergence_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    converged = _mean_bool(df["converged"])
    if np.isnan(converged):
        return float(np.nan)
    return float(1.0 - converged)


def cr_empty_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    return _mean_bool(df["cr_empty"])


def cr_disconnected_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "cr_disconnected" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["cr_disconnected"])


def boundary_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "at_grid_boundary" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["at_grid_boundary"])


def mean_failed_alpha_count(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "failed_alpha_count" not in df.columns:
        return float(np.nan)
    return _mean_numeric(df["failed_alpha_count"])


def mean_selected_controls(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "selected_controls" not in df.columns:
        return float(np.nan)
    return _mean_numeric(df["selected_controls"])


def mean_runtime_seconds(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    return _mean_numeric(df["runtime_seconds"])


def summarize_group(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize one already-filtered Monte Carlo result group."""
    validate_metric_input(df)
    valid_estimates = int(_to_numeric(df["alpha_hat"]).notna().sum())
    return {
        "replications": int(len(df)),
        "valid_estimates": valid_estimates,
        "bias": bias(df),
        "median_bias": median_bias(df),
        "rmse": rmse(df),
        "mae": mae(df),
        "coverage": coverage(df),
        "avg_cr_length": average_cr_length(df),
        "failure_rate": failure_rate(df),
        "non_convergence_rate": non_convergence_rate(df),
        "cr_empty_rate": cr_empty_rate(df),
        "cr_disconnected_rate": cr_disconnected_rate(df),
        "boundary_rate": boundary_rate(df),
        "mean_failed_alpha_count": mean_failed_alpha_count(df),
        "mean_selected_controls": mean_selected_controls(df),
        "mean_runtime_seconds": mean_runtime_seconds(df),
    }
