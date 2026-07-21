"""Core Monte Carlo performance metrics for raw estimator results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


RAW_RESULT_REQUIRED_COLUMNS = {
    "alpha_hat",
    "alpha_true",
    "converged",
    "cr_length",
}


def validate_metric_input(df: pd.DataFrame) -> None:
    """Validate the minimal raw-result columns needed for metric computation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("metric input must be a pandas DataFrame")

    if df.columns.duplicated().any():
        duplicated = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
        raise ValueError(f"metric input has duplicate columns: {duplicated}")

    missing = sorted(RAW_RESULT_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"metric input is missing required columns: {missing}")
    if "covered" not in df.columns and "cr_covers_true" not in df.columns:
        raise ValueError("metric input must contain covered or cr_covers_true")


def _coverage_column(df: pd.DataFrame) -> str:
    return "covered" if "covered" in df.columns else "cr_covers_true"


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_nonnegative_numeric(series: pd.Series) -> pd.Series:
    values = _to_numeric(series)
    values = values.where(np.isfinite(values))
    return values.where(values >= 0)


def _to_bool(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes"}
    false_values = {"false", "0", "no"}

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(value)):
                return pd.NA
            if float(value) == 1.0:
                return True
            if float(value) == 0.0:
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


def _mean_nonnegative_numeric(series: pd.Series) -> float:
    values = _to_numeric(series).dropna()
    values = values.loc[np.isfinite(values)]
    values = values.loc[values >= 0]
    if values.empty:
        return float(np.nan)
    return float(values.mean())


def _mean_bool(series: pd.Series) -> float:
    values = _to_bool(series).dropna()
    if values.empty:
        return float(np.nan)
    return float(values.astype(float).mean())


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    validate_metric_input(df)
    if "status" in df.columns and "failed" in df.columns:
        status_ok = df["status"].astype(str).str.strip().str.lower() == "ok"
        failed_mask = _to_bool(df["failed"]).fillna(False)
        return df.loc[status_ok & ~failed_mask]
    if "failed" in df.columns:
        return df.loc[~_to_bool(df["failed"]).fillna(False)]
    return df


def estimation_errors(df: pd.DataFrame) -> pd.Series:
    """Return alpha_hat - alpha_true for successful rows with numeric inputs.

    With a status column, successful rows require a stripped, case-insensitive
    status of ``"ok"`` and a failed flag that is not true. Without status,
    successful rows are those whose failed flag is not true.
    """
    successful = _successful_rows(df)
    alpha_hat = _to_numeric(successful["alpha_hat"])
    alpha_true = _to_numeric(successful["alpha_true"])
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
    """Return coverage conditional on explicitly resolved successful rows."""
    return coverage_valid_only(df)


def coverage_valid_only(df: pd.DataFrame) -> float:
    """Return coverage over successful rows with nonmissing coverage values."""
    successful = _successful_rows(df)
    if "coverage_status" in successful.columns:
        successful = successful.loc[
            successful["coverage_status"].isin(["covered", "not_covered"])
        ]
    return _mean_bool(successful[_coverage_column(successful)])


def coverage_unresolved_rate(df: pd.DataFrame) -> float:
    """Return unresolved-coverage rows as a share of all replications."""
    validate_metric_input(df)
    if "coverage_status" not in df.columns or df.empty:
        return float(np.nan)
    return float(df["coverage_status"].eq("coverage_unresolved").mean())


def average_cr_length_valid_only(df: pd.DataFrame) -> float:
    """Return mean valid nonnegative CR length over successful rows."""
    successful = _successful_rows(df)
    values = _to_nonnegative_numeric(successful["cr_length"]).dropna()
    if values.empty:
        return float(np.nan)
    return float(values.mean())


def average_cr_length_all(df: pd.DataFrame) -> float:
    """Return mean CR length over successful rows, using zero when invalid."""
    successful = _successful_rows(df)
    if successful.empty:
        return float(np.nan)
    values = _to_nonnegative_numeric(successful["cr_length"])
    return float(values.fillna(0.0).mean())


def average_cr_length(df: pd.DataFrame) -> float:
    return average_cr_length_valid_only(df)


def average_cr_hull_length(df: pd.DataFrame) -> float:
    """Return mean valid nonnegative CR hull length over successful rows."""
    validate_metric_input(df)
    if "cr_hull_length" not in df.columns:
        return float(np.nan)
    successful = _successful_rows(df)
    values = _to_nonnegative_numeric(successful["cr_hull_length"]).dropna()
    if values.empty:
        return float(np.nan)
    return float(values.mean())


def failure_rate(df: pd.DataFrame) -> float:
    """Return the mean failed indicator across all rows."""
    validate_metric_input(df)
    if "failed" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["failed"])


def non_convergence_rate(df: pd.DataFrame) -> float:
    """Return one minus the mean converged indicator across all rows."""
    validate_metric_input(df)
    converged = _mean_bool(df["converged"])
    if np.isnan(converged):
        return float(np.nan)
    return float(1.0 - converged)


def cr_empty_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "cr_empty" in df.columns:
        return _mean_bool(df["cr_empty"])
    if {"cr_lower", "cr_upper", "cr_length"}.issubset(df.columns):
        empty = df[["cr_lower", "cr_upper", "cr_length"]].isna().all(axis=1)
        return float(empty.mean())
    return float(np.nan)


def cr_disconnected_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "cr_disconnected" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["cr_disconnected"])


def boundary_rate(df: pd.DataFrame) -> float:
    """Backward-compatible point-estimate boundary rate.

    Prefer the current alpha-hat diagnostic and fall back to the legacy
    at-grid-boundary column for old result files.
    """
    validate_metric_input(df)
    if "alpha_hat_at_any_boundary" in df.columns:
        return _mean_bool(df["alpha_hat_at_any_boundary"])
    if "at_grid_boundary" in df.columns:
        return _mean_bool(df["at_grid_boundary"])
    return float(np.nan)


def alpha_hat_boundary_rate(df: pd.DataFrame) -> float:
    """Return share of rows where alpha_hat lies at the alpha-grid boundary."""
    validate_metric_input(df)
    if "alpha_hat_at_any_boundary" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["alpha_hat_at_any_boundary"])


def cr_boundary_hit_rate(df: pd.DataFrame) -> float:
    """Return share of rows where the confidence region touches a grid boundary."""
    validate_metric_input(df)
    if "cr_hits_any_boundary" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["cr_hits_any_boundary"])


def cr_lower_boundary_hit_rate(df: pd.DataFrame) -> float:
    """Return share of rows where the confidence region touches the lower boundary."""
    validate_metric_input(df)
    if "cr_hits_lower_boundary" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["cr_hits_lower_boundary"])


def cr_upper_boundary_hit_rate(df: pd.DataFrame) -> float:
    """Return share of rows where the confidence region touches the upper boundary."""
    validate_metric_input(df)
    if "cr_hits_upper_boundary" not in df.columns:
        return float(np.nan)
    return _mean_bool(df["cr_hits_upper_boundary"])


def mean_failed_alpha_count(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "failed_alpha_count" not in df.columns:
        return float(np.nan)
    return _mean_nonnegative_numeric(df["failed_alpha_count"])


def mean_failed_alpha_rate(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "failed_alpha_rate" not in df.columns:
        return float(np.nan)
    return _mean_nonnegative_numeric(df["failed_alpha_rate"])


def mean_selected_controls(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    for column in (
        "n_selected_controls",
        "selected_controls",
        "ps_n_selected_controls",
    ):
        if column in df.columns:
            return _mean_nonnegative_numeric(df[column])
    return float(np.nan)


def critical_value_multiplier(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "critical_value_multiplier" not in df.columns:
        return float(np.nan)
    values = _to_nonnegative_numeric(df["critical_value_multiplier"]).dropna()
    if values.empty:
        return float(np.nan)
    unique_values = values.unique()
    if unique_values.size == 1:
        return float(unique_values[0])
    return float(values.mean())


def selection_lasso_multiplier(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    column = next(
        (
            candidate
            for candidate in (
                "selection_lasso_multiplier",
                "ps_selection_lasso_multiplier",
            )
            if candidate in df.columns
        ),
        None,
    )
    if column is None:
        return float(np.nan)
    values = _to_nonnegative_numeric(df[column]).dropna()
    if values.empty:
        return float(np.nan)
    unique_values = values.unique()
    if unique_values.size == 1:
        return float(unique_values[0])
    return float(values.mean())


def ps_selection_lasso_multiplier(df: pd.DataFrame) -> float:
    """Backward-compatible alias for selection_lasso_multiplier."""
    return selection_lasso_multiplier(df)


def mean_critical_value_adjusted(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "critical_value_adjusted" in df.columns:
        return _mean_nonnegative_numeric(df["critical_value_adjusted"])
    if "critical_value" in df.columns:
        return _mean_nonnegative_numeric(df["critical_value"])
    return float(np.nan)


def mean_runtime_seconds(df: pd.DataFrame) -> float:
    validate_metric_input(df)
    if "runtime_seconds" not in df.columns:
        return float(np.nan)
    return _mean_nonnegative_numeric(df["runtime_seconds"])


def _mean_optional_nonnegative(df: pd.DataFrame, column: str) -> float:
    validate_metric_input(df)
    if column not in df.columns:
        return float(np.nan)
    return _mean_nonnegative_numeric(df[column])


def _median_optional_nonnegative(df: pd.DataFrame, column: str) -> float:
    validate_metric_input(df)
    if column not in df.columns:
        return float(np.nan)
    values = _to_nonnegative_numeric(df[column]).dropna()
    if values.empty:
        return float(np.nan)
    return float(values.median())


def mean_runtime_total_sec(df: pd.DataFrame) -> float:
    return _mean_optional_nonnegative(df, "runtime_total_sec")


def median_runtime_total_sec(df: pd.DataFrame) -> float:
    return _median_optional_nonnegative(df, "runtime_total_sec")


def mean_runtime_alpha_grid_sec(df: pd.DataFrame) -> float:
    return _mean_optional_nonnegative(df, "runtime_alpha_grid_sec")


def mean_runtime_confidence_region_sec(df: pd.DataFrame) -> float:
    return _mean_optional_nonnegative(df, "runtime_confidence_region_sec")


def mean_dml_runtime_crossfit_sec(df: pd.DataFrame) -> float:
    return _mean_optional_nonnegative(df, "dml_runtime_crossfit_sec")


def mean_ps_runtime_selection_sec(df: pd.DataFrame) -> float:
    return _mean_optional_nonnegative(df, "ps_runtime_selection_sec")


def summarize_group(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize one group using successful rows for estimation and CR metrics.

    Failure, convergence, empty-region, and diagnostic rates use all rows.
    """
    validate_metric_input(df)
    successful = _successful_rows(df)
    alpha_hat = _to_numeric(successful["alpha_hat"])
    alpha_true = _to_numeric(successful["alpha_true"])
    valid_estimates = int((alpha_hat.notna() & alpha_true.notna()).sum())
    return {
        "replications": int(len(df)),
        "valid_estimates": valid_estimates,
        "bias": bias(df),
        "median_bias": median_bias(df),
        "rmse": rmse(df),
        "mae": mae(df),
        "coverage": coverage(df),
        "coverage_valid_only": coverage_valid_only(df),
        "coverage_conditional_on_resolved": coverage_valid_only(df),
        "coverage_unresolved_rate": coverage_unresolved_rate(df),
        "avg_cr_length": average_cr_length_all(df),
        "avg_cr_length_valid_only": average_cr_length_valid_only(df),
        "avg_cr_hull_length": average_cr_hull_length(df),
        "failure_rate": failure_rate(df),
        "non_convergence_rate": non_convergence_rate(df),
        "cr_empty_rate": cr_empty_rate(df),
        "cr_disconnected_rate": cr_disconnected_rate(df),
        "boundary_rate": boundary_rate(df),
        "alpha_hat_boundary_rate": alpha_hat_boundary_rate(df),
        "cr_boundary_hit_rate": cr_boundary_hit_rate(df),
        "mean_failed_alpha_count": mean_failed_alpha_count(df),
        "mean_failed_alpha_rate": mean_failed_alpha_rate(df),
        "mean_selected_controls": mean_selected_controls(df),
        "critical_value_multiplier": critical_value_multiplier(df),
        "selection_lasso_multiplier": selection_lasso_multiplier(df),
        "mean_critical_value_adjusted": mean_critical_value_adjusted(df),
        "mean_runtime_seconds": mean_runtime_seconds(df),
        "mean_runtime_total_sec": mean_runtime_total_sec(df),
        "median_runtime_total_sec": median_runtime_total_sec(df),
        "mean_runtime_alpha_grid_sec": mean_runtime_alpha_grid_sec(df),
        "mean_runtime_confidence_region_sec": mean_runtime_confidence_region_sec(df),
        "mean_dml_runtime_crossfit_sec": mean_dml_runtime_crossfit_sec(df),
        "mean_ps_runtime_selection_sec": mean_ps_runtime_selection_sec(df),
    }


__all__ = [
    "RAW_RESULT_REQUIRED_COLUMNS",
    "validate_metric_input",
    "estimation_errors",
    "bias",
    "median_bias",
    "rmse",
    "mae",
    "coverage",
    "coverage_valid_only",
    "average_cr_length_valid_only",
    "average_cr_length_all",
    "average_cr_length",
    "average_cr_hull_length",
    "failure_rate",
    "non_convergence_rate",
    "cr_empty_rate",
    "cr_disconnected_rate",
    "boundary_rate",
    "alpha_hat_boundary_rate",
    "cr_boundary_hit_rate",
    "cr_lower_boundary_hit_rate",
    "cr_upper_boundary_hit_rate",
    "mean_failed_alpha_count",
    "mean_failed_alpha_rate",
    "mean_selected_controls",
    "critical_value_multiplier",
    "selection_lasso_multiplier",
    "ps_selection_lasso_multiplier",
    "mean_critical_value_adjusted",
    "mean_runtime_seconds",
    "mean_runtime_total_sec",
    "median_runtime_total_sec",
    "mean_runtime_alpha_grid_sec",
    "mean_runtime_confidence_region_sec",
    "mean_dml_runtime_crossfit_sec",
    "mean_ps_runtime_selection_sec",
    "summarize_group",
]

