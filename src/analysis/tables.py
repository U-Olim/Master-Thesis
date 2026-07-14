"""Thesis-focused Monte Carlo performance tables."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from analysis.data import require_unique_selection_lasso_multiplier


ESTIMATOR_NAMES = {
    "oracle": "Oracle IVQR",
    "post_selection": "Post-selection IVQR",
    "dml": "DML-IVQR",
}
ESTIMATOR_ORDER = list(ESTIMATOR_NAMES)
PERFORMANCE_COLUMNS = [
    "mean_estimate",
    "bias",
    "abs_bias",
    "mae",
    "rmse",
    "estimate_sd",
    "coverage",
    "average_cr_length",
    "median_cr_length",
    "valid_rate",
    "n_results",
]


def summarize_performance(
    results: pd.DataFrame, group_by: Sequence[str]
) -> pd.DataFrame:
    """Compute common Monte Carlo metrics for explicit groups."""
    required = {
        *group_by,
        "alpha_hat",
        "alpha_true",
        "covered",
        "cr_length",
        "converged",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(
            f"Cannot calculate metrics; missing columns: {sorted(missing)}"
        )

    records: list[dict[str, object]] = []
    grouper: str | list[str] = list(group_by)
    if len(group_by) == 1:
        grouper = group_by[0]
    for keys, group in results.groupby(grouper, dropna=False, sort=True):
        if len(group_by) == 1 or not isinstance(keys, tuple):
            key_values: tuple[object, ...] = (keys,)
        else:
            key_values = keys
        estimates = group["alpha_hat"].to_numpy(dtype=float)
        truth = group["alpha_true"].to_numpy(dtype=float)
        errors = estimates - truth
        lengths = group["cr_length"].to_numpy(dtype=float)
        coverage_values = group["covered"].to_numpy(dtype=float)
        valid_values = group["converged"].to_numpy(dtype=float)
        bias = float(np.mean(errors))
        record: dict[str, object] = {}
        for column, value in zip(group_by, key_values, strict=True):
            record[column] = value
        record.update(
            mean_estimate=float(np.mean(estimates)),
            bias=bias,
            abs_bias=abs(bias),
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(np.square(errors)))),
            estimate_sd=(
                float(np.std(estimates, ddof=1)) if len(group) > 1 else float("nan")
            ),
            coverage=float(np.mean(coverage_values)),
            average_cr_length=float(np.nanmean(lengths)),
            median_cr_length=float(np.nanmedian(lengths)),
            valid_rate=float(np.mean(valid_values)),
            n_results=int(len(group)),
        )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=[*group_by, *PERFORMANCE_COLUMNS])


def _display_estimators(table: pd.DataFrame) -> pd.DataFrame:
    displayed = table.copy()
    displayed["estimator"] = displayed["estimator"].map(ESTIMATOR_NAMES)
    order = list(ESTIMATOR_NAMES.values())
    displayed["estimator"] = pd.Categorical(
        displayed["estimator"], categories=order, ordered=True
    )
    sort_columns = [
        column for column in displayed.columns if column not in PERFORMANCE_COLUMNS
    ]
    displayed = displayed.sort_values(sort_columns).reset_index(drop=True)
    displayed["estimator"] = displayed["estimator"].astype(str)
    return displayed


def make_main_performance_table(results: pd.DataFrame) -> pd.DataFrame:
    return _display_estimators(summarize_performance(results, ["estimator"]))


def make_performance_by_strength_table(results: pd.DataFrame) -> pd.DataFrame:
    return _display_estimators(summarize_performance(results, ["pi", "estimator"]))


def make_performance_by_quantile_table(results: pd.DataFrame) -> pd.DataFrame:
    return _display_estimators(summarize_performance(results, ["tau", "estimator"]))


def make_performance_by_dgp_table(results: pd.DataFrame) -> pd.DataFrame:
    return _display_estimators(summarize_performance(results, ["dgp", "estimator"]))


def make_performance_by_design_cell_table(results: pd.DataFrame) -> pd.DataFrame:
    return _display_estimators(
        summarize_performance(results, ["dgp", "n", "p", "pi", "tau", "estimator"])
    )


def make_post_selection_table(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.loc[results["estimator"].eq("post_selection")]
    if selected.empty or "n_selected_controls" not in selected:
        raise ValueError("Post-selection diagnostics are unavailable")
    diagnostics = (
        selected.groupby(["dgp", "pi"], sort=True)["n_selected_controls"]
        .agg(
            n_results="size",
            mean_selected_controls="mean",
            median_selected_controls="median",
            min_selected_controls="min",
            max_selected_controls="max",
        )
        .reset_index()
    )
    diagnostics["selection_lasso_multiplier"] = (
        require_unique_selection_lasso_multiplier(
            selected["selection_lasso_multiplier"]
        )
    )
    return diagnostics


def _write_table(table: pd.DataFrame, output_dir: Path, name: str) -> list[Path]:
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    table.to_csv(csv_path, index=False, float_format="%.6f")
    _write_latex(table, tex_path)
    return [csv_path, tex_path]


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _latex_text(value: object) -> str:
    if _is_missing_scalar(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex(table: pd.DataFrame, path: Path) -> None:
    """Write a dependency-free LaTeX tabular with stable numeric formatting."""
    alignment = "".join(
        "r" if pd.api.types.is_numeric_dtype(table[column]) else "l"
        for column in table.columns
    )
    rows = [f"\\begin{{tabular}}{{{alignment}}}", r"\hline"]
    rows.append(" & ".join(_latex_text(column) for column in table.columns) + r" \\")
    rows.append(r"\hline")
    rows.extend(
        " & ".join(_latex_text(value) for value in row) + r" \\"
        for row in table.itertuples(index=False, name=None)
    )
    rows.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(rows), encoding="utf-8")


def write_all_tables(
    results: pd.DataFrame, output_dir: str | Path
) -> dict[str, list[Path]]:
    """Write the six final tables as CSV and LaTeX."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "overall_performance": make_main_performance_table(results),
        "performance_by_strength": make_performance_by_strength_table(results),
        "performance_by_quantile": make_performance_by_quantile_table(results),
        "performance_by_dgp": make_performance_by_dgp_table(results),
        "performance_by_design_cell": make_performance_by_design_cell_table(results),
        "post_selection_diagnostics": make_post_selection_table(results),
    }
    return {
        name: _write_table(table, destination, name) for name, table in tables.items()
    }
