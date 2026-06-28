"""Tests for Monte Carlo performance metrics."""

import numpy as np
import pandas as pd
import pytest

from inference.metrics import (
    average_cr_length,
    average_cr_length_all,
    average_cr_length_valid_only,
    alpha_hat_boundary_rate,
    bias,
    boundary_rate,
    coverage,
    coverage_valid_only,
    cr_boundary_hit_rate,
    cr_disconnected_rate,
    cr_empty_rate,
    estimation_errors,
    failure_rate,
    mae,
    mean_failed_alpha_count,
    mean_runtime_seconds,
    mean_selected_controls,
    median_bias,
    non_convergence_rate,
    rmse,
    summarize_group,
    validate_metric_input,
)


def _base_df(**overrides) -> pd.DataFrame:
    n_rows = max([3] + [len(value) for value in overrides.values()])

    def repeat(values: list[object]) -> list[object]:
        if len(values) >= n_rows:
            return values
        return values + [values[-1]] * (n_rows - len(values))

    data = {
        "alpha_hat": repeat([1.1, 0.9, 1.3]),
        "alpha_true": repeat([1.0, 1.0, 1.0]),
        "failed": repeat([False, False, False]),
        "converged": repeat([True, True, True]),
        "cr_length": repeat([1.0, 2.0, 3.0]),
        "cr_covers_true": repeat([True, False, True]),
        "cr_empty": repeat([False, False, False]),
        "runtime_seconds": repeat([0.1, 0.2, 0.3]),
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_validate_metric_input_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_metric_input(pd.DataFrame({"alpha_hat": [1.0]}))


def test_validate_metric_input_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_metric_input([])  # type: ignore[arg-type]


def test_validate_metric_input_rejects_duplicate_columns() -> None:
    df = _base_df()
    duplicated = pd.concat([df, df[["alpha_hat"]]], axis=1)

    with pytest.raises(ValueError, match="duplicate columns.*alpha_hat"):
        validate_metric_input(duplicated)


def test_validate_metric_input_accepts_empty_dataframe() -> None:
    validate_metric_input(_base_df().iloc[0:0])


def test_estimation_errors_recompute_from_alpha_hat_and_alpha_true() -> None:
    df = _base_df(alpha_hat=[1.1, 0.9, None], alpha_true=[1.0, 1.0, 1.0])

    errors = estimation_errors(df)

    assert errors.iloc[0] == pytest.approx(0.1)
    assert errors.iloc[1] == pytest.approx(-0.1)
    assert np.isnan(errors.iloc[2])


def test_bias_and_median_bias() -> None:
    df = _base_df(alpha_hat=[1.1, 0.9, 1.3], alpha_true=[1.0, 1.0, 1.0])

    assert bias(df) == pytest.approx(0.1)
    assert median_bias(df) == pytest.approx(0.1)


def test_rmse_and_mae() -> None:
    df = _base_df(alpha_hat=[2.0, 0.0, 3.0], alpha_true=[1.0, 1.0, 1.0])

    assert rmse(df) == pytest.approx(np.sqrt((1 + 1 + 4) / 3))
    assert mae(df) == pytest.approx(4 / 3)


def test_coverage_parses_bool_and_string_values() -> None:
    df = _base_df(cr_covers_true=[True, "False", "true", None])

    assert coverage(df) == pytest.approx(2 / 4)
    assert coverage_valid_only(df) == pytest.approx(2 / 3)


def test_boolean_metrics_parse_supported_scalar_representations() -> None:
    values = [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        1,
        0,
        np.int64(1),
        np.int64(0),
        1.0,
        0.0,
        np.float64(1.0),
        np.float64(0.0),
        "true",
        "false",
        "True",
        "False",
        "yes",
        "no",
        "1",
        "0",
    ]
    df = _base_df(failed=values)

    assert failure_rate(df) == pytest.approx(0.5)


def test_boolean_metrics_treat_invalid_values_as_missing() -> None:
    df = _base_df(
        cr_empty=[True, False, 2, -1, np.nan, np.inf, "maybe", "ok", "failed"],
    )

    assert cr_empty_rate(df) == pytest.approx(0.5)


def test_coverage_counts_all_missing_as_noncoverage() -> None:
    df = _base_df(cr_covers_true=[None, None, None])

    assert coverage(df) == pytest.approx(0.0)
    assert np.isnan(coverage_valid_only(df))


def test_coverage_empty_dataframe_returns_nan() -> None:
    df = _base_df().iloc[0:0]

    assert np.isnan(coverage(df))
    assert np.isnan(coverage_valid_only(df))


def test_average_cr_length_ignores_missing_values() -> None:
    df = _base_df(cr_length=[1.0, 2.0, None])

    assert average_cr_length(df) == pytest.approx(1.5)


def test_average_cr_length_all_uses_all_rows_denominator() -> None:
    df = _base_df(cr_length=[1.0, None, 3.0])

    assert average_cr_length_all(df) == pytest.approx(4.0 / 3)


def test_average_cr_length_valid_only_ignores_missing_lengths() -> None:
    df = _base_df(cr_length=[1.0, None, 3.0])

    assert average_cr_length_valid_only(df) == pytest.approx(2.0)


def test_average_cr_lengths_handle_negative_and_infinite_values() -> None:
    df = _base_df(cr_length=[1.0, -5.0, np.inf, 3.0])

    assert average_cr_length_valid_only(df) == pytest.approx(2.0)
    assert average_cr_length_all(df) == pytest.approx(1.0)


def test_average_cr_length_valid_only_returns_nan_when_all_invalid() -> None:
    df = _base_df(cr_length=[-5.0, np.inf, np.nan])

    assert np.isnan(average_cr_length_valid_only(df))
    assert average_cr_length_all(df) == pytest.approx(0.0)


def test_failure_and_non_convergence_rates() -> None:
    df = _base_df(
        failed=[False, True, False],
        converged=[True, False, True],
    )

    assert failure_rate(df) == pytest.approx(1 / 3)
    assert non_convergence_rate(df) == pytest.approx(1 / 3)


def test_successful_rows_require_ok_status_and_not_failed() -> None:
    df = _base_df(
        alpha_hat=[1.0, 100.0, 200.0, 300.0],
        alpha_true=[0.0, 0.0, 0.0, 0.0],
        status=["ok", "ok", "failed", "failed"],
        failed=[False, True, False, True],
    )

    assert estimation_errors(df).tolist() == [1.0]
    assert bias(df) == pytest.approx(1.0)


def test_successful_rows_without_status_use_failed_flag() -> None:
    df = _base_df(
        alpha_hat=[1.0, 100.0, 3.0],
        alpha_true=[0.0, 0.0, 0.0],
        failed=[False, True, False],
    )

    assert estimation_errors(df).tolist() == [1.0, 3.0]


def test_successful_rows_strip_status_whitespace() -> None:
    df = _base_df(
        alpha_hat=[1.0, 2.0, 100.0],
        alpha_true=[0.0, 0.0, 0.0],
        status=[" ok ", " OK ", " failed "],
        failed=[False, False, False],
    )

    assert estimation_errors(df).tolist() == [1.0, 2.0]


def test_empty_and_disconnected_rates_parse_mixed_bool_values() -> None:
    df = _base_df(
        cr_empty=[False, "true", "0", None],
        cr_disconnected=[True, "False", "yes", None],
    )

    assert cr_empty_rate(df) == pytest.approx(1 / 3)
    assert cr_disconnected_rate(df) == pytest.approx(2 / 3)


def test_optional_missing_diagnostic_columns_return_nan() -> None:
    df = _base_df()

    assert np.isnan(cr_disconnected_rate(df))
    assert np.isnan(boundary_rate(df))
    assert np.isnan(alpha_hat_boundary_rate(df))
    assert np.isnan(cr_boundary_hit_rate(df))
    assert np.isnan(mean_failed_alpha_count(df))
    assert np.isnan(mean_selected_controls(df))


def test_boundary_rate_uses_current_alpha_hat_boundary_column() -> None:
    df = _base_df(
        alpha_hat_at_any_boundary=[True, False, True],
        at_grid_boundary=[False, False, False],
    )

    assert boundary_rate(df) == pytest.approx(2 / 3)
    assert alpha_hat_boundary_rate(df) == pytest.approx(2 / 3)


def test_boundary_rate_falls_back_to_legacy_at_grid_boundary_column() -> None:
    df = _base_df().iloc[:2].copy()
    df["at_grid_boundary"] = [False, True]

    assert boundary_rate(df) == pytest.approx(1 / 2)


def test_cr_boundary_hit_rate_uses_cr_hits_any_boundary_column() -> None:
    df = _base_df(cr_hits_any_boundary=[True, True, False, False])

    assert cr_boundary_hit_rate(df) == pytest.approx(0.5)


def test_boundary_rates_return_nan_for_empty_dataframe() -> None:
    df = _base_df(
        alpha_hat_at_any_boundary=[True, False, True],
        cr_hits_any_boundary=[True, False, True],
    ).iloc[0:0]

    assert np.isnan(boundary_rate(df))
    assert np.isnan(alpha_hat_boundary_rate(df))
    assert np.isnan(cr_boundary_hit_rate(df))


def test_boundary_rates_parse_csv_style_string_booleans() -> None:
    df = _base_df(
        alpha_hat_at_any_boundary=["True", "False", "true", "0", "1"],
        cr_hits_any_boundary=["False", "true", "1", "0", None],
    )

    assert boundary_rate(df) == pytest.approx(3 / 5)
    assert alpha_hat_boundary_rate(df) == pytest.approx(3 / 5)
    assert cr_boundary_hit_rate(df) == pytest.approx(2 / 4)


def test_optional_diagnostic_means_and_rates() -> None:
    df = _base_df(
        alpha_hat_at_any_boundary=[True, False, "true", None],
        cr_hits_any_boundary=[False, True, "true", None],
        failed_alpha_count=[0, 2, None, "4"],
        selected_controls=[1, None, "3", 5],
        runtime_seconds=[0.5, "1.0", None, "bad"],
    )

    assert boundary_rate(df) == pytest.approx(2 / 3)
    assert alpha_hat_boundary_rate(df) == pytest.approx(2 / 3)
    assert cr_boundary_hit_rate(df) == pytest.approx(2 / 3)
    assert mean_failed_alpha_count(df) == pytest.approx(2.0)
    assert mean_selected_controls(df) == pytest.approx(3.0)
    assert mean_runtime_seconds(df) == pytest.approx(0.75)


def test_diagnostic_means_ignore_negative_values() -> None:
    df = _base_df(
        runtime_seconds=[-10.0, 1.0, 3.0],
        failed_alpha_count=[-5, 2, 4],
        selected_controls=[-1, 6, 8],
    )

    assert mean_runtime_seconds(df) == pytest.approx(2.0)
    assert mean_failed_alpha_count(df) == pytest.approx(3.0)
    assert mean_selected_controls(df) == pytest.approx(7.0)


def test_diagnostic_means_ignore_infinite_values() -> None:
    df = _base_df(
        runtime_seconds=[np.inf, 1.0, 3.0],
        failed_alpha_count=[np.inf, 2, 4],
        selected_controls=[np.inf, 6, 8],
    )

    assert mean_runtime_seconds(df) == pytest.approx(2.0)
    assert mean_failed_alpha_count(df) == pytest.approx(3.0)
    assert mean_selected_controls(df) == pytest.approx(7.0)


def test_diagnostic_means_return_nan_when_all_values_are_invalid() -> None:
    df = _base_df(
        runtime_seconds=[-1.0, np.inf, None],
        failed_alpha_count=[-2, np.inf, "bad"],
        selected_controls=[-3, np.inf, "bad"],
    )

    assert np.isnan(mean_runtime_seconds(df))
    assert np.isnan(mean_failed_alpha_count(df))
    assert np.isnan(mean_selected_controls(df))


def test_summarize_group_returns_expected_keys() -> None:
    df = _base_df(
        alpha_hat=[1.1, None, 1.3],
        cr_disconnected=[False, True, False],
        alpha_hat_at_any_boundary=[False, False, True],
        cr_hits_any_boundary=[False, True, True],
        failed_alpha_count=[0, 1, 2],
        selected_controls=[5, 6, None],
    )

    summary = summarize_group(df)

    assert summary["replications"] == 3
    assert summary["valid_estimates"] == 2
    assert set(summary) == {
        "replications",
        "valid_estimates",
        "bias",
        "median_bias",
        "rmse",
        "mae",
        "coverage",
        "coverage_valid_only",
        "avg_cr_length",
        "avg_cr_length_valid_only",
        "failure_rate",
        "non_convergence_rate",
        "cr_empty_rate",
        "cr_disconnected_rate",
        "boundary_rate",
        "alpha_hat_boundary_rate",
        "cr_boundary_hit_rate",
        "mean_failed_alpha_count",
        "mean_selected_controls",
        "mean_runtime_seconds",
        "mean_runtime_total_sec",
        "median_runtime_total_sec",
        "mean_runtime_alpha_grid_sec",
        "mean_runtime_confidence_region_sec",
        "mean_dml_runtime_crossfit_sec",
        "mean_ps_runtime_selection_sec",
    }
    assert summary["boundary_rate"] == pytest.approx(1 / 3)
    assert summary["alpha_hat_boundary_rate"] == pytest.approx(1 / 3)
    assert summary["cr_boundary_hit_rate"] == pytest.approx(2 / 3)


def test_summarize_group_valid_estimates_require_hat_and_true() -> None:
    df = _base_df(
        alpha_hat=[1.0, 2.0, 4.0],
        alpha_true=[0.0, None, 2.0],
    )

    summary = summarize_group(df)

    assert summary["valid_estimates"] == 2
    assert summary["bias"] == pytest.approx(1.5)
    assert summary["rmse"] == pytest.approx(np.sqrt(2.5))


def test_summarize_group_empty_dataframe_returns_zero_counts_and_nan_metrics() -> None:
    summary = summarize_group(_base_df().iloc[0:0])

    assert summary["replications"] == 0
    assert summary["valid_estimates"] == 0
    assert np.isnan(summary["bias"])
    assert np.isnan(summary["rmse"])
    assert np.isnan(summary["coverage"])
    assert np.isnan(summary["failure_rate"])


def test_all_invalid_alpha_hat_returns_nan_error_metrics_but_failure_rate_computes() -> None:
    df = _base_df(
        alpha_hat=[None, "bad", np.nan],
        failed=[True, True, False],
    )

    assert np.isnan(bias(df))
    assert np.isnan(median_bias(df))
    assert np.isnan(rmse(df))
    assert np.isnan(mae(df))
    assert failure_rate(df) == pytest.approx(2 / 3)
