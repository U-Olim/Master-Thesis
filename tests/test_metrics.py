import numpy as np
import pandas as pd
import pytest

from ivqr_sim.metrics import (
    average_cr_length,
    bias,
    boundary_rate,
    coverage,
    coverage_valid_only,
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


def test_failure_and_non_convergence_rates() -> None:
    df = _base_df(
        failed=[False, True, False],
        converged=[True, False, True],
    )

    assert failure_rate(df) == pytest.approx(1 / 3)
    assert non_convergence_rate(df) == pytest.approx(1 / 3)


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
    assert np.isnan(mean_failed_alpha_count(df))
    assert np.isnan(mean_selected_controls(df))


def test_optional_diagnostic_means_and_rates() -> None:
    df = _base_df(
        at_grid_boundary=[True, False, "true", None],
        failed_alpha_count=[0, 2, None, "4"],
        selected_controls=[1, None, "3", 5],
        runtime_seconds=[0.5, "1.0", None, "bad"],
    )

    assert boundary_rate(df) == pytest.approx(2 / 3)
    assert mean_failed_alpha_count(df) == pytest.approx(2.0)
    assert mean_selected_controls(df) == pytest.approx(3.0)
    assert mean_runtime_seconds(df) == pytest.approx(0.75)


def test_summarize_group_returns_expected_keys() -> None:
    df = _base_df(
        alpha_hat=[1.1, None, 1.3],
        cr_disconnected=[False, True, False],
        at_grid_boundary=[False, False, True],
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
        "failure_rate",
        "non_convergence_rate",
        "cr_empty_rate",
        "cr_disconnected_rate",
        "boundary_rate",
        "mean_failed_alpha_count",
        "mean_selected_controls",
        "mean_runtime_seconds",
    }


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
