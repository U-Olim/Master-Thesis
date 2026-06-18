# Consolidated tests for the thematic project structure.

import _path  # noqa: F401
import numpy as np
import pandas as pd
import pytest

from inference import (
    FAILED_ALPHA_STATISTIC,
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    is_disconnected_region,
    sanitize_grid_statistics,
    summarize_region,
)
from inference.metrics import (
    average_cr_length,
    average_cr_length_all,
    average_cr_length_valid_only,
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
from inference.moments import (
    alpha_grid,
    evaluate_grid,
    make_instruments,
    moment_contributions,
    moment_covariance,
    quantile_score,
    residuals_alpha,
    sample_moment,
    score_statistic,
    weighted_gmm_statistic,
)


def test_critical_value_chi_square_default_scalar_score() -> None:
    cv = critical_value_chi_square(level=0.95, df=1)

    assert cv == pytest.approx(3.841458820694124, rel=1e-6)


@pytest.mark.parametrize("level", [0.0, -0.1, 1.0, 1.1])
def test_critical_value_chi_square_validates_level(level: float) -> None:
    with pytest.raises(ValueError):
        critical_value_chi_square(level=level, df=1)


@pytest.mark.parametrize("df", [0, 1.5])
def test_critical_value_chi_square_validates_df(df: int) -> None:
    with pytest.raises(ValueError):
        critical_value_chi_square(level=0.95, df=df)  # type: ignore[arg-type]


def test_argmin_grid_returns_interior_minimum() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([4.0, 1.0, 3.0])

    alpha_hat, min_stat, at_boundary = argmin_grid(alphas, stats)

    assert alpha_hat == pytest.approx(0.0)
    assert min_stat == pytest.approx(1.0)
    assert at_boundary is False


def test_argmin_grid_reports_boundary_minimum() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([1.0, 2.0, 3.0])

    alpha_hat, min_stat, at_boundary = argmin_grid(alphas, stats)

    assert alpha_hat == pytest.approx(-1.0)
    assert min_stat == pytest.approx(1.0)
    assert at_boundary is True


def test_invert_score_test_connected_region_includes_critical_boundary() -> None:
    alphas = np.array([-2, -1, 0, 1, 2], dtype=float)
    stats = np.array([10, 4, 1, 4, 10], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=4.0, alpha_true=0.0)

    assert np.allclose(region.selected_grid, np.array([-1.0, 0.0, 1.0]))
    assert region.lower == pytest.approx(-1.0)
    assert region.upper == pytest.approx(1.0)
    assert region.length == pytest.approx(2.0)
    assert region.empty is False
    assert region.disconnected is False
    assert region.covers_true is True


def test_invert_score_test_empty_region() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([10.0, 10.0, 10.0])

    region = invert_score_test(alphas, stats, critical_value=1.0, alpha_true=0.0)

    assert region.empty is True
    assert region.lower is None
    assert region.upper is None
    assert region.length is None
    assert len(region.selected_grid) == 0
    assert region.disconnected is False
    assert region.covers_true is False


def test_invert_score_test_disconnected_region_uses_convex_hull_length() -> None:
    alphas = np.array([0, 1, 2, 3, 4], dtype=float)
    stats = np.array([1, 1, 10, 1, 1], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=2.0)

    assert np.allclose(region.selected_grid, np.array([0.0, 1.0, 3.0, 4.0]))
    assert region.lower == pytest.approx(0.0)
    assert region.upper == pytest.approx(4.0)
    assert region.length == pytest.approx(4.0)
    assert region.disconnected is True


def test_invert_score_test_coverage_false() -> None:
    alphas = np.array([0, 1, 2, 3, 4], dtype=float)
    stats = np.array([1, 1, 10, 1, 1], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=10.0)

    assert region.covers_true is False


def test_invert_score_test_rejects_nonfinite_statistics() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([1.0, np.inf, 2.0])

    with pytest.raises(ValueError):
        invert_score_test(alphas, stats, critical_value=3.0)


def test_is_disconnected_region_examples() -> None:
    full_grid = np.array([0, 1, 2, 3, 4], dtype=float)

    assert is_disconnected_region(np.array([0, 1, 4], dtype=float), full_grid) is True
    assert is_disconnected_region(np.array([1, 2, 3], dtype=float), full_grid) is False
    assert is_disconnected_region(np.array([], dtype=float), full_grid) is False


@pytest.mark.parametrize(
    ("alphas", "stats", "critical_value"),
    [
        (np.array([0.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, np.inf, 3.0]), 1.0),
        (np.array([0.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 0.0),
        (np.array([[0.0, 1.0, 2.0]]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 2.0]), np.array([[1.0, 2.0, 3.0]]), 1.0),
    ],
)
def test_invert_score_test_validates_inputs(
    alphas: np.ndarray,
    stats: np.ndarray,
    critical_value: float,
) -> None:
    with pytest.raises(ValueError):
        invert_score_test(alphas, stats, critical_value=critical_value)


@pytest.mark.parametrize(
    ("alphas", "stats"),
    [
        (np.array([0.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, np.nan, 3.0])),
        (np.array([0.0, np.inf, 2.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([[0.0, 1.0, 2.0]]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([[1.0, 2.0, 3.0]])),
    ],
)
def test_argmin_grid_validates_inputs(alphas: np.ndarray, stats: np.ndarray) -> None:
    with pytest.raises(ValueError):
        argmin_grid(alphas, stats)


def test_sanitize_grid_statistics_replaces_failed_points() -> None:
    statistics = np.array([1.0, np.inf, 3.0, np.nan])
    converged = [True, True, False, True]

    sanitized, num_failed = sanitize_grid_statistics(statistics, converged)

    assert np.all(np.isfinite(sanitized))
    assert sanitized[0] == pytest.approx(1.0)
    assert sanitized[1] == pytest.approx(FAILED_ALPHA_STATISTIC)
    assert sanitized[2] == pytest.approx(FAILED_ALPHA_STATISTIC)
    assert sanitized[3] == pytest.approx(FAILED_ALPHA_STATISTIC)
    assert num_failed == 3


def test_sanitize_grid_statistics_validates_lengths() -> None:
    with pytest.raises(ValueError):
        sanitize_grid_statistics(np.array([1.0, 2.0]), [True])


def test_summarize_region_returns_estimation_result_fields() -> None:
    alphas = np.array([-2, -1, 0, 1, 2], dtype=float)
    stats = np.array([10, 4, 1, 4, 10], dtype=float)
    region = invert_score_test(alphas, stats, critical_value=4.0, alpha_true=0.0)

    summary = summarize_region(region)

    assert summary == {
        "cr_lower": pytest.approx(-1.0),
        "cr_upper": pytest.approx(1.0),
        "cr_length": pytest.approx(2.0),
        "cr_empty": False,
        "cr_disconnected": False,
        "cr_covers_true": True,
    }

def test_quantile_score_uses_weak_inequality_at_zero() -> None:
    residuals = np.array([-1.0, 0.0, 2.0])

    scores = quantile_score(residuals, tau=0.5)

    assert np.allclose(scores, np.array([-0.5, -0.5, 0.5]))
    assert scores.shape == residuals.shape


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_quantile_score_validates_tau(tau: float) -> None:
    with pytest.raises(ValueError):
        quantile_score(np.array([1.0]), tau=tau)


def test_residuals_alpha_implements_formula() -> None:
    y = np.array([3.0, 4.0, 8.0])
    d = np.array([1.0, 0.0, 2.0])
    x_beta = np.array([0.5, 1.5, -1.0])

    residuals = residuals_alpha(y, d, x_beta, alpha=2.0)

    assert np.allclose(residuals, np.array([0.5, 2.5, 5.0]))


def test_residuals_alpha_validates_equal_lengths() -> None:
    with pytest.raises(ValueError):
        residuals_alpha(
            np.array([1.0, 2.0]),
            np.array([1.0]),
            np.array([0.0, 0.0]),
            alpha=1.0,
        )


def test_make_instruments_returns_z_column_without_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z)

    assert instruments.shape == (3, 1)
    assert np.allclose(instruments[:, 0], z)


def test_make_instruments_stacks_selected_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])
    x_selected = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

    instruments = make_instruments(z, x_selected)

    assert instruments.shape == (3, 3)
    assert np.allclose(instruments[:, 0], z)
    assert np.allclose(instruments[:, 1:], x_selected)


def test_make_instruments_validates_row_counts() -> None:
    with pytest.raises(ValueError):
        make_instruments(np.array([1.0, 0.0]), np.ones((3, 2)))


def test_sample_moment_returns_instrument_dimension() -> None:
    residuals = np.array([-1.0, 2.0, 3.0])
    instruments = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

    moment = sample_moment(residuals, tau=0.5, instruments=instruments)

    assert moment.shape == (2,)


def test_moment_contributions_shape() -> None:
    residuals = np.array([-1.0, 2.0, 3.0])
    instruments = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

    contributions = moment_contributions(residuals, tau=0.5, instruments=instruments)

    assert contributions.shape == (3, 2)
    assert np.all(np.isfinite(contributions))


def test_sample_moment_equals_mean_of_contributions() -> None:
    residuals = np.array([-1.0, 2.0, 3.0])
    instruments = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

    contributions = moment_contributions(residuals, tau=0.5, instruments=instruments)
    moment = sample_moment(residuals, tau=0.5, instruments=instruments)

    assert np.allclose(moment, contributions.mean(axis=0))


def test_moment_covariance_shape_and_symmetry() -> None:
    contributions = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]])

    sigma = moment_covariance(contributions, ridge=1e-8)

    assert sigma.shape == (2, 2)
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.isfinite(sigma))


def test_moment_covariance_ridge_adds_positive_diagonal() -> None:
    contributions = np.ones((4, 2))

    sigma = moment_covariance(contributions, ridge=1e-4)

    assert np.all(np.diag(sigma) > 0.0)
    assert np.allclose(np.diag(sigma), np.array([1e-4, 1e-4]))


def test_weighted_gmm_statistic_is_finite_and_nonnegative() -> None:
    contributions = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]])

    statistic = weighted_gmm_statistic(contributions, ridge=1e-8)

    assert np.isfinite(statistic)
    assert statistic >= 0.0


def test_weighted_gmm_statistic_matches_manual_scalar_case() -> None:
    contributions = np.array([[1.0], [2.0], [3.0], [4.0]])
    ridge = 1e-8
    n = contributions.shape[0]
    g_hat = contributions.mean(axis=0)
    centered = contributions - g_hat
    sigma = centered.T @ centered / n + ridge * np.eye(1)
    expected = n * g_hat @ np.linalg.inv(sigma) @ g_hat

    statistic = weighted_gmm_statistic(contributions, ridge=ridge, use_pinv=False)

    assert statistic == pytest.approx(float(expected))


@pytest.mark.parametrize(
    ("contributions", "ridge"),
    [
        (np.array([1.0, 2.0, 3.0]), 1e-8),
        (np.array([[1.0], [np.inf]]), 1e-8),
        (np.array([[1.0]]), 1e-8),
        (np.array([[1.0], [2.0]]), -1e-8),
    ],
)
def test_moment_covariance_validates_inputs(
    contributions: np.ndarray,
    ridge: float,
) -> None:
    with pytest.raises(ValueError):
        moment_covariance(contributions, ridge=ridge)


def test_score_statistic_is_nonnegative() -> None:
    moment_vector = np.array([-0.2, 0.4, 0.1])

    statistic = score_statistic(moment_vector)

    assert statistic >= 0.0
    assert statistic == pytest.approx(0.21)


def test_alpha_grid_has_expected_length_and_endpoint() -> None:
    grid = alpha_grid(-2.0, 4.0, 0.01)

    assert len(grid) == 601
    assert grid[0] == pytest.approx(-2.0)
    assert grid[-1] == pytest.approx(4.0)


def test_evaluate_grid_returns_finite_values() -> None:
    alphas = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([0.0, 1.0, 1.0, 0.0])
    x_beta = np.array([0.2, 0.3, 0.4, 0.5])
    instruments = make_instruments(
        np.array([1.0, 0.0, 1.0, 0.0]),
        np.array([[0.1], [0.2], [0.3], [0.4]]),
    )

    scores = evaluate_grid(alphas, y, d, x_beta, tau=0.5, instruments=instruments)

    assert scores.shape == alphas.shape
    assert np.all(np.isfinite(scores))


def test_moment_outputs_are_deterministic() -> None:
    alphas = alpha_grid(-0.5, 0.5, 0.25)
    y = np.array([1.0, -1.0, 2.0, -2.0])
    d = np.array([1.0, 0.0, 1.0, 0.0])
    x_beta = np.array([0.1, -0.2, 0.3, -0.4])
    instruments = make_instruments(np.array([1.0, 0.0, 1.0, 0.0]))

    first = evaluate_grid(alphas, y, d, x_beta, tau=0.25, instruments=instruments)
    second = evaluate_grid(alphas, y, d, x_beta, tau=0.25, instruments=instruments)

    assert np.allclose(first, second)

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


def test_average_cr_length_all_uses_all_rows_denominator() -> None:
    df = _base_df(cr_length=[1.0, None, 3.0])

    assert average_cr_length_all(df) == pytest.approx(4.0 / 3)


def test_average_cr_length_valid_only_ignores_missing_lengths() -> None:
    df = _base_df(cr_length=[1.0, None, 3.0])

    assert average_cr_length_valid_only(df) == pytest.approx(2.0)


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
        "avg_cr_length_valid_only",
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
