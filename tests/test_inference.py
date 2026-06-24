# Consolidated tests for the thematic project structure.

import numpy as np
import pandas as pd
import pytest

import inference.moments as moments
from inference import (
    FAILED_ALPHA_STATISTIC,
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    is_disconnected_region,
    sanitize_grid_statistics,
    summarize_region,
)
from inference.alpha_grid import alpha_grid
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


@pytest.mark.parametrize("level", [True, np.nan, np.inf, 0.0, -0.1, 1.0, 1.1])
def test_critical_value_chi_square_validates_level(level: float) -> None:
    with pytest.raises(ValueError):
        critical_value_chi_square(level=level, df=1)


@pytest.mark.parametrize("df", [True, 0, 1.5])
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
    assert region.region_length == pytest.approx(2.0)
    assert region.hull_length == pytest.approx(2.0)
    assert region.blocks == ((-1.0, 1.0),)
    assert region.accepted_alphas == (-1.0, 0.0, 1.0)
    assert region.n_blocks == 1
    assert region.empty is False
    assert region.is_empty is False
    assert region.disconnected is False
    assert region.is_disconnected is False
    assert region.covers_true is True
    assert region.critical_value == pytest.approx(4.0)


def test_invert_score_test_singleton_region() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([10.0, 1.0, 10.0])

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=0.0)

    assert np.allclose(region.selected_grid, np.array([0.0]))
    assert len(region.blocks) == 1
    assert region.blocks[0][0] == pytest.approx(-1.0 / 9.0)
    assert region.blocks[0][1] == pytest.approx(1.0 / 9.0)
    assert region.lower == pytest.approx(-1.0 / 9.0)
    assert region.upper == pytest.approx(1.0 / 9.0)
    assert region.length == pytest.approx(2.0 / 9.0)
    assert region.hull_length == pytest.approx(2.0 / 9.0)
    assert region.empty is False
    assert region.disconnected is False
    assert region.covers_true is True
    assert region.statistic_reference == pytest.approx(0.0)


def test_invert_score_test_empty_region() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([10.0, 10.0, 10.0])

    region = invert_score_test(alphas, stats, critical_value=1.0, alpha_true=0.0)

    assert region.empty is True
    assert region.lower is None
    assert region.upper is None
    assert region.length == pytest.approx(0.0)
    assert region.region_length == pytest.approx(0.0)
    assert region.hull_length == pytest.approx(0.0)
    assert region.blocks == ()
    assert region.accepted_alphas == ()
    assert region.n_blocks == 0
    assert len(region.selected_grid) == 0
    assert region.disconnected is False
    assert region.covers_true is False


def test_invert_score_test_disconnected_region_uses_block_length() -> None:
    alphas = np.array([0, 1, 2, 3, 4], dtype=float)
    stats = np.array([1, 1, 10, 1, 1], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=2.0)

    assert np.allclose(region.selected_grid, np.array([0.0, 1.0, 3.0, 4.0]))
    assert region.lower == pytest.approx(0.0)
    assert region.upper == pytest.approx(4.0)
    assert region.blocks == ((0.0, 10.0 / 9.0), (26.0 / 9.0, 4.0))
    assert region.n_blocks == 2
    assert region.length == pytest.approx(20.0 / 9.0)
    assert region.region_length == pytest.approx(20.0 / 9.0)
    assert region.hull_length == pytest.approx(4.0)
    assert region.disconnected is True
    assert region.covers_true is False


def test_invert_score_test_disconnected_region_covers_true_inside_block() -> None:
    alphas = np.array([0, 1, 2, 3, 4], dtype=float)
    stats = np.array([1, 1, 10, 1, 1], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=3.5)

    assert region.blocks == ((0.0, 10.0 / 9.0), (26.0 / 9.0, 4.0))
    assert region.covers_true is True


def test_invert_score_test_all_accepted_region() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([1.0, 1.0, 1.0])

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=0.5)

    assert region.blocks == ((-1.0, 1.0),)
    assert region.lower == pytest.approx(-1.0)
    assert region.upper == pytest.approx(1.0)
    assert region.length == pytest.approx(2.0)
    assert region.hull_length == pytest.approx(2.0)
    assert region.disconnected is False
    assert region.covers_true is True


def test_invert_score_test_sorts_unsorted_alpha_grid_with_aligned_statistics() -> None:
    alphas = np.array([2.0, 0.0, 1.0])
    stats = np.array([10.0, 1.0, 1.0])

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=0.5)

    assert np.allclose(region.selected_grid, np.array([0.0, 1.0]))
    assert region.blocks == ((0.0, 10.0 / 9.0),)
    assert region.lower == pytest.approx(0.0)
    assert region.upper == pytest.approx(10.0 / 9.0)
    assert region.length == pytest.approx(10.0 / 9.0)
    assert region.covers_true is True


def test_invert_score_test_can_use_profiled_statistic_difference() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([10.0, 6.0, 10.0])

    absolute = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=1.0)
    profiled = invert_score_test(
        alphas,
        stats,
        critical_value=2.0,
        alpha_true=1.0,
        statistic_reference=6.0,
        inversion_type="qlr",
    )

    assert absolute.empty is True
    assert profiled.empty is False
    assert profiled.blocks == ((0.5, 1.5),)
    assert profiled.covers_true is True
    assert profiled.statistic_reference == pytest.approx(6.0)


def test_invert_score_test_absolute_ignores_statistic_reference() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([10.0, 1.0, 10.0])

    region = invert_score_test(
        alphas,
        stats,
        critical_value=3.84,
        statistic_reference=1.0,
        inversion_type="absolute",
    )

    assert region.selected_grid.tolist() == [1.0]
    assert region.statistic_reference == pytest.approx(0.0)


def test_invert_score_test_absolute_accepts_statistics_at_or_below_critical_value() -> None:
    alphas = np.array([0.0, 1.0, 2.0, 3.0])
    stats = np.array([2.0, 2.01, 1.0, 4.0])

    region = invert_score_test(
        alphas,
        stats,
        critical_value=2.0,
        statistic_reference=True,
        inversion_type="absolute",
    )

    assert region.selected_grid.tolist() == [0.0, 2.0]
    assert region.statistic_reference == pytest.approx(0.0)


def test_invert_score_test_qlr_defaults_to_minimum_statistic() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([10.0, 6.0, 10.0])

    default = invert_score_test(
        alphas,
        stats,
        critical_value=2.0,
        inversion_type="qlr",
    )
    explicit = invert_score_test(
        alphas,
        stats,
        critical_value=2.0,
        statistic_reference=6.0,
        inversion_type="qlr",
    )

    assert default.statistic_reference == pytest.approx(6.0)
    assert default.selected_grid.tolist() == explicit.selected_grid.tolist() == [1.0]
    assert default.blocks == explicit.blocks


@pytest.mark.parametrize("statistic_reference", [True, np.nan, np.inf])
def test_invert_score_test_qlr_rejects_invalid_statistic_reference(
    statistic_reference: float,
) -> None:
    with pytest.raises(ValueError):
        invert_score_test(
            np.array([0.0, 1.0, 2.0]),
            np.array([10.0, 6.0, 10.0]),
            critical_value=2.0,
            statistic_reference=statistic_reference,
            inversion_type="qlr",
        )


def test_invert_score_test_qlr_rejects_reference_above_minimum() -> None:
    with pytest.raises(
        ValueError,
        match="statistic_reference cannot exceed the minimum statistic",
    ):
        invert_score_test(
            np.array([0.0, 1.0, 2.0]),
            np.array([10.0, 6.0, 10.0]),
            critical_value=2.0,
            statistic_reference=6.1,
            inversion_type="qlr",
        )


def test_invert_score_test_selected_grid_is_read_only() -> None:
    region = invert_score_test(
        np.array([0.0, 1.0, 2.0]),
        np.array([10.0, 1.0, 10.0]),
        critical_value=2.0,
    )

    with pytest.raises(ValueError):
        region.selected_grid[0] = 999.0

    assert region.selected_grid.tolist() == [1.0]


def test_invert_score_test_empty_selected_grid_is_read_only() -> None:
    region = invert_score_test(
        np.array([0.0, 1.0, 2.0]),
        np.array([10.0, 10.0, 10.0]),
        critical_value=2.0,
    )

    assert region.selected_grid.flags.writeable is False


def test_invert_score_test_interpolates_off_grid_coverage() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([10.0, 1.0, 10.0])

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=0.95)

    assert region.selected_grid.tolist() == [1.0]
    assert region.lower < 0.95 < region.upper
    assert region.covers_true is True


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


@pytest.mark.parametrize("critical_value", [True, False, np.nan, np.inf, 0.0, -1.0])
def test_invert_score_test_rejects_invalid_critical_value(
    critical_value: float,
) -> None:
    with pytest.raises(ValueError):
        invert_score_test(
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            critical_value=critical_value,
        )


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


def test_sanitize_grid_statistics_accepts_boolean_converged_mask() -> None:
    sanitized, num_failed = sanitize_grid_statistics(
        np.array([1.0, 2.0, 3.0]),
        np.array([True, False, True], dtype=bool),
    )

    assert sanitized.tolist() == [1.0, FAILED_ALPHA_STATISTIC, 3.0]
    assert num_failed == 1


def test_sanitize_grid_statistics_rejects_numeric_converged_mask() -> None:
    with pytest.raises(ValueError, match="converged must be boolean"):
        sanitize_grid_statistics(
            np.array([1.0, 2.0, 3.0]),
            np.array([1, 0, 1]),
        )


@pytest.mark.parametrize("failed_value", [True, np.nan, np.inf, 0.0])
def test_sanitize_grid_statistics_rejects_invalid_failed_value(
    failed_value: float,
) -> None:
    with pytest.raises(ValueError):
        sanitize_grid_statistics(
            np.array([1.0, 2.0, 3.0]),
            np.array([True, False, True]),
            failed_value=failed_value,
        )


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


def test_quantile_score_uses_validated_numpy_tau() -> None:
    scores = quantile_score(np.array([-1.0, 1.0]), tau=np.float64(0.5))

    assert scores.dtype == float
    assert np.allclose(scores, np.array([-0.5, 0.5]))


def test_quantile_score_rejects_empty_residuals() -> None:
    with pytest.raises(ValueError, match="residuals must be nonempty"):
        quantile_score(np.array([]), tau=0.5)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_quantile_score_validates_tau(tau: float) -> None:
    with pytest.raises(ValueError):
        quantile_score(np.array([1.0]), tau=tau)


@pytest.mark.parametrize(
    "residuals",
    [
        np.array([[1.0]]),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_quantile_score_validates_residuals(residuals: np.ndarray) -> None:
    with pytest.raises(ValueError):
        quantile_score(residuals, tau=0.5)


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


def test_residuals_alpha_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="y must be nonempty"):
        residuals_alpha(np.array([]), np.array([]), np.array([]), alpha=1.0)


@pytest.mark.parametrize(
    ("y", "d", "x_beta", "alpha"),
    [
        (np.array([[1.0]]), np.array([1.0]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([[1.0]]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([1.0]), np.array([[0.0]]), 1.0),
        (np.array([np.nan]), np.array([1.0]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([np.inf]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([1.0]), np.array([np.nan]), 1.0),
        (np.array([1.0]), np.array([1.0]), np.array([0.0]), np.nan),
        (np.array([1.0]), np.array([1.0]), np.array([0.0]), np.inf),
        (np.array([1.0]), np.array([1.0]), np.array([0.0]), True),
    ],
)
def test_residuals_alpha_validates_inputs(
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    alpha: float,
) -> None:
    with pytest.raises(ValueError):
        residuals_alpha(y, d, x_beta, alpha)


def test_make_instruments_returns_z_column_without_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z)

    assert instruments.shape == (3, 1)
    assert np.allclose(instruments[:, 0], z)


def test_make_instruments_returns_fresh_array() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z)
    instruments[0, 0] = 999.0

    assert z[0] == pytest.approx(1.0)


def test_make_instruments_stacks_selected_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])
    x_selected = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

    instruments = make_instruments(z, x_selected)

    assert instruments.shape == (3, 3)
    assert np.allclose(instruments[:, 0], z)
    assert np.allclose(instruments[:, 1:], x_selected)


def test_make_instruments_accepts_vector_valued_z() -> None:
    z = np.array([[1.0, 2.0], [0.0, 3.0], [1.0, 4.0]])

    instruments = make_instruments(z)

    assert instruments.shape == (3, 2)
    assert np.allclose(instruments, z)


def test_make_instruments_accepts_one_dimensional_controls() -> None:
    instruments = make_instruments(
        np.array([1.0, 0.0, 1.0]),
        np.array([2.0, 3.0, 4.0]),
    )

    assert instruments.shape == (3, 2)


def test_make_instruments_accepts_zero_selected_control_columns() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z, np.empty((3, 0)))

    assert instruments.shape == (3, 1)
    assert np.allclose(instruments[:, 0], z)


def test_make_instruments_rejects_empty_one_dimensional_controls() -> None:
    with pytest.raises(ValueError):
        make_instruments(np.ones(3), np.array([]))


def test_make_instruments_validates_row_counts() -> None:
    with pytest.raises(ValueError):
        make_instruments(np.array([1.0, 0.0]), np.ones((3, 2)))


@pytest.mark.parametrize(
    "z",
    [
        np.array(1.0),
        np.ones((2, 1, 1)),
        np.empty((2, 0)),
        np.array([1.0, np.nan]),
        np.array([[1.0], [np.inf]]),
    ],
)
def test_make_instruments_validates_z(z: np.ndarray) -> None:
    with pytest.raises(ValueError):
        make_instruments(z)


@pytest.mark.parametrize(
    "x_selected",
    [
        np.array(1.0),
        np.ones((2, 1, 1)),
        np.array([1.0, np.nan]),
        np.array([[1.0], [np.inf]]),
        np.empty((2, 0)),
    ],
)
def test_make_instruments_validates_selected_controls(
    x_selected: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        make_instruments(np.array([1.0, 0.0, 1.0]), x_selected)


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


@pytest.mark.parametrize(
    ("residuals", "instruments"),
    [
        (np.array([]), np.empty((0, 1))),
        (np.array([1.0, 2.0]), np.empty((2, 0))),
        (np.array([1.0]), np.ones((2, 1))),
    ],
)
def test_moment_contributions_validates_instrument_dimensions(
    residuals: np.ndarray,
    instruments: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        moment_contributions(residuals, tau=0.5, instruments=instruments)


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


def test_moment_covariance_constant_contributions_without_ridge_is_zero() -> None:
    sigma = moment_covariance(np.ones((4, 2)), ridge=0.0)

    assert sigma.shape == (2, 2)
    assert np.all(np.isfinite(sigma))
    assert np.allclose(sigma, np.zeros((2, 2)))


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
        (np.empty((2, 0)), 1e-8),
        (np.array([[1.0], [2.0]]), True),
        (np.array([[1.0], [2.0]]), np.nan),
        (np.array([[1.0], [2.0]]), np.inf),
        (np.array([[1.0], [2.0]]), -1e-8),
    ],
)
def test_moment_covariance_validates_inputs(
    contributions: np.ndarray,
    ridge: float,
) -> None:
    with pytest.raises(ValueError):
        moment_covariance(contributions, ridge=ridge)


@pytest.mark.parametrize(
    ("contributions", "ridge", "use_pinv"),
    [
        (np.array([[1.0]]), 1e-8, True),
        (np.empty((2, 0)), 1e-8, True),
        (np.array([[1.0], [2.0]]), True, True),
        (np.array([[1.0], [2.0]]), 1e-8, 1),
        (np.array([[1.0], [2.0]]), 1e-8, "yes"),
    ],
)
def test_weighted_gmm_statistic_validates_inputs(
    contributions: np.ndarray,
    ridge: float,
    use_pinv: bool,
) -> None:
    with pytest.raises(ValueError):
        weighted_gmm_statistic(
            contributions,
            ridge=ridge,
            use_pinv=use_pinv,
        )


def test_weighted_gmm_statistic_handles_singular_covariance_with_pinv() -> None:
    contributions = np.ones((3, 1))

    statistic = weighted_gmm_statistic(contributions, ridge=0.0, use_pinv=True)

    assert statistic == pytest.approx(0.0)


def test_weighted_gmm_statistic_singular_covariance_raises_without_pinv() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        weighted_gmm_statistic(np.ones((3, 1)), ridge=0.0, use_pinv=False)


def test_score_statistic_is_nonnegative() -> None:
    moment_vector = np.array([-0.2, 0.4, 0.1])

    statistic = score_statistic(moment_vector)

    assert statistic >= 0.0
    assert statistic == pytest.approx(0.21)


@pytest.mark.parametrize(
    "moment_vector",
    [
        np.array([[1.0, 2.0]]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_score_statistic_validates_moment_vector(
    moment_vector: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        score_statistic(moment_vector)


def test_alpha_grid_has_expected_length_and_endpoint() -> None:
    grid = alpha_grid(-2.0, 4.0, 0.01)

    assert len(grid) == 601
    assert grid[0] == pytest.approx(-2.0)
    assert grid[-1] == pytest.approx(4.0)


@pytest.mark.parametrize(("size", "step"), [(9, 0.5), (13, 1.0 / 3.0)])
def test_alpha_grid_supports_default_and_robustness_grid_sizes(
    size: int,
    step: float,
) -> None:
    grid = alpha_grid(-1.0, 3.0, step)

    assert len(grid) == size
    assert grid[0] == pytest.approx(-1.0)
    assert grid[-1] == pytest.approx(3.0)


def test_alpha_grid_appends_endpoint_for_non_dividing_step() -> None:
    grid = alpha_grid(0.0, 1.0, 0.3)

    assert np.all(np.diff(grid) > 0)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_min": np.nan, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": np.inf, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": np.nan},
        {"alpha_min": True, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": True, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": True},
        {"alpha_min": 1.0, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 2.0, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": 0.0},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": -0.1},
    ],
)
def test_alpha_grid_rejects_invalid_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        alpha_grid(**kwargs)


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


def test_moments_public_api_excludes_alpha_grid() -> None:
    assert "alpha_grid" not in moments.__all__
    assert "weighted_gmm_statistic" in moments.__all__
    assert "quantile_score" in moments.__all__


@pytest.mark.parametrize(
    "alphas",
    [
        np.array([]),
        np.array([0.0, -1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, np.nan]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_evaluate_grid_validates_alpha_grid(alphas: np.ndarray) -> None:
    with pytest.raises(ValueError):
        evaluate_grid(
            alphas,
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            tau=0.5,
            instruments=np.ones((2, 1)),
        )


@pytest.mark.parametrize(
    ("y", "d", "x_beta", "instruments"),
    [
        (
            np.array([[1.0, 2.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.ones((2, 1)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0]),
            np.array([0.0, 0.0]),
            np.ones((2, 1)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, np.nan]),
            np.ones((2, 1)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.ones(2),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.empty((2, 0)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.ones((3, 1)),
        ),
    ],
)
def test_evaluate_grid_validates_data(
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    instruments: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        evaluate_grid(
            np.array([0.0, 1.0]),
            y,
            d,
            x_beta,
            tau=0.5,
            instruments=instruments,
        )


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
