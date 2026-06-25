"""Tests for confidence-region inversion helpers."""

import numpy as np
import pytest

from inference.confidence_regions import (
    FAILED_ALPHA_STATISTIC,
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    is_disconnected_region,
    sanitize_grid_statistics,
    summarize_region,
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
    assert region.lower is not None
    assert region.upper is not None
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

