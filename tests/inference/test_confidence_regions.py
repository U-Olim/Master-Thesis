"""Tests for confidence-region inversion helpers."""

import numpy as np
import pytest

from inference.confidence_regions import (
    ConfidenceRegion,
    FAILED_ALPHA_STATISTIC,
    adjust_critical_value,
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    merge_region_and_grid_diagnostics,
    sanitize_grid_statistics,
    summarize_alpha_grid_diagnostics,
    validate_critical_value_multiplier,
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


@pytest.mark.parametrize("multiplier", [1.0, 1.1])
def test_validate_critical_value_multiplier_accepts_positive_finite(
    multiplier: float,
) -> None:
    assert validate_critical_value_multiplier(multiplier) == pytest.approx(multiplier)


@pytest.mark.parametrize("multiplier", [True, 0.0, -1.0, np.nan, np.inf])
def test_validate_critical_value_multiplier_rejects_invalid(
    multiplier: float,
) -> None:
    with pytest.raises(ValueError, match="critical_value_multiplier"):
        validate_critical_value_multiplier(multiplier)


def test_adjust_critical_value_multiplies_nominal_threshold() -> None:
    assert adjust_critical_value(2.0, 1.5) == pytest.approx(3.0)


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


def test_alpha_grid_diagnostics_empty_confidence_region() -> None:
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0]),
        accepted_mask=np.array([False, False, False]),
        alpha_hat=0.0,
        failed_alpha_count=1,
    )

    assert diagnostics["cr_empty"] is True
    assert diagnostics["cr_accepted_alpha_count"] == 0
    assert diagnostics["cr_acceptance_rate"] == pytest.approx(0.0)
    assert diagnostics["cr_n_blocks"] == 0
    assert diagnostics["cr_disconnected"] is False
    assert np.isnan(diagnostics["cr_lower"])
    assert np.isnan(diagnostics["cr_upper"])
    assert np.isnan(diagnostics["cr_length"])
    assert np.isnan(diagnostics["cr_hull_length"])
    assert diagnostics["failed_alpha_rate"] == pytest.approx(1.0 / 3.0)


def test_alpha_grid_diagnostics_one_contiguous_confidence_region() -> None:
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0, 2.0]),
        accepted_mask=np.array([False, True, True, False]),
        alpha_hat=1.0,
    )

    assert diagnostics["cr_empty"] is False
    assert diagnostics["cr_lower"] == pytest.approx(0.0)
    assert diagnostics["cr_upper"] == pytest.approx(1.0)
    assert diagnostics["cr_length"] == pytest.approx(1.0)
    assert diagnostics["cr_accepted_alpha_count"] == 2
    assert diagnostics["cr_acceptance_rate"] == pytest.approx(0.5)
    assert diagnostics["cr_n_blocks"] == 1
    assert diagnostics["cr_disconnected"] is False


def test_alpha_grid_diagnostics_two_disconnected_blocks() -> None:
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        accepted_mask=np.array([False, True, True, False, True]),
        alpha_hat=0.0,
    )

    assert diagnostics["cr_accepted_alpha_count"] == 3
    assert diagnostics["cr_n_blocks"] == 2
    assert diagnostics["cr_disconnected"] is True
    assert diagnostics["cr_lower"] == pytest.approx(0.0)
    assert diagnostics["cr_upper"] == pytest.approx(3.0)
    assert diagnostics["cr_hull_length"] == pytest.approx(3.0)


def test_alpha_grid_diagnostics_boundary_hits() -> None:
    lower = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0]),
        accepted_mask=np.array([True, True, False]),
        alpha_hat=-1.0,
    )
    upper = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0]),
        accepted_mask=np.array([False, True, True]),
        alpha_hat=1.0,
    )

    assert lower["cr_hits_lower_boundary"] is True
    assert lower["cr_hits_upper_boundary"] is False
    assert lower["cr_hits_any_boundary"] is True
    assert lower["alpha_hat_at_lower_boundary"] is True
    assert lower["alpha_hat_at_any_boundary"] is True
    assert upper["cr_hits_lower_boundary"] is False
    assert upper["cr_hits_upper_boundary"] is True
    assert upper["cr_hits_any_boundary"] is True
    assert upper["alpha_hat_at_upper_boundary"] is True
    assert upper["alpha_hat_at_any_boundary"] is True


def test_alpha_grid_diagnostics_test_statistics() -> None:
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0]),
        accepted_mask=np.array([False, True, False]),
        alpha_hat=0.0,
        test_stats=np.array([5.0, 1.0, 4.0]),
        critical_value=3.84,
    )

    assert diagnostics["min_test_stat"] == pytest.approx(1.0)
    assert diagnostics["max_test_stat"] == pytest.approx(5.0)
    assert diagnostics["test_stat_at_alpha_hat"] == pytest.approx(1.0)
    assert diagnostics["critical_value"] == pytest.approx(3.84)
    assert diagnostics["critical_value_nominal"] == pytest.approx(3.84)
    assert diagnostics["critical_value_multiplier"] == pytest.approx(1.0)
    assert diagnostics["critical_value_adjusted"] == pytest.approx(3.84)


def _confidence_region(
    *,
    lower: float | None,
    upper: float | None,
    length: float,
    hull_length: float,
    blocks: tuple[tuple[float, float], ...],
    empty: bool = False,
) -> ConfidenceRegion:
    return ConfidenceRegion(
        lower=lower,
        upper=upper,
        length=length,
        hull_length=hull_length,
        blocks=blocks,
        accepted_alphas=tuple(),
        n_blocks=len(blocks),
        empty=empty,
        disconnected=len(blocks) > 1,
        covers_true=None,
        selected_grid=np.array([], dtype=float),
        critical_value=3.84,
        critical_value_nominal=3.84,
        critical_value_multiplier=1.0,
        critical_value_adjusted=3.84,
        statistic_reference=0.0,
    )


def test_merge_region_diagnostics_uses_disconnected_region_block_length() -> None:
    grid_diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([0.0, 0.2, 0.8, 1.0]),
        accepted_mask=np.array([True, True, True, True]),
        alpha_hat=0.0,
    )
    region = _confidence_region(
        lower=0.0,
        upper=1.0,
        length=0.4,
        hull_length=1.0,
        blocks=((0.0, 0.2), (0.8, 1.0)),
    )

    diagnostics = merge_region_and_grid_diagnostics(region, grid_diagnostics)

    assert diagnostics["cr_lower"] == pytest.approx(0.0)
    assert diagnostics["cr_upper"] == pytest.approx(1.0)
    assert diagnostics["cr_hull_length"] == pytest.approx(1.0)
    assert diagnostics["cr_length"] == pytest.approx(0.4)
    assert diagnostics["cr_n_blocks"] == 2
    assert diagnostics["cr_disconnected"] is True


def test_merge_region_diagnostics_uses_contiguous_region_geometry() -> None:
    grid_diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([0.0, 0.5, 1.0]),
        accepted_mask=np.array([False, True, False]),
        alpha_hat=0.5,
    )
    region = _confidence_region(
        lower=0.2,
        upper=0.8,
        length=0.6,
        hull_length=0.6,
        blocks=((0.2, 0.8),),
    )

    diagnostics = merge_region_and_grid_diagnostics(region, grid_diagnostics)

    assert diagnostics["cr_n_blocks"] == 1
    assert diagnostics["cr_disconnected"] is False
    assert diagnostics["cr_length"] == pytest.approx(region.length)
    assert diagnostics["cr_hull_length"] == pytest.approx(region.hull_length)
    assert diagnostics["cr_lower"] == pytest.approx(region.lower)
    assert diagnostics["cr_upper"] == pytest.approx(region.upper)


def test_merge_region_diagnostics_preserves_empty_region_convention() -> None:
    grid_diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([0.0, 0.5, 1.0]),
        accepted_mask=np.array([False, False, False]),
        alpha_hat=0.5,
    )
    region = _confidence_region(
        lower=None,
        upper=None,
        length=0.0,
        hull_length=0.0,
        blocks=(),
        empty=True,
    )

    diagnostics = merge_region_and_grid_diagnostics(region, grid_diagnostics)

    assert diagnostics["cr_empty"] is True
    assert diagnostics["cr_n_blocks"] == 0
    assert diagnostics["cr_disconnected"] is False
    assert np.isnan(diagnostics["cr_lower"])
    assert np.isnan(diagnostics["cr_upper"])
    assert diagnostics["cr_length"] == pytest.approx(region.length)
    assert diagnostics["cr_hull_length"] == pytest.approx(region.hull_length)


def test_merge_region_diagnostics_keeps_boundary_hits_grid_based() -> None:
    lower_grid = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0]),
        accepted_mask=np.array([True, False, False]),
        alpha_hat=0.0,
    )
    upper_grid = summarize_alpha_grid_diagnostics(
        alpha_grid=np.array([-1.0, 0.0, 1.0]),
        accepted_mask=np.array([False, False, True]),
        alpha_hat=0.0,
    )
    region = _confidence_region(
        lower=-0.9,
        upper=0.9,
        length=1.8,
        hull_length=1.8,
        blocks=((-0.9, 0.9),),
    )

    lower = merge_region_and_grid_diagnostics(region, lower_grid)
    upper = merge_region_and_grid_diagnostics(region, upper_grid)

    assert lower["cr_hits_lower_boundary"] is True
    assert lower["cr_hits_upper_boundary"] is False
    assert lower["cr_hits_any_boundary"] is True
    assert upper["cr_hits_lower_boundary"] is False
    assert upper["cr_hits_upper_boundary"] is True
    assert upper["cr_hits_any_boundary"] is True


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
    assert region.critical_value_nominal == pytest.approx(4.0)
    assert region.critical_value_multiplier == pytest.approx(1.0)
    assert region.critical_value_adjusted == pytest.approx(4.0)


def test_invert_score_test_uses_adjusted_critical_value_for_acceptance() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([1.0, 2.0, 3.0])

    nominal = invert_score_test(alphas, stats, critical_value=2.0)
    adjusted = invert_score_test(
        alphas,
        stats,
        critical_value=2.0,
        critical_value_multiplier=1.5,
    )

    assert nominal.accepted_alphas == (0.0, 1.0)
    assert adjusted.accepted_alphas == (0.0, 1.0, 2.0)
    assert adjusted.critical_value == pytest.approx(3.0)
    assert adjusted.critical_value_nominal == pytest.approx(2.0)
    assert adjusted.critical_value_multiplier == pytest.approx(1.5)
    assert adjusted.critical_value_adjusted == pytest.approx(3.0)
    assert adjusted.length >= nominal.length


def test_critical_value_multiplier_does_not_change_argmin_grid() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([2.0, 0.5, 3.0])

    alpha_hat_nominal, min_nominal, _ = argmin_grid(alphas, stats)
    _ = invert_score_test(alphas, stats, critical_value=1.0)
    _ = invert_score_test(
        alphas,
        stats,
        critical_value=1.0,
        critical_value_multiplier=3.0,
    )
    alpha_hat_adjusted, min_adjusted, _ = argmin_grid(alphas, stats)

    assert alpha_hat_adjusted == pytest.approx(alpha_hat_nominal)
    assert min_adjusted == pytest.approx(min_nominal)


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


def test_invert_score_test_rejects_unsupported_inversion_type() -> None:
    alphas = np.array([0.0, 1.0, 2.0])
    stats = np.array([10.0, 6.0, 10.0])

    with pytest.raises(
        ValueError,
        match="Only absolute confidence-region inversion is supported",
    ):
        invert_score_test(alphas, stats, critical_value=2.0, inversion_type="qlr")


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


