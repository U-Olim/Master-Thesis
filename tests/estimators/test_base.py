"""Tests for shared estimator result containers."""

from estimators.base import EstimationResult


def test_estimation_result_can_be_instantiated() -> None:
    result = EstimationResult(
        estimator="full_control_ivqr",
        alpha_hat=None,
        alpha_true=1.0,
        tau=0.5,
        converged=False,
        failed=True,
        message="not implemented",
        objective_value=None,
        at_grid_boundary=False,
        alpha_grid_size=None,
        failed_alpha_count=None,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        cr_disconnected=None,
        selected_controls=None,
        runtime_seconds=0.0,
    )

    assert result.estimator == "full_control_ivqr"
    assert result.failed is True
    assert result.cr_empty is True
    assert result.cr_disconnected is None


def test_empty_confidence_region_is_separate_from_estimator_failure() -> None:
    result = EstimationResult(
        estimator="dml_ivqr",
        alpha_hat=1.0,
        alpha_true=1.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=10.0,
        at_grid_boundary=False,
        alpha_grid_size=5,
        failed_alpha_count=0,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=False,
        cr_empty=True,
        cr_disconnected=False,
        selected_controls=None,
        runtime_seconds=0.0,
    )

    assert result.failed is False
    assert result.converged is True
    assert result.cr_empty is True
