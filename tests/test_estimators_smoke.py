from ivqr_sim.estimators.base import EstimationResult


def test_estimation_result_can_be_instantiated() -> None:
    result = EstimationResult(
        estimator="full_ivqr",
        alpha_hat=None,
        alpha_true=1.0,
        tau=0.5,
        converged=False,
        failed=True,
        message="not implemented",
        objective_value=None,
        at_grid_boundary=False,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        selected_controls=None,
        runtime_seconds=0.0,
    )

    assert result.estimator == "full_ivqr"
    assert result.failed is True
    assert result.cr_empty is True
