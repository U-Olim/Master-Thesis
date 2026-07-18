import numpy as np
import pytest
import inspect
from unittest.mock import Mock

from dgp import Design, generate_data, get_oracle_control_indices
from estimators import EstimationResult
from estimators.dml import estimate_dml_ivqr
from estimators.oracle import estimate_oracle_ivqr
from estimators.post_selection import (
    estimate_post_selection_ivqr,
    evaluate_post_selection_alpha,
)
from ivqr.ch_inverse import (
    AlphaEvaluation,
    estimate_ch_ivqr_controls,
    evaluate_alpha_ch_ivqr,
)
from simulation.results import build_simulation_result_row


def _tiny_data():
    return generate_data(Design("dgp1", 80, 20, 1.0, 0.5, rep=0, seed=321))


def test_ch_and_post_selection_public_defaults_use_valid_warning_fits() -> None:
    functions = (
        evaluate_alpha_ch_ivqr,
        estimate_ch_ivqr_controls,
        evaluate_post_selection_alpha,
        estimate_post_selection_ivqr,
    )
    for function in functions:
        parameter = inspect.signature(function).parameters["iteration_warning_policy"]
        assert parameter.default == "use_if_valid"
    for function in (estimate_ch_ivqr_controls, estimate_post_selection_ivqr):
        parameter = inspect.signature(function).parameters["hard_failure_policy"]
        assert parameter.default == "unresolved"
    assert "hard_failure_policy" not in inspect.signature(estimate_dml_ivqr).parameters


def test_oracle_propagates_iteration_warning_policy(monkeypatch) -> None:
    captured: dict[str, object] = {}
    expected = Mock(estimator="oracle", selected_controls=1)

    def fake_estimate(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr("estimators.oracle.estimate_ch_ivqr_controls", fake_estimate)
    monkeypatch.setattr("estimators.oracle.replace", lambda result, **kwargs: result)
    data = _tiny_data()
    result = estimate_oracle_ivqr(
        data,
        tau=0.5,
        oracle_indices=np.array([0]),
        iteration_warning_policy="use_if_valid",
        hard_failure_policy="unresolved",
    )
    assert captured["iteration_warning_policy"] == "use_if_valid"
    assert captured["hard_failure_policy"] == "unresolved"
    assert result.estimator == "oracle"


def test_post_selection_evaluator_propagates_iteration_warning_policy(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    expected = AlphaEvaluation(
        statistic=1.0,
        gamma_hat=np.array([1.0]),
        cov_gamma=np.array([[1.0]]),
        dim_z=1,
        converged=False,
        usable=True,
        warning_type="iteration_limit",
        failure_reason=None,
        message="ok",
    )

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "estimators.post_selection._evaluate_alpha_ch_ivqr", fake_evaluate
    )
    result = evaluate_post_selection_alpha(
        np.arange(3.0),
        np.arange(3.0),
        np.arange(3.0),
        np.empty((3, 0)),
        1.0,
        0.5,
        iteration_warning_policy="use_if_valid",
    )
    assert captured["iteration_warning_policy"] == "use_if_valid"
    assert result is expected


def test_unresolved_status_fields_are_serialized() -> None:
    design = Design("dgp1", 80, 20, 1.0, 0.5, rep=0, seed=321)
    result = EstimationResult(
        estimator="oracle",
        alpha_hat=1.0,
        alpha_true=1.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=1.0,
        at_grid_boundary=False,
        alpha_grid_size=3,
        failed_alpha_count=1,
        cr_lower=1.0,
        cr_upper=1.0,
        cr_length=0.0,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=False,
        selected_controls=5,
        runtime_seconds=0.1,
        hard_failure_policy="unresolved",
        cr_status="partially_unresolved",
        cr_is_numerically_resolved=False,
        cr_unresolved_count=1,
        cr_unresolved_alphas="[0.0]",
        coverage_status="covered",
        point_estimate_status="potentially_unresolved",
        usable_alpha_evaluations=2,
        unresolved_alpha_evaluations=1,
    )
    row = build_simulation_result_row(design, result, np.array([0.0, 1.0, 2.0]))
    for field in (
        "hard_failure_policy",
        "cr_status",
        "cr_is_numerically_resolved",
        "cr_unresolved_count",
        "cr_unresolved_alphas",
        "coverage_status",
        "point_estimate_status",
        "usable_alpha_evaluations",
        "unresolved_alpha_evaluations",
    ):
        assert row[field] == getattr(result, field)


def test_ch_hard_failure_policy_controls_inference_not_alpha_usability(
    monkeypatch,
) -> None:
    def fake_evaluate(**kwargs):
        alpha = float(kwargs["alpha"])
        usable = alpha != 1.0
        return AlphaEvaluation(
            statistic={0.0: 1.0, 1.0: np.inf, 2.0: 5.0}[alpha],
            gamma_hat=np.array([0.0]),
            cov_gamma=np.array([[1.0]]),
            dim_z=1,
            converged=usable,
            usable=usable,
            warning_type=None,
            failure_reason=None if usable else "numerical_failure",
            message="ok" if usable else "numerical_failure",
        )

    monkeypatch.setattr("ivqr.ch_inverse.evaluate_alpha_ch_ivqr", fake_evaluate)
    data = _tiny_data()
    kwargs = {
        "data": data,
        "tau": 0.5,
        "x_controls": data.x[:, :1],
        "estimator_name": "oracle",
        "alphas": np.array([0.0, 1.0, 2.0]),
    }
    production = estimate_ch_ivqr_controls(**kwargs)
    legacy = estimate_ch_ivqr_controls(**kwargs, hard_failure_policy="legacy_reject")

    assert production.cr_status == "partially_unresolved"
    assert production.cr_covers_true is None
    assert production.coverage_status == "coverage_unresolved"
    assert production.point_estimate_status == "potentially_unresolved"
    assert production.unresolved_alpha_evaluations == 1

    assert legacy.hard_failure_policy == "legacy_reject"
    assert legacy.cr_status == "valid"
    assert legacy.cr_covers_true is False
    assert legacy.coverage_status == "not_covered"
    assert legacy.failed_alpha_count == 1


@pytest.mark.slow
def test_estimators_return_estimation_results_with_expected_names() -> None:
    data = _tiny_data()
    alphas = np.linspace(-1.0, 3.0, 5)
    results = [
        estimate_oracle_ivqr(
            data,
            tau=0.5,
            alphas=alphas,
            oracle_indices=get_oracle_control_indices("dgp1", data.x.shape[1]),
            max_iter=100,
        ),
        estimate_post_selection_ivqr(
            data,
            tau=0.5,
            alphas=alphas,
            selection_cv=2,
            quantreg_max_iter=100,
        ),
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=2),
    ]
    assert all(isinstance(result, EstimationResult) for result in results)
    assert [result.estimator for result in results] == [
        "oracle",
        "post_selection_ivqr",
        "dml_ivqr",
    ]
    for result in results:
        assert result.status in {"ok", "failed"}
        assert isinstance(result.message, str)
        assert isinstance(result.diagnostics, dict)
        assert isinstance(result.confidence_region, dict)
        assert "failed_alpha_rate" in result.diagnostics
        assert "ps_n_selected_controls" in result.diagnostics
        assert "runtime_total_sec" in result.diagnostics


@pytest.mark.slow
def test_dml_cached_and_uncached_paths_match_tiny_run() -> None:
    data = _tiny_data()
    alphas = np.linspace(-1.0, 3.0, 3)
    cached = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=2,
        fold_random_state=123,
        quantile_penalty=0.05,
        use_cache=True,
    )
    uncached = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=2,
        fold_random_state=123,
        quantile_penalty=0.05,
        use_cache=False,
    )

    assert cached.alpha_hat == pytest.approx(uncached.alpha_hat)
    assert cached.cr_lower == pytest.approx(uncached.cr_lower)
    assert cached.cr_upper == pytest.approx(uncached.cr_upper)
    assert cached.cr_covers_true == uncached.cr_covers_true
    assert cached.failed_alpha_count == uncached.failed_alpha_count
    assert cached.dml_qr_fit_count == uncached.dml_qr_fit_count


@pytest.mark.slow
def test_post_selection_lasso_multiplier_diagnostics() -> None:
    data = _tiny_data()
    alphas = np.linspace(-1.0, 3.0, 5)
    for multiplier in (1.0, 1.5):
        result = estimate_post_selection_ivqr(
            data,
            tau=0.5,
            alphas=alphas,
            selection_cv=2,
            selection_lasso_multiplier=multiplier,
            quantreg_max_iter=100,
        )
        assert isinstance(result, EstimationResult)
        assert result.status in {"ok", "failed"}
        assert result.ps_selection_lasso_multiplier == multiplier
        assert result.diagnostics["ps_selection_lasso_multiplier"] == multiplier
        if result.ps_lasso_alpha_y_cv is not None:
            assert result.ps_lasso_alpha_y_final == pytest.approx(
                multiplier * result.ps_lasso_alpha_y_cv
            )
        if result.ps_lasso_alpha_d_cv is not None:
            assert result.ps_lasso_alpha_d_final == pytest.approx(
                multiplier * result.ps_lasso_alpha_d_cv
            )


def test_post_selection_lasso_multiplier_must_be_positive() -> None:
    data = _tiny_data()
    with pytest.raises(ValueError, match="selection_lasso_multiplier must be positive"):
        estimate_post_selection_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(-1.0, 3.0, 5),
            selection_cv=2,
            selection_lasso_multiplier=0,
        )
