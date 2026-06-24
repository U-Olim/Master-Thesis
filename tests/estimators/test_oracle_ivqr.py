# Consolidated tests for the thematic project structure.

import numpy as np
import pytest
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp import generate_data
from dgp.designs import Design
from estimators import ch_inverse_ivqr
from estimators.base import EstimationResult
from estimators.dml_ivqr import (
    _build_dml_fold_cache,
    _evaluate_dml_ivqr_alpha_uncached,
    estimate_dml_ivqr,
    evaluate_dml_ivqr_alpha,
    fit_instrument_residualizer,
    fit_quantile_nuisance,
    make_folds,
    standardize_train_test,
)
from estimators.ch_inverse_ivqr import add_intercept
from estimators.full_control_ivqr import estimate_full_control_ivqr
from estimators.oracle_ivqr import estimate_oracle_ivqr
from estimators.post_selection_ivqr import (
    estimate_post_selection_ivqr,
    evaluate_post_selection_alpha,
    select_controls_lasso,
)


def require_float(value: float | None, name: str = "value") -> float:
    assert value is not None, f"{name} should not be None"
    return value


def require_array(value: np.ndarray | None, name: str = "array") -> np.ndarray:
    assert value is not None, f"{name} should not be None"
    return value


def test_estimate_oracle_ivqr_is_not_full_control_alias() -> None:
    assert estimate_oracle_ivqr is not estimate_full_control_ivqr


def test_estimate_oracle_ivqr_rejects_unknown_kwargs() -> None:
    data = generate_data(
        Design("dgp1", n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(TypeError, match="Unknown oracle IVQR keyword"):
        estimate_oracle_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            oracle_indices=np.arange(10),
            bad_argument=123,
        )


def test_estimate_oracle_ivqr_accepts_gmm_ridge_for_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_data(
        Design("dgp1", n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    captured: dict[str, object] = {}

    def fake_common_estimator(data, x_controls, estimator_name, **kwargs):
        captured.update(kwargs)
        return EstimationResult(
            estimator=estimator_name,
            alpha_hat=1.0,
            alpha_true=data.alpha_true,
            tau=kwargs["tau"],
            converged=True,
            failed=False,
            message="ok",
            objective_value=0.0,
            at_grid_boundary=False,
            alpha_grid_size=len(kwargs["alphas"]),
            failed_alpha_count=0,
            cr_lower=None,
            cr_upper=None,
            cr_length=None,
            cr_covers_true=None,
            cr_empty=True,
            cr_disconnected=False,
            selected_controls=None,
            runtime_seconds=0.0,
        )

    import estimators.oracle_ivqr as oracle_module

    monkeypatch.setattr(
        oracle_module,
        "estimate_ch_ivqr_controls",
        fake_common_estimator,
    )

    result = estimate_oracle_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        oracle_indices=np.arange(10),
        gmm_ridge=1e-6,
    )

    assert result.estimator == "oracle"
    assert "gmm_ridge" not in captured


def test_estimate_oracle_ivqr_alphas_alias_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_data(
        Design("dgp1", n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.linspace(0.0, 2.0, 3)
    captured: dict[str, np.ndarray] = {}

    def fake_common_estimator(data, x_controls, estimator_name, **kwargs):
        captured["alphas"] = kwargs["alphas"]
        return EstimationResult(
            estimator=estimator_name,
            alpha_hat=1.0,
            alpha_true=data.alpha_true,
            tau=kwargs["tau"],
            converged=True,
            failed=False,
            message="ok",
            objective_value=0.0,
            at_grid_boundary=False,
            alpha_grid_size=len(kwargs["alphas"]),
            failed_alpha_count=0,
            cr_lower=None,
            cr_upper=None,
            cr_length=None,
            cr_covers_true=None,
            cr_empty=True,
            cr_disconnected=False,
            selected_controls=None,
            runtime_seconds=0.0,
        )

    import estimators.oracle_ivqr as oracle_module

    monkeypatch.setattr(
        oracle_module,
        "estimate_ch_ivqr_controls",
        fake_common_estimator,
    )

    result = estimate_oracle_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        oracle_indices=np.arange(10),
    )

    assert result.estimator == "oracle"
    np.testing.assert_array_equal(captured["alphas"], alphas)


def test_estimate_oracle_ivqr_uses_reduced_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    data = generate_data(Design("dgp1", n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123))
    captured: dict[str, tuple[int, int]] = {}

    def fake_common_estimator(data, x_controls, estimator_name, **kwargs):
        captured["x_shape"] = x_controls.shape
        return EstimationResult(
            estimator=estimator_name,
            alpha_hat=1.0,
            alpha_true=data.alpha_true,
            tau=kwargs["tau"],
            converged=True,
            failed=False,
            message="ok",
            objective_value=0.0,
            at_grid_boundary=False,
            alpha_grid_size=len(kwargs["alphas"]),
            failed_alpha_count=0,
            cr_lower=None,
            cr_upper=None,
            cr_length=None,
            cr_covers_true=None,
            cr_empty=True,
            cr_disconnected=False,
            selected_controls=None,
            runtime_seconds=0.0,
        )

    import estimators.oracle_ivqr as oracle_module

    monkeypatch.setattr(oracle_module, "estimate_ch_ivqr_controls", fake_common_estimator)

    result = estimate_oracle_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        oracle_indices=np.arange(10),
    )

    assert captured["x_shape"] == (80, 10)
    assert result.estimator == "oracle"
    assert result.selected_controls == 10


def test_estimate_oracle_ivqr_accepts_array_api(monkeypatch: pytest.MonkeyPatch) -> None:
    data = generate_data(Design("dgp1", n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123))
    captured: dict[str, tuple[int, int]] = {}

    def fake_common_estimator(data, x_controls, estimator_name, **kwargs):
        captured["x_shape"] = x_controls.shape
        return EstimationResult(
            estimator=estimator_name,
            alpha_hat=1.0,
            alpha_true=data.alpha_true,
            tau=kwargs["tau"],
            converged=True,
            failed=False,
            message="ok",
            objective_value=0.0,
            at_grid_boundary=False,
            alpha_grid_size=len(kwargs["alphas"]),
            failed_alpha_count=0,
            cr_lower=None,
            cr_upper=None,
            cr_length=None,
            cr_covers_true=None,
            cr_empty=True,
            cr_disconnected=False,
            selected_controls=None,
            runtime_seconds=0.0,
        )

    import estimators.oracle_ivqr as oracle_module

    monkeypatch.setattr(oracle_module, "estimate_ch_ivqr_controls", fake_common_estimator)

    result = estimate_oracle_ivqr(
        data.y,
        data.d,
        data.x,
        data.z,
        tau=0.5,
        alpha_candidates=np.linspace(0.0, 2.0, 3),
        oracle_indices=np.arange(10),
        alpha_true=data.alpha_true,
    )

    assert captured["x_shape"] == (80, 10)
    assert result.estimator == "oracle"
    assert result.selected_controls == 10


def test_estimate_oracle_ivqr_alpha_hat_does_not_use_alpha_true() -> None:
    data = generate_data(Design("dgp1", n=200, p=50, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 3)

    without_truth = estimate_oracle_ivqr(
        data.y,
        data.d,
        data.x,
        data.z,
        tau=0.5,
        alpha_candidates=alphas,
        oracle_indices=np.arange(10),
    )
    with_wrong_truth = estimate_oracle_ivqr(
        data.y,
        data.d,
        data.x,
        data.z,
        tau=0.5,
        alpha_candidates=alphas,
        oracle_indices=np.arange(10),
        alpha_true=999.0,
    )

    assert without_truth.alpha_hat == pytest.approx(with_wrong_truth.alpha_hat)
    assert without_truth.objective_value == pytest.approx(with_wrong_truth.objective_value)
    assert without_truth.selected_controls == 10
    assert with_wrong_truth.selected_controls == 10


def test_estimate_oracle_ivqr_passes_max_iter_to_common_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_data(Design("dgp1", n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123))
    captured: dict[str, int] = {}

    def fake_common_estimator(data, x_controls, estimator_name, **kwargs):
        captured["max_iter"] = kwargs["max_iter"]
        captured["selected_controls"] = kwargs["selected_controls"]
        return EstimationResult(
            estimator=estimator_name,
            alpha_hat=1.0,
            alpha_true=data.alpha_true,
            tau=kwargs["tau"],
            converged=True,
            failed=False,
            message="ok",
            objective_value=0.0,
            at_grid_boundary=False,
            alpha_grid_size=len(kwargs["alphas"]),
            failed_alpha_count=0,
            cr_lower=None,
            cr_upper=None,
            cr_length=None,
            cr_covers_true=None,
            cr_empty=True,
            cr_disconnected=False,
            selected_controls=None,
            runtime_seconds=0.0,
        )

    import estimators.oracle_ivqr as oracle_module

    monkeypatch.setattr(oracle_module, "estimate_ch_ivqr_controls", fake_common_estimator)

    result = estimate_oracle_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        oracle_indices=np.arange(10),
        max_iter=2000,
    )

    assert result.estimator == "oracle"
    assert captured["max_iter"] == 2000
    assert captured["selected_controls"] == 10


def test_estimate_oracle_ivqr_rejects_invalid_indices() -> None:
    data = generate_data(Design("dgp1", n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123))

    with pytest.raises(ValueError, match="nonempty"):
        estimate_oracle_ivqr(data, tau=0.5, alphas=np.linspace(0.0, 2.0, 3), oracle_indices=[])
    with pytest.raises(ValueError, match="duplicates"):
        estimate_oracle_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            oracle_indices=np.array([0, 0]),
        )
    with pytest.raises(ValueError, match="between"):
        estimate_oracle_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            oracle_indices=np.array([20]),
        )

