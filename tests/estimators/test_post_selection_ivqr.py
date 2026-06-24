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


def test_select_controls_lasso_returns_valid_indices() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))

    selected, message = select_controls_lasso(
        data.y,
        data.d,
        data.x,
        tau=0.5,
        random_state=123,
        cv=3,
    )

    assert selected.ndim == 1
    assert np.issubdtype(selected.dtype, np.integer)
    assert np.all(selected >= 0)
    assert np.all(selected < data.x.shape[1])
    assert np.all(np.diff(selected) > 0) or selected.size <= 1
    assert isinstance(message, str)


def test_select_controls_lasso_handles_no_signal_artificial_data() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(40, 5))
    y = np.zeros(40)
    d = np.zeros(40)

    selected, message = select_controls_lasso(
        y,
        d,
        x,
        tau=0.5,
        random_state=123,
        cv=3,
    )

    assert selected.ndim == 1
    assert np.issubdtype(selected.dtype, np.integer)
    assert selected.size <= x.shape[1]
    assert isinstance(message, str)


@pytest.mark.parametrize("cv", [1, 41, True])
def test_select_controls_lasso_rejects_invalid_cv(cv: int) -> None:
    x = np.ones((40, 2))

    with pytest.raises(ValueError, match="cv must"):
        select_controls_lasso(
            np.ones(40),
            np.zeros(40),
            x,
            tau=0.5,
            cv=cv,
        )


@pytest.mark.parametrize("max_iter", [0, True])
def test_select_controls_lasso_rejects_invalid_max_iter(max_iter: int) -> None:
    x = np.ones((40, 2))

    with pytest.raises(ValueError, match="max_iter must"):
        select_controls_lasso(
            np.ones(40),
            np.zeros(40),
            x,
            tau=0.5,
            cv=3,
            max_iter=max_iter,
        )


def test_select_controls_lasso_handles_zero_control_matrix() -> None:
    selected, message = select_controls_lasso(
        np.ones(20),
        np.zeros(20),
        np.empty((20, 0)),
        tau=0.5,
        cv=3,
    )

    assert selected.shape == (0,)
    assert np.issubdtype(selected.dtype, np.integer)
    assert "selected_union=0" in message


def test_evaluate_post_selection_alpha_returns_finite_statistic() -> None:
    data = generate_data(Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123))
    x_selected = data.x[:, :3]
    alpha_true = require_float(data.alpha_true, "alpha_true")

    statistic, converged, message = evaluate_post_selection_alpha(
        data.y,
        data.d,
        data.z,
        x_selected,
        alpha=alpha_true,
        tau=0.5,
    )

    assert message == "ok"
    assert converged is True
    assert np.isfinite(statistic)
    assert statistic >= 0.0


def test_post_selection_alpha_uses_ch_ivqr_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[int, int]] = {}

    def fake_evaluate(**kwargs):
        captured["x_shape"] = kwargs["x_controls"].shape
        captured["z_shape"] = np.asarray(kwargs["z"]).shape
        return ch_inverse_ivqr.AlphaEvaluation(
            statistic=1.25,
            gamma_hat=np.array([0.5]),
            cov_gamma=np.array([[0.2]]),
            dim_z=1,
            converged=True,
            message="ok",
        )

    import estimators.post_selection_ivqr as post_module

    monkeypatch.setattr(post_module, "_evaluate_alpha_ch_ivqr", fake_evaluate)

    statistic, converged, message = evaluate_post_selection_alpha(
        y=np.arange(10, dtype=float),
        d=np.ones(10),
        z=np.arange(10, dtype=float),
        x_selected=np.ones((10, 3)),
        alpha=1.0,
        tau=0.5,
    )

    assert captured["x_shape"] == (10, 3)
    assert captured["z_shape"] == (10,)
    assert statistic == pytest.approx(1.25)
    assert converged is True
    assert message == "ok"


def test_estimate_post_selection_ivqr_returns_estimation_result() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 11)

    result = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_cv=3,
    )

    assert result.estimator == "post_selection_ivqr"
    assert result.alpha_true == pytest.approx(data.alpha_true)
    assert result.tau == 0.5
    assert result.failed is False
    assert result.alpha_hat is not None
    assert result.selected_controls is not None
    assert result.objective_value is not None
    assert np.isfinite(result.objective_value)
    assert result.cr_disconnected is not None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count is not None
    assert 0 < result.failed_alpha_count < len(alphas)
    assert (
        f"failed_alpha_points={result.failed_alpha_count}/{len(alphas)}"
        in result.message
    )
    assert result.runtime_seconds >= 0.0


def test_estimate_post_selection_ivqr_invalid_tau_raises_value_error() -> None:
    data = generate_data(Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123))

    with pytest.raises(ValueError):
        estimate_post_selection_ivqr(data, tau=0.0, alphas=np.linspace(0.0, 2.0, 5))


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("selection_cv", 1, "cv must"),
        ("selection_cv", 101, "cv must"),
        ("selection_cv", True, "cv must"),
        ("selection_max_iter", 0, "max_iter must"),
        ("selection_max_iter", True, "max_iter must"),
    ],
)
def test_estimate_post_selection_ivqr_rejects_invalid_selection_config(
    argument: str,
    value: int,
    message: str,
) -> None:
    data = generate_data(
        Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    kwargs = {argument: value}

    with pytest.raises(ValueError, match=message):
        estimate_post_selection_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 5),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        ({"runtime_seconds": -1.0}, "runtime_seconds must be nonnegative"),
        ({"alpha_grid_size": 0}, "alpha_grid_size must be at least 1"),
        ({"failed_alpha_count": -1}, "failed_alpha_count must be nonnegative"),
        (
            {"alpha_grid_size": 2, "failed_alpha_count": 3},
            "failed_alpha_count cannot exceed alpha_grid_size",
        ),
    ],
)
def test_post_selection_failed_result_rejects_invalid_diagnostics(
    diagnostics: dict[str, int | float],
    message: str,
) -> None:
    import estimators.post_selection_ivqr as post_module

    data = generate_data(
        Design("dgp1", n=20, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    arguments: dict[str, object] = {
        "data": data,
        "tau": 0.5,
        "message": "failed",
        "selected_controls": None,
        "runtime_seconds": 0.0,
        "alpha_grid_size": 3,
        "failed_alpha_count": 1,
    }
    arguments.update(diagnostics)

    with pytest.raises(ValueError, match=message):
        post_module._failed_result(**arguments)


def test_post_selection_qr_feasibility_counts_instruments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.post_selection_ivqr as post_module

    data = generate_data(
        Design("dgp1", n=5, p=3, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    def fake_select_controls(*args, **kwargs):
        return np.arange(3), "selected_y=3; selected_d=3; selected_union=3"

    monkeypatch.setattr(
        post_module,
        "select_controls_lasso",
        fake_select_controls,
    )

    result = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        selection_cv=2,
    )

    assert result.failed is True
    assert result.selected_controls == 3
    assert "QR design dimension is at least sample size" in result.message
    assert "regressors=5, n=5" in result.message


def test_estimate_post_selection_ivqr_output_is_deterministic() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 11)

    result_1 = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_random_state=123,
        selection_cv=3,
    )
    result_2 = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_random_state=123,
        selection_cv=3,
    )

    assert result_1.alpha_hat is not None
    assert result_2.alpha_hat is not None
    assert result_1.objective_value is not None
    assert result_2.objective_value is not None
    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.selected_controls == result_2.selected_controls
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
    assert result_1.failed_alpha_count == result_2.failed_alpha_count
    assert result_1.failed_alpha_count is not None
    assert result_1.failed_alpha_count > 0
    if result_1.cr_lower is None:
        assert result_2.cr_lower is None
    else:
        assert result_2.cr_lower is not None
        assert result_1.cr_lower == pytest.approx(result_2.cr_lower)
    if result_1.cr_upper is None:
        assert result_2.cr_upper is None
    else:
        assert result_2.cr_upper is not None
        assert result_1.cr_upper == pytest.approx(result_2.cr_upper)


