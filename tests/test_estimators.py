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


def test_add_intercept_prepends_ones() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    x_design = add_intercept(x)

    assert x_design.shape == (2, 3)
    assert np.allclose(x_design[:, 0], 1.0)
    assert np.allclose(x_design[:, 1:], x)


def test_add_intercept_rejects_zero_rows() -> None:
    with pytest.raises(ValueError, match="x must contain at least one row"):
        add_intercept(np.empty((0, 2)))


def test_as_2d_instruments_rejects_zero_rows() -> None:
    with pytest.raises(ValueError, match="z must contain at least one row"):
        ch_inverse_ivqr.as_2d_instruments(np.empty((0, 1)))


def test_wald_statistic_uses_covariance_scaling_without_extra_n() -> None:
    gamma_hat = np.array([2.0])
    # cov_gamma estimates Var(gamma_hat), so its inverse already includes
    # the CH sample-size scaling and no additional factor of n is applied.
    cov_gamma = np.array([[4.0]])

    assert ch_inverse_ivqr.wald_statistic(gamma_hat, cov_gamma) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize("alpha", [np.nan, np.inf, -np.inf])
def test_evaluate_alpha_ch_ivqr_rejects_nonfinite_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be finite"):
        ch_inverse_ivqr.evaluate_alpha_ch_ivqr(
            y=np.arange(6, dtype=float),
            d=np.ones(6),
            x_controls=np.ones((6, 2)),
            z=np.arange(6, dtype=float),
            alpha=alpha,
            tau=0.5,
        )


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        (
            {"runtime_seconds": -1.0},
            "runtime_seconds must be finite and nonnegative",
        ),
        ({"alpha_grid_size": 0}, "alpha_grid_size must be at least 1"),
        ({"failed_alpha_count": -1}, "failed_alpha_count must be nonnegative"),
        (
            {"alpha_grid_size": 2, "failed_alpha_count": 3},
            "failed_alpha_count must not exceed alpha_grid_size",
        ),
    ],
)
def test_failed_ch_ivqr_result_rejects_invalid_diagnostics(
    diagnostics: dict[str, int | float],
    message: str,
) -> None:
    data = generate_data(
        Design("dgp1", n=20, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    arguments: dict[str, object] = {
        "data": data,
        "tau": 0.5,
        "estimator": "test",
        "message": "failed",
        "runtime_seconds": 0.0,
        "alpha_grid_size": 3,
        "failed_alpha_count": 1,
    }
    arguments.update(diagnostics)

    with pytest.raises(ValueError, match=message):
        ch_inverse_ivqr.failed_ch_ivqr_result(**arguments)


def test_evaluate_full_control_ivqr_alpha_returns_finite_statistic() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alpha_true = require_float(data.alpha_true, "alpha_true")

    evaluation = ch_inverse_ivqr.evaluate_alpha_ch_ivqr(
        y=data.y,
        d=data.d,
        x_controls=data.x,
        z=data.z,
        alpha=alpha_true,
        tau=0.5,
    )

    assert evaluation.message == "ok"
    assert evaluation.converged is True
    assert evaluation.dim_z == 1
    assert evaluation.gamma_hat.shape == (1,)
    assert evaluation.cov_gamma.shape == (1, 1)
    assert np.isfinite(evaluation.statistic)
    assert evaluation.statistic >= 0.0


def test_ch_ivqr_design_includes_controls_and_excluded_instruments() -> None:
    x_controls = np.ones((8, 5))
    z = np.arange(8, dtype=float)

    design, z_block = ch_inverse_ivqr.ch_ivqr_design(x_controls, z)

    assert design.shape == (8, 7)
    assert z_block == slice(6, 7)
    assert np.allclose(design[:, 0], 1.0)
    assert np.allclose(design[:, 1:6], x_controls)
    assert np.allclose(design[:, 6], z)


def test_ch_ivqr_evaluator_extracts_gamma_after_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[int, int]] = {}

    class FakeQuantReg:
        def __init__(self, y_alpha, design):
            captured["design_shape"] = design.shape

        def fit(self, q, max_iter):
            class Result:
                params = np.array([0.0, 10.0, 20.0, 2.0])

                @staticmethod
                def cov_params():
                    return np.diag([1.0, 1.0, 1.0, 4.0])

            return Result()

    monkeypatch.setattr(ch_inverse_ivqr, "QuantReg", FakeQuantReg)

    evaluation = ch_inverse_ivqr.evaluate_alpha_ch_ivqr(
        y=np.arange(6, dtype=float),
        d=np.ones(6),
        x_controls=np.ones((6, 2)),
        z=np.arange(6, dtype=float),
        alpha=1.0,
        tau=0.5,
        max_iter=100,
    )

    assert captured["design_shape"] == (6, 4)
    assert evaluation.dim_z == 1
    assert evaluation.gamma_hat.tolist() == [2.0]
    assert evaluation.statistic == pytest.approx(1.0)


def test_ch_ivqr_evaluator_marks_iteration_limit_as_nonconverged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IterationLimitedQuantReg:
        def __init__(self, y_alpha, design):
            self.design = design

        def fit(self, q, max_iter):
            warnings.warn(
                "Maximum number of iterations reached",
                IterationLimitWarning,
            )

            class Result:
                params = np.zeros(4)

                @staticmethod
                def cov_params():
                    return np.eye(4)

            return Result()

    monkeypatch.setattr(
        ch_inverse_ivqr,
        "QuantReg",
        IterationLimitedQuantReg,
    )

    evaluation = ch_inverse_ivqr.evaluate_alpha_ch_ivqr(
        y=np.arange(6, dtype=float),
        d=np.ones(6),
        x_controls=np.ones((6, 2)),
        z=np.arange(6, dtype=float),
        alpha=1.0,
        tau=0.5,
        max_iter=10,
    )

    assert evaluation.converged is False
    assert evaluation.statistic == np.inf
    assert np.all(np.isnan(evaluation.gamma_hat))
    assert np.all(np.isnan(evaluation.cov_gamma))
    assert evaluation.message == "QuantReg reached iteration limit"


def test_estimate_full_control_ivqr_returns_estimation_result() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.linspace(0.0, 2.0, 11)

    result = estimate_full_control_ivqr(data, tau=0.5, alphas=alphas, gmm_ridge=1e-6)

    assert result.estimator == "full_control_ivqr"
    assert result.alpha_true == pytest.approx(data.alpha_true)
    assert result.tau == 0.5
    assert result.failed is False
    assert result.alpha_hat is not None
    assert result.objective_value is not None
    assert np.isfinite(result.objective_value)
    assert result.cr_disconnected is not None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == 0
    assert result.runtime_seconds >= 0.0


def test_full_control_delegates_all_controls_to_ch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.full_control_ivqr as full_control_module

    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.linspace(0.0, 2.0, 3)
    captured: dict[str, object] = {}

    def fake_ch_estimator(**kwargs):
        captured.update(kwargs)
        return EstimationResult(
            estimator=kwargs["estimator_name"],
            alpha_hat=1.0,
            alpha_true=kwargs["data"].alpha_true,
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
            selected_controls=kwargs["selected_controls"],
            runtime_seconds=0.0,
        )

    monkeypatch.setattr(
        full_control_module,
        "estimate_ch_ivqr_controls",
        fake_ch_estimator,
    )

    result = estimate_full_control_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        alpha_min=-1.0,
        alpha_max=3.0,
        alpha_step=0.25,
        confidence_level=0.9,
        max_iter=2000,
        gmm_ridge=1e-4,
    )

    assert captured["data"] is data
    assert captured["tau"] == 0.5
    assert captured["x_controls"] is data.x
    assert captured["estimator_name"] == "full_control_ivqr"
    np.testing.assert_array_equal(captured["alphas"], alphas)
    assert captured["alpha_min"] == -1.0
    assert captured["alpha_max"] == 3.0
    assert captured["alpha_step"] == 0.25
    assert captured["confidence_level"] == 0.9
    assert captured["max_iter"] == 2000
    assert captured["selected_controls"] == data.x.shape[1]
    assert result.selected_controls == data.x.shape[1]


def test_estimate_full_control_ivqr_infeasible_high_dimensional_case_fails_cleanly() -> None:
    design = Design("dgp1", n=20, p=25, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    result = estimate_full_control_ivqr(data, tau=0.5, alphas=np.linspace(0.0, 2.0, 5))

    assert result.estimator == "full_control_ivqr"
    assert result.failed is True
    assert result.converged is False
    assert result.alpha_hat is None
    assert "regressors=27" in result.message


def test_estimate_full_control_ivqr_some_failed_alphas_still_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.array([0.0, 1.0, 2.0])
    original = ch_inverse_ivqr.evaluate_alpha_ch_ivqr

    def fake_evaluate(*args: object, **kwargs: object):
        if kwargs["alpha"] == 1.0:
            return ch_inverse_ivqr.AlphaEvaluation(
                statistic=np.inf,
                gamma_hat=np.array([np.nan]),
                cov_gamma=np.array([[np.nan]]),
                dim_z=1,
                converged=False,
                message="forced failure",
            )
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ch_inverse_ivqr, "evaluate_alpha_ch_ivqr", fake_evaluate)

    result = estimate_full_control_ivqr(data, tau=0.5, alphas=alphas)

    assert result.failed is False
    assert result.converged is True
    assert result.alpha_hat is not None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == 1
    assert "failed_alpha_points=1/3" in result.message


def test_estimate_full_control_ivqr_all_failed_alphas_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.array([0.0, 1.0, 2.0])

    def fake_evaluate(*args: object, **kwargs: object):
        return ch_inverse_ivqr.AlphaEvaluation(
            statistic=np.inf,
            gamma_hat=np.array([np.nan]),
            cov_gamma=np.array([[np.nan]]),
            dim_z=1,
            converged=False,
            message="forced failure",
        )

    monkeypatch.setattr(ch_inverse_ivqr, "evaluate_alpha_ch_ivqr", fake_evaluate)

    result = estimate_full_control_ivqr(data, tau=0.5, alphas=alphas)

    assert result.failed is True
    assert result.converged is False
    assert result.alpha_hat is None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == len(alphas)
    assert "All alpha-grid evaluations failed" in result.message


def test_estimate_full_control_ivqr_invalid_tau_raises_value_error() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    with pytest.raises(ValueError):
        estimate_full_control_ivqr(data, tau=0.0, alphas=np.linspace(0.0, 2.0, 5))


@pytest.mark.parametrize("max_iter", [0, True, 1.5])
def test_estimate_full_control_ivqr_invalid_max_iter_raises_value_error(
    max_iter: object,
) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    with pytest.raises(ValueError, match="max_iter must"):
        estimate_full_control_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 5),
            max_iter=max_iter,
        )


@pytest.mark.parametrize("gmm_ridge", [-1.0, np.nan, np.inf, True])
def test_estimate_full_control_ivqr_rejects_invalid_gmm_ridge(
    gmm_ridge: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="gmm_ridge must be finite and nonnegative"):
        estimate_full_control_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            gmm_ridge=gmm_ridge,
        )


@pytest.mark.parametrize(
    "confidence_level",
    [0.0, 1.0, -0.1, np.nan, np.inf, True],
)
def test_estimate_full_control_ivqr_rejects_invalid_confidence_level(
    confidence_level: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(
        ValueError,
        match="confidence_level must satisfy 0 < confidence_level < 1",
    ):
        estimate_full_control_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"alpha_min": np.nan}, "alpha grid bounds must be finite"),
        ({"alpha_max": np.inf}, "alpha grid bounds must be finite"),
        ({"alpha_step": np.nan}, "alpha grid bounds must be finite"),
        ({"alpha_step": 0.0}, "alpha_step must be positive"),
        (
            {"alpha_step": True},
            "alpha grid bounds must be finite numeric values",
        ),
        (
            {"alpha_min": 1.0, "alpha_max": 1.0},
            "alpha_max must be greater than alpha_min",
        ),
    ],
)
def test_estimate_full_control_ivqr_rejects_invalid_alpha_grid_bounds(
    arguments: dict[str, object],
    message: str,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match=message):
        estimate_full_control_ivqr(
            data,
            tau=0.5,
            **arguments,
        )


def test_estimate_full_control_ivqr_passes_max_iter_to_quantreg_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alpha_true = require_float(data.alpha_true, "alpha_true")
    captured: dict[str, int] = {}
    original = ch_inverse_ivqr.evaluate_alpha_ch_ivqr

    def fake_evaluate_alpha_ch_ivqr(*args, **kwargs):
        captured["max_iter"] = kwargs["max_iter"]
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ch_inverse_ivqr,
        "evaluate_alpha_ch_ivqr",
        fake_evaluate_alpha_ch_ivqr,
    )

    result = estimate_full_control_ivqr(
        data,
        tau=0.5,
        alphas=np.array([alpha_true]),
        max_iter=2000,
        gmm_ridge=1e-6,
    )

    assert result.failed is False
    assert captured["max_iter"] == 2000


def test_estimate_full_control_ivqr_feasible_high_pn_emits_no_pn_warning() -> None:
    design = Design("dgp1", n=30, p=20, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alpha_true = require_float(data.alpha_true, "alpha_true")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate_full_control_ivqr(
            data,
            tau=0.5,
            alphas=np.array([alpha_true]),
            max_iter=50,
            gmm_ridge=1e-6,
        )

    messages = [str(warning.message).lower() for warning in caught]
    assert not any("p/n" in message or "high-dimensional" in message for message in messages)


def test_estimate_full_control_ivqr_confidence_region_fields_are_coherent() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    result = estimate_full_control_ivqr(data, tau=0.5, alphas=np.linspace(0.0, 2.0, 11))

    if result.cr_empty is False:
        assert result.cr_lower is not None
        assert result.cr_upper is not None
        assert result.cr_length is not None
        assert result.cr_upper >= result.cr_lower


def test_estimate_full_control_ivqr_output_is_deterministic() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.linspace(0.0, 2.0, 11)

    result_1 = estimate_full_control_ivqr(data, tau=0.5, alphas=alphas, gmm_ridge=1e-6)
    result_2 = estimate_full_control_ivqr(data, tau=0.5, alphas=alphas, gmm_ridge=1e-6)

    assert result_1.alpha_hat is not None
    assert result_2.alpha_hat is not None
    assert result_1.objective_value is not None
    assert result_2.objective_value is not None
    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
    assert result_1.cr_empty is result_2.cr_empty
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


def test_full_control_gmm_ridge_is_compatibility_only() -> None:
    data = generate_data(
        Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.linspace(0.0, 2.0, 11)

    result_1 = estimate_full_control_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        gmm_ridge=0.0,
    )
    result_2 = estimate_full_control_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        gmm_ridge=1e-4,
    )

    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
    if result_1.cr_lower is None:
        assert result_2.cr_lower is None
    else:
        assert result_2.cr_lower == pytest.approx(result_1.cr_lower)
    if result_1.cr_upper is None:
        assert result_2.cr_upper is None
    else:
        assert result_2.cr_upper == pytest.approx(result_1.cr_upper)


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

def test_make_folds_covers_each_observation_once() -> None:
    folds = make_folds(n=20, k_folds=5, random_state=123)

    assert len(folds) == 5

    test_indices = []
    for train_idx, test_idx in folds:
        assert np.intersect1d(train_idx, test_idx).size == 0
        test_indices.extend(test_idx.tolist())

    assert np.array_equal(np.sort(test_indices), np.arange(20))
    assert np.all(np.bincount(test_indices, minlength=20) == 1)


def test_standardize_train_test_uses_training_moments() -> None:
    x_train = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    x_test = np.array([[70.0, 140.0], [90.0, 180.0]])

    x_train_scaled, x_test_scaled, scaler = standardize_train_test(x_train, x_test)

    assert np.allclose(x_train_scaled.mean(axis=0), 0.0)
    assert np.allclose(x_train_scaled.std(axis=0), 1.0)
    assert x_test_scaled.shape == x_test.shape
    scaler_mean = require_array(scaler.mean_, "scaler.mean_")
    np.testing.assert_allclose(scaler_mean, x_train.mean(axis=0))
    assert not np.allclose(scaler_mean, np.vstack([x_train, x_test]).mean(axis=0))


def test_fit_quantile_nuisance_returns_fitted_model() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alpha_true = require_float(data.alpha_true, "alpha_true")

    model, converged, message = fit_quantile_nuisance(
        data.y,
        data.d,
        data.x,
        alpha_value=alpha_true,
        tau=0.5,
        penalty=0.01,
    )

    assert message == "ok"
    assert converged is True
    assert model is not None
    predictions = model.predict(data.x)
    assert predictions.shape == (data.x.shape[0],)
    assert np.all(np.isfinite(predictions))


def test_fit_instrument_residualizer_returns_fitted_model() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))

    model, converged, message = fit_instrument_residualizer(
        data.z,
        data.x,
        ridge_alpha=1.0,
    )

    assert message == "ok"
    assert converged is True
    assert model is not None
    predictions = model.predict(data.x)
    residuals = data.z - predictions
    assert predictions.shape == (data.x.shape[0],)
    assert residuals.shape == (data.x.shape[0],)
    assert np.all(np.isfinite(predictions))
    assert np.all(np.isfinite(residuals))


def test_evaluate_dml_ivqr_alpha_returns_finite_statistic() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alpha_true = require_float(data.alpha_true, "alpha_true")

    statistic, converged, message = evaluate_dml_ivqr_alpha(
        data.y,
        data.d,
        data.z,
        data.x,
        alpha_value=alpha_true,
        tau=0.5,
        k_folds=3,
        fold_random_state=123,
        gmm_ridge=1e-6,
    )

    assert message == "ok"
    assert converged is True
    assert np.isfinite(statistic)
    assert statistic >= 0.0


@pytest.mark.parametrize("alpha_value", [np.nan, np.inf, -np.inf, True])
def test_evaluate_dml_ivqr_alpha_rejects_nonfinite_alpha(
    alpha_value: float,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="alpha_value must be finite"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=alpha_value,
            tau=0.5,
            k_folds=3,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("quantile_penalty", np.nan),
        ("quantile_penalty", -1.0),
        ("quantile_penalty", True),
        ("ridge_alpha", np.nan),
        ("ridge_alpha", -1.0),
        ("ridge_alpha", True),
        ("gmm_ridge", np.nan),
        ("gmm_ridge", -1.0),
        ("gmm_ridge", True),
    ],
)
def test_evaluate_dml_ivqr_alpha_rejects_invalid_penalties(
    argument: str,
    value: float,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match=f"{argument} must"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
            **{argument: value},
        )


@pytest.mark.parametrize("solver", ["warn", "bad_solver"])
def test_evaluate_dml_ivqr_alpha_rejects_invalid_solver(solver: str) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="Unknown quantile solver"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
            quantile_solver=solver,
        )


def test_evaluate_dml_ivqr_accepts_single_column_instrument() -> None:
    data = generate_data(
        Design("dgp1", n=60, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    statistic, converged, message = evaluate_dml_ivqr_alpha(
        data.y,
        data.d,
        data.z[:, None],
        data.x,
        alpha_value=1.0,
        tau=0.5,
        k_folds=3,
    )

    assert converged is True
    assert message == "ok"
    assert np.isfinite(statistic)


def test_evaluate_dml_ivqr_rejects_multiple_instruments() -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    z_multiple = np.column_stack([data.z, data.z])

    with pytest.raises(
        ValueError,
        match="DML-IVQR currently supports exactly one excluded instrument",
    ):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            z_multiple,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
        )


def test_evaluate_dml_ivqr_rejects_zero_control_matrix() -> None:
    n = 30

    with pytest.raises(
        ValueError,
        match="DML-IVQR requires at least one control column",
    ):
        evaluate_dml_ivqr_alpha(
            np.ones(n),
            np.zeros(n),
            np.arange(n, dtype=float),
            np.empty((n, 0)),
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
        )


@pytest.mark.parametrize("use_cache", ["False", 1])
def test_evaluate_dml_ivqr_rejects_nonboolean_use_cache(
    use_cache: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="use_cache must be a boolean"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
            use_cache=use_cache,
        )


def test_dml_fold_cache_has_disjoint_train_test_splits() -> None:
    data = generate_data(Design("dgp1", n=90, p=8, pi=1.0, tau=0.5, rep=0, seed=123))

    cache = _build_dml_fold_cache(
        data.y,
        data.d,
        data.z,
        data.x,
        k_folds=3,
        random_state=123,
        ridge_alpha=1.0,
    )

    test_counts = np.zeros(data.x.shape[0], dtype=int)
    for fold in cache:
        assert np.intersect1d(fold.train_idx, fold.test_idx).size == 0
        assert fold.x_train_scaled.shape[0] == len(fold.train_idx)
        assert fold.x_test_scaled.shape[0] == len(fold.test_idx)
        assert fold.z_resid_test.shape == (len(fold.test_idx),)
        assert np.all(np.isfinite(fold.z_resid_test))
        test_counts[fold.test_idx] += 1

    assert np.all(test_counts == 1)


def test_dml_cached_and_uncached_alpha_statistics_match() -> None:
    data = generate_data(Design("dgp1", n=90, p=8, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.array([-0.5, 0.0, 0.5])

    cached_stats = []
    uncached_stats = []
    for alpha_value in alphas:
        cached_stat, cached_converged, cached_message = evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=float(alpha_value),
            tau=0.5,
            k_folds=3,
            fold_random_state=123,
            quantile_penalty=0.01,
            gmm_ridge=1e-6,
            use_cache=True,
        )
        uncached_stat, uncached_converged, uncached_message = (
            _evaluate_dml_ivqr_alpha_uncached(
                data.y,
                data.d,
                data.z,
                data.x,
                alpha_value=float(alpha_value),
                tau=0.5,
                k_folds=3,
                fold_random_state=123,
                quantile_penalty=0.01,
                gmm_ridge=1e-6,
            )
        )

        assert cached_message == uncached_message == "ok"
        assert cached_converged is uncached_converged is True
        cached_stats.append(cached_stat)
        uncached_stats.append(uncached_stat)

    np.testing.assert_allclose(cached_stats, uncached_stats, rtol=1e-12, atol=1e-12)


def test_estimate_dml_ivqr_returns_estimation_result() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 9)

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
    )

    assert result.estimator == "dml_ivqr"
    assert result.alpha_true == pytest.approx(data.alpha_true)
    assert result.tau == 0.5
    assert result.failed is False
    assert result.alpha_hat is not None
    assert result.objective_value is not None
    assert np.isfinite(result.objective_value)
    assert result.cr_disconnected is not None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == 0
    assert result.runtime_seconds >= 0.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"alpha_min": np.nan}, "alpha_min must be finite"),
        ({"alpha_max": np.inf}, "alpha_max must be finite"),
        ({"alpha_step": np.nan}, "alpha_step must be finite and positive"),
        ({"alpha_step": 0.0}, "alpha_step must be finite and positive"),
        ({"alpha_step": True}, "alpha_step must be finite and positive"),
        (
            {"alpha_min": 1.0, "alpha_max": 1.0},
            "alpha_max must exceed alpha_min",
        ),
        (
            {"alpha_min": 2.0, "alpha_max": 1.0},
            "alpha_max must exceed alpha_min",
        ),
    ],
)
def test_estimate_dml_ivqr_rejects_invalid_alpha_grid_bounds(
    arguments: dict[str, object],
    message: str,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match=message):
        estimate_dml_ivqr(
            data,
            tau=0.5,
            k_folds=3,
            **arguments,
        )


@pytest.mark.parametrize("use_cache", ["False", 1])
def test_estimate_dml_ivqr_rejects_nonboolean_use_cache(
    use_cache: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="use_cache must be a boolean"):
        estimate_dml_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            k_folds=3,
            use_cache=use_cache,
        )


@pytest.mark.parametrize("fold_random_state", [True, 1.5, "123"])
def test_estimate_dml_ivqr_rejects_invalid_fold_random_state(
    fold_random_state: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(
        ValueError,
        match="fold_random_state must be an integer or None",
    ):
        estimate_dml_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            k_folds=3,
            fold_random_state=fold_random_state,
        )


def test_estimate_dml_ivqr_uses_absolute_score_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.array([0.0, 1.0, 2.0])
    statistics = {0.0: 5.0, 1.0: 6.0, 2.0: 10.0}
    captured: dict[str, object] = {}
    original_invert = dml_module.invert_score_test

    monkeypatch.setattr(dml_module, "_build_dml_fold_cache", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        dml_module,
        "_evaluate_dml_ivqr_alpha_with_cache",
        lambda **kwargs: (statistics[kwargs["alpha_value"]], True, "ok"),
    )
    monkeypatch.setattr(
        dml_module,
        "critical_value_chi_square",
        lambda confidence_level, df: (
            captured.update({"critical_df": df}) or 3.84
        ),
    )

    def capture_inversion(**kwargs):
        captured.update(kwargs)
        return original_invert(**kwargs)

    monkeypatch.setattr(dml_module, "invert_score_test", capture_inversion)

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        use_cache=True,
    )

    assert captured["statistic_reference"] is None
    assert captured["inversion_type"] == "absolute"
    assert captured["critical_df"] == 1
    assert result.cr_empty is True


def test_estimate_dml_ivqr_cache_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    calls = {"cache": 0, "uncached": 0}

    def fail_cache(*args, **kwargs):
        calls["cache"] += 1
        raise RuntimeError("forced cache failure")

    def record_uncached(*args, **kwargs):
        calls["uncached"] += 1
        return 0.0, True, "ok"

    monkeypatch.setattr(dml_module, "_build_dml_fold_cache", fail_cache)
    monkeypatch.setattr(
        dml_module,
        "_evaluate_dml_ivqr_alpha_uncached",
        record_uncached,
    )

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        k_folds=3,
        use_cache=True,
    )

    assert result.failed is True
    assert "Fold cache construction failed: forced cache failure" in result.message
    assert calls == {"cache": 1, "uncached": 0}


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
def test_dml_failed_result_rejects_invalid_diagnostics(
    diagnostics: dict[str, int | float],
    message: str,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=20, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    arguments: dict[str, object] = {
        "data": data,
        "tau": 0.5,
        "message": "failed",
        "runtime_seconds": 0.0,
        "alpha_grid_size": 3,
        "failed_alpha_count": 1,
    }
    arguments.update(diagnostics)

    with pytest.raises(ValueError, match=message):
        dml_module._failed_result(**arguments)


def test_estimate_dml_ivqr_all_alpha_points_fail_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.linspace(0.0, 2.0, 3)
    monkeypatch.setattr(
        dml_module,
        "_build_dml_fold_cache",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        dml_module,
        "_evaluate_dml_ivqr_alpha_with_cache",
        lambda **kwargs: (np.inf, False, "forced alpha failure"),
    )

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
    )

    assert result.failed is True
    assert result.failed_alpha_count == len(alphas)
    assert result.alpha_grid_size == len(alphas)
    assert "All alpha grid points failed" in result.message
    assert "first_failure=forced alpha failure" in result.message


def test_estimate_dml_ivqr_cached_and_uncached_results_match() -> None:
    data = generate_data(Design("dgp1", n=90, p=8, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.array([-0.5, 0.0, 0.5])

    cached = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
        use_cache=True,
    )
    uncached = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
        use_cache=False,
    )

    assert cached.failed is False
    assert uncached.failed is False
    assert cached.alpha_hat is not None
    assert uncached.alpha_hat is not None
    assert cached.objective_value is not None
    assert uncached.objective_value is not None
    assert cached.alpha_hat == pytest.approx(uncached.alpha_hat)
    assert cached.objective_value == pytest.approx(uncached.objective_value)
    if cached.cr_lower is None:
        assert uncached.cr_lower is None
    else:
        assert uncached.cr_lower is not None
        assert cached.cr_lower == pytest.approx(uncached.cr_lower)
    if cached.cr_upper is None:
        assert uncached.cr_upper is None
    else:
        assert uncached.cr_upper is not None
        assert cached.cr_upper == pytest.approx(uncached.cr_upper)
    if cached.cr_length is None:
        assert uncached.cr_length is None
    else:
        assert uncached.cr_length is not None
        assert cached.cr_length == pytest.approx(uncached.cr_length)
    assert cached.cr_covers_true is uncached.cr_covers_true
    assert cached.cr_empty is uncached.cr_empty
    assert cached.cr_disconnected is uncached.cr_disconnected


def test_estimate_dml_ivqr_invalid_k_folds_raises_value_error() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 5)

    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=0)
    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=1)
    with pytest.raises(ValueError, match="k_folds must be an integer"):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=True)
    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=data.x.shape[0] + 1)


def test_estimate_dml_ivqr_accepts_five_folds_for_robustness() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        k_folds=5,
        fold_random_state=123,
        quantile_penalty=0.01,
    )

    assert result.alpha_grid_size == 3
    assert result.estimator == "dml_ivqr"


def test_estimate_dml_ivqr_invalid_tau_raises_value_error() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))

    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.0, alphas=np.linspace(0.0, 2.0, 5))


def test_estimate_dml_ivqr_output_is_deterministic() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 9)

    result_1 = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
    )
    result_2 = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
    )

    assert result_1.alpha_hat is not None
    assert result_2.alpha_hat is not None
    assert result_1.objective_value is not None
    assert result_2.objective_value is not None
    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
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
