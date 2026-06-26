"""Tests for the CH inverse-IVQR estimator."""

from typing import cast

import numpy as np
import pytest
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp import generate_data
from dgp.designs import Design, SimData
from estimators import ch_inverse_ivqr
from estimators.ch_inverse_ivqr import add_intercept
from inference.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
)


def _call_failed_ch_ivqr_result_with_objects(
    *,
    data: object,
    tau: object,
    estimator: object,
    message: object,
    runtime_seconds: object,
    alpha_grid_size: object = None,
    failed_alpha_count: object = None,
    selected_controls: object = None,
):
    return ch_inverse_ivqr.failed_ch_ivqr_result(
        data=cast(SimData, data),
        tau=cast(float, tau),
        estimator=cast(str, estimator),
        message=cast(str, message),
        runtime_seconds=cast(float, runtime_seconds),
        alpha_grid_size=cast(int | None, alpha_grid_size),
        failed_alpha_count=cast(int | None, failed_alpha_count),
        selected_controls=cast(int | None, selected_controls),
    )


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
        _call_failed_ch_ivqr_result_with_objects(**arguments)


def test_evaluate_full_control_ivqr_alpha_returns_finite_statistic() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alpha_true = data.alpha_true
    assert alpha_true is not None, "alpha_true should not be None"

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


def test_ch_ivqr_controls_fallback_grid_uses_project_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_data(Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123))
    fallback_grid = np.array([-1.0, 0.0, 3.0])
    captured: dict[str, object] = {}

    def fake_alpha_grid(alpha_min: float, alpha_max: float, step: float) -> np.ndarray:
        captured["alpha_min"] = alpha_min
        captured["alpha_max"] = alpha_max
        captured["step"] = step
        return fallback_grid

    monkeypatch.setattr(ch_inverse_ivqr, "alpha_grid", fake_alpha_grid)
    monkeypatch.setattr(
        ch_inverse_ivqr,
        "evaluate_alpha_ch_ivqr",
        lambda **kwargs: ch_inverse_ivqr.AlphaEvaluation(
            statistic=abs(float(kwargs["alpha"])),
            gamma_hat=np.zeros(1),
            cov_gamma=np.eye(1),
            dim_z=1,
            converged=True,
            message="ok",
        ),
    )

    result = ch_inverse_ivqr.estimate_ch_ivqr_controls(
        data=data,
        tau=0.5,
        x_controls=data.x[:, :1],
        estimator_name="test",
    )

    assert captured == {
        "alpha_min": DEFAULT_ALPHA_MIN,
        "alpha_max": DEFAULT_ALPHA_MAX,
        "step": DEFAULT_ALPHA_STEP,
    }
    assert result.alpha_grid_size == len(fallback_grid)


def test_ch_ivqr_controls_explicit_alphas_override_fallback_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_data(Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123))
    explicit_alphas = np.array([-0.25, 0.0, 0.25])

    def fail_alpha_grid(*args, **kwargs):
        raise AssertionError("fallback alpha grid should not be constructed")

    monkeypatch.setattr(ch_inverse_ivqr, "alpha_grid", fail_alpha_grid)
    monkeypatch.setattr(
        ch_inverse_ivqr,
        "evaluate_alpha_ch_ivqr",
        lambda **kwargs: ch_inverse_ivqr.AlphaEvaluation(
            statistic=abs(float(kwargs["alpha"])),
            gamma_hat=np.zeros(1),
            cov_gamma=np.eye(1),
            dim_z=1,
            converged=True,
            message="ok",
        ),
    )

    result = ch_inverse_ivqr.estimate_ch_ivqr_controls(
        data=data,
        tau=0.5,
        x_controls=data.x[:, :1],
        estimator_name="test",
        alphas=explicit_alphas,
    )

    assert result.alpha_grid_size == len(explicit_alphas)


