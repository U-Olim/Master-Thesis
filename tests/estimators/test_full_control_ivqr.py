"""Tests for the full-control IVQR estimator."""

from typing import cast

import numpy as np
import pytest
import warnings

from dgp import generate_data
from dgp.designs import Design, SimData
from estimators import ch_inverse_ivqr
from estimators.base import EstimationResult
from estimators.full_control_ivqr import estimate_full_control_ivqr


def _call_estimate_full_control_ivqr_with_objects(
    *,
    data: object,
    tau: object = 0.5,
    alphas: object = None,
    alpha_min: object = -2.0,
    alpha_max: object = 4.0,
    alpha_step: object = 0.05,
    confidence_level: object = 0.95,
    max_iter: object = 1000,
    gmm_ridge: object = 1e-8,
):
    return estimate_full_control_ivqr(
        data=cast(SimData, data),
        tau=cast(float, tau),
        alphas=cast(np.ndarray | None, alphas),
        alpha_min=cast(float, alpha_min),
        alpha_max=cast(float, alpha_max),
        alpha_step=cast(float, alpha_step),
        confidence_level=cast(float, confidence_level),
        max_iter=cast(int, max_iter),
        gmm_ridge=cast(float, gmm_ridge),
    )


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
        _call_estimate_full_control_ivqr_with_objects(
            data=data,
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
        _call_estimate_full_control_ivqr_with_objects(
            data=data,
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
        _call_estimate_full_control_ivqr_with_objects(
            data=data,
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
        _call_estimate_full_control_ivqr_with_objects(
            data=data,
            tau=0.5,
            **arguments,
        )


def test_estimate_full_control_ivqr_passes_max_iter_to_quantreg_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alpha_true = data.alpha_true
    assert alpha_true is not None, "alpha_true should not be None"
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
    alpha_true = data.alpha_true
    assert alpha_true is not None, "alpha_true should not be None"

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


