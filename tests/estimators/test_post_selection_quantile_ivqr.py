"""Tests for the experimental quantile-specific post-selection IVQR estimator."""

import numpy as np
import pytest

from dgp import generate_data
from dgp.designs import Design
import estimators.post_selection_quantile_ivqr as psq_module
from estimators.post_selection_quantile_ivqr import (
    estimate_post_selection_quantile_ivqr,
    select_controls_quantile_y,
    summarize_quantile_post_selection_diagnostics,
)


def test_select_controls_quantile_y_returns_valid_indices() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(36, 4))
    y = x[:, 0] + rng.normal(scale=0.1, size=36)

    result = select_controls_quantile_y(
        y,
        x,
        tau=0.5,
        candidate_alphas=(0.001, 0.01),
        cv=3,
        random_state=123,
    )

    assert result.selected_indices.ndim == 1
    assert np.issubdtype(result.selected_indices.dtype, np.integer)
    assert np.all(result.selected_indices >= 0)
    assert np.all(result.selected_indices < x.shape[1])
    assert result.alpha_selected in {0.001, 0.01}
    assert set(result.candidate_losses).issubset({0.001, 0.01})


def test_select_controls_quantile_y_handles_empty_selection() -> None:
    x = np.empty((20, 0))
    y = np.ones(20)

    result = select_controls_quantile_y(
        y,
        x,
        tau=0.5,
        candidate_alphas=(0.01,),
        cv=2,
    )

    assert result.selected_indices.shape == (0,)
    assert result.alpha_selected is None
    assert result.candidate_losses == {}


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_select_controls_quantile_y_rejects_invalid_tau(tau: float) -> None:
    with pytest.raises(ValueError, match="tau must satisfy"):
        select_controls_quantile_y(
            np.ones(20),
            np.ones((20, 2)),
            tau=tau,
            candidate_alphas=(0.01,),
            cv=2,
        )


def test_select_controls_quantile_y_failed_fits_raise_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        def fit(self, x, y):
            raise RuntimeError("forced fit failure")

    monkeypatch.setattr(psq_module, "_quantile_model", lambda tau, penalty: FailingModel())

    with pytest.raises(RuntimeError, match="All quantile-selection fits failed"):
        select_controls_quantile_y(
            np.ones(12),
            np.ones((12, 2)),
            tau=0.5,
            candidate_alphas=(0.01, 0.1),
            cv=3,
        )


def test_summarize_quantile_post_selection_diagnostics() -> None:
    diagnostics = summarize_quantile_post_selection_diagnostics(
        tau=0.25,
        n_controls=10,
        selected_quantile_y=[0, 1],
        selected_treatment_d=[1, 3],
        selected_union=[0, 1, 3],
        quantile_alpha_selected=0.01,
        quantile_cv_folds=3,
    )

    assert diagnostics["psq_selection_method"] == "quantile_l1_cv"
    assert diagnostics["psq_quantile_tau"] == pytest.approx(0.25)
    assert diagnostics["psq_quantile_alpha_selected"] == pytest.approx(0.01)
    assert diagnostics["psq_quantile_cv_folds"] == 3
    assert diagnostics["psq_n_selected_controls_quantile_y"] == 2
    assert diagnostics["psq_n_selected_controls_treatment_d"] == 2
    assert diagnostics["psq_n_selected_controls_union"] == 3
    assert diagnostics["psq_share_selected_controls_quantile_y"] == pytest.approx(0.2)
    assert diagnostics["psq_share_selected_controls_union"] == pytest.approx(0.3)
    assert diagnostics["psq_selection_failed"] is False


def test_estimate_post_selection_quantile_ivqr_returns_diagnostics() -> None:
    data = generate_data(
        Design("dgp1", n=80, p=6, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.linspace(0.0, 2.0, 5)

    result = estimate_post_selection_quantile_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_cv=2,
        quantile_selection_cv=2,
        quantile_selection_alphas=(0.01, 0.1),
    )

    assert result.estimator == "post_selection_quantile"
    assert result.failed is False
    assert result.alpha_true == pytest.approx(data.alpha_true)
    assert result.alpha_grid_size == len(alphas)
    assert result.selected_controls == result.psq_n_selected_controls_union
    assert result.ps_selection_method == "quantile_specific"
    assert result.psq_selection_method == "quantile_l1_cv"
    assert result.psq_quantile_tau == pytest.approx(0.5)
    assert result.psq_quantile_alpha_selected in {0.01, 0.1}
    assert result.ps_n_selected_controls == result.psq_n_selected_controls_union
    assert result.ps_instrument_selection_method == "all_instruments_retained"
    assert result.ps_n_candidate_instruments == 1
    assert result.ps_n_retained_instruments == 1
    assert result.ps_all_instruments_retained is True
    assert result.cr_disconnected is not None
    assert result.cr_hull_length is not None
    assert result.runtime_total_sec == pytest.approx(result.runtime_seconds)
    assert result.psq_runtime_total_sec == pytest.approx(result.runtime_seconds)
    assert result.psq_runtime_quantile_selection_sec is not None
    assert result.psq_runtime_quantile_selection_sec >= 0.0
    assert result.psq_runtime_treatment_selection_sec is not None
    assert result.psq_runtime_treatment_selection_sec >= 0.0
    assert result.psq_runtime_alpha_loop_sec is not None
    assert result.psq_runtime_alpha_loop_sec >= 0.0
    assert result.psq_runtime_confidence_region_sec is not None
    assert result.psq_runtime_confidence_region_sec >= 0.0
    assert result.psq_runtime_diagnostics_sec is not None
    assert result.psq_runtime_diagnostics_sec >= 0.0


def test_estimate_post_selection_quantile_selection_failure_returns_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=4, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    def fail_quantile_selection(*args, **kwargs):
        raise RuntimeError("forced quantile failure")

    monkeypatch.setattr(
        psq_module,
        "select_controls_quantile_y",
        fail_quantile_selection,
    )

    result = estimate_post_selection_quantile_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        selection_cv=2,
        quantile_selection_cv=2,
    )

    assert result.estimator == "post_selection_quantile"
    assert result.failed is True
    assert "forced quantile failure" in result.message
    assert result.ps_selection_method == "quantile_specific"
    assert result.ps_selection_failed is True
    assert result.psq_selection_failed is True
    assert result.psq_warning_code == "quantile_selection_failed"
    assert result.psq_runtime_total_sec is not None
    assert np.isfinite(result.psq_runtime_total_sec)
