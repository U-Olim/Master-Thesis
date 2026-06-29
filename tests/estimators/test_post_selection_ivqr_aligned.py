"""Tests for the experimental IVQR-aligned post-selection estimator."""

import numpy as np
import pytest

from dgp import generate_data
from dgp.designs import Design
import estimators.post_selection_ivqr_aligned as psa_module
from estimators.post_selection_ivqr_aligned import (
    estimate_post_selection_ivqr_aligned,
    select_controls_ivqr_aligned,
    summarize_aligned_post_selection_diagnostics,
)


def test_select_controls_ivqr_aligned_unions_anchor_and_treatment_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.ones((12, 4))
    y = np.arange(12, dtype=float)
    d = np.ones(12)
    anchors = np.array([0.0, 1.0, 2.0])
    selected_by_anchor = {
        0.0: np.array([2, 0]),
        1.0: np.array([2]),
        2.0: np.array([3]),
    }

    def fake_quantile_y(y_tilde, x_arg, tau, **kwargs):
        anchor = float((y - y_tilde)[0])
        return type(
            "Selection",
            (),
            {
                "selected_indices": selected_by_anchor[anchor],
                "alpha_selected": 0.01,
                "message": "ok",
            },
        )()

    monkeypatch.setattr(psa_module, "select_controls_quantile_y", fake_quantile_y)
    monkeypatch.setattr(
        psa_module,
        "_select_controls_treatment_lasso",
        lambda *args, **kwargs: (np.array([1, 3]), 0.2),
    )

    result = select_controls_ivqr_aligned(y, d, x, 0.5, anchors, treatment_cv=2)

    np.testing.assert_array_equal(result.selected_anchor_union, np.array([0, 2, 3]))
    np.testing.assert_array_equal(result.selected_treatment, np.array([1, 3]))
    np.testing.assert_array_equal(result.selected_final, np.array([0, 1, 2, 3]))
    assert result.treatment_alpha == pytest.approx(0.2)


def test_summarize_aligned_post_selection_diagnostics() -> None:
    diagnostics = summarize_aligned_post_selection_diagnostics(
        n_controls=10,
        alpha_anchors=np.array([0.0, 1.0, 2.0]),
        selected_anchor_union=[0, 2],
        selected_treatment=[1],
        selected_final=[0, 1, 2],
        anchor_results=(),
        quantile_cv_folds=3,
        quantile_penalty_grid=(0.001, 0.01),
    )

    assert diagnostics["psa_selection_method"] == "ivqr_aligned_quantile_l1_cv"
    assert diagnostics["psa_anchor_rule"] == "grid_quartiles"
    assert diagnostics["psa_alpha_anchor_count"] == 3
    assert diagnostics["psa_alpha_anchors"] == "0;1;2"
    assert diagnostics["psa_n_selected_controls_anchor_union"] == 2
    assert diagnostics["psa_n_selected_controls_treatment"] == 1
    assert diagnostics["psa_n_selected_controls_final_union"] == 3
    assert diagnostics["psa_share_selected_controls_final_union"] == pytest.approx(0.3)
    assert diagnostics["psa_quantile_cv_folds"] == 3
    assert diagnostics["psa_quantile_penalty_grid"] == "0.001;0.01"


def test_estimate_post_selection_ivqr_aligned_returns_diagnostics() -> None:
    data = generate_data(
        Design("dgp1", n=80, p=6, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.linspace(0.0, 2.0, 5)

    result = estimate_post_selection_ivqr_aligned(
        data,
        tau=0.5,
        alphas=alphas,
        selection_cv=2,
        quantile_selection_cv=2,
        quantile_selection_alphas=(0.01, 0.1),
    )

    assert result.estimator == "post_selection_ivqr_aligned"
    assert result.failed is False
    assert result.ps_selection_method == "ivqr_aligned"
    assert result.psa_selection_method == "ivqr_aligned_quantile_l1_cv"
    assert result.psa_anchor_rule == "grid_quartiles"
    assert result.psa_alpha_anchor_count == 3
    assert result.psa_alpha_anchors == "0.5;1;1.5"
    assert result.ps_n_selected_controls == result.psa_n_selected_controls_final_union
    assert result.ps_instrument_selection_method == "all_instruments_retained"
    assert result.ps_n_candidate_instruments == 1
    assert result.ps_n_retained_instruments == 1
    assert result.ps_all_instruments_retained is True
    assert result.cr_disconnected is not None
    assert result.cr_hull_length is not None
    assert result.runtime_total_sec == pytest.approx(result.runtime_seconds)
    assert result.psa_runtime_total_sec == pytest.approx(result.runtime_seconds)
    assert result.psa_runtime_anchor_selection_sec is not None
    assert result.psa_runtime_anchor_selection_sec >= 0.0
    assert result.psa_runtime_treatment_selection_sec is not None
    assert result.psa_runtime_treatment_selection_sec >= 0.0
    assert result.psa_runtime_alpha_loop_sec is not None
    assert result.psa_runtime_alpha_loop_sec >= 0.0
    assert result.psa_runtime_confidence_region_sec is not None
    assert result.psa_runtime_confidence_region_sec >= 0.0
    assert result.psa_runtime_diagnostics_sec is not None
    assert result.psa_runtime_diagnostics_sec >= 0.0


def test_estimate_post_selection_ivqr_aligned_selection_failure_returns_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=4, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    def fail_quantile_selection(*args, **kwargs):
        raise RuntimeError("forced anchor failure")

    monkeypatch.setattr(psa_module, "select_controls_quantile_y", fail_quantile_selection)

    result = estimate_post_selection_ivqr_aligned(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        selection_cv=2,
        quantile_selection_cv=2,
    )

    assert result.failed is True
    assert result.error_type == "quantile_selection_failed"
    assert "forced anchor failure" in result.message
    assert result.ps_selection_method == "ivqr_aligned"
    assert result.ps_selection_failed is True
    assert result.psa_anchor_selection_failed is True
    assert result.psa_n_failed_anchors == 3
    assert result.psa_runtime_total_sec is not None
    assert np.isfinite(result.psa_runtime_total_sec)
