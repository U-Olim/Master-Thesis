"""Tests for simulation result rows and tables."""

import inspect

import numpy as np
import pytest

from dgp.designs import Design
from estimators.base import (
    EstimationResult,
    POST_SELECTION_DIAGNOSTIC_FIELDS,
    POST_SELECTION_ALIGNED_DIAGNOSTIC_FIELDS,
    POST_SELECTION_QUANTILE_DIAGNOSTIC_FIELDS,
)
from inference.confidence_regions import (
    ConfidenceRegion,
    merge_region_and_grid_diagnostics,
)
import scenarios.full_control_ivqr as full_control_cli
import simulation.runner as runner_module
from simulation.runner import _result_to_row, run_single_replication, run_small_simulation
from simulation.results import (
    RESULT_COLUMNS,
    build_failure_result_row,
    build_simulation_result_row,
    empty_post_selection_diagnostics,
    empty_post_selection_aligned_diagnostics,
    empty_post_selection_quantile_diagnostics,
    runtime_diagnostics,
)
from tests.helpers import SIMULATION_RESULT_REQUIRED_KEYS
from utils.timing import RUNTIME_COLUMNS


def test_run_single_replication_returns_one_row_per_estimator() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    alphas = np.linspace(0.0, 2.0, 5)

    rows = run_single_replication(design, alphas)

    assert isinstance(rows, list)
    assert len(rows) == 3
    alpha_true = rows[0]["alpha_true"]

    for row in rows:
        assert SIMULATION_RESULT_REQUIRED_KEYS.issubset(row.keys())
        assert row["rep"] == 0
        assert row["seed"] == 123
        assert row["tau"] == 0.5
        assert row["alpha_true"] == alpha_true


def test_run_small_simulation_returns_dataframe() -> None:
    alphas = np.linspace(0.0, 2.0, 5)

    results = run_small_simulation(reps=2, n=80, p=10, alphas=alphas)

    assert len(results) == 6
    assert set(results["estimator"]) == {"oracle", "post_selection_ivqr", "dml_ivqr"}
    assert SIMULATION_RESULT_REQUIRED_KEYS.issubset(results.columns)
    assert {"cr_disconnected", "failed_alpha_count", "alpha_grid_size"}.issubset(
        results.columns
    )
    assert {
        "critical_value_nominal",
        "critical_value_multiplier",
        "critical_value_adjusted",
    }.issubset(results.columns)
    assert np.allclose(results["critical_value_multiplier"], 1.0)
    assert np.allclose(results["critical_value"], results["critical_value_adjusted"])
    assert (results["alpha_grid_min"] == 0.0).all()
    assert (results["alpha_grid_max"] == 2.0).all()
    assert (results["alpha_grid_size"] == 5).all()
    assert np.allclose(results["alpha_grid_step"], 0.5)
    assert {"cr_n_blocks", "cr_hits_any_boundary", "failed_alpha_rate"}.issubset(
        results.columns
    )
    assert set(RUNTIME_COLUMNS).issubset(results.columns)
    assert results["runtime_total_sec"].notna().all()
    assert np.allclose(results["runtime_seconds"], results["runtime_total_sec"])
    ps_columns = {
        "ps_n_selected_controls",
        "ps_n_selected_instruments",
        "ps_n_selected_total",
        "ps_share_selected_controls",
        "ps_share_selected_instruments",
        "ps_instrument_selection_method",
        "ps_n_candidate_instruments",
        "ps_n_retained_instruments",
        "ps_share_retained_instruments",
        "ps_all_instruments_retained",
        "ps_selected_no_controls",
        "ps_selected_no_instruments",
        "ps_selected_empty_total",
        "ps_first_stage_r2",
        "ps_first_stage_adj_r2",
        "ps_first_stage_partial_r2",
        "ps_first_stage_f_stat",
        "ps_first_stage_condition_number",
        "ps_selection_method",
        "ps_lasso_alpha_controls",
        "ps_lasso_alpha_instruments",
        "ps_lasso_alpha_first_stage",
        "ps_lasso_cv_folds",
        "ps_selection_failed",
        "ps_first_stage_failed",
        "ps_rank_deficient",
        "ps_warning_code",
    }
    assert ps_columns.issubset(results.columns)
    post_selection = results.loc[results["estimator"] == "post_selection_ivqr"]
    non_post_selection = results.loc[results["estimator"] != "post_selection_ivqr"]
    assert post_selection["ps_n_selected_controls"].notna().all()
    assert post_selection["ps_n_selected_instruments"].notna().all()
    assert (post_selection["ps_selection_method"] == "lassocv_control_union").all()
    assert (
        post_selection["ps_instrument_selection_method"]
        == "all_instruments_retained"
    ).all()
    assert post_selection["ps_n_candidate_instruments"].notna().all()
    assert post_selection["ps_n_retained_instruments"].notna().all()
    assert post_selection["ps_share_retained_instruments"].notna().all()
    assert (post_selection["ps_all_instruments_retained"] == True).all()  # noqa: E712
    assert non_post_selection["ps_n_selected_controls"].isna().all()
    assert non_post_selection["ps_n_retained_instruments"].isna().all()
    assert (non_post_selection["ps_selection_failed"] == False).all()  # noqa: E712

    oracle = results.loc[results["estimator"] == "oracle"]
    dml = results.loc[results["estimator"] == "dml_ivqr"]
    assert oracle["oracle_runtime_total_sec"].notna().all()
    assert oracle["oracle_runtime_alpha_loop_sec"].notna().all()
    assert oracle["dml_runtime_total_sec"].isna().all()
    assert post_selection["ps_runtime_total_sec"].notna().all()
    assert post_selection["ps_runtime_selection_sec"].notna().all()
    assert post_selection["ps_runtime_diagnostics_sec"].notna().all()
    assert post_selection["ps_runtime_first_stage_sec"].isna().all()
    assert dml["dml_runtime_total_sec"].notna().all()
    assert dml["dml_runtime_crossfit_sec"].notna().all()
    assert dml["dml_runtime_nuisance_fit_sec"].isna().all()
    assert dml["dml_runtime_nuisance_predict_sec"].isna().all()


def test_run_small_simulation_critical_value_multiplier_widens_cr_only() -> None:
    alphas = np.linspace(0.0, 2.0, 5)
    common_kwargs = {
        "reps": 1,
        "n": 120,
        "p": 20,
        "alphas": alphas,
        "estimators": ("oracle",),
        "quantreg_max_iter": 200,
    }

    nominal = run_small_simulation(**common_kwargs, critical_value_multiplier=1.0)
    adjusted = run_small_simulation(**common_kwargs, critical_value_multiplier=1.2)

    nominal_row = nominal.iloc[0]
    adjusted_row = adjusted.iloc[0]
    assert adjusted_row["alpha_hat"] == pytest.approx(nominal_row["alpha_hat"])
    assert adjusted_row["min_test_stat"] == pytest.approx(nominal_row["min_test_stat"])
    assert adjusted_row["critical_value_multiplier"] == pytest.approx(1.2)
    assert adjusted_row["critical_value_nominal"] == pytest.approx(
        nominal_row["critical_value_nominal"]
    )
    assert adjusted_row["critical_value_adjusted"] > nominal_row["critical_value_adjusted"]
    assert adjusted_row["critical_value"] == pytest.approx(
        adjusted_row["critical_value_adjusted"]
    )
    assert (
        adjusted_row["cr_accepted_alpha_count"]
        >= nominal_row["cr_accepted_alpha_count"]
    )
    assert adjusted_row["cr_hull_length"] >= nominal_row["cr_hull_length"]


def test_run_small_simulation_bias_logic() -> None:
    results = run_small_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    for row in results.to_dict("records"):
        if row["alpha_hat"] is not None and not np.isnan(row["alpha_hat"]):
            assert row["bias"] == row["alpha_hat"] - row["alpha_true"]


def test_result_row_uses_authoritative_region_geometry() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    result = EstimationResult(
        estimator="oracle",
        alpha_hat=0.0,
        alpha_true=0.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=4,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=0.4,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=True,
        selected_controls=5,
        runtime_seconds=0.1,
        cr_n_blocks=2,
        cr_hull_length=1.0,
        cr_accepted_alpha_count=4,
        cr_hits_any_boundary=True,
    )

    row = _result_to_row(design, result, np.array([0.0, 0.2, 0.8, 1.0]))

    assert row["cr_length"] == pytest.approx(0.4)
    assert row["cr_hull_length"] == pytest.approx(1.0)
    assert row["cr_n_blocks"] == 2
    assert row["cr_disconnected"] is True
    assert "cr_accepted_alpha_count" in row
    assert "cr_hits_any_boundary" in row


def test_runtime_diagnostics_returns_plain_dict() -> None:
    result = EstimationResult(
        estimator="oracle",
        alpha_hat=0.0,
        alpha_true=0.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=3,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=1.0,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=False,
        selected_controls=5,
        runtime_seconds=0.1,
        runtime_total_sec=0.1,
    )

    diagnostics = runtime_diagnostics(result)

    assert isinstance(diagnostics, dict)
    assert set(RUNTIME_COLUMNS).issubset(diagnostics)
    assert diagnostics["runtime_total_sec"] == pytest.approx(0.1)


def test_empty_post_selection_diagnostics_has_all_expected_defaults() -> None:
    diagnostics = empty_post_selection_diagnostics()

    assert set(diagnostics) == set(POST_SELECTION_DIAGNOSTIC_FIELDS)

    numeric_unavailable = {
        "ps_n_selected_controls",
        "ps_n_selected_instruments",
        "ps_n_selected_total",
        "ps_share_selected_controls",
        "ps_share_selected_instruments",
        "ps_n_candidate_instruments",
        "ps_n_retained_instruments",
        "ps_share_retained_instruments",
        "ps_first_stage_r2",
        "ps_first_stage_adj_r2",
        "ps_first_stage_partial_r2",
        "ps_first_stage_f_stat",
        "ps_first_stage_condition_number",
        "ps_lasso_alpha_controls",
        "ps_lasso_alpha_instruments",
        "ps_lasso_alpha_first_stage",
        "ps_lasso_cv_folds",
    }
    for name in numeric_unavailable:
        assert diagnostics[name] is None

    assert diagnostics["ps_selection_method"] is None
    assert diagnostics["ps_instrument_selection_method"] is None
    assert diagnostics["ps_warning_code"] == ""
    assert diagnostics["ps_selected_no_controls"] is False
    assert diagnostics["ps_selected_no_instruments"] is False
    assert diagnostics["ps_selected_empty_total"] is False
    assert diagnostics["ps_selection_failed"] is False
    assert diagnostics["ps_first_stage_failed"] is False
    assert diagnostics["ps_rank_deficient"] is False
    assert diagnostics["ps_all_instruments_retained"] is False


def test_empty_post_selection_quantile_diagnostics_has_all_expected_defaults() -> None:
    diagnostics = empty_post_selection_quantile_diagnostics()

    assert set(diagnostics) == set(POST_SELECTION_QUANTILE_DIAGNOSTIC_FIELDS)
    assert diagnostics["psq_selection_method"] is None
    assert diagnostics["psq_quantile_tau"] is None
    assert diagnostics["psq_quantile_alpha_selected"] is None
    assert diagnostics["psq_quantile_cv_folds"] is None
    assert diagnostics["psq_n_selected_controls_quantile_y"] is None
    assert diagnostics["psq_n_selected_controls_treatment_d"] is None
    assert diagnostics["psq_n_selected_controls_union"] is None
    assert diagnostics["psq_share_selected_controls_quantile_y"] is None
    assert diagnostics["psq_share_selected_controls_union"] is None
    assert diagnostics["psq_selection_failed"] is False
    assert diagnostics["psq_warning_code"] == ""


def test_empty_post_selection_aligned_diagnostics_has_all_expected_defaults() -> None:
    diagnostics = empty_post_selection_aligned_diagnostics()

    assert set(diagnostics) == set(POST_SELECTION_ALIGNED_DIAGNOSTIC_FIELDS)
    assert diagnostics["psa_selection_method"] is None
    assert diagnostics["psa_anchor_rule"] is None
    assert diagnostics["psa_alpha_anchor_count"] is None
    assert diagnostics["psa_alpha_anchors"] is None
    assert diagnostics["psa_n_selected_controls_anchor_union"] is None
    assert diagnostics["psa_n_selected_controls_treatment"] is None
    assert diagnostics["psa_n_selected_controls_final_union"] is None
    assert diagnostics["psa_anchor_selection_failed"] is False
    assert diagnostics["psa_selected_empty_anchor_union"] is False
    assert diagnostics["psa_selected_empty_final"] is False


def test_merge_region_and_grid_diagnostics_preserves_grid_boundary_fields() -> None:
    grid_diagnostics = {
        "cr_lower": -1.0,
        "cr_upper": 3.0,
        "cr_length": 4.0,
        "cr_hull_length": 4.0,
        "cr_empty": False,
        "cr_n_blocks": 1,
        "cr_disconnected": False,
        "cr_accepted_alpha_count": 3,
        "cr_acceptance_rate": 0.6,
        "cr_hits_lower_boundary": True,
        "cr_hits_upper_boundary": False,
        "cr_hits_any_boundary": True,
        "alpha_grid_min": -1.0,
        "alpha_grid_max": 3.0,
        "alpha_grid_size": 5,
        "alpha_grid_step": 1.0,
        "failed_alpha_count": 1,
        "failed_alpha_rate": 0.2,
        "min_test_stat": 0.1,
        "max_test_stat": 9.0,
        "critical_value": 3.84,
    }
    region = ConfidenceRegion(
        lower=-0.5,
        upper=2.5,
        length=1.25,
        hull_length=3.0,
        blocks=((-0.5, 0.0), (1.75, 2.5)),
        accepted_alphas=(-0.5, 0.0, 1.75, 2.5),
        n_blocks=2,
        empty=False,
        disconnected=True,
        covers_true=True,
        selected_grid=np.array([-0.5, 0.0, 1.75, 2.5]),
        critical_value=3.84,
        critical_value_nominal=3.84,
        critical_value_multiplier=1.0,
        critical_value_adjusted=3.84,
        statistic_reference=0.0,
    )

    diagnostics = merge_region_and_grid_diagnostics(region, grid_diagnostics)

    assert diagnostics["cr_lower"] == pytest.approx(-0.5)
    assert diagnostics["cr_upper"] == pytest.approx(2.5)
    assert diagnostics["cr_length"] == pytest.approx(1.25)
    assert diagnostics["cr_hull_length"] == pytest.approx(3.0)
    assert diagnostics["cr_n_blocks"] == 2
    assert diagnostics["cr_disconnected"] is True
    assert diagnostics["cr_hits_lower_boundary"] is True
    assert diagnostics["cr_hits_upper_boundary"] is False
    assert diagnostics["cr_accepted_alpha_count"] == 3


def test_build_simulation_result_row_defaults_and_extra_fields() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    result = EstimationResult(
        estimator="oracle",
        alpha_hat=0.4,
        alpha_true=0.2,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=3,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=1.0,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=False,
        selected_controls=5,
        runtime_seconds=0.1,
    )

    row = build_simulation_result_row(
        design,
        result,
        np.array([0.0, 0.5, 1.0]),
        extra={"custom_field": "kept"},
    )

    assert row["bias"] == pytest.approx(0.2)
    assert row["absolute_error"] == pytest.approx(0.2)
    assert row["squared_error"] == pytest.approx(0.04)
    assert row["custom_field"] == "kept"
    assert row["ps_n_selected_controls"] is None
    assert row["ps_selection_failed"] is False
    assert row["psq_selection_method"] is None
    assert row["psq_selection_failed"] is False
    assert row["psa_selection_method"] is None
    assert row["psa_anchor_selection_failed"] is False
    assert set(RESULT_COLUMNS).issubset(row)


def test_build_simulation_result_row_includes_quantile_post_selection_fields() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    result = EstimationResult(
        estimator="post_selection_quantile",
        alpha_hat=0.4,
        alpha_true=0.2,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=3,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=1.0,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=False,
        selected_controls=2,
        runtime_seconds=0.1,
        ps_selection_method="quantile_specific",
        ps_n_selected_controls=2,
        psq_selection_method="quantile_l1_cv",
        psq_quantile_tau=0.5,
        psq_quantile_alpha_selected=0.01,
        psq_quantile_cv_folds=3,
        psq_n_selected_controls_quantile_y=1,
        psq_n_selected_controls_treatment_d=1,
        psq_n_selected_controls_union=2,
        psq_share_selected_controls_quantile_y=0.2,
        psq_share_selected_controls_union=0.4,
        psq_selection_failed=False,
    )

    row = build_simulation_result_row(design, result, np.array([0.0, 0.5, 1.0]))

    assert row["estimator"] == "post_selection_quantile"
    assert row["ps_selection_method"] == "quantile_specific"
    assert row["psq_selection_method"] == "quantile_l1_cv"
    assert row["psq_quantile_tau"] == pytest.approx(0.5)
    assert row["psq_quantile_alpha_selected"] == pytest.approx(0.01)
    assert row["psq_n_selected_controls_union"] == 2
    assert set(RESULT_COLUMNS).issubset(row)


def test_build_simulation_result_row_includes_aligned_post_selection_fields() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    result = EstimationResult(
        estimator="post_selection_ivqr_aligned",
        alpha_hat=0.4,
        alpha_true=0.2,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=3,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=1.0,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=False,
        selected_controls=2,
        runtime_seconds=0.1,
        ps_selection_method="ivqr_aligned",
        ps_n_selected_controls=2,
        psa_selection_method="ivqr_aligned_quantile_l1_cv",
        psa_anchor_rule="grid_quartiles",
        psa_alpha_anchor_count=3,
        psa_alpha_anchors="0;1;2",
        psa_n_selected_controls_anchor_union=1,
        psa_n_selected_controls_treatment=1,
        psa_n_selected_controls_final_union=2,
        psa_share_selected_controls_final_union=0.4,
        psa_anchor_selection_failed=False,
        psa_n_failed_anchors=0,
        psa_quantile_cv_folds=3,
        psa_quantile_penalty_grid="0.001;0.01",
        psa_selected_penalties_by_anchor="0:0.01;1:0.01;2:0.01",
    )

    row = build_simulation_result_row(design, result, np.array([0.0, 0.5, 1.0]))

    assert row["estimator"] == "post_selection_ivqr_aligned"
    assert row["ps_selection_method"] == "ivqr_aligned"
    assert row["psa_selection_method"] == "ivqr_aligned_quantile_l1_cv"
    assert row["psa_anchor_rule"] == "grid_quartiles"
    assert row["psa_alpha_anchors"] == "0;1;2"
    assert row["psa_n_selected_controls_final_union"] == 2
    assert set(RESULT_COLUMNS).issubset(row)


def test_build_simulation_result_row_handles_failed_nan_alpha_hat() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    result = EstimationResult(
        estimator="oracle",
        alpha_hat=None,
        alpha_true=0.2,
        tau=0.5,
        converged=False,
        failed=True,
        message="failed",
        objective_value=None,
        at_grid_boundary=False,
        alpha_grid_size=3,
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

    row = build_simulation_result_row(design, result, np.array([0.0, 0.5, 1.0]))

    assert row["status"] == "failed"
    assert row["bias"] is None
    assert row["absolute_error"] is None
    assert row["squared_error"] is None
    assert row["error_type"] == "EstimatorFailure"


def test_build_failure_result_row_uses_common_schema() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)

    row = build_failure_result_row(
        design=design,
        estimator="oracle",
        alphas=np.array([0.0, 0.5, 1.0]),
        alpha_true=0.2,
        exc=RuntimeError("boom"),
        message="RuntimeError: boom",
    )

    assert row["status"] == "failed"
    assert row["estimator"] == "oracle"
    assert row["error_type"] == "RuntimeError"
    assert row["runtime_seconds"] is None
    assert np.isnan(row["runtime_total_sec"])
    assert row["ps_selected_empty_total"] is False
    assert set(RESULT_COLUMNS).issubset(row)


def test_result_builder_helpers_are_not_duplicated_in_scenario_modules() -> None:
    runner_source = inspect.getsource(runner_module)
    full_control_source = inspect.getsource(full_control_cli)

    for source in (runner_source, full_control_source):
        assert "def _result_diagnostics(" not in source
        assert "def _post_selection_diagnostics(" not in source
        assert "def _runtime_diagnostics(" not in source
        assert "def _diagnostic_value(" not in source


def test_main_and_full_control_result_rows_share_common_diagnostic_columns() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    main_result = EstimationResult(
        estimator="oracle",
        alpha_hat=0.0,
        alpha_true=0.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=3,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=1.0,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=False,
        selected_controls=5,
        runtime_seconds=0.1,
    )
    full_control_result = EstimationResult(
        estimator="full_control_ivqr",
        alpha_hat=0.0,
        alpha_true=0.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=3,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=1.0,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=False,
        selected_controls=5,
        runtime_seconds=0.1,
    )

    main_row = runner_module._result_to_row(
        design,
        main_result,
        np.array([0.0, 0.5, 1.0]),
    )
    full_control_row = full_control_cli._result_to_row(
        design,
        full_control_result,
        np.array([0.0, 0.5, 1.0]),
    )

    assert set(RESULT_COLUMNS).issubset(main_row)
    assert set(RESULT_COLUMNS).issubset(full_control_row)
    assert set(main_row) == set(full_control_row)


