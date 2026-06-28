"""Tests for simulation result rows and tables."""

import numpy as np

from dgp.designs import Design
from simulation.runner import run_single_replication, run_small_simulation
from tests.helpers import SIMULATION_RESULT_REQUIRED_KEYS


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

    results = run_small_simulation(reps=2, n=80, p=5, alphas=alphas)

    assert len(results) == 6
    assert set(results["estimator"]) == {"oracle", "post_selection_ivqr", "dml_ivqr"}
    assert SIMULATION_RESULT_REQUIRED_KEYS.issubset(results.columns)
    assert {"cr_disconnected", "failed_alpha_count", "alpha_grid_size"}.issubset(
        results.columns
    )
    assert (results["alpha_grid_min"] == 0.0).all()
    assert (results["alpha_grid_max"] == 2.0).all()
    assert (results["alpha_grid_size"] == 5).all()
    assert np.allclose(results["alpha_grid_step"], 0.5)
    assert {"cr_n_blocks", "cr_hits_any_boundary", "failed_alpha_rate"}.issubset(
        results.columns
    )
    ps_columns = {
        "ps_n_selected_controls",
        "ps_n_selected_instruments",
        "ps_n_selected_total",
        "ps_share_selected_controls",
        "ps_share_selected_instruments",
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
    assert non_post_selection["ps_n_selected_controls"].isna().all()
    assert (non_post_selection["ps_selection_failed"] == False).all()  # noqa: E712


def test_run_small_simulation_bias_logic() -> None:
    results = run_small_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    for row in results.to_dict("records"):
        if row["alpha_hat"] is not None and not np.isnan(row["alpha_hat"]):
            assert row["bias"] == row["alpha_hat"] - row["alpha_true"]


