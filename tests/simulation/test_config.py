"""Tests for simulation configuration and CLI defaults."""

from pathlib import Path
import warnings

import pytest
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from simulation.runner import (
    DEFAULT_DML_K_FOLDS,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_SIMULATION_ESTIMATORS,
    VALID_ESTIMATORS,
    quantreg_iteration_warning_filter,
    run_small_simulation,
)
from simulation.config import DEFAULT_N_JOBS
from tests.helpers import load_full_control_cli, load_main_simulation_cli

full_simulation_cli = load_main_simulation_cli()
full_simulation_main = full_simulation_cli.main

full_control_cli = load_full_control_cli()
full_control_main = full_control_cli.main


def test_run_small_simulation_default_grid_has_9_points() -> None:
    results = run_small_simulation(
        dgp="dgp1",
        n=80,
        p=5,
        pi=1.0,
        tau=0.5,
        reps=1,
        base_seed=123,
        alphas=None,
    )

    assert results["alpha_grid_size"].dropna().unique().tolist() == [9]


def test_run_small_simulation_explicit_grid_size_has_17_points() -> None:
    results = run_small_simulation(
        dgp="dgp1",
        n=80,
        p=5,
        pi=1.0,
        tau=0.5,
        reps=1,
        base_seed=123,
        alphas=None,
        alpha_grid_size=17,
    )

    assert results["alpha_grid_size"].dropna().unique().tolist() == [17]


def test_valid_estimators_includes_oracle() -> None:
    assert "oracle" in VALID_ESTIMATORS


def test_default_simulation_estimators_are_main_estimators() -> None:
    assert DEFAULT_SIMULATION_ESTIMATORS == ("oracle", "post_selection", "dml")


def test_default_dml_k_folds_is_three() -> None:
    assert DEFAULT_DML_K_FOLDS == 3


def test_default_n_jobs_is_six() -> None:
    assert DEFAULT_N_JOBS == 6


def test_default_quantreg_max_iter_is_1000() -> None:
    assert DEFAULT_QUANTREG_MAX_ITER == 1000


def test_quantreg_iteration_warning_filter_suppresses_by_default() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quantreg_iteration_warning_filter(show_warnings=False):
            warnings.warn("iteration limit", IterationLimitWarning)

    assert caught == []


def test_quantreg_iteration_warning_filter_can_show_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quantreg_iteration_warning_filter(show_warnings=True):
            warnings.warn("iteration limit", IterationLimitWarning)

    assert len(caught) == 1
    assert issubclass(caught[0].category, IterationLimitWarning)


def test_full_simulation_fast_mode_defaults_exclude_full_control() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=None,
        dgps=None,
        n_values=None,
        p_values=None,
        pi_values=None,
        taus=None,
        reps=None,
        alpha_grid_size=None,
        output=None,
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle", "post_selection", "dml"]
    assert "full" not in args.estimators
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.n_values == [500, 1000]
    assert args.p_values == [200, 500]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 10
    assert args.alpha_grid_size == 9
    assert args.output == Path("results/raw/fast_mode_results.csv")
    assert args.summary_output == Path("results/summary/main_simulation_summary.csv")
    assert args.tables_dir == Path("results/tables/main")
    assert args.figures_dir == Path("results/figures/main")


def test_full_simulation_full_mode_defaults_use_500_reps() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="full",
        estimators=None,
        dgps=None,
        n_values=None,
        p_values=None,
        pi_values=None,
        taus=None,
        reps=None,
        alpha_grid_size=None,
        output=None,
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle", "post_selection", "dml"]
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.n_values == [500, 1000]
    assert args.p_values == [200, 500]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 500
    assert args.alpha_grid_size == 9
    assert args.output == Path("results/raw/full_mode_results.csv")
    assert args.summary_output == Path("results/summary/main_simulation_summary.csv")


def test_full_simulation_mode_defaults_respect_explicit_overrides() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=["oracle"],
        dgps=["dgp1"],
        n_values=[500],
        p_values=[200],
        pi_values=None,
        taus=None,
        reps=10,
        alpha_grid_size=3,
        output="custom.csv",
        summary_output="custom_summary.csv",
        tables_dir="custom_tables",
        figures_dir="custom_figures",
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle"]
    assert args.dgps == ["dgp1"]
    assert args.n_values == [500]
    assert args.p_values == [200]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 10
    assert args.alpha_grid_size == 3
    assert args.output == Path("custom.csv")
    assert args.summary_output == "custom_summary.csv"
    assert args.tables_dir == "custom_tables"
    assert args.figures_dir == "custom_figures"


def test_full_simulation_main_mode_respects_alpha_grid_override() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=None,
        dgps=None,
        n_values=None,
        p_values=None,
        pi_values=None,
        taus=None,
        reps=None,
        alpha_grid_size=13,
        output=None,
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle", "post_selection", "dml"]
    assert args.alpha_grid_size == 13


def test_full_simulation_parser_dml_k_folds_default_and_override(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.dml_k_folds == 3

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--mode", "full", "--dml-k-folds", "5"],
    )
    args = full_simulation_cli._parse_args()
    assert args.dml_k_folds == 5


def test_full_simulation_parser_n_jobs_default_and_override(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.n_jobs == 6

    for n_jobs in (1, 4, 6):
        monkeypatch.setattr(
            "sys.argv",
            ["main_simulation.py", "--mode", "full", "--n-jobs", str(n_jobs)],
        )
        args = full_simulation_cli._parse_args()
        assert args.n_jobs == n_jobs


def test_full_simulation_parser_quantreg_max_iter_default_and_override(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.quantreg_max_iter == 1000
    assert args.show_quantreg_warnings is False

    monkeypatch.setattr(
        "sys.argv",
        [
            "main_simulation.py",
            "--mode",
            "full",
            "--quantreg-max-iter",
            "2000",
            "--show-quantreg-warnings",
        ],
    )
    args = full_simulation_cli._parse_args()
    assert args.quantreg_max_iter == 2000
    assert args.show_quantreg_warnings is True


def test_full_simulation_rejects_invalid_n_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--dry-run", "--n-jobs", "0"],
    )
    with pytest.raises(ValueError, match="--n-jobs must be at least 1"):
        full_simulation_main()

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--dry-run", "--n-jobs", "-1"],
    )
    with pytest.raises(ValueError, match="--n-jobs must be at least 1"):
        full_simulation_main()


def test_full_simulation_rejects_invalid_quantreg_max_iter(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main_simulation.py",
            "--dry-run",
            "--quantreg-max-iter",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="--quantreg-max-iter must be at least 1"):
        full_simulation_main()


def test_main_runner_validation_rejects_full_control() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=["full"],
        dgps=["dgp1"],
        n_values=[500],
        p_values=[100],
        pi_values=[1.0],
        taus=[0.5],
        reps=1,
        alpha_grid_size=3,
        output="manual.csv",
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
        batch_size=1,
        n_jobs=1,
        dml_k_folds=3,
        quantreg_max_iter=1000,
        alpha_min=-1.0,
        alpha_max=3.0,
        chunk_index=None,
        num_chunks=None,
        max_designs=None,
    )

    with pytest.raises(ValueError, match="Main runner only allows"):
        full_simulation_cli._validate_args(args)


def test_manual_oracle_uses_single_scenario_defaults() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=["oracle"],
        dgps=None,
        n_values=[100],
        p_values=[20],
        pi_values=None,
        taus=None,
        reps=1,
        alpha_grid_size=3,
        output="oracle.csv",
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle"]
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]


