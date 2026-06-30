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
from simulation.config import (
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_N_JOBS,
)
from simulation.estimators_config import normalize_estimator_names
from tests.helpers import load_full_control_cli, load_main_simulation_cli

full_simulation_cli = load_main_simulation_cli()
full_simulation_main = full_simulation_cli.main

full_control_cli = load_full_control_cli()
full_control_main = full_control_cli.main
RESUME_REQUIRES_MANIFEST_MESSAGE = (
    "--resume requires --manifest so run configuration compatibility can be validated."
)


def test_run_small_simulation_default_grid_has_21_points() -> None:
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

    assert results["alpha_grid_size"].dropna().unique().tolist() == [21]
    assert DEFAULT_ALPHA_MIN == -1.0
    assert DEFAULT_ALPHA_MAX == 3.0
    assert (DEFAULT_ALPHA_MAX - DEFAULT_ALPHA_MIN) / (
        DEFAULT_ALPHA_GRID_SIZE - 1
    ) == pytest.approx(0.2)


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
    assert DEFAULT_SIMULATION_ESTIMATORS == ("oracle", "dml", "post_selection")


def test_valid_estimators_includes_quantile_post_selection_experiment() -> None:
    assert "post_selection_quantile" in VALID_ESTIMATORS
    assert "post_selection_quantile" not in DEFAULT_SIMULATION_ESTIMATORS
    assert "post_selection_ivqr_aligned" in VALID_ESTIMATORS
    assert "post_selection_ivqr_aligned" not in DEFAULT_SIMULATION_ESTIMATORS


def test_normalize_estimator_names_defaults_and_aliases() -> None:
    assert normalize_estimator_names(None, scenario="main") == (
        "oracle",
        "dml",
        "post_selection",
    )
    assert normalize_estimator_names(["oracle"], scenario="main") == ("oracle",)
    assert normalize_estimator_names(["dml_ivqr"], scenario="main") == ("dml",)
    assert normalize_estimator_names(["DML-IVQR"], scenario="main") == ("dml",)
    assert normalize_estimator_names(["post-selection"], scenario="main") == (
        "post_selection",
    )
    assert normalize_estimator_names(["post_selection_quantile"], scenario="main") == (
        "post_selection_quantile",
    )
    assert normalize_estimator_names(["post_selection_q"], scenario="main") == (
        "post_selection_quantile",
    )
    assert normalize_estimator_names(["quantile_post_selection"], scenario="main") == (
        "post_selection_quantile",
    )
    assert normalize_estimator_names(["post_selection_ivqr_aligned"], scenario="main") == (
        "post_selection_ivqr_aligned",
    )
    assert normalize_estimator_names(["post_selection_aligned"], scenario="main") == (
        "post_selection_ivqr_aligned",
    )
    assert normalize_estimator_names(
        ["ivqr_aligned_post_selection"],
        scenario="main",
    ) == ("post_selection_ivqr_aligned",)
    assert normalize_estimator_names(
        ["oracle", "oracle", "post_selection_ivqr"],
        scenario="main",
    ) == ("oracle", "post_selection")


def test_normalize_estimator_names_rejects_invalid_and_unsupported() -> None:
    with pytest.raises(ValueError, match="Unknown estimator"):
        normalize_estimator_names(["bad_name"], scenario="main")
    assert normalize_estimator_names(["full_control"], scenario="main") == (
        "full_control",
    )
    with pytest.raises(ValueError, match="not supported"):
        normalize_estimator_names(["dml"], scenario="full_control")
    assert normalize_estimator_names(None, scenario="full_control") == ("full_control",)
    assert normalize_estimator_names(
        ["full_control_ivqr"],
        scenario="full_control",
    ) == ("full_control",)


def test_default_dml_k_folds_is_three() -> None:
    assert DEFAULT_DML_K_FOLDS == 3


def test_default_n_jobs_is_four() -> None:
    assert DEFAULT_N_JOBS == 4


def test_default_batch_size_is_ten() -> None:
    assert DEFAULT_BATCH_SIZE == 10


def test_default_quantreg_max_iter_is_1000() -> None:
    assert DEFAULT_QUANTREG_MAX_ITER == 1000


def test_default_critical_value_multiplier_is_one() -> None:
    assert DEFAULT_CRITICAL_VALUE_MULTIPLIER == pytest.approx(1.0)


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

    assert args.estimators == ("oracle", "dml", "post_selection")
    assert "full" not in args.estimators
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.n_values == [500, 1000]
    assert args.p_values == [200, 500]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 10
    assert args.alpha_grid_size == 21
    assert args.output == Path("results/raw/fast_mode_results.csv")
    assert args.summary_output == Path("results/summary/fast_mode_summary.csv")
    assert args.tables_dir == Path("results/tables/fast")
    assert args.figures_dir == Path("results/figures/fast")


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

    assert args.estimators == ("oracle", "dml", "post_selection")
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.n_values == [500, 1000]
    assert args.p_values == [200, 500]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 500
    assert args.alpha_grid_size == 21
    assert args.output == Path("results/raw/full_mode_results.csv")
    assert args.summary_output == Path("results/summary/full_mode_summary.csv")
    assert args.tables_dir == Path("results/tables/full")
    assert args.figures_dir == Path("results/figures/full")


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

    assert args.estimators == ("oracle",)
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

    assert args.estimators == ("oracle", "dml", "post_selection")
    assert args.alpha_grid_size == 13


def test_full_simulation_parser_respects_alpha_grid_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main_simulation.py",
            "--alpha-min",
            "-0.5",
            "--alpha-max",
            "2.5",
            "--alpha-grid-size",
            "41",
        ],
    )

    args = full_simulation_cli._parse_args()
    full_simulation_cli._apply_mode_defaults(args)

    assert args.alpha_min == -0.5
    assert args.alpha_max == 2.5
    assert args.alpha_grid_size == 41


def test_full_simulation_parser_supports_experiment_a_wide_grid(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main_simulation.py",
            "--mode",
            "fast",
            "--n-jobs",
            "4",
            "--batch-size",
            "10",
            "--alpha-min",
            "-2",
            "--alpha-max",
            "4",
            "--alpha-grid-size",
            "31",
            "--estimators",
            "oracle",
            "post_selection",
            "--output",
            "results/raw/fast_grid31_wide_oracle_post.csv",
            "--manifest",
            "results/raw/fast_grid31_wide_oracle_post_manifest.json",
        ],
    )

    args = full_simulation_cli._parse_args()
    full_simulation_cli._apply_mode_defaults(args)

    assert args.alpha_min == -2.0
    assert args.alpha_max == 4.0
    assert args.alpha_grid_size == 31
    assert (args.alpha_max - args.alpha_min) / (
        args.alpha_grid_size - 1
    ) == pytest.approx(0.2)
    assert args.estimators == ("oracle", "post_selection")


def test_full_simulation_parser_estimator_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    full_simulation_cli._apply_mode_defaults(args)
    assert args.estimators == ("oracle", "dml", "post_selection")

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--estimators", "oracle", "dml_ivqr"],
    )
    args = full_simulation_cli._parse_args()
    full_simulation_cli._apply_mode_defaults(args)
    assert args.estimators == ("oracle", "dml")

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--estimators", "bad_name"],
    )
    args = full_simulation_cli._parse_args()
    with pytest.raises(ValueError, match="Unknown estimator"):
        full_simulation_cli._apply_mode_defaults(args)


def test_full_control_parser_estimator_defaults_and_validation(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["full_control_ivqr.py"])
    args = full_control_cli._parse_args()
    assert args.estimators == ("full_control",)

    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--estimators", "full_control_ivqr"],
    )
    args = full_control_cli._parse_args()
    assert args.estimators == ("full_control",)

    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--estimators", "dml"],
    )
    with pytest.raises(ValueError, match="not supported"):
        full_control_cli._parse_args()


def test_full_control_parser_uses_default_alpha_grid(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["full_control_ivqr.py"])

    args = full_control_cli._parse_args()

    assert args.alpha_min == -1.0
    assert args.alpha_max == 3.0
    assert args.alpha_grid_size == 21
    assert (args.alpha_max - args.alpha_min) / (
        args.alpha_grid_size - 1
    ) == pytest.approx(0.2)


def test_full_control_parser_respects_alpha_grid_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "full_control_ivqr.py",
            "--alpha-min",
            "-0.5",
            "--alpha-max",
            "2.5",
            "--alpha-grid-size",
            "41",
        ],
    )

    args = full_control_cli._parse_args()

    assert args.alpha_min == -0.5
    assert args.alpha_max == 2.5
    assert args.alpha_grid_size == 41


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


def test_full_simulation_parser_chunking_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.chunk_index is None
    assert args.num_chunks is None

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--chunk-index", "1", "--num-chunks", "3"],
    )
    args = full_simulation_cli._parse_args()
    assert args.chunk_index == 1
    assert args.num_chunks == 3


def test_full_simulation_parser_n_jobs_default_and_override(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.n_jobs == 4

    for n_jobs in (1, 2, 4):
        monkeypatch.setattr(
            "sys.argv",
            ["main_simulation.py", "--mode", "full", "--n-jobs", str(n_jobs)],
        )
        args = full_simulation_cli._parse_args()
        assert args.n_jobs == n_jobs


def test_full_simulation_parser_batch_size_default_and_override(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.batch_size == 10

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--batch-size", "5"],
    )
    args = full_simulation_cli._parse_args()
    assert args.batch_size == 5


def test_full_control_parser_runtime_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["full_control_ivqr.py"])
    args = full_control_cli._parse_args()
    assert args.n_jobs == 4
    assert args.batch_size == 10

    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--n-jobs", "2", "--batch-size", "5"],
    )
    args = full_control_cli._parse_args()
    assert args.n_jobs == 2
    assert args.batch_size == 5


def test_full_simulation_parser_allows_no_resume_without_manifest(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()

    assert args.resume is False
    assert args.manifest is None


def test_full_simulation_parser_allows_resume_with_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--resume", "--manifest", "manifest.json"],
    )
    args = full_simulation_cli._parse_args()

    assert args.resume is True
    assert args.manifest == "manifest.json"


def test_full_simulation_parser_rejects_resume_without_manifest(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py", "--resume"])

    with pytest.raises(SystemExit) as exc_info:
        full_simulation_cli._parse_args()

    assert exc_info.value.code == 2
    assert RESUME_REQUIRES_MANIFEST_MESSAGE in capsys.readouterr().err


def test_full_control_parser_allows_no_resume_without_manifest(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["full_control_ivqr.py"])
    args = full_control_cli._parse_args()

    assert args.resume is False
    assert args.manifest is None


def test_full_control_parser_allows_resume_with_manifest(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--resume", "--manifest", "manifest.json"],
    )
    args = full_control_cli._parse_args()

    assert args.resume is True
    assert args.manifest == Path("manifest.json")


def test_full_control_parser_rejects_resume_without_manifest(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("sys.argv", ["full_control_ivqr.py", "--resume"])

    with pytest.raises(SystemExit) as exc_info:
        full_control_cli._parse_args()

    assert exc_info.value.code == 2
    assert RESUME_REQUIRES_MANIFEST_MESSAGE in capsys.readouterr().err


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


def test_full_simulation_parser_critical_value_multiplier_default_and_override(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.critical_value_multiplier == pytest.approx(1.0)

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--critical-value-multiplier", "1.2"],
    )
    args = full_simulation_cli._parse_args()
    assert args.critical_value_multiplier == pytest.approx(1.2)


@pytest.mark.parametrize("multiplier", ["0", "-1", "nan", "inf"])
def test_full_simulation_rejects_invalid_critical_value_multiplier(
    monkeypatch,
    multiplier: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--critical-value-multiplier", multiplier],
    )
    args = full_simulation_cli._parse_args()
    full_simulation_cli._apply_mode_defaults(args)
    with pytest.raises(ValueError, match="critical_value_multiplier"):
        full_simulation_cli._validate_args(args)


def test_full_control_parser_critical_value_multiplier_default_and_override(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["full_control_ivqr.py"])
    args = full_control_cli._parse_args()
    assert args.critical_value_multiplier == pytest.approx(1.0)

    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--critical-value-multiplier", "1.1"],
    )
    args = full_control_cli._parse_args()
    assert args.critical_value_multiplier == pytest.approx(1.1)


def test_full_simulation_rejects_invalid_n_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--dry-run", "--n-jobs", "0"],
    )
    with pytest.raises(ValueError, match="--n-jobs must be at least 1"):
        full_simulation_main()


def test_full_simulation_rejects_invalid_batch_size(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--dry-run", "--batch-size", "0"],
    )
    with pytest.raises(ValueError, match="--batch-size must be at least 1"):
        full_simulation_main()

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--dry-run", "--batch-size", "-1"],
    )
    with pytest.raises(ValueError, match="--batch-size must be at least 1"):
        full_simulation_main()


def test_full_control_rejects_invalid_runtime_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--dry-run", "--n-jobs", "0"],
    )
    with pytest.raises(ValueError, match="--n-jobs must be at least 1"):
        full_control_main()

    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--dry-run", "--batch-size", "0"],
    )
    with pytest.raises(ValueError, match="--batch-size must be at least 1"):
        full_control_main()

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


def test_main_runner_validation_accepts_full_control() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=["full_control_ivqr"],
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

    full_simulation_cli._apply_mode_defaults(args)
    assert args.estimators == ("full_control",)


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

    assert args.estimators == ("oracle",)
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
