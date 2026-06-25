from __future__ import annotations

import config
import simulation.config as sim_config
from simulation.config import (
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_COUNT,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_OUTPUT,
    DEFAULT_QUANTREG_MAX_ITER,
    FAST_FIGURES_DIR,
    FAST_OUTPUT,
    FAST_SUMMARY_OUTPUT,
    FAST_TABLES_DIR,
    FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE,
    FULL_CONTROL_BENCHMARK_DGPS,
    FULL_CONTROL_BENCHMARK_ESTIMATOR,
    FULL_CONTROL_BENCHMARK_N_VALUES,
    FULL_CONTROL_BENCHMARK_OUTPUT,
    FULL_CONTROL_BENCHMARK_PI_VALUES,
    FULL_CONTROL_BENCHMARK_P_VALUES,
    FULL_CONTROL_BENCHMARK_TAUS,
    FULL_FIGURES_DIR,
    FULL_OUTPUT,
    FULL_SUMMARY_OUTPUT,
    FULL_TABLES_DIR,
    DGPS,
    K_FOLDS,
    MAIN_ESTIMATORS,
    N_VALUES,
    PI_VALUES,
    P_VALUES,
    R_FAST,
    R_FULL_CONTROL_BENCHMARK,
    R_MAIN,
    TAUS,
)


def test_default_alpha_grid_size_is_nine() -> None:
    assert DEFAULT_ALPHA_GRID_SIZE == 9
    assert FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE == 9


def test_simulation_replication_defaults() -> None:
    assert R_FAST == 10
    assert R_MAIN == 500
    assert R_FULL_CONTROL_BENCHMARK == 500


def test_mode_output_paths_are_separate_and_consistent() -> None:
    assert FAST_OUTPUT == "results/raw/fast_mode_results.csv"
    assert FULL_OUTPUT == "results/raw/full_mode_results.csv"
    assert DEFAULT_OUTPUT == FULL_OUTPUT
    assert FAST_OUTPUT != FULL_OUTPUT


def test_mode_report_paths_are_separate() -> None:
    assert FAST_SUMMARY_OUTPUT == "results/summary/fast_mode_summary.csv"
    assert FULL_SUMMARY_OUTPUT == "results/summary/full_mode_summary.csv"
    assert FAST_TABLES_DIR == "results/tables/fast"
    assert FULL_TABLES_DIR == "results/tables/full"
    assert FAST_FIGURES_DIR == "results/figures/fast"
    assert FULL_FIGURES_DIR == "results/figures/full"

    assert FAST_SUMMARY_OUTPUT != FULL_SUMMARY_OUTPUT
    assert FAST_TABLES_DIR != FULL_TABLES_DIR
    assert FAST_FIGURES_DIR != FULL_FIGURES_DIR


def test_full_control_output_path_is_defined() -> None:
    assert FULL_CONTROL_BENCHMARK_OUTPUT == (
        "results/raw/full_control_ivqr_results.csv"
    )


def test_output_paths_are_nonempty_csv_paths() -> None:
    paths = {
        "FAST_OUTPUT": FAST_OUTPUT,
        "FULL_OUTPUT": FULL_OUTPUT,
        "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
        "FULL_CONTROL_BENCHMARK_OUTPUT": FULL_CONTROL_BENCHMARK_OUTPUT,
    }

    for name, path in paths.items():
        assert isinstance(path, str), name
        assert path, name
        assert path.endswith(".csv"), name
        assert path.startswith("results/raw/"), name

    assert len(set(paths.values())) == 3


def test_estimation_execution_defaults_are_positive_integers() -> None:
    defaults = {
        "DEFAULT_DML_K_FOLDS": DEFAULT_DML_K_FOLDS,
        "K_FOLDS": K_FOLDS,
        "DEFAULT_QUANTREG_MAX_ITER": DEFAULT_QUANTREG_MAX_ITER,
        "DEFAULT_N_JOBS": DEFAULT_N_JOBS,
        "DEFAULT_BATCH_SIZE": DEFAULT_BATCH_SIZE,
        "DEFAULT_CHUNK_COUNT": DEFAULT_CHUNK_COUNT,
    }

    for name, value in defaults.items():
        assert isinstance(value, int), name
        assert not isinstance(value, bool), name
        assert value > 0, name

    assert K_FOLDS == DEFAULT_DML_K_FOLDS


def test_full_control_benchmark_design_constants() -> None:
    assert FULL_CONTROL_BENCHMARK_DGPS == ("dgp1",)
    assert FULL_CONTROL_BENCHMARK_N_VALUES == (500, 1000)
    assert FULL_CONTROL_BENCHMARK_P_VALUES == (20, 50, 100)
    assert FULL_CONTROL_BENCHMARK_PI_VALUES == (1.0,)
    assert FULL_CONTROL_BENCHMARK_TAUS == (0.25, 0.5, 0.75)


def test_design_constants_are_immutable_tuples() -> None:
    tuple_constants = {
        "N_VALUES": N_VALUES,
        "P_VALUES": P_VALUES,
        "PI_VALUES": PI_VALUES,
        "TAUS": TAUS,
        "DGPS": DGPS,
        "MAIN_ESTIMATORS": MAIN_ESTIMATORS,
        "FULL_CONTROL_BENCHMARK_DGPS": FULL_CONTROL_BENCHMARK_DGPS,
        "FULL_CONTROL_BENCHMARK_N_VALUES": FULL_CONTROL_BENCHMARK_N_VALUES,
        "FULL_CONTROL_BENCHMARK_P_VALUES": FULL_CONTROL_BENCHMARK_P_VALUES,
        "FULL_CONTROL_BENCHMARK_PI_VALUES": FULL_CONTROL_BENCHMARK_PI_VALUES,
        "FULL_CONTROL_BENCHMARK_TAUS": FULL_CONTROL_BENCHMARK_TAUS,
    }

    for name, value in tuple_constants.items():
        assert isinstance(value, tuple), name
        assert value, name


def test_main_simulation_design_values() -> None:
    assert N_VALUES == (500, 1000)
    assert P_VALUES == (200, 500)
    assert PI_VALUES == (1.0, 0.5, 0.25, 0.10)
    assert TAUS == (0.25, 0.50, 0.75)
    assert DGPS == ("dgp1", "dgp2", "dgp3")


def test_estimator_name_constants() -> None:
    assert MAIN_ESTIMATORS == ("oracle", "post_selection", "dml")
    assert FULL_CONTROL_BENCHMARK_ESTIMATOR == "full_control_ivqr"


def test_simulation_config_all_is_complete_unique_and_public() -> None:
    assert isinstance(sim_config.__all__, list)
    assert len(sim_config.__all__) == len(set(sim_config.__all__))
    assert all(isinstance(name, str) and name for name in sim_config.__all__)
    assert all(not name.startswith("_") for name in sim_config.__all__)

    for name in sim_config.__all__:
        assert hasattr(sim_config, name), name


def test_compatibility_config_wrapper_reexports_same_public_surface() -> None:
    assert config.__all__ == sim_config.__all__

    for name in sim_config.__all__:
        assert getattr(config, name) == getattr(sim_config, name), name
