"""Tests for simulation and project configuration constants."""

import config
import simulation.config as sim_config
from simulation.config import (
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_COUNT,
    DEFAULT_N_JOBS,
    DEFAULT_OUTPUT,
    FAST_OUTPUT,
    FULL_CONTROL_BENCHMARK_ESTIMATOR,
    FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE,
    FULL_CONTROL_BENCHMARK_DGPS,
    FULL_CONTROL_BENCHMARK_N_VALUES,
    FULL_CONTROL_BENCHMARK_OUTPUT,
    FULL_CONTROL_BENCHMARK_PI_VALUES,
    FULL_CONTROL_BENCHMARK_P_VALUES,
    FULL_CONTROL_BENCHMARK_TAUS,
    FULL_OUTPUT,
    DGPS,
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


def test_full_control_output_path_is_defined() -> None:
    assert FULL_CONTROL_BENCHMARK_OUTPUT == (
        "results/raw/full_control_ivqr_results.csv"
    )


def test_execution_defaults_are_positive_integers() -> None:
    for value in (DEFAULT_N_JOBS, DEFAULT_BATCH_SIZE, DEFAULT_CHUNK_COUNT):
        assert isinstance(value, int)
        assert value > 0


def test_full_control_benchmark_design_constants() -> None:
    assert FULL_CONTROL_BENCHMARK_DGPS == ("dgp1",)
    assert FULL_CONTROL_BENCHMARK_N_VALUES == (500, 1000)
    assert FULL_CONTROL_BENCHMARK_P_VALUES == (20, 50, 100)
    assert FULL_CONTROL_BENCHMARK_PI_VALUES == (1.0,)
    assert FULL_CONTROL_BENCHMARK_TAUS == (0.25, 0.5, 0.75)


def test_main_simulation_design_constants_are_tuples() -> None:
    assert isinstance(N_VALUES, tuple)
    assert isinstance(P_VALUES, tuple)
    assert isinstance(PI_VALUES, tuple)
    assert isinstance(TAUS, tuple)
    assert isinstance(DGPS, tuple)


def test_estimator_name_constants() -> None:
    assert MAIN_ESTIMATORS == ("oracle", "post_selection", "dml")
    assert FULL_CONTROL_BENCHMARK_ESTIMATOR == "full_control_ivqr"


def test_simulation_config_all_is_complete_and_public() -> None:
    for name in sim_config.__all__:
        assert hasattr(sim_config, name)

    assert all(not name.startswith("_") for name in sim_config.__all__)


def test_compatibility_config_wrapper_reexports_public_surface() -> None:
    assert config.DEFAULT_ALPHA_GRID_SIZE == 9
    assert config.MAIN_ESTIMATORS == ("oracle", "post_selection", "dml")
    assert "DEFAULT_ALPHA_GRID_SIZE" in config.__all__
