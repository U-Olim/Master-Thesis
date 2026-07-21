from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenarios import run_simulation  # noqa: E402
from scenarios._dedicated_runner import build_parser, build_run_config  # noqa: E402
from simulation.config import (  # noqa: E402
    DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_HAT_GRID,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_BASE_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_DML_QUANTILE_PENALTY,
    DEFAULT_DML_QUANTILE_SOLVER,
    DEFAULT_DML_RIDGE_ALPHA,
    DEFAULT_GRID_STRATEGY,
    DEFAULT_HARD_FAILURE_POLICY,
    DEFAULT_ITERATION_WARNING_POLICY,
    DEFAULT_MAX_ALPHA_EVALUATIONS,
    DEFAULT_MAX_REFINEMENT_DEPTH,
    DEFAULT_N_JOBS,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_REFINEMENT_TOLERANCE,
    DEFAULT_SELECTION_LASSO_MULTIPLIER,
    DGPS,
    DMLRunConfig,
    FAST_OUTPUT,
    N_VALUES,
    OracleRunConfig,
    PI_VALUES,
    P_VALUES,
    PostSelectionRunConfig,
    R_FAST,
    TAUS,
    build_estimator_run_config,
    runner_kwargs,
)


ESTIMATOR_CONFIG_TYPES = {
    "oracle": OracleRunConfig,
    "post_selection": PostSelectionRunConfig,
    "dml": DMLRunConfig,
}


def _generic_config(estimator: str):
    args = run_simulation._parse_args(
        ["--mode", "fast", "--estimators", estimator]
    )
    run_simulation._apply_defaults(args)
    run_simulation._validate_args(args)
    return args, build_estimator_run_config(args)


@pytest.mark.parametrize("estimator", ESTIMATOR_CONFIG_TYPES)
def test_exact_default_configs_and_dedicated_generic_equivalence(
    estimator: str,
) -> None:
    dedicated_args = build_parser(estimator, prog="test").parse_args([])
    dedicated = build_run_config(estimator, dedicated_args)
    generic_args, generic = _generic_config(estimator)

    assert type(dedicated) is ESTIMATOR_CONFIG_TYPES[estimator]
    assert dedicated == generic
    assert generic.execution.reps == R_FAST
    assert generic.execution.rep_start == 0
    assert generic.execution.rep_end == R_FAST - 1
    assert generic.execution.base_seed == DEFAULT_BASE_SEED
    assert generic.execution.n_jobs == DEFAULT_N_JOBS
    assert generic.execution.batch_size == DEFAULT_BATCH_SIZE
    assert generic.execution.output_path == Path(FAST_OUTPUT)
    assert generic.execution.manifest_path is None
    assert generic.execution.resume is False
    assert generic.design.dgps == DGPS
    assert generic.design.sample_sizes == N_VALUES
    assert generic.design.dimensions == P_VALUES
    assert generic.design.instrument_strengths == PI_VALUES
    assert generic.design.quantiles == TAUS
    assert generic.alpha_grid.alpha_min == DEFAULT_ALPHA_MIN
    assert generic.alpha_grid.alpha_max == DEFAULT_ALPHA_MAX
    assert generic.alpha_grid.alpha_grid_size == DEFAULT_ALPHA_GRID_SIZE

    before = run_simulation._resume_signature(generic_args)
    build_estimator_run_config(generic_args)
    assert run_simulation._resume_signature(generic_args) == before


def test_ch_post_selection_and_dml_defaults_are_exact() -> None:
    _, oracle = _generic_config("oracle")
    _, post = _generic_config("post_selection")
    _, dml = _generic_config("dml")

    for inference in (oracle.inference, post.inference):
        assert inference.iteration_warning_policy == DEFAULT_ITERATION_WARNING_POLICY
        assert inference.hard_failure_policy == DEFAULT_HARD_FAILURE_POLICY
        assert inference.grid_strategy == DEFAULT_GRID_STRATEGY
        assert inference.adaptive_midpoint_probe is DEFAULT_ADAPTIVE_MIDPOINT_PROBE
        assert inference.refinement_tolerance == DEFAULT_REFINEMENT_TOLERANCE
        assert inference.max_refinement_depth == DEFAULT_MAX_REFINEMENT_DEPTH
        assert inference.max_alpha_evaluations == DEFAULT_MAX_ALPHA_EVALUATIONS
        assert inference.alpha_hat_grid == DEFAULT_ALPHA_HAT_GRID
        assert inference.critical_value_multiplier == DEFAULT_CRITICAL_VALUE_MULTIPLIER
        assert inference.quantreg_max_iter == DEFAULT_QUANTREG_MAX_ITER
        assert inference.show_quantreg_warnings is False
    assert post.selection.selection_lasso_multiplier == (
        DEFAULT_SELECTION_LASSO_MULTIPLIER
    )
    assert dml.dml.k_folds == DEFAULT_DML_K_FOLDS
    assert dml.dml.quantile_penalty == DEFAULT_DML_QUANTILE_PENALTY
    assert dml.dml.ridge_alpha == DEFAULT_DML_RIDGE_ALPHA
    assert dml.dml.quantile_solver == DEFAULT_DML_QUANTILE_SOLVER
    assert dml.dml.critical_value_multiplier == DEFAULT_CRITICAL_VALUE_MULTIPLIER


def test_configs_are_frozen_and_exclude_irrelevant_settings() -> None:
    _, oracle = _generic_config("oracle")
    _, post = _generic_config("post_selection")
    _, dml = _generic_config("dml")

    with pytest.raises(FrozenInstanceError):
        oracle.execution.reps = 20  # type: ignore[misc]
    assert not hasattr(oracle, "dml")
    assert not hasattr(oracle, "selection")
    assert not hasattr(post, "dml")
    assert not hasattr(dml, "inference")
    assert not hasattr(dml, "selection")


@pytest.mark.parametrize("estimator", ESTIMATOR_CONFIG_TYPES)
def test_critical_value_multiplier_reaches_owned_runner_path(estimator: str) -> None:
    args = run_simulation._parse_args(
        [
            "--mode",
            "fast",
            "--estimators",
            estimator,
            "--critical-value-multiplier",
            "1.75",
        ]
    )
    run_simulation._apply_defaults(args)
    config = build_estimator_run_config(args)
    assert runner_kwargs(config)["critical_value_multiplier"] == 1.75
