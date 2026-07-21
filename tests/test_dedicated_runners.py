from __future__ import annotations

import json
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenarios import (  # noqa: E402
    _cli_common,
    run_dml_ivqr,
    run_oracle_ivqr,
    run_post_selection_ivqr,
)
from dgp.generators import generate_data  # noqa: E402
from simulation.config import (  # noqa: E402
    DEFAULT_ALPHA_GRID_SIZE,
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
    DEFAULT_N_JOBS,
    runner_kwargs,
)
from simulation.output_schemas import (  # noqa: E402
    DML_OUTPUT_COLUMNS,
    ORACLE_OUTPUT_COLUMNS,
    POST_SELECTION_OUTPUT_COLUMNS,
)
from simulation.runner import make_simulation_grid, run_simulation_batch  # noqa: E402


RUNNERS = {
    "oracle": run_oracle_ivqr,
    "post_selection": run_post_selection_ivqr,
    "dml": run_dml_ivqr,
}
SCHEMAS = {
    "oracle": ORACLE_OUTPUT_COLUMNS,
    "post_selection": POST_SELECTION_OUTPUT_COLUMNS,
    "dml": DML_OUTPUT_COLUMNS,
}


@pytest.mark.parametrize("runner", RUNNERS.values(), ids=RUNNERS.keys())
@pytest.mark.parametrize(
    "forbidden", (("--estimators", "oracle"), ("--mode", "full"))
)
def test_dedicated_runner_rejects_estimator_and_mode_selection(
    runner, forbidden: tuple[str, str]
) -> None:
    with pytest.raises(SystemExit):
        runner._parser().parse_args(list(forbidden))


@pytest.mark.parametrize(
    ("runner", "forbidden_option"),
    [
        (run_oracle_ivqr, "--selection-lasso-multiplier"),
        (run_oracle_ivqr, "--dml-k-folds"),
        (run_post_selection_ivqr, "--dml-quantile-penalty"),
        (run_dml_ivqr, "--selection-lasso-multiplier"),
        (run_dml_ivqr, "--grid-strategy"),
        (run_dml_ivqr, "--max-refinement-depth"),
    ],
)
def test_dedicated_runner_rejects_irrelevant_options(
    runner, forbidden_option: str
) -> None:
    with pytest.raises(SystemExit):
        runner._parser().parse_args([forbidden_option, "1"])


@pytest.mark.parametrize("runner", RUNNERS.values(), ids=RUNNERS.keys())
def test_dedicated_runner_common_defaults_match_generic_defaults(runner) -> None:
    args = runner._parser().parse_args([])
    assert args.reps is None
    assert args.rep_start == 0
    assert args.rep_end is None
    assert args.n_jobs == DEFAULT_N_JOBS
    assert args.batch_size == DEFAULT_BATCH_SIZE
    assert args.base_seed == DEFAULT_BASE_SEED
    assert args.alpha_min == DEFAULT_ALPHA_MIN
    assert args.alpha_max == DEFAULT_ALPHA_MAX
    assert args.alpha_grid_size == DEFAULT_ALPHA_GRID_SIZE
    assert args.critical_value_multiplier == DEFAULT_CRITICAL_VALUE_MULTIPLIER
    assert args.output is None
    assert args.manifest is None


def test_dedicated_runner_specific_defaults_match_generic_defaults() -> None:
    oracle = run_oracle_ivqr._parser().parse_args([])
    post = run_post_selection_ivqr._parser().parse_args([])
    dml = run_dml_ivqr._parser().parse_args([])
    assert oracle.grid_strategy == post.grid_strategy == DEFAULT_GRID_STRATEGY
    assert post.selection_lasso_multiplier == 1.0
    assert dml.dml_k_folds == DEFAULT_DML_K_FOLDS
    assert dml.dml_quantile_penalty == DEFAULT_DML_QUANTILE_PENALTY
    assert dml.dml_ridge_alpha == DEFAULT_DML_RIDGE_ALPHA
    assert dml.dml_quantile_solver == DEFAULT_DML_QUANTILE_SOLVER
    for args in (oracle, post, dml):
        assert not hasattr(args, "estimators")
        assert not hasattr(args, "mode")


@pytest.mark.parametrize(
    ("estimator", "specific_args"),
    [
        ("oracle", ["--grid-strategy", "fixed"]),
        (
            "post_selection",
            ["--grid-strategy", "fixed", "--selection-lasso-multiplier", "1.2"],
        ),
        ("dml", ["--dml-k-folds", "2", "--dml-quantile-penalty", "0.05"]),
    ],
)
def test_dedicated_runner_matches_direct_infrastructure(
    tmp_path: Path,
    estimator: str,
    specific_args: list[str],
) -> None:
    direct_output = tmp_path / f"direct_{estimator}.csv"
    dedicated_output = tmp_path / f"dedicated_{estimator}.csv"
    dedicated_manifest = tmp_path / f"dedicated_{estimator}.json"
    common = [
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--n-values",
        "40",
        "--p-values",
        "5",
        "--pi-values",
        "0.5",
        "--taus",
        "0.5",
        "--n-jobs",
        "1",
        "--batch-size",
        "1",
        "--alpha-grid-size",
        "3",
        *specific_args,
    ]
    parsed = RUNNERS[estimator]._parser().parse_args(
        [*common, "--output", str(direct_output)]
    )
    direct_args = _cli_common.prepare_namespace(estimator, parsed)
    _cli_common.validate_namespace(direct_args)
    config = _cli_common.build_run_config(estimator, direct_args)
    alphas = np.linspace(
        config.alpha_grid.alpha_min,
        config.alpha_grid.alpha_max,
        config.alpha_grid.alpha_grid_size,
    )
    designs = make_simulation_grid(
        dgps=config.design.dgps,
        n_values=config.design.sample_sizes,
        p_values=config.design.dimensions,
        pi_values=config.design.instrument_strengths,
        taus=config.design.quantiles,
        reps=config.execution.reps,
        base_seed=config.execution.base_seed,
        rep_start=config.execution.rep_start,
        rep_end=config.execution.rep_end,
    )
    run_simulation_batch(
        designs,
        alphas,
        estimators=(estimator,),
        output_path=direct_output,
        n_jobs=config.execution.n_jobs,
        **runner_kwargs(config),
    )
    RUNNERS[estimator].main(
        [
            *common,
            "--output",
            str(dedicated_output),
            "--manifest",
            str(dedicated_manifest),
        ]
    )

    direct = pd.read_csv(direct_output)
    dedicated = pd.read_csv(dedicated_output)
    expected_schema = SCHEMAS[estimator]
    assert tuple(direct.columns) == tuple(dedicated.columns) == expected_schema
    assert len(expected_schema) == {"oracle": 26, "post_selection": 52, "dml": 43}[
        estimator
    ]
    pd.testing.assert_frame_equal(
        dedicated,
        direct,
        check_dtype=False,
        check_exact=True,
    )

    dedicated_provenance = json.loads(
        dedicated_manifest.read_text(encoding="utf-8")
    )
    assert dedicated_provenance["estimators"] == [estimator]
    assert dedicated_provenance["resume_signature"] == (
        _cli_common.resume_signature(direct_args)
    )


def test_dedicated_runner_modules_are_import_safe() -> None:
    for runner in RUNNERS.values():
        assert callable(runner.main)
        assert callable(runner._parser)


def test_separate_estimator_runs_use_identical_seed_and_dgp() -> None:
    designs = [
        make_simulation_grid(
            dgps=("dgp1",),
            n_values=(100,),
            p_values=(20,),
            pi_values=(0.5,),
            taus=(0.5,),
            reps=1,
            base_seed=12345,
        )[0]
        for _estimator in RUNNERS
    ]
    assert len({design.seed for design in designs}) == 1
    generated = [generate_data(design) for design in designs]
    for candidate in generated[1:]:
        np.testing.assert_array_equal(candidate.y, generated[0].y)
        np.testing.assert_array_equal(candidate.d, generated[0].d)
        np.testing.assert_array_equal(candidate.z, generated[0].z)
        np.testing.assert_array_equal(candidate.x, generated[0].x)


@pytest.mark.parametrize("estimator", RUNNERS)
def test_dedicated_runner_resume_completes_single_estimator_output(
    tmp_path: Path,
    estimator: str,
) -> None:
    output = tmp_path / f"{estimator}.csv"
    manifest = tmp_path / f"{estimator}.json"
    common = [
        "--reps",
        "2",
        "--dgps",
        "dgp1",
        "--n-values",
        "40",
        "--p-values",
        "5",
        "--pi-values",
        "0.5",
        "--taus",
        "0.5",
        "--n-jobs",
        "1",
        "--batch-size",
        "1",
        "--alpha-grid-size",
        "3",
        "--max-designs",
        "1",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]
    if estimator == "dml":
        common.extend(["--dml-k-folds", "2"])
    RUNNERS[estimator].main(common)
    RUNNERS[estimator].main([*common, "--resume"])
    resumed = pd.read_csv(output)
    assert resumed["rep"].tolist() == [0, 1]
    assert tuple(resumed.columns) == SCHEMAS[estimator]


def test_pixi_simulation_tasks_use_only_dedicated_runners() -> None:
    payload = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    tasks = payload["tool"]["pixi"]["tasks"]
    assert "fast" not in tasks
    assert "full" not in tasks
    simulation_tasks = {
        name: command
        for name, command in tasks.items()
        if name in {
            "oracle",
            "post_selection",
            "dml",
            "final_oracle",
            "final_post_selection",
            "final_dml",
            "final_dry_run",
        }
    }
    assert len(simulation_tasks) == 7
    for command in simulation_tasks.values():
        assert "scenarios/run_simulation.py" not in command
        assert "--mode full" not in command
        assert "--estimators" not in command
