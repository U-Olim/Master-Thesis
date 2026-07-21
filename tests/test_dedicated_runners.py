from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenarios import (  # noqa: E402
    run_dml_ivqr,
    run_oracle_ivqr,
    run_post_selection_ivqr,
    run_simulation,
)
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
)
from simulation.dml_output import REQUIRED_DML_COLUMNS  # noqa: E402
from simulation.oracle_output import ORACLE_OUTPUT_COLUMNS  # noqa: E402
from simulation.post_selection_output import (  # noqa: E402
    REQUIRED_POST_SELECTION_COLUMNS,
)


RUNNERS = {
    "oracle": run_oracle_ivqr,
    "post_selection": run_post_selection_ivqr,
    "dml": run_dml_ivqr,
}
SCHEMAS = {
    "oracle": ORACLE_OUTPUT_COLUMNS,
    "post_selection": REQUIRED_POST_SELECTION_COLUMNS,
    "dml": REQUIRED_DML_COLUMNS,
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
def test_dedicated_runner_matches_generic_single_estimator(
    tmp_path: Path,
    estimator: str,
    specific_args: list[str],
) -> None:
    generic_output = tmp_path / f"generic_{estimator}.csv"
    generic_manifest = tmp_path / f"generic_{estimator}.json"
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
    run_simulation.main(
        [
            "--mode",
            "fast",
            "--estimators",
            estimator,
            *common,
            "--output",
            str(generic_output),
            "--manifest",
            str(generic_manifest),
        ]
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

    generic = pd.read_csv(generic_output)
    dedicated = pd.read_csv(dedicated_output)
    expected_schema = SCHEMAS[estimator]
    assert tuple(generic.columns) == tuple(dedicated.columns) == expected_schema
    assert len(expected_schema) == {"oracle": 26, "post_selection": 52, "dml": 43}[
        estimator
    ]
    pd.testing.assert_frame_equal(
        dedicated,
        generic,
        check_dtype=False,
        check_exact=True,
    )

    generic_provenance = json.loads(generic_manifest.read_text(encoding="utf-8"))
    dedicated_provenance = json.loads(
        dedicated_manifest.read_text(encoding="utf-8")
    )
    assert dedicated_provenance["estimators"] == [estimator]
    assert dedicated_provenance["resume_signature"] == generic_provenance[
        "resume_signature"
    ]


def test_dedicated_runner_modules_are_import_safe() -> None:
    for runner in RUNNERS.values():
        assert callable(runner.main)
        assert callable(runner._parser)
