import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import simulation.runner as simulation_runner
from analysis.data import _read_results
from dgp import get_oracle_control_indices
from dgp.designs import Design
from ivqr.confidence_regions import parse_cr_components
from simulation.results import RESULT_SCHEMA_VERSION
from simulation.config import DEFAULT_BASE_SEED
from simulation.runner import (
    SEED_RULE_TEXT,
    make_design_seed,
    make_simulation_grid,
    normalize_estimator_names,
    run_simulation_batch,
    validate_oracle_support,
)
from simulation.dml_output import REQUIRED_CORE_COLUMNS, REQUIRED_DML_COLUMNS
from simulation.post_selection_output import REQUIRED_POST_SELECTION_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scenarios" / "run_simulation.py"


@pytest.mark.parametrize("supplied", [np.array([4, 3, 2, 1, 0]), np.arange(5)])
def test_oracle_support_invariant_accepts_matching_order(supplied: np.ndarray) -> None:
    design = Design("dgp1", 80, 20, 1.0, 0.5, 0, 123)
    np.testing.assert_array_equal(validate_oracle_support(design, supplied), np.arange(5))


@pytest.mark.parametrize("supplied", [np.arange(4), np.arange(6)])
def test_oracle_support_invariant_rejects_mismatch(supplied: np.ndarray) -> None:
    design = Design("dgp1", 80, 20, 1.0, 0.5, 7, 123)
    with pytest.raises(ValueError, match=r"Oracle support mismatch.*dgp=dgp1.*rep=7"):
        validate_oracle_support(design, supplied)


def test_oracle_support_invariant_rejects_duplicates() -> None:
    design = Design("dgp1", 80, 20, 1.0, 0.5, 0, 123)
    with pytest.raises(ValueError, match="duplicate"):
        validate_oracle_support(design, np.array([0, 1, 2, 3, 3]))


def test_oracle_runner_receives_the_unchanged_coefficient_implied_support(
    monkeypatch,
) -> None:
    captured: dict[str, np.ndarray] = {}

    def capture_oracle(*args, **kwargs):
        captured["oracle_indices"] = np.asarray(kwargs["oracle_indices"])
        raise RuntimeError("stop after support capture")

    monkeypatch.setattr(simulation_runner, "estimate_oracle_ivqr", capture_oracle)
    design = Design("dgp2", 40, 10, 1.0, 0.5, 0, 123)
    simulation_runner.run_simulation_design(
        design,
        np.array([-1.0, 0.0, 1.0]),
        estimators=("oracle",),
    )
    np.testing.assert_array_equal(
        captured["oracle_indices"],
        get_oracle_control_indices(design.dgp, design.p),
    )


def test_real_ch_csv_round_trip_preserves_components_and_schema(tmp_path: Path) -> None:
    design = Design("dgp1", 80, 20, 1.0, 0.5, 0, 321)
    alphas = np.linspace(-1.0, 3.0, 5)
    for estimator, label in (
        ("oracle", "oracle"),
        ("post_selection", "post_selection"),
    ):
        path = tmp_path / f"{estimator}.csv"
        run_simulation_batch(
            [design],
            alphas,
            estimators=(estimator,),
            output_path=path,
            n_jobs=1,
            adaptive_midpoint_probe=True,
            alpha_hat_grid="initial",
        )
        loaded = _read_results(path, label, expected_replications=1)
        row = loaded.iloc[0]
        components = parse_cr_components(row["cr_components"])
        assert components is not None
        assert int(row["result_schema_version"]) == RESULT_SCHEMA_VERSION
        assert int(row["cr_n_blocks"]) == len(components)
        assert bool(row["cr_disconnected"]) == (len(components) > 1)
        if components:
            assert row["cr_lower"] == pytest.approx(components[0][0])
            assert row["cr_upper"] == pytest.approx(components[-1][1])
            assert row["cr_length"] == pytest.approx(
                sum(upper - lower for lower, upper in components)
            )


def _load_run_simulation_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_simulation", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_simulation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_SIMULATION = _load_run_simulation_module()


def test_git_metadata_unavailable_is_nonfatal(monkeypatch) -> None:
    monkeypatch.setattr(RUN_SIMULATION.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    assert RUN_SIMULATION._git_metadata() == {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }


def _signature(*extra_args: str) -> dict[str, object]:
    args = RUN_SIMULATION._parse_args(["--mode", "fast", *extra_args])
    RUN_SIMULATION._apply_defaults(args)
    return RUN_SIMULATION._resume_signature(args)


def _validated_args(*extra_args: str):
    args = RUN_SIMULATION._parse_args(["--mode", "fast", *extra_args])
    RUN_SIMULATION._apply_defaults(args)
    RUN_SIMULATION._validate_args(args)
    return args


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )


def _cell_int(df: pd.DataFrame, column: str, row: int = 0) -> int:
    return int(str(df[column].iloc[row]))


def _cell_float(df: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(str(df[column].iloc[row]))


def _cell_str(df: pd.DataFrame, column: str, row: int = 0) -> str:
    return str(df[column].iloc[row])


def test_help_works(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--help")
    assert result.returncode == 0
    assert "--mode {fast,full}" in result.stdout


def test_default_estimators_and_aliases() -> None:
    assert normalize_estimator_names(None) == ("oracle", "post_selection", "dml")
    assert normalize_estimator_names(["dml-ivqr"]) == ("dml",)
    with pytest.raises(ValueError):
        normalize_estimator_names(["full_control_ivqr"])
    with pytest.raises(ValueError):
        normalize_estimator_names(["full-control-ivqr"])
    with pytest.raises(ValueError):
        normalize_estimator_names(["bad"])


def test_design_seed_is_stable_and_changes_by_design_cell() -> None:
    base_seed: int = DEFAULT_BASE_SEED
    dgp: str = "dgp1"
    n: int = 500
    p: int = 200
    pi: float = 1.0
    tau: float = 0.5

    seed = make_design_seed(
        base_seed=base_seed,
        dgp=dgp,
        n=n,
        p=p,
        pi=pi,
        tau=tau,
        rep=0,
    )
    assert seed == make_design_seed(
        base_seed=base_seed,
        dgp=dgp,
        n=n,
        p=p,
        pi=pi,
        tau=tau,
        rep=0,
    )
    assert seed != make_design_seed(
        base_seed=base_seed,
        dgp=dgp,
        n=n,
        p=p,
        pi=pi,
        tau=tau,
        rep=1,
    )
    assert seed != make_design_seed(
        base_seed=base_seed,
        dgp="dgp2",
        n=n,
        p=p,
        pi=pi,
        tau=tau,
        rep=0,
    )
    assert seed != make_design_seed(
        base_seed=base_seed,
        dgp=dgp,
        n=1000,
        p=p,
        pi=pi,
        tau=tau,
        rep=0,
    )
    assert seed != make_design_seed(
        base_seed=base_seed,
        dgp=dgp,
        n=n,
        p=500,
        pi=pi,
        tau=tau,
        rep=0,
    )
    assert seed != make_design_seed(
        base_seed=base_seed,
        dgp=dgp,
        n=n,
        p=p,
        pi=0.5,
        tau=tau,
        rep=0,
    )
    assert seed != make_design_seed(
        base_seed=base_seed,
        dgp=dgp,
        n=n,
        p=p,
        pi=pi,
        tau=0.75,
        rep=0,
    )


def test_design_seed_is_independent_of_estimator_list() -> None:
    oracle_only = normalize_estimator_names(["oracle"])
    all_estimators = normalize_estimator_names(
        ["oracle", "post_selection", "dml"]
    )
    assert oracle_only != all_estimators
    first = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(500,),
        p_values=(200,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=1,
        base_seed=DEFAULT_BASE_SEED,
    )[0]
    second = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(500,),
        p_values=(200,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=1,
        base_seed=DEFAULT_BASE_SEED,
    )[0]
    assert first.seed == second.seed


def test_simulation_grid_defaults_to_full_replication_range() -> None:
    designs = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(500,),
        p_values=(200,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=3,
        base_seed=DEFAULT_BASE_SEED,
    )
    assert [design.rep for design in designs] == [0, 1, 2]


@pytest.mark.parametrize(
    ("reps", "rep_start", "rep_end", "expected"),
    [
        (10, 3, 5, [3, 4, 5]),
        (10, 8, 9, [8, 9]),
        (10, 4, 4, [4]),
    ],
)
def test_simulation_grid_uses_global_replication_block(
    reps: int,
    rep_start: int,
    rep_end: int,
    expected: list[int],
) -> None:
    designs = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(500,),
        p_values=(200,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=reps,
        base_seed=DEFAULT_BASE_SEED,
        rep_start=rep_start,
        rep_end=rep_end,
    )
    assert [design.rep for design in designs] == expected


@pytest.mark.parametrize(
    ("rep_start", "rep_end"),
    [
        (-1, 3),
        (5, 4),
        (0, 10),
    ],
)
def test_cli_rejects_invalid_replication_blocks(rep_start: int, rep_end: int) -> None:
    with pytest.raises(ValueError):
        _validated_args(
            "--reps",
            "10",
            "--rep-start",
            str(rep_start),
            "--rep-end",
            str(rep_end),
        )


def test_cli_rejects_multi_estimator_rerun_failed_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="can duplicate successful rows"):
        _validated_args(
            "--resume",
            "--rerun-failed",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--estimators",
            "oracle",
            "dml",
        )


def test_resume_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    base_args = _validated_args("--manifest", str(manifest))
    manifest.write_text(
        json.dumps({"resume_signature": RUN_SIMULATION._resume_signature(base_args)}),
        encoding="utf-8",
    )

    changed_args = _validated_args(
        "--resume",
        "--manifest",
        str(manifest),
        "--base-seed",
        "54321",
    )
    with pytest.raises(ValueError, match="resume signature does not match"):
        RUN_SIMULATION._validate_resume_manifest(changed_args)


def test_resume_signature_seed_and_execution_invariance() -> None:
    base = _signature("--n-jobs", "1", "--batch-size", "1")
    changed_execution = _signature("--n-jobs", "4", "--batch-size", "10")
    changed_seed = _signature("--base-seed", "54321")
    changed_estimators = _signature("--estimators", "oracle")
    changed_selection_lasso = _signature("--selection-lasso-multiplier", "1.2")
    assert base == changed_execution
    assert base != changed_seed
    assert base != changed_estimators
    assert base != changed_selection_lasso
    assert base["base_seed"] == DEFAULT_BASE_SEED
    assert base["selection_lasso_multiplier"] == 1.0
    assert changed_selection_lasso["selection_lasso_multiplier"] == 1.2
    assert "n_jobs" not in base
    assert "batch_size" not in base


def test_resume_signature_changes_by_dml_quantile_penalty() -> None:
    base = _signature("--estimators", "dml", "--dml-quantile-penalty", "0.01")
    changed = _signature("--estimators", "dml", "--dml-quantile-penalty", "0.05")
    assert base != changed
    assert base["dml_quantile_penalty"] == 0.01
    assert changed["dml_quantile_penalty"] == 0.05


def test_resume_signature_changes_by_replication_block() -> None:
    first_block = _signature("--reps", "10", "--rep-start", "0", "--rep-end", "4")
    second_block = _signature("--reps", "10", "--rep-start", "5", "--rep-end", "9")
    assert first_block != second_block
    assert first_block["rep_start"] == 0
    assert first_block["rep_end"] == 4
    assert second_block["rep_start"] == 5
    assert second_block["rep_end"] == 9


def test_seed_uses_global_replication_index_inside_block() -> None:
    full_grid = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(500,),
        p_values=(200,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=10,
        base_seed=DEFAULT_BASE_SEED,
        rep_start=0,
        rep_end=9,
    )
    single_rep_block = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(500,),
        p_values=(200,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=10,
        base_seed=DEFAULT_BASE_SEED,
        rep_start=5,
        rep_end=5,
    )
    full_rep_5 = next(design for design in full_grid if design.rep == 5)
    assert single_rep_block[0].rep == 5
    assert single_rep_block[0].seed == full_rep_5.seed


def test_dry_run_uses_default_estimators(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--mode", "fast", "--dry-run")
    assert result.returncode == 0
    assert "Replications per design: 10" in result.stdout
    assert "Replication block: 0 to 9" in result.stdout
    assert f"Base seed: {DEFAULT_BASE_SEED}" in result.stdout
    assert "Seed rule: deterministic by design cell, independent of estimator/order" in result.stdout
    assert "First design seed:" in result.stdout
    assert "Estimators: oracle, post_selection, dml" in result.stdout
    assert "Post-selection Lasso multiplier: 1.0" in result.stdout
    assert "Expected design rows:" in result.stdout


def test_dry_run_accepts_selection_lasso_multiplier(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "post_selection",
        "--selection-lasso-multiplier",
        "1.2",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "Estimators: post_selection" in result.stdout
    assert "Post-selection Lasso multiplier: 1.2" in result.stdout


def test_selection_lasso_multiplier_rejects_nonpositive_values(tmp_path: Path) -> None:
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--selection-lasso-multiplier",
        "0",
        "--dry-run",
    )
    assert result.returncode != 0
    assert "--selection-lasso-multiplier must be positive" in result.stderr


def test_dry_run_rejects_full_control(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--mode", "fast", "--estimators", "full_control", "--dry-run")
    assert result.returncode != 0
    assert "Unknown estimator name" in result.stderr


@pytest.mark.slow
def test_tiny_one_design_run_writes_csv_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "tiny.csv"
    manifest = tmp_path / "raw" / "tiny_manifest.json"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "oracle",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--n-values",
        "50",
        "--p-values",
        "4",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--max-designs",
        "1",
        "--n-jobs",
        "1",
        "--alpha-grid-size",
        "5",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    )
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert written["estimator"].tolist() == ["oracle"]
    assert _cell_int(written, "seed") == make_design_seed(
        base_seed=DEFAULT_BASE_SEED,
        dgp="dgp1",
        n=50,
        p=4,
        pi=1.0,
        tau=0.5,
        rep=0,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert {"git_commit", "git_branch", "git_dirty"}.issubset(payload)
    assert payload["adaptive_midpoint_probe"] is True
    assert payload["alpha_hat_grid"] == "initial"
    assert payload["confidence_level"] == pytest.approx(0.95)
    assert payload["estimators"] == ["oracle"]
    assert payload["base_seed"] == DEFAULT_BASE_SEED
    assert payload["seed_rule"] == SEED_RULE_TEXT
    assert payload["resume_signature"]["base_seed"] == DEFAULT_BASE_SEED
    assert payload["parameters"]["reps"] == 1
    assert payload["parameters"]["rep_start"] == 0
    assert payload["parameters"]["rep_end"] == 0
    assert payload["resume_signature"]["reps"] == 1
    assert payload["resume_signature"]["rep_start"] == 0
    assert payload["resume_signature"]["rep_end"] == 0
    assert payload["parameters"]["dml_quantile_penalty"] == 0.01
    assert payload["parameters"]["dml_ridge_alpha"] == 1.0
    assert payload["parameters"]["dml_quantile_solver"] == "highs"
    assert payload["resume_signature"]["dml_quantile_penalty"] == 0.01
    assert payload["resume_signature"]["dml_ridge_alpha"] == 1.0
    assert payload["resume_signature"]["dml_quantile_solver"] == "highs"
    assert "n_jobs" not in payload["resume_signature"]
    assert "batch_size" not in payload["resume_signature"]


@pytest.mark.slow
def test_tiny_oracle_run_writes_clean_common_schema(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "tiny_oracle.csv"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "oracle",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--n-values",
        "40",
        "--p-values",
        "3",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--max-designs",
        "1",
        "--n-jobs",
        "1",
        "--alpha-grid-size",
        "3",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert tuple(written.columns) == REQUIRED_CORE_COLUMNS
    assert written["estimator"].tolist() == ["oracle"]
    assert not any("runtime" in column for column in written.columns)


@pytest.mark.slow
def test_tiny_dml_run_writes_clean_schema_and_keeps_options_in_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "tiny_dml.csv"
    manifest = tmp_path / "raw" / "tiny_dml_manifest.json"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "dml",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--n-values",
        "40",
        "--p-values",
        "3",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--max-designs",
        "1",
        "--n-jobs",
        "1",
        "--alpha-grid-size",
        "3",
        "--dml-k-folds",
        "2",
        "--dml-quantile-penalty",
        "0.05",
        "--dml-ridge-alpha",
        "0.5",
        "--dml-quantile-solver",
        "highs-ds",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    )
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert written["estimator"].tolist() == ["dml_ivqr"]
    assert tuple(written.columns) == REQUIRED_DML_COLUMNS

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["parameters"]["dml_quantile_penalty"] == 0.05
    assert payload["parameters"]["dml_ridge_alpha"] == 0.5
    assert payload["parameters"]["dml_quantile_solver"] == "highs-ds"
    assert payload["resume_signature"]["dml_quantile_penalty"] == 0.05
    assert payload["resume_signature"]["dml_ridge_alpha"] == 0.5
    assert payload["resume_signature"]["dml_quantile_solver"] == "highs-ds"


@pytest.mark.slow
def test_tiny_dml_run_excludes_diagnostics_and_runtime(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "tiny_dml_diagnostics.csv"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "dml",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--n-values",
        "40",
        "--p-values",
        "3",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--max-designs",
        "1",
        "--n-jobs",
        "1",
        "--alpha-grid-size",
        "3",
        "--dml-k-folds",
        "2",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert tuple(written.columns) == REQUIRED_DML_COLUMNS
    assert not any("runtime" in column for column in written.columns)
    assert "dml_quantile_solver" not in written.columns


def test_dml_append_rejects_different_output_schema(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "incompatible_dml.csv"
    output.parent.mkdir(parents=True)
    pd.DataFrame({"estimator": ["dml_ivqr"], "failed": [False]}).to_csv(
        output, index=False
    )

    with pytest.raises(ValueError, match="different schema"):
        run_simulation_batch(
            [],
            np.linspace(-1.0, 1.0, 3),
            estimators=("dml",),
            output_path=output,
            append=True,
        )


def test_append_rejects_incompatible_result_schema_version(tmp_path: Path) -> None:
    output = tmp_path / "incompatible_version.csv"
    current = run_simulation_batch(
        [],
        np.linspace(-1.0, 1.0, 3),
        estimators=("oracle",),
    )
    legacy_version = pd.DataFrame(
        [{column: np.nan for column in current.columns}], columns=current.columns
    )
    legacy_version.loc[0, "result_schema_version"] = RESULT_SCHEMA_VERSION - 1
    legacy_version.to_csv(output, index=False)

    with pytest.raises(ValueError, match="incompatible result schema version"):
        run_simulation_batch(
            [],
            np.linspace(-1.0, 1.0, 3),
            estimators=("oracle",),
            output_path=output,
            append=True,
        )


@pytest.mark.slow
def test_block_run_writes_requested_replication_block(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "block.csv"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "oracle",
        "--reps",
        "10",
        "--rep-start",
        "3",
        "--rep-end",
        "5",
        "--dgps",
        "dgp1",
        "--n-values",
        "50",
        "--p-values",
        "4",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-jobs",
        "1",
        "--batch-size",
        "3",
        "--alpha-grid-size",
        "5",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert sorted(written["rep"].unique().tolist()) == [3, 4, 5]


@pytest.mark.slow
def test_default_run_writes_default_replication_block(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "default.csv"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "oracle",
        "--reps",
        "3",
        "--dgps",
        "dgp1",
        "--n-values",
        "50",
        "--p-values",
        "4",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-jobs",
        "1",
        "--batch-size",
        "3",
        "--alpha-grid-size",
        "5",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert sorted(written["rep"].unique().tolist()) == [0, 1, 2]


@pytest.mark.slow
def test_tiny_post_selection_run_writes_clean_schema_and_manifest_options(
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "tiny_post_selection.csv"
    manifest = tmp_path / "raw" / "tiny_post_selection_manifest.json"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "post_selection",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--n-values",
        "80",
        "--p-values",
        "20",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--max-designs",
        "1",
        "--n-jobs",
        "1",
        "--batch-size",
        "10",
        "--alpha-grid-size",
        "5",
        "--selection-lasso-multiplier",
        "1.2",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    )
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert written["estimator"].tolist() == ["post_selection_ivqr"]
    assert tuple(written.columns) == REQUIRED_POST_SELECTION_COLUMNS
    assert _cell_float(written, "selection_lasso_multiplier") == 1.2
    assert "ps_lasso_alpha_y_cv" not in written.columns
    assert "runtime_seconds" not in written.columns
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["parameters"]["selection_lasso_multiplier"] == 1.2
    assert payload["resume_signature"]["selection_lasso_multiplier"] == 1.2
