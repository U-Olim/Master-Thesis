import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pandas as pd
import pytest

from simulation.config import DEFAULT_BASE_SEED
from simulation.runner import (
    SEED_RULE_TEXT,
    make_design_seed,
    make_simulation_grid,
    normalize_estimator_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scenarios" / "run_simulation.py"


def _load_run_simulation_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_simulation", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_simulation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_SIMULATION = _load_run_simulation_module()


def _signature(*extra_args: str) -> dict[str, object]:
    args = RUN_SIMULATION._parse_args(["--mode", "fast", *extra_args])
    RUN_SIMULATION._apply_defaults(args)
    return RUN_SIMULATION._resume_signature(args)


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


def test_help_works(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--help")
    assert result.returncode == 0
    assert "--mode {fast,full}" in result.stdout


def test_default_estimators_and_aliases() -> None:
    assert normalize_estimator_names(None) == ("oracle", "post_selection", "dml")
    assert normalize_estimator_names(["full_control_ivqr"]) == ("full_control",)
    assert normalize_estimator_names(["full-control-ivqr"]) == ("full_control",)
    assert normalize_estimator_names(["dml-ivqr"]) == ("dml",)
    with pytest.raises(ValueError):
        normalize_estimator_names(["bad"])


def test_design_seed_is_stable_and_changes_by_design_cell() -> None:
    base = dict(base_seed=DEFAULT_BASE_SEED, dgp="dgp1", n=500, p=200, pi=1.0, tau=0.5)
    seed = make_design_seed(**base, rep=0)
    assert seed == make_design_seed(**base, rep=0)
    assert seed != make_design_seed(**base, rep=1)
    assert seed != make_design_seed(**{**base, "dgp": "dgp2"}, rep=0)
    assert seed != make_design_seed(**{**base, "n": 1000}, rep=0)
    assert seed != make_design_seed(**{**base, "p": 500}, rep=0)
    assert seed != make_design_seed(**{**base, "pi": 0.5}, rep=0)
    assert seed != make_design_seed(**{**base, "tau": 0.75}, rep=0)


def test_design_seed_is_independent_of_estimator_list() -> None:
    oracle_only = normalize_estimator_names(["oracle"])
    all_estimators = normalize_estimator_names(
        ["oracle", "post_selection", "full_control", "dml"]
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


def test_resume_signature_seed_and_execution_invariance() -> None:
    base = _signature("--n-jobs", "1", "--batch-size", "1")
    changed_execution = _signature("--n-jobs", "4", "--batch-size", "10")
    changed_seed = _signature("--base-seed", "54321")
    changed_estimators = _signature("--estimators", "oracle")
    assert base == changed_execution
    assert base != changed_seed
    assert base != changed_estimators
    assert base["base_seed"] == DEFAULT_BASE_SEED
    assert "n_jobs" not in base
    assert "batch_size" not in base


def test_dry_run_uses_default_estimators(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--mode", "fast", "--dry-run")
    assert result.returncode == 0
    assert "Replications per design: 10" in result.stdout
    assert f"Base seed: {DEFAULT_BASE_SEED}" in result.stdout
    assert "Seed rule: deterministic by design cell, independent of estimator/order" in result.stdout
    assert "First design seed:" in result.stdout
    assert "Estimators: oracle, post_selection, dml" in result.stdout
    assert "Expected design rows:" in result.stdout
    assert "Reports: generated after successful run" in result.stdout


def test_dry_run_accepts_full_control(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--mode", "fast", "--estimators", "full_control", "--dry-run")
    assert result.returncode == 0
    assert "Estimators: full_control" in result.stdout


def test_tiny_one_design_run_writes_csv_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "raw" / "tiny.csv"
    manifest = tmp_path / "raw" / "tiny_manifest.json"
    result = _run_cli(
        tmp_path,
        "--mode",
        "fast",
        "--estimators",
        "full_control",
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
        "--no-reports",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    )
    assert result.returncode == 0, result.stderr
    written = pd.read_csv(output)
    assert written["estimator"].tolist() == ["full_control_ivqr"]
    assert int(written.loc[0, "seed"]) == make_design_seed(
        base_seed=DEFAULT_BASE_SEED,
        dgp="dgp1",
        n=50,
        p=4,
        pi=1.0,
        tau=0.5,
        rep=0,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["estimators"] == ["full_control"]
    assert payload["base_seed"] == DEFAULT_BASE_SEED
    assert payload["seed_rule"] == SEED_RULE_TEXT
    assert payload["resume_signature"]["base_seed"] == DEFAULT_BASE_SEED
    assert "n_jobs" not in payload["resume_signature"]
    assert "batch_size" not in payload["resume_signature"]
