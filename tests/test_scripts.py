# Consolidated tests for the thematic project structure.

from pathlib import Path
import subprocess
import sys

import inference
import pandas as pd
from estimators.base import EstimationResult
from dgp.designs import Design
from inference import metrics


FULL_SIMULATION_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "02_run_full_simulation.py"
)


def _run_full_simulation_dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FULL_SIMULATION_SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_core_phase1_imports_work() -> None:
    assert inference is not None
    assert metrics is not None
    assert Design is not None
    assert EstimationResult is not None


def test_full_simulation_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(FULL_SIMULATION_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "Run the full IVQR Monte Carlo simulation" in result.stdout
    assert "--preset" in result.stdout


def test_full_simulation_full_control_benchmark_preset_dry_run() -> None:
    result = _run_full_simulation_dry_run(
        "--preset",
        "full-control-benchmark",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "100",
    )

    assert result.returncode == 0
    assert "preset: full-control-benchmark" in result.stdout
    assert "estimators: full" in result.stdout
    assert "alpha grid: size=9" in result.stdout
    assert "DML folds: 3" in result.stdout
    assert "Parallel workers: 6" in result.stdout
    assert "QuantReg max iterations: 1000" in result.stdout
    assert "Show QuantReg warnings: False" in result.stdout


def test_full_simulation_main_preset_dry_run_includes_oracle_not_full() -> None:
    result = _run_full_simulation_dry_run(
        "--preset",
        "main",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "200",
    )

    assert result.returncode == 0
    assert "preset: main" in result.stdout
    assert "estimators: oracle,post_selection,dml" in result.stdout
    assert "estimators: full" not in result.stdout
    assert "alpha grid: size=9" in result.stdout
    assert "DML folds: 3" in result.stdout
    assert "Parallel workers: 6" in result.stdout
    assert "QuantReg max iterations: 1000" in result.stdout


def test_full_simulation_main_preset_alpha_grid_cli_override_dry_run() -> None:
    result = _run_full_simulation_dry_run(
        "--preset",
        "main",
        "--alpha-grid-size",
        "13",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "200",
    )

    assert result.returncode == 0
    assert "preset: main" in result.stdout
    assert "alpha grid: size=13" in result.stdout


def test_full_simulation_main_preset_dml_k_folds_cli_override_dry_run() -> None:
    result = _run_full_simulation_dry_run(
        "--preset",
        "main",
        "--dml-k-folds",
        "5",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "200",
    )

    assert result.returncode == 0
    assert "preset: main" in result.stdout
    assert "DML folds: 5" in result.stdout


def test_full_simulation_main_preset_n_jobs_cli_override_dry_run() -> None:
    result = _run_full_simulation_dry_run(
        "--preset",
        "main",
        "--n-jobs",
        "2",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "200",
    )

    assert result.returncode == 0
    assert "preset: main" in result.stdout
    assert "Parallel workers: 2" in result.stdout


def test_full_simulation_main_preset_quantreg_cli_override_dry_run() -> None:
    result = _run_full_simulation_dry_run(
        "--preset",
        "main",
        "--quantreg-max-iter",
        "2000",
        "--show-quantreg-warnings",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "200",
    )

    assert result.returncode == 0
    assert "preset: main" in result.stdout
    assert "QuantReg max iterations: 2000" in result.stdout
    assert "Show QuantReg warnings: True" in result.stdout


def test_full_simulation_full_control_alpha_grid_cli_override_dry_run() -> None:
    result = _run_full_simulation_dry_run(
        "--preset",
        "full-control-benchmark",
        "--alpha-grid-size",
        "3",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "100",
    )

    assert result.returncode == 0
    assert "preset: full-control-benchmark" in result.stdout
    assert "alpha grid: size=3" in result.stdout


def test_full_simulation_manual_full_control_dry_run() -> None:
    result = _run_full_simulation_dry_run(
        "--estimators",
        "full",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "100",
    )

    assert result.returncode == 0
    assert "preset: main" in result.stdout
    assert "estimators: full" in result.stdout


def test_full_simulation_oracle_estimator_runs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "02_run_full_simulation.py"
    output = tmp_path / "oracle_smoke.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--estimators",
            "oracle",
            "--reps",
            "1",
            "--dgps",
            "dgp1",
            "--n-values",
            "200",
            "--p-values",
            "50",
            "--pi-values",
            "1.0",
            "--taus",
            "0.5",
            "--alpha-grid-size",
            "3",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0
    assert output.exists()
    assert "estimators: oracle" in result.stdout
    written = pd.read_csv(output)
    assert written.loc[0, "estimator"] == "oracle"
    assert "status" in written.columns


def test_full_simulation_parallel_smoke_runs(tmp_path: Path) -> None:
    output = tmp_path / "parallel_smoke.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(FULL_SIMULATION_SCRIPT),
            "--estimators",
            "oracle",
            "post_selection",
            "dml",
            "--reps",
            "1",
            "--dgps",
            "dgp1",
            "--n-values",
            "200",
            "--p-values",
            "50",
            "--pi-values",
            "1.0",
            "--taus",
            "0.5",
            "--alpha-grid-size",
            "3",
            "--dml-k-folds",
            "3",
            "--n-jobs",
            "2",
            "--quantreg-max-iter",
            "1000",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0
    assert output.exists()
    assert "Parallel workers: 2" in result.stdout
    written = pd.read_csv(output)
    assert len(written) == 3
    assert set(written["estimator"]) == {"oracle", "post_selection_ivqr", "dml_ivqr"}
    assert not written.duplicated(["dgp", "n", "p", "pi", "tau", "rep", "seed", "estimator"]).any()


def test_make_tables_script_help_runs() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "03_make_tables.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "Create tables from IVQR simulation results" in result.stdout


def test_pilot_script_runs_end_to_end(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "01_pilot_simulation.py"

    result = subprocess.run(
        [sys.executable, str(script), "--estimators", "dml"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = tmp_path / "results" / "raw" / "pilot_quick_results.csv"
    assert result.returncode == 0
    assert output.exists()
    assert "dml_ivqr" in result.stdout
    assert "dml_k_folds=3" in result.stdout


def test_pilot_script_dml_k_folds_override(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "01_pilot_simulation.py"

    result = subprocess.run(
        [sys.executable, str(script), "--estimators", "dml", "--dml-k-folds", "5"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = tmp_path / "results" / "raw" / "pilot_quick_results.csv"
    assert result.returncode == 0
    assert output.exists()
    assert "dml_k_folds=5" in result.stdout
