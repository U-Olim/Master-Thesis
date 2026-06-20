# Consolidated tests for the thematic project structure.

from pathlib import Path
import subprocess
import sys

import inference
import pandas as pd
from estimators.base import EstimationResult
from dgp.designs import Design
from inference import metrics


def test_core_phase1_imports_work() -> None:
    assert inference is not None
    assert metrics is not None
    assert Design is not None
    assert EstimationResult is not None


def test_full_simulation_script_help_runs() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "02_run_full_simulation.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "Run the full IVQR Monte Carlo simulation" in result.stdout
    assert "--preset" in result.stdout


def test_full_simulation_full_control_benchmark_preset_dry_run() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "02_run_full_simulation.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
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
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0
    assert "preset: full-control-benchmark" in result.stdout
    assert "estimators: full" in result.stdout


def test_full_simulation_manual_full_control_dry_run() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "02_run_full_simulation.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
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
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
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
        timeout=60,
    )

    assert result.returncode == 0
    assert output.exists()
    assert "estimators: oracle" in result.stdout
    written = pd.read_csv(output)
    assert written.loc[0, "estimator"] == "oracle_ivqr"
    assert "status" in written.columns


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
