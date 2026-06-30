import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from simulation.runner import normalize_estimator_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scenarios" / "run_simulation.py"


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
    with pytest.raises(ValueError):
        normalize_estimator_names(["bad"])


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
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["estimators"] == ["full_control"]
