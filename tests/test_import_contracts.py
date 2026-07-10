import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
        timeout=120,
    )


def test_direct_ch_inverse_import_works_in_clean_process() -> None:
    result = _run_python(
        "from ivqr.ch_inverse import evaluate_alpha_ch_ivqr; "
        "print(evaluate_alpha_ch_ivqr.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "evaluate_alpha_ch_ivqr"


def test_public_estimator_imports_work_in_clean_process() -> None:
    result = _run_python(
        "from estimators import EstimationResult, estimate_oracle_ivqr, "
        "estimate_post_selection_ivqr, estimate_full_control_ivqr, estimate_dml_ivqr; "
        "print(EstimationResult.__name__, estimate_oracle_ivqr.__name__, "
        "estimate_post_selection_ivqr.__name__, estimate_full_control_ivqr.__name__, "
        "estimate_dml_ivqr.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "EstimationResult estimate_oracle_ivqr estimate_post_selection_ivqr "
        "estimate_full_control_ivqr estimate_dml_ivqr"
    )


def test_ch_inverse_then_public_estimator_import_has_no_cycle() -> None:
    result = _run_python(
        "from ivqr.ch_inverse import evaluate_alpha_ch_ivqr; "
        "from estimators import estimate_full_control_ivqr; "
        "print(evaluate_alpha_ch_ivqr.__name__, estimate_full_control_ivqr.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "evaluate_alpha_ch_ivqr estimate_full_control_ivqr"
