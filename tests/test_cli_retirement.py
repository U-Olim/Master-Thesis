from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenarios.run_simulation import MIGRATION_MESSAGE  # noqa: E402


RETIRED_SCRIPT = PROJECT_ROOT / "scenarios" / "run_simulation.py"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--mode", "fast", "--estimators", "oracle"],
    ],
)
def test_retired_generic_cli_only_prints_migration_error(
    tmp_path: Path, arguments: list[str]
) -> None:
    output = tmp_path / "must_not_exist.csv"
    result = subprocess.run(
        [sys.executable, str(RETIRED_SCRIPT), *arguments, "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == MIGRATION_MESSAGE
    assert not output.exists()


def test_dedicated_cli_sources_do_not_delegate_or_modify_sys_argv() -> None:
    for relative in (
        "scenarios/run_oracle_ivqr.py",
        "scenarios/run_post_selection_ivqr.py",
        "scenarios/run_dml_ivqr.py",
        "scenarios/_dedicated_runner.py",
        "scenarios/_cli_common.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "scenarios.run_simulation" not in source
        assert "run_simulation.main" not in source
        assert "sys.argv =" not in source
