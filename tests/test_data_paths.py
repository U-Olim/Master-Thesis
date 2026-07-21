from pathlib import Path

import pytest

from analysis.data import resolve_oracle_results_path


def test_oracle_path_prefers_flat_layout(tmp_path: Path) -> None:
    flat = tmp_path / "results/raw/oracle_ivqr.csv"
    legacy = tmp_path / "results/raw/oracle_ivqr/oracle_ivqr.csv"
    flat.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    flat.touch()
    legacy.touch()

    assert resolve_oracle_results_path(tmp_path) == flat


def test_oracle_path_falls_back_to_legacy_layout(tmp_path: Path) -> None:
    legacy = tmp_path / "results/raw/oracle_ivqr/oracle_ivqr.csv"
    legacy.parent.mkdir(parents=True)
    legacy.touch()

    assert resolve_oracle_results_path(tmp_path) == legacy


def test_oracle_path_failure_lists_checked_locations(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        resolve_oracle_results_path(tmp_path)

    message = str(error.value)
    assert str(tmp_path / "results/raw/oracle_ivqr.csv") in message
    assert str(tmp_path / "results/raw/oracle_ivqr/oracle_ivqr.csv") in message
