"""Shared pytest configuration for local and CI test runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TEST_CACHE_DIR = PROJECT_ROOT / ".pytest_tmp"
MPL_CACHE_DIR = TEST_CACHE_DIR / "matplotlib"


os.environ.setdefault("MPLBACKEND", "Agg")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))


def _prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    existing = {str(Path(entry).resolve()) for entry in sys.path if entry}
    if resolved not in existing:
        sys.path.insert(0, resolved)


def pytest_sessionstart(session: object) -> None:
    """Ensure the isolated Matplotlib cache exists after pytest initializes."""
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


_prepend_sys_path(SRC_DIR)
