"""Shared pytest configuration for local and CI test runs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _prepend_sys_path(path: Path) -> None:
    resolved = str(path.resolve())
    existing = {str(Path(entry).resolve()) for entry in sys.path if entry}
    if resolved not in existing:
        sys.path.insert(0, resolved)


_prepend_sys_path(SRC_DIR)
