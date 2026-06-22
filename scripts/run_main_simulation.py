"""Stable entry point for the main IVQR simulation."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).with_name("02_run_full_simulation.py")
    runpy.run_path(str(script), run_name="__main__")
