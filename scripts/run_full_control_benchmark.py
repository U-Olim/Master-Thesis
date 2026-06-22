"""Stable entry point for the separate Full-Control IVQR benchmark."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).with_name("04_run_full_control_ivqr.py")
    runpy.run_path(str(script), run_name="__main__")
