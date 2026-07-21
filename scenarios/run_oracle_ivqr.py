"""Run the Oracle IVQR simulation through the production runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scenarios._dedicated_runner import build_parser, run_dedicated  # noqa: E402


def _parser():
    return build_parser("oracle", prog="run_oracle_ivqr.py")


def main(argv: Sequence[str] | None = None) -> None:
    run_dedicated("oracle", prog="run_oracle_ivqr.py", argv=argv)


if __name__ == "__main__":
    main()
