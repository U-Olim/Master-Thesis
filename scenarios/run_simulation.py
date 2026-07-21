"""Retired generic simulation CLI migration stub."""

from __future__ import annotations

from collections.abc import Sequence
import sys


MIGRATION_MESSAGE = """The generic simulation CLI has been retired.

Run one estimator through its dedicated entry point:

  python scenarios/run_oracle_ivqr.py
  python scenarios/run_post_selection_ivqr.py
  python scenarios/run_dml_ivqr.py"""


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(MIGRATION_MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MIGRATION_MESSAGE", "main"]
