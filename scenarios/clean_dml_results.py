"""Convert a historical wide DML-IVQR CSV to the thesis-ready schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simulation.dml_output import clean_dml_results_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="historical wide DML CSV")
    parser.add_argument("output", type=Path, help="new clean DML CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = clean_dml_results_csv(args.input, args.output)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print(f"Rows:        {summary.output_rows}")
    print(f"Input cols:  {summary.input_columns}")
    print(f"Output cols: {summary.output_columns}")
    print(f"Removed:     {summary.removed_columns}")
    print(f"Missing CRs: {summary.empty_confidence_regions}")
    print(f"Duplicate identifiers: {summary.duplicate_identifiers}")
    print("Missing values:")
    for column, count in summary.missing_values.items():
        print(f"  {column}: {count}")
    print("Status:      PASS")


if __name__ == "__main__":
    main()
