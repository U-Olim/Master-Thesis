"""Convert a historical Post-selection IVQR CSV to the thesis-ready schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simulation.post_selection_output import clean_post_selection_results_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="historical wide Post-selection CSV")
    parser.add_argument("output", type=Path, help="new clean Post-selection CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = clean_post_selection_results_csv(args.input, args.output)
    print(f"Input file:      {args.input}")
    print(f"Output file:     {args.output}")
    print(f"Rows:            {summary.output_rows}")
    print(f"Input columns:   {summary.input_columns}")
    print(f"Output columns:  {summary.output_columns}")
    print(f"Removed:         {summary.removed_columns}")
    print(f"Missing CRs:     {summary.empty_confidence_regions}")
    print(f"Duplicate IDs:   {summary.duplicate_identifiers}")
    print("Missing values:")
    for column, count in summary.missing_values.items():
        print(f"  {column}: {count}")
    print(
        "Selected controls (min/max/mean): "
        f"{summary.selected_controls_min:g}/"
        f"{summary.selected_controls_max:g}/"
        f"{summary.selected_controls_mean:g}"
    )
    print(f"Selection multipliers: {list(summary.selection_lasso_multipliers)}")
    print("Status:          PASS")


if __name__ == "__main__":
    main()
