"""Create tables from aggregated or raw simulation results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reporting.tables import load_summary, write_tables  # noqa: E402
from reporting.summaries import aggregate_results_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tables from IVQR simulation results."
    )
    parser.add_argument("--input", default=None, help="Aggregated summary CSV.")
    parser.add_argument(
        "--raw-input", default=None, help="Raw estimator-level result CSV."
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary CSV to write when aggregating from --raw-input.",
    )
    parser.add_argument("--expected-replications", type=int, default=None)
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--round-digits", type=int, default=4)
    args = parser.parse_args()
    if args.input is None and args.raw_input is None:
        parser.error("provide either --input or --raw-input")
    if args.raw_input is not None and args.summary_output is None:
        parser.error("--summary-output is required when using --raw-input")
    return args


def main() -> None:
    args = _parse_args()
    if args.raw_input is not None:
        summary = aggregate_results_file(
            args.raw_input,
            args.summary_output,
            expected_replications=args.expected_replications,
        )
        input_path = Path(args.raw_input)
        summary_path = Path(args.summary_output)
    else:
        summary = load_summary(args.input)
        input_path = Path(args.input)
        summary_path = input_path

    output_dir = Path(args.output_dir)
    written = write_tables(summary, output_dir, round_digits=args.round_digits)

    print(f"Input path: {input_path}")
    print(f"Summary path: {summary_path}")
    print(f"Output dir: {output_dir}")
    print(f"Summary rows: {len(summary)}")
    print("Written tables:")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
