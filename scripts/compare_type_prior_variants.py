# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.type_prior_comparison import (
    TypePriorComparisonError,
    write_type_prior_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare validation type-prior ablation run directories."
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        write_type_prior_comparison(
            [Path(value) for value in args.run],
            Path(args.output),
        )
    except (OSError, KeyError, TypeError, TypePriorComparisonError) as exc:
        print(f"type-prior comparison failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
