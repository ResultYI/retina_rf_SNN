from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.mechanistic_retina.clean_sampled_benchmark import (
    CleanBenchmarkConfig,
    run_clean_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the single clean sampled-spike mechanistic benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/bounded_differentiable_delay_learning"),
        help="artifact directory (default: output/bounded_differentiable_delay_learning)",
    )
    args = parser.parse_args()
    result = run_clean_benchmark(
        CleanBenchmarkConfig(), args.output_dir, pathway_diagnostics=True
    )
    print(
        f"validation sampled-spike NLL: "
        f"{result.validation_nll_raw:.6f} -> {result.validation_nll_trained:.6f}"
    )
    print(f"artifacts: {result.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
