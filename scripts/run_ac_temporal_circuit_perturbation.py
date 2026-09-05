from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.mechanistic_retina.ac_temporal_perturbation import (
    run_ac_temporal_circuit_perturbation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run checkpoint-only AC temporal chirp/flicker perturbation."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("output/bounded_differentiable_delay_learning"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/bounded_differentiable_delay_learning_ac_temporal_perturbation"
        ),
    )
    args = parser.parse_args()
    result = run_ac_temporal_circuit_perturbation(
        args.benchmark_dir,
        args.output_dir,
    )
    print(
        "AC temporal clamp mean absolute change: "
        f"logit={result.mean_absolute_logit_change:.6f}, "
        f"probability={result.mean_absolute_probability_change:.6f}"
    )
    print(f"temporal RF cosine={result.temporal_rf_cosine:.6f}")
    print(f"artifacts: {result.artifact_dir.resolve()}")


if __name__ == "__main__":
    main()
