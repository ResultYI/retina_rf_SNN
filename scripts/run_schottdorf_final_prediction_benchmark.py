from __future__ import annotations

import argparse
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.mechanistic_retina.schottdorf_final_benchmark_run import (
    run_final_prediction_benchmark,
)
from evaluation.mechanistic_retina.schottdorf_final_benchmark_types import (
    FinalBenchmarkConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("data/real/schottdorf_lee_2021_repository"),
    )
    parser.add_argument(
        "--movie",
        type=Path,
        default=Path("data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"),
    )
    parser.add_argument(
        "--retinal-artifact",
        type=Path,
        default=Path(
            "output/real_data/"
            "schottdorf_lee_2021_22cell_canonical_v1_revision4_fresh_20260829"
        ),
    )
    parser.add_argument(
        "--glm-artifact",
        type=Path,
        default=Path(
            "output/real_data/"
            "schottdorf_lee_2021_22cell_matched_prediction_baselines_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/real_data/"
            "schottdorf_lee_2021_22cell_final_fair_prediction_benchmark_revision4"
        ),
    )
    parser.add_argument("--neural-maximum-steps", type=int, default=2_000)
    parser.add_argument("--neural-patience", type=int, default=200)
    args = parser.parse_args()
    result = run_final_prediction_benchmark(
        FinalBenchmarkConfig(
            repository_dir=args.repository,
            movie_path=args.movie,
            retinal_artifact_dir=args.retinal_artifact,
            glm_artifact_dir=args.glm_artifact,
            output_dir=args.output,
            neural_maximum_steps=args.neural_maximum_steps,
            neural_patience=args.neural_patience,
        )
    )
    print(result.artifact_dir.resolve())
    print(result.mean_validation_nll)


if __name__ == "__main__":
    main()
