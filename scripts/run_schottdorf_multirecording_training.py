from __future__ import annotations

import argparse
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.mechanistic_retina.schottdorf_multirecording_run import (
    SchottdorfMultiRunConfig,
    run_schottdorf_multirecording_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one fresh Canonical V1 model per public macaque MC/PC cell, "
            "pooling that cell's recordings."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output/real_data/"
            "schottdorf_lee_2021_cellwise_multirecording_v1_local_seed20260828_revision2"
        ),
    )
    args = parser.parse_args()
    result = run_schottdorf_multirecording_training(
        SchottdorfMultiRunConfig(
            repository_dir=Path("data/real/schottdorf_lee_2021_repository"),
            movie_path=Path("data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"),
            output_dir=args.output_dir,
        )
    )
    print(result)


if __name__ == "__main__":
    main()
