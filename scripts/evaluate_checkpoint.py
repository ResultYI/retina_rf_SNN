# /// script
# requires-python = ">=3.11"
# dependencies = ["h5py", "numpy", "scipy", "torch"]
# ///

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.checkpoint_contracts import CheckpointEvaluationConfig
from evaluation.checkpoint_runner import run_checkpoint_evaluation


def main(argv: Sequence[str] | None = None) -> int:
    run_checkpoint_evaluation(_parse_args(argv))
    return 0


def _parse_args(argv: Sequence[str] | None) -> CheckpointEvaluationConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate one Retina SNN checkpoint without training."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
    parser.add_argument("--train-h5", nargs="+", type=Path, required=True)
    parser.add_argument("--eval-h5", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-steps", type=int, default=16)
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 2, 4))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rf-sample-count", type=int, default=32)
    parser.add_argument("--glm-max-steps", type=int, default=20)
    parser.add_argument("--humret-root", type=Path)
    parser.add_argument("--humret-model-grating", type=Path)
    parser.add_argument("--formal-evidence", action="store_true")
    args = parser.parse_args(argv)
    return CheckpointEvaluationConfig(
        checkpoint=args.checkpoint,
        normalization_stats=args.normalization_stats,
        train_h5=tuple(args.train_h5),
        eval_h5=tuple(args.eval_h5),
        output_dir=args.output_dir,
        input_steps=args.input_steps,
        horizons=args.horizons,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        rf_sample_count=args.rf_sample_count,
        glm_max_steps=args.glm_max_steps,
        humret_root=args.humret_root,
        humret_model_grating=args.humret_model_grating,
        formal_evidence=args.formal_evidence,
    )


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons must be comma-separated integers") from exc
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    return horizons


if __name__ == "__main__":
    raise SystemExit(main())
