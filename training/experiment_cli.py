from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from training.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ExperimentArguments:
    config: Path
    device: str
    resume: Path | None
    output: Path
    stop_after_steps: int | None
    diagnostics_only: bool
    representation_diagnostic_steps: int | None
    diagnostic_core_lr: float


def parse_experiment_args(
    argv: Sequence[str] | None,
) -> ExperimentArguments:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the canonical retina model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/experiment"),
    )
    parser.add_argument("--stop-after-steps", type=_positive_int)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument(
        "--representation-diagnostic-steps",
        type=_positive_int,
    )
    parser.add_argument(
        "--diagnostic-core-lr",
        type=_positive_float,
        default=2e-4,
    )
    parsed = parser.parse_args(argv)
    read_only_resume = parsed.resume is not None and parsed.diagnostics_only
    if parsed.representation_diagnostic_steps is not None and (
        parsed.stop_after_steps is not None
        or ((parsed.resume is not None or parsed.diagnostics_only) and not read_only_resume)
    ):
        parser.error(
            "representation diagnostic mode cannot be combined with "
            "stop or a mutating resume"
        )
    return ExperimentArguments(
        config=parsed.config,
        device=parsed.device,
        resume=parsed.resume,
        output=parsed.output,
        stop_after_steps=parsed.stop_after_steps,
        diagnostics_only=parsed.diagnostics_only,
        representation_diagnostic_steps=(
            parsed.representation_diagnostic_steps
        ),
        diagnostic_core_lr=parsed.diagnostic_core_lr,
    )


def apply_invocation_overrides(
    config: ExperimentConfig,
    args: ExperimentArguments,
) -> ExperimentConfig:
    steps = args.representation_diagnostic_steps
    if steps is None:
        return config
    return replace(
        config,
        training=replace(
            config.training,
            core_lr=args.diagnostic_core_lr,
            decoder_freeze_steps=steps,
            validation_interval_steps=min(10, steps),
        ),
        objective=replace(
            config.objective,
            phenotype_repulsion_weight=0.0,
        ),
    )


def execution_limit(
    configured_limit: int,
    stop_after_steps: int | None,
    representation_diagnostic_steps: int | None = None,
) -> int:
    requested = (
        representation_diagnostic_steps
        if representation_diagnostic_steps is not None
        else stop_after_steps
    )
    return min(configured_limit, requested) if requested is not None else configured_limit


def diagnostic_should_stop(
    initial_mse: float,
    validation_mses: Sequence[float],
    args: ExperimentArguments,
) -> bool:
    if args.representation_diagnostic_steps is None or len(validation_mses) < 3:
        return False
    threshold = 1.01 * initial_mse
    return all(value > threshold for value in validation_mses[-3:])


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


__all__ = [
    "ExperimentArguments",
    "apply_invocation_overrides",
    "diagnostic_should_stop",
    "execution_limit",
    "parse_experiment_args",
    "seed_everything",
]
