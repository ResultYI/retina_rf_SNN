from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from loss.rgc_response import response_nll
from models.response_snn import ResponseRetinaModel
from training.response_data import masked_history_counts
from training.response_trainer import ResponseTrainer
from training.response_unroll import ResponseUnrollRequest, unroll_response


class ProbeLikelihoodError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProbeLikelihoodResult:
    probe_steps: int
    trained_nll: float
    initialized_nll: float
    improvement_fraction: float


@torch.no_grad()
def evaluate_validation_probe_likelihood(
    trainer: ResponseTrainer,
    initialized_model: ResponseRetinaModel,
    probe_steps: int,
) -> ProbeLikelihoodResult:
    if not 1 <= probe_steps <= trainer.config.training.differentiable_steps:
        raise ProbeLikelihoodError("probe_steps must fit the differentiable window")
    trained_nll = _probe_nll(trainer, trainer.model, probe_steps)
    initialized_nll = _probe_nll(trainer, initialized_model, probe_steps)
    return ProbeLikelihoodResult(
        probe_steps,
        trained_nll,
        initialized_nll,
        (initialized_nll - trained_nll) / initialized_nll,
    )


def write_probe_likelihood(
    path: str | Path,
    result: ProbeLikelihoodResult,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _probe_nll(
    trainer: ResponseTrainer,
    model: ResponseRetinaModel,
    probe_steps: int,
) -> float:
    split = trainer.data.validation
    model.eval()
    logits = []
    for stimulus in range(split.cone_response.shape[0]):
        trial_logits = []
        cones = split.cone_response[stimulus : stimulus + 1].to(trainer.device)
        for trial in range(split.spike_counts.shape[1]):
            counts = split.spike_counts[stimulus : stimulus + 1, trial].to(
                trainer.device
            )
            mask = split.valid_mask[stimulus : stimulus + 1, trial].to(trainer.device)
            output, _ = unroll_response(
                ResponseUnrollRequest(
                    model,
                    cones,
                    masked_history_counts(counts, mask),
                    trainer.config.training.burn_in_steps,
                    trainer.config.training.differentiable_steps,
                    trainer.config.training.checkpoint_block_steps,
                    False,
                )
            )
            trial_logits.append(output.spike_logits[0, -probe_steps:])
        logits.append(torch.stack(trial_logits))
    targets = split.spike_counts[:, :, -probe_steps:].to(trainer.device)
    valid_mask = split.valid_mask[:, :, -probe_steps:].to(trainer.device)
    return float(
        response_nll(
            torch.stack(logits),
            targets,
            valid_mask,
            trainer.data.target_kind,
        )
    )


__all__ = [
    "ProbeLikelihoodError",
    "ProbeLikelihoodResult",
    "evaluate_validation_probe_likelihood",
    "write_probe_likelihood",
]
