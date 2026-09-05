from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, assert_never

import torch

from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit, masked_history_counts


ResponseHistoryMode: TypeAlias = Literal[
    "observed",
    "zero",
    "shuffled",
    "thresholded_free_running",
    "free_running",
]
ConditionalHistoryMode: TypeAlias = Literal["observed", "zero", "shuffled"]


class ResponseHistoryModeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponseHistoryTrial:
    split: ResponseSplit
    stimulus_index: int
    trial_index: int


@dataclass(frozen=True, slots=True)
class ResponsePredictionRequest:
    model: ResponseRetinaModel
    split: ResponseSplit
    burn_in_steps: int
    device: torch.device
    history_mode: ResponseHistoryMode = "observed"


@dataclass(frozen=True, slots=True)
class ResponsePredictionTensors:
    logits: torch.Tensor
    generator_potential: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor


@torch.no_grad()
def collect_response_predictions(
    request: ResponsePredictionRequest,
) -> ResponsePredictionTensors:
    was_training = request.model.training
    request.model.eval()
    try:
        return _collect_response_predictions(request)
    finally:
        request.model.train(was_training)


def _collect_response_predictions(
    request: ResponsePredictionRequest,
) -> ResponsePredictionTensors:
    split = request.split
    if not 0 <= request.burn_in_steps < split.cone_response.shape[1]:
        raise ResponseHistoryModeError("burn-in must leave prediction bins")
    logits: list[torch.Tensor] = []
    generators: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for stimulus in range(split.cone_response.shape[0]):
        stimulus_logits: list[torch.Tensor] = []
        stimulus_generators: list[torch.Tensor] = []
        stimulus_targets: list[torch.Tensor] = []
        stimulus_masks: list[torch.Tensor] = []
        cones = split.cone_response[stimulus : stimulus + 1].to(request.device)
        for trial in range(split.spike_counts.shape[1]):
            match request.history_mode:
                case "thresholded_free_running" | "free_running":
                    output, _ = request.model.forward_sequence(cones)
                case "observed" | "zero" | "shuffled" as conditional_mode:
                    history = evaluation_history_counts(
                        ResponseHistoryTrial(split, stimulus, trial),
                        conditional_mode,
                    ).to(request.device)
                    output, _ = request.model.forward_sequence(
                        cones,
                        observed_counts=history,
                    )
                case unreachable:
                    assert_never(unreachable)
            burn = request.burn_in_steps
            stimulus_logits.append(output.spike_logits[:, burn:].squeeze(0))
            stimulus_generators.append(
                output.generator_potential[:, burn:].squeeze(0)
            )
            stimulus_targets.append(split.spike_counts[stimulus, trial, burn:])
            stimulus_masks.append(split.valid_mask[stimulus, trial, burn:])
        logits.append(torch.stack(stimulus_logits))
        generators.append(torch.stack(stimulus_generators))
        targets.append(torch.stack(stimulus_targets))
        masks.append(torch.stack(stimulus_masks))
    return ResponsePredictionTensors(
        torch.stack(logits),
        torch.stack(generators),
        torch.stack(targets).to(request.device),
        torch.stack(masks).to(request.device),
    )


def evaluation_history_counts(
    trial: ResponseHistoryTrial,
    history_mode: ConditionalHistoryMode,
) -> torch.Tensor:
    match history_mode:
        case "observed":
            history_trial = trial.trial_index
        case "zero":
            return torch.zeros_like(
                trial.split.spike_counts[
                    trial.stimulus_index : trial.stimulus_index + 1,
                    trial.trial_index,
                ]
            )
        case "shuffled":
            trial_count = trial.split.spike_counts.shape[1]
            if trial_count < 2:
                raise ResponseHistoryModeError(
                    "Shuffled response history requires at least two trials"
                )
            history_trial = (trial.trial_index + 1) % trial_count
        case unreachable:
            assert_never(unreachable)
    counts = trial.split.spike_counts[
        trial.stimulus_index : trial.stimulus_index + 1,
        history_trial,
    ]
    mask = trial.split.valid_mask[
        trial.stimulus_index : trial.stimulus_index + 1,
        history_trial,
    ]
    return masked_history_counts(counts, mask)


__all__ = [
    "ConditionalHistoryMode",
    "ResponseHistoryMode",
    "ResponseHistoryModeError",
    "ResponseHistoryTrial",
    "ResponsePredictionRequest",
    "ResponsePredictionTensors",
    "collect_response_predictions",
    "evaluation_history_counts",
]
