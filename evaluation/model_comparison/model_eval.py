from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.mechanistic_retina.rf_base import Candidate0Reference
from evaluation.model_comparison.prediction import (
    LogitFunction,
    PredictionMetrics,
    predict_trials,
    prediction_metrics,
)
from evaluation.model_comparison.rf import (
    RFComparison,
    conditional_total_dynamic_rf,
    evaluate_comparison_rf,
)
from evaluation.model_comparison.types import RunResult, TrainingPoint


@dataclass(frozen=True, slots=True)
class ModelEvaluationRequest:
    model_name: str
    bank_seed: int
    model_seed: int | None
    parameter_count: int
    logits: LogitFunction
    validation_cones: torch.Tensor
    validation_spikes: torch.Tensor
    validation_mask: torch.Tensor
    validation_probability: torch.Tensor
    bias_logits: torch.Tensor
    candidate: Candidate0Reference
    cone_positions: torch.Tensor
    cell_positions: torch.Tensor
    training: tuple[TrainingPoint, ...]
    gradients_finite: bool
    extras: Mapping[str, JsonValue]
    auxiliary_tensor: torch.Tensor | None = None
    rf_enabled: bool = True


def evaluate_run(request: ModelEvaluationRequest) -> RunResult:
    logits = predict_trials(
        request.logits, request.validation_cones, request.validation_spikes
    )
    prediction = prediction_metrics(
        logits,
        request.validation_spikes,
        request.validation_mask,
        request.validation_probability,
        request.bias_logits,
    )
    rf_tensor, rf = _rf(request)
    return RunResult(
        request.model_name,
        request.bank_seed,
        request.model_seed,
        request.parameter_count,
        prediction,
        rf,
        rf_tensor,
        request.auxiliary_tensor,
        logits.mean(dim=1).detach(),
        request.training,
        request.gradients_finite,
        request.extras,
    )


def _rf(
    request: ModelEvaluationRequest,
) -> tuple[torch.Tensor | None, RFComparison | None]:
    if not request.rf_enabled:
        return None, None
    tensor = conditional_total_dynamic_rf(
        request.logits,
        request.validation_cones,
        request.validation_spikes[:, 0],
        16,
    )
    result = evaluate_comparison_rf(
        tensor,
        request.candidate,
        request.cone_positions,
        request.cell_positions,
    )
    return tensor, result


__all__ = ["ModelEvaluationRequest", "evaluate_run"]
