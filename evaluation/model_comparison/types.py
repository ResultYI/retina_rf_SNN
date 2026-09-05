from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from evaluation.mechanistic_retina.metrics import JsonValue
from evaluation.model_comparison.prediction import PredictionMetrics
from evaluation.model_comparison.rf import RFComparison


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    step: int
    train_nll: float
    gradient_infinity_norm: float


@dataclass(frozen=True, slots=True)
class RunResult:
    model: str
    bank_seed: int
    model_seed: int | None
    parameter_count: int
    prediction: PredictionMetrics
    rf: RFComparison | None
    rf_tensor: torch.Tensor | None
    auxiliary_tensor: torch.Tensor | None
    mean_logits: torch.Tensor
    training: tuple[TrainingPoint, ...]
    gradients_finite: bool
    extras: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    model: str
    bank_seed: int
    model_seed: int | None
    step: int
    loss: float


__all__ = ["ProgressEvent", "RunResult", "TrainingPoint"]
