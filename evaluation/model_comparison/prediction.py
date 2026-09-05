from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable

import torch
from torch.nn import functional as F

from training.mechanistic_retina.losses import expected_bernoulli_nll


LogitFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    teacher_expected_ce: float
    sampled_nll: float
    bits_per_spike: float
    logit_rmse: float
    brier_score: float
    per_cell_expected_ce: tuple[float, ...]
    per_cell_sampled_nll: tuple[float, ...]
    per_cell_logit_rmse: tuple[float, ...]
    per_cell_brier: tuple[float, ...]


def masked_bernoulli_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return expected_bernoulli_nll(logits, targets, mask)


def fit_bias(train_spikes: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    dimensions = (0, 1, 2)
    rates = (train_spikes * train_mask).sum(dim=dimensions)
    rates /= train_mask.sum(dim=dimensions).clamp_min(1)
    return torch.logit(rates.clamp(1e-5, 1 - 1e-5))


def predict_trials(
    logits: LogitFunction,
    cones: torch.Tensor,
    spikes: torch.Tensor,
    *,
    chunk_size: int = 32,
) -> torch.Tensor:
    stimuli, trials = spikes.shape[:2]
    flat_cones = cones[:, None].expand(-1, trials, -1, -1).flatten(0, 1)
    flat_spikes = spikes.flatten(0, 1)
    rows = []
    with torch.no_grad():
        for start in range(0, flat_spikes.shape[0], chunk_size):
            stop = start + chunk_size
            rows.append(logits(flat_cones[start:stop], flat_spikes[start:stop]))
    return torch.cat(rows).reshape(stimuli, trials, *flat_spikes.shape[1:])


def prediction_metrics(
    logits: torch.Tensor,
    sampled: torch.Tensor,
    mask: torch.Tensor,
    expected_probability: torch.Tensor,
    bias_logits: torch.Tensor,
) -> PredictionMetrics:
    expected = expected_probability[:, None].expand_as(logits)
    baseline = bias_logits.view(1, 1, 1, -1).expand_as(logits)
    float_mask = mask.to(dtype=logits.dtype)
    dimensions = (0, 1, 2)
    denominator = float_mask.sum().clamp_min(1)
    cell_denominator = float_mask.sum(dim=dimensions).clamp_min(1)
    expected_loss = F.binary_cross_entropy_with_logits(logits, expected, reduction="none")
    sampled_loss = F.binary_cross_entropy_with_logits(logits, sampled, reduction="none")
    baseline_loss = F.binary_cross_entropy_with_logits(baseline, sampled, reduction="none")
    teacher_logits = torch.logit(expected.clamp(1e-7, 1 - 1e-7))
    squared_logit = (logits - teacher_logits).square()
    squared_probability = (torch.sigmoid(logits) - expected).square()
    spikes = (sampled * float_mask).sum().clamp_min(1)
    bits = ((baseline_loss - sampled_loss) * float_mask).sum() / (spikes * math.log(2.0))
    return PredictionMetrics(
        float((expected_loss * float_mask).sum() / denominator),
        float((sampled_loss * float_mask).sum() / denominator),
        float(bits),
        float(((squared_logit * float_mask).sum() / denominator).sqrt()),
        float((squared_probability * float_mask).sum() / denominator),
        tuple(float(value) for value in (expected_loss * float_mask).sum(dim=dimensions) / cell_denominator),
        tuple(float(value) for value in (sampled_loss * float_mask).sum(dim=dimensions) / cell_denominator),
        tuple(float(value) for value in ((squared_logit * float_mask).sum(dim=dimensions) / cell_denominator).sqrt()),
        tuple(float(value) for value in (squared_probability * float_mask).sum(dim=dimensions) / cell_denominator),
    )


__all__ = [
    "LogitFunction",
    "PredictionMetrics",
    "fit_bias",
    "masked_bernoulli_loss",
    "predict_trials",
    "prediction_metrics",
]
