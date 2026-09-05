from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import torch
from torch.nn import functional as F

from evaluation.mechanistic_retina.metrics import JsonValue, RFMetric, evaluate_rf, rf_metric_payload
from evaluation.phase1_metrics import prediction_metrics_by_cell, rf_recovery_by_cell
from evaluation.v4_identity_endpoint import CellIdentityMetadata


@dataclass(frozen=True, slots=True)
class PredictionSummary:
    teacher_expected_ce: float
    sampled_nll: float
    logit_rmse: float
    per_cell_expected_ce: tuple[float, ...]
    per_cell_sampled_nll: tuple[float, ...]
    per_cell_logit_rmse: tuple[float, ...]

    def finite(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (self.teacher_expected_ce, self.sampled_nll, self.logit_rmse)
        )


@dataclass(frozen=True, slots=True)
class DirectRFSummary:
    metric: RFMetric
    mean_norm_error: float
    per_cell_norm_error: tuple[float, ...]

    def finite(self) -> bool:
        return self.metric.finite() and math.isfinite(self.mean_norm_error)


@dataclass(frozen=True, slots=True)
class DirectMetricError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def prediction_summary(
    logits: torch.Tensor,
    sampled: torch.Tensor,
    mask: torch.Tensor,
    expected_probability: torch.Tensor,
) -> PredictionSummary:
    expected = expected_probability[:, None].expand_as(logits)
    cells = prediction_metrics_by_cell(logits, sampled, mask, expected)
    float_mask = mask.to(dtype=logits.dtype)
    denominator = float_mask.sum().clamp_min(1)
    expected_ce = (F.binary_cross_entropy_with_logits(logits, expected, reduction="none") * float_mask).sum() / denominator
    sampled_nll = (F.binary_cross_entropy_with_logits(logits, sampled, reduction="none") * float_mask).sum() / denominator
    teacher_logits = torch.logit(expected.clamp(1e-7, 1 - 1e-7))
    rmse = (((logits - teacher_logits).square() * float_mask).sum() / denominator).sqrt()
    return PredictionSummary(
        float(expected_ce),
        float(sampled_nll),
        float(rmse),
        tuple(value.teacher_expected_cross_entropy for value in cells),
        tuple(value.empirical_nll for value in cells),
        tuple(value.validation_logit_rmse for value in cells),
    )


def rf_summary(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
    metadata: tuple[CellIdentityMetadata, ...],
    *,
    pair_count: int = 6,
) -> DirectRFSummary:
    predicted_grid = _rf_grid(predicted, pair_count)
    teacher_grid = _rf_grid(teacher, pair_count)
    metric = evaluate_rf(
        predicted_grid, teacher_grid, cone_positions, cell_positions, metadata
    )
    cells = rf_recovery_by_cell(
        predicted_grid, teacher_grid, cone_positions, cell_positions
    )
    errors = tuple(value.norm_error for value in cells)
    return DirectRFSummary(metric, sum(errors) / len(errors), errors)


def prediction_payload(summary: PredictionSummary) -> Mapping[str, JsonValue]:
    return {
        "teacher_expected_validation_ce": summary.teacher_expected_ce,
        "independent_sampled_validation_nll": summary.sampled_nll,
        "validation_logit_rmse": summary.logit_rmse,
    }


def rf_payload(summary: DirectRFSummary) -> Mapping[str, JsonValue]:
    payload = dict(rf_metric_payload(summary.metric))
    payload["mean_rf_norm_error"] = summary.mean_norm_error
    return payload


def _rf_grid(value: torch.Tensor, pair_count: int) -> torch.Tensor:
    if value.ndim == 4:
        if value.shape[0] != pair_count * 2:
            raise DirectMetricError(
                "effective RF validation identity is not six paired contexts"
            )
        return value.reshape(pair_count, 2, *value.shape[1:])
    if value.ndim == 3:
        return value.unsqueeze(0).unsqueeze(0).expand(pair_count, 2, *value.shape).clone()
    if value.ndim == 5:
        return value
    raise DirectMetricError("RF tensor has unsupported dimensions")


__all__ = [
    "DirectRFSummary",
    "DirectMetricError",
    "PredictionSummary",
    "prediction_payload",
    "prediction_summary",
    "rf_payload",
    "rf_summary",
]
