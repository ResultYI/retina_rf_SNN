from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch.nn import functional as F


RECOVERY_COSINE: Final = 0.20
NULL_PERCENTILE: Final = 0.95


class Phase1MetricError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CellRFMetrics:
    full_cosine: float
    signed_spatial_cosine: float
    temporal_cosine: float
    norm_error: float
    positive_energy_recovery: float
    negative_energy_recovery: float
    center_surround_sign_recovery: bool
    temporal_biphasic_recovery: float
    identifiable: bool
    null_percentile: float


@dataclass(frozen=True, slots=True)
class CellPredictionMetrics:
    teacher_expected_cross_entropy: float
    empirical_nll: float
    validation_logit_rmse: float


@dataclass(frozen=True, slots=True)
class PathContributionMetrics:
    cancellation_ratio: float
    direct_teacher_projection_fraction: float
    direct_norm_fraction: float


def rf_recovery_by_cell(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
) -> tuple[CellRFMetrics, ...]:
    _validate_rf_grid(predicted, teacher, cone_positions, cell_positions)
    full = _cosine_by_cell(predicted, teacher)
    spatial = _cosine_by_cell(predicted.sum(dim=-2), teacher.sum(dim=-2))
    temporal = _cosine_by_cell(predicted.sum(dim=-1), teacher.sum(dim=-1))
    pred_norm = predicted.flatten(0, 1).flatten(2).norm(dim=-1).mean(dim=0)
    teacher_norm = teacher.flatten(0, 1).flatten(2).norm(dim=-1).mean(dim=0)
    norm_error = (pred_norm - teacher_norm).abs() / teacher_norm.clamp_min(1e-12)
    positive = _energy_recovery(predicted.clamp_min(0), teacher.clamp_min(0))
    negative = _energy_recovery(-predicted.clamp_max(0), -teacher.clamp_max(0))
    center_surround = _center_surround_recovery(
        predicted,
        teacher,
        cone_positions,
        cell_positions,
    )
    biphasic = _biphasic_recovery(predicted, teacher)
    nulls = _cell_permutation_percentiles(predicted, teacher, full)
    return tuple(
        CellRFMetrics(
            float(full[cell]),
            float(spatial[cell]),
            float(temporal[cell]),
            float(norm_error[cell]),
            float(positive[cell]),
            float(negative[cell]),
            bool(center_surround[cell]),
            float(biphasic[cell]),
            bool(full[cell] >= RECOVERY_COSINE and nulls[cell] >= NULL_PERCENTILE),
            float(nulls[cell]),
        )
        for cell in range(predicted.shape[2])
    )


def prediction_metrics_by_cell(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> tuple[CellPredictionMetrics, ...]:
    if not (
        logits.shape == targets.shape == valid_mask.shape == teacher_probabilities.shape
    ):
        raise Phase1MetricError("prediction tensors must share one shape")
    mask = valid_mask.bool()
    teacher = teacher_probabilities.clamp(1e-6, 1 - 1e-6)
    expected = F.binary_cross_entropy_with_logits(logits, teacher, reduction="none")
    empirical = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    teacher_logits = torch.logit(teacher)
    squared_error = (logits - teacher_logits).square()
    dimensions = (0, 1, 2)
    denominator = mask.sum(dim=dimensions).clamp_min(1)
    expected_cell = (expected * mask).sum(dim=dimensions) / denominator
    empirical_cell = (empirical * mask).sum(dim=dimensions) / denominator
    rmse_cell = torch.sqrt((squared_error * mask).sum(dim=dimensions) / denominator)
    return tuple(
        CellPredictionMetrics(
            float(expected_cell[cell]),
            float(empirical_cell[cell]),
            float(rmse_cell[cell]),
        )
        for cell in range(logits.shape[-1])
    )


def path_contribution_metrics(
    total: torch.Tensor,
    core: torch.Tensor,
    bipolar_direct: torch.Tensor,
    amacrine_direct: torch.Tensor,
    teacher: torch.Tensor,
) -> PathContributionMetrics:
    direct = bipolar_direct + amacrine_direct
    flattened_total = total.flatten()
    flattened_core = core.flatten()
    flattened_direct = direct.flatten()
    flattened_teacher = teacher.flatten()
    cancellation = (
        flattened_core.norm() + bipolar_direct.flatten().norm() + amacrine_direct.flatten().norm()
    ) / flattened_total.norm().clamp_min(1e-12)
    direct_projection = torch.dot(flattened_direct, flattened_teacher)
    total_projection = torch.dot(flattened_total, flattened_teacher)
    projection_sign = torch.where(total_projection < 0, -1.0, 1.0)
    safe_total_projection = projection_sign * total_projection.abs().clamp_min(1e-12)
    return PathContributionMetrics(
        float(cancellation),
        float(direct_projection / safe_total_projection),
        float(flattened_direct.norm() / flattened_total.norm().clamp_min(1e-12)),
    )


def repeat_static_rf(
    rf: torch.Tensor,
    pair_count: int,
) -> torch.Tensor:
    if rf.ndim != 3 or pair_count < 1:
        raise Phase1MetricError("static RF must be [cell,lag,cone]")
    return rf.unsqueeze(0).unsqueeze(0).expand(pair_count, 2, *rf.shape).clone()


def _validate_rf_grid(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
) -> None:
    if predicted.shape != teacher.shape or predicted.ndim != 5:
        raise Phase1MetricError("RF grids must share [pair,context,cell,lag,cone]")
    if cone_positions.shape != (predicted.shape[-1], 2):
        raise Phase1MetricError("cone positions do not match RF grid")
    if cell_positions.shape != (predicted.shape[2], 2):
        raise Phase1MetricError("cell positions do not match RF grid")


def _cosine_by_cell(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    cells = left.shape[2]
    first = left.permute(2, 0, 1, *range(3, left.ndim)).reshape(cells, -1).double()
    second = right.permute(2, 0, 1, *range(3, right.ndim)).reshape(cells, -1).double()
    return (first * second).sum(dim=1) / (
        first.norm(dim=1) * second.norm(dim=1)
    ).clamp_min(1e-12)


def _energy_recovery(predicted: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    dimensions = (0, 1, 3, 4)
    first = predicted.square().sum(dim=dimensions)
    second = teacher.square().sum(dim=dimensions)
    ratio = first / second.clamp_min(1e-12)
    return torch.minimum(ratio, ratio.reciprocal()).clamp(0, 1)


def _center_surround_recovery(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
) -> torch.Tensor:
    predicted_spatial = predicted.mean(dim=(0, 1, 3))
    teacher_spatial = teacher.mean(dim=(0, 1, 3))
    order = torch.cdist(cell_positions.float(), cone_positions.float()).argsort(dim=1)
    quartile = max(1, cone_positions.shape[0] // 4)
    center = order[:, :quartile]
    surround = order[:, -quartile:]
    pred_center = predicted_spatial.gather(1, center).mean(dim=1)
    true_center = teacher_spatial.gather(1, center).mean(dim=1)
    pred_surround = predicted_spatial.gather(1, surround).mean(dim=1)
    true_surround = teacher_spatial.gather(1, surround).mean(dim=1)
    return (torch.sign(pred_center) == torch.sign(true_center)) & (
        torch.sign(pred_surround) == torch.sign(true_surround)
    )


def _biphasic_recovery(predicted: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    predicted_temporal = predicted.mean(dim=(0, 1, 4))
    teacher_temporal = teacher.mean(dim=(0, 1, 4))
    first = _biphasic_index(predicted_temporal)
    second = _biphasic_index(teacher_temporal)
    return (1 - (first - second).abs()).clamp(0, 1)


def _biphasic_index(temporal: torch.Tensor) -> torch.Tensor:
    positive = temporal.clamp_min(0).sum(dim=1)
    negative = -temporal.clamp_max(0).sum(dim=1)
    return 2 * torch.minimum(positive, negative) / temporal.abs().sum(dim=1).clamp_min(1e-12)


def _cell_permutation_percentiles(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    true_cosines: torch.Tensor,
) -> torch.Tensor:
    values = []
    for cell in range(predicted.shape[2]):
        candidate = predicted[:, :, cell].reshape(1, -1).double()
        references = teacher.permute(2, 0, 1, 3, 4).reshape(teacher.shape[2], -1).double()
        cosines = (candidate * references).sum(dim=1) / (
            candidate.norm(dim=1) * references.norm(dim=1)
        ).clamp_min(1e-12)
        wrong = torch.cat((cosines[:cell], cosines[cell + 1 :]))
        values.append((wrong <= true_cosines[cell]).float().mean())
    return torch.stack(values)
