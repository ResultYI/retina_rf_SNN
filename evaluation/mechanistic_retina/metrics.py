from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import torch

from evaluation.phase1_metrics import rf_recovery_by_cell
from evaluation.rf_geometry_metrics import RFGeometryMetric, RFGeometryRequest, rf_geometry_metrics
from evaluation.v4_group_endpoint import GroupEndpointRequest, audit_group_endpoint
from evaluation.v4_group_scoring import GroupEndpointKind
from evaluation.v4_identity_endpoint import CellIdentityMetadata


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@unique
class ScientificCase(StrEnum):
    SUPPORTED = "MECHANISTIC-PHYSIOLOGICAL-CORE-SUPPORTED"
    EXPLICIT_SPAN_INSUFFICIENT = "EXPLICIT-SUBUNIT-SPAN-INSUFFICIENT"
    CONSTRAINED_SPAN_INSUFFICIENT = "PHYSIOLOGICAL-CONSTRAINT-SPAN-INSUFFICIENT"
    MATCHED_CONTROL_INVALID = "MATCHED-LIKELIHOOD-CONTROL-INVALID"
    CORE_NOT_COMPETITIVE = "MECHANISTIC-CORE-LIKELIHOOD-NOT-COMPETITIVE"
    RUNTIME_BLOCKED = "RUNTIME-BLOCKED"


@dataclass(frozen=True, slots=True)
class RFCellMetric:
    cell_id: str
    full_cosine: float
    spatial_cosine: float
    temporal_cosine: float
    exact_margin: float
    exact_resolved: bool
    type_polarity_margin: float | None
    type_polarity_resolved: bool | None
    center_error: float
    radius_error: float
    rf_norm: float


@dataclass(frozen=True, slots=True)
class RFMetric:
    global_cosine: float
    spatial_cosine: float
    temporal_cosine: float
    exact_fraction: float
    type_polarity_fraction: float
    median_exact_margin: float
    geometry: RFGeometryMetric
    cells: tuple[RFCellMetric, ...]

    def finite(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (
                self.global_cosine,
                self.spatial_cosine,
                self.temporal_cosine,
                self.exact_fraction,
                self.type_polarity_fraction,
                self.median_exact_margin,
            )
        )


@dataclass(frozen=True, slots=True)
class ProjectionDecision:
    case: ScientificCase
    signed_passed: bool
    constrained_passed: bool


@dataclass(frozen=True, slots=True)
class MechanisticMetricError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message


def evaluate_rf(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    cone_positions: torch.Tensor,
    cell_positions: torch.Tensor,
    metadata: Sequence[CellIdentityMetadata],
) -> RFMetric:
    predicted_grid = _grid(predicted)
    teacher_grid = _grid(teacher)
    recovery = rf_recovery_by_cell(
        predicted_grid, teacher_grid, cone_positions, cell_positions
    )
    correct, wrong, resolved = _identity(predicted_grid, teacher_grid)
    group = audit_group_endpoint(
        GroupEndpointRequest(
            teacher_grid,
            metadata,
            GroupEndpointKind.TYPE_POLARITY,
            predicted_grid,
        )
    )
    geometry = rf_geometry_metrics(
        RFGeometryRequest(predicted_grid, teacher_grid, cone_positions, metadata)
    )
    if group.group_resolved_fraction is None:
        raise MechanisticMetricError("type-polarity endpoint is undefined")
    groups = {cell.cell_id: cell for cell in group.cells}
    cells = tuple(
        RFCellMetric(
            metadata[index].cell_id,
            recovery[index].full_cosine,
            recovery[index].signed_spatial_cosine,
            recovery[index].temporal_cosine,
            float(correct[index] - wrong[index]),
            bool(resolved[index]),
            groups[metadata[index].cell_id].margin,
            groups[metadata[index].cell_id].resolved,
            geometry.cells[index].center_error,
            geometry.cells[index].radius_error,
            float(predicted_grid[:, :, index].norm()),
        )
        for index in range(predicted.shape[-3])
    )
    return RFMetric(
        _cosine(predicted_grid, teacher_grid),
        statistics.fmean(cell.spatial_cosine for cell in cells),
        statistics.fmean(cell.temporal_cosine for cell in cells),
        statistics.fmean(float(cell.exact_resolved) for cell in cells),
        group.group_resolved_fraction,
        float(statistics.median(cell.exact_margin for cell in cells)),
        geometry,
        cells,
    )


def projection_decision(
    signed: RFMetric,
    constrained: RFMetric,
    ratio: float,
    *,
    constrained_solver_valid: bool,
) -> ProjectionDecision:
    signed_passed = (
        signed.global_cosine >= 0.65
        and signed.temporal_cosine >= 0.75
        and signed.exact_fraction >= 12 / 16
        and signed.type_polarity_fraction >= 15 / 16
    )
    constrained_passed = (
        constrained_solver_valid
        and constrained.global_cosine >= 0.55
        and ratio >= 0.80
        and constrained.temporal_cosine >= 0.65
        and constrained.exact_fraction >= 10 / 16
        and constrained.type_polarity_fraction >= 15 / 16
        and constrained.median_exact_margin > 0
    )
    if not signed_passed:
        case = ScientificCase.EXPLICIT_SPAN_INSUFFICIENT
    elif not constrained_passed:
        case = ScientificCase.CONSTRAINED_SPAN_INSUFFICIENT
    else:
        case = ScientificCase.SUPPORTED
    return ProjectionDecision(case, signed_passed, constrained_passed)


def rf_metric_payload(metric: RFMetric) -> Mapping[str, JsonValue]:
    return {
        "global_cosine": metric.global_cosine,
        "spatial_cosine": metric.spatial_cosine,
        "temporal_cosine": metric.temporal_cosine,
        "exact_fraction": metric.exact_fraction,
        "type_polarity_fraction": metric.type_polarity_fraction,
        "median_exact_margin": metric.median_exact_margin,
        "mean_center_error": metric.geometry.mean_center_error,
        "mean_radius_error": metric.geometry.mean_radius_error,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "full_cosine": cell.full_cosine,
                "spatial_cosine": cell.spatial_cosine,
                "temporal_cosine": cell.temporal_cosine,
                "exact_margin": cell.exact_margin,
                "exact_resolved": cell.exact_resolved,
                "type_polarity_margin": cell.type_polarity_margin,
                "type_polarity_resolved": cell.type_polarity_resolved,
                "center_error": cell.center_error,
                "radius_error": cell.radius_error,
                "rf_norm": cell.rf_norm,
            }
            for cell in metric.cells
        ],
    }


def _grid(values: torch.Tensor) -> torch.Tensor:
    return values if values.ndim == 5 else values.unsqueeze(0).unsqueeze(0)


def _identity(
    predicted: torch.Tensor, teacher: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = predicted.permute(2, 0, 1, 3, 4).reshape(predicted.shape[2], -1).double()
    right = teacher.permute(2, 0, 1, 3, 4).reshape(teacher.shape[2], -1).double()
    cosine = left @ right.T / (
        left.norm(dim=1, keepdim=True) * right.norm(dim=1)[None]
    ).clamp_min(1e-12)
    correct = cosine.diagonal()
    masked = cosine.masked_fill(torch.eye(cosine.shape[0], dtype=torch.bool), -torch.inf)
    wrong = masked.max(dim=1).values
    return correct, wrong, correct > wrong


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    left, right = first.flatten().double(), second.flatten().double()
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


__all__ = [
    "ProjectionDecision",
    "MechanisticMetricError",
    "RFCellMetric",
    "RFMetric",
    "ScientificCase",
    "evaluate_rf",
    "projection_decision",
    "rf_metric_payload",
]
