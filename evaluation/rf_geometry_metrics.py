from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from evaluation.v4_identity_endpoint import CellIdentityMetadata


@dataclass(frozen=True, slots=True)
class RFGeometryRequest:
    predicted: torch.Tensor
    teacher: torch.Tensor
    cone_positions: torch.Tensor
    metadata: Sequence[CellIdentityMetadata]


@dataclass(frozen=True, slots=True)
class RFGeometryCell:
    cell_id: str
    predicted_center: tuple[float, float]
    teacher_center: tuple[float, float]
    center_error: float
    predicted_radius: float
    teacher_radius: float
    radius_error: float


@dataclass(frozen=True, slots=True)
class RFGeometryMetric:
    cells: tuple[RFGeometryCell, ...]
    mean_center_error: float
    median_center_error: float
    mean_radius_error: float
    pairwise_location_distance_correlation: float
    mean_paired_center_distance_error: float


@dataclass(frozen=True, slots=True)
class RFGeometryError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def rf_geometry_metrics(request: RFGeometryRequest) -> RFGeometryMetric:
    predicted, teacher = request.predicted, request.teacher
    if predicted.shape != teacher.shape or predicted.ndim < 5:
        raise RFGeometryError("geometry RF tensors must share source/context/cell/lag/cone shape")
    if predicted.shape[2] != len(request.metadata):
        raise RFGeometryError("geometry metadata must match the RF cell axis")
    if request.cone_positions.shape != (predicted.shape[-1], 2):
        raise RFGeometryError("cone positions must have shape [cone,2]")
    predicted_centers, predicted_radii = _centers_and_radii(
        predicted, request.cone_positions
    )
    teacher_centers, teacher_radii = _centers_and_radii(
        teacher, request.cone_positions
    )
    center_errors = torch.linalg.vector_norm(predicted_centers - teacher_centers, dim=1)
    radius_errors = (predicted_radii - teacher_radii).abs()
    cells = tuple(
        RFGeometryCell(
            metadata.cell_id,
            tuple(float(value) for value in predicted_centers[index]),
            tuple(float(value) for value in teacher_centers[index]),
            float(center_errors[index]),
            float(predicted_radii[index]),
            float(teacher_radii[index]),
            float(radius_errors[index]),
        )
        for index, metadata in enumerate(request.metadata)
    )
    return RFGeometryMetric(
        cells,
        float(center_errors.mean()),
        float(center_errors.median()),
        float(radius_errors.mean()),
        _distance_correlation(predicted_centers, teacher_centers),
        _paired_distance_error(
            predicted_centers,
            teacher_centers,
            request.metadata,
        ),
    )


def _centers_and_radii(
    rf: torch.Tensor,
    cone_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = rf.movedim(2, 0).double()
    energy = values.square().reshape(values.shape[0], -1, values.shape[-1]).sum(dim=1)
    totals = energy.sum(dim=1)
    if bool((totals <= torch.finfo(energy.dtype).tiny).any()):
        raise RFGeometryError("geometry metrics require nonzero RF energy")
    positions = cone_positions.double()
    centers = energy @ positions / totals[:, None]
    squared_distance = (positions[None] - centers[:, None]).square().sum(dim=2)
    radii = (energy.mul(squared_distance).sum(dim=1) / totals).sqrt()
    return centers, radii


def _distance_correlation(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
) -> float:
    indices = torch.triu_indices(predicted.shape[0], predicted.shape[0], offset=1)
    predicted_distances = torch.linalg.vector_norm(
        predicted[indices[0]] - predicted[indices[1]], dim=1
    )
    teacher_distances = torch.linalg.vector_norm(
        teacher[indices[0]] - teacher[indices[1]], dim=1
    )
    left = predicted_distances - predicted_distances.mean()
    right = teacher_distances - teacher_distances.mean()
    denominator = left.norm() * right.norm()
    return 0.0 if float(denominator) == 0 else float(torch.dot(left, right) / denominator)


def _paired_distance_error(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    metadata: Sequence[CellIdentityMetadata],
) -> float:
    pairs = tuple(
        (left, right)
        for left, first in enumerate(metadata)
        for right, second in enumerate(metadata[left + 1 :], start=left + 1)
        if first.polarity == second.polarity
        and first.replicate_id == second.replicate_id
        and first.type_id != second.type_id
        and math.isclose(first.position_x, second.position_x, abs_tol=1e-7)
        and math.isclose(first.position_y, second.position_y, abs_tol=1e-7)
    )
    if not pairs:
        raise RFGeometryError("geometry metrics require cross-type paired locations")
    errors = tuple(
        abs(
            float(torch.linalg.vector_norm(predicted[left] - predicted[right]))
            - float(torch.linalg.vector_norm(teacher[left] - teacher[right]))
        )
        for left, right in pairs
    )
    return sum(errors) / len(errors)


__all__ = [
    "RFGeometryCell",
    "RFGeometryError",
    "RFGeometryMetric",
    "RFGeometryRequest",
    "rf_geometry_metrics",
]
