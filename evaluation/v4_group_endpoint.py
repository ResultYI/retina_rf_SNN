from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Sequence

import torch

from evaluation.v4_identity_endpoint import CellIdentityMetadata
from evaluation.phase1_data import Phase1Assets
from evaluation.v4_group_scoring import (
    GroupEndpointKind,
    GroupScore,
    GroupSpec,
    between_group_distance,
    full_vectors,
    group_spec,
    score_groups,
    within_group_variance,
)


@dataclass(frozen=True, slots=True)
class GroupEndpointCell:
    cell_id: str
    group_key: str
    leave_one_out_defined: bool
    correct_group_cosine: float | None
    best_wrong_group: str | None
    best_wrong_group_cosine: float | None
    margin: float | None
    resolved: bool | None
    spatial_margin: float | None
    temporal_margin: float | None
    wrong_same_polarity: bool | None
    wrong_same_type: bool | None
    wrong_same_position: bool | None
    wrong_same_replicate: bool | None


@dataclass(frozen=True, slots=True)
class GroupEndpointAudit:
    endpoint: GroupEndpointKind
    group_keys: tuple[str, ...]
    valid_cell_count: int
    cells: tuple[GroupEndpointCell, ...]
    median_margin: float | None
    minimum_margin: float | None
    maximum_margin: float | None
    margin_at_least_0_01_fraction: float | None
    group_resolved_fraction: float | None
    mean_correct_group_cosine: float | None
    mean_best_wrong_group_cosine: float | None
    confusion_matrix: tuple[tuple[float | None, ...], ...]
    within_group_variance: float
    between_group_distance: float | None
    dominant_error_group_fraction: float
    gate_passed: bool


@dataclass(frozen=True, slots=True)
class GroupEndpointRequest:
    reference_rf: torch.Tensor
    metadata: Sequence[CellIdentityMetadata]
    endpoint: GroupEndpointKind
    query_rf: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class GroupEndpointError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class _CellContext:
    metadata: Sequence[CellIdentityMetadata]
    spec: GroupSpec


def audit_group_endpoint(request: GroupEndpointRequest) -> GroupEndpointAudit:
    reference = request.reference_rf
    query = request.query_rf if request.query_rf is not None else reference
    if reference.shape != query.shape or reference.ndim < 3:
        raise GroupEndpointError("reference and query RF tensors must share shape")
    if reference.shape[2] != len(request.metadata):
        raise GroupEndpointError("RF cell axis must match metadata")
    spec = group_spec(request.metadata, request.endpoint)
    full = score_groups(full_vectors(reference), full_vectors(query), spec)
    spatial = score_groups(
        full_vectors(reference.sum(dim=3)),
        full_vectors(query.sum(dim=3)),
        spec,
    )
    temporal = score_groups(
        full_vectors(reference.sum(dim=4)),
        full_vectors(query.sum(dim=4)),
        spec,
    )
    cell_context = _CellContext(request.metadata, spec)
    cells = tuple(
        _cell_result(index, cell_context, (full, spatial, temporal))
        for index in range(len(request.metadata))
    )
    valid = tuple(cell for cell in cells if cell.leave_one_out_defined)
    margins = tuple(cell.margin for cell in valid if cell.margin is not None)
    resolved = tuple(cell for cell in valid if cell.resolved)
    errors = tuple(cell for cell in valid if cell.resolved is False)
    dominant = _dominant_error_fraction(errors)
    margin_fraction = _fraction_at_threshold(margins)
    resolved_fraction = len(resolved) / len(valid) if valid else None
    median = float(statistics.median(margins)) if margins else None
    gate_passed = (
        len(valid) >= 12
        and margin_fraction is not None
        and margin_fraction >= 0.75
        and resolved_fraction is not None
        and resolved_fraction >= 0.75
        and median is not None
        and median > 0
        and dominant <= 0.5
    )
    return GroupEndpointAudit(
        request.endpoint,
        spec.keys,
        len(valid),
        cells,
        median,
        min(margins) if margins else None,
        max(margins) if margins else None,
        margin_fraction,
        resolved_fraction,
        _mean_optional(tuple(cell.correct_group_cosine for cell in valid)),
        _mean_optional(tuple(cell.best_wrong_group_cosine for cell in valid)),
        tuple(score.row for score in full),
        within_group_variance(full_vectors(reference), spec),
        between_group_distance(full_vectors(reference), spec),
        dominant,
        gate_passed,
    )


def _cell_result(
    index: int,
    context: _CellContext,
    scores: tuple[
        tuple[GroupScore, ...],
        tuple[GroupScore, ...],
        tuple[GroupScore, ...],
    ],
) -> GroupEndpointCell:
    full, spatial, temporal = scores
    score = full[index]
    wrong_indices = (
        () if score.wrong_group is None else context.spec.groups[score.wrong_group]
    )
    wrong = tuple(context.metadata[value] for value in wrong_indices)
    cell = context.metadata[index]
    return GroupEndpointCell(
        cell.cell_id,
        next(key for key, members in context.spec.groups.items() if index in members),
        score.correct is not None,
        score.correct,
        score.wrong_group,
        score.wrong,
        score.margin,
        score.resolved,
        spatial[index].margin,
        temporal[index].margin,
        _any_match(cell, wrong, "polarity"),
        _any_match(cell, wrong, "type_id"),
        _same_position(cell, wrong),
        _any_match(cell, wrong, "replicate_id"),
    )


def _any_match(
    cell: CellIdentityMetadata,
    wrong: Sequence[CellIdentityMetadata],
    field: str,
) -> bool | None:
    if not wrong:
        return None
    return any(getattr(other, field) == getattr(cell, field) for other in wrong)


def _same_position(
    cell: CellIdentityMetadata,
    wrong: Sequence[CellIdentityMetadata],
) -> bool | None:
    if not wrong:
        return None
    return any(
        math.isclose(other.position_x, cell.position_x, abs_tol=1e-7)
        and math.isclose(other.position_y, cell.position_y, abs_tol=1e-7)
        for other in wrong
    )


def endpoint_metadata(assets: Phase1Assets) -> tuple[CellIdentityMetadata, ...]:
    cells = assets.canonical_data.cells
    replicates = assets.validation_result.kernels["cell_replicate_id"]
    return tuple(
        CellIdentityMetadata(
            cell_id,
            type_id,
            "ON" if int(polarity) == 0 else "OFF",
            float(position[0]),
            float(position[1]),
            str(replicate),
        )
        for cell_id, type_id, polarity, position, replicate in zip(
            cells.ids,
            cells.type_ids,
            cells.polarities,
            cells.positions_degs,
            replicates,
            strict=True,
        )
    )


def _dominant_error_fraction(errors: Sequence[GroupEndpointCell]) -> float:
    if not errors:
        return 0.0
    counts = {cell.best_wrong_group: sum(other.best_wrong_group == cell.best_wrong_group for other in errors) for cell in errors}
    return max(counts.values()) / len(errors)


def _fraction_at_threshold(margins: tuple[float, ...]) -> float | None:
    return sum(value >= 0.01 for value in margins) / len(margins) if margins else None


def _mean_optional(values: tuple[float | None, ...]) -> float | None:
    present = tuple(value for value in values if value is not None)
    return sum(present) / len(present) if present else None


__all__ = [
    "GroupEndpointAudit",
    "GroupEndpointRequest",
    "GroupEndpointKind",
    "audit_group_endpoint",
    "endpoint_metadata",
]
