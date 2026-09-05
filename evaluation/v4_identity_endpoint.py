from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Sequence

import torch


@dataclass(frozen=True, slots=True)
class CellIdentityMetadata:
    cell_id: str
    type_id: str
    polarity: str
    position_x: float
    position_y: float
    replicate_id: str


@dataclass(frozen=True, slots=True)
class ProjectedCellMetric:
    cell_id: str
    correct_cosine: float
    best_wrong_cosine: float
    identity_margin: float
    identity_resolved: bool


@dataclass(frozen=True, slots=True)
class IntrinsicCellIdentity:
    cell_id: str
    best_wrong_cell_id: str
    best_wrong_cosine: float
    intrinsic_margin: float
    intrinsically_separable: bool
    same_position: bool
    different_type: bool
    same_type: bool
    different_position: bool
    midget_parasol_confusion: bool
    on_off_confusion: bool
    replicate_ambiguity: bool


@dataclass(frozen=True, slots=True)
class TeacherIdentityAudit:
    pairwise_cosine: tuple[tuple[float, ...], ...]
    cells: tuple[IntrinsicCellIdentity, ...]
    ambiguous_fraction: float
    near_duplicate_pairs: tuple[tuple[str, str], ...]
    endpoint_valid: bool


@dataclass(frozen=True, slots=True)
class ProjectionIdentityCell:
    cell_id: str
    teacher_best_wrong_cell_id: str
    teacher_intrinsic_margin: float
    projected_margin: float
    normalized_margin_recovery: float
    identity_resolved: bool
    intrinsically_separable: bool


@dataclass(frozen=True, slots=True)
class ProjectionIdentityAudit:
    cells: tuple[ProjectionIdentityCell, ...]
    all_cell_identity_fraction: float
    separable_cell_identity_fraction: float | None
    same_position_cross_type_resolved_fraction: float | None
    median_identity_margin: float


def audit_teacher_identity(
    teacher_rf: torch.Tensor,
    metadata: Sequence[CellIdentityMetadata],
    *,
    margin_threshold: float = 0.01,
    duplicate_tolerance: float = 1e-8,
) -> TeacherIdentityAudit:
    if teacher_rf.ndim < 3:
        raise ValueError("teacher_rf must include source, context, and cell axes")
    if teacher_rf.shape[2] != len(metadata) or len(metadata) < 2:
        raise ValueError("teacher_rf cell axis must match at least two metadata rows")
    if margin_threshold <= 0 or duplicate_tolerance < 0:
        raise ValueError("identity thresholds must be positive")

    vectors = teacher_rf.movedim(2, 0).reshape(len(metadata), -1).double()
    norms = torch.linalg.vector_norm(vectors, dim=1)
    if not bool(torch.isfinite(vectors).all()) or bool((norms == 0).any()):
        raise ValueError("teacher RFs must be finite and nonzero")
    normalized = vectors / norms[:, None]
    pairwise = (normalized @ normalized.T).clamp(-1.0, 1.0)
    wrong = pairwise.clone()
    wrong.fill_diagonal_(-torch.inf)
    best_values, best_indices = wrong.max(dim=1)

    cells = tuple(
        _intrinsic_cell(
            metadata[index],
            metadata[int(best_indices[index])],
            float(best_values[index]),
            margin_threshold,
        )
        for index in range(len(metadata))
    )
    near_duplicates = tuple(
        (metadata[left].cell_id, metadata[right].cell_id)
        for left in range(len(metadata))
        for right in range(left + 1, len(metadata))
        if float(pairwise[left, right]) >= 1.0 - duplicate_tolerance
    )
    ambiguous_fraction = sum(
        cell.intrinsic_margin < margin_threshold for cell in cells
    ) / len(cells)
    return TeacherIdentityAudit(
        tuple(tuple(float(value) for value in row) for row in pairwise),
        cells,
        ambiguous_fraction,
        near_duplicates,
        ambiguous_fraction <= 0.25 and not near_duplicates,
    )


def audit_projection_identity(
    projected: Sequence[ProjectedCellMetric],
    teacher: TeacherIdentityAudit,
) -> ProjectionIdentityAudit:
    teacher_by_id = {cell.cell_id: cell for cell in teacher.cells}
    if {cell.cell_id for cell in projected} != set(teacher_by_id):
        raise ValueError("projected and teacher cell identities must match")
    if not all(
        all(
            math.isfinite(value)
            for value in (
                cell.correct_cosine,
                cell.best_wrong_cosine,
                cell.identity_margin,
            )
        )
        for cell in projected
    ):
        raise ValueError("projected identity metrics must be finite")

    cells = tuple(
        ProjectionIdentityCell(
            cell.cell_id,
            teacher_by_id[cell.cell_id].best_wrong_cell_id,
            teacher_by_id[cell.cell_id].intrinsic_margin,
            cell.identity_margin,
            cell.identity_margin / (teacher_by_id[cell.cell_id].intrinsic_margin + 1e-12),
            cell.identity_resolved,
            teacher_by_id[cell.cell_id].intrinsically_separable,
        )
        for cell in projected
    )
    separable = tuple(cell for cell in cells if cell.intrinsically_separable)
    cross_type_ids = {
        cell.cell_id
        for cell in teacher.cells
        if cell.same_position and cell.different_type
    }
    cross_type = tuple(cell for cell in cells if cell.cell_id in cross_type_ids)
    return ProjectionIdentityAudit(
        cells,
        _resolved_fraction(cells),
        _resolved_fraction(separable) if separable else None,
        _resolved_fraction(cross_type) if cross_type else None,
        float(statistics.median(cell.projected_margin for cell in cells)),
    )


def _intrinsic_cell(
    cell: CellIdentityMetadata,
    wrong: CellIdentityMetadata,
    best_wrong_cosine: float,
    margin_threshold: float,
) -> IntrinsicCellIdentity:
    same_position = (
        math.isclose(cell.position_x, wrong.position_x, abs_tol=1e-7)
        and math.isclose(cell.position_y, wrong.position_y, abs_tol=1e-7)
    )
    different_type = cell.type_id != wrong.type_id
    return IntrinsicCellIdentity(
        cell.cell_id,
        wrong.cell_id,
        best_wrong_cosine,
        1.0 - best_wrong_cosine,
        1.0 - best_wrong_cosine >= margin_threshold,
        same_position,
        different_type,
        not different_type,
        not same_position,
        {cell.type_id, wrong.type_id} == {"midget", "parasol"},
        cell.polarity != wrong.polarity,
        cell.replicate_id == wrong.replicate_id,
    )


def _resolved_fraction(cells: Sequence[ProjectionIdentityCell]) -> float:
    return sum(cell.identity_resolved for cell in cells) / len(cells)


__all__ = [
    "CellIdentityMetadata",
    "ProjectedCellMetric",
    "ProjectionIdentityAudit",
    "TeacherIdentityAudit",
    "audit_projection_identity",
    "audit_teacher_identity",
]
