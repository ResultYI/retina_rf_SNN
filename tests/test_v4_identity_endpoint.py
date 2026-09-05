from __future__ import annotations

import pytest
import torch

from evaluation.v4_identity_endpoint import (
    CellIdentityMetadata,
    ProjectedCellMetric,
    audit_projection_identity,
    audit_teacher_identity,
)


def _metadata() -> tuple[CellIdentityMetadata, ...]:
    return (
        CellIdentityMetadata("midget-on-r0", "midget", "ON", 0.0, 0.0, "r0"),
        CellIdentityMetadata("parasol-on-r0", "parasol", "ON", 0.0, 0.0, "r0"),
        CellIdentityMetadata("midget-off-r1", "midget", "OFF", 1.0, 0.0, "r1"),
    )


def test_teacher_duplicate_rf_invalidates_exact_cell_endpoint() -> None:
    # Given
    teacher = torch.tensor(
        [[[[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]]]
    )

    # When
    audit = audit_teacher_identity(teacher, _metadata())

    # Then
    assert not audit.endpoint_valid
    assert audit.ambiguous_fraction == pytest.approx(2 / 3)
    assert audit.near_duplicate_pairs == (("midget-on-r0", "parasol-on-r0"),)
    first = audit.cells[0]
    assert first.same_position and first.different_type
    assert first.midget_parasol_confusion
    assert first.replicate_ambiguity


def test_orthogonal_teacher_rf_supports_exact_cell_endpoint() -> None:
    # Given
    teacher = torch.eye(3).reshape(1, 1, 3, 1, 3)

    # When
    audit = audit_teacher_identity(teacher, _metadata())

    # Then
    assert audit.endpoint_valid
    assert audit.ambiguous_fraction == 0.0
    assert all(cell.intrinsic_margin == pytest.approx(1.0) for cell in audit.cells)


def test_projection_identity_is_stratified_by_teacher_separability() -> None:
    # Given
    teacher = audit_teacher_identity(
        torch.eye(3).reshape(1, 1, 3, 1, 3), _metadata()
    )
    projected = (
        ProjectedCellMetric("midget-on-r0", 0.8, 0.7, 0.1, True),
        ProjectedCellMetric("parasol-on-r0", 0.6, 0.7, -0.1, False),
        ProjectedCellMetric("midget-off-r1", 0.9, 0.2, 0.7, True),
    )

    # When
    audit = audit_projection_identity(projected, teacher)

    # Then
    assert audit.all_cell_identity_fraction == pytest.approx(2 / 3)
    assert audit.separable_cell_identity_fraction == pytest.approx(2 / 3)
    assert audit.same_position_cross_type_resolved_fraction == pytest.approx(0.5)
    assert audit.median_identity_margin == pytest.approx(0.1)
