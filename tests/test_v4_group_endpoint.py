from __future__ import annotations

import pytest
import torch

from evaluation.v4_group_endpoint import (
    GroupEndpointKind,
    GroupEndpointRequest,
    audit_group_endpoint,
)
from evaluation.v4_identity_endpoint import CellIdentityMetadata


def _paired_metadata(group_count: int) -> tuple[CellIdentityMetadata, ...]:
    rows = []
    for group in range(group_count):
        polarity = "ON" if group < group_count // 2 else "OFF"
        for type_id in ("midget", "parasol"):
            rows.append(
                CellIdentityMetadata(
                    f"{type_id}-{polarity}-r{group}",
                    type_id,
                    polarity,
                    float(group),
                    0.0,
                    f"r{group}",
                )
            )
    return tuple(rows)


def test_type_collapsed_location_uses_leave_one_out_centroid() -> None:
    # Given
    metadata = _paired_metadata(2)
    vectors = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    ).reshape(1, 1, 4, 1, 2)

    # When
    audit = audit_group_endpoint(
        GroupEndpointRequest(
            vectors,
            metadata,
            GroupEndpointKind.TYPE_COLLAPSED_POLARITY_LOCATION,
        )
    )

    # Then
    assert audit.valid_cell_count == 4
    assert audit.group_resolved_fraction == 1.0
    assert all(cell.correct_group_cosine == pytest.approx(1.0) for cell in audit.cells)
    assert all(cell.margin == pytest.approx(1.0) for cell in audit.cells)


def test_singleton_group_is_excluded_from_endpoint_gate() -> None:
    # Given
    metadata = _paired_metadata(2)
    vectors = torch.eye(4).reshape(1, 1, 4, 1, 4)

    # When
    audit = audit_group_endpoint(
        GroupEndpointRequest(
            vectors,
            metadata,
            GroupEndpointKind.TYPE_POLARITY_LOCATION,
        )
    )

    # Then
    assert audit.valid_cell_count == 0
    assert audit.group_resolved_fraction is None
    assert not audit.gate_passed
    assert all(not cell.leave_one_out_defined for cell in audit.cells)


def test_endpoint_gate_passes_for_sixteen_separable_paired_cells() -> None:
    # Given
    metadata = _paired_metadata(8)
    vectors = torch.eye(8).repeat_interleave(2, dim=0).reshape(1, 1, 16, 1, 8)

    # When
    audit = audit_group_endpoint(
        GroupEndpointRequest(
            vectors,
            metadata,
            GroupEndpointKind.TYPE_COLLAPSED_POLARITY_LOCATION,
        )
    )

    # Then
    assert audit.valid_cell_count == 16
    assert audit.margin_at_least_0_01_fraction == 1.0
    assert audit.group_resolved_fraction == 1.0
    assert audit.median_margin == pytest.approx(1.0)
    assert audit.dominant_error_group_fraction == 0.0
    assert audit.gate_passed
