from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.candidate0_projection_solver import (
    physiological_path_signs,
    project_rf_columns_nonnegative,
)
from evaluation.rf_geometry_metrics import RFGeometryRequest, rf_geometry_metrics
from evaluation.v4_identity_endpoint import CellIdentityMetadata


def test_nonnegative_projection_recovers_feasible_path_signed_target() -> None:
    # Given
    generator = torch.Generator().manual_seed(81201)
    columns = torch.randn(2, 1, 16, 24, generator=generator, dtype=torch.float64)
    weights = torch.linspace(0.05, 1.2, 24, dtype=torch.float64)
    signed = columns * physiological_path_signs().view(1, 1, 1, 24)
    target = (signed * weights.view(1, 1, 1, 24)).sum(dim=-1)

    # When
    projection = project_rf_columns_nonnegative(columns, target)

    # Then
    torch.testing.assert_close(projection.projected, target, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(projection.weights[0], weights, atol=1e-8, rtol=1e-8)
    assert projection.cells[0].converged
    assert projection.cells[0].kkt_residual <= 1e-9


def test_nonnegative_projection_excludes_required_negative_weight() -> None:
    # Given
    columns = torch.eye(24, dtype=torch.float64).reshape(1, 1, 24, 24)
    signed = columns * physiological_path_signs().view(1, 1, 1, 24)
    target = -signed[..., 0]

    # When
    projection = project_rf_columns_nonnegative(columns, target)

    # Then
    assert projection.weights[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert projection.cells[0].excluded_energy == pytest.approx(1.0, abs=1e-12)
    assert projection.cells[0].converged


def test_rf_geometry_uses_spatiotemporal_energy_centers() -> None:
    # Given
    cone_positions = torch.tensor(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), dtype=torch.float64
    )
    teacher = torch.zeros(1, 1, 4, 2, 3, dtype=torch.float64)
    predicted = torch.zeros_like(teacher)
    teacher[0, 0, 0, 0, 0] = 1
    teacher[0, 0, 1, 1, 1] = 1
    teacher[0, 0, 2, 0, 1] = 1
    teacher[0, 0, 3, 1, 2] = 1
    predicted.copy_(teacher)
    predicted[0, 0, 1].zero_()
    predicted[0, 0, 1, 1, 2] = 1
    metadata = (
        CellIdentityMetadata("m-on", "midget", "ON", 0, 0, "r0"),
        CellIdentityMetadata("p-on", "parasol", "ON", 0, 0, "r0"),
        CellIdentityMetadata("m-off", "midget", "OFF", 0, 0, "r0"),
        CellIdentityMetadata("p-off", "parasol", "OFF", 0, 0, "r0"),
    )

    # When
    metric = rf_geometry_metrics(
        RFGeometryRequest(predicted, teacher, cone_positions, metadata)
    )

    # Then
    assert metric.cells[1].center_error == pytest.approx(1.0)
    assert metric.mean_center_error == pytest.approx(0.25)
    assert metric.mean_radius_error == pytest.approx(0.0)
    assert metric.mean_paired_center_distance_error == pytest.approx(0.5)
    assert np.isfinite(metric.pairwise_location_distance_correlation)
