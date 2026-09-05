from __future__ import annotations

from pathlib import Path

import torch

from evaluation.mechanistic_retina.rf_base import (
    CandidateTeacherUsage,
    base_rf,
    load_candidate0,
    project_base_rf,
)
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina


def _model():
    cone_positions = torch.tensor(
        [[0.00, 0.00], [0.05, 0.00], [0.10, 0.00], [0.15, 0.00]],
        dtype=torch.float32,
    )
    cell_positions = torch.tensor(
        [[0.04, 0.00], [0.11, 0.00]], dtype=torch.float32
    )
    return build_mechanistic_retina(
        MechanisticRetinaConfig(),
        cone_positions,
        cell_positions,
        ("midget", "parasol"),
        ("ON", "OFF"),
    )


def test_signed_projection_exactly_recovers_an_in_span_rf() -> None:
    # Given
    model = _model()
    target = base_rf(model).detach()

    # When
    projection = project_base_rf(model, target, constrained=False)

    # Then
    assert torch.allclose(projection.projected_rf, target, atol=1e-6, rtol=0)
    assert projection.converged


def test_constrained_projection_exactly_recovers_positive_path_weights() -> None:
    # Given
    model = _model()
    target = base_rf(model).detach()

    # When
    projection = project_base_rf(model, target, constrained=True)

    # Then
    assert torch.all(projection.coefficients >= 0)
    assert torch.allclose(projection.projected_rf, target, atol=1e-6, rtol=0)
    assert projection.converged


def test_candidate0_loader_preserves_frozen_tensor_identity() -> None:
    # Given
    path = Path(
        ".omo/evidence/hierarchical-endpoint-and-v4-decision/teacher-preflight-results.json"
    )

    # When
    candidate = load_candidate0(
        path,
        usage=CandidateTeacherUsage.DEVELOPMENT_REFERENCE,
        reference_candidate_index=0,
    )

    # Then
    assert candidate.rf.shape == (16, 16, 29)
    assert len(candidate.rf_sha256) == 64
    assert tuple(value.cell_id for value in candidate.metadata)[0] == (
        "synthetic-midget-on-r0"
    )
    assert candidate.teacher_usage is CandidateTeacherUsage.DEVELOPMENT_REFERENCE
    assert not candidate.preflight_passed
