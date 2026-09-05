from __future__ import annotations

import torch

from evaluation.candidate0_likelihood_math import (
    StaticTargetRequest,
    SupportAuditRequest,
    audit_projection_support,
    build_static_teacher_targets,
    causal_static_drive,
)


def test_causal_static_drive_uses_oldest_to_current_lag_order() -> None:
    # Given
    cones = torch.tensor([[[1.0], [2.0], [3.0]]])
    rf = torch.tensor([[[10.0], [1.0]]])

    # When
    drive = causal_static_drive(cones, rf)

    # Then
    torch.testing.assert_close(drive[0, :, 0], torch.tensor([1.0, 12.0, 23.0]))


def test_static_teacher_targets_use_train_only_calibration() -> None:
    # Given
    train = torch.tensor([[[0.0], [2.0], [4.0]]])
    validation = torch.tensor([[[6.0], [8.0]]])
    mask = torch.ones(1, 2, 3, 1, dtype=torch.bool)

    # When
    result = build_static_teacher_targets(
        StaticTargetRequest(train, validation, mask, 2, 0, -2.0)
    )

    # Then
    assert result.train_probabilities.shape == (1, 2, 3, 1)
    torch.testing.assert_close(result.drive_mean, torch.tensor([2.0]))
    torch.testing.assert_close(result.drive_std, torch.tensor([8.0 / 3.0]).sqrt())
    torch.testing.assert_close(
        result.validation_logits[0, 0, :, 0],
        torch.tensor([(6.0 - 2.0) / (8.0 / 3.0) ** 0.5 - 2.0,
                      (8.0 - 2.0) / (8.0 / 3.0) ** 0.5 - 2.0]),
    )


def test_projection_support_audit_finds_feasible_target_direction() -> None:
    # Given
    initial = torch.tensor([0.0, 0.0, 1.0, 1.0])
    gradient = torch.tensor([-2.0, 1.0, -1.0, 1.0])
    target = torch.tensor([1.0, 1.0, 2.0, 0.0])

    # When
    result = audit_projection_support(
        SupportAuditRequest(initial, gradient, target, 1e-8)
    )

    # Then
    assert result.positive_zero_count == 2
    assert result.positive_zero_feasible_fraction == 0.5
    assert result.parameter_direction_cosine > 0
