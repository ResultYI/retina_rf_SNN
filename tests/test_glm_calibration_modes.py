from __future__ import annotations

import pytest
import torch

from baselines.point_process_glm import (
    GLMError,
    PointProcessGLM,
    fit_point_process_glm,
)
from tests.calibration_fixture import calibration_data


def test_glm_modes_isolate_bias_history_and_stimulus_terms() -> None:
    # Given
    cones = torch.ones(1, 3, 1)
    counts = torch.tensor([[[1.0], [0.0], [0.0]]])
    models = {
        mode: PointProcessGLM(1, 1, 1, mode=mode)
        for mode in ("bias_only", "bias_plus_history", "full_glm")
    }
    with torch.no_grad():
        for model in models.values():
            model.bias.fill_(1.0)
            model.history.fill_(2.0)
            model.kernel.fill_(3.0)

    # When
    outputs = {mode: model(cones, counts) for mode, model in models.items()}

    # Then
    assert torch.equal(outputs["bias_only"], torch.ones_like(counts))
    assert outputs["bias_plus_history"][0, 0, 0] == 1.0
    assert outputs["bias_plus_history"][0, 1, 0] == 3.0
    assert outputs["full_glm"][0, 0, 0] == 4.0
    assert models["bias_only"].kernel.requires_grad is False
    assert models["bias_only"].history.requires_grad is False
    assert models["bias_plus_history"].kernel.requires_grad is False
    assert models["bias_plus_history"].history.requires_grad is True


def test_glm_full_mode_remains_the_default() -> None:
    # Given
    cones = torch.ones(1, 2, 1)
    counts = torch.zeros(1, 2, 1)
    default = PointProcessGLM(1, 1, 1)
    explicit = PointProcessGLM(1, 1, 1, mode="full_glm")
    explicit.load_state_dict(default.state_dict())

    # When / Then
    assert torch.equal(default(cones, counts), explicit(cones, counts))


def test_glm_rejects_unknown_mode() -> None:
    # Given / When / Then
    with pytest.raises(GLMError, match="GLM mode"):
        PointProcessGLM(1, 1, 1, mode="unknown")


def test_bias_only_starts_at_exact_training_rate_baseline() -> None:
    # Given
    data = calibration_data()

    # When
    result = fit_point_process_glm(
        data,
        device=torch.device("cpu"),
        steps=1,
        burn_in_steps=1,
        mode="bias_only",
    )

    # Then
    assert result.validation_metrics.nll == pytest.approx(
        result.validation_metrics.constant_rate_nll,
        abs=1e-7,
    )
