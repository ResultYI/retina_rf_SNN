from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.static_teacher_preflight import (
    build_static_teacher_candidate,
    noise_free_glm_rf_cosine,
)


def _cone_positions() -> np.ndarray:
    x = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    return np.stack((x, np.zeros_like(x)), axis=1)


def test_static_teacher_candidate_is_deterministic_and_polarity_opposed() -> None:
    # Given
    cone_positions = _cone_positions()

    # When
    first = build_static_teacher_candidate(cone_positions, 0)
    second = build_static_teacher_candidate(cone_positions, 0)

    # Then
    torch.testing.assert_close(first.rf, second.rf, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.rf[:, :, :4], -first.rf[:, :, 4:8])
    torch.testing.assert_close(first.rf[:, :, 8:12], -first.rf[:, :, 12:16])
    assert all(float(norm) > 1e-8 for norm in first.rf.flatten(3).norm(dim=-1).flatten())


def test_noise_free_full_glm_recovers_static_rf() -> None:
    # Given
    torch.manual_seed(73101)
    rf = torch.randn(1, 1, 3, 4, 5)

    # When
    cosine = noise_free_glm_rf_cosine(rf, seed=73102)

    # Then
    assert cosine == pytest.approx(1.0, abs=1e-8)
