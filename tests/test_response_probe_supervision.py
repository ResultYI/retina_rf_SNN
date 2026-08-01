from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from training.response_config import ResponseConfigurationError, load_response_config


ROOT = Path(__file__).resolve().parents[1]


def test_probe_supervision_selects_final_differentiable_bins() -> None:
    # Given
    config = load_response_config(ROOT / "configs" / "synthetic_smoke.yaml")

    # When
    training = replace(config.training, supervised_tail_steps=64)
    selected = torch.zeros(1, 256, 4)[:, training.supervision_slice]

    # Then
    assert selected.shape == (1, 64, 4)


def test_probe_supervision_rejects_tail_longer_than_differentiable_window() -> None:
    # Given
    config = load_response_config(ROOT / "configs" / "synthetic_smoke.yaml")

    # When
    with pytest.raises(ResponseConfigurationError, match="supervised tail"):
        # Then
        replace(config.training, supervised_tail_steps=257)
