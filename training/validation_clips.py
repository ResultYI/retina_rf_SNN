from __future__ import annotations

from typing import Sequence

import torch

from training.augmentation import (
    AugmentedClip,
    ValidationScenario,
    _build_augmented_clip,
)
from training.config import DataConfig
from training.data import PreparedClip


def fixed_validation_clips(
    clips: Sequence[PreparedClip],
    config: DataConfig,
    seed: int,
    device: torch.device,
) -> tuple[AugmentedClip, ...]:
    scenarios = (
        ValidationScenario(
            name="low_gain_high_noise_to_high_gain_low_noise",
            gain_before=config.context_gain_min,
            gain_after=config.context_gain_max,
            noise_before=config.noise_std_max,
            noise_after=config.noise_std_min,
        ),
        ValidationScenario(
            name="high_gain_low_noise_to_low_gain_high_noise",
            gain_before=config.context_gain_max,
            gain_after=config.context_gain_min,
            noise_before=config.noise_std_min,
            noise_after=config.noise_std_max,
        ),
    )
    result: list[AugmentedClip] = []
    for source_index, clip in enumerate(clips):
        for scenario_index, scenario in enumerate(scenarios):
            augmented = _build_augmented_clip(
                clip,
                config,
                torch.Generator().manual_seed(
                    seed + 2 * source_index + scenario_index
                ),
                scenario,
                config.context_transition_latest_step,
            )
            result.append(
                AugmentedClip(
                    noisy_input=augmented.noisy_input.unsqueeze(0).to(device),
                    clean_target=augmented.clean_target.unsqueeze(0).to(device),
                    metadata=augmented.metadata,
                )
            )
    return tuple(result)


__all__ = ["fixed_validation_clips"]
