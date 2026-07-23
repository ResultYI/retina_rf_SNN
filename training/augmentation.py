from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from training.config import DataConfig
from training.data import PreparedClip, TrainingDataError


@dataclass(frozen=True, slots=True)
class AugmentedClip:
    noisy_input: torch.Tensor
    clean_target: torch.Tensor
    metadata: dict[str, float | int | bool | str]

    @classmethod
    def stack(cls, clips: Sequence[AugmentedClip]) -> AugmentedClip:
        if not clips:
            raise TrainingDataError("At least one augmented clip is required")
        return cls(
            noisy_input=torch.cat([clip.noisy_input for clip in clips]),
            clean_target=torch.cat([clip.clean_target for clip in clips]),
            metadata={"batch_size": len(clips)},
        )


@dataclass(frozen=True, slots=True)
class ValidationScenario:
    name: str
    gain_before: float
    gain_after: float
    noise_before: float
    noise_after: float


def augment_clip(
    clip: PreparedClip,
    config: DataConfig,
    generator: torch.Generator,
) -> AugmentedClip:
    has_gain_transition = (
        torch.rand((), generator=generator).item()
        < config.context_transition_probability
    )
    has_noise_transition = (
        torch.rand((), generator=generator).item()
        < config.noise_transition_probability
    )
    gain_before = _log_uniform(
        config.context_gain_min,
        config.context_gain_max,
        generator,
    )
    noise_before = _uniform(
        config.noise_std_min,
        config.noise_std_max,
        generator,
    )
    gain_after = (
        _log_uniform(config.context_gain_min, config.context_gain_max, generator)
        if has_gain_transition
        else gain_before
    )
    noise_after = (
        _uniform(config.noise_std_min, config.noise_std_max, generator)
        if has_noise_transition
        else noise_before
    )
    transition_step = (
        int(
            torch.randint(
                max(1, config.sequence_steps // 3),
                config.context_transition_latest_step + 1,
                (),
                generator=generator,
            ).item()
        )
        if has_gain_transition or has_noise_transition
        else -1
    )
    return _build_augmented_clip(
        clip,
        config,
        generator,
        ValidationScenario(
            name="random_training",
            gain_before=gain_before,
            gain_after=gain_after,
            noise_before=noise_before,
            noise_after=noise_after,
        ),
        transition_step,
    )


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


def _build_augmented_clip(
    clip: PreparedClip,
    config: DataConfig,
    generator: torch.Generator,
    scenario: ValidationScenario,
    transition_step: int,
) -> AugmentedClip:
    clean = clip.clean
    if clean.ndim != 2 or clean.shape[0] != config.sequence_steps:
        raise TrainingDataError("Clean clip shape does not match data.sequence_steps")
    transition_width = max(2, config.sequence_steps // 64) if transition_step >= 0 else 0
    gain_envelope = _transition_envelope(
        clean,
        config.sequence_steps,
        scenario.gain_before,
        scenario.gain_after,
        transition_step,
        transition_width,
    )
    noise_envelope = _transition_envelope(
        clean,
        config.sequence_steps,
        scenario.noise_before,
        scenario.noise_after,
        transition_step,
        transition_width,
    )
    clean_target = clean * gain_envelope[:, None]
    noisy_input = clean_target + noise_envelope[:, None] * torch.randn(
        clean.shape,
        generator=generator,
        dtype=clean.dtype,
    )
    return AugmentedClip(
        noisy_input=noisy_input,
        clean_target=clean_target,
        metadata={
            "source_id": clip.source_id,
            "scenario": scenario.name,
            "has_transition": scenario.gain_before != scenario.gain_after,
            "has_noise_transition": scenario.noise_before != scenario.noise_after,
            "gain_before": scenario.gain_before,
            "gain_after": scenario.gain_after,
            "transition_step": transition_step,
            "transition_width_steps": transition_width,
            "noise_std": scenario.noise_before,
            "noise_std_before": scenario.noise_before,
            "noise_std_after": scenario.noise_after,
        },
    )


def _transition_envelope(
    reference: torch.Tensor,
    steps: int,
    before: float,
    after: float,
    transition_step: int,
    transition_width: int,
) -> torch.Tensor:
    envelope = reference.new_full((steps,), before)
    if transition_step < 0:
        return envelope
    start = max(0, transition_step - transition_width // 2)
    end = min(steps, start + transition_width)
    phase = torch.linspace(0.0, math.pi, end - start, dtype=reference.dtype)
    blend = (1.0 - torch.cos(phase)) / 2.0
    envelope[start:end] = before + (after - before) * blend
    envelope[end:] = after
    return envelope


def _log_uniform(
    minimum: float,
    maximum: float,
    generator: torch.Generator,
) -> float:
    lower = math.log(minimum)
    upper = math.log(maximum)
    return math.exp(
        lower + (upper - lower) * torch.rand((), generator=generator).item()
    )


def _uniform(
    minimum: float,
    maximum: float,
    generator: torch.Generator,
) -> float:
    return minimum + (maximum - minimum) * torch.rand(
        (),
        generator=generator,
    ).item()


__all__ = [
    "AugmentedClip",
    "ValidationScenario",
    "augment_clip",
    "fixed_validation_clips",
]
