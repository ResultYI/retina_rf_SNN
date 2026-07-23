from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from training.augmentation import (
    AugmentedClip,
    augment_clip,
    augment_clip_pair,
)
from training.config import ExperimentConfig
from training.data import PreparedClip
from training.runtime import sample_unique_source_indices


@dataclass(frozen=True, slots=True)
class TrainingGenerators:
    sampling: torch.Generator
    augmentation: torch.Generator


@dataclass(frozen=True, slots=True)
class TrainingBatchRequest:
    sources: Sequence[PreparedClip]
    config: ExperimentConfig
    device: torch.device
    generators: TrainingGenerators
    optimizer_step: int


def build_training_batch(
    request: TrainingBatchRequest,
) -> tuple[AugmentedClip, ...]:
    source_indices = sample_unique_source_indices(
        len(request.sources),
        request.config.training.batch_size,
        request.generators.sampling,
    )
    first_views: list[AugmentedClip] = []
    second_views: list[AugmentedClip] = []
    bootstrap_active = (
        request.optimizer_step
        < request.config.training.reconstruction_bootstrap_steps
    )
    for source_index_tensor in source_indices:
        source = request.sources[int(source_index_tensor)]
        if bootstrap_active:
            first, second = augment_clip_pair(
                source,
                request.config.data,
                request.generators.augmentation,
            )
            first_views.append(_to_device(first, request.device))
            second_views.append(_to_device(second, request.device))
        else:
            first_views.append(
                _to_device(
                    augment_clip(
                        source,
                        request.config.data,
                        request.generators.augmentation,
                    ),
                    request.device,
                )
            )
    return tuple((*first_views, *second_views))


def _to_device(
    clip: AugmentedClip,
    device: torch.device,
) -> AugmentedClip:
    return AugmentedClip(
        noisy_input=clip.noisy_input.unsqueeze(0).to(device),
        clean_target=clip.clean_target.unsqueeze(0).to(device),
        metadata=clip.metadata,
    )


__all__ = [
    "TrainingBatchRequest",
    "TrainingGenerators",
    "build_training_batch",
]
