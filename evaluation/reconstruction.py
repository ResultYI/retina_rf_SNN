from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from training.config import DataConfig
from training.data import PreparedClip, augment_clip


class ReconstructionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    mse: float
    mean_baseline_mse: float
    representation_skill: float


def fit_reconstruction_scale(clips: Sequence[PreparedClip]) -> float:
    if not clips:
        raise ReconstructionError("At least one training clip is required")
    samples = torch.cat([clip.clean for clip in clips], dim=0).float()
    mean = samples.mean(dim=0, keepdim=True)
    return float((samples - mean).square().mean().clamp_min(1e-12))


def fit_augmented_reconstruction_scale(
    clips: Sequence[PreparedClip],
    data_config: DataConfig,
    *,
    seed: int,
    augmentations_per_source: int = 1,
) -> float:
    if not clips or augmentations_per_source < 1:
        raise ReconstructionError("Training clips and augmentations_per_source are required")
    generator = torch.Generator().manual_seed(seed)
    targets = [
        augment_clip(clip, data_config, generator).clean_target
        for clip in clips
        for _ in range(augmentations_per_source)
    ]
    samples = torch.cat(targets, dim=0).float()
    mean = samples.mean(dim=0, keepdim=True)
    return float((samples - mean).square().mean().clamp_min(1e-12))


def reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    train_mean: torch.Tensor,
) -> ReconstructionMetrics:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ReconstructionError("prediction and target must match [batch,time,cone]")
    if train_mean.shape != (target.shape[-1],):
        raise ReconstructionError("train_mean must have shape [cone]")
    mse = float((prediction - target).square().mean())
    baseline = float((target - train_mean.view(1, 1, -1)).square().mean())
    skill = 1.0 - mse / max(baseline, 1e-12)
    return ReconstructionMetrics(mse, baseline, skill)


__all__ = [
    "ReconstructionError",
    "ReconstructionMetrics",
    "fit_augmented_reconstruction_scale",
    "fit_reconstruction_scale",
    "reconstruction_metrics",
]
