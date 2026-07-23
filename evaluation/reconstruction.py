from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from training.config import DataConfig
from training.augmentation import augment_clip
from training.data import PreparedClip


class ReconstructionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    mse: float
    mean_baseline_mse: float
    noisy_current_mse: float
    causal_ema_mse: float
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
    noisy_input: torch.Tensor,
    ema_alpha: float,
) -> ReconstructionMetrics:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ReconstructionError("prediction and target must match [batch,time,cone]")
    if noisy_input.shape != target.shape:
        raise ReconstructionError("noisy_input and target must match")
    if train_mean.shape != (target.shape[-1],):
        raise ReconstructionError("train_mean must have shape [cone]")
    mse = float((prediction - target).square().mean())
    baseline = float((target - train_mean.view(1, 1, -1)).square().mean())
    noisy_mse = float((noisy_input - target).square().mean())
    ema_mse = float((causal_ema(noisy_input, ema_alpha) - target).square().mean())
    skill = 1.0 - mse / max(baseline, 1e-12)
    return ReconstructionMetrics(mse, baseline, noisy_mse, ema_mse, skill)


def causal_ema(noisy_input: torch.Tensor, alpha: float) -> torch.Tensor:
    if noisy_input.ndim != 3 or not 0 <= alpha < 1:
        raise ReconstructionError(
            "noisy_input must be [batch,time,cone] and alpha inside [0,1)"
        )
    filtered = torch.empty_like(noisy_input)
    filtered[:, 0] = noisy_input[:, 0]
    for time_index in range(1, noisy_input.shape[1]):
        filtered[:, time_index] = (
            alpha * filtered[:, time_index - 1]
            + (1.0 - alpha) * noisy_input[:, time_index]
        )
    return filtered


def fit_causal_ema_alpha(
    noisy_input: torch.Tensor,
    target: torch.Tensor,
    *,
    candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9),
) -> float:
    if noisy_input.shape != target.shape or not candidates:
        raise ReconstructionError("Matching training observations are required")
    scored = (
        (
            float((causal_ema(noisy_input, alpha) - target).square().mean()),
            alpha,
        )
        for alpha in candidates
    )
    return min(scored)[1]


__all__ = [
    "ReconstructionError",
    "ReconstructionMetrics",
    "causal_ema",
    "fit_causal_ema_alpha",
    "fit_augmented_reconstruction_scale",
    "fit_reconstruction_scale",
    "reconstruction_metrics",
]
