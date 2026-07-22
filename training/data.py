from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from data.cone_response import (
    ConeResponseExport,
    load_cone_response,
    validate_formal_stimulus_splits,
)
from data.dataset import (
    apply_log_cone_stats,
    fit_log_cone_stats,
    validate_compatible_cone_exports,
)
from training.config import DataConfig


class TrainingDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedClip:
    clean: torch.Tensor
    source_id: str


@dataclass(frozen=True, slots=True)
class PreparedData:
    train: tuple[PreparedClip, ...]
    validation: tuple[PreparedClip, ...]
    positions_degs: np.ndarray
    dt_ms: float
    eccentricity_deg: float
    normalization_mean: np.ndarray
    normalization_scale: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AugmentedClip:
    noisy_input: torch.Tensor
    clean_target: torch.Tensor
    metadata: dict[str, float | int | bool | str]


def prepare_data(config: DataConfig) -> PreparedData:
    train_paths = tuple(Path(path) for path in sorted(glob.glob(config.train_glob)))
    validation_paths = tuple(
        Path(path) for path in sorted(glob.glob(config.validation_glob))
    )
    if not train_paths or not validation_paths:
        raise TrainingDataError("Training and validation globs must both match HDF5 files")
    train_exports = tuple(load_cone_response(path) for path in train_paths)
    validation_exports = tuple(load_cone_response(path) for path in validation_paths)
    validate_formal_stimulus_splits(train_exports, validation_exports)
    validate_compatible_cone_exports((*train_paths, *validation_paths))
    reference = train_exports[0]
    if any(export.response.shape[0] != config.sequence_steps for export in (*train_exports, *validation_exports)):
        raise TrainingDataError("Every export length must equal data.sequence_steps")
    mean, scale = fit_log_cone_stats(train_paths)

    def normalized(export: ConeResponseExport) -> PreparedClip:
        source_id = export.source_id or export.source_movie_id
        if not source_id:
            raise TrainingDataError("Every formal export requires a source id")
        clean = torch.from_numpy(
            apply_log_cone_stats(export.response, mean, scale, clip=5.0)
        )
        return PreparedClip(clean=clean, source_id=source_id)

    train_clips = tuple(normalized(export) for export in train_exports)
    validation_clips = tuple(normalized(export) for export in validation_exports)
    dt_ms = float(np.median(np.diff(reference.time_axis_seconds)) * 1000.0)
    manifest = {
        "mode": "synthetic_training_noise",
        "train_files": [str(path.resolve()) for path in train_paths],
        "validation_files": [str(path.resolve()) for path in validation_paths],
        "train_source_ids": [clip.source_id for clip in train_clips],
        "validation_source_ids": [clip.source_id for clip in validation_clips],
        "source_disjoint": True,
        "sequence_steps": config.sequence_steps,
        "dt_ms": dt_ms,
        "normalization": "per-cone train-only log mean/std; validation reuses train stats",
    }
    return PreparedData(
        train=train_clips,
        validation=validation_clips,
        positions_degs=reference.positions_degs,
        dt_ms=dt_ms,
        eccentricity_deg=reference.eccentricity_deg,
        normalization_mean=mean,
        normalization_scale=scale,
        manifest=manifest,
    )


def augment_clip(
    clip: PreparedClip,
    config: DataConfig,
    generator: torch.Generator,
) -> AugmentedClip:
    clean = clip.clean
    if clean.ndim != 2 or clean.shape[0] != config.sequence_steps:
        raise TrainingDataError("Clean clip shape does not match data.sequence_steps")
    gain_before = _log_uniform(
        config.context_gain_min, config.context_gain_max, generator
    )
    gain_after = gain_before
    has_gain_transition = (
        torch.rand((), generator=generator).item()
        < config.context_transition_probability
    )
    noise_before = config.noise_std_min + (
        config.noise_std_max - config.noise_std_min
    ) * torch.rand((), generator=generator).item()
    noise_after = noise_before
    has_noise_transition = (
        torch.rand((), generator=generator).item()
        < config.noise_transition_probability
    )
    gain_envelope = clean.new_full((config.sequence_steps,), gain_before)
    noise_envelope = clean.new_full((config.sequence_steps,), noise_before)
    transition_step = -1
    transition_width = 0
    if has_gain_transition or has_noise_transition:
        transition_low = max(1, config.sequence_steps // 3)
        transition_high = max(
            transition_low + 1,
            min(config.context_transition_latest_step + 1, config.sequence_steps),
        )
        transition_step = int(
            torch.randint(transition_low, transition_high, (), generator=generator).item()
        )
        transition_width = max(2, config.sequence_steps // 64)
        start = max(0, transition_step - transition_width // 2)
        end = min(config.sequence_steps, start + transition_width)
        phase = torch.linspace(0.0, math.pi, end - start, dtype=clean.dtype)
        blend = (1.0 - torch.cos(phase)) / 2.0
        if has_gain_transition:
            gain_after = _log_uniform(
                config.context_gain_min, config.context_gain_max, generator
            )
            gain_envelope[start:end] = gain_before + (
                gain_after - gain_before
            ) * blend
            gain_envelope[end:] = gain_after
        if has_noise_transition:
            noise_after = config.noise_std_min + (
                config.noise_std_max - config.noise_std_min
            ) * torch.rand((), generator=generator).item()
            noise_envelope[start:end] = noise_before + (
                noise_after - noise_before
            ) * blend
            noise_envelope[end:] = noise_after
    clean_target = clean * gain_envelope[:, None]
    noisy_input = clean_target + noise_envelope[:, None] * torch.randn(
        clean.shape, generator=generator, dtype=clean.dtype
    )
    return AugmentedClip(
        noisy_input=noisy_input,
        clean_target=clean_target,
        metadata={
            "source_id": clip.source_id,
            "has_transition": has_gain_transition,
            "has_noise_transition": has_noise_transition,
            "gain_before": gain_before,
            "gain_after": gain_after,
            "transition_step": transition_step,
            "transition_width_steps": transition_width,
            "noise_std": noise_before,
            "noise_std_before": noise_before,
            "noise_std_after": noise_after,
        },
    )


def fixed_validation_clips(
    clips: Sequence[PreparedClip],
    config: DataConfig,
    seed: int,
    device: torch.device,
) -> tuple[AugmentedClip, ...]:
    result: list[AugmentedClip] = []
    for index, clip in enumerate(clips):
        augmented = augment_clip(
            clip,
            config,
            torch.Generator().manual_seed(seed + index),
        )
        result.append(
            AugmentedClip(
                noisy_input=augmented.noisy_input.unsqueeze(0).to(device),
                clean_target=augmented.clean_target.unsqueeze(0).to(device),
                metadata=augmented.metadata,
            )
        )
    return tuple(result)


def _log_uniform(minimum: float, maximum: float, generator: torch.Generator) -> float:
    lower = math.log(minimum)
    upper = math.log(maximum)
    return math.exp(lower + (upper - lower) * torch.rand((), generator=generator).item())


__all__ = [
    "AugmentedClip",
    "PreparedClip",
    "PreparedData",
    "TrainingDataError",
    "augment_clip",
    "fixed_validation_clips",
    "prepare_data",
]
