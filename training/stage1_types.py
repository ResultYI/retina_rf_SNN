from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.hybrid import RetinaTrainingBatch, TrainingStage


class TrainStage1Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainStage1Config:
    train_h5: tuple[Path, ...]
    val_h5: tuple[Path, ...]
    output_dir: Path
    epochs: int
    batch_size: int
    input_steps: int
    horizons: tuple[int, ...]
    stage: TrainingStage
    device: torch.device
    seed: int
    t_bptt: int
    lr_core: float
    lr_decoder: float
    num_workers: int
    max_clip_fraction: float
    resume: Path | None
    formal_evidence: bool = False


@dataclass(frozen=True, slots=True)
class Stage1Loaders:
    train: DataLoader[RetinaTrainingBatch]
    val: DataLoader[RetinaTrainingBatch] | None


@dataclass(frozen=True, slots=True)
class CheckpointProgress:
    epoch: int
    step: int
    best_loss: float
