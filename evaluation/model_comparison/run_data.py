from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import torch

from evaluation.mechanistic_retina.rf_base import Candidate0Reference
from evaluation.model_comparison.config import ComparisonConfig
from evaluation.model_comparison.types import ProgressEvent
from training.mechanistic_retina.stages import MechanisticSeedData


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class BankRunData:
    root: Path
    config: ComparisonConfig
    data: MechanisticSeedData
    candidate: Candidate0Reference
    bank_seed: int
    match_target_parameters: int
    train_spikes: torch.Tensor
    validation_spikes: torch.Tensor
    train_mask: torch.Tensor
    validation_mask: torch.Tensor
    progress: ProgressCallback


__all__ = ["BankRunData", "ProgressCallback"]
