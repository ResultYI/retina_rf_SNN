from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


class RetinaTrainingBatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetinaTrainingSample:
    x_cone: torch.Tensor
    target_current: torch.Tensor
    time_index: torch.Tensor


@dataclass(frozen=True, slots=True)
class RetinaTrainingBatch:
    x_cone: torch.Tensor
    target_current: torch.Tensor


def collate_retina_training_batch(
    samples: Sequence[RetinaTrainingSample],
) -> RetinaTrainingBatch:
    if not samples:
        raise RetinaTrainingBatchError("Cannot collate an empty batch")
    return RetinaTrainingBatch(
        x_cone=torch.stack(tuple(sample.x_cone for sample in samples), dim=0),
        target_current=torch.stack(
            tuple(sample.target_current for sample in samples), dim=0
        ),
    )


def retina_training_sample_from_mapping(
    sample: Mapping[str, torch.Tensor],
) -> RetinaTrainingSample:
    if "target_current" not in sample:
        raise RetinaTrainingBatchError("Training samples require a current target")
    return RetinaTrainingSample(
        x_cone=sample["x_cone"],
        target_current=sample["target_current"],
        time_index=sample["time_index"],
    )

