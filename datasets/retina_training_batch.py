from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from training.hybrid import RetinaTargets, RetinaTrainingBatch


class RetinaTrainingBatchError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetinaTrainingSample:
    x_cone: torch.Tensor
    target_fine: torch.Tensor
    target_coarse: torch.Tensor
    target_delta: torch.Tensor | None
    time_index: torch.Tensor


def collate_retina_training_batch(
    samples: Sequence[RetinaTrainingSample],
) -> RetinaTrainingBatch:
    if not samples:
        raise RetinaTrainingBatchError("Cannot collate an empty batch")
    return RetinaTrainingBatch(
        x_cone=torch.stack(tuple(sample.x_cone for sample in samples), dim=0),
        targets=RetinaTargets(
            fine=torch.stack(tuple(sample.target_fine for sample in samples), dim=0),
            coarse=torch.stack(tuple(sample.target_coarse for sample in samples), dim=0),
        ),
    )


def retina_training_sample_from_mapping(
    sample: Mapping[str, torch.Tensor],
) -> RetinaTrainingSample:
    if "target_fine" not in sample or "target_coarse" not in sample:
        raise RetinaTrainingBatchError(
            "ISETBio HDF5 training samples require target_fine and target_coarse"
        )
    return RetinaTrainingSample(
        x_cone=sample["x_cone"],
        target_fine=sample["target_fine"],
        target_coarse=sample["target_coarse"],
        target_delta=sample.get("target_delta"),
        time_index=sample["time_index"],
    )
