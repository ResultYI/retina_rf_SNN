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
    target_current: torch.Tensor
    time_index: torch.Tensor


def collate_retina_training_batch(
    samples: Sequence[RetinaTrainingSample],
) -> RetinaTrainingBatch:
    if not samples:
        raise RetinaTrainingBatchError("Cannot collate an empty batch")
    return RetinaTrainingBatch(
        x_cone=torch.stack(tuple(sample.x_cone for sample in samples), dim=0),
        targets=RetinaTargets(
            target_current=torch.stack(
                tuple(sample.target_current for sample in samples), dim=0
            ),
        ),
    )


def retina_training_sample_from_mapping(
    sample: Mapping[str, torch.Tensor],
) -> RetinaTrainingSample:
    required = {"target_current"}
    if not required <= sample.keys():
        raise RetinaTrainingBatchError(
            "ISETBio HDF5 training samples require a current target"
        )
    return RetinaTrainingSample(
        x_cone=sample["x_cone"],
        target_current=sample["target_current"],
        time_index=sample["time_index"],
    )
