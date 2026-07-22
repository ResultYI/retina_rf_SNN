from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from datasets.retina_training_batch import (
    RetinaTrainingSample,
    collate_retina_training_batch,
    retina_training_sample_from_mapping,
)
from datasets.retina_training_batch import RetinaTrainingBatch


@dataclass(frozen=True, slots=True)
class ConeNormalizationStats:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True, slots=True)
class ISETBioH5DatasetConfig:
    h5_path: Path
    input_steps: int = 16
    eps: float = 1e-6
    clip: float = 5.0
    allow_fit_stats: bool = False


class ISETBioH5Dataset(Dataset[RetinaTrainingSample]):
    collate_fn = staticmethod(collate_retina_training_batch)

    def __init__(
        self,
        config: ISETBioH5DatasetConfig,
        stats: ConeNormalizationStats | None = None,
    ) -> None:
        from data.dataset import ISETBioDataset, ISETBioDatasetConfig

        dataset_config = ISETBioDatasetConfig(
            h5_path=config.h5_path,
            input_steps=config.input_steps,
            eps=config.eps,
            clip=config.clip,
            allow_fit_stats=config.allow_fit_stats,
        )
        self._dataset = ISETBioDataset(
            dataset_config,
            None if stats is None else stats.mean,
            None if stats is None else stats.scale,
        )

    @property
    def positions_degs(self) -> np.ndarray:
        return self._dataset.positions_degs

    @property
    def cone_types(self) -> np.ndarray:
        return self._dataset.cone_types

    @property
    def time_axis_seconds(self) -> np.ndarray:
        return self._dataset.time_axis_seconds

    @property
    def dt_ms(self) -> float:
        return self._dataset.dt_ms

    @property
    def eccentricity_deg(self) -> float:
        return self._dataset.eccentricity_deg

    @property
    def clip_fraction(self) -> float:
        return self._dataset.clip_fraction

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> RetinaTrainingSample:
        return retina_training_sample_from_mapping(self._dataset[index])


def collate_isetbio_h5_batch(
    samples: list[RetinaTrainingSample],
) -> RetinaTrainingBatch:
    return collate_retina_training_batch(samples)
