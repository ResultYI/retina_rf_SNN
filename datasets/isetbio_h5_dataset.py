from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.retina_training_batch import (
    RetinaTrainingSample,
    collate_retina_training_batch,
    retina_training_sample_from_mapping,
)
from training.hybrid import RetinaTrainingBatch


class ISETBioH5DatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConeNormalizationStats:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True, slots=True)
class ISETBioH5DatasetConfig:
    h5_path: Path
    input_steps: int = 16
    horizons: tuple[int, ...] = (1, 2, 4)
    eps: float = 1e-6
    clip: float = 5.0
    allow_fit_stats: bool = False
    target_fine_pool: torch.Tensor | None = None
    target_coarse_pool: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.target_fine_pool is None) or (self.target_coarse_pool is None):
            raise ISETBioH5DatasetError(
                "SNN training requires target_fine_pool and target_coarse_pool"
            )


class ISETBioH5Dataset(Dataset[RetinaTrainingSample]):
    collate_fn = staticmethod(collate_retina_training_batch)

    def __init__(
        self,
        config: ISETBioH5DatasetConfig,
        stats: ConeNormalizationStats | None = None,
    ) -> None:
        from data.dataset import ISETBioDataset, ISETBioDatasetConfig

        legacy_config = ISETBioDatasetConfig(
            h5_path=config.h5_path,
            input_steps=config.input_steps,
            horizons=config.horizons,
            eps=config.eps,
            clip=config.clip,
            allow_fit_stats=config.allow_fit_stats,
            target_fine_pool=config.target_fine_pool,
            target_coarse_pool=config.target_coarse_pool,
        )
        self._dataset = ISETBioDataset(
            legacy_config,
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
    def clip_fraction(self) -> float:
        return self._dataset.clip_fraction

    @property
    def horizons(self) -> tuple[int, ...]:
        return self._dataset.horizons

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> RetinaTrainingSample:
        return retina_training_sample_from_mapping(self._dataset[index])


def collate_isetbio_h5_batch(
    samples: list[RetinaTrainingSample],
) -> RetinaTrainingBatch:
    return collate_retina_training_batch(samples)
