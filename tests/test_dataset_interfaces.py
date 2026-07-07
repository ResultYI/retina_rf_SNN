from __future__ import annotations

from pathlib import Path

import pytest
import torch

from datasets.image_folder_stimulus import ImageFolderStimulusConfig, ImageFolderStimulusDataset
from datasets.raw_stimulus_dataset import DownloadSpec, RawStimulusDatasetError
from datasets.retina_training_batch import RetinaTrainingSample, collate_retina_training_batch
from datasets.rgc_response_dataset import RGCResponseDataset
from training.hybrid import RetinaTrainingBatch


def test_image_folder_stimulus_lists_sorted_image_paths(tmp_path: Path) -> None:
    for name in ("b.jpg", "a.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")

    dataset = ImageFolderStimulusDataset(
        ImageFolderStimulusConfig(root=tmp_path, max_items=2)
    )

    assert len(dataset) == 2
    assert [dataset[index].source_id for index in range(len(dataset))] == [
        "a.png",
        "b.jpg",
    ]


def test_missing_raw_stimulus_root_requires_explicit_download(tmp_path: Path) -> None:
    config = ImageFolderStimulusConfig(
        root=tmp_path / "missing",
        download=DownloadSpec(
            url="https://example.invalid/images.zip",
            target_path=tmp_path / "images.zip",
        ),
    )

    with pytest.raises(RawStimulusDatasetError, match="download"):
        ImageFolderStimulusDataset(config)


def test_collate_retina_training_batch_stacks_samples() -> None:
    samples = [
        RetinaTrainingSample(
            x_cone=torch.full((2, 3), float(index)),
            target_fine=torch.full((1, 3), float(index)),
            target_coarse=torch.full((1, 2), float(index)),
            target_delta=None,
            time_index=torch.tensor(index),
        )
        for index in range(2)
    ]

    batch = collate_retina_training_batch(samples)

    assert isinstance(batch, RetinaTrainingBatch)
    assert batch.x_cone.shape == (2, 2, 3)
    assert batch.targets.fine.shape == (2, 1, 3)
    assert batch.targets.coarse.shape == (2, 1, 2)


def test_rgc_response_dataset_is_analysis_only_path_index(tmp_path: Path) -> None:
    path = tmp_path / "rgc.h5"
    path.write_bytes(b"placeholder")

    dataset = RGCResponseDataset([path])

    assert len(dataset) == 1
    assert dataset[0].path == path
