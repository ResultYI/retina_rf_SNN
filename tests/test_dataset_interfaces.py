from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.isetbio_h5_dataset import ISETBioH5Dataset, ISETBioH5DatasetConfig
from datasets.image_folder_stimulus import ImageFolderStimulusConfig, ImageFolderStimulusDataset
from datasets.raw_stimulus_dataset import DownloadSpec, RawStimulusDatasetError
from datasets.retina_training_batch import RetinaTrainingSample, collate_retina_training_batch
from datasets.rgc_response_dataset import RGCResponseDataset
from training.hybrid import RetinaTrainingBatch

try:
    import h5py
except (ImportError, ValueError):
    h5py = None


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


def test_isetbio_h5_dataset_derives_dt_ms_from_real_time_axis() -> None:
    # Given
    if h5py is None:
        pytest.skip("h5py is not available")
    path = Path("data/isetbio_h5_input_png_test/input_seed7.h5")
    if not path.exists():
        pytest.skip("real ISETBio smoke HDF5 is not present")
    with h5py.File(path, "r") as handle:
        time_axis_seconds = np.asarray(handle["time_axis_seconds"][()])
        cone_count = int(np.asarray(handle["cone_type"][()]).reshape(-1).size)
    expected_dt_ms = float(np.median(np.diff(time_axis_seconds)) * 1000.0)
    indices = torch.tensor([[0, 1], [0, 1]])
    values = torch.ones(2)
    pool = torch.sparse_coo_tensor(indices, values, (2, cone_count)).coalesce()

    # When
    dataset = ISETBioH5Dataset(
        ISETBioH5DatasetConfig(
            h5_path=path,
            input_steps=3,
            horizons=(1, 2),
            allow_fit_stats=True,
            target_fine_pool=pool,
            target_coarse_pool=pool,
        )
    )
    sample = dataset[0]

    # Then
    assert dataset.dt_ms == pytest.approx(expected_dt_ms)
    assert sample.x_cone.shape == (3, cone_count)
    assert sample.target_fine.shape == (2, 2)
    assert sample.target_coarse.shape == (2, 2)
    assert sample.time_index.item() == 2
