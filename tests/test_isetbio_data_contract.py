from pathlib import Path

import numpy as np
import pytest
import torch

import data.dataset as dataset_module
from data.geometry import local_gaussian_weights, nearest_one_to_one_weights


def test_isetbio_dataset_returns_cone_windows_and_future_deltas(monkeypatch) -> None:
    response = np.exp(np.arange(30, dtype=np.float32).reshape(6, 5) / 10.0)

    def load_cone_response(_path: Path) -> dataset_module.ConeResponseExport:
        return dataset_module.ConeResponseExport(
            response=response,
            positions_degs=np.zeros((5, 2), dtype=np.float32),
            cone_types=np.ones(5, dtype=np.uint8),
            time_axis_seconds=np.arange(6, dtype=np.float64),
            eye_trace_degs=np.zeros((6, 2), dtype=np.float32),
            units="test_units",
        )

    monkeypatch.setattr(dataset_module, "load_cone_response", load_cone_response)

    config = dataset_module.ISETBioDatasetConfig(
        h5_path=Path("fake.h5"),
        input_steps=3,
        horizons=(1, 2),
        clip=100.0,
    )
    dataset = dataset_module.ISETBioDataset(
        config,
        mean=np.zeros(5, dtype=np.float32),
        scale=np.ones(5, dtype=np.float32),
    )

    sample = dataset[0]
    assert len(dataset) == 2
    assert dataset.response_units == "test_units"
    assert sample["x_cone"].shape == (3, 5)
    assert sample["target_delta"].shape == (2, 5)
    assert sample["time_index"].item() == 2
    torch.testing.assert_close(
        sample["target_delta"][0],
        torch.from_numpy(np.log(response[3]) - np.log(response[2])),
    )


def test_dataset_requires_explicit_permission_to_fit_stats(monkeypatch) -> None:
    response = np.ones((6, 5), dtype=np.float32)

    def load_cone_response(_path: Path) -> dataset_module.ConeResponseExport:
        return dataset_module.ConeResponseExport(
            response=response,
            positions_degs=np.zeros((5, 2), dtype=np.float32),
            cone_types=np.ones(5, dtype=np.uint8),
            time_axis_seconds=np.arange(6, dtype=np.float64),
            eye_trace_degs=np.zeros((6, 2), dtype=np.float32),
            units="test_units",
        )

    monkeypatch.setattr(dataset_module, "load_cone_response", load_cone_response)
    config = dataset_module.ISETBioDatasetConfig(h5_path=Path("fake.h5"))

    with pytest.raises(ValueError, match="train-only normalization stats"):
        dataset_module.ISETBioDataset(config)

    smoke_config = dataset_module.ISETBioDatasetConfig(
        h5_path=Path("fake.h5"),
        input_steps=2,
        horizons=(1,),
        allow_fit_stats=True,
    )
    assert len(dataset_module.ISETBioDataset(smoke_config)) == 4


def test_dataset_reports_clipping_and_multiscale_targets(monkeypatch) -> None:
    response = np.exp(np.arange(30, dtype=np.float32).reshape(6, 5) / 10.0)

    def load_cone_response(_path: Path) -> dataset_module.ConeResponseExport:
        return dataset_module.ConeResponseExport(
            response=response,
            positions_degs=np.zeros((5, 2), dtype=np.float32),
            cone_types=np.ones(5, dtype=np.uint8),
            time_axis_seconds=np.arange(6, dtype=np.float64),
            eye_trace_degs=np.zeros((6, 2), dtype=np.float32),
            units="test_units",
        )

    monkeypatch.setattr(dataset_module, "load_cone_response", load_cone_response)
    fine_pool = torch.eye(5).to_sparse()
    coarse_pool = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5]],
        dtype=torch.float32,
    ).to_sparse()
    config = dataset_module.ISETBioDatasetConfig(
        h5_path=Path("fake.h5"),
        input_steps=3,
        horizons=(1,),
        clip=0.5,
        target_fine_pool=fine_pool,
        target_coarse_pool=coarse_pool,
    )

    dataset = dataset_module.ISETBioDataset(
        config,
        mean=np.zeros(5, dtype=np.float32),
        scale=np.ones(5, dtype=np.float32),
    )
    sample = dataset[0]

    assert 0.0 < dataset.clip_fraction < 1.0
    torch.testing.assert_close(sample["target_fine"], sample["target_delta"])
    torch.testing.assert_close(
        sample["target_coarse"],
        torch.sparse.mm(coarse_pool, sample["target_delta"].T).T,
    )


def test_log_cone_stats_round_trip(tmp_path) -> None:
    mean = np.asarray([1.0, 2.0], dtype=np.float32)
    scale = np.asarray([0.5, 0.25], dtype=np.float32)
    path = tmp_path / "normalization_stats.npz"

    dataset_module.save_log_cone_stats(path, mean, scale, eps=1e-5)
    loaded_mean, loaded_scale, loaded_eps = dataset_module.load_log_cone_stats(path)

    np.testing.assert_array_equal(loaded_mean, mean)
    np.testing.assert_array_equal(loaded_scale, scale)
    assert np.isclose(loaded_eps, 1e-5)


def test_local_geometry_weights_are_sparse_and_row_normalized() -> None:
    source = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    pooled = local_gaussian_weights(source, target, radius_degs=0.2, sigma_degs=0.1).to_dense()
    private = nearest_one_to_one_weights(source, target).to_dense()

    torch.testing.assert_close(pooled.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(private.sum(dim=1), torch.ones(2))
    assert pooled[0, 0] > pooled[0, 1]
    assert private[0, 0] == 1
    assert private[1, 2] == 1
