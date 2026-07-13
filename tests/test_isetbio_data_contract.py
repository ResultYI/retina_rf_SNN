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
            eccentricity_deg=2.0,
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
            eccentricity_deg=2.0,
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
            eccentricity_deg=2.0,
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


def test_compatible_exports_require_shared_mosaic_and_frame_interval(monkeypatch) -> None:
    base = dataset_module.ConeResponseExport(
        response=np.ones((4, 2), dtype=np.float32),
        positions_degs=np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
        cone_types=np.asarray([1, 2], dtype=np.uint8),
        time_axis_seconds=np.asarray([0.0, 0.005, 0.010, 0.015]),
        eye_trace_degs=np.zeros((4, 2), dtype=np.float32),
        units="test_units",
        eccentricity_deg=2.0,
    )
    mismatched_dt = dataset_module.ConeResponseExport(
        response=base.response,
        positions_degs=base.positions_degs,
        cone_types=base.cone_types,
        time_axis_seconds=np.asarray([0.0, 0.010, 0.020, 0.030]),
        eye_trace_degs=base.eye_trace_degs,
        units=base.units,
        eccentricity_deg=base.eccentricity_deg,
    )

    monkeypatch.setattr(
        dataset_module,
        "load_cone_response",
        lambda path: base if path.name == "first.h5" else mismatched_dt,
    )

    with pytest.raises(ValueError, match="temporal sampling"):
        dataset_module.validate_compatible_cone_exports(
            (Path("first.h5"), Path("second.h5"))
        )


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


def test_target_pool_accepts_float32_accumulation_error_for_large_rows() -> None:
    count = 4401
    indices = torch.arange(count)
    pool = torch.sparse_coo_tensor(
        torch.stack((torch.zeros(count, dtype=torch.long), indices)),
        torch.full((count,), 1 / count),
        (1, count),
    ).coalesce()

    validated = dataset_module._validate_target_pool("pool", pool, count)

    assert validated is not None
