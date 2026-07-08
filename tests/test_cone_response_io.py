from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import (
    apply_log_cone_stats,
    fit_log_cone_stats,
    load_cone_response,
    load_log_cone_stats,
    save_log_cone_stats,
)


def _write_text(handle: h5py.File, name: str, value: str) -> None:
    handle.create_dataset(name, data=np.frombuffer(value.encode("utf-8"), dtype=np.uint8))


def _write_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    time_steps, cone_count = 4, 5
    response = np.arange(time_steps * cone_count, dtype=np.float32).reshape(
        time_steps, cone_count
    )
    positions = np.column_stack(
        [np.linspace(-0.5, 0.5, cone_count), np.zeros(cone_count)]
    ).astype(np.float32)

    with h5py.File(path, "w") as handle:
        # MATLAB/HDF5 presents 2-D datasets to h5py with reversed axes.
        handle.create_dataset("cone_response", data=response.T)
        handle.create_dataset("cone_positions_degs", data=positions.T)
        handle.create_dataset("cone_types", data=np.arange(cone_count, dtype=np.uint8))
        handle.create_dataset("time_axis_seconds", data=np.arange(time_steps) * 0.005)
        handle.create_dataset(
            "eye_trace_degs", data=np.zeros((2, time_steps), np.float32)
        )
        handle.create_dataset(
            "response_shape_time_cone",
            data=np.asarray([time_steps, cone_count], dtype=np.int64),
        )
        _write_text(handle, "format_version", "retina-snn-cone-response-v1")
        _write_text(handle, "response_units", "isomerizations_per_integration_time")

    return response, positions


def test_load_dynamic_cone_response_contract(tmp_path) -> None:
    path = tmp_path / "sample.h5"
    response, positions = _write_sample(path)
    sample = load_cone_response(path)

    assert sample.response.shape == response.shape
    assert np.any(np.diff(sample.response, axis=0) != 0)
    np.testing.assert_array_equal(sample.positions_degs, positions)
    mean, scale = fit_log_cone_stats([path])
    normalized = apply_log_cone_stats(sample.response, mean, scale)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-6)

    stats_path = tmp_path / "normalization_stats.npz"
    save_log_cone_stats(stats_path, mean, scale)
    loaded_mean, loaded_scale, loaded_eps = load_log_cone_stats(stats_path)
    np.testing.assert_array_equal(loaded_mean, mean)
    np.testing.assert_array_equal(loaded_scale, scale)
    assert np.isclose(loaded_eps, 1e-6)


def test_load_accepts_current_isetbio_h5_contract_names(tmp_path) -> None:
    path = tmp_path / "current_contract.h5"
    response = np.arange(20, dtype=np.float32).reshape(4, 5)
    positions = np.column_stack(
        [np.linspace(-0.5, 0.5, 5), np.zeros(5)]
    ).astype(np.float32)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("cone_response_achromatic", data=response)
        handle.create_dataset("cone_xy_deg", data=positions)
        handle.create_dataset("cone_type", data=np.arange(5, dtype=np.uint8))
        handle.create_dataset("time_axis_seconds", data=np.arange(4) * 0.005)
        handle.create_dataset("eye_movement_xy_deg", data=np.zeros((4, 2), np.float32))
        handle.create_dataset(
            "response_shape_time_cone",
            data=np.asarray([4, 5], dtype=np.int64),
        )
        metadata = handle.create_group("metadata")
        _write_text(metadata, "config", '{"source":"synthetic/test-only"}')
        _write_text(handle, "format_version", "retina-snn-cone-response-v1")
        _write_text(handle, "response_units", "isomerizations_per_integration_time")

    sample = load_cone_response(path)

    np.testing.assert_array_equal(sample.response, response)
    np.testing.assert_array_equal(sample.positions_degs, positions)
    assert sample.metadata_config == '{"source":"synthetic/test-only"}'


def test_fit_stats_rejects_mismatched_mosaic_geometry(tmp_path) -> None:
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    _write_sample(first_path)
    _write_sample(second_path)
    with h5py.File(second_path, "r+") as handle:
        handle["cone_positions_degs"][0, 0] += 0.1

    with pytest.raises(ValueError, match="cone positions"):
        fit_log_cone_stats([first_path, second_path])


def test_fit_stats_rejects_mismatched_cone_type_order(tmp_path) -> None:
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    _write_sample(first_path)
    _write_sample(second_path)
    with h5py.File(second_path, "r+") as handle:
        cone_types = handle["cone_types"][:]
        handle["cone_types"][:] = cone_types[::-1]

    with pytest.raises(ValueError, match="cone type ordering"):
        fit_log_cone_stats([first_path, second_path])


@pytest.mark.parametrize(
    ("time_axis", "message"),
    [
        (np.asarray([0.0, 0.005, np.nan, 0.015]), "must be finite"),
        (np.asarray([0.0, 0.005, 0.005, 0.015]), "strictly increasing"),
        (np.asarray([0.0, 0.005, 0.011, 0.016]), "stable frame interval"),
    ],
)
def test_load_rejects_invalid_time_axis(
    tmp_path,
    time_axis: np.ndarray,
    message: str,
) -> None:
    path = tmp_path / "invalid_time.h5"
    _write_sample(path)
    with h5py.File(path, "r+") as handle:
        handle["time_axis_seconds"][:] = time_axis

    with pytest.raises(ValueError, match=message):
        load_cone_response(path)


def main() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sample.h5"
        response, positions = _write_sample(path)

        sample = load_cone_response(path)
        mean, scale = fit_log_cone_stats([path])
        normalized = apply_log_cone_stats(sample.response, mean, scale)

    np.testing.assert_array_equal(sample.response, response)
    np.testing.assert_array_equal(sample.positions_degs, positions)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-6)
    print("cone_response_io self-check passed")


if __name__ == "__main__":
    main()
