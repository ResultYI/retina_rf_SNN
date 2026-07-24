from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from data.rgc_response import (
    RGCResponseContractError,
    ResponseTargetKind,
    load_rgc_response,
    validate_response_splits,
)
from training.response_config import ResponseDataConfig
from training.response_data import prepare_response_data


def _text(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)


def _write_response(
    path: Path,
    *,
    source: str,
    kind: str = "bernoulli",
    cone_value: float = 1.0,
    teacher_mean: np.ndarray | None = None,
    teacher_std: np.ndarray | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("format_version", data=_text("retina-rgc-response-v1"))
        handle.attrs["response_target_kind"] = kind
        handle.create_dataset(
            "cone_response",
            data=np.full((2, 6, 3), cone_value, dtype=np.float32),
        )
        spikes = np.zeros((2, 2, 6, 2), dtype=np.float32)
        spikes[:, :, 2, 0] = 1
        handle.create_dataset("spike_counts", data=spikes)
        handle.create_dataset("valid_mask", data=np.ones_like(spikes, dtype=np.uint8))
        handle.create_dataset("time_axis_seconds", data=np.arange(6) * 0.005)
        handle.create_dataset("cone/position_degs", data=np.zeros((3, 2)))
        handle.create_dataset("cell/id", data=np.asarray([b"c0", b"c1"]))
        handle.create_dataset("cell/type_id", data=np.asarray([b"midget", b"parasol"]))
        handle.create_dataset("cell/polarity", data=np.asarray([0, 1], dtype=np.uint8))
        handle.create_dataset("cell/position_degs", data=np.zeros((2, 2)))
        handle.create_dataset("cell/eccentricity_deg", data=np.asarray([4.0, 4.0]))
        handle.create_dataset(
            "stimulus/source_id",
            data=np.asarray([f"{source}-0".encode(), f"{source}-1".encode()]),
        )
        handle.create_dataset(
            "stimulus/context_id",
            data=np.asarray([b"low", b"high"]),
        )
        if teacher_mean is not None or teacher_std is not None:
            group = handle.create_group("teacher")
            if teacher_mean is not None:
                group.create_dataset("input_mean", data=teacher_mean)
            if teacher_std is not None:
                group.create_dataset("input_std", data=teacher_std)


def test_loads_strict_response_contract(tmp_path: Path) -> None:
    path = tmp_path / "response.h5"
    _write_response(path, source="train")

    session = load_rgc_response(path)

    assert session.target_kind is ResponseTargetKind.BERNOULLI
    assert session.cone_response.shape == (2, 6, 3)
    assert session.spike_counts.shape == (2, 2, 6, 2)
    assert session.cells.polarities.tolist() == [0, 1]
    assert session.source_ids == ("train-0", "train-1")


def test_rejects_invalid_bernoulli_targets(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    _write_response(path, source="train")
    with h5py.File(path, "r+") as handle:
        handle["spike_counts"][0, 0, 0, 0] = 2

    with pytest.raises(RGCResponseContractError, match="binary"):
        load_rgc_response(path)


def test_rejects_source_leakage(tmp_path: Path) -> None:
    train_path = tmp_path / "train.h5"
    validation_path = tmp_path / "validation.h5"
    _write_response(train_path, source="shared")
    _write_response(validation_path, source="shared")

    with pytest.raises(RGCResponseContractError, match="source-disjoint"):
        validate_response_splits(
            (load_rgc_response(train_path),),
            (load_rgc_response(validation_path),),
        )


def test_canonical_training_rejects_poisson_targets(tmp_path: Path) -> None:
    _write_split_files(tmp_path, kind="poisson")

    with pytest.raises(RGCResponseContractError, match="Bernoulli"):
        prepare_response_data(_data_config(tmp_path))


def test_dataset_fingerprint_includes_response_content(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_split_files(first)
    _write_split_files(second)
    with h5py.File(second / "test.h5", "r+") as handle:
        handle["spike_counts"][0, 0, 3, 1] = 1

    first_data = prepare_response_data(_data_config(first))
    second_data = prepare_response_data(_data_config(second))

    assert first_data.fingerprint != second_data.fingerprint


def test_prepare_response_data_keeps_real_train_fitted_normalization(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.h5"
    validation_path = tmp_path / "validation.h5"
    test_path = tmp_path / "test.h5"
    _write_response(train_path, source="train", cone_value=2.0)
    _write_response(validation_path, source="validation", cone_value=7.0)
    _write_response(test_path, source="test", cone_value=11.0)

    data = prepare_response_data(_data_config(tmp_path))

    assert np.array_equal(data.normalization_mean, np.full(3, 2.0, dtype=np.float32))
    assert np.array_equal(data.normalization_std, np.full(3, 1e-6, dtype=np.float32))


def test_prepare_response_data_uses_synthetic_teacher_normalization(
    tmp_path: Path,
) -> None:
    teacher_mean = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    teacher_std = np.asarray([2.0, 4.0, 8.0], dtype=np.float32)
    for split in ("train", "validation", "test"):
        _write_response(
            tmp_path / f"{split}.h5",
            source=split,
            cone_value=1.0,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
        )

    data = prepare_response_data(_data_config(tmp_path))

    assert np.array_equal(data.normalization_mean, teacher_mean)
    assert np.array_equal(data.normalization_std, teacher_std)
    assert np.allclose(data.train.cone_response.numpy(), (1.0 - teacher_mean) / teacher_std)


def test_prepare_response_data_rejects_mismatched_synthetic_normalization(
    tmp_path: Path,
) -> None:
    teacher_std = np.ones(3, dtype=np.float32)
    _write_response(
        tmp_path / "train.h5",
        source="train",
        teacher_mean=np.zeros(3, dtype=np.float32),
        teacher_std=teacher_std,
    )
    _write_response(
        tmp_path / "validation.h5",
        source="validation",
        teacher_mean=np.ones(3, dtype=np.float32),
        teacher_std=teacher_std,
    )
    _write_response(
        tmp_path / "test.h5",
        source="test",
        teacher_mean=np.zeros(3, dtype=np.float32),
        teacher_std=teacher_std,
    )

    with pytest.raises(RGCResponseContractError, match="teacher normalization"):
        prepare_response_data(_data_config(tmp_path))


def test_prepare_response_data_rejects_malformed_synthetic_normalization(
    tmp_path: Path,
) -> None:
    _write_response(
        tmp_path / "train.h5",
        source="train",
        teacher_mean=np.zeros(3, dtype=np.float32),
    )
    _write_response(tmp_path / "validation.h5", source="validation")
    _write_response(tmp_path / "test.h5", source="test")

    with pytest.raises(RGCResponseContractError, match="teacher normalization"):
        prepare_response_data(_data_config(tmp_path))


def _write_split_files(root: Path, *, kind: str = "bernoulli") -> None:
    root.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        _write_response(root / f"{split}.h5", source=split, kind=kind)


def _data_config(root: Path) -> ResponseDataConfig:
    return ResponseDataConfig(
        train_glob=str(root / "train.h5"),
        validation_glob=str(root / "validation.h5"),
        test_glob=str(root / "test.h5"),
        sequence_steps=6,
    )
