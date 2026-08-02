from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from benchmarks.point_process_teacher import generate_teacher_responses
from data.input_identity import synthetic_input_identity
from data.rgc_response import (
    CellMetadata,
    RGCResponseContractError,
    RGCResponseSession,
    ResponseTargetKind,
    load_rgc_response,
    validate_response_splits,
)
from data.rgc_response_export import write_rgc_response
from data.synthetic_teacher import fit_teacher_input_normalization, load_teacher_rf_metadata
from training.response_config import ResponseDataConfig
from training.response_data import _fingerprint, prepare_response_data


def test_export_persists_teacher_identity_metadata(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    normalization = fit_teacher_input_normalization(cones)
    result = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        np.arange(80) * 0.005,
        trials=1,
        seed=3,
        adaptive=False,
        teacher_normalization=normalization,
    )

    write_rgc_response(
        tmp_path / "synthetic.h5",
        result.session,
        teacher_kernels=result.kernels,
        teacher_normalization=result.teacher_normalization,
    )

    with h5py.File(tmp_path / "synthetic.h5", "r") as handle:
        assert "teacher/input_mean" in handle
        assert "teacher/input_std" in handle
        assert tuple(
            value.decode("utf-8") for value in handle["teacher/component_id"][()]
        ) == (
            "population",
            "type",
            "polarity",
            "cell_residual",
        )
    metadata = load_teacher_rf_metadata(tmp_path / "synthetic.h5")
    assert metadata is not None
    assert metadata.revision == "hierarchical-synthetic-teacher-v1"
    assert metadata.generation_seed == 3
    assert metadata.residual_seed == 3
    assert len(metadata.cell_group_ids) == 16


def test_response_splits_reject_stale_teacher_seed_metadata(tmp_path: Path) -> None:
    train = _write_teacher_response(tmp_path / "train.h5", ("train-a", "train-b"))
    validation = _write_teacher_response(
        tmp_path / "validation.h5",
        ("validation-a", "validation-b"),
    )
    test = _write_teacher_response(tmp_path / "test.h5", ("test-a", "test-b"))
    with h5py.File(validation, "r+") as handle:
        handle["teacher/generation_seed"][0] = 99

    with pytest.raises(RGCResponseContractError, match="teacher identity"):
        validate_response_splits(
            (load_rgc_response(train),),
            (load_rgc_response(validation),),
            (load_rgc_response(test),),
        )
    with pytest.raises(RGCResponseContractError, match="teacher identity"):
        prepare_response_data(
            ResponseDataConfig(
                str(train),
                str(validation),
                str(test),
                80,
            )
        )


def test_load_rejects_wrong_length_teacher_identity_vector(tmp_path: Path) -> None:
    path = _write_teacher_response(tmp_path / "synthetic.h5", ("train-a", "train-b"))
    with h5py.File(path, "r+") as handle:
        del handle["teacher/context_gain_cell_residual"]
        handle["teacher"].create_dataset(
            "context_gain_cell_residual",
            data=np.zeros(15, dtype=np.float32),
        )

    with pytest.raises(RGCResponseContractError, match="teacher identity"):
        load_rgc_response(path)


def test_fingerprint_frames_adjacent_arrays() -> None:
    session = _fingerprint_session()
    splits = ((session,), (), ())

    first = _fingerprint(
        splits,
        1,
        np.asarray([1, 2], dtype=np.uint8),
        np.asarray([], dtype=np.uint8),
    )
    second = _fingerprint(
        splits,
        1,
        np.asarray([1], dtype=np.uint8),
        np.asarray([2], dtype=np.uint8),
    )

    assert first != second


def _write_teacher_response(path: Path, source_ids: tuple[str, str]) -> Path:
    rng = np.random.default_rng(31)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    normalization = fit_teacher_input_normalization(cones)
    result = generate_teacher_responses(
        cones,
        positions,
        source_ids,
        np.arange(80) * 0.005,
        trials=1,
        seed=3,
        adaptive=True,
        teacher_normalization=normalization,
    )
    write_rgc_response(
        path,
        result.session,
        teacher_kernels=result.kernels,
        teacher_normalization=normalization,
    )
    return path


def _fingerprint_session() -> RGCResponseSession:
    return RGCResponseSession(
        cone_response=np.zeros((1, 1, 1), dtype=np.float32),
        spike_counts=np.zeros((1, 1, 1, 1), dtype=np.float32),
        valid_mask=np.ones((1, 1, 1, 1), dtype=bool),
        time_axis_seconds=np.asarray([0.0], dtype=np.float64),
        cone_positions_degs=np.zeros((1, 2), dtype=np.float32),
        cells=CellMetadata(
            ids=("cell-a",),
            type_ids=("midget",),
            polarities=np.asarray([0], dtype=np.int64),
            positions_degs=np.zeros((1, 2), dtype=np.float32),
            eccentricities_deg=np.asarray([4.0], dtype=np.float32),
        ),
        source_ids=("source-a",),
        context_ids=("low",),
        target_kind=ResponseTargetKind.BERNOULLI,
        path=Path("<memory>"),
        input_identity=synthetic_input_identity(1, ("source-fingerprint",)),
    )
