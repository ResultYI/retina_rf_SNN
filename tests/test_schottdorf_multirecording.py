from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig, SchottdorfDataError
from data.schottdorf_lee_catalog import mc_pc_recordings
from data.schottdorf_lee_multirecording import (
    SchottdorfMovieDrive,
    load_schottdorf_cell,
    load_schottdorf_movie_drive,
    load_schottdorf_recording,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_run import (
    SchottdorfMultiRunConfig,
    run_schottdorf_multirecording_training,
)
from evaluation.mechanistic_retina.schottdorf_fresh_evaluation import (
    learned_parameter_values,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    require_unchanged_source,
    sha256_file,
    tensor_summary,
)
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina


_REPOSITORY = Path("data/real/schottdorf_lee_2021_repository")
_MOVIE = Path("data/real/schottdorf_lee_2021_macaque/1x10_256.mpg")
_AVAILABLE = _REPOSITORY.is_dir() and _MOVIE.is_file()


def test_tensor_summary_reports_effective_parameter_values() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])

    summary = tensor_summary(values)

    assert summary == {
        "values": [1.0, 2.0, 3.0],
        "minimum": 1.0,
        "maximum": 3.0,
        "mean": 2.0,
        "norm": pytest.approx(14.0**0.5),
    }


def test_learned_parameter_values_separates_gates_tau_and_delay() -> None:
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(cell_specific_gains=True),
        torch.tensor([[0.0, 0.0], [0.12, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        ("parasol",),
        ("ON",),
    )

    values = learned_parameter_values(model)

    assert "H1_effective_amplitude" in values
    assert "history_gate" in values
    assert "tau_H1" in values
    assert "delay_H1" in values
    assert "H1" not in values


def test_source_lineage_rejects_mid_run_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("before", encoding="utf-8")
    expected = sha256_file(source)
    source.write_text("after", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source changed during run"):
        require_unchanged_source(source, expected)


@pytest.mark.skipif(not _AVAILABLE, reason="public macaque dataset is unavailable")
def test_catalog_selects_all_public_mc_pc_recordings() -> None:
    recordings = mc_pc_recordings(_REPOSITORY / "data")

    assert len(recordings) == 37
    assert len({recording.cell_id for recording in recordings}) == 22
    assert {
        (recording.retinal_class, recording.polarity): sum(
            candidate.retinal_class == recording.retinal_class
            and candidate.polarity == recording.polarity
            for candidate in recordings
        )
        for recording in recordings
    } == {
        ("MC", "ON"): 8,
        ("MC", "OFF"): 8,
        ("PC", "ON"): 15,
        ("PC", "OFF"): 6,
    }
    assert all(recording.path.is_file() for recording in recordings)
    assert all(
        recording.canonical_cell_type
        == ("parasol" if recording.retinal_class == "MC" else "midget")
        for recording in recordings
    )
    conflicting = next(
        recording for recording in recordings if recording.recording_id == "lSS01184"
    )
    assert conflicting.catalog_recording_kind.value == "10min"
    assert conflicting.recording_kind.value == "6x1min"


@pytest.mark.skipif(not _AVAILABLE, reason="public macaque dataset is unavailable")
def test_repeated_recording_rejects_window_past_first_movie_minute() -> None:
    config = SchottdorfAdapterConfig(
        train_sequence_count=60,
        validation_sequence_count=1,
        sequence_steps=150,
        warmup_steps=30,
    )
    movie = SchottdorfMovieDrive(
        sequences=np.zeros((61, 150, 1), dtype=np.float32),
        cone_positions_degs=torch.zeros((1, 2)),
        dt_ms=1000.0 / 150.0,
        stimulus_rate_hz=150.0,
    )
    recording = next(
        item
        for item in mc_pc_recordings(_REPOSITORY / "data")
        if item.recording_id == "lSS01254"
    )

    with pytest.raises(SchottdorfDataError, match="one-minute movie"):
        load_schottdorf_recording(recording, movie, config)


@pytest.mark.skipif(not _AVAILABLE, reason="public macaque dataset is unavailable")
def test_cellwise_adapter_keeps_trials_and_temporal_split_disjoint() -> None:
    config = SchottdorfAdapterConfig(
        train_sequence_count=2,
        validation_sequence_count=1,
        sequence_steps=60,
        warmup_steps=10,
    )
    movie = load_schottdorf_movie_drive(_MOVIE, config)
    recording = next(
        item
        for item in mc_pc_recordings(_REPOSITORY / "data")
        if item.recording_id == "lSS01254"
    )

    data = load_schottdorf_recording(recording, movie, config)

    assert data.cell_ids == ("69#4",)
    assert data.recorded_cell_classes == ("MC on",)
    assert data.cell_types == ("parasol",)
    assert data.polarities == ("ON",)
    assert data.trial_count == 6
    assert data.train.cone_drive.shape == (12, 60, 289)
    assert data.validation.cone_drive.shape == (6, 60, 289)
    assert set(data.train.trial_indices) == set(range(6))
    assert set(data.validation.trial_indices) == set(range(6))
    assert set(data.train.source_image_ids).isdisjoint(data.validation.source_image_ids)
    assert data.dt_ms == pytest.approx(1000.0 / 150.0)
    assert torch.equal(data.cell_positions_degs, torch.zeros((1, 2)))
    assert not data.population_locality_constructed


@pytest.mark.skipif(not _AVAILABLE, reason="public macaque dataset is unavailable")
def test_cellwise_adapter_pools_recordings_without_merging_trial_identity() -> None:
    config = SchottdorfAdapterConfig(
        train_sequence_count=2,
        validation_sequence_count=1,
        sequence_steps=60,
        warmup_steps=10,
    )
    movie = load_schottdorf_movie_drive(_MOVIE, config)
    recordings = tuple(
        item
        for item in mc_pc_recordings(_REPOSITORY / "data")
        if item.recording_id in {"lSS01299", "lSS01300"}
    )

    data = load_schottdorf_cell(recordings, movie, config)

    assert data.recording_ids == ("lSS01299", "lSS01300")
    assert data.cell_ids == ("70#34",)
    assert data.recorded_cell_classes == ("MC on",)
    assert data.trial_count == 7
    assert data.train.cone_drive.shape == (14, 60, 289)
    assert data.validation.cone_drive.shape == (7, 60, 289)
    assert set(data.train.trial_indices) == set(range(7))
    assert set(data.validation.trial_indices) == set(range(7))
    assert set(data.train.source_image_ids).isdisjoint(data.validation.source_image_ids)
    assert all(
        source_id.startswith(("lSS01299-", "lSS01300-"))
        for source_id in data.train.source_image_ids + data.validation.source_image_ids
    )


@pytest.mark.skipif(not _AVAILABLE, reason="public macaque dataset is unavailable")
def test_multirecording_runner_uses_v1_gains_and_no_population_geometry(
    tmp_path: Path,
) -> None:
    result = run_schottdorf_multirecording_training(
        SchottdorfMultiRunConfig(
            repository_dir=_REPOSITORY,
            movie_path=_MOVIE,
            output_dir=tmp_path / "macaque-multi",
            recording_ids=("lSS01299", "lSS01300"),
            steps=1,
            batch_size=1,
            adapter=SchottdorfAdapterConfig(
                train_sequence_count=1,
                validation_sequence_count=1,
                sequence_steps=60,
                warmup_steps=10,
            ),
        )
    )
    payload = json.loads((result.artifact_dir / "results.json").read_text())
    checkpoint = torch.load(
        result.artifact_dir / "cells" / "70_34" / "model-trained.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert payload["recording_count"] == 2
    assert payload["cell_count"] == 1
    assert payload["population_locality_constructed"] is False
    assert checkpoint["model_config"]["cell_specific_gains"] is True
    assert checkpoint["model_config"]["cell_specific_pathway_mixture"] is False
    assert set(checkpoint["source_sha256"]) == {
        "1x10_256.mpg",
        "data/Cell List.docx",
        "data/CellsList.docx",
        "README.md",
        "stimuli/1x10_256.mpg",
        "lSS01299.txt",
        "lSS01300.txt",
    }
    assert checkpoint["recording_ids"] == ("lSS01299", "lSS01300")
    assert payload["cells"][0]["biological_trials"] == 7
    updates = payload["cells"][0]["training"]["major_parameter_groups_updated"]
    assert updates["aggregate_bc_ac_gains"]
    assert all(updates.values())
    assert not payload["cells"][0]["training"]["self_edge_connection_parameter_updated"]
    assert "teacher" not in json.dumps(payload).lower()
