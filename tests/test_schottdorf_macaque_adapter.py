from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from data.schottdorf_lee_2021 import (
    SchottdorfAdapterConfig,
    load_minimal_macaque_natural_movie,
    parse_spike_time_table,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from evaluation.mechanistic_retina.schottdorf_real_run import (
    SchottdorfRealRunConfig,
    run_schottdorf_real_training,
)


_RECORDING = Path("data/real/schottdorf_lee_2021_macaque")


@pytest.mark.skipif(not _RECORDING.exists(), reason="macaque recording not downloaded")
def test_spike_table_preserves_recorded_timing_resolution() -> None:
    # Given: the published lSS01300 spike-time table.
    # When: its live-video spike times are parsed without smoothing.
    spikes = parse_spike_time_table(_RECORDING / "lSS01300.txt")

    # Then: identity and 0.1 ms acquisition timing remain explicit.
    assert spikes.total_spikes == 24_001
    assert spikes.resolution_ms == pytest.approx(0.1)
    assert spikes.times_ms.ndim == 1
    assert torch.all(spikes.times_ms >= 0)
    assert torch.all(spikes.times_ms < 600_000.0)


@pytest.mark.skipif(not _RECORDING.exists(), reason="macaque recording not downloaded")
def test_adapter_preserves_native_movie_dt_and_mc_identity() -> None:
    # Given: a bounded prefix of the published natural movie and MC ON recording.
    # When: it is adapted to three short canonical-model sequences.
    data = load_minimal_macaque_natural_movie(
        _RECORDING,
        SchottdorfAdapterConfig(
            train_sequence_count=2,
            validation_sequence_count=1,
            sequence_steps=60,
            warmup_steps=10,
        ),
    )

    # Then: experimental timing and the PC/MC cell annotation are not inferred.
    assert data.recording_id == "lSS01300"
    assert data.cell_ids == ("70#34",)
    assert data.recorded_cell_classes == ("MC on",)
    assert data.cell_types == ("parasol",)
    assert data.polarities == ("ON",)
    assert data.spike_time_resolution_ms == pytest.approx(0.1)
    assert data.stimulus_rate_hz == pytest.approx(150.0)
    assert data.dt_ms == pytest.approx(1000.0 / 150.0)


@pytest.mark.skipif(not _RECORDING.exists(), reason="macaque recording not downloaded")
def test_adapter_builds_time_disjoint_real_spike_splits() -> None:
    # Given: one real biological trial with measured stimulus and spike timing.
    # When: contiguous, non-overlapping sequence windows are constructed.
    data = load_minimal_macaque_natural_movie(
        _RECORDING,
        SchottdorfAdapterConfig(
            train_sequence_count=2,
            validation_sequence_count=1,
            sequence_steps=60,
            warmup_steps=10,
        ),
    )

    # Then: the model receives calibrated cone drive and measured spike events only.
    assert data.trial_count == 1
    assert data.train.cone_drive.shape == (2, 60, 289)
    assert data.train.spike_events.shape == (2, 60, 1)
    assert data.validation.cone_drive.shape == (1, 60, 289)
    assert data.validation.spike_events.shape == (1, 60, 1)
    assert set(data.train.source_image_ids).isdisjoint(
        data.validation.source_image_ids
    )
    assert torch.all((data.train.spike_events == 0) | (data.train.spike_events == 1))
    assert not data.train.valid_mask[:, :10].any()
    assert data.train.valid_mask[:, 10:].all()


@pytest.mark.skipif(not _RECORDING.exists(), reason="macaque recording not downloaded")
def test_real_macaque_sequence_enters_canonical_model() -> None:
    # Given: one adapted MC ON natural-movie sequence at the native 150 Hz dt.
    data = load_minimal_macaque_natural_movie(
        _RECORDING,
        SchottdorfAdapterConfig(
            train_sequence_count=1,
            validation_sequence_count=1,
            sequence_steps=60,
            warmup_steps=10,
        ),
    )
    model = build_mechanistic_retina(
        MechanisticRetinaConfig(
            architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
            dt_ms=data.dt_ms,
        ),
        data.cone_positions_degs,
        data.cell_positions_degs,
        data.cell_types,
        data.polarities,
    )

    # When: the measured visual drive and causal real spike history are forwarded.
    output = model.forward_sequence(
        data.train.cone_drive,
        observed_counts=data.train.spike_events,
    )

    # Then: the unchanged canonical model produces finite aligned logits.
    assert output.logits.shape == (1, 60, 1)
    assert torch.isfinite(output.logits).all()


@pytest.mark.skipif(not _RECORDING.exists(), reason="macaque recording not downloaded")
def test_real_macaque_run_writes_teacher_free_fresh_artifacts(tmp_path: Path) -> None:
    # Given: one minimal fresh run on measured macaque stimulus and spikes.
    config = SchottdorfRealRunConfig(
        recording_dir=_RECORDING,
        output_dir=tmp_path / "macaque-run",
        steps=1,
        learning_rate=0.03,
        batch_size=1,
        seed=202_608_27,
        adapter=SchottdorfAdapterConfig(
            train_sequence_count=1,
            validation_sequence_count=1,
            sequence_steps=60,
            warmup_steps=10,
        ),
    )

    # When: the full real-data fitting surface is executed once.
    result = run_schottdorf_real_training(config)
    payload = json.loads((result.artifact_dir / "results.json").read_text())

    # Then: raw/trained checkpoints and real-only lineage are auditable.
    assert (result.artifact_dir / "model-raw.pt").is_file()
    assert (result.artifact_dir / "model-trained.pt").is_file()
    assert payload["training_target"] == "measured_macaque_spike_events_only"
    assert payload["fresh_initialization"]
    assert payload["fresh_optimizer"]
    assert payload["native_dt_ms"] == pytest.approx(1000.0 / 150.0)
    assert "teacher" not in json.dumps(payload).lower()
    assert all(payload["training"]["major_parameter_groups_updated"].values())
