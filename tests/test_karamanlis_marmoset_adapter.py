from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from data.karamanlis_2024 import load_minimal_marmoset_imagesequence
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from evaluation.mechanistic_retina.karamanlis_real_run import (
    KaramanlisRealRunConfig,
    run_karamanlis_real_training,
)
from training.mechanistic_retina.real_sampled import (
    RealSpikeTrainingRequest,
    fit_real_spike_model,
)


_SESSION = Path(
    "data/real/karamanlis_2024/sessions/"
    "20220301_252MEA_marmoset_left_n1"
)


@pytest.mark.skipif(not _SESSION.exists(), reason="Karamanlis session not downloaded")
def test_adapter_preserves_real_time_and_metadata() -> None:
    # Given: the downloaded 252MEA marmoset session.
    # When: its natural-image trials are adapted for the canonical core.
    data = load_minimal_marmoset_imagesequence(_SESSION)

    # Then: native timing and measured cell metadata are preserved.
    assert data.session_id == "20220301_252MEA_marmoset_left_n1"
    assert data.dt_ms == pytest.approx(1000.0 / 85.0)
    assert data.recording_sampling_rate_hz == pytest.approx(25_000.0)
    assert data.cell_types == ("parasol", "midget", "parasol", "midget")
    assert data.polarities == ("ON", "ON", "OFF", "OFF")
    assert data.cell_ids == ("14218", "11185", "116", "1806")


@pytest.mark.skipif(not _SESSION.exists(), reason="Karamanlis session not downloaded")
def test_adapter_builds_source_disjoint_real_spike_splits() -> None:
    # Given: repeated presentations of measured natural images.
    # When: deterministic train and validation splits are constructed.
    data = load_minimal_marmoset_imagesequence(_SESSION)

    # Then: image identities are disjoint and spike targets remain measured events.
    assert set(data.train.source_image_ids).isdisjoint(
        data.validation.source_image_ids
    )
    assert data.train.cone_drive.shape[1:] == (68, 289)
    assert data.train.spike_events.shape[1:] == (68, 4)
    assert data.validation.cone_drive.shape[1:] == (68, 289)
    assert data.validation.spike_events.shape[1:] == (68, 4)
    assert torch.all((data.train.spike_events == 0) | (data.train.spike_events == 1))
    assert int(data.train.spike_counts.max()) >= 2


@pytest.mark.skipif(not _SESSION.exists(), reason="Karamanlis session not downloaded")
def test_adapter_enters_canonical_model_with_native_dt() -> None:
    # Given: real marmoset inputs and the unchanged canonical topology.
    data = load_minimal_marmoset_imagesequence(_SESSION)
    config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        dt_ms=data.dt_ms,
    )
    model = build_mechanistic_retina(
        config,
        data.cone_positions_degs,
        data.cell_positions_degs,
        data.cell_types,
        data.polarities,
    )

    # When: a real trial is forwarded with its causal observed spike history.
    output = model.forward_sequence(
        data.train.cone_drive[:1],
        observed_counts=data.train.spike_events[:1],
    )

    # Then: the output is finite and aligned to the native 85 Hz sequence.
    assert output.logits.shape == (1, 68, 4)
    assert torch.isfinite(output.logits).all()
    assert model.h1.decay.item() == pytest.approx(
        torch.exp(torch.tensor(-data.dt_ms / 50.0)).item()
    )


@pytest.mark.skipif(not _SESSION.exists(), reason="Karamanlis session not downloaded")
def test_real_training_uses_only_measured_spike_events() -> None:
    # Given: a fresh canonical model and measured marmoset train/validation trials.
    data = load_minimal_marmoset_imagesequence(_SESSION)
    torch.manual_seed(202_603_01)
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

    # When: one optimizer step is fitted against real binary spike events only.
    result = fit_real_spike_model(
        RealSpikeTrainingRequest(
            model=model,
            train=data.train,
            validation=data.validation,
            steps=1,
            learning_rate=0.03,
            batch_size=1,
            seed=202_603_01,
        )
    )

    # Then: the measured-target likelihood is finite and parameters update.
    assert torch.isfinite(torch.tensor(result.validation_nll_raw))
    assert torch.isfinite(torch.tensor(result.validation_nll_trained))
    assert result.gradients_finite
    assert result.actually_updated


@pytest.mark.skipif(not _SESSION.exists(), reason="Karamanlis session not downloaded")
def test_real_training_run_writes_auditable_artifacts(tmp_path: Path) -> None:
    # Given: a one-step real-data run with a fresh output directory.
    config = KaramanlisRealRunConfig(
        session_dir=_SESSION,
        output_dir=tmp_path / "real-run",
        steps=1,
        learning_rate=0.03,
        batch_size=1,
        seed=202_603_01,
    )

    # When: the complete real-data training surface is executed.
    result = run_karamanlis_real_training(config)

    # Then: checkpoints and a teacher-free lineage report are persisted.
    payload = json.loads((result.artifact_dir / "results.json").read_text())
    assert (result.artifact_dir / "model-raw.pt").is_file()
    assert (result.artifact_dir / "model-trained.pt").is_file()
    checkpoint = torch.load(
        result.artifact_dir / "model-trained.pt",
        weights_only=True,
    )
    assert checkpoint["model_config"]["architecture_mode"] == "mechanism_identifiable"
    assert payload["training_target"] == "measured_marmoset_spike_events_only"
    assert payload["native_dt_ms"] == pytest.approx(1000.0 / 85.0)
    assert "teacher" not in json.dumps(payload).lower()
