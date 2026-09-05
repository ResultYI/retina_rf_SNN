from __future__ import annotations

from dataclasses import asdict
import json
import math

import torch

from data.schottdorf_lee_catalog import SchottdorfRecording
from data.schottdorf_lee_multirecording import (
    SchottdorfMovieDrive,
    load_schottdorf_cell,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    major_parameter_group_updates,
    require_unchanged_source,
    sha256_file,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_types import (
    CellFitRecord,
    RecordingMetadata,
    SchottdorfMultiRunConfig,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.real_sampled import (
    RealSpikeTrainingRequest,
    fit_real_spike_model,
)


def fit_schottdorf_cell(
    config: SchottdorfMultiRunConfig,
    movie: SchottdorfMovieDrive,
    recordings: tuple[SchottdorfRecording, ...],
    index: int,
    *,
    movie_sha256: str,
    catalog_source_sha256: dict[str, str],
) -> CellFitRecord:
    spike_sha256 = {item.path.name: sha256_file(item.path) for item in recordings}
    data = load_schottdorf_cell(recordings, movie, config.adapter)
    for recording in recordings:
        require_unchanged_source(recording.path, spike_sha256[recording.path.name])
    model_config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        cell_specific_gains=True,
        dt_ms=data.dt_ms,
    )
    seed = config.seed + index
    torch.manual_seed(seed)
    model = build_mechanistic_retina(
        model_config,
        data.cone_positions_degs,
        data.cell_positions_degs,
        data.cell_types,
        data.polarities,
    )
    artifact_dir = config.output_dir / "cells" / data.cell_ids[0].replace("#", "_")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    recording_metadata = _recording_metadata(recordings, spike_sha256)
    checkpoint = {
        "schema": "schottdorf_lee_2021_macaque_cellwise_canonical_v1",
        "revision": MECHANISTIC_MODEL_REVISION,
        "recording_ids": data.recording_ids,
        "cell_id": data.cell_ids[0],
        "recorded_cell_classes": data.recorded_cell_classes,
        "retinal_class": data.retinal_classes[0],
        "canonical_cell_type": data.cell_types[0],
        "polarity": data.polarities[0],
        "recording_metadata": recording_metadata,
        "input_representation": data.input_representation,
        "model_config": asdict(model_config)
        | {"architecture_mode": model_config.architecture_mode.value},
        "adapter_config": asdict(config.adapter),
        "cell_positions_degs": data.cell_positions_degs,
        "cone_positions_degs": data.cone_positions_degs,
        "population_locality_constructed": False,
        "seed": seed,
        "source_sha256": {config.movie_path.name: movie_sha256}
        | catalog_source_sha256
        | spike_sha256,
    }
    torch.save(
        checkpoint | {"stage": "raw", "model": model.state_dict()},
        artifact_dir / "model-raw.pt",
    )
    training = fit_real_spike_model(
        RealSpikeTrainingRequest(
            model=model,
            train=data.train,
            validation=data.validation,
            steps=config.steps,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            seed=seed,
        )
    )
    torch.save(
        checkpoint | {"stage": "trained", "model": model.state_dict()},
        artifact_dir / "model-trained.pt",
    )
    result: CellFitRecord = {
        "recording_ids": data.recording_ids,
        "recording_kinds": data.recording_kinds,
        "recording_count": len(recordings),
        "recording_metadata": recording_metadata,
        "input_representation": data.input_representation,
        "cell_id": data.cell_ids[0],
        "recorded_cell_classes": data.recorded_cell_classes,
        "retinal_class": data.retinal_classes[0],
        "canonical_cell_type": data.cell_types[0],
        "polarity": data.polarities[0],
        "eccentricity_deg": data.eccentricity_deg,
        "native_dt_ms": data.dt_ms,
        "stimulus_rate_hz": data.stimulus_rate_hz,
        "spike_time_resolution_ms": data.spike_time_resolution_ms,
        "biological_trials": data.trial_count,
        "train_sequences": data.train.cone_drive.shape[0],
        "validation_sequences": data.validation.cone_drive.shape[0],
        "train_valid_bins": int(data.train.valid_mask.sum()),
        "validation_valid_bins": int(data.validation.valid_mask.sum()),
        "train_event_rate": _valid_event_rate(
            data.train.spike_events, data.train.valid_mask
        ),
        "validation_event_rate": training.observed_event_rate,
        "train_multi_spike_bin_fraction": _valid_multi_spike_fraction(
            data.train.spike_counts, data.train.valid_mask
        ),
        "validation_multi_spike_bin_fraction": _valid_multi_spike_fraction(
            data.validation.spike_counts, data.validation.valid_mask
        ),
        "time_segment_disjoint": set(data.train.source_image_ids).isdisjoint(
            data.validation.source_image_ids
        ),
        "validation_nll_raw": training.validation_nll_raw,
        "validation_nll_trained": training.validation_nll_trained,
        "nll_improvement": training.validation_nll_raw
        - training.validation_nll_trained,
        "prediction_improved": bool(
            math.isfinite(training.validation_nll_trained)
            and training.validation_nll_trained < training.validation_nll_raw
        ),
        "training": {
            "seed": seed,
            "gradients_finite": training.gradients_finite,
            "actually_updated": training.actually_updated,
            "major_parameter_groups_updated": major_parameter_group_updates(
                training.actually_updated
            ),
            "self_edge_connection_parameter_updated": (
                "shared_subunits.raw_connections" in training.actually_updated
            ),
        },
        "source_sha256": spike_sha256,
    }
    (artifact_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _recording_metadata(
    recordings: tuple[SchottdorfRecording, ...],
    spike_sha256: dict[str, str],
) -> list[RecordingMetadata]:
    return [
        RecordingMetadata(
            recording_id=recording.recording_id,
            recorded_cell_class=recording.recorded_cell_class,
            retinal_class=recording.retinal_class,
            canonical_cell_type=recording.canonical_cell_type,
            polarity=recording.polarity,
            recording_kind=recording.recording_kind.value,
            catalog_recording_kind=recording.catalog_recording_kind.value,
            eccentricity_deg=recording.eccentricity_deg,
            spike_sha256=spike_sha256[recording.path.name],
        )
        for recording in recordings
    ]


def _valid_event_rate(events: torch.Tensor, valid_mask: torch.Tensor) -> float:
    mask = valid_mask.to(dtype=events.dtype)
    return float((events * mask).sum() / mask.sum().clamp_min(1))


def _valid_multi_spike_fraction(
    counts: torch.Tensor, valid_mask: torch.Tensor
) -> float:
    mask = valid_mask.to(dtype=counts.dtype)
    return float(((counts > 1).to(counts.dtype) * mask).sum() / mask.sum().clamp_min(1))


__all__ = ["fit_schottdorf_cell"]
