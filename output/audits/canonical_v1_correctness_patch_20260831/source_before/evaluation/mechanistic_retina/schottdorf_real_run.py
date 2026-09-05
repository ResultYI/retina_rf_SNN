from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from data.schottdorf_lee_2021 import (
    SchottdorfAdapterConfig,
    load_minimal_macaque_natural_movie,
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


@dataclass(frozen=True, slots=True)
class SchottdorfRealRunConfig:
    recording_dir: Path
    output_dir: Path
    steps: int = 50
    learning_rate: float = 0.03
    batch_size: int = 4
    seed: int = 202_608_27
    adapter: SchottdorfAdapterConfig = SchottdorfAdapterConfig()


@dataclass(frozen=True, slots=True)
class SchottdorfRealRunResult:
    artifact_dir: Path
    validation_nll_raw: float
    validation_nll_trained: float


def run_schottdorf_real_training(
    run_config: SchottdorfRealRunConfig,
) -> SchottdorfRealRunResult:
    if run_config.output_dir.exists() and any(run_config.output_dir.iterdir()):
        raise FileExistsError("real-data output directory must be empty")
    data = load_minimal_macaque_natural_movie(
        run_config.recording_dir, run_config.adapter
    )
    model_config = MechanisticRetinaConfig(
        architecture_mode=ArchitectureMode.MECHANISM_IDENTIFIABLE,
        dt_ms=data.dt_ms,
    )
    torch.manual_seed(run_config.seed)
    model = build_mechanistic_retina(
        model_config,
        data.cone_positions_degs,
        data.cell_positions_degs,
        data.cell_types,
        data.polarities,
    )
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_base = {
        "schema": "schottdorf_lee_2021_macaque_canonical_retina",
        "revision": MECHANISTIC_MODEL_REVISION,
        "recording_id": data.recording_id,
        "model_config": asdict(model_config)
        | {"architecture_mode": model_config.architecture_mode.value},
        "adapter_config": asdict(run_config.adapter),
        "cell_ids": data.cell_ids,
        "recorded_cell_classes": data.recorded_cell_classes,
        "cell_types": data.cell_types,
        "polarities": data.polarities,
        "cell_positions_degs": data.cell_positions_degs,
        "cone_positions_degs": data.cone_positions_degs,
    }
    torch.save(
        checkpoint_base | {"stage": "raw", "model": model.state_dict()},
        run_config.output_dir / "model-raw.pt",
    )
    training = fit_real_spike_model(
        RealSpikeTrainingRequest(
            model=model,
            train=data.train,
            validation=data.validation,
            steps=run_config.steps,
            learning_rate=run_config.learning_rate,
            batch_size=run_config.batch_size,
            seed=run_config.seed,
        )
    )
    torch.save(
        checkpoint_base | {"stage": "trained", "model": model.state_dict()},
        run_config.output_dir / "model-trained.pt",
    )
    updated_groups = _major_parameter_group_updates(training.actually_updated)
    movie_path = run_config.recording_dir / "1x10_256.mpg"
    spike_path = run_config.recording_dir / "lSS01300.txt"
    result_payload = {
        "model_class": "physiology-constrained recurrent/state-space retinal point-process model",
        "model_revision": MECHANISTIC_MODEL_REVISION,
        "causal_contract": model.config.causal_contract,
        "topology": "Cone->H1->shared BC encoder; narrow BC->RGC; broad BC->AC->RGC; RGC->spike likelihood",
        "dataset": "Schottdorf and Lee 2021, G-Node 10.12751/g-node.xage77",
        "species": "Macaca fascicularis",
        "recording_id": data.recording_id,
        "training_target": "measured_macaque_spike_events_only",
        "fresh_initialization": True,
        "fresh_optimizer": True,
        "native_dt_ms": data.dt_ms,
        "stimulus_rate_hz": data.stimulus_rate_hz,
        "spike_time_resolution_ms": data.spike_time_resolution_ms,
        "input_representation": data.input_representation,
        "cells": [
            {
                "id": data.cell_ids[0],
                "recorded_class": data.recorded_cell_classes[0],
                "canonical_type": data.cell_types[0],
                "polarity": data.polarities[0],
            }
        ],
        "stimulus": {
            "kind": "natural color movie with recorded eye-like motion",
            "native_size_pixels": [256, 256],
            "native_live_rate_hz": data.stimulus_rate_hz,
            "used_live_frames": (
                run_config.adapter.train_sequence_count
                + run_config.adapter.validation_sequence_count
            )
            * run_config.adapter.sequence_steps,
            "used_duration_s": (
                run_config.adapter.train_sequence_count
                + run_config.adapter.validation_sequence_count
            )
            * run_config.adapter.sequence_steps
            / data.stimulus_rate_hz,
        },
        "splits": {
            "biological_trials": data.trial_count,
            "train_sequences": data.train.cone_drive.shape[0],
            "validation_sequences": data.validation.cone_drive.shape[0],
            "sequence_steps": data.train.cone_drive.shape[1],
            "train_valid_bins": int(data.train.valid_mask.sum()),
            "validation_valid_bins": int(data.validation.valid_mask.sum()),
            "time_segment_disjoint": set(data.train.source_image_ids).isdisjoint(
                data.validation.source_image_ids
            ),
        },
        "temporal_definitions": {
            "tau": "bounded learnable physiological time constant in ms",
            "explicit_pathway_delay": "bounded learnable fractional delay in ms",
            "rf_lag_window": {
                "lag_steps": model_config.lag_steps,
                "window_ms": model_config.lag_steps * data.dt_ms,
            },
            "rgc_history_shift": {
                "steps": 1,
                "shift_ms": data.dt_ms,
            },
        },
        "spike_encoding": {
            "likelihood": "Bernoulli event per native stimulus frame",
            "observed_validation_event_rate": training.observed_event_rate,
            "train_multi_spike_bin_fraction": float(
                (data.train.spike_counts > 1).float().mean()
            ),
        },
        "validation_prediction": {
            "nll_raw": training.validation_nll_raw,
            "nll_trained": training.validation_nll_trained,
            "mean_probability_raw": training.mean_probability_raw,
            "mean_probability_trained": training.mean_probability_trained,
            "observed_event_rate": training.observed_event_rate,
        },
        "training": {
            "steps": run_config.steps,
            "learning_rate": run_config.learning_rate,
            "batch_size": run_config.batch_size,
            "seed": run_config.seed,
            "gradients_finite": training.gradients_finite,
            "actually_updated": training.actually_updated,
            "major_parameter_groups_updated": updated_groups,
        },
        "source_sha256": {
            movie_path.name: _sha256_file(movie_path),
            spike_path.name: _sha256_file(spike_path),
        },
    }
    (run_config.output_dir / "results.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return SchottdorfRealRunResult(
        artifact_dir=run_config.output_dir,
        validation_nll_raw=training.validation_nll_raw,
        validation_nll_trained=training.validation_nll_trained,
    )


def _major_parameter_group_updates(updated: tuple[str, ...]) -> dict[str, bool]:
    updated_set = set(updated)
    expected = {
        "weights": ("bipolar.raw_weights",),
        "gates": (
            "gates.raw_h1_amplitude",
            "gates.ac_local",
            "gates.ac_transient",
            "gates.history",
        ),
        "bounded_tau": (
            "h1.raw_tau",
            "feature_bank.raw_tau",
            "amacrine.raw_tau",
        ),
        "bounded_explicit_delay": (
            "h1.raw_delay", "feature_bank.raw_delay", "amacrine.raw_delay"
        ),
        "rgc_bias": ("rgc.response_bias",),
    }
    return {
        name: all(parameter in updated_set for parameter in parameters)
        for name, parameters in expected.items()
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "SchottdorfRealRunConfig",
    "SchottdorfRealRunResult",
    "run_schottdorf_real_training",
]
