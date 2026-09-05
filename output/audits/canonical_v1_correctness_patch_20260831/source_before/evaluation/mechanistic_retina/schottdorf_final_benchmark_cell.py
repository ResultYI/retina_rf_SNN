from __future__ import annotations

from dataclasses import asdict

import torch

from baselines.local_point_process_glm import LocalPointProcessGLM
from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import SchottdorfRecording
from data.schottdorf_lee_multirecording import (
    SchottdorfMovieDrive,
    load_schottdorf_cell,
)
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    constant_rate_logits,
    evaluate_retinal_model,
)
from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.schottdorf_final_benchmark_types import (
    FinalBenchmarkConfig,
    FinalBenchmarkError,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    sha256_file,
)
from evaluation.mechanistic_retina.schottdorf_neural_baseline import (
    CompactNeuralTrainingRequest,
    evaluate_compact_neural,
    fit_compact_neural_baseline,
)
from evaluation.mechanistic_retina.schottdorf_prediction_baselines import (
    evaluate_dynamic_glm,
)
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.real_sampled import spike_prediction_metrics


def evaluate_final_benchmark_cell(
    config: FinalBenchmarkConfig,
    adapter: SchottdorfAdapterConfig,
    movie: SchottdorfMovieDrive,
    recordings: tuple[SchottdorfRecording, ...],
    source_cell: dict[str, JsonValue],
    glm_cell: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    _verify_recording_hashes(recordings, source_cell)
    data = load_schottdorf_cell(recordings, movie, adapter)
    _verify_data_contract(data, source_cell, glm_cell, adapter)
    retinal_path = _cell_path(config.retinal_artifact_dir, data.cell_ids[0], "model-trained.pt")
    glm_path = _cell_path(config.glm_artifact_dir, data.cell_ids[0], "glm-trained.pt")
    retinal_sha = sha256_file(retinal_path)
    glm_sha = sha256_file(glm_path)
    checkpoint = torch.load(retinal_path, map_location="cpu", weights_only=True)
    model_config = _model_config(checkpoint)
    retinal = build_mechanistic_retina(
        model_config,
        data.cone_positions_degs,
        data.cell_positions_degs,
        data.cell_types,
        data.polarities,
    )
    retinal.load_state_dict(checkpoint["model"], strict=True)
    retinal_metrics, _ = evaluate_retinal_model(retinal, data.validation)
    replay_error = abs(retinal_metrics.population_nll - source_cell["validation_nll_trained"])
    if replay_error >= 1e-7:
        raise FinalBenchmarkError("Revision 4 checkpoint replay NLL mismatch")

    constant_logits = constant_rate_logits(
        data.train.spike_events,
        data.train.valid_mask,
        data.validation.spike_events,
        data.validation.valid_mask,
    )
    constant_metrics = spike_prediction_metrics(
        constant_logits, data.validation.spike_events, data.validation.valid_mask
    )
    glm = _load_glm(glm_path, data, model_config.lag_steps)
    glm_metrics, _ = evaluate_dynamic_glm(glm, data.validation)
    if abs(glm_metrics.population_nll - glm_cell["glm_nll"]) >= 1e-7:
        raise FinalBenchmarkError("causal GLM replay NLL mismatch")

    retinal_parameters = _parameter_counts(retinal, phase1_parameters(retinal))
    neural = fit_compact_neural_baseline(
        CompactNeuralTrainingRequest(
            train=data.train,
            cone_positions=data.cone_positions_degs,
            cell_positions=data.cell_positions_degs,
            target_parameters=retinal_parameters["total"],
            seed=int(checkpoint["seed"]),
            maximum_steps=config.neural_maximum_steps,
            patience=config.neural_patience,
            learning_rate=config.neural_learning_rate,
            weight_decay=config.neural_weight_decay,
        )
    )
    neural_metrics, _ = evaluate_compact_neural(neural.model, data.validation)
    _verify_unchanged(retinal_path, retinal_sha, glm_path, glm_sha)
    neural_path = _save_neural(config, data, neural, retinal_sha, glm_sha)

    nll = {
        "constant": constant_metrics.population_nll,
        "glm": glm_metrics.population_nll,
        "neural": neural_metrics.population_nll,
        "retinal": retinal_metrics.population_nll,
    }
    return {
        "cell_id": data.cell_ids[0],
        "recording_ids": list(data.recording_ids),
        "recording_count": len(data.recording_ids),
        "retinal_class": data.retinal_classes[0],
        "canonical_cell_type": data.cell_types[0],
        "polarity": data.polarities[0],
        "native_dt_ms": data.dt_ms,
        "train_sequences": data.train.cone_drive.shape[0],
        "validation_sequences": data.validation.cone_drive.shape[0],
        "train_valid_bins": int(data.train.valid_mask.sum()),
        "validation_valid_bins": int(data.validation.valid_mask.sum()),
        "constant_nll": nll["constant"],
        "glm_nll": nll["glm"],
        "neural_nll": nll["neural"],
        "retinal_nll": nll["retinal"],
        "winner": min(nll, key=nll.__getitem__),
        "neural_train_nll_initial": neural.train_nll_initial,
        "neural_train_nll_trained": neural.train_nll_trained,
        "neural_best_step": neural.best_step,
        "neural_stop_step": neural.stop_step,
        "neural_gradients_finite": neural.gradients_finite,
        "neural_actually_updated": list(neural.actually_updated),
        "constant_parameters": {"total": 1, "requires_grad": 0, "optimizer_listed": 0},
        "glm_parameters": _parameter_counts(glm, tuple(glm.parameters())),
        "neural_parameters": _parameter_counts(neural.model, tuple(neural.model.parameters())),
        "retinal_parameters": retinal_parameters,
        "retinal_nll_replay_error": replay_error,
        "glm_nll_replay_error": abs(glm_metrics.population_nll - glm_cell["glm_nll"]),
        "source_retinal_checkpoint": str(retinal_path.resolve()),
        "source_retinal_checkpoint_sha256": retinal_sha,
        "source_glm_checkpoint": str(glm_path.resolve()),
        "source_glm_checkpoint_sha256": glm_sha,
        "neural_checkpoint": str(neural_path.resolve()),
    }


def _verify_data_contract(
    data,
    source: dict[str, JsonValue],
    glm: dict[str, JsonValue],
    adapter,
) -> None:
    keys = ("cell_id", "recording_ids", "train_sequences", "validation_sequences", "train_valid_bins", "validation_valid_bins")
    current = {
        "cell_id": data.cell_ids[0],
        "recording_ids": list(data.recording_ids),
        "train_sequences": data.train.cone_drive.shape[0],
        "validation_sequences": data.validation.cone_drive.shape[0],
        "train_valid_bins": int(data.train.valid_mask.sum()),
        "validation_valid_bins": int(data.validation.valid_mask.sum()),
    }
    if any(current[key] != source[key] or current[key] != glm[key] for key in keys):
        raise FinalBenchmarkError("baseline data/split contract mismatch")
    if set(data.train.source_image_ids) & set(data.validation.source_image_ids):
        raise FinalBenchmarkError("training and validation temporal segments overlap")
    if asdict(adapter) != source.get("adapter_config", asdict(adapter)):
        raise FinalBenchmarkError("adapter contract mismatch")


def _verify_recording_hashes(
    recordings: tuple[SchottdorfRecording, ...],
    source: dict[str, JsonValue],
) -> None:
    expected = source["source_sha256"]
    if not isinstance(expected, dict):
        raise FinalBenchmarkError("source spike hash contract is missing")
    for recording in recordings:
        digest = expected.get(recording.path.name)
        if not isinstance(digest, str) or sha256_file(recording.path) != digest:
            raise FinalBenchmarkError("spike/source artifact hash mismatch")


def _model_config(checkpoint) -> MechanisticRetinaConfig:
    payload = dict(checkpoint["model_config"])
    payload["architecture_mode"] = ArchitectureMode(payload["architecture_mode"])
    return MechanisticRetinaConfig(**payload)


def _load_glm(path, data, temporal_lags: int) -> LocalPointProcessGLM:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    valid = (
        payload["schema"] == "schottdorf_lee_2021_causal_per_cell_glm_v1"
        and payload["cell_id"] == data.cell_ids[0]
        and tuple(payload["recording_ids"]) == data.recording_ids
        and payload["temporal_lags"] == temporal_lags
        and payload["history_lags"] == 4
        and payload["l2_penalty"] == 1e-4
    )
    if not valid:
        raise FinalBenchmarkError("stored causal GLM contract mismatch")
    model = LocalPointProcessGLM(
        data.cone_positions_degs,
        data.cell_positions_degs,
        radius_deg=None,
        temporal_lags=temporal_lags,
        history_lags=4,
        support_mask=torch.ones((1, data.cone_positions_degs.shape[0]), dtype=torch.bool),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model


def _save_neural(config, data, result, retinal_sha: str, glm_sha: str):
    cell_dir = config.output_dir / "cells" / data.cell_ids[0].replace("#", "_")
    cell_dir.mkdir(parents=True, exist_ok=False)
    path = cell_dir / "compact-neural-trained.pt"
    torch.save(
        {
            "schema": "schottdorf_lee_2021_compact_graph_tcn_v1",
            "cell_id": data.cell_ids[0],
            "recording_ids": data.recording_ids,
            "source_retinal_checkpoint_sha256": retinal_sha,
            "source_glm_checkpoint_sha256": glm_sha,
            "history_lags": 4,
            "stimulus_receptive_field_steps": result.model.receptive_field_steps,
            "hidden_width": result.model.width,
            "model": result.model.state_dict(),
        },
        path,
    )
    return path


def _cell_path(root, cell_id: str, name: str):
    return root / "cells" / cell_id.replace("#", "_") / name


def _verify_unchanged(retinal_path, retinal_sha, glm_path, glm_sha) -> None:
    if sha256_file(retinal_path) != retinal_sha or sha256_file(glm_path) != glm_sha:
        raise FinalBenchmarkError("frozen source checkpoint changed during benchmark")


def _parameter_counts(model, optimizer_parameters) -> dict[str, int]:
    return {
        "total": sum(value.numel() for value in model.parameters()),
        "requires_grad": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "optimizer_listed": sum(value.numel() for value in optimizer_parameters),
    }


__all__ = ["evaluate_final_benchmark_cell"]
