from __future__ import annotations

from dataclasses import asdict

import torch

from data.schottdorf_lee_2021 import SchottdorfAdapterConfig
from data.schottdorf_lee_catalog import SchottdorfRecording
from data.schottdorf_lee_multirecording import (
    SchottdorfCellwiseData,
    SchottdorfMovieDrive,
    load_schottdorf_cell,
)
from evaluation.mechanistic_retina.karamanlis_prediction_baselines import (
    constant_rate_logits,
    evaluate_retinal_model,
)
from evaluation.mechanistic_retina.schottdorf_baseline_types import (
    BaselineCellRecord,
    ParameterCounts,
    SchottdorfBaselineRunConfig,
    SchottdorfBaselineRunError,
)
from evaluation.mechanistic_retina.schottdorf_multirecording_reporting import (
    sha256_file,
)
from evaluation.mechanistic_retina.schottdorf_prediction_baselines import (
    DynamicGLMTrainingRequest,
    evaluate_dynamic_glm,
    fit_dynamic_glm,
)
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MECHANISTIC_MODEL_REVISION,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina
from training.mechanistic_retina.optimizer import phase1_parameters
from training.mechanistic_retina.real_sampled import spike_prediction_metrics


_HISTORY_LAGS = 4
_L2_PENALTY = 1e-4
_SOURCE_SCHEMA = "schottdorf_lee_2021_macaque_cellwise_canonical_v1"


def evaluate_baseline_cell(
    config: SchottdorfBaselineRunConfig,
    adapter: SchottdorfAdapterConfig,
    movie: SchottdorfMovieDrive,
    recordings: tuple[SchottdorfRecording, ...],
    source_cell,
) -> BaselineCellRecord:
    spike_sha256 = {
        recording.path.name: sha256_file(recording.path)
        for recording in recordings
    }
    if any(
        source_cell["source_sha256"][name] != digest
        for name, digest in spike_sha256.items()
    ):
        raise SchottdorfBaselineRunError("spike/source artifact hash mismatch")
    data = load_schottdorf_cell(recordings, movie, adapter)
    checkpoint_path = (
        config.retinal_artifact_dir
        / "cells"
        / data.cell_ids[0].replace("#", "_")
        / "model-trained.pt"
    )
    before = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = _verify_and_build_config(checkpoint, source_cell, data, adapter)
    retinal = build_mechanistic_retina(
        model_config,
        data.cone_positions_degs,
        data.cell_positions_degs,
        data.cell_types,
        data.polarities,
    )
    retinal.load_state_dict(checkpoint["model"], strict=True)
    retinal_metrics, _ = evaluate_retinal_model(retinal, data.validation)
    replay_error = abs(
        retinal_metrics.population_nll - source_cell["validation_nll_trained"]
    )
    if replay_error >= 1e-7:
        raise SchottdorfBaselineRunError("retinal checkpoint replay NLL mismatch")
    constant_logits = constant_rate_logits(
        data.train.spike_events,
        data.train.valid_mask,
        data.validation.spike_events,
        data.validation.valid_mask,
    )
    constant_metrics = spike_prediction_metrics(
        constant_logits, data.validation.spike_events, data.validation.valid_mask
    )
    glm = fit_dynamic_glm(
        DynamicGLMTrainingRequest(
            train=data.train,
            cone_positions=data.cone_positions_degs,
            cell_positions=data.cell_positions_degs,
            temporal_lags=model_config.lag_steps,
            history_lags=_HISTORY_LAGS,
            max_iterations=config.glm_max_iterations,
            seed=int(checkpoint["seed"]),
            l2_penalty=_L2_PENALTY,
        )
    )
    glm_metrics, _ = evaluate_dynamic_glm(glm.model, data.validation)
    after = sha256_file(checkpoint_path)
    if after != before:
        raise SchottdorfBaselineRunError("retinal checkpoint changed during baseline run")
    if any(
        sha256_file(recording.path) != spike_sha256[recording.path.name]
        for recording in recordings
    ):
        raise SchottdorfBaselineRunError("spike recording changed during baseline run")
    _save_glm(config, data, checkpoint, before, glm.model.state_dict())
    nll = {
        "constant_rate": constant_metrics.population_nll,
        "glm": glm_metrics.population_nll,
        "retinal": retinal_metrics.population_nll,
    }
    return BaselineCellRecord(
        cell_id=data.cell_ids[0],
        recording_ids=list(data.recording_ids),
        recording_count=len(recordings),
        retinal_class=data.retinal_classes[0],
        canonical_cell_type=data.cell_types[0],
        polarity=data.polarities[0],
        native_dt_ms=data.dt_ms,
        train_sequences=data.train.cone_drive.shape[0],
        validation_sequences=data.validation.cone_drive.shape[0],
        train_valid_bins=int(data.train.valid_mask.sum()),
        validation_valid_bins=int(data.validation.valid_mask.sum()),
        time_segment_disjoint=set(data.train.source_image_ids).isdisjoint(
            data.validation.source_image_ids
        ),
        constant_rate_nll=nll["constant_rate"],
        glm_nll=nll["glm"],
        retinal_nll=nll["retinal"],
        winner=min(nll, key=nll.__getitem__),
        retinal_strictly_better_than_constant=nll["retinal"]
        < nll["constant_rate"],
        glm_train_nll_initial=glm.train_nll_initial,
        glm_train_nll_trained=glm.train_nll_trained,
        glm_solver_iterations=glm.solver_iterations,
        glm_solver_evaluations=glm.solver_evaluations,
        glm_final_gradient_max=glm.final_gradient_max,
        glm_strict_gradient_converged=glm.strict_gradient_converged,
        glm_solver_terminated_before_budget=glm.solver_terminated_before_budget,
        glm_converged=glm.converged,
        glm_gradients_finite=glm.gradients_finite,
        glm_actually_updated=list(glm.actually_updated),
        constant_parameters={"total": 1, "requires_grad": 0, "optimizer_listed": 0},
        glm_parameters=_parameter_counts(glm.model, tuple(glm.model.parameters())),
        retinal_parameters=_parameter_counts(retinal, phase1_parameters(retinal)),
        retinal_nll_replay_error=replay_error,
        source_checkpoint=str(checkpoint_path.resolve()),
        source_checkpoint_sha256_before=before,
        source_checkpoint_sha256_after=after,
    )


def _save_glm(config, data, checkpoint, source_sha256: str, model_state) -> None:
    cell_dir = config.output_dir / "cells" / data.cell_ids[0].replace("#", "_")
    cell_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema": "schottdorf_lee_2021_causal_per_cell_glm_v1",
            "cell_id": data.cell_ids[0],
            "recording_ids": data.recording_ids,
            "source_retinal_checkpoint_sha256": source_sha256,
            "temporal_lags": checkpoint["model_config"]["lag_steps"],
            "history_lags": _HISTORY_LAGS,
            "l2_penalty": _L2_PENALTY,
            "model": model_state,
        },
        cell_dir / "glm-trained.pt",
    )


def _verify_and_build_config(
    checkpoint, source_cell, data: SchottdorfCellwiseData, adapter
) -> MechanisticRetinaConfig:
    valid = (
        checkpoint["schema"] == _SOURCE_SCHEMA
        and checkpoint["stage"] == "trained"
        and int(checkpoint["revision"]) == MECHANISTIC_MODEL_REVISION
        and checkpoint["cell_id"] == data.cell_ids[0] == source_cell["cell_id"]
        and tuple(checkpoint["recording_ids"]) == data.recording_ids
        and checkpoint["adapter_config"] == asdict(adapter)
        and checkpoint["input_representation"] == data.input_representation
        and torch.equal(checkpoint["cone_positions_degs"], data.cone_positions_degs)
        and torch.equal(checkpoint["cell_positions_degs"], data.cell_positions_degs)
        and data.train.cone_drive.shape[0] == source_cell["train_sequences"]
        and data.validation.cone_drive.shape[0]
        == source_cell["validation_sequences"]
        and int(data.train.valid_mask.sum()) == source_cell["train_valid_bins"]
        and int(data.validation.valid_mask.sum())
        == source_cell["validation_valid_bins"]
        and set(data.train.source_image_ids).isdisjoint(data.validation.source_image_ids)
    )
    if not valid:
        raise SchottdorfBaselineRunError("checkpoint/data lineage contract mismatch")
    payload = dict(checkpoint["model_config"])
    payload["architecture_mode"] = ArchitectureMode(payload["architecture_mode"])
    model_config = MechanisticRetinaConfig(**payload)
    if model_config.lag_steps != 16:
        raise SchottdorfBaselineRunError("retinal/GLM temporal lag contract mismatch")
    return model_config


def _parameter_counts(model, optimizer_parameters) -> ParameterCounts:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "requires_grad": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "optimizer_listed": sum(parameter.numel() for parameter in optimizer_parameters),
    }


__all__ = ["evaluate_baseline_cell"]
