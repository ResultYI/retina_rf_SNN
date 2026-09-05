from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from data.karamanlis_rf_population import (
    RFPopulationAdapterConfig,
    load_rf_population_geometry,
    load_rf_population_imagesequence,
)
from evaluation.mechanistic_retina.ac_circuit_support import (
    parameter_invariance,
    state_snapshot,
)
from evaluation.mechanistic_retina.ac_temporal_lineage import temporal_timing_contract
from evaluation.mechanistic_retina.ac_temporal_support import (
    temporal_parameter_invariance,
)
from evaluation.mechanistic_retina.atomic_artifacts import (
    atomic_write_text,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_artifacts import (
    PerturbationArtifactRequest,
    TemporalRFIdentityRequest,
    build_lineage,
    build_temporal_rf_identity,
    save_perturbation_artifacts,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_metrics import (
    CellPerturbationRequest,
    cell_perturbation_metrics,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_payload import (
    PerturbationPayloadRequest,
    build_results_payload,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_reporting import (
    PerturbationReportRequest,
    build_perturbation_summary,
)
from evaluation.mechanistic_retina.karamanlis_v1_ac_runtime import (
    AC_CLAMPS,
    EvaluationRequest,
    collect_responses,
    mean_temporal_rf,
    stimulus_onset_step,
    validate_ac_clamp,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_artifacts import (
    validate_checkpoint_data,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_validation import (
    _one_trial_per_source,
    validate_v1_checkpoint,
)
from models.mechanistic_retina.contracts import (
    MECHANISTIC_MODEL_REVISION,
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.model import build_mechanistic_retina


@dataclass(frozen=True, slots=True)
class V1ACPerturbationConfig:
    session_dir: Path
    graph_dir: Path
    checkpoint_path: Path
    output_dir: Path
    response_batch_size: int = 4
    rf_batch_size: int = 2


@dataclass(frozen=True, slots=True)
class V1ACPerturbationResult:
    artifact_dir: Path
    mean_absolute_logit_change: float
    mean_absolute_probability_change: float
    temporal_rf_cosine: float


class V1ACPerturbationError(ValueError):
    pass


def run_v1_ac_perturbation(
    config: V1ACPerturbationConfig,
) -> V1ACPerturbationResult:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError("V1 AC perturbation output directory must be empty")
    if min(config.response_batch_size, config.rf_batch_size) < 1:
        raise V1ACPerturbationError("evaluation batch sizes must be positive")
    checkpoint = torch.load(
        config.checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise V1ACPerturbationError("checkpoint payload must be a mapping")
    validate_v1_checkpoint(checkpoint)
    if checkpoint.get("revision") != MECHANISTIC_MODEL_REVISION:
        raise V1ACPerturbationError("checkpoint model revision is unsupported")
    geometry = load_rf_population_geometry(config.graph_dir, grid_size=51)
    data = load_rf_population_imagesequence(
        config.session_dir, geometry, RFPopulationAdapterConfig()
    )
    validate_checkpoint_data(checkpoint, data)
    model_config_values = dict(checkpoint["model_config"])
    model_config_values["architecture_mode"] = ArchitectureMode(
        model_config_values["architecture_mode"]
    )
    model_config = MechanisticRetinaConfig(**model_config_values)
    if model_config.dt_ms != data.dt_ms:
        raise V1ACPerturbationError("checkpoint and held-out data dt differ")
    model = build_mechanistic_retina(
        model_config,
        data.model_cone_positions,
        data.model_cell_positions,
        data.cell_types,
        data.polarities,
        shared_subunit_edge_index=data.edge_index,
        pathway_spatial_geometry=data.pathway_spatial_geometry,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    state_before = state_snapshot(model)
    response_request = EvaluationRequest(
        model,
        data.validation.cone_drive,
        data.validation.spike_events,
        config.response_batch_size,
    )
    responses = collect_responses(response_request)
    context_indices = _one_trial_per_source(data.validation.source_image_ids)
    rf_request = EvaluationRequest(
        model,
        data.validation.cone_drive,
        data.validation.spike_events,
        config.rf_batch_size,
    )
    normal_temporal_rf = mean_temporal_rf(rf_request, context_indices, frozenset())
    clamped_temporal_rf = mean_temporal_rf(rf_request, context_indices, AC_CLAMPS)
    state_after = state_snapshot(model)
    invariance = temporal_parameter_invariance(
        model,
        state_before,
        state_after,
        parameter_invariance(model, state_before, state_after),
    )
    clamp_verification = validate_ac_clamp(
        responses,
        state_unchanged=bool(invariance["all_state_tensors_unchanged"]),
    )
    baseline_steps = stimulus_onset_step(data.validation.cone_drive)
    metrics = cell_perturbation_metrics(
        CellPerturbationRequest(
            responses.normal.logits,
            responses.clamped.logits,
            responses.normal.probability,
            responses.clamped.probability,
            normal_temporal_rf,
            clamped_temporal_rf,
            baseline_steps,
            data.dt_ms,
        )
    )
    summary = build_perturbation_summary(
        PerturbationReportRequest(
            data.cell_ids, data.cell_types, data.polarities, metrics
        )
    )
    lag_ms = torch.arange(model.config.lag_steps) * data.dt_ms
    lineage = build_lineage(
        config, data.validation.cone_drive, data.validation.spike_events
    )
    selected_source_ids = tuple(
        data.validation.source_image_ids[index] for index in context_indices
    )
    selected_trial_indices = tuple(
        data.validation.trial_indices[index] for index in context_indices
    )
    artifact_identity = build_temporal_rf_identity(
        TemporalRFIdentityRequest(
            checkpoint_sha256=str(lineage["checkpoint_sha256"]),
            checkpoint_stage=str(checkpoint["stage"]),
            checkpoint_best_step=int(checkpoint["best_step"]),
            training_seed=int(checkpoint["training_seed"]),
            model_revision=int(checkpoint["revision"]),
            dt_ms=data.dt_ms,
            lag_ms=lag_ms,
            cell_ids=data.cell_ids,
            cell_types=data.cell_types,
            polarities=data.polarities,
            context_indices=context_indices,
            source_image_ids=selected_source_ids,
            trial_indices=selected_trial_indices,
            source_sha256=lineage["source_sha256"],
        )
    )
    timing_contract = dict(temporal_timing_contract(model, invariance)) | {
        "rf_lag_window": {
            "lag_steps": model.config.lag_steps,
            "dt_ms": model.config.dt_ms,
            "lag_order_semantics": "current_to_past",
            "lag_ms": lag_ms.tolist(),
            "learnable": False,
        }
    }
    payload = build_results_payload(
        PerturbationPayloadRequest(
            clamps=sorted(clamp.value for clamp in AC_CLAMPS),
            lineage=lineage,
            checkpoint_stage=str(checkpoint["stage"]),
            checkpoint_best_step=int(checkpoint["best_step"]),
            training_seed=int(checkpoint["training_seed"]),
            sequence_count=data.validation.cone_drive.shape[0],
            source_image_count=len(set(data.validation.source_image_ids)),
            time_steps=data.validation.cone_drive.shape[1],
            cell_count=len(data.cell_ids),
            dt_ms=data.dt_ms,
            stimulus_onset_step=baseline_steps,
            context_indices=context_indices,
            source_image_ids=selected_source_ids,
            trial_indices=selected_trial_indices,
            artifact_identity=artifact_identity,
            timing_contract=timing_contract,
            invariance=invariance,
            clamp=clamp_verification,
            upstream_outputs_unchanged=responses.upstream_outputs_unchanged,
            summary=summary,
        )
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_perturbation_artifacts(
        PerturbationArtifactRequest(
            config.output_dir,
            responses,
            normal_temporal_rf,
            clamped_temporal_rf,
            lag_ms,
            artifact_identity,
        )
    )
    atomic_write_text(
        config.output_dir / "results.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )
    population = summary["population"]
    return V1ACPerturbationResult(
        config.output_dir,
        float(population["response_change"]["mean_absolute_logit"]),
        float(population["response_change"]["mean_absolute_probability"]),
        float(population["temporal_rf"]["cosine"]),
    )


__all__ = ["V1ACPerturbationConfig", "V1ACPerturbationError", "V1ACPerturbationResult", "run_v1_ac_perturbation"]
