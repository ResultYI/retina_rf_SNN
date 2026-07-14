from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from configs.physiology_profiles import dt_ms_from_time_axis_seconds
from data.cone_response import validate_natural_video_splits
from data.dataset import load_log_cone_stats, validate_compatible_cone_exports
from datasets.isetbio_h5_dataset import (
    ConeNormalizationStats,
    ISETBioH5Dataset,
    ISETBioH5DatasetConfig,
    collate_isetbio_h5_batch,
)
from evaluation.checkpoint_contracts import (
    CheckpointEvaluationConfig,
    CheckpointEvaluationPayload,
    CheckpointMetadata,
    EvaluationArtifacts,
    EvaluationDataMetadata,
    HumRetPayload,
    ParameterAuditPayload,
)
from evaluation.checkpoint_metrics import (
    HeldOutEvaluationRequest,
    evaluate_held_out,
    fit_evaluation_baselines,
)
from evaluation.checkpoint_probes import (
    RFProbeRequest,
    run_rf_probes,
    run_temporal_probes,
)
from evaluation.humret import (
    compare_humret_grating_population,
    load_humret_reference,
)
from evaluation.parameter_audit import audit_stage1_parameters
from evaluation.prediction_baselines import LocalARSupports
from training.hybrid import RetinaTrainingBatch
from training.stage1 import (
    MidgetSamplingMode,
    Stage1BuildConfig,
    Stage1Components,
    build_stage1_components,
)
from training.stage1_runtime import load_checkpoint


@dataclass(frozen=True, slots=True)
class DatasetLoaderRequest:
    paths: tuple[Path, ...]
    config: CheckpointEvaluationConfig
    components: Stage1Components
    stats: ConeNormalizationStats
    eps: float


def run_checkpoint_evaluation(
    config: CheckpointEvaluationConfig,
) -> CheckpointEvaluationPayload:
    exports = validate_compatible_cone_exports((*config.train_h5, *config.eval_h5))
    train_exports = exports[: len(config.train_h5)]
    eval_exports = exports[len(config.train_h5) :]
    if config.formal_evidence:
        validate_natural_video_splits(train_exports, eval_exports)
    reference = train_exports[0]
    dt_ms = dt_ms_from_time_axis_seconds(reference.time_axis_seconds)
    components = build_stage1_components(
        reference.positions_degs,
        Stage1BuildConfig(
            dt_ms=dt_ms,
            horizon_count=len(config.horizons),
            eccentricity_deg=reference.eccentricity_deg,
            midget_sampling=(
                MidgetSamplingMode.FOVEAL_PRIVATE_LINE
                if reference.eccentricity_deg == 0
                else MidgetSamplingMode.CONVERGENT
            ),
        ),
    )
    components.core.to(config.device)
    components.decoder.to(config.device)
    checkpoint = load_checkpoint(config.checkpoint, config.device)
    components.core.load_state_dict(checkpoint["core"])
    components.decoder.load_state_dict(checkpoint["decoder"])

    mean, scale, eps = load_log_cone_stats(config.normalization_stats)
    stats = ConeNormalizationStats(mean, scale)
    train_loader = _loader(
        DatasetLoaderRequest(config.train_h5, config, components, stats, eps)
    )
    eval_loader = _loader(
        DatasetLoaderRequest(config.eval_h5, config, components, stats, eps)
    )
    baselines = fit_evaluation_baselines(
        train_loader,
        eval_loader,
        LocalARSupports(
            components.target_pools.fine,
            components.target_pools.coarse,
        ),
    )
    held_out = evaluate_held_out(
        HeldOutEvaluationRequest(
            components,
            eval_loader,
            baselines,
            config.device,
            config.rf_sample_count,
        )
    )
    rf = run_rf_probes(
        RFProbeRequest(
            components,
            held_out,
            config.rf_sample_count,
            config.glm_max_steps,
        )
    )
    temporal = run_temporal_probes(
        components.core,
        reference.positions_degs.shape[0],
        dt_ms,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "evaluation_summary.json"
    rf_path = config.output_dir / "rf_probes.npz"
    payload = CheckpointEvaluationPayload(
        checkpoint=CheckpointMetadata(
            path=str(config.checkpoint),
            stage=str(checkpoint["stage"]),
            epoch=int(checkpoint["epoch"]),
            step=int(checkpoint["step"]),
        ),
        data=EvaluationDataMetadata(
            normalization_stats=str(config.normalization_stats),
            train_h5=tuple(str(path) for path in config.train_h5),
            eval_h5=tuple(str(path) for path in config.eval_h5),
            train_exports=len(train_exports),
            eval_exports=len(eval_exports),
            eval_samples=len(eval_loader.dataset),
            cone_count=reference.positions_degs.shape[0],
            dt_ms=dt_ms,
            input_steps=config.input_steps,
            horizons=config.horizons,
        ),
        prediction=held_out.prediction,
        population_usage=held_out.population_usage,
        population_ablation=held_out.population_ablation,
        temporal_probe_interpretation="direct_normalized_contrast_diagnostic",
        temporal_probes=temporal,
        rf_probes=rf.metrics,
        parameter_audit=_parameter_payload(components),
        humret=_humret_payload(config),
        artifacts=EvaluationArtifacts(
            summary=str(summary_path),
            rf_probes=str(rf_path),
        ),
    )
    np.savez_compressed(rf_path, **rf.arrays)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _loader(request: DatasetLoaderRequest) -> DataLoader[RetinaTrainingBatch]:
    datasets = tuple(
        ISETBioH5Dataset(
            ISETBioH5DatasetConfig(
                h5_path=path,
                input_steps=request.config.input_steps,
                horizons=request.config.horizons,
                eps=request.eps,
                target_fine_pool=request.components.target_pools.fine,
                target_coarse_pool=request.components.target_pools.coarse,
            ),
            request.stats,
        )
        for path in request.paths
    )
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=request.config.batch_size,
        shuffle=False,
        collate_fn=collate_isetbio_h5_batch,
    )


def _parameter_payload(
    components: Stage1Components,
) -> tuple[ParameterAuditPayload, ...]:
    return tuple(
        ParameterAuditPayload(
            name=item.name,
            value=item.value,
            lower=item.lower,
            upper=item.upper,
            boundary_fraction=item.boundary_fraction,
            near_boundary=item.near_boundary,
        )
        for item in audit_stage1_parameters(components)
    )


def _humret_payload(config: CheckpointEvaluationConfig) -> HumRetPayload:
    if config.humret_root is None or config.humret_model_grating is None:
        return HumRetPayload(
            status="not_run",
            reason="requires ISETBio-derived model grating F1 artifact",
        )
    reference = load_humret_reference(config.humret_root)
    model_tuning = torch.from_numpy(
        np.asarray(np.load(config.humret_model_grating), dtype=np.float32)
    )
    agreement = compare_humret_grating_population(model_tuning, reference)
    return HumRetPayload(
        status="ok",
        reference_root=str(config.humret_root),
        model_grating_artifact=str(config.humret_model_grating),
        human_cells=reference.grating_f1_normalized.shape[0],
        model_units=model_tuning.shape[0],
        mean_tuning_cosine_similarity=agreement.mean_tuning_cosine_similarity,
        spatial_preference_total_variation=(
            agreement.spatial_preference_total_variation
        ),
        temporal_preference_total_variation=(
            agreement.temporal_preference_total_variation
        ),
    )
