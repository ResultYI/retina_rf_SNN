from __future__ import annotations

import json

import numpy as np
import torch

from configs.physiology_profiles import dt_ms_from_time_axis_seconds
from data.cone_response import validate_formal_stimulus_splits
from data.dataset import load_log_cone_stats, validate_compatible_cone_exports
from datasets.isetbio_h5_dataset import (
    ConeNormalizationStats,
)
from evaluation.checkpoint_contracts import (
    CheckpointEvaluationConfig,
    CheckpointEvaluationPayload,
    CheckpointMetadata,
    EvaluationArtifacts,
    EvaluationDataMetadata,
    RFProbeStatusPayload,
)
from evaluation.checkpoint_context import (
    EmpiricalContextRequest,
    architecture_compliance_payload,
    context_audit_payload,
    evaluate_empirical_context,
    extended_filtering_steps,
    rf_probe_gate,
    RFGateRequest,
)
from evaluation.checkpoint_loaders import DatasetLoaderRequest, checkpoint_loader
from evaluation.checkpoint_metrics import (
    HeldOutEvaluationRequest,
    evaluate_held_out,
    fit_evaluation_baselines,
)
from evaluation.checkpoint_probes import (
    RFProbeBundle,
    RFProbeRequest,
    run_rf_probes,
)
from evaluation.checkpoint_payloads import humret_payload, parameter_payload
from evaluation.reconstruction_baselines import LocalLinearSupport
from evaluation.rf_identifiability import (
    RFGLMIdentifiabilityRequest,
    rf_glm_identifiability,
)
from evaluation.temporal_probes import run_temporal_probes
from training.stage1 import (
    MidgetSamplingMode,
    Stage1BuildConfig,
    Stage1Components,
    build_stage1_components,
)
from training.stage1_runtime import (
    batch_to_device,
    filtering_context_requirement,
    load_checkpoint,
)
from training.stage1_types import TrainStage1Error

def run_checkpoint_evaluation(
    config: CheckpointEvaluationConfig,
) -> CheckpointEvaluationPayload:
    exports = validate_compatible_cone_exports((*config.train_h5, *config.eval_h5))
    train_exports = exports[: len(config.train_h5)]
    eval_exports = exports[len(config.train_h5) :]
    if config.formal_evidence:
        validate_formal_stimulus_splits(train_exports, eval_exports)
    reference = train_exports[0]
    dt_ms = dt_ms_from_time_axis_seconds(reference.time_axis_seconds)
    components = build_stage1_components(
        reference.positions_degs,
        Stage1BuildConfig(
            dt_ms=dt_ms,
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
    if "rgc.raw_kinetic_mix" in checkpoint["core"]:
        raise TrainStage1Error(
            "Checkpoint is ordered-kinetics incompatible; fresh run required: "
            "rgc.raw_kinetic_mix"
        )
    components.core.load_state_dict(checkpoint["core"])
    components.decoder.load_state_dict(checkpoint["decoder"])

    mean, scale, eps = load_log_cone_stats(config.normalization_stats)
    stats = ConeNormalizationStats(mean, scale)
    requirement = filtering_context_requirement(components.profile)
    analytic_ok = config.input_steps >= requirement.required_steps
    if config.formal_evidence and not analytic_ok:
        raise TrainStage1Error(
            "Formal evidence requires filtering context of at least "
            f"{requirement.required_steps} input steps; got {config.input_steps}"
        )
    extended_steps = extended_filtering_steps(
        requirement.required_steps,
        requirement.tau_upper_ms,
        requirement.dt_ms,
    )
    empirical = evaluate_empirical_context(
        EmpiricalContextRequest(
            config.eval_h5,
            components,
            stats,
            eps,
            requirement.required_steps,
            extended_steps,
            config.device,
            requirement.residual_tolerance,
        )
    )
    if config.formal_evidence and not empirical.sufficient:
        reason = empirical.reason or "paired reconstruction filtering context exceeded tolerance"
        raise TrainStage1Error(f"Formal evidence requires {reason}")
    train_loader = checkpoint_loader(
        DatasetLoaderRequest(config.train_h5, config, components, stats, eps)
    )
    eval_loader = checkpoint_loader(
        DatasetLoaderRequest(config.eval_h5, config, components, stats, eps)
    )
    baselines = fit_evaluation_baselines(
        train_loader,
        eval_loader,
        LocalLinearSupport(components.current_reconstruction_support),
    )
    held_out = evaluate_held_out(
        HeldOutEvaluationRequest(
            components,
            eval_loader,
            baselines,
            config.device,
        )
    )
    rf_gate = rf_probe_gate(
        RFGateRequest(
            analytic_ok,
            empirical.sufficient,
            held_out.population_usage,
        )
    )
    rf_status = rf_gate
    if rf_status.status == "run":
        identifiability = rf_glm_identifiability(
            RFGLMIdentifiabilityRequest(
                sequence_count=len(eval_loader.dataset),
                input_steps=config.input_steps,
                source_counts=_rf_source_counts(components),
            )
        )
        if not identifiability.sufficient:
            rf_status = type(rf_gate)("not_identifiable", identifiability.reason)
    if rf_status.status == "run":
        probe_batch = batch_to_device(next(iter(eval_loader)), config.device)
        with torch.no_grad():
            probe_output, _ = components.core.forward_sequence(probe_batch.x_cone)
        rf = run_rf_probes(
            RFProbeRequest(
                components,
                held_out,
                probe_batch.x_cone,
                probe_output,
                config.rf_sample_count,
                config.glm_max_steps,
            )
        )
    else:
        rf = RFProbeBundle((), {})
    temporal = run_temporal_probes(
        components.core,
        reference.positions_degs.shape[0],
        dt_ms,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "evaluation_summary.json"
    rf_path = config.output_dir / "rf_probes.npz"
    payload = CheckpointEvaluationPayload(
        evidence_class="formal_candidate" if config.formal_evidence else "non_formal_smoke",
        context_audit=context_audit_payload(
            config.input_steps,
            requirement,
            analytic_ok,
            empirical,
        ),
        architecture_compliance=architecture_compliance_payload(components),
        rf_probe_status=RFProbeStatusPayload(
            status=rf_status.status,
            reason=rf_status.reason,
        ),
        checkpoint=CheckpointMetadata(
            path=str(config.checkpoint),
            stage=str(checkpoint["phase"]),
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
        ),
        reconstruction=held_out.reconstruction,
        population_usage=held_out.population_usage,
        population_ablation=held_out.population_ablation,
        temporal_probe_interpretation="direct_normalized_contrast_diagnostic",
        temporal_probes=temporal,
        rf_probes=rf.metrics,
        parameter_audit=parameter_payload(components),
        humret=humret_payload(config),
        artifacts=EvaluationArtifacts(
            summary=str(summary_path),
        ),
    )
    if rf_status.status == "run":
        payload["artifacts"]["rf_probes"] = str(rf_path)
        np.savez_compressed(rf_path, **rf.arrays)
    elif rf_path.exists():
        rf_path.unlink()
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _rf_source_counts(components: Stage1Components) -> dict[str, int]:
    def count(pool: torch.Tensor) -> int:
        indices = pool.coalesce().indices()
        return int((indices[0] == 0).sum())

    return {
        "midget": count(components.core.rgc.midget_pool),
        "parasol": count(components.core.rgc.parasol_pool),
    }
