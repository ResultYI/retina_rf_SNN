from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import torch

from evaluation.dynamic_rf import (
    build_matched_context_pairs,
    evaluate_dynamic_rf,
    select_dynamic_rf_units,
)
from evaluation.dynamic_rf_summary import (
    compare_dynamic_rf,
    not_run_dynamic_rf_summary,
)
from evaluation.parameter_audit import audit_parameters
from evaluation.representation_diagnostics import (
    RepresentationDiagnostics,
    collect_decoder_examples,
    compare_representation_diagnostics,
    representation_diagnostics,
)
from evaluation.reporting import summarize_evaluation, write_evaluation_report
from evaluation.rgc_types import identify_rgc_types
from evaluation.temporal_probes import run_temporal_probes
from models.decoder.local_decoder import TiedLocalDecoder
from training.augmentation import AugmentedClip
from training.config import ExperimentConfig
from training.data import PreparedData
from training.runtime import (
    InitialReference,
    ValidationContext,
    build_network,
    evaluate_validation,
)
from training.trainer import RetinaTrainer


@dataclass(frozen=True, slots=True)
class FinalEvaluationRequest:
    trainer: RetinaTrainer
    prepared: PreparedData
    config: ExperimentConfig
    validation: ValidationContext
    calibration_clips: tuple[AugmentedClip, ...]
    initial_reference: InitialReference
    initial_diagnostics: RepresentationDiagnostics
    output_dir: Path
    device: torch.device


def run_final_evaluation(request: FinalEvaluationRequest) -> None:
    trainer = request.trainer
    config = request.config
    model = trainer.model
    decoder = trainer.decoder
    reconstruction, output, target_energy_ratio = evaluate_validation(
        trainer,
        request.validation,
        config,
    )

    initialized_model, initialized_decoder = build_network(
        config,
        request.prepared,
        request.device,
    )
    initialized_model.load_state_dict(request.initial_reference.model_state)
    initialized_decoder.load_state_dict(request.initial_reference.decoder_state)
    selected_diagnostics = _write_selected_representation(
        request,
        initialized_decoder,
    )
    comparison = compare_representation_diagnostics(
        request.initial_diagnostics,
        selected_diagnostics,
    )
    (request.output_dir / "representation_comparison.json").write_text(
        json.dumps(asdict(comparison), indent=2),
        encoding="utf-8",
    )

    minimum_skill_passed = (
        reconstruction.representation_skill
        >= config.evaluation.minimum_representation_skill
    )
    energy_passed = (
        trainer.optimizer_step >= config.training.budget_ramp_end_step
        and target_energy_ratio is not None
        and target_energy_ratio <= config.evaluation.maximum_energy_budget_ratio
    )
    if minimum_skill_passed and energy_passed:
        pairs = build_matched_context_pairs(
            request.prepared.validation,
            config.data,
            config.evaluation,
        )
        selection_plan = select_dynamic_rf_units(model, pairs, config.evaluation)
        trained_dynamic_rf = evaluate_dynamic_rf(
            model,
            pairs,
            config.evaluation,
            dt_ms=request.prepared.dt_ms,
            selection_plan=selection_plan,
        )
        initialized_dynamic_rf = evaluate_dynamic_rf(
            initialized_model,
            pairs,
            config.evaluation,
            dt_ms=request.prepared.dt_ms,
            selection_plan=selection_plan,
        )
        onset_step = (
            config.training.burn_in_steps + config.training.context_only_steps
        )
        probes = run_temporal_probes(
            model,
            request.validation.train_mean,
            sequence_steps=config.data.sequence_steps,
            onset_step=onset_step,
            dt_ms=request.prepared.dt_ms,
        )
        initialized_probes = run_temporal_probes(
            initialized_model,
            request.validation.train_mean,
            sequence_steps=config.data.sequence_steps,
            onset_step=onset_step,
            dt_ms=request.prepared.dt_ms,
        )
        type_report = identify_rgc_types(
            model.rgc,
            output,
            probes=probes,
            initialized_rgc=initialized_model.rgc,
            initialized_probes=initialized_probes,
            config=config.evaluation,
            seed=config.seed,
        )
        dynamic_rf_summary, dynamic_rf_sources = compare_dynamic_rf(
            trained_dynamic_rf,
            initialized_dynamic_rf,
            config.evaluation,
            seed=config.seed,
        )
    else:
        selection_plan = ()
        trained_dynamic_rf = ()
        initialized_dynamic_rf = ()
        dynamic_rf_sources = ()
        dynamic_rf_summary = not_run_dynamic_rf_summary()
        type_report = None

    summary = summarize_evaluation(
        reconstruction,
        target_energy_ratio,
        trained_dynamic_rf,
        config,
        dynamic_rf_status=dynamic_rf_summary.status,
        rgc_type_status=type_report.status if type_report is not None else "not_run",
        budget_ramp_complete=(
            trainer.optimizer_step >= config.training.budget_ramp_end_step
        ),
    )
    write_evaluation_report(
        request.output_dir,
        summary,
        trained_dynamic_rf,
        initialized_dynamic_rf,
        selection_plan,
        dynamic_rf_summary,
        dynamic_rf_sources,
        type_report,
        config,
    )
    audit = [
        asdict(entry)
        for entry in audit_parameters(
            model,
            decoder,
            initialized_model=initialized_model,
            initialized_decoder=initialized_decoder,
        )
    ]
    (request.output_dir / "parameter_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )


def _write_selected_representation(
    request: FinalEvaluationRequest,
    fixed_calibrated_decoder: TiedLocalDecoder,
) -> RepresentationDiagnostics:
    trainer = request.trainer
    spatial_weights = trainer.model.rgc.compute_spatial_weights()
    diagnostics = representation_diagnostics(
        trainer.decoder,
        fixed_calibrated_decoder,
        collect_decoder_examples(
            trainer.model,
            request.calibration_clips,
            request.config.training.supervised_steps,
        ),
        collect_decoder_examples(
            trainer.model,
            request.validation.clips,
            request.config.training.supervised_steps,
        ),
        spatial_weights,
        torch.as_tensor(
            request.prepared.positions_degs,
            device=request.device,
            dtype=spatial_weights.dtype,
        ),
        request.validation.train_mean,
        request.validation.ema_alpha,
    )
    (request.output_dir / "representation_selected.json").write_text(
        json.dumps(asdict(diagnostics), indent=2),
        encoding="utf-8",
    )
    return diagnostics


__all__ = ["FinalEvaluationRequest", "run_final_evaluation"]
