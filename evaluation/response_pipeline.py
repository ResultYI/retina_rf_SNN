from __future__ import annotations

import glob
import json
from dataclasses import asdict
from pathlib import Path
from typing import assert_never

import torch

from baselines.point_process_glm import fit_point_process_glm
from data.synthetic_teacher import TeacherRFMetadata, load_teacher_rf_metadata
from evaluation.parameter_audit import ParameterAuditContext, audit_parameter_deltas
from evaluation.response_prediction import evaluate_response_prediction
from evaluation.response_report_schema import (
    EvaluationSplit,
    KernelReferenceComparison,
    RFModeEvidence,
    ResponseReportEvidence,
)
from evaluation.response_reporting import write_response_report
from evaluation.rf_artifacts import write_rf_artifacts
from evaluation.rf_dynamic import compare_dynamic_rf, evaluate_dynamic_rf
from evaluation.rf_history_contracts import standard_train_rate_history_counts
from evaluation.rf_history_pipeline import conditional_rf_by_history, mean_static_rf
from evaluation.rf_static import StaticRFResult, compare_rf_kernels, extract_static_rf
from models.response_snn import ResponseRetinaModel
from training.response_config import ResponseExperimentConfig
from training.response_data import PreparedResponseData
from training.response_trainer import ResponseTrainer


def evaluate_and_report_response_experiment(
    output: Path,
    *,
    model: ResponseRetinaModel,
    initialized_model: ResponseRetinaModel,
    trainer: ResponseTrainer,
    data: PreparedResponseData,
    config: ResponseExperimentConfig,
    checkpoint: Path,
    evaluation_split: EvaluationSplit = "validation",
) -> None:
    match evaluation_split:
        case "validation":
            split, teacher_pattern = data.validation, config.data.validation_glob
        case "test":
            split, teacher_pattern = data.test, config.data.test_glob
        case unreachable:
            assert_never(unreachable)
    response_prediction = evaluate_response_prediction(trainer, split, initialized_model)
    glm = fit_point_process_glm(
        data,
        device=trainer.device,
        burn_in_steps=config.training.burn_in_steps,
        evaluate_test=evaluation_split == "test",
    )
    teacher = _teacher_metadata(teacher_pattern)
    teacher_dynamic = (
        None
        if teacher is None
        else (
            torch.as_tensor(
                teacher.context_kernel_low,
                device=model.rgc.support_mask.device,
            ),
            torch.as_tensor(
                teacher.context_kernel_high,
                device=model.rgc.support_mask.device,
            ),
        )
    )
    teacher_envelope = (
        None
        if teacher is None or teacher.context_gain_envelope is None
        else torch.as_tensor(
            teacher.context_gain_envelope,
            device=model.rgc.support_mask.device,
        )
    )
    standard_history = standard_train_rate_history_counts(
        data.train,
        burn_in_steps=config.training.burn_in_steps,
        sequence_steps=split.cone_response.shape[1],
        device=trainer.device,
    )
    conditional_by_history = conditional_rf_by_history(
        model,
        initialized_model,
        split,
        config,
        data.dt_ms,
        config.seed,
        teacher_dynamic,
        teacher_envelope,
        standard_history,
    )
    finite_difference_tolerance = config.evaluation.finite_difference_tolerance
    conditional_static = conditional_by_history["matched_observed"].static_rf
    initialized_conditional_static = conditional_by_history[
        "matched_observed"
    ].initialized_static_rf
    conditional_dynamic = conditional_by_history["matched_observed"].dynamic_rf
    initialized_conditional_dynamic = conditional_by_history[
        "matched_observed"
    ].initialized_dynamic_rf
    conditional_comparison = conditional_by_history[
        "matched_observed"
    ].dynamic_comparison
    free_static = mean_static_rf(
        tuple(
            extract_static_rf(
                model,
                split.cone_response[index : index + 1].to(trainer.device),
                lag_steps=config.evaluation.rf_lag_steps,
                finite_difference_tolerance=finite_difference_tolerance,
            )
            for index in range(split.cone_response.shape[0])
        )
    )
    initialized_free_static = mean_static_rf(
        tuple(
            extract_static_rf(
                initialized_model,
                split.cone_response[index : index + 1].to(trainer.device),
                lag_steps=config.evaluation.rf_lag_steps,
                finite_difference_tolerance=finite_difference_tolerance,
            )
            for index in range(split.cone_response.shape[0])
        )
    )
    free_dynamic = evaluate_dynamic_rf(
        model,
        split,
        lag_steps=config.evaluation.rf_lag_steps,
        condition_on_observed=False,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
        teacher_context_gain_envelope=teacher_envelope,
        finite_difference_tolerance=finite_difference_tolerance,
    )
    initialized_free_dynamic = evaluate_dynamic_rf(
        initialized_model,
        split,
        lag_steps=config.evaluation.rf_lag_steps,
        condition_on_observed=False,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
        teacher_context_gain_envelope=teacher_envelope,
        finite_difference_tolerance=finite_difference_tolerance,
    )
    free_comparison = compare_dynamic_rf(
        free_dynamic,
        initialized_free_dynamic,
        seed=config.seed,
    )
    static_reference = _static_reference(conditional_static, teacher)
    free_static_reference = _static_reference(free_static, teacher)
    evidence = ResponseReportEvidence(
        response_prediction=response_prediction,
        parameter_deltas=audit_parameter_deltas(
            model, initialized_model, ParameterAuditContext.from_cells(data.cells)
        ),
        glm=glm,
        conditional_rf=RFModeEvidence(
            static_rf=conditional_static,
            initialized_static_rf=initialized_conditional_static,
            static_reference=static_reference,
            dynamic_rf=conditional_dynamic,
            initialized_dynamic_rf=initialized_conditional_dynamic,
            dynamic_comparison=conditional_comparison,
            initialized_static_reference=_static_reference(
                initialized_conditional_static,
                teacher,
            ),
        ),
        free_running_rf=RFModeEvidence(
            static_rf=free_static,
            initialized_static_rf=initialized_free_static,
            static_reference=free_static_reference,
            dynamic_rf=free_dynamic,
            initialized_dynamic_rf=initialized_free_dynamic,
            dynamic_comparison=free_comparison,
            initialized_static_reference=_static_reference(
                initialized_free_static,
                teacher,
            ),
        ),
        synthetic=teacher is not None,
        checkpoint=str(checkpoint.resolve()),
        evaluation_split=evaluation_split,
        conditional_rf_by_history=conditional_by_history,
    )
    write_response_report(output, evidence)
    write_rf_artifacts(output, data, evidence)
    torch.save(initialized_model.state_dict(), output / "initialized_model_state.pt")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "dataset_fingerprint": data.fingerprint,
                "target_kind": data.target_kind.value,
                "cell_count": len(data.cells.ids),
                "input_identity": asdict(data.input_identity),
                "evaluation_split": evaluation_split,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _static_reference(
    static_rf: StaticRFResult,
    teacher: TeacherRFMetadata | None,
) -> KernelReferenceComparison | None:
    if teacher is None:
        return None
    comparison = compare_rf_kernels(
        static_rf.kernels,
        torch.as_tensor(
            teacher.static_kernel,
            device=static_rf.kernels.device,
            dtype=static_rf.kernels.dtype,
        ),
    )
    return KernelReferenceComparison(
        comparison["mean_kernel_correlation"], comparison["mean_kernel_norm"]
    )


def _teacher_metadata(pattern: str) -> TeacherRFMetadata | None:
    paths = sorted(glob.glob(pattern))
    return load_teacher_rf_metadata(paths[0]) if paths else None
