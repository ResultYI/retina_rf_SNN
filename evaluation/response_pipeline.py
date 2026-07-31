from __future__ import annotations

import glob
import json
from dataclasses import asdict
from pathlib import Path

import torch

from baselines.point_process_glm import fit_point_process_glm
from data.synthetic_teacher import TeacherRFMetadata, load_teacher_rf_metadata
from evaluation.response_report_schema import (
    KernelReferenceComparison,
    RFModeEvidence,
    ResponseReportEvidence,
)
from evaluation.response_reporting import write_response_report
from evaluation.rf_artifacts import write_rf_artifacts
from evaluation.rf_dynamic import compare_dynamic_rf, evaluate_dynamic_rf
from evaluation.rf_static import compare_rf_kernels, extract_static_rf
from evaluation.rf_static import StaticRFResult
from evaluation.rf_dynamic_metrics import trial_conditioned_rf
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
) -> None:
    conditional = trainer.evaluate(data.test)
    free_running = trainer.evaluate(data.test, free_running=True)
    glm = fit_point_process_glm(
        data,
        device=trainer.device,
        burn_in_steps=config.training.burn_in_steps,
    )
    teacher = _teacher_metadata(config.data.test_glob)
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
    conditional_static = _mean_static_rf(
        tuple(
            trial_conditioned_rf(
                model,
                data.test,
                index,
                config.evaluation.rf_lag_steps,
            )
            for index in range(data.test.cone_response.shape[0])
        )
    )
    initialized_conditional_static = _mean_static_rf(
        tuple(
            trial_conditioned_rf(
                initialized_model,
                data.test,
                index,
                config.evaluation.rf_lag_steps,
            )
            for index in range(data.test.cone_response.shape[0])
        )
    )
    free_static = _mean_static_rf(
        tuple(
            extract_static_rf(
                model,
                data.test.cone_response[index : index + 1].to(trainer.device),
                lag_steps=config.evaluation.rf_lag_steps,
            )
            for index in range(data.test.cone_response.shape[0])
        )
    )
    initialized_free_static = _mean_static_rf(
        tuple(
            extract_static_rf(
                initialized_model,
                data.test.cone_response[index : index + 1].to(trainer.device),
                lag_steps=config.evaluation.rf_lag_steps,
            )
            for index in range(data.test.cone_response.shape[0])
        )
    )
    conditional_dynamic = evaluate_dynamic_rf(
        model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
        condition_on_observed=True,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
        teacher_context_gain_envelope=teacher_envelope,
    )
    initialized_conditional_dynamic = evaluate_dynamic_rf(
        initialized_model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
        condition_on_observed=True,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
        teacher_context_gain_envelope=teacher_envelope,
    )
    free_dynamic = evaluate_dynamic_rf(
        model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
        condition_on_observed=False,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
        teacher_context_gain_envelope=teacher_envelope,
    )
    initialized_free_dynamic = evaluate_dynamic_rf(
        initialized_model,
        data.test,
        lag_steps=config.evaluation.rf_lag_steps,
        condition_on_observed=False,
        recovery_delays_ms=config.evaluation.recovery_delays_ms,
        dt_ms=data.dt_ms,
        seed=config.seed,
        teacher_kernels=teacher_dynamic,
        teacher_context_gain_envelope=teacher_envelope,
    )
    conditional_comparison = compare_dynamic_rf(
        conditional_dynamic,
        initialized_conditional_dynamic,
        seed=config.seed,
    )
    free_comparison = compare_dynamic_rf(
        free_dynamic,
        initialized_free_dynamic,
        seed=config.seed,
    )
    static_reference = _static_reference(
        conditional_static,
        teacher,
    )
    free_static_reference = _static_reference(
        free_static,
        teacher,
    )
    evidence = ResponseReportEvidence(
        conditional=conditional,
        free_running=free_running,
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
        comparison["mean_kernel_correlation"],
        comparison["mean_kernel_norm"],
    )


def _mean_static_rf(results: tuple[StaticRFResult, ...]) -> StaticRFResult:
    return StaticRFResult(
        torch.stack([result.kernels for result in results]).mean(dim=0),
        max(result.finite_difference_relative_error for result in results),
        all(result.identifiable for result in results),
    )


def _teacher_metadata(pattern: str) -> TeacherRFMetadata | None:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    return load_teacher_rf_metadata(paths[0])


__all__ = ["evaluate_and_report_response_experiment"]
