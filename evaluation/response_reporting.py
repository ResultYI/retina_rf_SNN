from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

from evaluation.response_report_schema import (
    KernelReferenceComparison,
    RFModeEvidence,
    ResponseReportEvidence,
)
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_history_contracts import require_exact_history_contracts
from evaluation.rf_static import StaticRFResult

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap: TypeAlias = dict[str, JsonValue]
ReportValue: TypeAlias = (
    JsonValue
    | np.generic
    | np.ndarray
    | torch.Tensor
    | tuple["ReportValue", ...]
)


class ResponseReportSerializationError(TypeError):
    pass


def write_response_report(
    output_dir: str | Path,
    evidence: ResponseReportEvidence,
) -> None:
    output = Path(output_dir)
    prediction = evidence.response_prediction
    metrics = {
        "evidence_kind": (
            "synthetic_method_validation" if evidence.synthetic else "real_recording"
        ),
        "checkpoint": evidence.checkpoint,
        "evaluation_split": evidence.evaluation_split,
        "parameter_delta_audit": [
            asdict(parameter_delta) for parameter_delta in evidence.parameter_deltas
        ],
        "response_prediction": {
            "conditional": asdict(prediction.conditional),
            "initialized_conditional": asdict(prediction.initialized_conditional),
            "history_diagnostic": {
                "observed": asdict(prediction.conditional),
                "zero": asdict(prediction.zero_history),
                "shuffled": (
                    None
                    if prediction.shuffled_history is None
                    else asdict(prediction.shuffled_history)
                ),
                "observed_minus_zero_nll": (
                    prediction.conditional.nll - prediction.zero_history.nll
                ),
            },
            "free_running": asdict(prediction.free_running),
            "glm_validation": asdict(evidence.glm.validation_metrics),
            "glm_test": (
                None
                if evidence.glm.test_metrics is None
                else asdict(evidence.glm.test_metrics)
            ),
            "glm_evaluation": asdict(evidence.glm.evaluation_metrics),
            "glm_best_step": evidence.glm.best_step,
        },
        "static_rf": {
            "conditional": _static_block(evidence.conditional_rf),
            "free_running": _static_block(evidence.free_running_rf),
            "by_history": _static_by_history(evidence),
        },
        "dynamic_rf": _dynamic_block(evidence),
    }
    (output / "final_metrics.json").write_text(
        json.dumps(
            _json_native(metrics),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output / "parameter_delta.json").write_text(
        json.dumps(
            _json_native(metrics["parameter_delta_audit"]),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output / "run_status.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "synthetic": evidence.synthetic,
                "evaluation_split": evidence.evaluation_split,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output / "final_report_zh.md").write_text(
        _markdown_report(evidence),
        encoding="utf-8",
    )


def _json_native(value: ReportValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _json_native(value.item())
        return _json_native(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool | int | str) or value is None:
        return value
    raise ResponseReportSerializationError(
        f"Unsupported report value type: {type(value).__name__}"
    )


def _static_block(evidence: RFModeEvidence) -> JsonMap:
    value: JsonMap = {
        "trained": _static_result(evidence.static_rf),
        "initialized": _static_result(evidence.initialized_static_rf),
    }
    if evidence.static_reference is not None:
        value["reference_comparison"] = _kernel_reference(evidence.static_reference)
    if evidence.initialized_static_reference is not None:
        value["initialized_reference_comparison"] = _kernel_reference(
            evidence.initialized_static_reference
        )
    return value


def _static_result(result: StaticRFResult) -> JsonMap:
    return {
        "identifiable": result.identifiable,
        "finite_difference_relative_error": result.finite_difference_relative_error,
        "kernel_shape": list(result.kernels.shape),
        "per_cell_kernel_norm": result.kernels.norm(dim=(1, 2)),
    }


def _dynamic_block(evidence: ResponseReportEvidence) -> JsonMap:
    conditional = evidence.conditional_rf.dynamic_comparison.status
    free_running = evidence.free_running_rf.dynamic_comparison.status
    return {
        "status": conditional,
        "support_reason": _support_reason(conditional, evidence.synthetic),
        "mode_agreement": _mode_agreement(conditional, free_running),
        "conditional": _dynamic_mode(evidence.conditional_rf),
        "free_running": _dynamic_mode(evidence.free_running_rf),
        "by_history": _dynamic_by_history(evidence),
        "teacher_alignment": _teacher_alignment(evidence.conditional_rf.dynamic_rf),
    }


def _dynamic_mode(evidence: RFModeEvidence) -> JsonMap:
    return {
        "trained": _dynamic_result(evidence.dynamic_rf),
        "initialized": _dynamic_result(evidence.initialized_dynamic_rf),
        "trained_minus_initialized": asdict(evidence.dynamic_comparison),
    }


def _dynamic_result(result: DynamicRFResult) -> JsonMap:
    value = asdict(result)
    del value["mean_low_kernel"]
    del value["mean_high_kernel"]
    value["per_cell_signed_log_gain_shift"] = _dynamic_cell_gain_shift(result)
    if result.teacher_model_signed_gains:
        return value
    for key in (
        "teacher_primary_errors",
        "teacher_recovery_errors",
        "teacher_gain_direction_agreement",
        "teacher_model_signed_gains",
        "teacher_reference_signed_gains",
        "teacher_signed_gain_correlation",
        "teacher_delta_cosine_distance",
    ):
        del value[key]
    return value


def _static_by_history(evidence: ResponseReportEvidence) -> JsonMap:
    by_history = require_exact_history_contracts(evidence.conditional_rf_by_history)
    return {key: _static_block(value) for key, value in by_history.items()}


def _dynamic_by_history(evidence: ResponseReportEvidence) -> JsonMap:
    by_history = require_exact_history_contracts(evidence.conditional_rf_by_history)
    return {key: _dynamic_mode(value) for key, value in by_history.items()}


def _dynamic_cell_gain_shift(result: DynamicRFResult) -> torch.Tensor:
    if result.mean_low_kernel is None or result.mean_high_kernel is None:
        return torch.empty(0)
    return (
        (result.mean_high_kernel.norm(dim=(1, 2)) + 1e-8).log()
        - (result.mean_low_kernel.norm(dim=(1, 2)) + 1e-8).log()
    )


def _teacher_alignment(result: DynamicRFResult) -> JsonMap:
    if not result.teacher_model_signed_gains:
        return {"available": False}
    return {
        "available": True,
        "status": result.status,
        "model_signed_gains": list(result.teacher_model_signed_gains),
        "reference_signed_gains": list(result.teacher_reference_signed_gains),
        "direction_agreement": list(result.teacher_gain_direction_agreement),
        "signed_gain_correlation": result.teacher_signed_gain_correlation,
        "delta_cosine_distance": result.teacher_delta_cosine_distance,
        "primary_errors": list(result.teacher_primary_errors),
        "recovery_errors": list(result.teacher_recovery_errors),
    }


def _kernel_reference(reference: KernelReferenceComparison) -> JsonMap:
    return {
        "mean_kernel_correlation": reference.mean_kernel_correlation,
        "mean_kernel_norm": reference.mean_kernel_norm,
    }


def _support_reason(status: str, synthetic: bool) -> str:
    if synthetic and status == "supported":
        return "conditional_rf_supported"
    return status


def _mode_agreement(conditional: str, free_running: str) -> str:
    if "not_identifiable" in (conditional, free_running):
        return "not_identifiable"
    return "agree" if conditional == free_running else "mismatch"


def _markdown_report(evidence: ResponseReportEvidence) -> str:
    title = "合成方法验证" if evidence.synthetic else "真实 RGC 响应拟合"
    dynamic = _dynamic_block(evidence)
    prediction = evidence.response_prediction
    report = (
        "# RGC 响应拟合报告\n\n"
        f"证据类型：{title}\n\n"
        f"- 评估数据集：{evidence.evaluation_split}\n"
        f"- 条件评估 NLL：{prediction.conditional.nll:.6f}\n"
        f"- 自由运行评估 NLL：{prediction.free_running.nll:.6f}\n"
        f"- 静态 GLM 评估 NLL：{evidence.glm.evaluation_metrics.nll:.6f}\n"
        f"- Conditional Dynamic RF：{dynamic['status']}\n"
        f"- Free-running Dynamic RF：{evidence.free_running_rf.dynamic_comparison.status}\n"
        f"- Mode agreement：{dynamic['mode_agreement']}\n"
        f"- Support reason：{dynamic['support_reason']}\n\n"
        "主结论只使用 conditional RF；free-running RF 仅作为确定性生成诊断。\n"
    )
    if evidence.synthetic:
        report += "\n本报告仅验证方法链路，不构成真实视网膜生理结论。\n"
    return report


__all__ = ["write_response_report"]
