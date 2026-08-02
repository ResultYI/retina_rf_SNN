from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

from baselines.point_process_glm import GLMFitResult
from evaluation.parameter_audit import ParameterDelta
from evaluation.response_metrics import ResponseMetrics
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_dynamic_compare import DynamicRFComparison
from evaluation.rf_static import StaticRFResult
from evaluation.rf_history_contracts import RFHistoryContract


EvaluationSplit: TypeAlias = Literal["validation", "test"]


@dataclass(frozen=True, slots=True)
class KernelReferenceComparison:
    mean_kernel_correlation: float
    mean_kernel_norm: float


@dataclass(frozen=True, slots=True)
class RFModeEvidence:
    static_rf: StaticRFResult
    initialized_static_rf: StaticRFResult
    static_reference: KernelReferenceComparison | None
    dynamic_rf: DynamicRFResult
    initialized_dynamic_rf: DynamicRFResult
    dynamic_comparison: DynamicRFComparison
    initialized_static_reference: KernelReferenceComparison | None = None


RFModeEvidenceByHistory: TypeAlias = Mapping[RFHistoryContract, RFModeEvidence]


@dataclass(frozen=True, slots=True)
class ResponsePredictionEvidence:
    conditional: ResponseMetrics
    initialized_conditional: ResponseMetrics
    zero_history: ResponseMetrics
    shuffled_history: ResponseMetrics | None
    free_running: ResponseMetrics


@dataclass(frozen=True, slots=True)
class ResponseReportEvidence:
    response_prediction: ResponsePredictionEvidence
    parameter_deltas: tuple[ParameterDelta, ...]
    glm: GLMFitResult
    conditional_rf: RFModeEvidence
    free_running_rf: RFModeEvidence
    synthetic: bool
    checkpoint: str
    evaluation_split: EvaluationSplit = "validation"
    conditional_rf_by_history: RFModeEvidenceByHistory | None = None


__all__ = [
    "EvaluationSplit",
    "KernelReferenceComparison",
    "RFModeEvidenceByHistory",
    "RFModeEvidence",
    "ResponsePredictionEvidence",
    "ResponseReportEvidence",
]
