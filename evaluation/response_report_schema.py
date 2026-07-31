from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from baselines.point_process_glm import GLMFitResult
from evaluation.response_metrics import ResponseMetrics
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_dynamic_compare import DynamicRFComparison
from evaluation.rf_static import StaticRFResult


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


@dataclass(frozen=True, slots=True)
class ResponseReportEvidence:
    conditional: ResponseMetrics
    free_running: ResponseMetrics
    glm: GLMFitResult
    conditional_rf: RFModeEvidence
    free_running_rf: RFModeEvidence
    synthetic: bool
    checkpoint: str
    evaluation_split: EvaluationSplit = "validation"


__all__ = [
    "EvaluationSplit",
    "KernelReferenceComparison",
    "RFModeEvidence",
    "ResponseReportEvidence",
]
