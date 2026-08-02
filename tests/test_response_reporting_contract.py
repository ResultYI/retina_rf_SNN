from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.point_process_glm import GLMFitResult, PointProcessGLM
from evaluation.response_metrics import ResponseMetrics
from evaluation.response_report_schema import (
    KernelReferenceComparison,
    RFModeEvidence,
    ResponsePredictionEvidence,
    ResponseReportEvidence,
)
from evaluation.response_reporting import write_response_report
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_dynamic_compare import DynamicRFComparison
from evaluation.rf_history_contracts import RF_HISTORY_CONTRACTS
from evaluation.rf_static import StaticRFResult


@pytest.mark.parametrize("teacher_status", ("teacher_mismatch", "supported"))
def test_report_separates_primary_dynamic_status_from_teacher_alignment(
    tmp_path: Path,
    teacher_status: str,
) -> None:
    # Given
    evidence = ResponseReportEvidence(
        response_prediction=_prediction(_metrics(0.4), _metrics(0.5)),
        parameter_deltas=(),
        glm=_glm(),
        conditional_rf=RFModeEvidence(
            static_rf=_static_rf(),
            initialized_static_rf=_static_rf(),
            static_reference=KernelReferenceComparison(0.7, 1.0),
            dynamic_rf=_dynamic(teacher_status, teacher=True),
            initialized_dynamic_rf=_dynamic("supported", teacher=True),
            dynamic_comparison=_comparison("supported"),
        ),
        free_running_rf=RFModeEvidence(
            static_rf=_static_rf(),
            initialized_static_rf=_static_rf(),
            static_reference=KernelReferenceComparison(0.8, 1.1),
            dynamic_rf=_dynamic("supported", teacher=True),
            initialized_dynamic_rf=_dynamic("supported", teacher=True),
            dynamic_comparison=_comparison("supported"),
        ),
        synthetic=True,
        checkpoint="checkpoint.pt",
    )
    evidence = _complete_history(evidence)

    # When
    write_response_report(tmp_path, evidence)

    # Then
    metrics = json.loads((tmp_path / "final_metrics.json").read_text())
    assert metrics["dynamic_rf"]["status"] == "supported"
    assert metrics["dynamic_rf"]["support_reason"] == "conditional_rf_supported"
    assert metrics["dynamic_rf"]["mode_agreement"] == "agree"
    assert metrics["dynamic_rf"]["conditional"]["trained"]["status"] == teacher_status
    assert metrics["dynamic_rf"]["conditional"]["trained_minus_initialized"]["status"] == "supported"
    assert metrics["dynamic_rf"]["free_running"]["trained_minus_initialized"]["status"] == "supported"
    assert metrics["dynamic_rf"]["teacher_alignment"]["status"] == teacher_status
    assert metrics["dynamic_rf"]["teacher_alignment"]["model_signed_gains"] == [0.2, -0.1]
    assert metrics["dynamic_rf"]["teacher_alignment"]["direction_agreement"] == [True, False]
    assert len(metrics["dynamic_rf"]["teacher_alignment"]["direction_agreement"]) == len(
        metrics["dynamic_rf"]["teacher_alignment"]["model_signed_gains"]
    )
    assert metrics["static_rf"]["conditional"]["reference_comparison"]["mean_kernel_correlation"] == 0.7


def test_report_omits_teacher_arrays_for_real_data(tmp_path: Path) -> None:
    # Given
    evidence = ResponseReportEvidence(
        response_prediction=_prediction(_metrics(0.4), _metrics(0.5)),
        parameter_deltas=(),
        glm=_glm(),
        conditional_rf=RFModeEvidence(
            static_rf=_static_rf(),
            initialized_static_rf=_static_rf(),
            static_reference=None,
            dynamic_rf=_dynamic("not_supported", teacher=False),
            initialized_dynamic_rf=_dynamic("not_supported", teacher=False),
            dynamic_comparison=_comparison("not_supported"),
        ),
        free_running_rf=RFModeEvidence(
            static_rf=_static_rf(),
            initialized_static_rf=_static_rf(),
            static_reference=None,
            dynamic_rf=_dynamic("not_supported", teacher=False),
            initialized_dynamic_rf=_dynamic("not_supported", teacher=False),
            dynamic_comparison=_comparison("not_supported"),
        ),
        synthetic=False,
        checkpoint="checkpoint.pt",
    )
    evidence = _complete_history(evidence)

    # When
    write_response_report(tmp_path, evidence)

    # Then
    metrics = json.loads((tmp_path / "final_metrics.json").read_text())
    assert "conditional" in metrics["static_rf"]
    assert "free_running" in metrics["static_rf"]
    assert metrics["dynamic_rf"]["teacher_alignment"] == {"available": False}
    assert "teacher_model_signed_gains" not in json.dumps(metrics)


def test_report_writes_strict_json_nulls_when_metrics_are_nonfinite(
    tmp_path: Path,
) -> None:
    # Given
    evidence = ResponseReportEvidence(
        response_prediction=_prediction(
            ResponseMetrics(
                float("nan"),
                np.float64(float("inf")),
                torch.tensor(float("-inf")),
                0.3,
                0.4,
                (0.5, np.float32(float("nan")), torch.tensor(float("inf"))),
            ),
            _metrics(0.5),
        ),
        parameter_deltas=(),
        glm=_glm(),
        conditional_rf=RFModeEvidence(
            static_rf=StaticRFResult(torch.ones(2, 3, 1), float("inf"), True),
            initialized_static_rf=_static_rf(),
            static_reference=KernelReferenceComparison(np.float32(0.7), float("-inf")),
            dynamic_rf=_dynamic("supported", teacher=True),
            initialized_dynamic_rf=_dynamic("supported", teacher=True),
            dynamic_comparison=DynamicRFComparison(
                pair_count=3,
                learned_shape_delta=0.1,
                learned_gain_delta=0.1,
                shape_delta_ci=(0.05, np.float64(float("inf"))),
                gain_delta_ci=(torch.tensor(float("-inf")), 0.2),
                status="supported",
                teacher_primary_error_delta=float("nan"),
                teacher_primary_error_delta_ci=(0.1, 0.3),
                teacher_recovery_error_delta=0.1,
                teacher_recovery_error_delta_ci=(0.05, 0.2),
            ),
        ),
        free_running_rf=RFModeEvidence(
            static_rf=_static_rf(),
            initialized_static_rf=_static_rf(),
            static_reference=None,
            dynamic_rf=_dynamic("supported", teacher=False),
            initialized_dynamic_rf=_dynamic("supported", teacher=False),
            dynamic_comparison=_comparison("supported"),
        ),
        synthetic=True,
        checkpoint="checkpoint.pt",
    )
    evidence = _complete_history(evidence)

    # When
    write_response_report(tmp_path, evidence)

    # Then
    text = (tmp_path / "final_metrics.json").read_text(encoding="utf-8")
    metrics = json.loads(
        text,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    assert "NaN" not in text
    assert "Infinity" not in text
    assert metrics["response_prediction"]["conditional"]["nll"] is None
    assert metrics["response_prediction"]["conditional"]["micro_bits_per_spike"] is None
    assert metrics["response_prediction"]["conditional"]["macro_bits_per_spike"] is None
    assert metrics["response_prediction"]["conditional"]["per_cell_nll"] == [
        0.5,
        None,
        None,
    ]
    assert metrics["static_rf"]["conditional"]["trained"]["finite_difference_relative_error"] is None
    assert metrics["static_rf"]["conditional"]["reference_comparison"] == {
        "mean_kernel_correlation": 0.699999988079071,
        "mean_kernel_norm": None,
    }
    comparison = metrics["dynamic_rf"]["conditional"]["trained_minus_initialized"]
    assert comparison["shape_delta_ci"] == [0.05, None]
    assert comparison["gain_delta_ci"] == [None, 0.2]
    assert comparison["teacher_primary_error_delta"] is None


def _metrics(nll: float) -> ResponseMetrics:
    return ResponseMetrics(nll, 0.1, 0.2, 0.3, 0.4, (nll,))


def _prediction(
    conditional: ResponseMetrics,
    free_running: ResponseMetrics,
) -> ResponsePredictionEvidence:
    return ResponsePredictionEvidence(
        conditional=conditional,
        initialized_conditional=conditional,
        zero_history=conditional,
        shuffled_history=conditional,
        free_running=free_running,
    )


def _glm() -> GLMFitResult:
    metrics = _metrics(0.6)
    return GLMFitResult(PointProcessGLM(1, 1, 1), metrics, metrics, 1)


def _static_rf() -> StaticRFResult:
    return StaticRFResult(torch.ones(2, 3, 1), 0.01, True)


def _dynamic(status: str, *, teacher: bool) -> DynamicRFResult:
    return DynamicRFResult(
        pair_count=3,
        mean_shape_distance=0.2,
        mean_log_gain_shift=0.2,
        shape_distance_ci=(0.1, 0.3),
        gain_shift_ci=(0.1, 0.3),
        reset_shape_distance=0.0,
        recovery_shape_distances=(0.2, 0.1),
        finite_difference_relative_error=0.01,
        teacher_shape_error=0.1 if teacher else None,
        teacher_gain_error=0.2 if teacher else None,
        per_source_shape_distances=(0.2, 0.2, 0.2),
        per_source_gain_shifts=(0.2, 0.2, 0.2),
        status=status,
        teacher_primary_errors=(0.1, 0.1, 0.1) if teacher else (),
        teacher_recovery_errors=(0.05, 0.05, 0.05) if teacher else (),
        teacher_gain_direction_agreement=(True, False) if teacher else (),
        teacher_model_signed_gains=(0.2, -0.1) if teacher else (),
        teacher_reference_signed_gains=(0.3, 0.1) if teacher else (),
        teacher_signed_gain_correlation=-0.5 if teacher else None,
        teacher_delta_cosine_distance=0.4 if teacher else None,
    )


def _comparison(status: str) -> DynamicRFComparison:
    return DynamicRFComparison(
        pair_count=3,
        learned_shape_delta=0.1,
        learned_gain_delta=0.1,
        shape_delta_ci=(0.05, 0.2),
        gain_delta_ci=(0.05, 0.2),
        status=status,
        teacher_primary_error_delta=0.2,
        teacher_primary_error_delta_ci=(0.1, 0.3),
        teacher_recovery_error_delta=0.1,
        teacher_recovery_error_delta_ci=(0.05, 0.2),
    )


def _complete_history(evidence: ResponseReportEvidence) -> ResponseReportEvidence:
    return replace(
        evidence,
        conditional_rf_by_history={
            key: evidence.conditional_rf for key in RF_HISTORY_CONTRACTS
        },
    )
