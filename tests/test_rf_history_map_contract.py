from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.point_process_glm import GLMFitResult, PointProcessGLM
from data.input_identity import synthetic_input_identity
from data.rgc_response import CellMetadata, ResponseTargetKind
from evaluation.response_metrics import ResponseMetrics
from evaluation.response_report_schema import (
    RFModeEvidence,
    ResponsePredictionEvidence,
    ResponseReportEvidence,
)
from evaluation.response_reporting import write_response_report
from evaluation.rf_artifacts import write_rf_artifacts
from evaluation.rf_dynamic import DynamicRFResult
from evaluation.rf_dynamic_compare import DynamicRFComparison
from evaluation.rf_history_contracts import RFHistoryContractError
from evaluation.rf_static import StaticRFResult
from training.response_data import PreparedResponseData, ResponseSplit


def test_report_rejects_incomplete_conditional_history_map(tmp_path: Path) -> None:
    evidence = _incomplete_history(_minimal_evidence())

    with pytest.raises(RFHistoryContractError, match="exact history keys"):
        write_response_report(tmp_path, evidence)


def test_rf_artifacts_reject_incomplete_conditional_history_map(
    tmp_path: Path,
) -> None:
    evidence = _incomplete_history(_minimal_evidence())

    with pytest.raises(RFHistoryContractError, match="exact history keys"):
        write_rf_artifacts(tmp_path, _prepared_data(), evidence)


def _minimal_evidence() -> ResponseReportEvidence:
    metrics = _metrics(0.4)
    return ResponseReportEvidence(
        response_prediction=ResponsePredictionEvidence(
            metrics,
            metrics,
            metrics,
            metrics,
            metrics,
        ),
        parameter_deltas=(),
        glm=GLMFitResult(PointProcessGLM(1, 1, 1), metrics, None, 1),
        conditional_rf=_mode(),
        free_running_rf=_mode(),
        synthetic=False,
        checkpoint="checkpoint.pt",
    )


def _mode() -> RFModeEvidence:
    return RFModeEvidence(
        static_rf=_static_rf(),
        initialized_static_rf=_static_rf(),
        static_reference=None,
        dynamic_rf=_dynamic(),
        initialized_dynamic_rf=_dynamic(),
        dynamic_comparison=_comparison(),
    )


def _incomplete_history(evidence: ResponseReportEvidence) -> ResponseReportEvidence:
    return replace(
        evidence,
        conditional_rf_by_history={"matched_observed": evidence.conditional_rf},
    )


def _metrics(nll: float) -> ResponseMetrics:
    return ResponseMetrics(nll, 0.1, 0.2, 0.3, 0.4, (nll,))


def _static_rf() -> StaticRFResult:
    return StaticRFResult(torch.ones(2, 3, 1), 0.01, True)


def _dynamic() -> DynamicRFResult:
    return DynamicRFResult(
        3,
        0.2,
        0.2,
        (0.1, 0.3),
        (0.1, 0.3),
        0.0,
        (0.2, 0.1),
        0.01,
        None,
        None,
        (0.2, 0.2, 0.2),
        (0.2, 0.2, 0.2),
        "not_supported",
        mean_low_kernel=torch.ones(2, 3, 1),
        mean_high_kernel=torch.full((2, 3, 1), 2.0),
    )


def _comparison() -> DynamicRFComparison:
    return DynamicRFComparison(
        3,
        0.1,
        0.1,
        (0.05, 0.2),
        (0.05, 0.2),
        "not_supported",
    )


def _prepared_data() -> PreparedResponseData:
    split = ResponseSplit(
        cone_response=torch.ones(6, 4, 1),
        spike_counts=torch.zeros(6, 1, 4, 1),
        valid_mask=torch.ones(6, 1, 4, 1, dtype=torch.bool),
        source_ids=("a", "a", "b", "b", "c", "c"),
        context_ids=("low", "high", "low", "high", "low", "high"),
    )
    cells = CellMetadata(
        ids=("cell",),
        type_ids=("midget",),
        polarities=np.zeros(1, dtype=np.int64),
        positions_degs=np.zeros((1, 2), dtype=np.float32),
        eccentricities_deg=np.ones(1, dtype=np.float32),
    )
    return PreparedResponseData(
        train=split,
        validation=split,
        test=split,
        cells=cells,
        cone_positions_degs=np.zeros((1, 2), dtype=np.float32),
        time_axis_seconds=np.arange(4, dtype=np.float64) * 0.005,
        target_kind=ResponseTargetKind.BERNOULLI,
        normalization_mean=np.zeros(1, dtype=np.float32),
        normalization_std=np.ones(1, dtype=np.float32),
        fingerprint="fingerprint",
        input_identity=synthetic_input_identity(1, ("source",)),
    )
