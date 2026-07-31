from __future__ import annotations

from pathlib import Path

import torch

from evaluation.response_report_schema import ResponseReportEvidence
from training.response_data import PreparedResponseData


def write_rf_artifacts(
    output: Path,
    data: PreparedResponseData,
    evidence: ResponseReportEvidence,
) -> None:
    conditional = evidence.conditional_rf
    free_running = evidence.free_running_rf
    torch.save(
        {
            "schema": "retina-rf-artifacts-v1",
            "cell_ids": data.cells.ids,
            "cone_positions_degs": torch.as_tensor(data.cone_positions_degs),
            "lag_order": "oldest_to_current",
            "conditional_static_trained": (
                conditional.static_rf.kernels.detach().cpu()
            ),
            "conditional_static_initialized": (
                conditional.initialized_static_rf.kernels.detach().cpu()
            ),
            "free_static_trained": free_running.static_rf.kernels.detach().cpu(),
            "free_static_initialized": (
                free_running.initialized_static_rf.kernels.detach().cpu()
            ),
            "conditional_dynamic_trained_low": conditional.dynamic_rf.mean_low_kernel,
            "conditional_dynamic_trained_high": conditional.dynamic_rf.mean_high_kernel,
            "conditional_dynamic_initialized_low": (
                conditional.initialized_dynamic_rf.mean_low_kernel
            ),
            "conditional_dynamic_initialized_high": (
                conditional.initialized_dynamic_rf.mean_high_kernel
            ),
            "free_dynamic_trained_low": free_running.dynamic_rf.mean_low_kernel,
            "free_dynamic_trained_high": free_running.dynamic_rf.mean_high_kernel,
            "free_dynamic_initialized_low": (
                free_running.initialized_dynamic_rf.mean_low_kernel
            ),
            "free_dynamic_initialized_high": (
                free_running.initialized_dynamic_rf.mean_high_kernel
            ),
        },
        output / "rf_artifacts.pt",
    )


__all__ = ["write_rf_artifacts"]
