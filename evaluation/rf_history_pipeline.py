from __future__ import annotations

import torch

from evaluation.response_report_schema import RFModeEvidence
from evaluation.rf_dynamic import compare_dynamic_rf, evaluate_dynamic_rf
from evaluation.rf_dynamic_metrics import context_pairs, trial_conditioned_rf
from evaluation.rf_history_contracts import RF_HISTORY_CONTRACTS, RFHistoryContract
from evaluation.rf_static import StaticRFResult
from models.response_snn import ResponseRetinaModel
from training.response_config import ResponseExperimentConfig
from training.response_data import ResponseSplit


def conditional_rf_by_history(
    model: ResponseRetinaModel,
    initialized_model: ResponseRetinaModel,
    split: ResponseSplit,
    config: ResponseExperimentConfig,
    dt_ms: float,
    seed: int,
    teacher_dynamic: tuple[torch.Tensor, torch.Tensor] | None,
    teacher_envelope: torch.Tensor | None,
    standard_history: torch.Tensor,
) -> dict[RFHistoryContract, RFModeEvidence]:
    pairs = context_pairs(split, require_complete=True)
    by_history: dict[RFHistoryContract, RFModeEvidence] = {}
    finite_difference_tolerance = config.evaluation.finite_difference_tolerance
    for history in RF_HISTORY_CONTRACTS:
        trained_static = history_static_rf(
            model,
            split,
            pairs,
            config.evaluation.rf_lag_steps,
            history,
            standard_history,
            finite_difference_tolerance,
        )
        initialized_static = history_static_rf(
            initialized_model,
            split,
            pairs,
            config.evaluation.rf_lag_steps,
            history,
            standard_history,
            finite_difference_tolerance,
        )
        trained_dynamic = evaluate_dynamic_rf(
            model,
            split,
            lag_steps=config.evaluation.rf_lag_steps,
            condition_on_observed=True,
            recovery_delays_ms=config.evaluation.recovery_delays_ms,
            dt_ms=dt_ms,
            seed=seed,
            teacher_kernels=teacher_dynamic,
            teacher_context_gain_envelope=teacher_envelope,
            history_mode=history,
            standard_history_counts=standard_history,
            finite_difference_tolerance=finite_difference_tolerance,
        )
        initialized_dynamic = evaluate_dynamic_rf(
            initialized_model,
            split,
            lag_steps=config.evaluation.rf_lag_steps,
            condition_on_observed=True,
            recovery_delays_ms=config.evaluation.recovery_delays_ms,
            dt_ms=dt_ms,
            seed=seed,
            teacher_kernels=teacher_dynamic,
            teacher_context_gain_envelope=teacher_envelope,
            history_mode=history,
            standard_history_counts=standard_history,
            finite_difference_tolerance=finite_difference_tolerance,
        )
        by_history[history] = RFModeEvidence(
            static_rf=trained_static,
            initialized_static_rf=initialized_static,
            static_reference=None,
            dynamic_rf=trained_dynamic,
            initialized_dynamic_rf=initialized_dynamic,
            dynamic_comparison=compare_dynamic_rf(
                trained_dynamic,
                initialized_dynamic,
                seed=seed,
            ),
        )
    return by_history


def history_static_rf(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    pairs: tuple[tuple[int, int], ...],
    lag_steps: int,
    history: RFHistoryContract,
    standard_history: torch.Tensor,
    finite_difference_tolerance: float | None = 0.05,
) -> StaticRFResult:
    return mean_static_rf(
        tuple(
            trial_conditioned_rf(
                model,
                split,
                index,
                lag_steps,
                history_mode=history,
                matched_history_index=pair[0],
                standard_history_counts=standard_history,
                finite_difference_tolerance=finite_difference_tolerance,
            )
            for pair in pairs
            for index in pair
        )
    )


def mean_static_rf(results: tuple[StaticRFResult, ...]) -> StaticRFResult:
    return StaticRFResult(
        torch.stack([result.kernels for result in results]).mean(dim=0),
        max(result.finite_difference_relative_error for result in results),
        all(result.identifiable for result in results),
    )


__all__ = ["conditional_rf_by_history", "history_static_rf", "mean_static_rf"]
