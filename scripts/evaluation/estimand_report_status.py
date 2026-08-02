from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

from evaluation.rf_history_contracts import RF_HISTORY_CONTRACTS

MIN_SELF_FIT_KERNEL_CORRELATION: Final = 0.0
MAX_SELF_FIT_KERNEL_ERROR: Final = 1.0


def parameter_recovery_status(
    controlled_results: Mapping[str, Mapping[str, Sequence[float] | str]],
    *,
    per_cell_kernel_correlation: Sequence[float],
    per_cell_kernel_error: Sequence[float],
) -> str:
    controlled_status = controlled_history_status(controlled_results)
    if controlled_status != "supported":
        return controlled_status
    if not all(
        math.isfinite(value)
        for value in (*per_cell_kernel_correlation, *per_cell_kernel_error)
    ):
        return "not_supported"
    self_fit_supported = all(
        value > MIN_SELF_FIT_KERNEL_CORRELATION
        for value in per_cell_kernel_correlation
    ) and all(
        0.0 <= value < MAX_SELF_FIT_KERNEL_ERROR
        for value in per_cell_kernel_error
    )
    return "supported" if self_fit_supported else "not_supported"


def controlled_history_status(
    controlled_results: Mapping[str, Mapping[str, Sequence[float] | str]],
) -> str:
    if set(controlled_results) != set(RF_HISTORY_CONTRACTS):
        return "not_identifiable"
    for contract in RF_HISTORY_CONTRACTS:
        result = controlled_results[contract]
        metric_values = tuple(
            result.get(key)
            for key in (
                "conditional_logit_rf_gain",
                "probability_rf_gain",
                "response_gain",
            )
        )
        if not all(isinstance(values, Sequence) and values for values in metric_values):
            return "not_identifiable"
        if not all(
            math.isfinite(float(value))
            for values in metric_values
            for value in values
        ):
            return "not_identifiable"
        logit_gain = metric_values[0]
        if not any(abs(float(value)) > 1e-8 for value in logit_gain):
            return "not_supported"
    return "supported"


def exploratory_type_effects(parameter):
    self_fit = parameter.get("self_fit_power")
    if not isinstance(self_fit, dict):
        return {"available": False}
    return {
        "available": True,
        "teacher_signed_gains": self_fit["teacher_context_gains"],
        "group_effect_recovery": parameter["group_effect_recovery"],
        "signed_gain_status": "exploratory",
    }


def overall_status(parameter_status: str, observational_status: str) -> str:
    if observational_status != "supported":
        return observational_status
    if parameter_status == "not_identifiable":
        return "observational_only"
    return parameter_status
