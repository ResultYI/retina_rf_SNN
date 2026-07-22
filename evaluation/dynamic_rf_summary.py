from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from evaluation.dynamic_rf import DynamicRFUnitResult
from evaluation.dynamic_rf_sources import (
    DynamicRFSourceSummary,
    summarize_dynamic_rf_sources,
)
from training.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class DynamicRFComparisonSummary:
    status: str
    valid_source_count: int
    total_source_count: int
    trained_shape_median: float | None
    initialized_shape_median: float | None
    shape_delta_median: float | None
    shape_delta_bootstrap_ci: tuple[float, float] | None
    trained_gain_median: float | None
    initialized_gain_median: float | None
    gain_delta_median: float | None
    gain_delta_bootstrap_ci: tuple[float, float] | None
    trained_recovery_fraction_median: float | None
    initialized_recovery_fraction_median: float | None
    trained_finite_difference_valid_fraction: float
    initialized_finite_difference_valid_fraction: float


def compare_dynamic_rf(
    trained: Sequence[DynamicRFUnitResult],
    initialized: Sequence[DynamicRFUnitResult],
    config: EvaluationConfig,
    *,
    seed: int,
) -> tuple[DynamicRFComparisonSummary, tuple[DynamicRFSourceSummary, ...]]:
    sources = summarize_dynamic_rf_sources(trained, initialized, config)
    valid = tuple(
        source
        for source in sources
        if source.valid_record_count
        >= config.dynamic_rf_min_valid_records_per_source
        and source.valid_record_fraction
        >= config.dynamic_rf_min_valid_record_fraction_per_source
    )
    trained_fd_fraction = (
        float(np.mean([source.trained_finite_difference_valid_fraction for source in sources]))
        if sources
        else 0.0
    )
    initialized_fd_fraction = (
        float(
            np.mean(
                [source.initialized_finite_difference_valid_fraction for source in sources]
            )
        )
        if sources
        else 0.0
    )
    if len(valid) < config.dynamic_rf_min_valid_sources:
        return (
            _empty_summary(
                "not_identifiable",
                len(valid),
                len(sources),
                trained_fd_fraction,
                initialized_fd_fraction,
            ),
            sources,
        )

    trained_shape = _values(valid, "trained_shape_median")
    initialized_shape = _values(valid, "initialized_shape_median")
    trained_gain = _values(valid, "trained_gain_median")
    initialized_gain = _values(valid, "initialized_gain_median")
    trained_recovery = _values(valid, "trained_recovery_fraction_median")
    initialized_recovery = _values(valid, "initialized_recovery_fraction_median")
    if not all(
        values.size == len(valid)
        for values in (
            trained_shape,
            initialized_shape,
            trained_gain,
            initialized_gain,
            trained_recovery,
            initialized_recovery,
        )
    ):
        return (
            _empty_summary(
                "not_identifiable",
                len(valid),
                len(sources),
                trained_fd_fraction,
                initialized_fd_fraction,
            ),
            sources,
        )

    shape_delta = trained_shape - initialized_shape
    gain_delta = trained_gain - initialized_gain
    shape_ci = _paired_bootstrap_ci(
        shape_delta,
        config.dynamic_rf_bootstrap_iterations,
        seed,
    )
    gain_ci = _paired_bootstrap_ci(
        gain_delta,
        config.dynamic_rf_bootstrap_iterations,
        seed + 1,
    )
    trained_shape_median = float(np.median(trained_shape))
    initialized_shape_median = float(np.median(initialized_shape))
    trained_gain_median = float(np.median(trained_gain))
    initialized_gain_median = float(np.median(initialized_gain))
    trained_recovery_median = float(np.median(trained_recovery))
    recovery_passed = (
        trained_recovery_median <= config.dynamic_rf_recovery_fraction_max
    )
    shape_effect = trained_shape_median >= config.dynamic_rf_shape_distance_min
    gain_effect = trained_gain_median >= config.dynamic_rf_gain_log_shift_min
    initialized_effect = (
        initialized_shape_median >= config.dynamic_rf_shape_distance_min
        or initialized_gain_median >= config.dynamic_rf_gain_log_shift_min
    )
    if shape_effect and shape_ci[0] > 0.0 and recovery_passed:
        status = "learned_dynamic_rf_supported"
    elif gain_effect and gain_ci[0] > 0.0 and recovery_passed:
        status = "learned_gain_only"
    elif initialized_effect or (shape_effect and shape_ci[0] <= 0.0) or (
        gain_effect and gain_ci[0] <= 0.0
    ):
        status = "architecture_induced_context_dependence"
    else:
        status = "not_supported"
    return (
        DynamicRFComparisonSummary(
            status=status,
            valid_source_count=len(valid),
            total_source_count=len(sources),
            trained_shape_median=trained_shape_median,
            initialized_shape_median=initialized_shape_median,
            shape_delta_median=float(np.median(shape_delta)),
            shape_delta_bootstrap_ci=shape_ci,
            trained_gain_median=trained_gain_median,
            initialized_gain_median=initialized_gain_median,
            gain_delta_median=float(np.median(gain_delta)),
            gain_delta_bootstrap_ci=gain_ci,
            trained_recovery_fraction_median=trained_recovery_median,
            initialized_recovery_fraction_median=float(
                np.median(initialized_recovery)
            ),
            trained_finite_difference_valid_fraction=trained_fd_fraction,
            initialized_finite_difference_valid_fraction=initialized_fd_fraction,
        ),
        sources,
    )


def not_run_dynamic_rf_summary() -> DynamicRFComparisonSummary:
    return _empty_summary("not_run", 0, 0, 0.0, 0.0)


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, deltas.size, size=(iterations, deltas.size))
    medians = np.median(deltas[indices], axis=1)
    lower, upper = np.quantile(medians, (0.025, 0.975))
    return float(lower), float(upper)


def _values(
    sources: Sequence[DynamicRFSourceSummary],
    name: str,
) -> np.ndarray:
    return np.asarray(
        [value for source in sources if (value := getattr(source, name)) is not None],
        dtype=np.float64,
    )


def _empty_summary(
    status: str,
    valid_source_count: int,
    total_source_count: int,
    trained_finite_difference_valid_fraction: float,
    initialized_finite_difference_valid_fraction: float,
) -> DynamicRFComparisonSummary:
    return DynamicRFComparisonSummary(
        status=status,
        valid_source_count=valid_source_count,
        total_source_count=total_source_count,
        trained_shape_median=None,
        initialized_shape_median=None,
        shape_delta_median=None,
        shape_delta_bootstrap_ci=None,
        trained_gain_median=None,
        initialized_gain_median=None,
        gain_delta_median=None,
        gain_delta_bootstrap_ci=None,
        trained_recovery_fraction_median=None,
        initialized_recovery_fraction_median=None,
        trained_finite_difference_valid_fraction=(
            trained_finite_difference_valid_fraction
        ),
        initialized_finite_difference_valid_fraction=(
            initialized_finite_difference_valid_fraction
        ),
    )


__all__ = [
    "DynamicRFComparisonSummary",
    "DynamicRFSourceSummary",
    "compare_dynamic_rf",
    "not_run_dynamic_rf_summary",
]
