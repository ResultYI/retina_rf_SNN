from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from evaluation.dynamic_rf import DynamicRFUnitResult
from training.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class DynamicRFSourceSummary:
    source_id: str
    valid_record_count: int
    total_record_count: int
    trained_shape_median: float | None
    initialized_shape_median: float | None
    trained_gain_median: float | None
    initialized_gain_median: float | None
    trained_recovery_fraction_median: float | None
    initialized_recovery_fraction_median: float | None
    trained_finite_difference_valid_fraction: float
    initialized_finite_difference_valid_fraction: float
    trained_reset_error_max: float | None
    initialized_reset_error_max: float | None


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
    finite_difference_valid_fraction: float


def compare_dynamic_rf(
    trained: Sequence[DynamicRFUnitResult],
    initialized: Sequence[DynamicRFUnitResult],
    config: EvaluationConfig,
    *,
    seed: int,
) -> tuple[DynamicRFComparisonSummary, tuple[DynamicRFSourceSummary, ...]]:
    trained_by_key = {_key(row): row for row in trained}
    initialized_by_key = {_key(row): row for row in initialized}
    source_ids = sorted({row.source_id for row in (*trained, *initialized)})
    sources = tuple(
        _summarize_source(
            source_id,
            trained_by_key,
            initialized_by_key,
            config,
        )
        for source_id in source_ids
    )
    valid = tuple(source for source in sources if source.valid_record_count > 0)
    finite_difference_fraction = (
        float(np.mean([source.trained_finite_difference_valid_fraction for source in sources]))
        if sources
        else 0.0
    )
    if len(valid) < config.dynamic_rf_min_valid_sources:
        return (
            _empty_summary(
                "not_identifiable",
                len(valid),
                len(sources),
                finite_difference_fraction,
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
                finite_difference_fraction,
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
            finite_difference_valid_fraction=finite_difference_fraction,
        ),
        sources,
    )


def not_run_dynamic_rf_summary() -> DynamicRFComparisonSummary:
    return _empty_summary("not_run", 0, 0, 0.0)


def _summarize_source(
    source_id: str,
    trained: dict[tuple[str, int, int], DynamicRFUnitResult],
    initialized: dict[tuple[str, int, int], DynamicRFUnitResult],
    config: EvaluationConfig,
) -> DynamicRFSourceSummary:
    keys = sorted(
        key
        for key in trained.keys() & initialized.keys()
        if key[0] == source_id
    )
    pairs = [(trained[key], initialized[key]) for key in keys]
    quality_pairs = [
        pair
        for pair in pairs
        if _record_valid(pair[0], config) and _record_valid(pair[1], config)
    ]
    valid_pairs = []
    for trained_row, initialized_row in quality_pairs:
        trained_recovery = _recovery_fraction(trained_row, config)
        initialized_recovery = _recovery_fraction(initialized_row, config)
        if trained_recovery is not None and initialized_recovery is not None:
            valid_pairs.append(
                (
                    trained_row,
                    initialized_row,
                    trained_recovery,
                    initialized_recovery,
                )
            )
    usable = bool(valid_pairs)
    return DynamicRFSourceSummary(
        source_id=source_id,
        valid_record_count=len(valid_pairs) if usable else 0,
        total_record_count=len(pairs),
        trained_shape_median=_median(
            [row.gain_normalized_cosine_distance for row, _, _, _ in valid_pairs]
        ) if usable else None,
        initialized_shape_median=_median(
            [row.gain_normalized_cosine_distance for _, row, _, _ in valid_pairs]
        ) if usable else None,
        trained_gain_median=_median(
            [
                abs(np.log(max(row.kernel_norm_ratio, 1e-12)))
                for row, _, _, _ in valid_pairs
            ]
        ) if usable else None,
        initialized_gain_median=_median(
            [
                abs(np.log(max(row.kernel_norm_ratio, 1e-12)))
                for _, row, _, _ in valid_pairs
            ]
        ) if usable else None,
        trained_recovery_fraction_median=_median(
            [recovery for _, _, recovery, _ in valid_pairs]
        ) if usable else None,
        initialized_recovery_fraction_median=(
            _median([recovery for _, _, _, recovery in valid_pairs])
            if usable
            else None
        ),
        trained_finite_difference_valid_fraction=_fd_fraction(
            [row for row, _ in pairs], config
        ),
        initialized_finite_difference_valid_fraction=_fd_fraction(
            [row for _, row in pairs], config
        ),
        trained_reset_error_max=(
            max((row.identical_reset_kernel_error for row, _ in pairs), default=None)
        ),
        initialized_reset_error_max=(
            max((row.identical_reset_kernel_error for _, row in pairs), default=None)
        ),
    )


def _record_valid(row: DynamicRFUnitResult, config: EvaluationConfig) -> bool:
    check = row.finite_difference
    return bool(
        check.status == "local_continuous_check"
        and check.relative_error is not None
        and check.relative_error <= config.dynamic_rf_fd_max_relative_error
        and row.low_kernel_norm >= config.dynamic_rf_kernel_norm_min
        and row.high_kernel_norm >= config.dynamic_rf_kernel_norm_min
        and row.identical_reset_kernel_error <= config.dynamic_rf_reset_error_max
    )


def _recovery_fraction(
    row: DynamicRFUnitResult,
    config: EvaluationConfig,
) -> float | None:
    if not row.recovery_curve:
        return None
    ordered = sorted(row.recovery_curve)
    zero = next((distance for delay, distance in ordered if delay == 0), None)
    if zero is None or zero < config.dynamic_rf_kernel_norm_min:
        return None
    return ordered[-1][1] / max(zero, 1e-12)


def _fd_fraction(
    rows: Sequence[DynamicRFUnitResult],
    config: EvaluationConfig,
) -> float:
    if not rows:
        return 0.0
    return float(
        np.mean(
            [
                row.finite_difference.status == "local_continuous_check"
                and row.finite_difference.relative_error is not None
                and row.finite_difference.relative_error
                <= config.dynamic_rf_fd_max_relative_error
                for row in rows
            ]
        )
    )


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


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _key(row: DynamicRFUnitResult) -> tuple[str, int, int]:
    return row.source_id, row.polarity, row.unit


def _empty_summary(
    status: str,
    valid_source_count: int,
    total_source_count: int,
    finite_difference_valid_fraction: float,
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
        finite_difference_valid_fraction=finite_difference_valid_fraction,
    )


__all__ = [
    "DynamicRFComparisonSummary",
    "DynamicRFSourceSummary",
    "compare_dynamic_rf",
    "not_run_dynamic_rf_summary",
]
