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
    valid_record_fraction: float
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


def summarize_dynamic_rf_sources(
    trained: Sequence[DynamicRFUnitResult],
    initialized: Sequence[DynamicRFUnitResult],
    config: EvaluationConfig,
) -> tuple[DynamicRFSourceSummary, ...]:
    trained_by_key = {_key(row): row for row in trained}
    initialized_by_key = {_key(row): row for row in initialized}
    source_ids = sorted({row.source_id for row in (*trained, *initialized)})
    return tuple(
        _summarize_source(
            source_id,
            trained_by_key,
            initialized_by_key,
            config,
        )
        for source_id in source_ids
    )


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
    valid_count = len(valid_pairs)
    total_count = len(pairs)
    return DynamicRFSourceSummary(
        source_id=source_id,
        valid_record_count=valid_count,
        total_record_count=total_count,
        valid_record_fraction=valid_count / total_count if total_count else 0.0,
        trained_shape_median=_median(
            [row.gain_normalized_cosine_distance for row, _, _, _ in valid_pairs]
        ),
        initialized_shape_median=_median(
            [row.gain_normalized_cosine_distance for _, row, _, _ in valid_pairs]
        ),
        trained_gain_median=_median(
            [
                abs(np.log(max(row.kernel_norm_ratio, 1e-12)))
                for row, _, _, _ in valid_pairs
            ]
        ),
        initialized_gain_median=_median(
            [
                abs(np.log(max(row.kernel_norm_ratio, 1e-12)))
                for _, row, _, _ in valid_pairs
            ]
        ),
        trained_recovery_fraction_median=_median(
            [recovery for _, _, recovery, _ in valid_pairs]
        ),
        initialized_recovery_fraction_median=_median(
            [recovery for _, _, _, recovery in valid_pairs]
        ),
        trained_finite_difference_valid_fraction=_fd_fraction(
            [row for row, _ in pairs], config
        ),
        initialized_finite_difference_valid_fraction=_fd_fraction(
            [row for _, row in pairs], config
        ),
        trained_reset_error_max=max(
            (row.identical_reset_kernel_error for row, _ in pairs), default=None
        ),
        initialized_reset_error_max=max(
            (row.identical_reset_kernel_error for _, row in pairs), default=None
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


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _key(row: DynamicRFUnitResult) -> tuple[str, int, int]:
    return row.source_id, row.polarity, row.unit


__all__ = ["DynamicRFSourceSummary", "summarize_dynamic_rf_sources"]
