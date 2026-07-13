from __future__ import annotations

from collections.abc import Sequence

MetricRow = dict[str, str]
_METADATA_FIELDS = frozenset({"split", "epoch", "step"})


class EpochMetricsError(ValueError):
    pass


def weighted_mean_row(rows: Sequence[tuple[int, MetricRow]]) -> MetricRow:
    if not rows:
        raise EpochMetricsError("Cannot average an empty set of metric rows")
    total_samples = sum(sample_count for sample_count, _ in rows)
    if total_samples < 1:
        raise EpochMetricsError("Metric rows must have positive sample counts")
    first = rows[0][1]
    result = {
        field: first[field]
        for field in _METADATA_FIELDS
        if field in first
    }
    for field in first:
        if field not in _METADATA_FIELDS:
            value = sum(
                sample_count * float(row[field])
                for sample_count, row in rows
            ) / total_samples
            result[field] = f"{value:.8g}"
    return result
