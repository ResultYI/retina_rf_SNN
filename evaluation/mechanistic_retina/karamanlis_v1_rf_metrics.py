from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.karamanlis_locality_graph import (
    RFLocalityError,
    RFMapGrid,
    extract_rf_spatial_extent,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_validation_math import cosine

RFComparisonRow = dict[str, JsonValue]


class RFMetricError(ValueError):
    pass


def compare_population_rfs(
    *,
    cell_ids: Sequence[str],
    cell_types: Sequence[str],
    polarities: Sequence[str],
    model_spatial: np.ndarray,
    model_temporal: np.ndarray,
    empirical_spatial: np.ndarray,
    empirical_temporal: np.ndarray,
    empirical_even_temporal: np.ndarray,
    empirical_odd_temporal: np.ndarray,
    model_lag_ms: np.ndarray,
    empirical_lag_ms: np.ndarray,
    grid: RFMapGrid,
) -> list[RFComparisonRow]:
    if any(polarity not in ("ON", "OFF") for polarity in polarities):
        raise RFMetricError("RF comparison polarity must be ON or OFF")
    rows = []
    for index, cell_id in enumerate(cell_ids):
        rows.append(
            _cell_metrics(
                cell_id=str(cell_id),
                cell_type=str(cell_types[index]),
                polarity=str(polarities[index]),
                model_spatial=model_spatial[index],
                model_temporal=model_temporal[index],
                empirical_spatial=empirical_spatial[index],
                empirical_temporal=empirical_temporal[index],
                empirical_even_temporal=empirical_even_temporal[index],
                empirical_odd_temporal=empirical_odd_temporal[index],
                model_lag_ms=model_lag_ms,
                empirical_lag_ms=empirical_lag_ms,
                grid=grid,
            )
        )
    return rows


def summarize_rf_rows(rows: Sequence[RFComparisonRow]) -> dict[str, JsonValue]:
    return {
        "population": _summarize_group(rows),
        "by_cell_class": {
            label: _summarize_group(
                tuple(row for row in rows if row["cell_class"] == label)
            )
            for label in ("ON midget", "OFF midget", "ON parasol", "OFF parasol")
        },
    }


def _cell_metrics(
    *,
    cell_id: str,
    cell_type: str,
    polarity: str,
    model_spatial: np.ndarray,
    model_temporal: np.ndarray,
    empirical_spatial: np.ndarray,
    empirical_temporal: np.ndarray,
    empirical_even_temporal: np.ndarray,
    empirical_odd_temporal: np.ndarray,
    model_lag_ms: np.ndarray,
    empirical_lag_ms: np.ndarray,
    grid: RFMapGrid,
) -> RFComparisonRow:
    expected_sign = 1 if polarity == "ON" else -1
    model_peak_index = int(np.argmax(np.abs(model_temporal)))
    empirical_peak_index = int(np.argmax(np.abs(empirical_temporal)))
    matched_count = model_temporal.size
    empirical_matched = empirical_temporal[:matched_count]
    empirical_matched_peak = int(np.argmax(np.abs(empirical_matched)))
    split_cosine = cosine(empirical_even_temporal, empirical_odd_temporal)
    model_sign = int(np.sign(model_temporal[model_peak_index]))
    empirical_sign = int(np.sign(empirical_temporal[empirical_peak_index]))
    temporal_reliable = bool(
        np.isfinite(split_cosine)
        and split_cosine >= 0.5
        and model_sign != 0
        and empirical_sign != 0
        and empirical_lag_ms[empirical_peak_index]
        <= model_lag_ms[-1] + 0.5 * np.diff(model_lag_ms).mean()
    )
    tail_energy = float(np.square(empirical_temporal[matched_count:]).sum())
    full_energy = float(np.square(empirical_temporal).sum())
    row: RFComparisonRow = {
        "cell_id": cell_id,
        "cell_type": cell_type,
        "polarity": polarity,
        "cell_class": f"{polarity} {cell_type}",
        "spatial_cosine": cosine(model_spatial, empirical_spatial),
        "model_temporal_peak_sign": model_sign,
        "empirical_temporal_peak_sign": empirical_sign,
        "polarity_consistent": bool(model_sign == empirical_sign == expected_sign),
        "temporal_split_half_cosine": split_cosine,
        "temporal_reliable": temporal_reliable,
        "temporal_cosine_matched_window": cosine(model_temporal, empirical_matched),
        "model_peak_latency_ms": float(model_lag_ms[model_peak_index]),
        "empirical_peak_latency_matched_ms": float(
            empirical_lag_ms[empirical_matched_peak]
        ),
        "empirical_peak_latency_full_ms": float(empirical_lag_ms[empirical_peak_index]),
        "peak_latency_absolute_error_ms": float(
            abs(
                model_lag_ms[model_peak_index]
                - empirical_lag_ms[empirical_matched_peak]
            )
        ),
        "empirical_temporal_tail_energy_fraction": (
            tail_energy / full_energy if full_energy > 0 else float("nan")
        ),
    }
    try:
        model_extent = extract_rf_spatial_extent(model_spatial, grid)
        empirical_extent = extract_rf_spatial_extent(empirical_spatial, grid)
    except RFLocalityError as error:
        row |= {"static_reliable": False, "static_failure": str(error)}
        return row
    intersection = np.logical_and(
        model_extent.support_mask, empirical_extent.support_mask
    ).sum()
    union = np.logical_or(
        model_extent.support_mask, empirical_extent.support_mask
    ).sum()
    row |= {
        "static_reliable": True,
        "static_failure": None,
        "center_distance_um": float(
            np.linalg.norm(model_extent.center_um - empirical_extent.center_um)
        ),
        "model_center_um": model_extent.center_um.tolist(),
        "empirical_center_um": empirical_extent.center_um.tolist(),
        "model_equivalent_radius_um": model_extent.equivalent_radius_um,
        "empirical_equivalent_radius_um": empirical_extent.equivalent_radius_um,
        "radius_ratio": (
            model_extent.equivalent_radius_um / empirical_extent.equivalent_radius_um
        ),
        "radius_absolute_error_um": abs(
            model_extent.equivalent_radius_um - empirical_extent.equivalent_radius_um
        ),
        "support_iou": float(intersection / union) if union else float("nan"),
    }
    return row


def _summarize_group(rows: Sequence[RFComparisonRow]) -> dict[str, JsonValue]:
    static = tuple(row for row in rows if row.get("static_reliable") is True)
    temporal = tuple(row for row in rows if row.get("temporal_reliable") is True)
    return {
        "cell_count": len(rows),
        "static_reliable_count": len(static),
        "temporal_reliable_count": len(temporal),
        "center_distance_um": _distribution(static, "center_distance_um"),
        "spatial_cosine": _distribution(static, "spatial_cosine"),
        "support_iou": _distribution(static, "support_iou"),
        "radius_ratio": _distribution(static, "radius_ratio"),
        "radius_absolute_error_um": _distribution(static, "radius_absolute_error_um"),
        "polarity_consistent_count": sum(
            row["polarity_consistent"] is True for row in rows
        ),
        "temporal_split_half_cosine": _distribution(rows, "temporal_split_half_cosine"),
        "temporal_cosine_matched_window": _distribution(
            temporal, "temporal_cosine_matched_window"
        ),
        "peak_latency_absolute_error_ms": _distribution(
            temporal, "peak_latency_absolute_error_ms"
        ),
        "empirical_temporal_tail_energy_fraction": _distribution(
            rows, "empirical_temporal_tail_energy_fraction"
        ),
    }


def _distribution(rows: Sequence[RFComparisonRow], key: str) -> dict[str, float] | None:
    values = np.asarray(
        tuple(float(row[key]) for row in rows if key in row), dtype=np.float64
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


__all__ = ["RFMetricError", "compare_population_rfs", "summarize_rf_rows"]
