from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.karamanlis_v1_ac_metrics import (
    CellPerturbationMetrics,
)


@dataclass(frozen=True, slots=True)
class PerturbationReportRequest:
    cell_ids: Sequence[str]
    cell_types: Sequence[str]
    polarities: Sequence[str]
    metrics: CellPerturbationMetrics


class PerturbationReportError(ValueError):
    pass


def build_perturbation_summary(
    request: PerturbationReportRequest,
) -> Mapping[str, JsonValue]:
    indices = torch.arange(len(request.cell_ids))
    classes = {
        label: _group_summary(request.metrics, _class_indices(request, label))
        for label in ("ON midget", "OFF midget", "ON parasol", "OFF parasol")
    }
    return {
        "population": _group_summary(request.metrics, indices),
        "by_cell_class": classes,
        "per_cell": [
            {
                "cell_id": str(cell_id),
                "cell_class": f"{request.polarities[index]} {request.cell_types[index]}",
                **_cell_summary(request.metrics, index),
            }
            for index, cell_id in enumerate(request.cell_ids)
        ],
    }


def _class_indices(
    request: PerturbationReportRequest,
    label: str,
) -> torch.Tensor:
    indices = torch.as_tensor(
        tuple(
            index
            for index, (polarity, cell_type) in enumerate(
                zip(request.polarities, request.cell_types, strict=True)
            )
            if f"{polarity} {cell_type}" == label
        ),
        dtype=torch.long,
    )
    if indices.numel() == 0:
        raise PerturbationReportError(f"cell class is absent: {label}")
    return indices


def _group_summary(
    metrics: CellPerturbationMetrics,
    indices: torch.Tensor,
) -> Mapping[str, JsonValue]:
    return {
        "cell_count": int(indices.numel()),
        "response_change": {
            "mean_absolute_logit": _mean(metrics.mean_absolute_logit_change, indices),
            "mean_logit": _mean(metrics.mean_logit_change, indices),
            "mean_absolute_probability": _mean(
                metrics.mean_absolute_probability_change, indices
            ),
            "mean_probability": _mean(metrics.mean_probability_change, indices),
        },
        "response_peak": {
            "normal_logit_magnitude": _mean(
                metrics.normal_logit_peak_magnitude, indices
            ),
            "clamped_logit_magnitude": _mean(
                metrics.clamped_logit_peak_magnitude, indices
            ),
            "logit_magnitude_change": _mean(
                metrics.logit_peak_magnitude_change, indices
            ),
            "normal_probability_magnitude": _mean(
                metrics.normal_probability_peak_magnitude, indices
            ),
            "clamped_probability_magnitude": _mean(
                metrics.clamped_probability_peak_magnitude, indices
            ),
            "probability_magnitude_change": _mean(
                metrics.probability_peak_magnitude_change, indices
            ),
        },
        "response_latency": {
            "normal_logit_ms": _mean(metrics.normal_logit_peak_latency_ms, indices),
            "clamped_logit_ms": _mean(metrics.clamped_logit_peak_latency_ms, indices),
            "logit_change_ms": _mean(metrics.logit_peak_latency_change_ms, indices),
            "logit_absolute_shift_ms": _mean(
                metrics.logit_peak_latency_absolute_shift_ms, indices
            ),
            "normal_probability_ms": _mean(
                metrics.normal_probability_peak_latency_ms, indices
            ),
            "clamped_probability_ms": _mean(
                metrics.clamped_probability_peak_latency_ms, indices
            ),
            "probability_change_ms": _mean(
                metrics.probability_peak_latency_change_ms, indices
            ),
            "probability_absolute_shift_ms": _mean(
                metrics.probability_peak_latency_absolute_shift_ms, indices
            ),
        },
        "temporal_rf": {
            "normal_norm": _mean(metrics.temporal_rf_normal_norm, indices),
            "clamped_norm": _mean(metrics.temporal_rf_clamped_norm, indices),
            "norm_change": _mean(metrics.temporal_rf_norm_change, indices),
            "change_norm": _mean(metrics.temporal_rf_change_norm, indices),
            "cosine": _mean(metrics.temporal_rf_cosine, indices),
        },
    }


def _cell_summary(
    metrics: CellPerturbationMetrics,
    index: int,
) -> dict[str, JsonValue]:
    return {
        "mean_absolute_logit_change": float(metrics.mean_absolute_logit_change[index]),
        "mean_absolute_probability_change": float(
            metrics.mean_absolute_probability_change[index]
        ),
        "logit_peak_magnitude_change": float(
            metrics.logit_peak_magnitude_change[index]
        ),
        "logit_peak_latency_change_ms": float(
            metrics.logit_peak_latency_change_ms[index]
        ),
        "temporal_rf_normal_norm": float(metrics.temporal_rf_normal_norm[index]),
        "temporal_rf_clamped_norm": float(metrics.temporal_rf_clamped_norm[index]),
        "temporal_rf_change_norm": float(metrics.temporal_rf_change_norm[index]),
        "temporal_rf_cosine": float(metrics.temporal_rf_cosine[index]),
    }


def _mean(values: torch.Tensor, indices: torch.Tensor) -> float:
    return float(values[indices].mean())


__all__ = [
    "PerturbationReportError",
    "PerturbationReportRequest",
    "build_perturbation_summary",
]
