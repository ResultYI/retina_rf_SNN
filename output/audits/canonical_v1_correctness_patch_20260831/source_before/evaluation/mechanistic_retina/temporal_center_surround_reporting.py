from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import statistics

import torch

from evaluation.mechanistic_retina.temporal_center_surround import (
    CenterSurroundProbe,
    CenterSurroundProbeConfig,
    ResponseSummary,
    summarize_response,
)


@dataclass(frozen=True, slots=True)
class MetricRow:
    cell_id: str
    group: str
    condition: str
    mode: str
    offset_ms: float | None
    peak_response_probability: float
    peak_response_logit: float
    peak_latency_ms_probability: float
    peak_latency_ms_logit: float
    response_integral_probability_s: float
    response_integral_logit_s: float
    integral_change_vs_center_only: float
    integral_fraction_vs_center_only: float
    peak_change_vs_center_only: float
    peak_fraction_vs_center_only: float
    center_onset_response: float | None
    center_offset_response: float | None
    surround_onset_response: float | None
    surround_offset_response: float | None


@dataclass(frozen=True, slots=True)
class CellTrace:
    cell_id: str
    group: str
    time_ms: torch.Tensor
    condition_names: tuple[str, ...]
    probability_delta: dict[str, torch.Tensor]


def condition_metric_rows(
    cell_id: str,
    group: str,
    mode: str,
    probe: CenterSurroundProbe,
    logit_delta: torch.Tensor,
    probability_delta: torch.Tensor,
    config: CenterSurroundProbeConfig,
) -> tuple[MetricRow, ...]:
    summaries = _summaries(probe, logit_delta, probability_delta, config)
    center_probability = summaries[0][1]
    output = []
    for index, (logit_summary, probability_summary) in enumerate(summaries):
        offset = float(probe.offset_ms[index])
        output.append(
            _metric_row(
                cell_id,
                group,
                probe.names[index],
                mode,
                None if math.isnan(offset) else offset,
                logit_summary,
                probability_summary,
                center_probability,
            )
        )
    return tuple(output)


def group_metric_rows(
    rows: list[MetricRow],
) -> list[dict[str, str | int | float | None]]:
    keys = sorted({(row.group, row.condition, row.mode) for row in rows})
    numeric = tuple(
        field.name
        for field in fields(MetricRow)
        if field.name not in {"cell_id", "group", "condition", "mode", "offset_ms"}
    )
    output = []
    for group, condition, mode in keys:
        selected = [
            row
            for row in rows
            if (row.group, row.condition, row.mode) == (group, condition, mode)
        ]
        summary: dict[str, str | int | float | None] = {
            "group": group,
            "condition": condition,
            "mode": mode,
            "cells": len(selected),
        }
        for name in numeric:
            values = [getattr(row, name) for row in selected]
            finite = [float(value) for value in values if value is not None]
            summary[f"mean_{name}"] = statistics.fmean(finite) if finite else None
        output.append(summary)
    return output


def write_metric_tables(
    output_dir: Path,
    rows: list[MetricRow],
    group_rows: list[dict[str, str | int | float | None]],
) -> None:
    with (output_dir / "per_cell_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(MetricRow)]
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    with (output_dir / "group_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)


def _summaries(
    probe: CenterSurroundProbe,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    config: CenterSurroundProbeConfig,
) -> tuple[tuple[ResponseSummary, ResponseSummary], ...]:
    values = []
    for index in range(len(probe.names)):
        arguments = {
            "dt_ms": probe.dt_ms,
            "center_onset_ms": _optional_float(probe.center_onset_ms[index]),
            "surround_onset_ms": _optional_float(probe.surround_onset_ms[index]),
            "pulse_duration_ms": probe.pulse_duration_ms,
            "event_window_ms": config.event_window_ms,
        }
        values.append(
            (
                summarize_response(logits[index], **arguments),
                summarize_response(probabilities[index], **arguments),
            )
        )
    return tuple(values)


def _metric_row(
    cell_id: str,
    group: str,
    condition: str,
    mode: str,
    offset: float | None,
    logit: ResponseSummary,
    probability: ResponseSummary,
    center: ResponseSummary,
) -> MetricRow:
    integral_difference = probability.response_integral - center.response_integral
    peak_difference = probability.peak_response - center.peak_response
    return MetricRow(
        cell_id,
        group,
        condition,
        mode,
        offset,
        probability.peak_response,
        logit.peak_response,
        probability.peak_latency_ms,
        logit.peak_latency_ms,
        probability.response_integral,
        logit.response_integral,
        integral_difference,
        integral_difference / max(abs(center.response_integral), 1e-12),
        peak_difference,
        peak_difference / max(center.peak_absolute_response, 1e-12),
        probability.center_onset_response,
        probability.center_offset_response,
        probability.surround_onset_response,
        probability.surround_offset_response,
    )


def _optional_float(value: torch.Tensor) -> float | None:
    scalar = float(value)
    return None if math.isnan(scalar) else scalar


__all__ = [
    "CellTrace",
    "MetricRow",
    "condition_metric_rows",
    "group_metric_rows",
    "write_metric_tables",
]
