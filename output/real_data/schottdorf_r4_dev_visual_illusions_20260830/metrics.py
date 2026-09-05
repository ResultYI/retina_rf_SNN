from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeAlias

import torch

from stimuli import DT_MS, DURATION_MS, ONSET_MS, Pair, Stimuli

Scalar: TypeAlias = str | int | float | bool | None
Row: TypeAlias = dict[str, Scalar]


def response_metrics(trace: torch.Tensor, time_ms: torch.Tensor) -> dict[str, float | None]:
    active = (time_ms >= ONSET_MS) & (time_ms < ONSET_MS + DURATION_MS)
    post = time_ms >= ONSET_MS
    peak_index = int(trace[post].abs().argmax()) + int(post.nonzero()[0])
    peak = float(trace[peak_index])
    return {"mean_on": float(trace[active].mean()), "peak_signed": peak,
            "peak_absolute": abs(peak),
            "peak_latency_from_onset_ms": float(time_ms[peak_index]) - ONSET_MS if peak != 0 else None,
            "integral_response_seconds": float(trace.sum()) * DT_MS / 1000,
            "onset_50ms_mean": float(trace[(time_ms >= ONSET_MS) & (time_ms < ONSET_MS + 50)].mean()),
            "offset_50ms_mean": float(trace[(time_ms >= ONSET_MS + DURATION_MS) &
                                             (time_ms < ONSET_MS + DURATION_MS + 50)].mean())}


def cell_rows(cell_id: str, group: str, bank: Stimuli,
              responses: dict[str, dict[str, torch.Tensor]], time_ms: torch.Tensor) -> list[Row]:
    rows: list[Row] = []
    for mode, channels in responses.items():
        for channel in ("logit", "probability"):
            value = channels[channel]
            for index, name in enumerate(bank.names):
                common = {"cell_id": cell_id, "group": group, "mode": mode, "channel": channel,
                          "family": name.split("_")[0], "name": name}
                for kind, trace in (("response_minus_blank", value[index] - value[-1]),
                                    ("clamp_minus_normal", value[index] - responses["normal"][channel][index])):
                    rows.append(common | {"kind": kind} | response_metrics(trace, time_ms))
            for pair in bank.pairs:
                common = {"cell_id": cell_id, "group": group, "mode": mode, "channel": channel,
                          "family": pair.family, "name": pair.name, "control": pair.control,
                          "x_pixels": pair.x_pixels}
                trace = value[pair.a] - value[pair.b]
                normal = responses["normal"][channel][pair.a] - responses["normal"][channel][pair.b]
                for kind, curve in (("pair_difference", trace), ("pair_clamp_minus_normal", trace - normal)):
                    rows.append(common | {"kind": kind} | response_metrics(curve, time_ms))
    return rows


def mach_rows(cell_id: str, group: str, bank: Stimuli,
              responses: dict[str, dict[str, torch.Tensor]], time_ms: torch.Tensor) -> list[Row]:
    active = (time_ms >= ONSET_MS) & (time_ms < ONSET_MS + DURATION_MS)
    rows: list[Row] = []
    for mode, channels in responses.items():
        for channel in ("logit", "probability"):
            for label, start in (("ramp", 0), ("matched_uniform", 25)):
                profile = channels[channel][start:start + 25, active].mean(dim=1)
                upper, lower = profile[[0, 24]].max(), profile[[0, 24]].min()
                for name, left, right in (("dark_boundary", -6, -2), ("bright_boundary", 2, 6)):
                    values = profile[left + 12:right + 13]
                    hi, lo = float((values - upper).max()), float((values - lower).min())
                    rows.append({"cell_id": cell_id, "group": group, "mode": mode, "channel": channel,
                                 "profile": label, "region": name, "overshoot_above_plateaus": max(0.0, hi),
                                 "undershoot_below_plateaus": min(0.0, lo),
                                 "max_minus_upper_plateau": hi, "min_minus_lower_plateau": lo,
                                 "max_x_pixels": left + int(values.argmax()),
                                 "min_x_pixels": left + int(values.argmin())})
    return rows


def aggregate(rows: list[Row]) -> list[Row]:
    groups: dict[tuple[Scalar, ...], list[Row]] = {}
    keys = ("group", "mode", "channel", "family", "name", "kind")
    for row in rows:
        for group in (str(row["group"]), "all"):
            key = (group,) + tuple(row.get(k) for k in keys[1:])
            groups.setdefault(key, []).append(row)
    output: list[Row] = []
    for key, values in groups.items():
        item = dict(zip(keys, key, strict=True))
        item["n_cells"] = len(values)
        for metric in ("mean_on", "peak_signed", "peak_absolute", "peak_latency_from_onset_ms",
                       "integral_response_seconds", "onset_50ms_mean", "offset_50ms_mean"):
            numbers = [float(v[metric]) for v in values if v[metric] is not None]
            item[metric] = sum(numbers) / len(numbers) if numbers else None
        signs = [float(v["mean_on"]) for v in values]
        item.update({"positive_cells": sum(v > 1e-9 for v in signs),
                     "negative_cells": sum(v < -1e-9 for v in signs),
                     "zero_within_1e9_cells": sum(abs(v) <= 1e-9 for v in signs)})
        output.append(item)
    return output


def write_csv(path: Path, rows: list[Row]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
