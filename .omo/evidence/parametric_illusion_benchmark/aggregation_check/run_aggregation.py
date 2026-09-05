#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///

# ─── How to run ───
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
# 2. Run: uv run run_aggregation.py
# ──────────────────

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


OUT: Final = Path(__file__).resolve().parent
SOURCE: Final = OUT.parent / "per_cell_curves.csv"
BOOTSTRAP_SAMPLES: Final = 100_000
BOOTSTRAP_SEED: Final = 20260902
sys.path.insert(0, str(OUT))

from aggregation_core import GROUPS, AggregationSpec, Group, Stratum, polarity  # noqa: E402

FloatArray = NDArray[np.float64]
Scalar = str | int | float | bool
CsvRow = Mapping[str, Scalar]
Condition = Literal["normal", "AC_off", "AC_effect"]


@dataclass(frozen=True, slots=True, order=True)
class Point:
    family: str
    signature: str
    contrast: float
    extent_px: int


@dataclass(frozen=True, slots=True)
class InputData:
    cells: tuple[str, ...]
    groups: tuple[Group, ...]
    points: tuple[Point, ...]
    normal: FloatArray
    ac_off: FloatArray


@dataclass(frozen=True, slots=True)
class BootstrapStratum:
    indices: NDArray[np.int64]
    coefficient: float
    weights: FloatArray


@dataclass(frozen=True, slots=True)
class BootstrapSpec:
    name: str
    strata: tuple[BootstrapStratum, ...]


@dataclass(frozen=True, slots=True)
class Summary:
    mean: FloatArray
    low: FloatArray
    high: FloatArray


@dataclass(frozen=True, slots=True)
class AggregationInputError(Exception):
    value: str

    def __str__(self) -> str:
        return f"invalid group {self.value!r}"


def parse_group(value: str) -> Group:
    if value == "MC_ON": return "MC_ON"
    if value == "MC_OFF": return "MC_OFF"
    if value == "PC_ON": return "PC_ON"
    if value == "PC_OFF": return "PC_OFF"
    raise AggregationInputError(value)


def load_input() -> InputData:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["model"] == "canonical" and row["mode"] in ("normal", "AC_off")]
    cells = tuple(dict.fromkeys(row["cell_id"] for row in rows))
    group_lookup = {row["cell_id"]: parse_group(row["group"]) for row in rows}
    points = tuple(sorted({Point(row["family"], row["signature"], float(row["contrast"]),
                                 int(row["extent_px"])) for row in rows}))
    lookup = {(row["cell_id"], row["mode"], Point(row["family"], row["signature"],
               float(row["contrast"]), int(row["extent_px"]))): float(row["paired_logit"]) for row in rows}
    matrix = lambda mode: np.asarray([[lookup[(cell, mode, point)] for point in points] for cell in cells])
    return InputData(cells, tuple(group_lookup[cell] for cell in cells), points, matrix("normal"), matrix("AC_off"))


def bootstrap_stratum(indices: NDArray[np.int64], coefficient: float, seed: int) -> BootstrapStratum:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(indices), size=(BOOTSTRAP_SAMPLES, len(indices)))
    weights = np.zeros((BOOTSTRAP_SAMPLES, len(indices)), dtype=np.float64)
    np.add.at(weights, (np.arange(BOOTSTRAP_SAMPLES)[:, None], samples), 1.0 / len(indices))
    return BootstrapStratum(indices, coefficient, weights)


def specs(data: InputData) -> tuple[tuple[BootstrapSpec, ...], tuple[BootstrapSpec, ...]]:
    indices = {group: np.asarray([i for i, value in enumerate(data.groups) if value == group]) for group in GROUPS}
    on = np.concatenate((indices["MC_ON"], indices["PC_ON"]))
    off = np.concatenate((indices["MC_OFF"], indices["PC_OFF"]))
    raw = bootstrap_stratum(np.arange(len(data.cells)), 1.0, BOOTSTRAP_SEED)
    by_group = tuple(bootstrap_stratum(indices[group], 1.0, BOOTSTRAP_SEED + 10 + i)
                     for i, group in enumerate(GROUPS))
    on_stratum = bootstrap_stratum(on, 0.5, BOOTSTRAP_SEED + 20)
    off_stratum = bootstrap_stratum(off, 0.5, BOOTSTRAP_SEED + 21)
    equal_groups = tuple(BootstrapStratum(value.indices, 0.25, value.weights) for value in by_group)
    aggregation = (BootstrapSpec("raw_22cell", (raw,)), BootstrapSpec("ON_OFF_equal", (on_stratum, off_stratum)),
                   BootstrapSpec("four_group_equal", equal_groups))
    groups = tuple(BootstrapSpec(group, (value,)) for group, value in zip(GROUPS, by_group, strict=True))
    return aggregation, groups


def summarize(matrix: FloatArray, specification: BootstrapSpec) -> Summary:
    mean = sum(stratum.coefficient * matrix[stratum.indices].mean(axis=0) for stratum in specification.strata)
    low, high = np.empty(matrix.shape[1]), np.empty(matrix.shape[1])
    for start in range(0, matrix.shape[1], 16):
        stop = min(start + 16, matrix.shape[1])
        draws = sum(stratum.coefficient * (stratum.weights @ matrix[stratum.indices, start:stop])
                    for stratum in specification.strata)
        low[start:stop], high[start:stop] = np.quantile(draws, (0.025, 0.975), axis=0)
    return Summary(mean, low, high)


def write_csv(name: str, rows: Sequence[CsvRow]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def point_fields(point: Point) -> dict[str, Scalar]:
    return {"family": point.family, "signature": point.signature, "contrast": point.contrast,
            "extent_px": point.extent_px}


def direction(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "exact_zero"


def curve_rows(data: InputData, specifications: tuple[BootstrapSpec, ...], label: str) -> list[dict[str, Scalar]]:
    output: list[dict[str, Scalar]] = []
    for condition, matrix in (("normal", data.normal), ("AC_off", data.ac_off),
                              ("AC_effect", data.ac_off - data.normal)):
        for specification in specifications:
            values = summarize(matrix, specification)
            for index, point in enumerate(data.points):
                output.append({label: specification.name, "condition": condition, **point_fields(point),
                               "mean": values.mean[index], "ci95_low": values.low[index],
                               "ci95_high": values.high[index]})
    return output


def interaction_data(data: InputData) -> tuple[tuple[Point, ...], FloatArray, FloatArray, FloatArray]:
    keys, normal, ac_off, effect = [], [], [], []
    point_index = {point: index for index, point in enumerate(data.points)}
    for signature in ("dark_ramp_minus_uniform", "bright_ramp_minus_uniform"):
        for contrast in (0.0, 0.0625, 0.125, 0.25, 0.375, 0.5):
            control = point_index[Point("Mach", signature, contrast, 0)]
            for width in (2, 4, 8, 12, 16):
                current = point_index[Point("Mach", signature, contrast, width)]
                keys.append(Point("Mach", signature, contrast, width))
                normal.append(data.normal[:, current] - data.normal[:, control])
                ac_off.append(data.ac_off[:, current] - data.ac_off[:, control])
                effect.append((data.ac_off[:, current] - data.normal[:, current])
                              - (data.ac_off[:, control] - data.normal[:, control]))
    return tuple(keys), np.stack(normal, axis=1), np.stack(ac_off, axis=1), np.stack(effect, axis=1)


def main() -> None:
    data = load_input()
    aggregation_specs, group_specs = specs(data)
    effect = data.ac_off - data.normal
    labeled = [{"cell_id": cell, "group": group, "polarity": polarity(group), **point_fields(point),
                "normal": data.normal[cell_index, point_index], "AC_off": data.ac_off[cell_index, point_index],
                "AC_effect": effect[cell_index, point_index]}
               for cell_index, (cell, group) in enumerate(zip(data.cells, data.groups, strict=True))
               for point_index, point in enumerate(data.points)]
    write_csv("labeled_surfaces.csv", labeled)
    write_csv("group_surface_curves.csv", curve_rows(data, group_specs, "group"))
    write_csv("aggregation_surface_curves.csv", curve_rows(data, aggregation_specs, "aggregation"))

    keys, normal_interaction, ac_interaction, effect_interaction = interaction_data(data)
    per_cell = [{"cell_id": cell, "group": group, "polarity": polarity(group), **point_fields(point),
                 "normal_width_minus_width0": normal_interaction[cell_index, point_index],
                 "ACoff_width_minus_width0": ac_interaction[cell_index, point_index],
                 "AC_effect_interaction": effect_interaction[cell_index, point_index],
                 "interaction_direction": direction(effect_interaction[cell_index, point_index])}
                for cell_index, (cell, group) in enumerate(zip(data.cells, data.groups, strict=True))
                for point_index, point in enumerate(keys)]
    write_csv("mach_control_subtracted_per_cell.csv", per_cell)
    for filename, specifications, label in (("mach_control_subtracted_groups.csv", group_specs, "group"),
                                             ("mach_control_subtracted_aggregations.csv", aggregation_specs, "aggregation")):
        rows: list[dict[str, Scalar]] = []
        for metric, matrix in (("normal_width_minus_width0", normal_interaction),
                               ("ACoff_width_minus_width0", ac_interaction),
                               ("AC_effect_interaction", effect_interaction)):
            for specification in specifications:
                values = summarize(matrix, specification)
                rows.extend({label: specification.name, "metric": metric, **point_fields(point),
                             "mean": values.mean[i], "ci95_low": values.low[i], "ci95_high": values.high[i]}
                            for i, point in enumerate(keys))
        write_csv(filename, rows)
    metadata = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "cells": len(data.cells), "group_counts": {group: data.groups.count(group) for group in GROUPS},
                "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                "models_loaded": False, "training_performed": False, "new_probes_generated": False}
    (OUT / "aggregation_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
