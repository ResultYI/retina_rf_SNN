#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping


OUT: Final = Path(__file__).resolve().parent
GROUPS: Final = ("MC_ON", "MC_OFF", "PC_ON", "PC_OFF")
SIGNATURES: Final = (
    "bright_minus_dark",
    "dark_ramp_minus_uniform",
    "bright_ramp_minus_uniform",
)
Scalar = str | int | float | bool
Row = dict[str, Scalar]


@dataclass(frozen=True, slots=True)
class UnsupportedSignatureError(Exception):
    signature: str

    def __str__(self) -> str:
        return f"unsupported signature {self.signature!r}"


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[Row]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def direction(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "exact_zero"


def expected_positive_polarity(signature: str) -> str:
    if signature in ("bright_minus_dark", "dark_ramp_minus_uniform"):
        return "ON"
    if signature == "bright_ramp_minus_uniform":
        return "OFF"
    raise UnsupportedSignatureError(signature)


def is_sign_audit_point(row: Mapping[str, str]) -> bool:
    if float(row["contrast"]) == 0:
        return False
    return row["family"] == "Mach" or int(row["extent_px"]) > 0


def sign_audit(rows: list[dict[str, str]]) -> tuple[list[Row], list[Row]]:
    cell_groups = {row["cell_id"]: row["group"] for row in rows}
    by_point: dict[tuple[str, float, int], list[dict[str, str]]] = defaultdict(list)
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not is_sign_audit_point(row):
            continue
        key = row["signature"], float(row["contrast"]), int(row["extent_px"])
        by_point[key].append(row)
        by_cell[(row["cell_id"], row["signature"])].append(float(row["AC_effect"]))
    points: list[Row] = []
    for (signature, contrast, extent), values in sorted(by_point.items()):
        positive = {row["cell_id"] for row in values if float(row["AC_effect"]) > 0}
        negative = {row["cell_id"] for row in values if float(row["AC_effect"]) < 0}
        expected = expected_positive_polarity(signature)
        expected_positive = {cell for cell, group in cell_groups.items() if group.endswith(expected)}
        points.append({
            "signature": signature,
            "contrast": contrast,
            "extent_px": extent,
            "positive_cells": len(positive),
            "negative_cells": len(negative),
            "zero_cells": 22 - len(positive) - len(negative),
            "expected_positive_polarity": expected,
            "exact_polarity_partition": positive == expected_positive and negative == set(cell_groups) - expected_positive,
        })
    cells: list[Row] = []
    for (cell, signature), values in sorted(by_cell.items()):
        expected = "positive" if cell_groups[cell].endswith(expected_positive_polarity(signature)) else "negative"
        observed = {direction(value) for value in values}
        cells.append({
            "cell_id": cell,
            "group": cell_groups[cell],
            "signature": signature,
            "points": len(values),
            "observed_directions": ";".join(sorted(observed)),
            "expected_direction": expected,
            "all_points_match": observed == {expected},
        })
    return points, cells


def reversal_audit(rows: list[dict[str, str]]) -> list[Row]:
    values: dict[tuple[str, str, float, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["condition"] not in ("normal", "AC_off") or float(row["contrast"]) == 0:
            continue
        if row["family"] == "SBC" and int(row["extent_px"]) == 0:
            continue
        key = row["aggregation"], row["signature"], float(row["contrast"]), int(row["extent_px"])
        values[key][row["condition"]] = float(row["mean"])
    output: list[Row] = []
    for (aggregation, signature, contrast, extent), pair in sorted(values.items()):
        normal, ac_off = pair["normal"], pair["AC_off"]
        output.append({
            "aggregation": aggregation,
            "signature": signature,
            "contrast": contrast,
            "extent_px": extent,
            "normal": normal,
            "AC_off": ac_off,
            "normal_direction": direction(normal),
            "AC_off_direction": direction(ac_off),
            "reversed": normal * ac_off < 0,
        })
    return output


def interaction_directions(
    per_cell: list[dict[str, str]], group_rows: list[dict[str, str]]
) -> list[Row]:
    means = {
        (row["group"], row["signature"], float(row["contrast"]), int(row["extent_px"])): row
        for row in group_rows if row["metric"] == "AC_effect_interaction"
    }
    buckets: dict[tuple[str, float, int, str], list[float]] = defaultdict(list)
    for row in per_cell:
        if float(row["contrast"]) == 0:
            continue
        buckets[(row["signature"], float(row["contrast"]), int(row["extent_px"]), row["group"])].append(
            float(row["AC_effect_interaction"])
        )
    output: list[Row] = []
    for (signature, contrast, extent, group), values in sorted(buckets.items()):
        summary = means[(group, signature, contrast, extent)]
        output.append({
            "signature": signature,
            "contrast": contrast,
            "extent_px": extent,
            "group": group,
            "positive_cells": sum(value > 0 for value in values),
            "negative_cells": sum(value < 0 for value in values),
            "zero_cells": sum(value == 0 for value in values),
            "mean": float(summary["mean"]),
            "ci95_low": float(summary["ci95_low"]),
            "ci95_high": float(summary["ci95_high"]),
        })
    return output


def count_reversals(rows: Iterable[Mapping[str, Scalar]]) -> dict[str, dict[str, str]]:
    counts: dict[str, dict[str, str]] = defaultdict(dict)
    buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["aggregation"]), str(row["signature"]))].append(bool(row["reversed"]))
    for (aggregation, signature), values in buckets.items():
        counts[aggregation][signature] = f"{sum(values)}/{len(values)}"
    return dict(counts)


def main() -> None:
    labeled = read_csv("labeled_surfaces.csv")
    aggregations = read_csv("aggregation_surface_curves.csv")
    interactions = read_csv("mach_control_subtracted_per_cell.csv")
    interaction_groups = read_csv("mach_control_subtracted_groups.csv")
    point_audit, cell_audit = sign_audit(labeled)
    reversal_rows = reversal_audit(aggregations)
    direction_rows = interaction_directions(interactions, interaction_groups)
    write_csv("effect_sign_point_audit.csv", point_audit)
    write_csv("effect_sign_cell_audit.csv", cell_audit)
    write_csv("cohort_reversal_audit.csv", reversal_rows)
    write_csv("mach_interaction_direction_summary.csv", direction_rows)

    manifest = json.loads((OUT / "aggregation_manifest.json").read_text(encoding="utf-8"))
    conclusion = {
        "group_counts": manifest["group_counts"],
        "polarity_counts": {"ON": 14, "OFF": 8},
        "effect_sign_points_checked": len(point_audit),
        "effect_sign_exact_polarity_partition": all(bool(row["exact_polarity_partition"]) for row in point_audit),
        "effect_sign_cells_checked": len(cell_audit),
        "effect_sign_all_cell_surfaces_match": all(bool(row["all_points_match"]) for row in cell_audit),
        "cohort_reversal_counts": count_reversals(reversal_rows),
    }
    (OUT / "conclusion_metrics.json").write_text(json.dumps(conclusion, indent=2), encoding="utf-8")
    source = OUT.parent / "per_cell_curves.csv"
    verification = {
        "source_sha256_matches_manifest": hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"],
        "row_counts": {
            "labeled_surfaces": len(labeled),
            "aggregation_surface_curves": len(aggregations),
            "mach_control_subtracted_per_cell": len(interactions),
            "effect_sign_point_audit": len(point_audit),
            "effect_sign_cell_audit": len(cell_audit),
        },
        "cells": len({row["cell_id"] for row in labeled}),
        "groups": sorted({row["group"] for row in labeled}),
        "effect_sign_contract_pass": conclusion["effect_sign_exact_polarity_partition"]
        and conclusion["effect_sign_all_cell_surfaces_match"],
        "models_loaded": manifest["models_loaded"],
        "training_performed": manifest["training_performed"],
        "new_probes_generated": manifest["new_probes_generated"],
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
