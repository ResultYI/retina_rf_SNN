#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.6.0"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run aggregate_81.py
# 3. Or make executable and run:
#      chmod +x aggregate_81.py && ./aggregate_81.py
# ──────────────────

from __future__ import annotations

import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from typing import Final

import torch


ROOT: Final = Path(__file__).resolve().parents[3]
OUT: Final = Path(__file__).resolve().parent
APPLICATION: Final = ROOT / "output/real_data/schottdorf_canonical_v1_shared_bc_frozen_applications_20260830"
SELECTED: Final = ("69#4", "67#6", "68#4", "67#4")
FITS: Final = ("primary", "fresh_1", "fresh_2")
SIGNATURES: Final = {
    "Mach dark": "ramp_minus_matched_uniform_x-04",
    "Mach bright": "ramp_minus_matched_uniform_x+04",
    "SBC": "bright_surround_minus_dark_surround",
    "Hermann": "intersection_minus_corridor",
    "White": "on_bright_bar_minus_on_dark_bar",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_mean(logits: torch.Tensor, pair: tuple[int, int], active: torch.Tensor) -> float:
    first, second = pair
    return float((logits[first, active] - logits[second, active]).to(torch.float64).mean())


def aggregate(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def is_reversed(normal: float, ac_off: float) -> bool:
    return normal * ac_off < 0.0


def write_csv(path: Path, rows: list[dict[str, str | float | bool]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_csv = OUT / "aggregate_81_combinations.csv"
    output_summary = OUT / "aggregate_summary.md"
    assert not output_csv.exists() and not output_summary.exists(), "fresh aggregate output required"
    assert aggregate((1.0, -3.0)) == -1.0
    assert is_reversed(-2.0, 3.0) and not is_reversed(0.0, -1.0)
    inputs_path = APPLICATION / "illusion/inputs.pt"
    responses_path = APPLICATION / "illusion/responses.pt"
    inputs = torch.load(inputs_path, map_location="cpu", weights_only=True)
    responses = torch.load(responses_path, map_location="cpu", weights_only=True)
    assert tuple(sorted(responses["cells"])) and len(responses["cells"]) == 22
    assert set(SELECTED) <= set(responses["cells"])
    pair_lookup = {row["name"]: (int(row["a"]), int(row["b"])) for row in inputs["pairs"]}
    assert set(SIGNATURES.values()) <= set(pair_lookup)
    time_ms = inputs["time_ms"]
    active = (time_ms >= 300.0) & (time_ms < 400.0)
    assert int(active.sum()) == 15
    hashes = {str(inputs_path): digest(inputs_path), str(responses_path): digest(responses_path)}
    fixed_cells = tuple(cell for cell in sorted(responses["cells"]) if cell not in SELECTED)
    assert len(fixed_cells) == 18
    values: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for cell_id in fixed_cells:
        values[cell_id] = {"primary": {}}
        for mode in ("normal", "AC_off"):
            values[cell_id]["primary"][mode] = {
                label: paired_mean(responses["cells"][cell_id][mode]["logit"], pair_lookup[name], active)
                for label, name in SIGNATURES.items()
            }
    for cell_id in SELECTED:
        values[cell_id] = {}
        historical = responses["cells"][cell_id]
        for fit in FITS:
            evaluation_path = OUT / "fits" / cell_id.replace("#", "_") / fit / "evaluation.pt"
            hashes[str(evaluation_path)] = digest(evaluation_path)
            saved = torch.load(evaluation_path, map_location="cpu", weights_only=True)["illusion_logits"]
            if fit == "primary":
                assert all(torch.equal(saved[mode], historical[mode]["logit"]) for mode in ("normal", "AC_off"))
            values[cell_id][fit] = {
                mode: {label: paired_mean(saved[mode], pair_lookup[name], active) for label, name in SIGNATURES.items()}
                for mode in ("normal", "AC_off")
            }
    combinations = list(product(FITS, repeat=len(SELECTED)))
    rows: list[dict[str, str | float | bool]] = []
    primary_row: dict[str, str | float | bool] | None = None
    for index, choice in enumerate(combinations, start=1):
        assignment = dict(zip(SELECTED, choice, strict=True))
        row: dict[str, str | float | bool] = {"combination_id": f"C{index:02d}", **assignment}
        for label in SIGNATURES:
            normal_values = tuple(values[cell][assignment.get(cell, "primary")]["normal"][label] for cell in sorted(values))
            ac_values = tuple(values[cell][assignment.get(cell, "primary")]["AC_off"][label] for cell in sorted(values))
            normal, ac_off = aggregate(normal_values), aggregate(ac_values)
            row[f"{label}_normal"] = normal
            row[f"{label}_AC_off"] = ac_off
            row[f"{label}_reversed"] = is_reversed(normal, ac_off)
        rows.append(row)
        if choice == ("primary",) * len(SELECTED):
            primary_row = row
    assert len(rows) == 81 and primary_row is not None
    all_primary: dict[str, tuple[float, float]] = {}
    for label in SIGNATURES:
        normal = aggregate(tuple(values[cell]["primary"]["normal"][label] for cell in sorted(values)))
        ac_off = aggregate(tuple(values[cell]["primary"]["AC_off"][label] for cell in sorted(values)))
        assert primary_row[f"{label}_normal"] == normal and primary_row[f"{label}_AC_off"] == ac_off
        all_primary[label] = (normal, ac_off)
    write_csv(output_csv, rows)
    lines = ["# 81-combination full-precision aggregate check", "", "STATUS: COMPLETED", "",
        "18 cells are fixed at saved primary logits. The selected cells are `69#4`, `67#6`, `68#4`, and `67#4`; P/S1/S2 form all 3^4 = 81 assignments.",
        "Each paired logit is the existing 300–400 ms paired trace mean. Inputs are raw saved float32 logits, converted to float64 before the within-cell mean and equal-cell `math.fsum / 22` aggregate. No numeric threshold is applied: reversed means `normal * AC-off < 0`.", "",
        "| Signature | Reversed combinations | Normal range | AC-off range | Closest normal-to-zero | Closest AC-off-to-zero |",
        "|---|---:|---|---|---|---|"]
    for label in SIGNATURES:
        reversed_count = sum(bool(row[f"{label}_reversed"]) for row in rows)
        normal_rows = sorted(rows, key=lambda row: (abs(float(row[f"{label}_normal"])), str(row["combination_id"])))
        ac_rows = sorted(rows, key=lambda row: (abs(float(row[f"{label}_AC_off"])), str(row["combination_id"])))
        normal_values = [float(row[f"{label}_normal"]) for row in rows]
        ac_values = [float(row[f"{label}_AC_off"]) for row in rows]
        near_normal, near_ac = normal_rows[0], ac_rows[0]
        lines.append("| " + " | ".join((label, str(reversed_count),
            f"[{min(normal_values):.17g}, {max(normal_values):.17g}]",
            f"[{min(ac_values):.17g}, {max(ac_values):.17g}]",
            f"{near_normal['combination_id']} = {float(near_normal[f'{label}_normal']):.17g}",
            f"{near_ac['combination_id']} = {float(near_ac[f'{label}_AC_off']):.17g}")) + " |")
    lines += ["", "Combination labels in CSV are in Cartesian-product order `69#4, 67#6, 68#4, 67#4`, each taking `primary`, `fresh_1`, `fresh_2`. `C01` is all-primary.",
        "", "## Validation", "", "- 81 rows were emitted exactly once.",
        "- The C01 aggregate equals the direct aggregate of all 22 saved primary tensors for every signature and condition.",
        "- For every selected cell, its primary normal and AC-off tensor is bitwise identical to the historical 22-cell frozen application tensor.",
        "- All source tensor hashes used by this aggregate are recorded below.", "",
        "```json", json.dumps(hashes, indent=2), "```", ""]
    output_summary.write_text("\n".join(lines), encoding="utf-8")
    print("COMPLETED", len(rows), "combinations", flush=True)


if __name__ == "__main__":
    main()
