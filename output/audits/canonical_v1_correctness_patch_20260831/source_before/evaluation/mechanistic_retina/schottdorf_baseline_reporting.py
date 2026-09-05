from __future__ import annotations

import csv
import statistics
from pathlib import Path

from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.schottdorf_baseline_types import (
    BaselineCellRecord,
    SchottdorfBaselineRunError,
)


def summarize_baselines(
    cells: list[BaselineCellRecord],
) -> dict[str, JsonValue]:
    if not cells:
        raise SchottdorfBaselineRunError(
            "baseline summary requires at least one cell"
        )
    names = ("constant_rate", "glm", "retinal")
    mean_nll = {
        name: statistics.fmean(float(cell[f"{name}_nll"]) for cell in cells)
        for name in names
    }
    total_valid = sum(cell["validation_valid_bins"] for cell in cells)
    pooled_nll = {
        name: sum(
            float(cell[f"{name}_nll"]) * cell["validation_valid_bins"]
            for cell in cells
        )
        / total_valid
        for name in names
    }
    groups: dict[str, JsonValue] = {}
    for retinal_class in ("MC", "PC"):
        for polarity in ("ON", "OFF"):
            selected = [
                cell
                for cell in cells
                if cell["retinal_class"] == retinal_class
                and cell["polarity"] == polarity
            ]
            groups[f"{retinal_class}_{polarity}"] = _group_row(selected)
    winners = {
        name: sum(cell["winner"] == name for cell in cells) for name in names
    }
    parameter_counts = {
        name: {
            field: sum(
                cell[
                    "constant_parameters"
                    if name == "constant_rate"
                    else f"{name}_parameters"
                ][field]
                for cell in cells
            )
            for field in ("total", "requires_grad", "optimizer_listed")
        }
        for name in names
    }
    return {
        "mean_validation_nll_across_cells": mean_nll,
        "pooled_validation_nll_weighted_by_valid_bins": pooled_nll,
        "nll_differences": {
            "retinal_minus_constant_rate": mean_nll["retinal"]
            - mean_nll["constant_rate"],
            "retinal_minus_glm": mean_nll["retinal"] - mean_nll["glm"],
        },
        "winner_counts": winners,
        "retinal_strictly_better_than_constant_count": sum(
            cell["retinal_strictly_better_than_constant"] for cell in cells
        ),
        "group_mean_validation_nll": groups,
        "aggregate_parameter_counts": parameter_counts,
        "glm_converged_cells": sum(cell["glm_converged"] for cell in cells),
        "glm_strict_gradient_converged_cells": sum(
            cell["glm_strict_gradient_converged"] for cell in cells
        ),
        "glm_finite_gradient_cells": sum(
            cell["glm_gradients_finite"] for cell in cells
        ),
        "maximum_glm_final_gradient": max(
            cell["glm_final_gradient_max"] for cell in cells
        ),
        "maximum_retinal_replay_error": max(
            cell["retinal_nll_replay_error"] for cell in cells
        ),
    }


def _group_row(cells: list[BaselineCellRecord]) -> dict[str, JsonValue]:
    if not cells:
        return {
            "cell_count": 0,
            "constant_rate": None,
            "glm": None,
            "retinal": None,
        }
    return {
        "cell_count": len(cells),
        "constant_rate": statistics.fmean(
            cell["constant_rate_nll"] for cell in cells
        ),
        "glm": statistics.fmean(cell["glm_nll"] for cell in cells),
        "retinal": statistics.fmean(cell["retinal_nll"] for cell in cells),
    }


def write_cell_csv(path: Path, cells: list[BaselineCellRecord]) -> None:
    fields = (
        "cell_id",
        "recording_ids",
        "recording_count",
        "retinal_class",
        "canonical_cell_type",
        "polarity",
        "native_dt_ms",
        "train_sequences",
        "validation_sequences",
        "train_valid_bins",
        "validation_valid_bins",
        "constant_rate_nll",
        "glm_nll",
        "retinal_nll",
        "winner",
        "retinal_strictly_better_than_constant",
        "glm_solver_iterations",
        "glm_solver_evaluations",
        "glm_final_gradient_max",
        "glm_strict_gradient_converged",
        "glm_solver_terminated_before_budget",
        "glm_converged",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)


__all__ = ["summarize_baselines", "write_cell_csv"]
