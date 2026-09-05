from __future__ import annotations

import csv
import statistics

from evaluation.json_types import JsonValue


_MODELS = ("constant", "glm", "neural", "retinal")


def summarize_final_benchmark(
    cells: list[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    total_valid = sum(cell["validation_valid_bins"] for cell in cells)
    means = {
        name: statistics.fmean(cell[f"{name}_nll"] for cell in cells)
        for name in _MODELS
    }
    pooled = {
        name: sum(
            cell[f"{name}_nll"] * cell["validation_valid_bins"] for cell in cells
        )
        / total_valid
        for name in _MODELS
    }
    groups = {}
    for retinal_class in ("MC", "PC"):
        for polarity in ("ON", "OFF"):
            selected = [
                cell
                for cell in cells
                if cell["retinal_class"] == retinal_class
                and cell["polarity"] == polarity
            ]
            groups[f"{retinal_class}_{polarity}"] = {
                "cell_count": len(selected),
                **{
                    name: (
                        statistics.fmean(cell[f"{name}_nll"] for cell in selected)
                        if selected
                        else None
                    )
                    for name in _MODELS
                },
            }
    parameters = {
        name: {
            field: sum(cell[f"{name}_parameters"][field] for cell in cells)
            for field in ("total", "requires_grad", "optimizer_listed")
        }
        for name in _MODELS
    }
    return {
        "mean_validation_nll_across_cells": means,
        "pooled_validation_nll_weighted_by_valid_bins": pooled,
        "group_mean_validation_nll": groups,
        "winner_counts": {
            name: sum(cell["winner"] == name for cell in cells) for name in _MODELS
        },
        "retinal_minus_baselines": {
            name: means["retinal"] - means[name]
            for name in ("constant", "glm", "neural")
        },
        "aggregate_parameter_counts": parameters,
        "maximum_retinal_replay_error": max(
            cell["retinal_nll_replay_error"] for cell in cells
        ),
        "maximum_glm_replay_error": max(cell["glm_nll_replay_error"] for cell in cells),
        "neural_finite_gradient_cells": sum(
            cell["neural_gradients_finite"] for cell in cells
        ),
    }


def write_final_cell_csv(path, cells: list[dict[str, JsonValue]]) -> None:
    fields = (
        "cell_id",
        "retinal_class",
        "canonical_cell_type",
        "polarity",
        "recording_count",
        "train_sequences",
        "validation_sequences",
        "train_valid_bins",
        "validation_valid_bins",
        "constant_nll",
        "glm_nll",
        "neural_nll",
        "retinal_nll",
        "winner",
        "neural_best_step",
        "neural_stop_step",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)


__all__ = ["summarize_final_benchmark", "write_final_cell_csv"]
