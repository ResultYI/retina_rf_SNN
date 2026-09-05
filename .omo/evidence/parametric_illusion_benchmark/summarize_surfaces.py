from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np


OUT: Final = Path(__file__).resolve().parent
Scalar = str | int | float | bool
Row = dict[str, Scalar]


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[Row]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def noncontrol(row: dict[str, str]) -> bool:
    return float(row["contrast"]) > 0 and (row["family"] == "Mach" or int(row["extent_px"]) > 0)


def pointwise_summary(name: str) -> list[Row]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(name):
        if noncontrol(row):
            groups[(row["model"], row["family"], row["signature"])].append(row)
    output = []
    for key, values in groups.items():
        means = np.asarray([float(row["mean_paired_logit_difference"]) for row in values])
        directions = [row["direction"] for row in values]
        same_sign_fractions = []
        for row, mean in zip(values, means, strict=True):
            same = int(row["positive_cells"] if mean > 0 else row["negative_cells"] if mean < 0 else row["exact_zero_cells"])
            same_sign_fractions.append(same / int(row["n_cells"]))
        output.append(dict(zip(("comparison", "family", "signature"), key, strict=True)) | {
            "noncontrol_points": len(values), "positive_ci_points": directions.count("positive"),
            "negative_ci_points": directions.count("negative"),
            "ci_includes_zero_points": directions.count("CI_includes_zero"),
            "mean_absolute_cohort_difference": float(np.abs(means).mean()),
            "max_absolute_cohort_difference": float(np.abs(means).max()),
            "mean_same_sign_cell_fraction": float(np.mean(same_sign_fractions)),
            "minimum_same_sign_cell_fraction": float(np.min(same_sign_fractions)),
        })
    return output


def alignment_rows(name: str, value_name: str) -> list[Row]:
    rows = [row for row in read_csv(name) if noncontrol(row)]
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family"], row["signature"])].append(row)
    output = []
    for key, values in grouped.items():
        cells = sorted({row["cell_id"] for row in values})
        points = sorted({(float(row["contrast"]), int(row["extent_px"])) for row in values})
        lookup = {(row["cell_id"], float(row["contrast"]), int(row["extent_px"])): float(row[value_name]) for row in values}
        matrix = np.asarray([[lookup[(cell, *point)] for point in points] for cell in cells])
        cohort = matrix.mean(axis=0)
        for cell, vector in zip(cells, matrix, strict=True):
            denominator = np.linalg.norm(vector) * np.linalg.norm(cohort)
            cosine = float(vector @ cohort / denominator) if denominator else 0.0
            correlation = float(np.corrcoef(vector, cohort)[0, 1]) if vector.std() and cohort.std() else 0.0
            active = cohort != 0
            sign_agreement = float(np.mean(np.sign(vector[active]) == np.sign(cohort[active]))) if active.any() else 1.0
            output.append(dict(zip(("comparison", "family", "signature"), key, strict=True)) | {
                "cell_id": cell, "surface_cosine_to_cohort": cosine,
                "surface_correlation_to_cohort": correlation, "pointwise_sign_agreement": sign_agreement,
                "surface_norm": float(np.linalg.norm(vector)),
            })
    return output


def monotonic_summary() -> list[Row]:
    rows = read_csv("per_cell_monotonicity.csv")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    keys = ("model", "mode", "family", "signature", "sweep_axis")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, values in groups.items():
        fractions = np.asarray([float(row["monotonic_fraction"]) for row in values])
        rho = np.asarray([float(row["spearman_rho"]) for row in values])
        output.append(dict(zip(keys, key, strict=True)) | {
            "curves": len(values), "increasing": sum(row["direction"] == "increasing" for row in values),
            "decreasing": sum(row["direction"] == "decreasing" for row in values),
            "flat": sum(row["direction"] == "flat" for row in values),
            "strictly_monotonic": sum(row["strictly_monotonic"] == "True" for row in values),
            "median_monotonic_fraction": float(np.median(fractions)), "mean_spearman_rho": float(rho.mean()),
        })
    return output


def main() -> None:
    model = pointwise_summary("cohort_model_differences.csv")
    clamp = pointwise_summary("cohort_clamp_differences.csv")
    model_alignment = alignment_rows("per_cell_model_differences.csv", "paired_logit")
    clamp_alignment = alignment_rows("per_cell_clamp_differences.csv", "paired_logit")
    monotonic = monotonic_summary()
    write_csv("model_difference_summary.csv", model)
    write_csv("clamp_difference_summary.csv", clamp)
    write_csv("model_difference_cell_alignment.csv", model_alignment)
    write_csv("clamp_difference_cell_alignment.csv", clamp_alignment)
    write_csv("monotonicity_summary.csv", monotonic)
    (OUT / "summary_metrics.json").write_text(json.dumps({"model_differences": model,
        "clamp_differences": clamp, "bootstrap_samples": 100000}, indent=2), encoding="utf-8")
    print(f"WROTE {len(model)} model and {len(clamp)} clamp surface summaries")


if __name__ == "__main__":
    main()
