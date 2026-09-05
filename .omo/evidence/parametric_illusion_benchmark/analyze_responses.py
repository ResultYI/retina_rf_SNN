from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final, Iterable

import numpy as np
import torch


OUT: Final = Path(__file__).resolve().parent
ACTIVE: Final = slice(45, 60)
BOOTSTRAP_SAMPLES: Final = 100_000
BOOTSTRAP_SEED: Final = 20260901
KEYS: Final = ("model", "mode", "family", "signature", "contrast", "extent_px")
Scalar = str | int | float | bool
Row = dict[str, Scalar]


def write_csv(path: Path, rows: list[Row]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_weights(cell_count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(cell_count, (BOOTSTRAP_SAMPLES, cell_count), generator=generator)
    weights = torch.zeros(BOOTSTRAP_SAMPLES, cell_count)
    weights.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float32))
    return weights / cell_count


def aggregate(rows: list[Row], cells: list[str], value_name: str,
              output_value_name: str) -> list[Row]:
    lookup = {(str(row["cell_id"]), *(row[key] for key in KEYS)): float(row[value_name]) for row in rows}
    keys = sorted({tuple(row[key] for key in KEYS) for row in rows})
    matrix = torch.tensor([[lookup[(cell, *key)] for key in keys] for cell in cells])
    weights = bootstrap_weights(len(cells))
    output: list[Row] = []
    for start in range(0, len(keys), 32):
        bootstrap = weights @ matrix[:, start : start + 32]
        lower = torch.quantile(bootstrap, 0.025, dim=0)
        upper = torch.quantile(bootstrap, 0.975, dim=0)
        for offset, key in enumerate(keys[start : start + 32]):
            values = matrix[:, start + offset]
            mean, lo, hi = float(values.mean()), float(lower[offset]), float(upper[offset])
            direction = "positive" if lo > 0 else "negative" if hi < 0 else "exact_zero" if mean == lo == hi == 0 else "CI_includes_zero"
            output.append(dict(zip(KEYS, key, strict=True)) | {
                "n_cells": len(cells), output_value_name: mean, "median": float(values.median()),
                "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi, "direction": direction,
                "positive_cells": int((values > 0).sum()), "negative_cells": int((values < 0).sum()),
                "exact_zero_cells": int((values == 0).sum()),
            })
    return output


def paired_rows(rows: list[Row], left: tuple[str, str], right: tuple[str, str],
                label: str) -> list[Row]:
    identity = ("cell_id", "group", "family", "signature", "contrast", "extent_px")
    right_lookup = {tuple(row[name] for name in identity): row for row in rows
                    if (row["model"], row["mode"]) == right}
    left_rows = [row for row in rows if (row["model"], row["mode"]) == left]
    output = []
    for row in left_rows:
        key = tuple(row[name] for name in identity)
        match = right_lookup[key]
        output.append({name: row[name] for name in identity} | dict(model=label, mode="difference",
                      paired_logit=float(row["paired_logit"]) - float(match["paired_logit"])))
    return output


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def monotonicity(values: Iterable[float], axis_values: Iterable[float]) -> Row:
    x, y = np.asarray(tuple(axis_values), dtype=float), np.asarray(tuple(values), dtype=float)
    delta = np.diff(y)
    endpoint = float(y[-1] - y[0])
    direction = "increasing" if endpoint > 0 else "decreasing" if endpoint < 0 else "flat"
    agreement = delta >= 0 if endpoint > 0 else delta <= 0 if endpoint < 0 else delta == 0
    rank_x, rank_y = average_ranks(x), average_ranks(y)
    correlation = 0.0 if rank_y.std() == 0 else float(np.corrcoef(rank_x, rank_y)[0, 1])
    return {"endpoint_change": endpoint, "direction": direction, "monotonic_fraction": float(agreement.mean()),
            "strictly_monotonic": bool((delta > 0).all() if endpoint > 0 else (delta < 0).all() if endpoint < 0 else False),
            "spearman_rho": correlation}


def monotonic_rows(rows: list[Row], entity: str) -> list[Row]:
    axes = (("contrast", "extent_px"), ("extent_px", "contrast"))
    output: list[Row] = []
    for axis, fixed in axes:
        groups: dict[tuple[Scalar, ...], list[Row]] = {}
        names = (entity, "model", "mode", "family", "signature", fixed)
        for row in rows:
            groups.setdefault(tuple(row[name] for name in names), []).append(row)
        for key, values in groups.items():
            values.sort(key=lambda row: float(row[axis]))
            expected = 6 if axis == "contrast" else 5 if values[0]["family"] == "SBC" else 6
            if len(values) != expected:
                continue
            output.append(dict(zip(names, key, strict=True)) | {"sweep_axis": axis, "points": len(values)}
                          | monotonicity((float(row["paired_logit"]) for row in values),
                                         (float(row[axis]) for row in values)))
    return output


def main() -> None:
    artifact = torch.load(OUT / "responses.pt", map_location="cpu", weights_only=True)
    cells = list(artifact["cells"])
    rows: list[Row] = []
    for cell_id, cell in artifact["cells"].items():
        for model in ("canonical", "ln", "cnn"):
            for mode, logits in cell[model].items():
                for comparison in artifact["comparisons"]:
                    value = float((logits[comparison["a"], ACTIVE] - logits[comparison["b"], ACTIVE]).mean())
                    rows.append({"cell_id": cell_id, "group": cell["group"], "model": model, "mode": mode,
                                 "family": comparison["family"], "signature": comparison["signature"],
                                 "contrast": comparison["contrast"], "extent_px": comparison["extent_px"],
                                 "paired_logit": value})
    cohort = aggregate(rows, cells, "paired_logit", "mean_paired_logit")
    model_differences = sum((paired_rows(rows, left, right, label) for left, right, label in (
        (("canonical", "normal"), ("ln", "normal"), "canonical_minus_ln"),
        (("canonical", "normal"), ("cnn", "normal"), "canonical_minus_cnn"),
        (("cnn", "normal"), ("ln", "normal"), "cnn_minus_ln"))), [])
    clamp_differences = sum((paired_rows(rows, ("canonical", mode), ("canonical", "normal"), f"{mode}_minus_normal")
                             for mode in ("H1_off", "direct_BC_off", "AC_off")), [])
    cohort_model = aggregate(model_differences, cells, "paired_logit", "mean_paired_logit_difference")
    cohort_clamp = aggregate(clamp_differences, cells, "paired_logit", "mean_paired_logit_difference")
    cohort_for_monotonic = [row | {"cell_id": "cohort", "paired_logit": row["mean_paired_logit"]} for row in cohort]
    write_csv(OUT / "per_cell_curves.csv", rows)
    write_csv(OUT / "cohort_curves.csv", cohort)
    write_csv(OUT / "per_cell_model_differences.csv", model_differences)
    write_csv(OUT / "cohort_model_differences.csv", cohort_model)
    write_csv(OUT / "per_cell_clamp_differences.csv", clamp_differences)
    write_csv(OUT / "cohort_clamp_differences.csv", cohort_clamp)
    write_csv(OUT / "per_cell_monotonicity.csv", monotonic_rows(rows, "cell_id"))
    write_csv(OUT / "cohort_monotonicity.csv", monotonic_rows(cohort_for_monotonic, "cell_id"))
    (OUT / "analysis_contract.json").write_text(json.dumps({"paired_logit_window_ms": [300, 400],
        "paired_bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "cell",
                             "cell_count": len(cells)}, "training_performed": False}, indent=2), encoding="utf-8")
    print(f"WROTE {len(rows)} per-cell curve points; {len(cohort)} cohort points")


if __name__ == "__main__":
    main()
