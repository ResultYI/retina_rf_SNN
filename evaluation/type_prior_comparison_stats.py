from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from evaluation.type_prior_comparison_types import (
    HISTORY_CONTRACTS,
    JsonMap,
    JsonValue,
    RunGrid,
    RunInput,
    ValueStatus,
    Variant,
)


def stability_summary(
    runs: tuple[RunInput, ...],
    *,
    learned_delta: bool = False,
) -> JsonMap:
    values = _stability_values(runs, learned_delta=learned_delta)
    per_cell = _stability_values_by_cell(runs, learned_delta=learned_delta)
    result: JsonMap = {
        "paired_sample_count": len(values),
        "mean_cosine": _mean(values),
        "ci": _ci(values),
        "per_cell": [
            {
                "cell_id": cell_id,
                "paired_sample_count": len(cell_values),
                "mean_cosine": _mean(cell_values),
                "ci": _ci(cell_values),
            }
            for cell_id, cell_values in zip(runs[0].cell_ids, per_cell, strict=True)
        ],
    }
    if learned_delta:
        norms = tuple(
            float(np.linalg.norm(kernel))
            for run in runs
            for kernel in (
                run.low_kernel - run.initialized_low_kernel,
                run.high_kernel - run.initialized_high_kernel,
            )
        )
        trained_initialized_distances = tuple(
            1.0 - cosine
            for run in runs
            for cosine in (
                *_kernel_cosines(run.low_kernel, run.initialized_low_kernel),
                *_kernel_cosines(run.high_kernel, run.initialized_high_kernel),
            )
        )
        result["mean_delta_norm"] = _mean(norms)
        result["delta_norm_ci"] = _ci(norms)
        result["mean_trained_initialized_cosine_distance"] = _mean(
            trained_initialized_distances
        )
        result["trained_initialized_cosine_distance_ci"] = _ci(
            trained_initialized_distances
        )
    return result


def stability_comparison(
    grid: RunGrid,
    variant: Variant,
    *,
    learned_delta: bool = False,
) -> JsonMap:
    aware_runs = tuple(items["type_aware"] for items in grid.values())
    other_runs = tuple(items[variant] for items in grid.values())
    aware = _stability_values(aware_runs, learned_delta=learned_delta)
    other = _stability_values(other_runs, learned_delta=learned_delta)
    count = min(len(aware), len(other))
    deltas = tuple(aware[index] - other[index] for index in range(count))
    aware_by_cell = _stability_values_by_cell(
        aware_runs,
        learned_delta=learned_delta,
    )
    other_by_cell = _stability_values_by_cell(
        other_runs,
        learned_delta=learned_delta,
    )
    per_cell = []
    for cell_id, aware_values, other_values in zip(
        aware_runs[0].cell_ids,
        aware_by_cell,
        other_by_cell,
        strict=True,
    ):
        cell_deltas = tuple(
            left - right
            for left, right in zip(aware_values, other_values, strict=True)
        )
        per_cell.append(
            {
                "cell_id": cell_id,
                "paired_sample_count": len(cell_deltas),
                "cosine_delta": _mean(cell_deltas),
                "cosine_delta_ci": _ci(cell_deltas),
            }
        )
    return {
        "paired_sample_count": count,
        "cosine_delta": _mean(deltas),
        "cosine_delta_ci": _ci(deltas),
        "per_cell_cosine_delta": per_cell,
    }


def value_status(
    intervals: tuple[JsonValue, JsonValue],
    *,
    identifiable: bool,
) -> ValueStatus:
    if not identifiable:
        return "not_identifiable"
    lows = tuple(float(interval[0]) for interval in intervals)
    highs = tuple(float(interval[1]) for interval in intervals)
    if all(low > 0 for low in lows):
        return "supported"
    if any(high < 0 for high in highs):
        return "significant_disadvantage"
    return "not_supported"


def run_contract(runs: tuple[RunInput, ...]) -> JsonMap:
    return {
        "dataset_fingerprint": runs[0].fingerprint,
        "evaluation_split": "validation",
        "history_contracts": list(HISTORY_CONTRACTS),
        "source_pair_count": runs[0].source_pairs,
        "cell_ids": list(runs[0].cell_ids),
        "cone_positions_degs": [list(row) for row in runs[0].cone_positions],
        "lag_order": runs[0].lag_order,
        "seeds": sorted({run.seed for run in runs}),
        "trial_budgets": sorted({run.budget for run in runs}),
        "matched_initialization": runs[0].matched_initialization,
        "shuffle_contracts": sorted({run.shuffle_contract for run in runs}),
        "shuffle_assignments": sorted(
            {
                tuple(run.effective_type_labels)
                for run in runs
                if run.variant in ("shuffled_type", "balanced_shuffled_type")
            }
        ),
        "input_runs": [
            {
                "path": str(run.path),
                "variant": run.variant,
                "seed": run.seed,
                "training_budget": run.budget,
            }
            for run in sorted(runs, key=lambda item: (item.budget, item.seed, item.variant))
        ],
    }


def _stability_values(
    runs: tuple[RunInput, ...],
    *,
    learned_delta: bool,
) -> tuple[float, ...]:
    values = []
    by_budget: dict[int, list[RunInput]] = defaultdict(list)
    for run in runs:
        by_budget[run.budget].append(run)
    for budget_runs in by_budget.values():
        ordered = sorted(budget_runs, key=lambda run: run.seed)
        for left, right in combinations(ordered, 2):
            values.extend(
                _kernel_cosines(
                    _kernel(left, high=False, learned_delta=learned_delta),
                    _kernel(right, high=False, learned_delta=learned_delta),
                )
            )
            values.extend(
                _kernel_cosines(
                    _kernel(left, high=True, learned_delta=learned_delta),
                    _kernel(right, high=True, learned_delta=learned_delta),
                )
            )
    return tuple(values)


def _stability_values_by_cell(
    runs: tuple[RunInput, ...],
    *,
    learned_delta: bool,
) -> tuple[tuple[float, ...], ...]:
    values: list[list[float]] = [[] for _ in runs[0].cell_ids]
    by_budget: dict[int, list[RunInput]] = defaultdict(list)
    for run in runs:
        by_budget[run.budget].append(run)
    for budget_runs in by_budget.values():
        ordered = sorted(budget_runs, key=lambda run: run.seed)
        for left, right in combinations(ordered, 2):
            for cosines in (
                _kernel_cosines(
                    _kernel(left, high=False, learned_delta=learned_delta),
                    _kernel(right, high=False, learned_delta=learned_delta),
                ),
                _kernel_cosines(
                    _kernel(left, high=True, learned_delta=learned_delta),
                    _kernel(right, high=True, learned_delta=learned_delta),
                ),
            ):
                for cell, value in enumerate(cosines):
                    values[cell].append(value)
    return tuple(tuple(cell_values) for cell_values in values)


def _kernel(run: RunInput, *, high: bool, learned_delta: bool) -> np.ndarray:
    trained = run.high_kernel if high else run.low_kernel
    if not learned_delta:
        return trained
    initialized = run.initialized_high_kernel if high else run.initialized_low_kernel
    return trained - initialized


def _kernel_cosines(left: np.ndarray, right: np.ndarray) -> tuple[float, ...]:
    values = []
    for cell in range(left.shape[0]):
        a = left[cell].reshape(-1)
        b = right[cell].reshape(-1)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        values.append(0.0 if denominator == 0 else float(np.dot(a, b) / denominator))
    return tuple(values)


def _ci(values: tuple[float, ...]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = np.random.default_rng(0)
    samples = np.asarray(values, dtype=np.float64)
    draws = rng.choice(samples, size=(1000, samples.shape[0]), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


__all__ = [
    "run_contract",
    "stability_comparison",
    "stability_summary",
    "value_status",
]
