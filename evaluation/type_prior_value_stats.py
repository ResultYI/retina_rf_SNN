from __future__ import annotations

import numpy as np

from evaluation.factorial_contrasts import FactorialContrasts, factorial_contrasts
from evaluation.rf_dynamic_metrics import bootstrap_ci
from evaluation.type_prior_comparison_types import (
    JsonMap,
    RunGrid,
    RunInput,
    Variant,
)


def predictive_comparison(grid: RunGrid, variant: Variant) -> JsonMap:
    nll = tuple(items[variant].nll - items["type_aware"].nll for items in grid.values())
    bits = tuple(items["type_aware"].bits - items[variant].bits for items in grid.values())
    calibration = tuple(
        items[variant].calibration - items["type_aware"].calibration
        for items in grid.values()
    )
    teacher = tuple(
        items[variant].teacher_error - items["type_aware"].teacher_error
        for items in grid.values()
        if items[variant].teacher_error is not None
        and items["type_aware"].teacher_error is not None
    )
    reference = next(iter(grid.values()))["type_aware"]
    per_cell = []
    for cell, cell_id in enumerate(reference.cell_ids):
        deltas = tuple(
            items[variant].per_cell_nll[cell]
            - items["type_aware"].per_cell_nll[cell]
            for items in grid.values()
        )
        per_cell.append(
            {
                "cell_id": cell_id,
                "paired_sample_count": len(deltas),
                "nll_delta": _mean(deltas),
                "nll_delta_ci": _ci(deltas),
            }
        )
    return {
        "bootstrap_unit": "seed",
        "paired_sample_count": len(nll),
        "nll_delta": _mean(nll),
        "nll_delta_ci": _ci(nll),
        "bits_delta": _mean(bits),
        "bits_delta_ci": _ci(bits),
        "calibration_delta": _mean(calibration),
        "calibration_delta_ci": _ci(calibration),
        "teacher_error_delta": None if not teacher else _mean(teacher),
        "teacher_error_delta_ci": None if not teacher else _ci(teacher),
        "per_cell_nll_delta": per_cell,
    }


def learning_gain_comparison(grid: RunGrid, variant: Variant) -> JsonMap:
    deltas = tuple(
        (items["type_aware"].initialized_nll - items["type_aware"].nll)
        - (items[variant].initialized_nll - items[variant].nll)
        for items in grid.values()
    )
    reference = next(iter(grid.values()))["type_aware"]
    per_cell = []
    for cell, cell_id in enumerate(reference.cell_ids):
        cell_deltas = tuple(
            (
                items["type_aware"].initialized_per_cell_nll[cell]
                - items["type_aware"].per_cell_nll[cell]
            )
            - (
                items[variant].initialized_per_cell_nll[cell]
                - items[variant].per_cell_nll[cell]
            )
            for items in grid.values()
        )
        per_cell.append(
            {
                "cell_id": cell_id,
                "learning_gain_delta": _mean(cell_deltas),
                "learning_gain_delta_ci": _ci(cell_deltas),
            }
        )
    return {
        "bootstrap_unit": "seed",
        "paired_sample_count": len(deltas),
        "learning_gain_delta": _mean(deltas),
        "learning_gain_delta_ci": _ci(deltas),
        "per_seed_learning_gain_delta": list(deltas),
        "seed_direction_consistent": all(value > 0 for value in deltas),
        "per_cell_learning_gain_delta": per_cell,
        "factorial_contrasts": _factorial_contrast_summary(grid, variant),
    }


def _factorial_contrast_summary(grid: RunGrid, variant: Variant) -> JsonMap:
    names = ("common", "type", "polarity", "interaction")
    samples: dict[str, dict[str, list[float]]] = {
        estimand: {name: [] for name in names}
        for estimand in ("step0_nll_delta", "learning_gain_delta", "step20_nll_delta")
    }
    for items in grid.values():
        aware = items["type_aware"]
        control = items[variant]
        initial = np.asarray(control.initialized_per_cell_nll) - np.asarray(
            aware.initialized_per_cell_nll
        )
        final = np.asarray(control.per_cell_nll) - np.asarray(aware.per_cell_nll)
        for estimand, values in (
            ("step0_nll_delta", initial),
            ("learning_gain_delta", final - initial),
            ("step20_nll_delta", final),
        ):
            contrast = _cell_factorial_contrast(
                values,
                aware.observed_type_labels,
                aware.cell_polarities,
            )
            if contrast is None:
                return {"identifiable": False}
            for name in names:
                samples[estimand][name].append(float(getattr(contrast, name)))
    return {
        "identifiable": True,
        "cell_order": ["midget_ON", "midget_OFF", "parasol_ON", "parasol_OFF"],
        "by_estimand": {
            estimand: {
                name: {
                    "mean": _mean(tuple(values)),
                    "ci": _ci(tuple(values)),
                    "per_seed": values,
                }
                for name, values in contrasts.items()
            }
            for estimand, contrasts in samples.items()
        },
    }


def _cell_factorial_contrast(
    values: np.ndarray,
    type_labels: tuple[str, ...],
    polarities: tuple[int, ...],
) -> FactorialContrasts | None:
    order = (("midget", 0), ("midget", 1), ("parasol", 0), ("parasol", 1))
    grouped = [
        [
            float(value)
            for value, label, polarity in zip(
                values,
                type_labels,
                polarities,
                strict=True,
            )
            if (label, polarity) == key
        ]
        for key in order
    ]
    if any(not group for group in grouped):
        return None
    return factorial_contrasts(np.asarray([np.mean(group) for group in grouped]))


def parameter_delta_variance(runs: tuple[RunInput, ...]) -> JsonMap:
    arrays = [np.asarray(run.parameter_delta, dtype=np.float64) for run in runs]
    if len(arrays) < 2:
        return {"sample_count": len(arrays), "mean_variance": None}
    width = min(array.shape[0] for array in arrays)
    matrix = np.stack([array[:width] for array in arrays])
    return {
        "sample_count": len(arrays),
        "mean_variance": float(matrix.var(axis=0).mean()),
    }


def matched_initialization_audit(
    grid: RunGrid,
    variants: tuple[Variant, ...],
) -> JsonMap:
    parameter_differences = []
    nll_differences = []
    for items in grid.values():
        aware = items["type_aware"]
        for variant in variants:
            control = items[variant]
            parameter_differences.extend(
                abs(left - right)
                for left, right in zip(
                    aware.initial_effective_parameters,
                    control.initial_effective_parameters,
                    strict=True,
                )
            )
            nll_differences.append(abs(aware.initialized_nll - control.initialized_nll))
    max_parameter = max(parameter_differences, default=float("inf"))
    max_nll = max(nll_differences, default=float("inf"))
    return {
        "parameter_tolerance": 1e-7,
        "step0_nll_tolerance": 1e-7,
        "max_effective_parameter_difference": max_parameter,
        "max_step0_nll_difference": max_nll,
        "passed": max_parameter < 1e-7 and max_nll < 1e-7,
    }


def _ci(values: tuple[float, ...]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    return list(bootstrap_ci(list(values), 1000, 0))


def _mean(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


__all__ = [
    "learning_gain_comparison",
    "matched_initialization_audit",
    "parameter_delta_variance",
    "predictive_comparison",
]
