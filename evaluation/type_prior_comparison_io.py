from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch

from evaluation.rf_artifacts import (
    RFArtifactError,
    load_rf_artifact,
)
from evaluation.rf_history_contracts import (
    RFHistoryContractError,
    require_exact_history_contracts,
)
from evaluation.type_prior_comparison_types import (
    BASE_VARIANTS,
    HISTORY_CONTRACTS,
    REFERENCE_VARIANT,
    SHUFFLE_VARIANTS,
    JsonMap,
    JsonValue,
    RunGrid,
    RunInput,
    TypePriorComparisonError,
    Variant,
)

_VARIANT_BY_NAME: dict[str, Variant] = {
    "type_aware": "type_aware",
    "type_blind": "type_blind",
    "cell_only": "cell_only",
    "shuffled_type": "shuffled_type",
    "balanced_shuffled_type": "balanced_shuffled_type",
}


def load_comparable_grid(paths: list[Path]) -> tuple[tuple[RunInput, ...], RunGrid]:
    runs = tuple(_load_run(path) for path in paths)
    _require_comparable_contracts(runs)
    return runs, _variant_grid(runs)


def _load_run(path: Path) -> RunInput:
    manifest = _json_file(path / "run_manifest.json")
    metrics = _json_file(path / "final_metrics.json")
    status = _json_file(path / "run_status.json")
    _require_completed_validation(manifest, metrics, status)
    try:
        artifact = load_rf_artifact(path / "rf_artifacts.pt")
    except RFArtifactError as exc:
        raise TypePriorComparisonError(str(exc)) from exc
    dynamic_history = artifact.conditional_dynamic_by_history
    _require_history_contract(metrics["dynamic_rf"]["by_history"], "metrics")
    prediction = metrics["response_prediction"]["conditional"]
    initialized_prediction = metrics["response_prediction"]["initialized_conditional"]
    per_cell_nll = tuple(
        _finite_float(value, "per-cell nll")
        for value in prediction["per_cell_nll"]
    )
    if len(per_cell_nll) != len(artifact.cell_ids):
        raise TypePriorComparisonError("per-cell NLL identity mismatch")
    initialized_per_cell_nll = tuple(
        _finite_float(value, "initialized per-cell nll")
        for value in initialized_prediction["per_cell_nll"]
    )
    if len(initialized_per_cell_nll) != len(artifact.cell_ids):
        raise TypePriorComparisonError("initialized per-cell NLL identity mismatch")
    sharing = manifest["parameter_sharing"]
    teacher_errors = metrics["dynamic_rf"]["by_history"]["matched_observed"]["trained"].get(
        "teacher_primary_errors",
        [],
    )
    return RunInput(
        path=path,
        variant=_variant(manifest["parameter_sharing"]["mode"]),
        seed=int(manifest["config"]["seed"]),
        budget=int(manifest["config"]["training"]["max_optimizer_steps"]),
        fingerprint=str(manifest["dataset_fingerprint"]),
        source_pairs=_source_pair_count(metrics["dynamic_rf"]["by_history"]),
        nll=_finite_float(prediction["nll"], "nll"),
        initialized_nll=_finite_float(initialized_prediction["nll"], "initialized nll"),
        bits=_finite_float(prediction["micro_bits_per_spike"], "bits"),
        calibration=_finite_float(prediction["calibration_error"], "calibration"),
        per_cell_nll=per_cell_nll,
        initialized_per_cell_nll=initialized_per_cell_nll,
        teacher_error=_mean_or_none(tuple(float(value) for value in teacher_errors)),
        parameter_delta=_parameter_delta(path / "parameter_delta.json"),
        cell_ids=artifact.cell_ids,
        cone_positions=artifact.cone_positions_degs,
        lag_order=artifact.lag_order,
        low_kernel=_kernel_array(
            dynamic_history["matched_observed"]["trained_low"],
        ),
        high_kernel=_kernel_array(
            dynamic_history["matched_observed"]["trained_high"],
        ),
        initialized_low_kernel=_kernel_array(
            dynamic_history["matched_observed"]["initialized_low"],
        ),
        initialized_high_kernel=_kernel_array(
            dynamic_history["matched_observed"]["initialized_high"],
        ),
        matched_initialization=bool(sharing.get("matched_initialization", False)),
        shuffle_contract=str(sharing.get("shuffle_contract", "legacy_unrecorded")),
        effective_type_labels=tuple(str(value) for value in sharing["effective_type_labels"]),
        initial_effective_parameters=_initial_effective_parameters(
            sharing.get("initial_effective_parameters")
        ),
        observed_type_labels=tuple(
            str(value) for value in sharing.get("observed_type_labels", ())
        ),
        cell_polarities=tuple(int(value) for value in sharing.get("cell_polarities", ())),
    )


def _json_file(path: Path) -> JsonMap:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypePriorComparisonError(f"{path.name} must be a JSON object")
    return value


def _require_completed_validation(manifest: JsonMap, metrics: JsonMap, status: JsonMap) -> None:
    splits = (
        manifest.get("evaluation_split"),
        metrics.get("evaluation_split"),
        status.get("evaluation_split"),
    )
    if splits != ("validation", "validation", "validation"):
        raise TypePriorComparisonError("type-prior comparator only accepts validation split runs")
    if status.get("status") != "COMPLETED":
        raise TypePriorComparisonError("type-prior comparator only accepts completed runs")


def _variant(value: JsonValue) -> Variant:
    if not isinstance(value, str) or value not in _VARIANT_BY_NAME:
        raise TypePriorComparisonError(f"unknown parameter-sharing variant: {value!r}")
    return _VARIANT_BY_NAME[value]


def _require_history_contract(value: JsonValue, label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypePriorComparisonError(f"{label} history contract must be exact")
    try:
        require_exact_history_contracts(value)
    except RFHistoryContractError as exc:
        raise TypePriorComparisonError(
            f"{label} history contract must be exact"
        ) from exc
    if tuple(value) != HISTORY_CONTRACTS:
        raise TypePriorComparisonError(f"{label} history contract must be exact")


def _kernel_array(value: torch.Tensor):
    return value.detach().cpu().numpy()


def _source_pair_count(by_history: JsonMap) -> int:
    counts = tuple(
        int(by_history[history]["trained"]["pair_count"])
        for history in HISTORY_CONTRACTS
    )
    if len(set(counts)) != 1:
        raise TypePriorComparisonError("source-pair count must match across histories")
    return counts[0]


def _finite_float(value: JsonValue, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TypePriorComparisonError(f"{label} must be finite")
    return number


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _parameter_delta(path: Path) -> tuple[float, ...]:
    values = json.loads(path.read_text(encoding="utf-8"))
    deltas: list[float] = []
    for item in values:
        deltas.extend(float(value) for value in item["delta_values"])
    return tuple(deltas)


def _initial_effective_parameters(value: JsonValue) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypePriorComparisonError("initial effective parameters must be a mapping")
    flattened = []
    for name in sorted(value):
        values = value[name]
        if not isinstance(values, list):
            raise TypePriorComparisonError("initial effective parameters must be vectors")
        flattened.extend(
            _finite_float(item, "initial effective parameter") for item in values
        )
    return tuple(flattened)


def _require_comparable_contracts(runs: tuple[RunInput, ...]) -> None:
    if not runs:
        raise TypePriorComparisonError("at least one run directory is required")
    first = runs[0]
    for run in runs[1:]:
        if run.fingerprint != first.fingerprint:
            raise TypePriorComparisonError("dataset fingerprint mismatch")
        if run.cell_ids != first.cell_ids:
            raise TypePriorComparisonError("cell identity mismatch")
        if run.cone_positions != first.cone_positions:
            raise TypePriorComparisonError("cone identity mismatch")
        if run.lag_order != first.lag_order:
            raise TypePriorComparisonError("lag order mismatch")
        if run.source_pairs != first.source_pairs:
            raise TypePriorComparisonError("source-pair count mismatch")
        if run.matched_initialization != first.matched_initialization:
            raise TypePriorComparisonError("matched-initialization contract mismatch")
        if run.matched_initialization and (
            not run.initial_effective_parameters
            or len(run.initial_effective_parameters)
            != len(first.initial_effective_parameters)
        ):
            raise TypePriorComparisonError("initial effective parameter identity mismatch")


def _variant_grid(runs: tuple[RunInput, ...]) -> RunGrid:
    grid: RunGrid = defaultdict(dict)
    for run in runs:
        key = (run.seed, run.budget)
        if run.variant in grid[key]:
            raise TypePriorComparisonError("duplicate variant for seed and budget")
        grid[key][run.variant] = run
    for variants in grid.values():
        missing = set(BASE_VARIANTS) - set(variants)
        if missing:
            raise TypePriorComparisonError("every seed/budget needs a complete variant grid")
        shuffle = set(variants).intersection(SHUFFLE_VARIANTS)
        if len(shuffle) != 1:
            raise TypePriorComparisonError(
                "every seed/budget needs exactly one shuffled-type control"
            )
    budget_sets: dict[int, set[int]] = defaultdict(set)
    for seed, budget in grid:
        budget_sets[seed].add(budget)
    if len({tuple(sorted(budgets)) for budgets in budget_sets.values()}) != 1:
        raise TypePriorComparisonError("training budget mismatch across seeds")
    shuffle_modes = {
        next(iter(set(variants).intersection(SHUFFLE_VARIANTS)))
        for variants in grid.values()
    }
    if len(shuffle_modes) != 1:
        raise TypePriorComparisonError("shuffled-type control mismatch across seeds")
    has_reference = tuple(REFERENCE_VARIANT in variants for variants in grid.values())
    if any(has_reference) and not all(has_reference):
        raise TypePriorComparisonError("cell-only reference must cover every seed/budget")
    return dict(grid)


__all__ = ["load_comparable_grid"]
