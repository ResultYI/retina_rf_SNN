from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation.dynamic_rf import DynamicRFUnitResult
from evaluation.reconstruction import ReconstructionMetrics
from evaluation.rgc_types import FEATURE_NAMES, RGCTypeReport
from training.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    representation_passed: bool
    energy_passed: bool
    reconstruction: ReconstructionMetrics
    energy_budget_ratio: float
    dynamic_rf_units: int


def summarize_evaluation(
    reconstruction: ReconstructionMetrics,
    energy_budget_ratio: float,
    dynamic_rf: Sequence[DynamicRFUnitResult],
    config: ExperimentConfig,
) -> EvaluationSummary:
    return EvaluationSummary(
        representation_passed=(
            reconstruction.representation_skill
            >= config.evaluation.minimum_representation_skill
        ),
        energy_passed=(
            energy_budget_ratio <= config.evaluation.maximum_energy_budget_ratio
        ),
        reconstruction=reconstruction,
        energy_budget_ratio=energy_budget_ratio,
        dynamic_rf_units=len(dynamic_rf),
    )


def write_evaluation_report(
    output_dir: str | Path,
    summary: EvaluationSummary,
    dynamic_rf: Sequence[DynamicRFUnitResult],
    rgc_types: RGCTypeReport,
    config: ExperimentConfig,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "summary": asdict(summary),
        "dynamic_rf": [asdict(row) for row in dynamic_rf],
        "rgc_types": {
            "feature_names": FEATURE_NAMES,
            "features": rgc_types.features,
            "standardized_features": rgc_types.standardized_features,
            "assignments": rgc_types.assignments,
            "cluster_names": rgc_types.cluster_names,
            "candidate_labels": rgc_types.candidate_labels,
        },
        "resolved_config": config.resolved(),
        "local_linear_baseline": config.evaluation.local_linear_baseline,
    }
    with (destination / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    lines = [
        "# Retina RF SNN evaluation",
        "",
        f"- Representation skill: {summary.reconstruction.representation_skill:.6f}",
        f"- Representation gate: {summary.representation_passed}",
        f"- Energy budget ratio: {summary.energy_budget_ratio:.6f}",
        f"- Energy gate: {summary.energy_passed}",
        f"- Dynamic RF unit records: {summary.dynamic_rf_units}",
        f"- RGC clusters: {', '.join(rgc_types.cluster_names)}",
    ]
    (destination / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


__all__ = [
    "EvaluationSummary",
    "summarize_evaluation",
    "write_evaluation_report",
]

