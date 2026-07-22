from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluation.dynamic_rf import DynamicRFSelection, DynamicRFUnitResult
from evaluation.dynamic_rf_summary import (
    DynamicRFComparisonSummary,
    DynamicRFSourceSummary,
)
from evaluation.reconstruction import ReconstructionMetrics
from evaluation.rgc_types import FEATURE_NAMES, RGCTypeReport
from training.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    representation_passed: bool
    energy_passed: bool
    energy_status: str
    reconstruction: ReconstructionMetrics
    target_energy_ratio: float | None
    dynamic_rf_units: int
    dynamic_rf_status: str
    rgc_type_status: str


def summarize_evaluation(
    reconstruction: ReconstructionMetrics,
    target_energy_ratio: float | None,
    dynamic_rf: Sequence[DynamicRFUnitResult],
    config: ExperimentConfig,
    *,
    dynamic_rf_status: str,
    rgc_type_status: str,
    budget_ramp_complete: bool,
) -> EvaluationSummary:
    energy_passed = bool(
        budget_ramp_complete
        and target_energy_ratio is not None
        and target_energy_ratio <= config.evaluation.maximum_energy_budget_ratio
    )
    return EvaluationSummary(
        representation_passed=(
            reconstruction.representation_skill
            >= config.evaluation.minimum_representation_skill
        ),
        energy_passed=energy_passed,
        energy_status=(
            "not_identifiable"
            if target_energy_ratio is None
            else "passed" if energy_passed else "failed"
        ),
        reconstruction=reconstruction,
        target_energy_ratio=target_energy_ratio,
        dynamic_rf_units=len(dynamic_rf),
        dynamic_rf_status=dynamic_rf_status,
        rgc_type_status=rgc_type_status,
    )


def write_evaluation_report(
    output_dir: str | Path,
    summary: EvaluationSummary,
    trained_dynamic_rf: Sequence[DynamicRFUnitResult],
    initialized_dynamic_rf: Sequence[DynamicRFUnitResult],
    dynamic_rf_selection: Sequence[DynamicRFSelection],
    dynamic_rf_comparison: DynamicRFComparisonSummary,
    dynamic_rf_sources: Sequence[DynamicRFSourceSummary],
    rgc_types: RGCTypeReport | None,
    config: ExperimentConfig,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "summary": asdict(summary),
        "dynamic_rf": {
            "selection": [asdict(selection) for selection in dynamic_rf_selection],
            "comparison": asdict(dynamic_rf_comparison),
            "sources": [asdict(source) for source in dynamic_rf_sources],
            "trained_units": [asdict(row) for row in trained_dynamic_rf],
            "initialized_units": [asdict(row) for row in initialized_dynamic_rf],
        },
        "rgc_types": (
            {"status": summary.rgc_type_status}
            if rgc_types is None
            else {"feature_names": FEATURE_NAMES, **asdict(rgc_types)}
        ),
        "resolved_config": config.resolved(),
        "local_linear_baseline": config.evaluation.local_linear_baseline,
    }
    with (destination / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    ratio = (
        "not_identifiable"
        if summary.target_energy_ratio is None
        else f"{summary.target_energy_ratio:.6f}"
    )
    lines = [
        "# Retina RF SNN evaluation",
        "",
        f"- Representation skill: {summary.reconstruction.representation_skill:.6f}",
        f"- Representation gate: {summary.representation_passed}",
        f"- Target energy ratio: {ratio}",
        f"- Energy gate: {summary.energy_status}",
        f"- Dynamic RF unit records: {summary.dynamic_rf_units}",
        f"- Dynamic RF status: {summary.dynamic_rf_status}",
        f"- RGC type status: {summary.rgc_type_status}",
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
