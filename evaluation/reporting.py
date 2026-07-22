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
    dynamic_rf_status: str
    rgc_type_status: str


def summarize_evaluation(
    reconstruction: ReconstructionMetrics,
    energy_budget_ratio: float,
    dynamic_rf: Sequence[DynamicRFUnitResult],
    config: ExperimentConfig,
    *,
    dynamic_rf_status: str,
    rgc_type_status: str,
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
        dynamic_rf_status=dynamic_rf_status,
        rgc_type_status=rgc_type_status,
    )


def write_evaluation_report(
    output_dir: str | Path,
    summary: EvaluationSummary,
    dynamic_rf: Sequence[DynamicRFUnitResult],
    rgc_types: RGCTypeReport | None,
    config: ExperimentConfig,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "summary": asdict(summary),
        "dynamic_rf": {
            "status": summary.dynamic_rf_status,
            "units": [asdict(row) for row in dynamic_rf],
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
    lines = [
        "# Retina RF SNN evaluation",
        "",
        f"- Representation skill: {summary.reconstruction.representation_skill:.6f}",
        f"- Representation gate: {summary.representation_passed}",
        f"- Energy budget ratio: {summary.energy_budget_ratio:.6f}",
        f"- Energy gate: {summary.energy_passed}",
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


def classify_dynamic_rf(dynamic_rf: Sequence[DynamicRFUnitResult]) -> str:
    if not dynamic_rf:
        return "not_identifiable"
    if all(
        row.finite_difference.status == "threshold_crossing_not_local"
        for row in dynamic_rf
    ):
        return "not_identifiable"
    shape_shift = float(
        np.median([row.gain_normalized_cosine_distance for row in dynamic_rf])
    )
    gain_shift = float(
        np.median(
            [abs(np.log(max(row.kernel_norm_ratio, 1e-12))) for row in dynamic_rf]
        )
    )
    if not np.isfinite(shape_shift) or not np.isfinite(gain_shift):
        return "not_identifiable"
    if shape_shift >= 0.05:
        return "supported"
    if gain_shift >= 0.05:
        return "gain_only"
    return "not_supported"


__all__ = [
    "EvaluationSummary",
    "classify_dynamic_rf",
    "summarize_evaluation",
    "write_evaluation_report",
]
