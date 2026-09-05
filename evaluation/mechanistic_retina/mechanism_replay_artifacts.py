from __future__ import annotations

from collections.abc import Mapping

from evaluation.mechanistic_retina.mechanism_replay_types import (
    CheckpointManifestEntry,
    ReplayMetricComparison,
    ReplayRunSet,
)
from evaluation.mechanistic_retina.metrics import JsonValue


def checkpoint_manifest_payload(
    entries: tuple[CheckpointManifestEntry, ...],
) -> Mapping[str, JsonValue]:
    return {
        "schema": "mechanism-final-checkpoints-v1",
        "count": len(entries),
        "checkpoints": [
            {
                "path": entry.relative_path,
                "sha256": entry.saved.sha256,
                "bytes": entry.saved.bytes,
                "teacher_id": entry.saved.identity.teacher_id,
                "teacher_hash": entry.saved.identity.teacher_hash,
                "condition": entry.saved.identity.condition,
                "structural_variant": entry.saved.identity.structural_variant.value,
                "seed": entry.saved.identity.seed,
                "step": entry.saved.identity.step,
                "run_id": entry.saved.identity.run_id,
                "roundtrip_state_equal": entry.roundtrip_state_equal,
                "roundtrip_gate_difference": entry.roundtrip_gate_difference,
            }
            for entry in entries
        ],
    }


def replay_results_payload(run_set: ReplayRunSet) -> Mapping[str, JsonValue]:
    return {
        "schema": "mechanism-scientific-replay-v1",
        "thresholds": {
            "ce_difference_max": 1e-4,
            "rf_cosine_difference_max": 1e-4,
            "gate_difference_max": 1e-3,
        },
        "passed": run_set.passed(),
        "runs": [
            {
                "teacher": run.teacher,
                "structural_variant": run.ablation.value,
                "seed": run.seed,
                "validation_ce": run.validation_ce,
                "gates": {
                    "h1": run.gates.h1,
                    "ac_local": run.gates.ac_local,
                    "ac_transient": run.gates.ac_transient,
                    "history": run.gates.history,
                },
                "pathway_cosines": dict(run.pathway_cosines),
                "pathway_norms": dict(run.pathway_norms),
                "total_rf_global_cosine": run.rf.metric.global_cosine,
                "total_rf_temporal_cosine": run.rf.metric.temporal_cosine,
                "exact_cell_fraction": run.rf.metric.exact_fraction,
                "comparison": comparison_payload(comparison),
            }
            for run, comparison in zip(
                run_set.runs, run_set.comparisons, strict=True
            )
        ],
    }


def comparison_payload(value: ReplayMetricComparison) -> Mapping[str, JsonValue]:
    return {
        "ce_difference": value.ce_difference,
        "gate_difference": value.gate_difference,
        "pathway_rf_cosine_difference": value.pathway_rf_cosine_difference,
        "total_rf_global_difference": value.total_rf_global_difference,
        "exact_cell_fraction_difference": value.exact_cell_fraction_difference,
        "passed": value.passed,
    }


__all__ = [
    "checkpoint_manifest_payload",
    "comparison_payload",
    "replay_results_payload",
]
