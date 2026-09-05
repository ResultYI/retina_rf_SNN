from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.karamanlis_v1_ac_runtime import (
    ACClampVerification,
)


@dataclass(frozen=True, slots=True)
class PerturbationPayloadRequest:
    clamps: Sequence[str]
    lineage: Mapping[str, JsonValue]
    checkpoint_stage: str
    checkpoint_best_step: int
    training_seed: int
    sequence_count: int
    source_image_count: int
    time_steps: int
    cell_count: int
    dt_ms: float
    stimulus_onset_step: int
    context_indices: Sequence[int]
    source_image_ids: Sequence[str]
    trial_indices: Sequence[int]
    artifact_identity: Mapping[str, JsonValue]
    timing_contract: Mapping[str, JsonValue]
    invariance: Mapping[str, JsonValue]
    clamp: ACClampVerification
    upstream_outputs_unchanged: bool
    summary: Mapping[str, JsonValue]


def build_results_payload(
    request: PerturbationPayloadRequest,
) -> dict[str, JsonValue]:
    return {
        "schema": "karamanlis-marmoset-v1-ac-perturbation-v2",
        "analysis": "exploratory AC structural in-silico perturbation",
        "execution": {
            "training_performed": False,
            "optimizer_created": False,
            "clamps": list(request.clamps),
        },
        "lineage": dict(request.lineage),
        "checkpoint_context": {
            "stage": request.checkpoint_stage,
            "best_step": request.checkpoint_best_step,
            "training_seed": request.training_seed,
            "source_image_disjoint_split": True,
            "training_source_image_count": 176,
            "validation_source_image_count": 44,
        },
        "validation": {
            "sequence_count": request.sequence_count,
            "source_image_count": request.source_image_count,
            "time_steps": request.time_steps,
            "cell_count": request.cell_count,
            "native_dt_ms": request.dt_ms,
            "stimulus_onset_step": request.stimulus_onset_step,
            "stimulus_onset_ms": request.stimulus_onset_step * request.dt_ms,
            "rf_context_count": len(request.context_indices),
            "rf_context": "one recorded held-out trial per source image",
            "selected_sequence_indices": list(request.context_indices),
            "selected_source_image_ids": list(request.source_image_ids),
            "selected_trial_indices": list(request.trial_indices),
        },
        "rf_artifact_identity": dict(request.artifact_identity),
        "metric_definitions": {
            "response_change": "AC-off minus normal, averaged over held-out sequences and time bins per cell",
            "response_peak_magnitude": "maximum absolute deviation from each sequence's mean pre-flash baseline, then averaged per cell",
            "response_peak_latency": "time from flash onset to each sequence's maximum absolute baseline deviation, then averaged per cell",
            "temporal_rf": request.artifact_identity["rf_estimand"],
        },
        "timing_contract": dict(request.timing_contract),
        "parameter_invariance": dict(request.invariance),
        "structural_clamp": {
            "ac_current_exact_zero": (
                request.clamp.local_exact_zero
                and request.clamp.transient_exact_zero
            ),
            "ac_local_current_exact_zero": request.clamp.local_exact_zero,
            "ac_transient_current_exact_zero": request.clamp.transient_exact_zero,
            "clamped_ac_local_current_max_abs": request.clamp.local_max_abs,
            "clamped_ac_transient_current_max_abs": request.clamp.transient_max_abs,
            "h1_bc_and_ac_pre_gate_states_exactly_unchanged": request.upstream_outputs_unchanged,
        },
        "summary": dict(request.summary),
    }


__all__ = ["PerturbationPayloadRequest", "build_results_payload"]
