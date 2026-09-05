from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.ac_circuit_inputs import (
    CheckpointPayload,
    rebuild_model,
)
from evaluation.mechanistic_retina.ac_circuit_lineage import file_sha256
from evaluation.mechanistic_retina.ac_circuit_support import (
    JsonValue,
    parameter_invariance,
    response_block,
    response_metrics,
    state_snapshot,
    tensor_change_metrics,
)
from evaluation.mechanistic_retina.ac_temporal_probe import (
    TemporalProbeConfig,
    build_temporal_probe,
)
from evaluation.mechanistic_retina.ac_temporal_lineage import (
    TEMPORAL_RF_ARTIFACT_SCHEMA,
    temporal_rf_lineage,
    temporal_timing_contract,
)
from evaluation.mechanistic_retina.ac_temporal_support import (
    peak_latency_change,
    temporal_parameter_invariance,
    validate_temporal_rf_artifact,
)
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.mechanistic_retina.spike_banks import tensor_sha256
from models.mechanistic_retina.contracts import PathwayClamp


_AC_CLAMPS = frozenset({PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT})


class ACTemporalPerturbationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ACTemporalPerturbationResult:
    artifact_dir: Path
    parameter_state_unchanged: bool
    mean_absolute_logit_change: float
    mean_absolute_probability_change: float
    temporal_rf_cosine: float


def run_ac_temporal_circuit_perturbation(
    benchmark_dir: Path,
    output_dir: Path,
    *,
    probe_config: TemporalProbeConfig = TemporalProbeConfig(),
) -> ACTemporalPerturbationResult:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("AC temporal perturbation output directory must be empty")
    checkpoint_path = benchmark_dir / "student-trained.pt"
    checkpoint: CheckpointPayload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = rebuild_model(checkpoint)
    probe = build_temporal_probe(
        checkpoint["cone_positions"], model.config.dt_ms, probe_config
    )
    stimulus_count, time_steps, _ = probe.cone_response.shape
    cell_count = checkpoint["cell_positions"].shape[0]
    observed_history = torch.zeros(stimulus_count, time_steps, cell_count)
    state_before = state_snapshot(model)

    model.eval()
    with torch.no_grad():
        normal_output = model.forward_sequence(
            probe.cone_response, observed_counts=observed_history
        )
        clamped_output = model.forward_sequence(
            probe.cone_response,
            observed_counts=observed_history,
            clamps=_AC_CLAMPS,
        )
    normal_response = response_block(
        normal_output.logits,
        normal_output.spike_probability,
        normal_output.amacrine_local_current,
        normal_output.amacrine_transient_current,
        stimulus_count,
        1,
    )
    clamped_response = response_block(
        clamped_output.logits,
        clamped_output.spike_probability,
        clamped_output.amacrine_local_current,
        clamped_output.amacrine_transient_current,
        stimulus_count,
        1,
    )
    normal_global_rf = effective_rf(model, probe.cone_response, observed_history)
    clamped_global_rf = effective_rf(
        model,
        probe.cone_response,
        observed_history,
        clamps=_AC_CLAMPS,
    )
    normal_temporal_rf = normal_global_rf.sum(dim=-1)
    clamped_temporal_rf = clamped_global_rf.sum(dim=-1)
    ac_temporal_rf = normal_temporal_rf - clamped_temporal_rf

    state_after = state_snapshot(model)
    base_invariance = parameter_invariance(model, state_before, state_after)
    invariance = temporal_parameter_invariance(
        model, state_before, state_after, base_invariance
    )
    response_delta = {
        "logits": clamped_response["logits"] - normal_response["logits"],
        "spike_probability": clamped_response["spike_probability"]
        - normal_response["spike_probability"],
    }
    ac_response_contribution = {
        "logits": -response_delta["logits"],
        "spike_probability": -response_delta["spike_probability"],
        "ac_total_current": normal_response["ac_total_current"],
    }
    response_change = response_metrics(
        response_delta, normal_response, clamped_response
    )
    temporal_rf_change = tensor_change_metrics(normal_temporal_rf, clamped_temporal_rf)
    peaks = peak_latency_change(
        normal_response,
        clamped_response,
        probe.baseline_steps,
        probe.dt_ms,
    )
    exact_zero = float(clamped_response["ac_total_current"].abs().max()) == 0.0
    lineage = temporal_rf_lineage(checkpoint, checkpoint_path, probe)
    identity = lineage.identity
    rf_tensors = {
        "schema": TEMPORAL_RF_ARTIFACT_SCHEMA,
        "schema_revision": 1,
        "identity": identity,
        "normal": {
            "temporal": normal_temporal_rf,
            "ac_pathway": ac_temporal_rf,
        },
        "ac_structural_clamp": {
            "temporal": clamped_temporal_rf,
            "ac_pathway": torch.zeros_like(ac_temporal_rf),
        },
        "clamp_minus_normal": {
            "temporal": clamped_temporal_rf - normal_temporal_rf,
            "ac_pathway": -ac_temporal_rf,
        },
        "ac_contribution": {
            "temporal": ac_temporal_rf,
            "ac_pathway": ac_temporal_rf,
        },
    }
    validate_temporal_rf_artifact(rf_tensors)
    payload: dict[str, JsonValue] = {
        "analysis": "AC temporal in-silico structural circuit perturbation",
        "model_class": "physiology-constrained recurrent/state-space retinal point-process model",
        "execution": {
            "training_performed": False,
            "optimizer_created": False,
            "checkpoint_role": checkpoint["role"],
            "model_revision": checkpoint["model_revision"],
            "clamps": sorted(clamp.value for clamp in _AC_CLAMPS),
        },
        "probe": {
            "names": list(probe.names),
            "space": "cone representation",
            "observed_history_context": "all-zero",
            "dt_ms": probe.dt_ms,
            "time_steps": time_steps,
            "baseline_steps": probe.baseline_steps,
            "baseline_duration_ms": probe.baseline_steps * probe.dt_ms,
            "config": asdict(probe_config),
        },
        "lineage": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "probe_sha256": tensor_sha256(probe.cone_response),
            "source_sha256": dict(lineage.source_sha256),
        },
        "rf_artifact_identity": identity,
        "timing_contract": dict(temporal_timing_contract(model, invariance)),
        "parameter_invariance": dict(invariance),
        "structural_clamp": {
            "ac_current_exact_zero": exact_zero,
            "clamped_ac_current_max_abs": float(
                clamped_response["ac_total_current"].abs().max()
            ),
        },
        "response_change": response_change,
        "peak_latency_change": dict(peaks),
        "temporal_rf_change": temporal_rf_change,
        "ac_contribution": {
            "mean_absolute_logit": float(
                ac_response_contribution["logits"].abs().mean()
            ),
            "mean_absolute_probability": float(
                ac_response_contribution["spike_probability"].abs().mean()
            ),
            "normal_ac_current_mean_abs": float(
                normal_response["ac_total_current"].abs().mean()
            ),
            "temporal_rf_norm": float(torch.linalg.vector_norm(ac_temporal_rf)),
        },
    }
    if not exact_zero or not bool(invariance["all_parameters_unchanged"]):
        raise ACTemporalPerturbationError(
            "AC temporal clamp failed exact-zero or parameter-invariance contract"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "ac-temporal-probe-v1",
            "names": probe.names,
            "cone_response": probe.cone_response,
            "temporal_drive": probe.temporal_drive,
            "spatial_pattern": probe.spatial_pattern,
            "time_ms": probe.time_ms,
            "baseline_steps": probe.baseline_steps,
            "dt_ms": probe.dt_ms,
        },
        output_dir / "stimuli.pt",
    )
    torch.save(
        {
            "normal": normal_response,
            "ac_structural_clamp": clamped_response,
            "clamp_minus_normal": response_delta,
            "ac_contribution": ac_response_contribution,
        },
        output_dir / "responses.pt",
    )
    torch.save(rf_tensors, output_dir / "rf-tensors.pt")
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return ACTemporalPerturbationResult(
        output_dir,
        True,
        float(response_change["mean_absolute_logit_change"]),
        float(response_change["mean_absolute_probability_change"]),
        float(temporal_rf_change["cosine"]),
    )
