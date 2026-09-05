from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from evaluation.mechanistic_retina.ac_circuit_inputs import (
    CheckpointPayload,
    DataPayload,
    rebuild_model,
    validation_tensors,
)
from evaluation.mechanistic_retina.ac_circuit_lineage import (
    file_sha256,
    implementation_source_sha256,
)
from evaluation.mechanistic_retina.ac_circuit_support import (
    AC_RF_ARTIFACT_SCHEMA,
    AC_RF_ESTIMAND,
    JsonValue,
    parameter_invariance,
    response_block,
    response_metrics,
    state_snapshot,
    tensor_change_metrics,
    validate_ac_rf_artifact,
)
from evaluation.mechanistic_retina.rf_effective import effective_rf
from evaluation.mechanistic_retina.spike_banks import tensor_sha256
from models.mechanistic_retina.contracts import PathwayClamp

_AC_CLAMPS = frozenset(
    {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
)


@dataclass(frozen=True, slots=True)
class ACCircuitPerturbationResult:
    artifact_dir: Path
    parameter_state_unchanged: bool
    mean_absolute_logit_change: float
    mean_absolute_probability_change: float
    global_rf_cosine: float
    temporal_rf_cosine: float


class ACCircuitPerturbationError(ValueError):
    pass


def run_ac_circuit_perturbation(
    benchmark_dir: Path,
    output_dir: Path,
) -> ACCircuitPerturbationResult:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("AC perturbation output directory must be empty")
    checkpoint_path = benchmark_dir / "student-trained.pt"
    data_path = benchmark_dir / "sampled-data.pt"
    checkpoint: CheckpointPayload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    data: DataPayload = torch.load(data_path, map_location="cpu", weights_only=True)
    model = rebuild_model(checkpoint)
    validation_cones, validation_spikes = validation_tensors(data, checkpoint)
    stimulus_count, trial_count, time_steps, cell_count = validation_spikes.shape
    repeated_cones = validation_cones[:, None].expand(
        -1, trial_count, -1, -1
    ).reshape(stimulus_count * trial_count, time_steps, -1)
    histories = validation_spikes.reshape(
        stimulus_count * trial_count, time_steps, cell_count
    )
    state_before = state_snapshot(model)

    model.eval()
    with torch.no_grad():
        normal_output = model.forward_sequence(
            repeated_cones, observed_counts=histories
        )
        clamped_output = model.forward_sequence(
            repeated_cones,
            observed_counts=histories,
            clamps=_AC_CLAMPS,
        )
    normal_response = response_block(
        normal_output.logits,
        normal_output.spike_probability,
        normal_output.amacrine_local_current,
        normal_output.amacrine_transient_current,
        stimulus_count,
        trial_count,
    )
    clamped_response = response_block(
        clamped_output.logits,
        clamped_output.spike_probability,
        clamped_output.amacrine_local_current,
        clamped_output.amacrine_transient_current,
        stimulus_count,
        trial_count,
    )
    normal_rf = effective_rf(model, repeated_cones, histories).reshape(
        stimulus_count, trial_count, cell_count, model.config.lag_steps, -1
    )
    clamped_rf = effective_rf(
        model,
        repeated_cones,
        histories,
        clamps=_AC_CLAMPS,
    ).reshape_as(normal_rf)
    normal_temporal = normal_rf.sum(dim=-1)
    clamped_temporal = clamped_rf.sum(dim=-1)
    ac_rf = normal_rf - clamped_rf
    state_after = state_snapshot(model)
    invariance = parameter_invariance(model, state_before, state_after)

    response_delta = {
        "logits": clamped_response["logits"] - normal_response["logits"],
        "spike_probability": clamped_response["spike_probability"]
        - normal_response["spike_probability"],
    }
    ac_response_contribution = {
        name: -value for name, value in response_delta.items()
    }
    artifact_identity: dict[str, JsonValue] = {
        "rf_estimand": AC_RF_ESTIMAND,
        "lag_order": list(range(model.config.lag_steps)),
        "lag_order_semantics": "oldest_to_current",
        "cell_order": list(range(cell_count)),
        "cell_types": list(checkpoint["cell_types"]),
        "polarities": list(checkpoint["polarities"]),
        "cell_positions_degs": checkpoint["cell_positions"].tolist(),
        "cone_order": list(range(repeated_cones.shape[-1])),
        "cone_positions_degs": checkpoint["cone_positions"].tolist(),
        "validation_split": "validation",
        "validation_context": "all validation stimuli and sampled-history trials",
        "validation_cones_sha256": tensor_sha256(validation_cones),
        "validation_spikes_sha256": tensor_sha256(validation_spikes),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "sampled_data_sha256": file_sha256(data_path),
        "source_sha256": implementation_source_sha256(),
    }
    rf_tensors = {
        "schema": AC_RF_ARTIFACT_SCHEMA,
        "schema_revision": 1,
        "identity": artifact_identity,
        "normal": {
            "global": normal_rf,
            "temporal": normal_temporal,
            "ac_pathway": ac_rf,
        },
        "ac_structural_clamp": {
            "global": clamped_rf,
            "temporal": clamped_temporal,
            "ac_pathway": torch.zeros_like(ac_rf),
        },
        "clamp_minus_normal": {
            "global": clamped_rf - normal_rf,
            "temporal": clamped_temporal - normal_temporal,
        },
        "ac_contribution": {
            "global": normal_rf - clamped_rf,
            "temporal": normal_temporal - clamped_temporal,
        },
    }
    validate_ac_rf_artifact(rf_tensors)
    response_change = response_metrics(response_delta, normal_response, clamped_response)
    global_change = tensor_change_metrics(normal_rf, clamped_rf)
    temporal_change = tensor_change_metrics(normal_temporal, clamped_temporal)
    exact_zero = float(clamped_response["ac_total_current"].abs().max()) == 0.0
    payload: dict[str, JsonValue] = {
        "analysis": "AC in-silico structural circuit perturbation",
        "model_class": "physiology-constrained recurrent/state-space retinal point-process model",
        "execution": {
            "training_performed": False,
            "optimizer_created": False,
            "checkpoint_role": checkpoint["role"],
            "clamps": sorted(clamp.value for clamp in _AC_CLAMPS),
        },
        "lineage": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "sampled_data": str(data_path.resolve()),
            "sampled_data_sha256": file_sha256(data_path),
            "validation_cones_sha256": tensor_sha256(validation_cones),
            "validation_spikes_sha256": tensor_sha256(validation_spikes),
        },
        "rf_artifact_identity": artifact_identity,
        "validation": {
            "stimulus_count": stimulus_count,
            "trial_count": trial_count,
            "time_steps": time_steps,
            "cell_count": cell_count,
            "rf_context": "all validation stimulus/trial histories",
            "temporal_rf_definition": "signed sum of global RF over cone dimension",
        },
        "parameter_invariance": invariance,
        "structural_clamp": {
            "ac_current_exact_zero": exact_zero,
            "clamped_ac_current_max_abs": float(
                clamped_response["ac_total_current"].abs().max()
            ),
        },
        "response_change": response_change,
        "global_rf_change": global_change,
        "temporal_rf_change": temporal_change,
        "ac_contribution": {
            "mean_absolute_logit": float(ac_response_contribution["logits"].abs().mean()),
            "mean_absolute_probability": float(
                ac_response_contribution["spike_probability"].abs().mean()
            ),
            "global_rf_norm": float(torch.linalg.vector_norm(ac_rf)),
            "temporal_rf_norm": float(
                torch.linalg.vector_norm(normal_temporal - clamped_temporal)
            ),
            "normal_ac_current_mean_abs": float(
                normal_response["ac_total_current"].abs().mean()
            ),
        },
    }
    if not exact_zero or not bool(invariance["all_parameters_unchanged"]):
        raise ACCircuitPerturbationError(
            "AC clamp failed exact-zero or parameter-invariance contract"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
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
    return ACCircuitPerturbationResult(
        output_dir,
        True,
        float(response_change["mean_absolute_logit_change"]),
        float(response_change["mean_absolute_probability_change"]),
        float(global_change["cosine"]),
        float(temporal_change["cosine"]),
    )


__all__ = [
    "ACCircuitPerturbationError",
    "ACCircuitPerturbationResult",
    "run_ac_circuit_perturbation",
]
