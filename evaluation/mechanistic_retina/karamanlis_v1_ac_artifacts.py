from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import torch

from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.ac_circuit_lineage import file_sha256
from evaluation.mechanistic_retina.atomic_artifacts import atomic_torch_save
from evaluation.mechanistic_retina.karamanlis_v1_ac_runtime import CollectedResponses
from evaluation.mechanistic_retina.spike_banks import tensor_sha256

V1_AC_TEMPORAL_RF_SCHEMA: Final = "karamanlis-marmoset-v1-ac-temporal-rf-v2"
V1_AC_RF_ESTIMAND: Final = (
    "final-bin conditional RGC-logit Jacobian averaged over one recorded "
    "validation trial per held-out natural image"
)
V1_AC_TEMPORAL_RF_DEFINITION: Final = (
    "signed sum of final-bin global RF over cone dimension"
)
V1_AC_HISTORY_CONTEXT: Final = (
    "recorded Bernoulli validation spike events with the model's causal "
    "one-bin RGC history shift"
)


class ArtifactSourcePaths(Protocol):
    session_dir: Path
    graph_dir: Path
    checkpoint_path: Path


class V1ACTemporalRFArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PerturbationArtifactRequest:
    output_dir: Path
    responses: CollectedResponses
    normal_temporal_rf: torch.Tensor
    clamped_temporal_rf: torch.Tensor
    lag_ms: torch.Tensor
    identity: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class TemporalRFIdentityRequest:
    checkpoint_sha256: str
    checkpoint_stage: str
    checkpoint_best_step: int
    training_seed: int
    model_revision: int
    dt_ms: float
    lag_ms: torch.Tensor
    cell_ids: Sequence[str]
    cell_types: Sequence[str]
    polarities: Sequence[str]
    context_indices: Sequence[int]
    source_image_ids: Sequence[str]
    trial_indices: Sequence[int]
    source_sha256: Mapping[str, str]


def build_lineage(
    config: ArtifactSourcePaths,
    cones: torch.Tensor,
    spikes: torch.Tensor,
) -> dict[str, JsonValue]:
    training_results = config.checkpoint_path.parent / "results.json"
    lineage: dict[str, JsonValue] = {
        "checkpoint": str(config.checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(config.checkpoint_path),
        "validation_cones_sha256": tensor_sha256(cones),
        "validation_spikes_sha256": tensor_sha256(spikes),
        "expdata_sha256": file_sha256(config.session_dir / "expdata.mat"),
        "imagesequence_sha256": file_sha256(
            config.session_dir / "imagesequence_data.mat"
        ),
        "locality_graph_sha256": file_sha256(config.graph_dir / "locality_graph.npz"),
        "source_sha256": dict(implementation_source_sha256()),
    }
    if training_results.is_file():
        lineage["training_results"] = str(training_results.resolve())
        lineage["training_results_sha256"] = file_sha256(training_results)
    return lineage


def implementation_source_sha256() -> Mapping[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    analysis_names = (
        "ac_circuit_lineage.py",
        "ac_circuit_support.py",
        "ac_temporal_lineage.py",
        "ac_temporal_support.py",
        "atomic_artifacts.py",
        "karamanlis_v1_ac_artifacts.py",
        "karamanlis_v1_ac_metrics.py",
        "karamanlis_v1_ac_payload.py",
        "karamanlis_v1_ac_perturbation.py",
        "karamanlis_v1_ac_reporting.py",
        "karamanlis_v1_ac_runtime.py",
        "karamanlis_v1_rf_artifacts.py",
        "karamanlis_v1_rf_validation.py",
        "rf_effective.py",
        "spike_banks.py",
    )
    analysis_paths = tuple(
        repo_root / "evaluation/mechanistic_retina" / name for name in analysis_names
    ) + (repo_root / "evaluation/json_types.py",)
    data_paths = tuple(sorted((repo_root / "data").glob("karamanlis*.py"))) + (
        repo_root / "data/retinal_recording.py",
    )
    model_paths = tuple(sorted((repo_root / "models/mechanistic_retina").glob("*.py")))
    script_path = repo_root / "scripts/run_karamanlis_v1_ac_perturbation.py"
    paths = tuple(sorted((*analysis_paths, *data_paths, *model_paths, script_path)))
    return {
        path.relative_to(repo_root).as_posix(): file_sha256(path) for path in paths
    }


def build_temporal_rf_identity(
    request: TemporalRFIdentityRequest,
) -> dict[str, JsonValue]:
    return {
        "rf_estimand": V1_AC_RF_ESTIMAND,
        "temporal_rf_definition": V1_AC_TEMPORAL_RF_DEFINITION,
        "observed_history_context": V1_AC_HISTORY_CONTEXT,
        "lag_order_semantics": "current_to_past",
        "lag_ms": request.lag_ms.tolist(),
        "dt_ms": request.dt_ms,
        "model_revision": request.model_revision,
        "checkpoint_stage": request.checkpoint_stage,
        "checkpoint_best_step": request.checkpoint_best_step,
        "training_seed": request.training_seed,
        "checkpoint_sha256": request.checkpoint_sha256,
        "cell_order": list(request.cell_ids),
        "cell_types": list(request.cell_types),
        "polarities": list(request.polarities),
        "selected_sequence_indices": list(request.context_indices),
        "selected_source_image_ids": list(request.source_image_ids),
        "selected_trial_indices": list(request.trial_indices),
        "source_sha256": dict(request.source_sha256),
    }


def validate_v1_ac_temporal_rf_artifact(
    artifact: Mapping[str, JsonValue | torch.Tensor],
) -> None:
    identity = artifact.get("identity")
    required_identity = {
        "rf_estimand",
        "temporal_rf_definition",
        "observed_history_context",
        "lag_order_semantics",
        "lag_ms",
        "model_revision",
        "checkpoint_sha256",
        "cell_order",
        "cell_types",
        "polarities",
        "selected_sequence_indices",
        "selected_source_image_ids",
        "selected_trial_indices",
        "source_sha256",
    }
    if (
        artifact.get("schema") != V1_AC_TEMPORAL_RF_SCHEMA
        or artifact.get("schema_revision") != 2
        or not isinstance(identity, Mapping)
        or not required_identity.issubset(identity)
    ):
        raise V1ACTemporalRFArtifactError(
            "V1 AC temporal RF artifact identity is invalid"
        )
    tensors = tuple(
        artifact.get(key)
        for key in (
            "normal",
            "ac_structural_clamp",
            "ac_contribution",
            "clamped_ac_pathway",
        )
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise V1ACTemporalRFArtifactError(
            "V1 AC temporal RF artifact tensors are invalid"
        )
    normal, clamped, contribution, clamped_pathway = tensors
    assert isinstance(normal, torch.Tensor)
    assert isinstance(clamped, torch.Tensor)
    assert isinstance(contribution, torch.Tensor)
    assert isinstance(clamped_pathway, torch.Tensor)
    if not (
        normal.ndim == 2
        and clamped.shape == normal.shape
        and contribution.shape == normal.shape
        and clamped_pathway.shape == normal.shape
        and torch.equal(contribution, normal - clamped)
        and torch.count_nonzero(clamped_pathway).item() == 0
        and len(identity["cell_order"]) == normal.shape[0]
    ):
        raise V1ACTemporalRFArtifactError(
            "V1 AC temporal RF artifact decomposition is invalid"
        )


def save_perturbation_artifacts(
    request: PerturbationArtifactRequest,
) -> None:
    atomic_torch_save(
        {
            "normal": {
                "logits": request.responses.normal.logits,
                "spike_probability": request.responses.normal.probability,
                "ac_local_current": request.responses.normal.ac_local,
                "ac_transient_current": request.responses.normal.ac_transient,
            },
            "ac_structural_clamp": {
                "logits": request.responses.clamped.logits,
                "spike_probability": request.responses.clamped.probability,
                "ac_local_current": request.responses.clamped.ac_local,
                "ac_transient_current": request.responses.clamped.ac_transient,
            },
            "ac_contribution": {
                "logits": request.responses.normal.logits
                - request.responses.clamped.logits,
                "spike_probability": request.responses.normal.probability
                - request.responses.clamped.probability,
            },
        },
        request.output_dir / "responses.pt",
    )
    ac_temporal_rf = request.normal_temporal_rf - request.clamped_temporal_rf
    temporal_rf_artifact = {
        "schema": V1_AC_TEMPORAL_RF_SCHEMA,
        "schema_revision": 2,
        "identity": request.identity,
        "lag_ms": request.lag_ms,
        "lag_order_semantics": "current_to_past",
        "normal": request.normal_temporal_rf,
        "ac_structural_clamp": request.clamped_temporal_rf,
        "ac_contribution": ac_temporal_rf,
        "clamped_ac_pathway": torch.zeros_like(ac_temporal_rf),
    }
    validate_v1_ac_temporal_rf_artifact(temporal_rf_artifact)
    atomic_torch_save(temporal_rf_artifact, request.output_dir / "temporal-rf.pt")


__all__ = [
    "V1_AC_HISTORY_CONTEXT",
    "V1_AC_RF_ESTIMAND",
    "V1_AC_TEMPORAL_RF_DEFINITION",
    "V1_AC_TEMPORAL_RF_SCHEMA",
    "ArtifactSourcePaths",
    "PerturbationArtifactRequest",
    "TemporalRFIdentityRequest",
    "V1ACTemporalRFArtifactError",
    "build_lineage",
    "build_temporal_rf_identity",
    "implementation_source_sha256",
    "save_perturbation_artifacts",
    "validate_v1_ac_temporal_rf_artifact",
]
