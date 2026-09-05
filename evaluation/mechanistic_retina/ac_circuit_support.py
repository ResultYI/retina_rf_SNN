from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Final

import torch

from models.mechanistic_retina.model import MechanisticGraphTemporalRetina

type TensorBlock = dict[str, torch.Tensor]
type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

AC_RF_ARTIFACT_SCHEMA: Final = "ac-circuit-perturbation-rf-v1"
AC_RF_ESTIMAND: Final = (
    "final-time conditional RGC-logit Jacobian with sampled spike history held fixed"
)
_TOP_LEVEL_KEYS: Final = {
    "schema",
    "schema_revision",
    "identity",
    "normal",
    "ac_structural_clamp",
    "clamp_minus_normal",
    "ac_contribution",
}
_IDENTITY_KEYS: Final = {
    "rf_estimand",
    "lag_order",
    "lag_order_semantics",
    "cell_order",
    "cell_types",
    "polarities",
    "cell_positions_degs",
    "cone_order",
    "cone_positions_degs",
    "validation_split",
    "validation_context",
    "validation_cones_sha256",
    "validation_spikes_sha256",
    "checkpoint_sha256",
    "sampled_data_sha256",
    "source_sha256",
}


class ACRFArtifactError(ValueError):
    pass


def load_ac_rf_artifact(path: Path) -> Mapping:
    return validate_ac_rf_artifact(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def validate_ac_rf_artifact(value) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_KEYS:
        raise ACRFArtifactError("AC RF artifact top-level contract is invalid")
    if value["schema"] != AC_RF_ARTIFACT_SCHEMA or value["schema_revision"] != 1:
        raise ACRFArtifactError("AC RF artifact schema is invalid")
    identity = value["identity"]
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_KEYS:
        raise ACRFArtifactError("AC RF artifact identity contract is invalid")
    if (
        identity["rf_estimand"] != AC_RF_ESTIMAND
        or identity["lag_order_semantics"] != "oldest_to_current"
        or identity["validation_split"] != "validation"
    ):
        raise ACRFArtifactError("AC RF artifact estimand, lag order, or split is invalid")
    normal = _tensor_block(value["normal"], "normal", {"global", "temporal", "ac_pathway"})
    clamped = _tensor_block(
        value["ac_structural_clamp"],
        "AC structural clamp",
        {"global", "temporal", "ac_pathway"},
    )
    clamp_delta = _tensor_block(
        value["clamp_minus_normal"], "clamp-minus-normal", {"global", "temporal"}
    )
    contribution = _tensor_block(
        value["ac_contribution"], "AC contribution", {"global", "temporal"}
    )
    global_rf = normal["global"]
    if global_rf.ndim != 5 or clamped["global"].shape != global_rf.shape:
        raise ACRFArtifactError("AC RF global tensors have invalid shapes")
    expected_temporal_shape = global_rf.shape[:-1]
    if (
        normal["ac_pathway"].shape != global_rf.shape
        or clamped["ac_pathway"].shape != global_rf.shape
        or clamp_delta["global"].shape != global_rf.shape
        or contribution["global"].shape != global_rf.shape
        or normal["temporal"].shape != expected_temporal_shape
        or clamped["temporal"].shape != expected_temporal_shape
        or clamp_delta["temporal"].shape != expected_temporal_shape
        or contribution["temporal"].shape != expected_temporal_shape
    ):
        raise ACRFArtifactError("AC RF pathway or temporal tensors have invalid shapes")
    cell_count, lag_count, cone_count = global_rf.shape[-3:]
    _validate_identity(identity, cell_count, lag_count, cone_count)
    if not (
        torch.equal(normal["temporal"], global_rf.sum(dim=-1))
        and torch.equal(clamped["temporal"], clamped["global"].sum(dim=-1))
        and torch.equal(normal["ac_pathway"], global_rf - clamped["global"])
        and torch.count_nonzero(clamped["ac_pathway"]).item() == 0
        and torch.equal(clamp_delta["global"], clamped["global"] - global_rf)
        and torch.equal(
            clamp_delta["temporal"], clamped["temporal"] - normal["temporal"]
        )
        and torch.equal(contribution["global"], global_rf - clamped["global"])
        and torch.equal(
            contribution["temporal"], normal["temporal"] - clamped["temporal"]
        )
    ):
        raise ACRFArtifactError("AC RF tensor decomposition contract is invalid")
    return value


def _tensor_block(value, label: str, keys: set[str]) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ACRFArtifactError(f"{label} RF tensor block is invalid")
    if any(not isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise ACRFArtifactError(f"{label} RF values must be tensors")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in value.values()):
        raise ACRFArtifactError(f"{label} RF tensors must be finite")
    return value


def _validate_identity(
    identity: Mapping,
    cell_count: int,
    lag_count: int,
    cone_count: int,
) -> None:
    if identity["cell_order"] != list(range(cell_count)):
        raise ACRFArtifactError("AC RF cell order is invalid")
    if identity["cone_order"] != list(range(cone_count)):
        raise ACRFArtifactError("AC RF cone order is invalid")
    if identity["lag_order"] != list(range(lag_count)):
        raise ACRFArtifactError("AC RF lag order is invalid")
    cell_positions = torch.as_tensor(identity["cell_positions_degs"])
    cone_positions = torch.as_tensor(identity["cone_positions_degs"])
    if cell_positions.shape != (cell_count, 2) or cone_positions.shape != (cone_count, 2):
        raise ACRFArtifactError("AC RF geometry identity is invalid")
    if not bool(torch.isfinite(cell_positions).all() and torch.isfinite(cone_positions).all()):
        raise ACRFArtifactError("AC RF geometry must be finite")
    for key in ("cell_types", "polarities"):
        values = identity[key]
        if not isinstance(values, Sequence) or isinstance(values, str | bytes):
            raise ACRFArtifactError(f"AC RF {key} identity is invalid")
        if len(values) != cell_count or any(not isinstance(item, str) for item in values):
            raise ACRFArtifactError(f"AC RF {key} identity is invalid")
    hashes = identity["source_sha256"]
    if not isinstance(hashes, Mapping) or not hashes:
        raise ACRFArtifactError("AC RF source hashes are invalid")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for name, digest in hashes.items()
    ):
        raise ACRFArtifactError("AC RF source hashes are invalid")


def response_block(
    logits: torch.Tensor,
    probability: torch.Tensor,
    local_current: torch.Tensor,
    transient_current: torch.Tensor,
    stimulus_count: int,
    trial_count: int,
) -> TensorBlock:
    shape = (stimulus_count, trial_count, *logits.shape[1:])
    local = local_current.reshape(shape).detach()
    transient = transient_current.reshape(shape).detach()
    return {
        "logits": logits.reshape(shape).detach(),
        "spike_probability": probability.reshape(shape).detach(),
        "ac_local_current": local,
        "ac_transient_current": transient,
        "ac_total_current": local + transient,
    }


def response_metrics(
    delta: TensorBlock,
    normal: TensorBlock,
    clamped: TensorBlock,
) -> dict[str, JsonValue]:
    return {
        "mean_absolute_logit_change": float(delta["logits"].abs().mean()),
        "mean_logit_change": float(delta["logits"].mean()),
        "mean_absolute_probability_change": float(
            delta["spike_probability"].abs().mean()
        ),
        "mean_probability_change": float(delta["spike_probability"].mean()),
        "normal_mean_probability": float(normal["spike_probability"].mean()),
        "clamped_mean_probability": float(clamped["spike_probability"].mean()),
    }


def tensor_change_metrics(
    normal: torch.Tensor, clamped: torch.Tensor
) -> dict[str, JsonValue]:
    normal_norm = torch.linalg.vector_norm(normal)
    clamped_norm = torch.linalg.vector_norm(clamped)
    difference = clamped - normal
    denominator = normal_norm * clamped_norm
    cosine = torch.where(
        denominator > 0,
        (normal.flatten() @ clamped.flatten()) / denominator,
        torch.tensor(1.0 if normal_norm == clamped_norm else 0.0),
    )
    norm_change = clamped_norm - normal_norm
    return {
        "cosine": float(cosine),
        "normal_norm": float(normal_norm),
        "clamped_norm": float(clamped_norm),
        "norm_change": float(norm_change),
        "relative_norm_change": float(norm_change / normal_norm.clamp_min(1e-12)),
        "difference_norm": float(torch.linalg.vector_norm(difference)),
        "mean_absolute_change": float(difference.abs().mean()),
    }


def state_snapshot(
    model: MechanisticGraphTemporalRetina,
) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def parameter_invariance(
    model: MechanisticGraphTemporalRetina,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, JsonValue]:
    unchanged = {name: torch.equal(before[name], after[name]) for name in before}
    parameter_names = tuple(name for name, _ in model.named_parameters())
    return {
        "all_parameters_unchanged": all(unchanged[name] for name in parameter_names),
        "all_state_tensors_unchanged": all(unchanged.values()),
        "state_sha256_before": _state_sha256(before),
        "state_sha256_after": _state_sha256(after),
        "H1": all(
            unchanged[name]
            for name in parameter_names
            if name.startswith("h1.") or name == "gates.raw_h1_amplitude"
        ),
        "BC": all(unchanged[name] for name in parameter_names if name.startswith("bipolar.")),
        "RGC": all(
            unchanged[name]
            for name in parameter_names
            if name.startswith("rgc.") or name == "gates.history"
        ),
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
