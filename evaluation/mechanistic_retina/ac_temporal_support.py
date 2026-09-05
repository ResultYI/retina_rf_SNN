from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import torch

from evaluation.mechanistic_retina.ac_circuit_support import JsonValue
from evaluation.mechanistic_retina.ac_temporal_lineage import (
    TEMPORAL_RF_ARTIFACT_SCHEMA,
    TEMPORAL_RF_DEFINITION,
    TEMPORAL_RF_ESTIMAND,
)
from models.mechanistic_retina.contracts import MECHANISTIC_MODEL_REVISION
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina

_IDENTITY_KEYS: Final = {
    "rf_estimand",
    "temporal_rf_definition",
    "lag_order",
    "lag_order_semantics",
    "probe_names",
    "observed_history_context",
    "model_revision",
    "checkpoint_role",
    "dt_ms",
    "cell_order",
    "cell_types",
    "polarities",
    "cell_positions_degs",
    "cone_order",
    "cone_positions_degs",
    "checkpoint_sha256",
    "probe_sha256",
    "source_sha256",
}


class ACTemporalArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PeakSummary:
    peak_absolute_mean: float
    latency_mean_ms: float


def peak_latency_change(
    normal: Mapping[str, torch.Tensor],
    clamped: Mapping[str, torch.Tensor],
    baseline_steps: int,
    dt_ms: float,
) -> Mapping[str, JsonValue]:
    normal_logit = _peak_summary(normal["logits"], baseline_steps, dt_ms)
    clamped_logit = _peak_summary(clamped["logits"], baseline_steps, dt_ms)
    normal_probability = _peak_summary(
        normal["spike_probability"], baseline_steps, dt_ms
    )
    clamped_probability = _peak_summary(
        clamped["spike_probability"], baseline_steps, dt_ms
    )
    return {
        "definition": "mean per-probe/cell absolute deviation peak from pre-probe baseline",
        "normal_logit_peak_absolute": normal_logit.peak_absolute_mean,
        "clamped_logit_peak_absolute": clamped_logit.peak_absolute_mean,
        "logit_peak_absolute_change": clamped_logit.peak_absolute_mean
        - normal_logit.peak_absolute_mean,
        "normal_logit_latency_ms": normal_logit.latency_mean_ms,
        "clamped_logit_latency_ms": clamped_logit.latency_mean_ms,
        "logit_latency_change_ms": clamped_logit.latency_mean_ms
        - normal_logit.latency_mean_ms,
        "normal_probability_peak_absolute": normal_probability.peak_absolute_mean,
        "clamped_probability_peak_absolute": clamped_probability.peak_absolute_mean,
        "probability_peak_absolute_change": clamped_probability.peak_absolute_mean
        - normal_probability.peak_absolute_mean,
        "normal_probability_latency_ms": normal_probability.latency_mean_ms,
        "clamped_probability_latency_ms": clamped_probability.latency_mean_ms,
        "probability_latency_change_ms": clamped_probability.latency_mean_ms
        - normal_probability.latency_mean_ms,
    }


def temporal_parameter_invariance(
    model: MechanisticGraphTemporalRetina,
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    base: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    names = tuple(name for name, _ in model.named_parameters())
    unchanged = {name: torch.equal(before[name], after[name]) for name in names}
    tau_names = tuple(name for name in names if name.endswith("raw_tau"))
    delay_names = tuple(name for name in names if name.endswith("raw_delay"))
    gate_names = tuple(name for name in names if name.startswith("gates."))
    temporal_names = {*tau_names, *delay_names, *gate_names}
    weight_names = tuple(name for name in names if name not in temporal_names)
    return {
        **base,
        "weights": all(unchanged[name] for name in weight_names),
        "gates": all(unchanged[name] for name in gate_names),
        "tau": all(unchanged[name] for name in tau_names),
        "delay": all(unchanged[name] for name in delay_names),
    }


def validate_temporal_rf_artifact(value: Mapping) -> Mapping:
    expected_keys = {
        "schema",
        "schema_revision",
        "identity",
        "normal",
        "ac_structural_clamp",
        "clamp_minus_normal",
        "ac_contribution",
    }
    if set(value) != expected_keys:
        raise ACTemporalArtifactError("temporal RF artifact keys are invalid")
    if value["schema"] != TEMPORAL_RF_ARTIFACT_SCHEMA or value["schema_revision"] != 1:
        raise ACTemporalArtifactError("temporal RF artifact schema is invalid")
    identity = value["identity"]
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_KEYS:
        raise ACTemporalArtifactError("temporal RF artifact identity is invalid")
    normal, normal_pathway = _temporal_tensors(value["normal"], "normal")
    clamped, clamped_pathway = _temporal_tensors(
        value["ac_structural_clamp"], "clamped"
    )
    delta, delta_pathway = _temporal_tensors(value["clamp_minus_normal"], "delta")
    contribution, contribution_pathway = _temporal_tensors(
        value["ac_contribution"], "contribution"
    )
    if not (
        normal.shape == clamped.shape == delta.shape == contribution.shape
        and normal.ndim == 3
        and torch.equal(delta, clamped - normal)
        and torch.equal(contribution, normal - clamped)
        and torch.equal(normal_pathway, contribution)
        and torch.count_nonzero(clamped_pathway).item() == 0
        and torch.equal(delta_pathway, -normal_pathway)
        and torch.equal(contribution_pathway, normal_pathway)
    ):
        raise ACTemporalArtifactError("temporal RF decomposition is invalid")
    _validate_identity(identity, normal.shape)
    return value


def _peak_summary(
    response: torch.Tensor,
    baseline_steps: int,
    dt_ms: float,
) -> PeakSummary:
    baseline = response[:, :, :baseline_steps].mean(dim=2, keepdim=True)
    deviation = (response[:, :, baseline_steps:] - baseline).abs()
    peak, peak_index = deviation.max(dim=2)
    return PeakSummary(float(peak.mean()), float(peak_index.float().mean() * dt_ms))


def _temporal_tensors(value, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != {"temporal", "ac_pathway"}:
        raise ACTemporalArtifactError(f"{label} temporal RF block is invalid")
    temporal = value["temporal"]
    pathway = value["ac_pathway"]
    if not isinstance(temporal, torch.Tensor) or not isinstance(pathway, torch.Tensor):
        raise ACTemporalArtifactError(f"{label} temporal RF values must be tensors")
    if pathway.shape != temporal.shape or not bool(
        torch.isfinite(temporal).all() and torch.isfinite(pathway).all()
    ):
        raise ACTemporalArtifactError(f"{label} temporal RF tensors are invalid")
    return temporal, pathway


def _validate_identity(identity: Mapping, shape: torch.Size) -> None:
    probe_count, cell_count, lag_count = shape
    cone_positions = torch.as_tensor(identity["cone_positions_degs"])
    cell_positions = torch.as_tensor(identity["cell_positions_degs"])
    valid_sequences = all(
        isinstance(identity[key], Sequence)
        and not isinstance(identity[key], str | bytes)
        for key in ("probe_names", "cell_types", "polarities")
    )
    if not (
        identity["rf_estimand"] == TEMPORAL_RF_ESTIMAND
        and identity["temporal_rf_definition"] == TEMPORAL_RF_DEFINITION
        and identity["lag_order_semantics"] == "oldest_to_current"
        and identity["observed_history_context"] == "all-zero"
        and identity["model_revision"] == MECHANISTIC_MODEL_REVISION
        and identity["checkpoint_role"] == "student-trained"
        and identity["lag_order"] == list(range(lag_count))
        and identity["cell_order"] == list(range(cell_count))
        and identity["cone_order"] == list(range(cone_positions.shape[0]))
        and valid_sequences
        and len(identity["probe_names"]) == probe_count
        and len(identity["cell_types"]) == cell_count
        and len(identity["polarities"]) == cell_count
        and cell_positions.shape == (cell_count, 2)
        and cone_positions.ndim == 2
        and cone_positions.shape[1] == 2
        and bool(torch.isfinite(cell_positions).all())
        and bool(torch.isfinite(cone_positions).all())
    ):
        raise ACTemporalArtifactError("temporal RF artifact identity is invalid")
    hashes = identity["source_sha256"]
    if not isinstance(hashes, Mapping) or not hashes:
        raise ACTemporalArtifactError("temporal RF source identity is invalid")
    digest_values = (
        identity["checkpoint_sha256"],
        identity["probe_sha256"],
        *hashes.values(),
    )
    if any(
        not isinstance(digest, str) or len(digest) != 64 for digest in digest_values
    ):
        raise ACTemporalArtifactError("temporal RF hash identity is invalid")


__all__ = [
    "ACTemporalArtifactError",
    "peak_latency_change",
    "temporal_parameter_invariance",
    "validate_temporal_rf_artifact",
]
