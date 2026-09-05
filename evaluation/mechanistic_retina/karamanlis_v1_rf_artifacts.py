from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
import torch

from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.karamanlis_v1_rf_metrics import (
    RFComparisonRow,
    summarize_rf_rows,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_validation_math import (
    SeparableRF,
    StixelProjection,
)

CheckpointValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | torch.Tensor
    | list["CheckpointValue"]
    | tuple["CheckpointValue", ...]
    | Mapping[str, "CheckpointValue"]
)


class ValidationPaths(Protocol):
    session_dir: Path
    checkpoint_path: Path
    sta_dir: Path
    graph_dir: Path


class RFValidationData(Protocol):
    cell_ids: tuple[str, ...]
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    edge_index: torch.Tensor
    cone_blocks_screen_indices: torch.Tensor
    model_cell_positions: torch.Tensor
    model_cone_positions: torch.Tensor
    cell_positions_um: torch.Tensor
    cone_positions_um: torch.Tensor
    validation: RFValidationSplit


class RFValidationSplit(Protocol):
    source_image_ids: tuple[str, ...]
    trial_indices: tuple[int, ...]


class RFValidationArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LoadedSTA:
    lag_ms: np.ndarray
    x_centers_px: np.ndarray
    y_centers_px: np.ndarray
    spatial: np.ndarray
    temporal: np.ndarray
    even_temporal: np.ndarray
    odd_temporal: np.ndarray
    pixel_um: float


def load_sta(path: Path, cell_ids: tuple[str, ...]) -> LoadedSTA:
    metadata = json.loads((path / "results.json").read_text(encoding="utf-8"))
    records = metadata["cells"]
    row_by_id = {str(row["cell_id"]): index for index, row in enumerate(records)}
    indices = np.asarray(tuple(row_by_id[cell_id] for cell_id in cell_ids))
    if not all(records[index]["reliable_for_locality"] for index in indices):
        raise RFValidationArtifactError(
            "V1 cells must all pass frozen-white-noise RF QC"
        )
    with np.load(path / "rf_maps.npz") as arrays:
        return LoadedSTA(
            lag_ms=np.asarray(arrays["lag_ms"], dtype=np.float64),
            x_centers_px=np.asarray(arrays["x_stimulus_pixels"], dtype=np.float64),
            y_centers_px=np.asarray(arrays["y_stimulus_pixels"], dtype=np.float64),
            spatial=np.asarray(arrays["full_spatial_rfs"][indices]),
            temporal=np.asarray(arrays["full_temporal_filters"][indices]),
            even_temporal=np.asarray(
                arrays["split_even_trials_temporal_filters"][indices]
            ),
            odd_temporal=np.asarray(
                arrays["split_odd_trials_temporal_filters"][indices]
            ),
            pixel_um=float(
                metadata["coordinates"]["screen_pixel_size_micrometers_on_retina"]
            ),
        )


def validate_checkpoint_data(
    checkpoint: Mapping[str, CheckpointValue], data: RFValidationData
) -> None:
    checks = (
        tuple(checkpoint["cell_ids"]) == data.cell_ids,
        tuple(checkpoint["cell_types"]) == data.cell_types,
        tuple(checkpoint["polarities"]) == data.polarities,
        torch.equal(checkpoint["edge_index"], data.edge_index),
        torch.equal(
            checkpoint["cone_blocks_screen_indices"],
            data.cone_blocks_screen_indices,
        ),
        torch.equal(checkpoint["model_cell_positions"], data.model_cell_positions),
        torch.equal(checkpoint["model_cone_positions"], data.model_cone_positions),
        torch.equal(checkpoint["cell_positions_um"], data.cell_positions_um),
        torch.equal(checkpoint["cone_positions_um"], data.cone_positions_um),
    )
    if not all(checks):
        raise RFValidationArtifactError(
            "checkpoint and current 60-cell RF data contract differ"
        )


def build_results_payload(
    *,
    config: ValidationPaths,
    checkpoint: Mapping[str, CheckpointValue],
    data: RFValidationData,
    context_indices: Sequence[int],
    rows: Sequence[RFComparisonRow],
    residual: torch.Tensor,
    model_lag_ms: np.ndarray,
) -> dict[str, JsonValue]:
    source_image_ids = tuple(
        data.validation.source_image_ids[index] for index in context_indices
    )
    trial_indices = tuple(
        data.validation.trial_indices[index] for index in context_indices
    )
    return {
        "schema": "karamanlis_marmoset_v1_real_data_rf_validation_v1",
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_sha256": sha256(config.checkpoint_path),
        "checkpoint_stage": checkpoint["stage"],
        "checkpoint_best_step": checkpoint["best_step"],
        "retrained": False,
        "cell_count": len(data.cell_ids),
        "rf_context": {
            "definition": "conditional final-logit Jacobian averaged over one real validation trial per held-out natural image",
            "context_count": len(context_indices),
            "source_image_count": len(context_indices),
            "selected_sequence_indices": list(context_indices),
            "selected_source_image_ids": list(source_image_ids),
            "selected_trial_indices": list(trial_indices),
            "causal_history": "recorded Bernoulli spike events with the model's existing RGC history shift",
        },
        "time_concepts": {
            "tau": "checkpoint bounded-learnable state decay in ms",
            "explicit_pathway_delay": "checkpoint bounded-learnable fractional pathway delay in ms",
            "rf_lag_window": f"{len(model_lag_ms)} bins, 0 to {model_lag_ms[-1]:.6f} ms",
            "rgc_history_shift": "strictly past recorded spike events; no current-target input",
        },
        "independence": {
            "sta_used_in_training_loss": False,
            "sta_used_in_checkpoint_selection": False,
            "same_sta_spatial_geometry_used_by_model": True,
            "static_scope": "weight/checkpoint-independent target, but not architecture-independent because RF center/contour/extent define V1 support and graph",
            "temporal_scope": "empirical temporal STA was not used by V1 geometry, training loss, or checkpoint selection",
        },
        "pathway_decomposition_residual_norm": float(residual),
        "summary": summarize_rf_rows(rows),
        "per_cell": list(rows),
        "artifacts": {"rf_arrays": "rf_validation_arrays.npz"},
        "source_sha256": {
            "sta_results": sha256(config.sta_dir / "results.json"),
            "sta_rf_maps": sha256(config.sta_dir / "rf_maps.npz"),
            "locality_graph": sha256(config.graph_dir / "locality_graph.npz"),
            "expdata.mat": sha256(config.session_dir / "expdata.mat"),
            "imagesequence_data.mat": sha256(
                config.session_dir / "imagesequence_data.mat"
            ),
        },
    }


def save_rf_arrays(
    *,
    path: Path,
    data: RFValidationData,
    projection: StixelProjection,
    global_rf: torch.Tensor,
    model_projection: SeparableRF,
    pathways: Mapping[str, torch.Tensor],
    pathway_projections: Mapping[str, SeparableRF],
    sta: LoadedSTA,
    model_lag_ms: np.ndarray,
) -> None:
    values = {
        "cell_ids": np.asarray(data.cell_ids),
        "model_lag_ms": model_lag_ms,
        "empirical_lag_ms": sta.lag_ms,
        "cone_blocks_screen_indices": data.cone_blocks_screen_indices.numpy(),
        "stixel_projection_indices": projection.indices,
        "stixel_projection_weights": projection.weights,
        "global_rf_cone": global_rf.numpy(),
        "global_spatial_rf": model_projection.spatial.numpy(),
        "global_temporal_rf": model_projection.temporal.numpy(),
        "empirical_spatial_sta": sta.spatial,
        "empirical_temporal_sta": sta.temporal,
    }
    for name, rf in pathways.items():
        values[f"{name}_pathway_rf_cone"] = torch.flip(rf, dims=(1,)).numpy()
        values[f"{name}_pathway_spatial_rf"] = pathway_projections[name].spatial.numpy()
        values[f"{name}_pathway_temporal_rf"] = pathway_projections[
            name
        ].temporal.numpy()
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


__all__ = [
    "LoadedSTA",
    "RFValidationArtifactError",
    "atomic_json",
    "build_results_payload",
    "load_sta",
    "save_rf_arrays",
    "validate_checkpoint_data",
]
