from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.karamanlis_rf_population import (
    RFPopulationAdapterConfig,
    load_rf_population_geometry,
    load_rf_population_imagesequence,
)
from evaluation.json_types import JsonValue
from evaluation.mechanistic_retina.karamanlis_locality_graph import RFMapGrid
from evaluation.mechanistic_retina.karamanlis_v1_rf_artifacts import (
    CheckpointValue,
    atomic_json,
    build_results_payload,
    load_sta,
    save_rf_arrays,
    validate_checkpoint_data,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_metrics import (
    compare_population_rfs,
)
from evaluation.mechanistic_retina.karamanlis_v1_rf_validation_math import (
    build_stixel_projection,
    project_cone_rf_to_stixels,
    separable_projection,
)
from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
    PathwayClamp,
)
from models.mechanistic_retina.model import build_mechanistic_retina


class RFValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class V1RFValidationConfig:
    session_dir: Path
    graph_dir: Path
    sta_dir: Path
    checkpoint_path: Path
    output_dir: Path
    rf_batch_size: int = 4


def validate_v1_checkpoint(checkpoint: Mapping[str, CheckpointValue]) -> None:
    model_config = checkpoint.get("model_config")
    if (
        checkpoint.get("schema")
        != "karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1"
        or checkpoint.get("stage") != "best_trained"
        or not isinstance(model_config, Mapping)
    ):
        raise RFValidationError("checkpoint is not the trained V1 canonical candidate")
    if (
        model_config.get("cell_specific_gains") is not True
        or model_config.get("cell_specific_pathway_mixture", False) is not False
    ):
        raise RFValidationError(
            "V1 requires aggregate BC/AC gains and forbids pathway mixture"
        )


def run_v1_rf_validation(config: V1RFValidationConfig) -> dict[str, JsonValue]:
    if config.rf_batch_size < 1:
        raise RFValidationError("RF batch size must be positive")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise FileExistsError("RF validation output directory must be empty")
    checkpoint = torch.load(
        config.checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise RFValidationError("checkpoint payload must be a mapping")
    validate_v1_checkpoint(checkpoint)
    geometry = load_rf_population_geometry(config.graph_dir, grid_size=51)
    data = load_rf_population_imagesequence(
        config.session_dir, geometry, RFPopulationAdapterConfig()
    )
    validate_checkpoint_data(checkpoint, data)
    model_config_values = dict(checkpoint["model_config"])
    model_config_values["architecture_mode"] = ArchitectureMode(
        model_config_values["architecture_mode"]
    )
    model_config = MechanisticRetinaConfig(**model_config_values)
    model = build_mechanistic_retina(
        model_config,
        data.model_cone_positions,
        data.model_cell_positions,
        data.cell_types,
        data.polarities,
        shared_subunit_edge_index=data.edge_index,
        pathway_spatial_geometry=data.pathway_spatial_geometry,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    context_indices = _one_trial_per_source(data.validation.source_image_ids)
    full = _mean_effective_rf(
        model, data.validation, context_indices, frozenset(), config.rf_batch_size
    )
    no_h1 = _mean_effective_rf(
        model,
        data.validation,
        context_indices,
        frozenset({PathwayClamp.H1}),
        config.rf_batch_size,
    )
    bc = _mean_effective_rf(
        model,
        data.validation,
        context_indices,
        frozenset(
            {
                PathwayClamp.H1,
                PathwayClamp.AMACRINE_LOCAL,
                PathwayClamp.AMACRINE_TRANSIENT,
            }
        ),
        config.rf_batch_size,
    )
    pathways = {"H1": full - no_h1, "BC": bc, "AC": no_h1 - bc}
    residual = torch.linalg.vector_norm(
        full - sum(pathways.values(), torch.zeros_like(full))
    )
    if not bool(torch.isfinite(full).all()) or float(residual) > 1e-5:
        raise RFValidationError("global/pathway RF decomposition is invalid")
    sta = load_sta(config.sta_dir, data.cell_ids)
    projection = build_stixel_projection(
        data.cone_blocks_screen_indices.numpy(),
        x_centers_px=sta.x_centers_px,
        y_centers_px=sta.y_centers_px,
        stixel_width_px=5,
        stixel_height_px=5,
    )
    aligned_global = torch.flip(full, dims=(1,))
    stixel_global = project_cone_rf_to_stixels(aligned_global, projection)
    model_projection = separable_projection(stixel_global)
    model_lag_ms = np.arange(model_config.lag_steps) * data.dt_ms
    grid = RFMapGrid(
        sta.x_centers_px,
        sta.y_centers_px,
        5,
        5,
        sta.pixel_um,
        np.asarray((400.0, 300.0)),
    )
    rows = compare_population_rfs(
        cell_ids=data.cell_ids,
        cell_types=data.cell_types,
        polarities=data.polarities,
        model_spatial=model_projection.spatial.numpy(),
        model_temporal=model_projection.temporal.numpy(),
        empirical_spatial=sta.spatial,
        empirical_temporal=sta.temporal,
        empirical_even_temporal=sta.even_temporal,
        empirical_odd_temporal=sta.odd_temporal,
        model_lag_ms=model_lag_ms,
        empirical_lag_ms=sta.lag_ms,
        grid=grid,
    )
    pathway_projections = {
        name: separable_projection(
            project_cone_rf_to_stixels(torch.flip(rf, dims=(1,)), projection)
        )
        for name, rf in pathways.items()
    }
    payload = build_results_payload(
        config=config,
        checkpoint=checkpoint,
        data=data,
        context_indices=context_indices,
        rows=rows,
        residual=residual,
        model_lag_ms=model_lag_ms,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_rf_arrays(
        path=config.output_dir / "rf_validation_arrays.npz",
        data=data,
        projection=projection,
        global_rf=aligned_global,
        model_projection=model_projection,
        pathways=pathways,
        pathway_projections=pathway_projections,
        sta=sta,
        model_lag_ms=model_lag_ms,
    )
    atomic_json(config.output_dir / "results.json", payload)
    return payload


def _mean_effective_rf(model, split, indices, clamps, batch_size) -> torch.Tensor:
    total = None
    for start in range(0, len(indices), batch_size):
        batch = torch.as_tensor(indices[start : start + batch_size])
        rf = effective_rf(
            model,
            split.cone_drive[batch],
            split.spike_events[batch],
            clamps=clamps,
        ).double()
        total = rf.sum(dim=0) if total is None else total + rf.sum(dim=0)
    if total is None:
        raise RFValidationError("RF context selection is empty")
    return (total / len(indices)).float()


def _one_trial_per_source(source_ids: tuple[str, ...]) -> tuple[int, ...]:
    selected: dict[str, int] = {}
    for index, source_id in enumerate(source_ids):
        selected.setdefault(source_id, index)
    return tuple(selected.values())


__all__ = [
    "RFValidationError",
    "V1RFValidationConfig",
    "run_v1_rf_validation",
    "validate_v1_checkpoint",
]
