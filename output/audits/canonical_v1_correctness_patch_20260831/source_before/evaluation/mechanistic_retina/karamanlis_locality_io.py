from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np

from evaluation.mechanistic_retina.karamanlis_locality_graph import (
    NEIGHBOR_CONNECTION_INITIAL,
    RFLocalityGraph,
    SELF_CONNECTION_INITIAL,
)


OUTPUT_SCHEMA: Final = "karamanlis_marmoset_rf_locality_graph_v1"
CELL_CLASSES: Final = (
    ("ON", "midget"),
    ("OFF", "midget"),
    ("ON", "parasol"),
    ("OFF", "parasol"),
)


@dataclass(frozen=True, slots=True)
class ExperimentProjection:
    origin_px: np.ndarray
    screen_pixel_size_um: float
    screen_shape_px: tuple[int, int]


@dataclass(frozen=True, slots=True)
class LocalityGraphWriteRequest:
    graph: RFLocalityGraph
    session_id: str
    projection: ExperimentProjection
    excluded_ids: tuple[str, ...]
    source_dir: Path
    experiment_path: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class LocalityGraphWriteResult:
    results_path: Path
    graph_path: Path


def write_locality_graph_artifact(
    request: LocalityGraphWriteRequest,
) -> LocalityGraphWriteResult:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = request.output_dir / "locality_graph.npz"
    _save_graph_arrays(graph_path, request.graph)
    results_path = request.output_dir / "results.json"
    results_path.write_text(
        json.dumps(_results_payload(request), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return LocalityGraphWriteResult(results_path, graph_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_graph_arrays(path: Path, graph: RFLocalityGraph) -> None:
    contour_lengths = np.asarray(
        tuple(cell.extent.contour_um.shape[0] for cell in graph.cells), dtype=np.int64
    )
    contour_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(contour_lengths))
    )
    positive_connections = np.where(
        graph.edge_index[0] == graph.edge_index[1],
        SELF_CONNECTION_INITIAL,
        NEIGHBOR_CONNECTION_INITIAL,
    ).astype(np.float32)
    np.savez_compressed(
        path,
        adjacency=graph.adjacency,
        overlap_adjacency=graph.overlap_adjacency,
        proximity_adjacency=graph.proximity_adjacency,
        center_distance_um=graph.center_distance_um,
        edge_index=graph.edge_index,
        raw_connections=graph.raw_connections,
        positive_connections=positive_connections,
        original_cell_indices=np.asarray(
            tuple(cell.original_index for cell in graph.cells), dtype=np.int64
        ),
        cell_ids=np.asarray(tuple(cell.cell_id for cell in graph.cells)),
        cell_types=np.asarray(tuple(cell.cell_type for cell in graph.cells)),
        polarities=np.asarray(tuple(cell.polarity for cell in graph.cells)),
        centers_um=np.stack(tuple(cell.extent.center_um for cell in graph.cells)),
        areas_um2=np.asarray(tuple(cell.extent.area_um2 for cell in graph.cells)),
        equivalent_radii_um=np.asarray(
            tuple(cell.extent.equivalent_radius_um for cell in graph.cells)
        ),
        widths_um=np.asarray(tuple(cell.extent.width_um for cell in graph.cells)),
        heights_um=np.asarray(tuple(cell.extent.height_um for cell in graph.cells)),
        support_masks=np.stack(tuple(cell.extent.support_mask for cell in graph.cells)),
        contour_offsets=contour_offsets,
        contour_points_um=np.concatenate(
            tuple(cell.extent.contour_um for cell in graph.cells), axis=0
        ),
    )


def _results_payload(request: LocalityGraphWriteRequest):
    graph = request.graph
    neighbors = graph.adjacency.sum(axis=1).astype(np.int64) - 1
    class_summary = {}
    for polarity, cell_type in CELL_CLASSES:
        indices = np.asarray(
            [
                cell.polarity == polarity and cell.cell_type == cell_type
                for cell in graph.cells
            ],
            dtype=bool,
        )
        class_summary[f"{polarity} {cell_type}"] = {
            "cell_count": int(indices.sum()),
            "mean_nonself_neighbor_count": float(neighbors[indices].mean()),
            "self_only_cell_count": int(np.sum(neighbors[indices] == 0)),
        }
    edge_count = int(graph.adjacency.sum())
    cells = [
        {
            "cell_id": cell.cell_id,
            "cell_type": cell.cell_type,
            "polarity": cell.polarity,
            "center_um": cell.extent.center_um.tolist(),
            "area_um2": cell.extent.area_um2,
            "equivalent_radius_um": cell.extent.equivalent_radius_um,
            "width_um": cell.extent.width_um,
            "height_um": cell.extent.height_um,
            "contour_point_count": int(cell.extent.contour_um.shape[0]),
            "nonself_neighbor_count": int(neighbors[index]),
        }
        for index, cell in enumerate(graph.cells)
    ]
    return {
        "schema": OUTPUT_SCHEMA,
        "session_id": request.session_id,
        "coordinates": {
            "unit": "retinal_micrometers",
            "projection": "screen pixels multiplied by projector pixel size from expdata.mat",
            "screen_pixel_size_um": request.projection.screen_pixel_size_um,
            "origin_screen_pixels": request.projection.origin_px.tolist(),
            "screen_shape_pixels": list(request.projection.screen_shape_px),
            "axis_order": ["x", "y"],
            "x_positive": "right",
            "y_positive": "down",
        },
        "criterion": {
            "rf_support": "S_i is the connected central component at spatial RF >= 0.25 peak after sigma-4-screen-pixel smoothing",
            "rf_extent": "A_i = |S_i| times pixel_area; r_i = sqrt(A_i/pi)",
            "nonself_edge": "same cell_type and polarity, and either |S_i intersection S_j| > 0 or Euclidean(c_i,c_j) <= r_i + r_j",
            "self_edge": "A_ii = 1 for every retained cell",
            "selection": "split-half QC only; no prediction metric or tuned radius",
        },
        "raw_connections_contract": {
            "edge_index_axis_0": "target",
            "edge_index_axis_1": "source",
            "parameterization": "positive connection = softplus(raw_connections), then target-row normalization in canonical forward",
            "self_positive_initial": SELF_CONNECTION_INITIAL,
            "nonself_positive_initial": NEIGHBOR_CONNECTION_INITIAL,
        },
        "summary": {
            "cell_count": len(graph.cells),
            "all_centers_and_extents_usable": True,
            "edge_count": edge_count,
            "self_edge_count": len(graph.cells),
            "nonself_edge_count": edge_count - len(graph.cells),
            "class_neighbor_summary": class_summary,
        },
        "excluded_cell_ids": list(request.excluded_ids),
        "cells": cells,
        "source_sha256": {
            "rf_results_json": sha256_file(request.source_dir / "results.json"),
            "rf_maps_npz": sha256_file(request.source_dir / "rf_maps.npz"),
            "expdata_mat": sha256_file(request.experiment_path),
        },
    }


__all__ = [
    "ExperimentProjection",
    "LocalityGraphWriteRequest",
    "LocalityGraphWriteResult",
    "sha256_file",
    "write_locality_graph_artifact",
]
