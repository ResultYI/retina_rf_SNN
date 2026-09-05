from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

import h5py
import numpy as np

from evaluation.mechanistic_retina.karamanlis_locality_io import (
    ExperimentProjection,
    LocalityGraphWriteRequest,
    sha256_file,
    write_locality_graph_artifact,
)
from evaluation.mechanistic_retina.karamanlis_locality_graph import (
    RFLocalityCell,
    RFLocalityGraph,
    RFMapGrid,
    build_rf_locality_graph,
    extract_rf_spatial_extent,
)


EXPECTED_SOURCE_SCHEMA: Final = "karamanlis_marmoset_frozen_white_noise_rf_centers_v1"
EXPECTED_RELIABLE_CELL_COUNT: Final = 60


@dataclass(frozen=True, slots=True)
class RFLocalityArtifactError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class LocalityGraphRunResult:
    results_path: Path
    graph_path: Path
    graph: RFLocalityGraph


def run_karamanlis_locality_graph(
    rf_source_dir: str | Path,
    output_dir: str | Path,
) -> LocalityGraphRunResult:
    source = Path(rf_source_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise RFLocalityArtifactError("locality graph output directory must be empty")
    payload = _load_json(source / "results.json")
    if payload.get("schema") != EXPECTED_SOURCE_SCHEMA:
        raise RFLocalityArtifactError("RF-center source schema is unsupported")
    session_id = str(payload["session_id"])
    experiment_path = (
        Path("data/real/karamanlis_2024/sessions") / session_id / "expdata.mat"
    )
    projection = _load_experiment_projection(experiment_path)
    expected_experiment_hash = str(payload["source_sha256"]["expdata.mat"])
    if sha256_file(experiment_path) != expected_experiment_hash:
        raise RFLocalityArtifactError("expdata hash differs from the RF-center lineage")
    stored_pixel_size = float(
        payload["coordinates"]["screen_pixel_size_micrometers_on_retina"]
    )
    if not np.isclose(stored_pixel_size, projection.screen_pixel_size_um):
        raise RFLocalityArtifactError("RF source and expdata projector pixel sizes differ")
    with np.load(source / "rf_maps.npz") as arrays:
        x_centers = np.asarray(arrays["x_stimulus_pixels"], dtype=np.float64)
        y_centers = np.asarray(arrays["y_stimulus_pixels"], dtype=np.float64)
        spatial_rfs = np.asarray(arrays["full_spatial_rfs"], dtype=np.float64)
    records = payload["cells"]
    if not isinstance(records, list) or spatial_rfs.shape[0] != len(records):
        raise RFLocalityArtifactError("RF maps and cell metadata do not align")
    reliable_indices = tuple(
        index for index, record in enumerate(records) if record["reliable_for_locality"]
    )
    if len(reliable_indices) != EXPECTED_RELIABLE_CELL_COUNT:
        raise RFLocalityArtifactError("RF source does not contain exactly 60 reliable cells")
    grid = RFMapGrid(
        x_centers_px=x_centers,
        y_centers_px=y_centers,
        stixel_width_px=_uniform_axis_step(x_centers),
        stixel_height_px=_uniform_axis_step(y_centers),
        screen_pixel_size_um=projection.screen_pixel_size_um,
        origin_px=projection.origin_px,
    )
    cells: list[RFLocalityCell] = []
    for index in reliable_indices:
        record = records[index]
        extent = extract_rf_spatial_extent(spatial_rfs[index], grid)
        stored_center = np.asarray(
            record["full_center"]["retinal_micrometers_from_stimulus_center"],
            dtype=np.float64,
        )
        if np.linalg.norm(extent.center_um - stored_center) > projection.screen_pixel_size_um:
            raise RFLocalityArtifactError("recomputed RF center differs from source contour")
        if extent.touches_boundary:
            raise RFLocalityArtifactError("reliable RF contour unexpectedly touches boundary")
        cells.append(
            RFLocalityCell(
                original_index=index,
                cell_id=str(record["cell_id"]),
                cell_type=str(record["cell_type"]),
                polarity=str(record["polarity"]),
                extent=extent,
            )
        )
    graph = build_rf_locality_graph(tuple(cells))
    excluded_ids = tuple(
        str(record["cell_id"])
        for record in records
        if not record["reliable_for_locality"]
    )
    written = write_locality_graph_artifact(
        LocalityGraphWriteRequest(
            graph=graph,
            session_id=session_id,
            projection=projection,
            excluded_ids=excluded_ids,
            source_dir=source,
            experiment_path=experiment_path,
            output_dir=destination,
        )
    )
    return LocalityGraphRunResult(written.results_path, written.graph_path, graph)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RFLocalityArtifactError("RF-center results JSON is unreadable") from error


def _load_experiment_projection(path: Path) -> ExperimentProjection:
    try:
        with h5py.File(path, "r") as experiment:
            screen_y, screen_x = np.asarray(
                experiment["projector/screen"]
            ).reshape(-1)
            pixel_um = float(experiment["projector/pixelsize"][0, 0]) * 1e6
    except (OSError, KeyError, ValueError) as error:
        raise RFLocalityArtifactError("experimental projector geometry is unreadable") from error
    shape = int(screen_y), int(screen_x)
    origin = np.asarray([screen_x / 2.0, screen_y / 2.0], dtype=np.float64)
    if min(shape) <= 0 or not np.isfinite(pixel_um) or pixel_um <= 0:
        raise RFLocalityArtifactError("experimental projector geometry is invalid")
    return ExperimentProjection(origin, pixel_um, shape)


def _uniform_axis_step(axis: np.ndarray) -> int:
    differences = np.diff(axis)
    step = int(round(float(np.median(differences))))
    if step <= 0 or not np.allclose(differences, step):
        raise RFLocalityArtifactError("RF stimulus axis lacks uniform stixel spacing")
    return step


__all__ = [
    "LocalityGraphRunResult",
    "RFLocalityArtifactError",
    "run_karamanlis_locality_graph",
]
