from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import torch

from data.karamanlis_2024 import _image_trial_onsets, _make_split, _matlab_text
from data.karamanlis_cells import CellSelection, TARGET_LABELS
from data.karamanlis_rf_artifact import (
    RFPopulationDataError,
    RFPopulationGeometry,
    build_rf_pathway_geometry,
    load_rf_population_geometry,
)
from data.karamanlis_projection import (
    FlashTiming,
    ProjectionGeometry,
    StimulusImages,
    project_achromatic_cone_drive,
)
from data.karamanlis_spikes import bin_spikes_to_frames
from data.retinal_recording import RealSequenceSplit
from models.mechanistic_retina.pathway_spatial_geometry import PathwaySpatialGeometry


_CANONICAL_GRID_STEP: Final = 0.1125


@dataclass(frozen=True, slots=True)
class RFPopulationAdapterConfig:
    train_image_count: int = 176
    validation_image_count: int = 44
    cell_selection: CellSelection = CellSelection.ALL_QUALITY_1_TARGETS

    def __post_init__(self) -> None:
        if self.train_image_count < 1 or self.validation_image_count < 1:
            raise RFPopulationDataError("image split sizes must be positive")


@dataclass(frozen=True, slots=True)
class RFPopulationMarmosetData:
    session_id: str
    train: RealSequenceSplit
    validation: RealSequenceSplit
    cell_ids: tuple[str, ...]
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    model_cell_positions: torch.Tensor
    model_cone_positions: torch.Tensor
    cell_positions_um: torch.Tensor
    cone_positions_um: torch.Tensor
    cone_blocks_screen_indices: torch.Tensor
    pathway_spatial_geometry: PathwaySpatialGeometry
    edge_index: torch.Tensor
    dt_ms: float
    recording_sampling_rate_hz: float
    projector_rate_hz: float
    cone_drive_coordinate_unit: str
    geometry_lineage: str
    crop_pixels: int
    pool_factor: int
    pooled_grid_size: int


def load_rf_population_imagesequence(
    session_dir: str | Path,
    geometry: RFPopulationGeometry,
    config: RFPopulationAdapterConfig,
) -> RFPopulationMarmosetData:
    if config.cell_selection is not CellSelection.ALL_QUALITY_1_TARGETS:
        raise RFPopulationDataError("RF population requires quality-1 target selection")
    session = Path(session_dir)
    if session.name != geometry.session_id:
        raise RFPopulationDataError("RF geometry and recording session differ")
    metadata = _read_experiment(session / "expdata.mat", geometry)
    stimulus = _read_stimulus(session / "imagesequence_data.mat")
    required = config.train_image_count + config.validation_image_count
    natural_ids = np.flatnonzero(~stimulus["artificial"])
    if natural_ids.size < required:
        raise RFPopulationDataError("recording has too few natural image identities")
    selected_ids = natural_ids[:required]
    timing = FlashTiming(stimulus["trial_steps"], stimulus["flash_start"], stimulus["flash_stop"])
    projected = project_achromatic_cone_drive(
        StimulusImages(stimulus["images"], stimulus["space_x"], stimulus["space_y"], stimulus["background"]),
        timing,
        ProjectionGeometry(
            selected_ids,
            geometry.crop_center_px,
            geometry.screen_pixel_size_um,
            geometry.crop_pixels,
            geometry.pool_factor,
            1.0,
        ),
    )
    cone_blocks, cone_um = _cone_geometry(stimulus, geometry)
    pathway = build_rf_pathway_geometry(
        geometry.support_masks,
        geometry.centers_um,
        geometry.equivalent_radii_um,
        geometry.cell_types,
        cone_blocks,
        cone_um,
    )
    template_by_id = {int(image_id): projected.templates[index] for index, image_id in enumerate(selected_ids)}
    trial_onsets = _image_trial_onsets(
        stimulus["onsets"], stimulus["images"].shape[0], stimulus["present_order"].size
    )
    spike_counts = bin_spikes_to_frames(
        stimulus["spike_table"],
        trial_onsets,
        metadata["selected_rows"] + 1,
        metadata["sampling_rate_hz"],
        metadata["projector_rate_hz"],
        stimulus["trial_steps"],
    )
    train_ids = selected_ids[: config.train_image_count]
    validation_ids = selected_ids[config.train_image_count :]
    train = _make_split(stimulus["present_order"], train_ids, template_by_id, spike_counts)
    validation = _make_split(stimulus["present_order"], validation_ids, template_by_id, spike_counts)
    scale = _CANONICAL_GRID_STEP / (geometry.pool_factor * geometry.screen_pixel_size_um)
    return RFPopulationMarmosetData(
        geometry.session_id,
        train,
        validation,
        geometry.cell_ids,
        geometry.cell_types,
        geometry.polarities,
        torch.from_numpy((geometry.centers_um * scale).astype(np.float32)),
        torch.from_numpy((cone_um * scale).astype(np.float32)),
        torch.from_numpy(geometry.centers_um.astype(np.float32)),
        torch.from_numpy(cone_um.astype(np.float32)),
        torch.from_numpy(cone_blocks.copy()),
        pathway,
        geometry.edge_index,
        1000.0 / metadata["projector_rate_hz"],
        metadata["sampling_rate_hz"],
        metadata["projector_rate_hz"],
        "retinal_micrometers",
        "white-noise RF center/contour/extent in retinal micrometers with x-right/y-up, and RF-derived explicit locality graph",
        geometry.crop_pixels,
        geometry.pool_factor,
        geometry.grid_size,
    )


def _read_experiment(path: Path, geometry: RFPopulationGeometry):
    with h5py.File(path, "r") as exp:
        if _matlab_text(exp["animal"]) != "marmoset":
            raise RFPopulationDataError("recording is not marmoset")
        labels = tuple(_matlab_text(exp[ref]) for ref in np.asarray(exp["typelabels"]).reshape(-1, order="F"))
        classes = np.asarray(exp["cellclus_id"]).reshape(-1).astype(np.int64)
        units = np.asarray(exp["units"]).T.astype(np.int64)
        sampling = float(exp["fs"][0, 0])
        projector = float(exp["projector/refreshrate"][0, 0])
    candidates = np.flatnonzero(
        np.isin(classes, tuple(labels.index(label) + 1 for label in TARGET_LABELS))
        & (units[:, 3] == 1)
    )
    row_by_id = {str(units[row, 0]): int(row) for row in candidates}
    try:
        rows = np.asarray(tuple(row_by_id[cell_id] for cell_id in geometry.cell_ids), dtype=np.int64)
    except KeyError as error:
        raise RFPopulationDataError("RF cell is absent from quality-1 recording units") from error
    selected_labels = tuple(labels[classes[row] - 1] for row in rows)
    if tuple(label.split()[1] for label in selected_labels) != geometry.cell_types or tuple(label.split()[0] for label in selected_labels) != geometry.polarities:
        raise RFPopulationDataError("RF cell metadata differs from recording metadata")
    return {"selected_rows": rows, "sampling_rate_hz": sampling, "projector_rate_hz": projector}


def _read_stimulus(path: Path):
    with h5py.File(path, "r") as stimulus:
        return {
            "present_order": np.asarray(stimulus["presentOrder"]).reshape(-1).astype(np.int64) - 1,
            "artificial": np.asarray(stimulus["isartificial"]).reshape(-1).astype(bool),
            "images": np.asarray(stimulus["imageEnsemble"]),
            "space_x": np.asarray(stimulus["spaceVecX"]).reshape(-1),
            "space_y": np.asarray(stimulus["spaceVecY"]).reshape(-1),
            "onsets": np.asarray(stimulus["rawdata/fonsets"]).reshape(-1),
            "spike_table": np.asarray(stimulus["rawdata/spiketimes"]),
            "trial_steps": int(stimulus["rawdata/stimPara/trialduration"][0, 0]),
            "flash_start": int(stimulus["rawdata/stimPara/flashstart"][0, 0]) - 1,
            "flash_stop": int(stimulus["rawdata/stimPara/flashstop"][0, 0]) - 1,
            "background": float(stimulus["rawdata/stimPara/backgroundIntensity"][0, 0]),
        }


def _cone_geometry(stimulus, geometry: RFPopulationGeometry) -> tuple[np.ndarray, np.ndarray]:
    half = geometry.crop_pixels // 2
    center_x = int(np.argmin(np.abs(stimulus["space_x"] - geometry.crop_center_px[0])))
    center_y = int(np.argmin(np.abs(stimulus["space_y"] - geometry.crop_center_px[1])))
    x_start, y_start = center_x - half, center_y - half
    blocks = []
    positions = []
    for row in range(geometry.grid_size):
        for column in range(geometry.grid_size):
            xs = stimulus["space_x"][x_start + column * geometry.pool_factor : x_start + (column + 1) * geometry.pool_factor]
            ys = stimulus["space_y"][y_start + row * geometry.pool_factor : y_start + (row + 1) * geometry.pool_factor]
            blocks.append((int(ys[0] - 1), int(ys[-1]), int(xs[0] - 1), int(xs[-1])))
            positions.append(((xs.mean() - geometry.screen_origin_px[0]) * geometry.screen_pixel_size_um, (geometry.screen_origin_px[1] - ys.mean()) * geometry.screen_pixel_size_um))
    return np.asarray(blocks, dtype=np.int64), np.asarray(positions, dtype=np.float64)


__all__ = [
    "RFPopulationAdapterConfig",
    "RFPopulationDataError",
    "RFPopulationGeometry",
    "RFPopulationMarmosetData",
    "build_rf_pathway_geometry",
    "load_rf_population_geometry",
    "load_rf_population_imagesequence",
]
