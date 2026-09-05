from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import torch

from data.retinal_recording import RealSequenceSplit
from data.karamanlis_cells import (
    CellCatalog,
    CellSelection,
    KaramanlisCellSelectionError,
    electrode_grid,
    select_cells,
)
from data.karamanlis_projection import (
    FlashTiming,
    ProjectionGeometry,
    StimulusImages,
    project_achromatic_cone_drive,
)
from data.karamanlis_spikes import bin_spikes_to_frames


_TRAIN_IMAGE_COUNT: Final = 12
_VALIDATION_IMAGE_COUNT: Final = 4
_RETINAL_UM_PER_DEGREE: Final = 200.0


@dataclass(frozen=True, slots=True)
class KaramanlisMarmosetData:
    session_id: str
    train: RealSequenceSplit
    validation: RealSequenceSplit
    cell_ids: tuple[str, ...]
    cell_types: tuple[str, ...]
    polarities: tuple[str, ...]
    cell_positions_degs: torch.Tensor
    cone_positions_degs: torch.Tensor
    dt_ms: float
    recording_sampling_rate_hz: float
    projector_rate_hz: float
    input_representation: str
    crop_pixels: int
    pooled_grid_size: int


@dataclass(frozen=True, slots=True)
class KaramanlisDataError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class KaramanlisAdapterConfig:
    train_image_count: int = _TRAIN_IMAGE_COUNT
    validation_image_count: int = _VALIDATION_IMAGE_COUNT
    cell_selection: CellSelection = CellSelection.COLOCALIZED_QUARTET
    crop_pixels: int = 51
    pool_factor: int = 3

    def __post_init__(self) -> None:
        if self.train_image_count < 1 or self.validation_image_count < 1:
            raise KaramanlisDataError("image split sizes must be positive")
        if (
            self.crop_pixels < 1
            or self.pool_factor < 1
            or self.crop_pixels % 2 == 0
            or self.crop_pixels % self.pool_factor != 0
        ):
            raise KaramanlisDataError(
                "crop_pixels must be positive, odd, and divisible by pool_factor"
            )


def load_marmoset_imagesequence(
    session_dir: str | Path,
    config: KaramanlisAdapterConfig,
) -> KaramanlisMarmosetData:
    session_path = Path(session_dir)
    exp_path = session_path / "expdata.mat"
    stimulus_path = session_path / "imagesequence_data.mat"
    if not exp_path.is_file() or not stimulus_path.is_file():
        raise KaramanlisDataError("session lacks expdata or imagesequence data")

    with h5py.File(exp_path, "r") as exp:
        animal = _matlab_text(exp["animal"])
        if animal != "marmoset":
            raise KaramanlisDataError(f"expected marmoset session, found {animal}")
        labels = tuple(
            _matlab_text(exp[reference])
            for reference in np.asarray(exp["typelabels"]).reshape(-1, order="F")
        )
        classes = np.asarray(exp["cellclus_id"]).reshape(-1).astype(np.int64)
        units = np.asarray(exp["units"]).T.astype(np.int64)
        sampling_rate_hz = float(exp["fs"][0, 0])
        projector_rate_hz = float(exp["projector/refreshrate"][0, 0])
        pixel_size_um = float(exp["projector/pixelsize"][0, 0]) * 1e6
        screen_y, screen_x = np.asarray(exp["projector/screen"]).reshape(-1)
        electrode_spacing_um = float(exp["array/distelectrodes"][0, 0]) * 1e6

    array_positions = electrode_grid() * electrode_spacing_um
    try:
        selected_rows = select_cells(
            CellCatalog(labels, classes, units, array_positions), config.cell_selection
        )
    except KaramanlisCellSelectionError as error:
        raise KaramanlisDataError(str(error)) from error
    selected_units = units[selected_rows]
    cell_grid_um = array_positions[selected_units[:, 1] - 1]
    cluster_center_um = cell_grid_um.mean(axis=0)
    cell_positions = (cell_grid_um - cluster_center_um) / _RETINAL_UM_PER_DEGREE
    cell_ids = tuple(str(value) for value in selected_units[:, 0])
    selected_labels = tuple(labels[classes[row] - 1] for row in selected_rows)
    cell_types = tuple(label.split()[1] for label in selected_labels)
    polarities = tuple(label.split()[0] for label in selected_labels)

    with h5py.File(stimulus_path, "r") as stimulus:
        present_order = np.asarray(stimulus["presentOrder"]).reshape(-1).astype(np.int64) - 1
        artificial = np.asarray(stimulus["isartificial"]).reshape(-1).astype(bool)
        images = np.asarray(stimulus["imageEnsemble"])
        space_x = np.asarray(stimulus["spaceVecX"]).reshape(-1)
        space_y = np.asarray(stimulus["spaceVecY"]).reshape(-1)
        onsets = np.asarray(stimulus["rawdata/fonsets"]).reshape(-1)
        spike_table = np.asarray(stimulus["rawdata/spiketimes"])
        trial_steps = int(stimulus["rawdata/stimPara/trialduration"][0, 0])
        flash_start = int(stimulus["rawdata/stimPara/flashstart"][0, 0]) - 1
        flash_stop = int(stimulus["rawdata/stimPara/flashstop"][0, 0]) - 1
        background = float(stimulus["rawdata/stimPara/backgroundIntensity"][0, 0])

    natural_ids = np.flatnonzero(~artificial)
    required_images = config.train_image_count + config.validation_image_count
    if natural_ids.size < required_images:
        raise KaramanlisDataError("session has too few natural image identities")
    train_ids = natural_ids[:config.train_image_count]
    validation_ids = natural_ids[config.train_image_count:required_images]
    trial_onsets = _image_trial_onsets(onsets, images.shape[0], present_order.size)
    crop_center = np.array([screen_x, screen_y]) / 2 + np.array(
        [cluster_center_um[0] - array_positions[:, 0].mean(),
         array_positions[:, 1].mean() - cluster_center_um[1]]
    ) / pixel_size_um
    projection = project_achromatic_cone_drive(
        StimulusImages(images, space_x, space_y, background),
        FlashTiming(trial_steps, flash_start, flash_stop),
        ProjectionGeometry(
            np.concatenate((train_ids, validation_ids)),
            crop_center,
            pixel_size_um,
            config.crop_pixels,
            config.pool_factor,
            _RETINAL_UM_PER_DEGREE,
        ),
    )
    template_by_id = {
        int(image_id): projection.templates[index]
        for index, image_id in enumerate(np.concatenate((train_ids, validation_ids)))
    }
    spike_counts = bin_spikes_to_frames(
        spike_table,
        trial_onsets,
        selected_rows + 1,
        sampling_rate_hz,
        projector_rate_hz,
        trial_steps,
    )
    train = _make_split(present_order, train_ids, template_by_id, spike_counts)
    validation = _make_split(
        present_order, validation_ids, template_by_id, spike_counts
    )
    return KaramanlisMarmosetData(
        session_id=session_path.name,
        train=train,
        validation=validation,
        cell_ids=cell_ids,
        cell_types=cell_types,
        polarities=polarities,
        cell_positions_degs=torch.from_numpy(cell_positions.astype(np.float32)),
        cone_positions_degs=torch.from_numpy(
            projection.positions_degs.astype(np.float32)
        ),
        dt_ms=1000.0 / projector_rate_hz,
        recording_sampling_rate_hz=sampling_rate_hz,
        projector_rate_hz=projector_rate_hz,
        input_representation="marmoset_projected_achromatic_cone_drive_v1",
        crop_pixels=config.crop_pixels,
        pooled_grid_size=config.crop_pixels // config.pool_factor,
    )


def load_minimal_marmoset_imagesequence(
    session_dir: str | Path,
) -> KaramanlisMarmosetData:
    return load_marmoset_imagesequence(session_dir, KaramanlisAdapterConfig())


def _matlab_text(dataset: h5py.Dataset) -> str:
    return "".join(
        chr(int(value))
        for value in np.asarray(dataset).reshape(-1, order="F")
    )


def _image_trial_onsets(
    onsets: np.ndarray, image_count: int, trial_count: int
) -> np.ndarray:
    trial_onsets = np.delete(onsets, np.arange(0, onsets.size, image_count + 1))
    if trial_onsets.size != trial_count:
        raise KaramanlisDataError("image trials do not align with frame onsets")
    return trial_onsets


def _make_split(
    present_order: np.ndarray,
    image_ids: np.ndarray,
    templates: dict[int, np.ndarray],
    spike_counts: np.ndarray,
) -> RealSequenceSplit:
    trial_indices = np.flatnonzero(np.isin(present_order, image_ids))
    cones = np.stack(tuple(templates[int(present_order[index])] for index in trial_indices))
    counts = spike_counts[trial_indices]
    events = (counts > 0).astype(np.float32)
    return RealSequenceSplit(
        cone_drive=torch.from_numpy(cones),
        spike_counts=torch.from_numpy(counts),
        spike_events=torch.from_numpy(events),
        valid_mask=torch.ones_like(torch.from_numpy(events), dtype=torch.bool),
        source_image_ids=tuple(f"image-{int(present_order[index]) + 1:03d}" for index in trial_indices),
        trial_indices=tuple(int(index) for index in trial_indices),
    )


__all__ = [
    "CellSelection",
    "KaramanlisAdapterConfig",
    "KaramanlisDataError",
    "KaramanlisMarmosetData",
    "RealSequenceSplit",
    "load_minimal_marmoset_imagesequence",
    "load_marmoset_imagesequence",
]
