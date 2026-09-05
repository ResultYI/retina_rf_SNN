# /// script
# requires-python = ">=3.12"
# dependencies = ["h5py", "numba>=0.62.1", "numpy", "torch"]
# ///
# Run: python -m scripts.run_karamanlis_rf_centers

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import h5py
import numpy as np
import torch

from evaluation.mechanistic_retina.karamanlis_rf_centers import (
    RFGrid,
    estimate_condition,
)
from evaluation.mechanistic_retina.gollisch_white_noise import (
    initialize_ran1,
    recreate_binary_white_noise_block,
)
from evaluation.mechanistic_retina.karamanlis_running_sta import StreamingSTA


SESSION: Final = Path("data/real/karamanlis_2024/sessions/20220301_252MEA_marmoset_left_n1")
CHECKPOINT: Final = Path("output/real_data/karamanlis_2024_population_v1/model-trained.pt")
OUTPUT: Final = Path("output/real_data/karamanlis_2024_population_rf_centers_v1")
MARMOSET_UM_PER_DEGREE: Final = 100.0
RELIABILITY_LIMIT_UM: Final = 75.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinates(center_px: np.ndarray, pixel_um: float) -> dict[str, list[float]] | None:
    if not np.isfinite(center_px).all():
        return None
    centered_um = (center_px - np.asarray([400.0, 300.0])) * pixel_um
    return {
        "stimulus_screen_pixels": center_px.tolist(),
        "retinal_micrometers_from_stimulus_center": centered_um.tolist(),
        "visual_degrees_from_stimulus_center": (
            centered_um / MARMOSET_UM_PER_DEGREE
        ).tolist(),
    }


def _load_experiment() -> tuple[dict[str, object], np.ndarray, RFGrid, float]:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cell_ids = tuple(str(value) for value in checkpoint["cell_ids"])
    with h5py.File(SESSION / "expdata.mat", "r") as exp:
        units = np.asarray(exp["units"]).T.astype(np.int64)
        pixel_um = float(exp["projector/pixelsize"][0, 0]) * 1e6
        projector_hz = float(exp["projector/refreshrate"][0, 0])
    rows_by_id = {str(unit_id): row for row, unit_id in enumerate(units[:, 0])}
    selected_rows = np.asarray([rows_by_id[cell_id] for cell_id in cell_ids])
    if not np.all(units[selected_rows, 3] == 1):
        raise ValueError("checkpoint contains a non-quality-1 cell")
    with h5py.File(SESSION / "frozencheckerflicker_data.mat", "r") as frozen:
        spike_counts = np.asarray(frozen["runningbin"][:, :, selected_rows])
        width = int(frozen["stimPara/Nx"][0, 0])
        height = int(frozen["stimPara/Ny"][0, 0])
        frame_count = int(frozen["stimPara/RunningFrames"][0, 0])
        seed = int(frozen["stimPara/seed"][0, 0])
        stixel_width = int(frozen["stimPara/stixelwidth"][0, 0])
        stixel_height = int(frozen["stimPara/stixelheight"][0, 0])
        x_centers = np.asarray(frozen["spaceVecX"]).reshape(-1)
        y_centers = np.asarray(frozen["spaceVecY"]).reshape(-1)
    if spike_counts.shape != (40, frame_count, len(cell_ids)):
        raise ValueError("running white-noise tensor does not match checkpoint cells")
    grid = RFGrid(x_centers, y_centers, stixel_width, stixel_height)
    metadata = {
        "checkpoint": checkpoint,
        "cell_ids": cell_ids,
        "selected_rows": selected_rows,
        "stimulus_seed": seed,
        "running_frame_count": frame_count,
        "projector_hz": projector_hz,
    }
    return metadata, spike_counts, grid, pixel_um


def main() -> None:
    metadata, spike_counts, grid, pixel_um = _load_experiment()
    checkpoint = metadata["checkpoint"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    dt_ms = 1000.0 / float(metadata["projector_hz"])
    lag_count = int(np.floor(500.0 / dt_ms + 0.5))
    accumulators = (
        StreamingSTA.empty(lag_count, len(metadata["cell_ids"]), grid.y_centers_px.size, grid.x_centers_px.size),
        StreamingSTA.empty(lag_count, len(metadata["cell_ids"]), grid.y_centers_px.size, grid.x_centers_px.size),
    )
    ran1_state = initialize_ran1(int(metadata["stimulus_seed"]))
    for trial_index, response in enumerate(spike_counts):
        stimulus, ran1_state = recreate_binary_white_noise_block(
            ran1_state,
            grid.x_centers_px.size,
            grid.y_centers_px.size,
            int(metadata["running_frame_count"]),
        )
        accumulators[trial_index % 2].add(stimulus, response)
    full_accumulator = accumulators[0].combined(accumulators[1])
    results = {
        "full": estimate_condition(full_accumulator.finalize(), grid),
        "split_even_trials": estimate_condition(accumulators[0].finalize(), grid),
        "split_odd_trials": estimate_condition(accumulators[1].finalize(), grid),
    }
    first = results["split_even_trials"]
    second = results["split_odd_trials"]
    distances_px = np.linalg.norm(first.centers_px - second.centers_px, axis=1)
    distances_um = distances_px * pixel_um
    full = results["full"]
    reliable = (
        np.isfinite(full.centers_px).all(axis=1)
        & np.isfinite(distances_um)
        & ~full.touches_boundary
        & ~first.touches_boundary
        & ~second.touches_boundary
        & (distances_um <= RELIABILITY_LIMIT_UM)
    )
    cells = []
    for index, cell_id in enumerate(metadata["cell_ids"]):
        reasons = [
            reason
            for reason in (
                full.failure_reasons[index],
                first.failure_reasons[index],
                second.failure_reasons[index],
            )
            if reason is not None
        ]
        if bool(full.touches_boundary[index] or first.touches_boundary[index] or second.touches_boundary[index]):
            reasons.append("25% contour touches stimulus boundary")
        if np.isfinite(distances_um[index]) and distances_um[index] > RELIABILITY_LIMIT_UM:
            reasons.append("split-half center distance exceeds 75 micrometers")
        cells.append(
            {
                "cell_id": cell_id,
                "cell_type": checkpoint["cell_types"][index],
                "polarity": checkpoint["polarities"][index],
                "full_center": _coordinates(full.centers_px[index], pixel_um),
                "split_even_center": _coordinates(first.centers_px[index], pixel_um),
                "split_odd_center": _coordinates(second.centers_px[index], pixel_um),
                "split_half_distance_micrometers": float(distances_um[index]),
                "split_half_distance_visual_degrees": float(
                    distances_um[index] / MARMOSET_UM_PER_DEGREE
                ),
                "reliable_for_locality": bool(reliable[index]),
                "reliability_flags": reasons,
                "spike_count_full": int(spike_counts[:, :, index].sum()),
            }
        )
    valid_distances = distances_um[np.isfinite(distances_um)]
    successful = np.isfinite(full.centers_px).all(axis=1)
    full_degrees = (
        (full.centers_px[successful] - np.asarray([400.0, 300.0]))
        * pixel_um
        / MARMOSET_UM_PER_DEGREE
    )
    payload = {
        "schema": "karamanlis_marmoset_frozen_white_noise_rf_centers_v1",
        "session_id": SESSION.name,
        "source_checkpoint": str(CHECKPOINT),
        "method": {
            "sta_window_requested_ms": 500.0,
            "rf_lag_bin_count": lag_count,
            "rf_lag_window_ms": lag_count * dt_ms,
            "rf_max_lag_ms": (lag_count - 1) * dt_ms,
            "frame_dt_ms": dt_ms,
            "white_noise_segment": "runningbin varying sequence within frozencheckerflicker_data",
            "running_frames_per_trial": int(metadata["running_frame_count"]),
            "split_half_trials": {"even": 20, "odd": 20},
            "stimulus_reconstruction": "Gollisch Lab ran1 with continuous running-sequence state",
            "spatial_projection": "STA projected onto >4.5 robust-SD temporal filter",
            "center": "25%-maximum contour median after 5x upsampling and sigma-4-pixel blur",
            "split_reliability_limit_micrometers": RELIABILITY_LIMIT_UM,
            "split_reliability_limit_basis": (
                "conservative reuse of the paper's 75-micrometer center-distance QC; "
                "the paper applied it to spot versus white-noise centers, not split halves"
            ),
        },
        "coordinates": {
            "axis_order": ["x", "y"],
            "origin": "stimulus center (screen pixel 400,300)",
            "x_positive": "right",
            "y_positive": "up",
            "screen_pixel_size_micrometers_on_retina": pixel_um,
            "marmoset_retinal_micrometers_per_visual_degree": MARMOSET_UM_PER_DEGREE,
        },
        "summary": {
            "cell_count": len(cells),
            "full_center_success_count": int(successful.sum()),
            "reliable_for_locality_count": int(reliable.sum()),
            "visual_degree_range": {
                "x": [float(full_degrees[:, 0].min()), float(full_degrees[:, 0].max())],
                "y": [float(full_degrees[:, 1].min()), float(full_degrees[:, 1].max())],
            },
            "split_half_distance_micrometers": {
                "median": float(np.median(valid_distances)),
                "q75": float(np.quantile(valid_distances, 0.75)),
                "maximum": float(valid_distances.max()),
            },
            "unreliable_cell_ids": [
                cell["cell_id"] for cell in cells if not cell["reliable_for_locality"]
            ],
        },
        "source_sha256": {
            "expdata.mat": _sha256(SESSION / "expdata.mat"),
            "frozencheckerflicker_data.mat": _sha256(
                SESSION / "frozencheckerflicker_data.mat"
            ),
            "model-trained.pt": _sha256(CHECKPOINT),
        },
        "cells": cells,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        OUTPUT / "rf_maps.npz",
        lag_ms=np.arange(lag_count) * dt_ms,
        x_stimulus_pixels=grid.x_centers_px,
        y_stimulus_pixels=grid.y_centers_px,
        **{
            f"{name}_{field}": getattr(result, field)
            for name, result in results.items()
            for field in ("spatial_rfs", "temporal_filters", "centers_px")
        },
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
