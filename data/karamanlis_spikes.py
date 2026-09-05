from __future__ import annotations

import numpy as np


def bin_spikes_to_frames(
    spike_table: np.ndarray,
    onsets: np.ndarray,
    unit_rows: np.ndarray,
    sampling_rate_hz: float,
    projector_rate_hz: float,
    trial_steps: int,
) -> np.ndarray:
    timestamps = spike_table[0]
    assignments = spike_table[1].astype(np.int64)
    counts = np.zeros((onsets.size, trial_steps, unit_rows.size), dtype=np.float32)
    offsets = np.arange(trial_steps + 1) * sampling_rate_hz / projector_rate_hz
    for cell, unit_row in enumerate(unit_rows):
        unit_spikes = timestamps[assignments == unit_row]
        for trial, onset in enumerate(onsets):
            indices = np.searchsorted(unit_spikes, onset + offsets)
            counts[trial, :, cell] = np.diff(indices)
    return counts


__all__ = ["bin_spikes_to_frames"]
