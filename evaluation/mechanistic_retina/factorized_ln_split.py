from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from baselines.center_surround_ln import CONTEXT_BINS, LNError
from data.retinal_recording import RealSequenceSplit


@dataclass(frozen=True, slots=True)
class TrialBoundary:
    trial_index: int
    trial_start: int
    fit_stop: int
    dev_start: int
    trial_stop: int


@dataclass(frozen=True, slots=True)
class InnerDevSplit:
    train: RealSequenceSplit
    development: RealSequenceSplit
    boundaries: tuple[TrialBoundary, ...]


def make_inner_dev(source: RealSequenceSplit) -> InnerDevSplit:
    batch, time, _ = source.cone_drive.shape
    if len(source.source_image_ids) != batch or len(source.trial_indices) != batch:
        raise LNError("trial metadata differs from the sequence count")
    starts = []
    for name in source.source_image_ids:
        match = re.search(r"-live-frames-(\d+)-(\d+)-trial-\d+$", name)
        if match is None or int(match[2]) - int(match[1]) + 1 != time:
            raise LNError("expected native Schottdorf frame-segment identity")
        starts.append(int(match[1]))
    fit_support = torch.zeros_like(source.valid_mask)
    dev_support = torch.zeros_like(source.valid_mask)
    dev_context = torch.zeros_like(source.valid_mask)
    boundaries = []
    for trial in sorted(set(source.trial_indices)):
        rows = [i for i, value in enumerate(source.trial_indices) if value == trial]
        frame_starts = [starts[i] for i in rows]
        if frame_starts != list(range(frame_starts[0], frame_starts[-1] + time, time)):
            raise LNError("each training trial must have contiguous ordered segments")
        first, stop = frame_starts[0], frame_starts[-1] + time
        dev_start = first + (stop - first) * 4 // 5
        fit_stop = dev_start - CONTEXT_BINS
        if fit_stop <= first:
            raise LNError("trial is too short for the inner-dev guard")
        boundaries.append(TrialBoundary(trial, first, fit_stop, dev_start, stop))
        for row in rows:
            positions = starts[row] + torch.arange(time, device=source.valid_mask.device)
            fit_support[row, :, 0] = positions < fit_stop
            dev_support[row, :, 0] = positions >= dev_start
            dev_context[row, :, 0] = positions >= fit_stop
    train = _masked_split(source, fit_support, fit_support)
    development = _masked_split(source, dev_support, dev_context)
    return InnerDevSplit(train, development, tuple(boundaries))


def _masked_split(
    source: RealSequenceSplit, score_support: torch.Tensor, input_support: torch.Tensor,
) -> RealSequenceSplit:
    mask = source.valid_mask & score_support
    rows = torch.nonzero(mask.any(dim=(1, 2)), as_tuple=False).flatten()
    if rows.numel() == 0:
        raise LNError("inner split has no valid scoring bins")
    return RealSequenceSplit(
        cone_drive=(source.cone_drive * input_support)[rows],
        spike_counts=(source.spike_counts * input_support)[rows],
        spike_events=(source.spike_events * input_support)[rows],
        valid_mask=mask[rows],
        source_image_ids=tuple(source.source_image_ids[i] for i in rows.tolist()),
        trial_indices=tuple(source.trial_indices[i] for i in rows.tolist()),
    )
