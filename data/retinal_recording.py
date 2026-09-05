from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RealSequenceSplit:
    cone_drive: torch.Tensor
    spike_counts: torch.Tensor
    spike_events: torch.Tensor
    valid_mask: torch.Tensor
    source_image_ids: tuple[str, ...]
    trial_indices: tuple[int, ...]


__all__ = ["RealSequenceSplit"]
