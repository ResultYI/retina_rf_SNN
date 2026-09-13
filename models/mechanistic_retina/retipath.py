"""Formal RetiPath entry point; preserves the frozen spatial conductance model."""

from __future__ import annotations

import torch

from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.retipath_spatial_ei import SpatialEIBackend, SpatialEIRetiPath


class RetiPath(SpatialEIRetiPath):
    def __init__(self, config: MechanisticRetinaConfig, cone_positions: torch.Tensor,
                 cell_positions: torch.Tensor, cell_types: tuple[str, ...],
                 polarities: tuple[str, ...], *, rms_e: torch.Tensor, rms_i: torch.Tensor) -> None:
        super().__init__(config, cone_positions, cell_positions, cell_types, polarities,
                         backend=SpatialEIBackend.CONDUCTANCE, rms_e=rms_e, rms_i=rms_i)
