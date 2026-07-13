from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

import torch

from configs.physiology_profiles import PhysiologyProfile, human_macaque_v1
from data.geometry import (
    PositionArray,
    local_gaussian_weights,
    nearest_one_to_one_weights,
)
from models.cells.amacrine import LocalAmacrineLayer
from models.cells.bipolar import BipolarLayer
from models.cells.horizontal import H1HorizontalNetwork
from models.cells.rgc import RGCMosaic, RGCPopulationLayer
from models.decoder.local_decoder import DecoderTargets, LocalDecoder
from models.retina_snn import RetinaSNNCore


class Stage1BuildError(ValueError):
    pass


class MidgetSamplingMode(StrEnum):
    FOVEAL_PRIVATE_LINE = "foveal_private_line"
    CONVERGENT = "convergent"


@dataclass(frozen=True, slots=True)
class Stage1BuildConfig:
    dt_ms: float
    horizon_count: int
    eccentricity_deg: float = 0.0
    midget_sampling: MidgetSamplingMode = MidgetSamplingMode.FOVEAL_PRIVATE_LINE
    midget_stride: int = 2
    parasol_stride: int = 4
    residual_stride: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0:
            raise Stage1BuildError("dt_ms must be positive and finite")
        if self.horizon_count < 1:
            raise Stage1BuildError("horizon_count must be positive")
        if not math.isfinite(self.eccentricity_deg) or self.eccentricity_deg < 0:
            raise Stage1BuildError(
                "eccentricity_deg must be finite and non-negative"
            )
        if (
            self.midget_sampling is MidgetSamplingMode.FOVEAL_PRIVATE_LINE
            and self.eccentricity_deg > 0
        ):
            raise Stage1BuildError(
                "Foveal private-line sampling requires zero nominal eccentricity"
            )
        if self.midget_stride < 2:
            raise Stage1BuildError("midget_stride must be at least 2")
        if self.parasol_stride < 2:
            raise Stage1BuildError("parasol_stride must be at least 2")
        if self.residual_stride < self.parasol_stride:
            raise Stage1BuildError(
                "residual_stride must be at least parasol_stride"
            )


@dataclass(frozen=True, slots=True)
class Stage1OptimizerConfig:
    core_lr: float = 1e-4
    decoder_lr: float = 1e-3
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        rates = (self.core_lr, self.decoder_lr)
        if not all(math.isfinite(rate) and rate > 0 for rate in rates):
            raise Stage1BuildError("Optimizer learning rates must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise Stage1BuildError("weight_decay must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Stage1TargetPools:
    fine: torch.Tensor
    coarse: torch.Tensor


@dataclass(frozen=True, slots=True)
class Stage1Components:
    profile: PhysiologyProfile
    mosaic: RGCMosaic
    decoder_targets: DecoderTargets
    target_pools: Stage1TargetPools
    core: RetinaSNNCore
    decoder: LocalDecoder


def build_stage1_components(
    cone_positions_degs: PositionArray,
    config: Stage1BuildConfig,
) -> Stage1Components:
    positions = _positions_tensor(cone_positions_degs)
    profile = human_macaque_v1(
        dt_ms=config.dt_ms,
        horizon_count=config.horizon_count,
        cone_spacing_deg=_median_nearest_neighbor_spacing(positions),
        eccentricity_deg=config.eccentricity_deg,
    )
    parasol_positions = _spatial_subsample_positions(
        positions,
        config.parasol_stride,
    )
    residual_positions = _spatial_subsample_positions(
        parasol_positions,
        math.ceil(config.residual_stride / config.parasol_stride),
    )
    match config.midget_sampling:
        case MidgetSamplingMode.FOVEAL_PRIVATE_LINE:
            midget_positions = positions
        case MidgetSamplingMode.CONVERGENT:
            midget_positions = _spatial_subsample_positions(
                positions,
                config.midget_stride,
            )
        case unreachable:
            assert_never(unreachable)

    mosaic = RGCMosaic(
        bipolar_positions_degs=positions,
        midget_positions_degs=midget_positions,
        parasol_positions_degs=parasol_positions,
        residual_positions_degs=residual_positions,
    )
    targets = DecoderTargets(
        fine_positions_degs=positions,
        coarse_positions_degs=parasol_positions,
    )
    pools = Stage1TargetPools(
        fine=nearest_one_to_one_weights(positions, positions),
        coarse=local_gaussian_weights(
            positions,
            parasol_positions,
            profile.decoder.coarse_radius_degs,
            profile.decoder.coarse_sigma_degs,
        ),
    )
    core = RetinaSNNCore(
        H1HorizontalNetwork(positions, profile.h1),
        BipolarLayer(positions, profile.bipolar),
        LocalAmacrineLayer(positions, profile.amacrine),
        RGCPopulationLayer(mosaic, profile.rgc),
    )
    return Stage1Components(
        profile=profile,
        mosaic=mosaic,
        decoder_targets=targets,
        target_pools=pools,
        core=core,
        decoder=LocalDecoder(mosaic, targets, profile.decoder),
    )


def build_stage1_optimizer(
    core: RetinaSNNCore,
    decoder: LocalDecoder,
    config: Stage1OptimizerConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        (
            {"params": core.parameters(), "lr": config.core_lr},
            {"params": decoder.parameters(), "lr": config.decoder_lr},
        ),
        weight_decay=config.weight_decay,
    )


def _positions_tensor(positions_degs: PositionArray) -> torch.Tensor:
    positions = torch.as_tensor(positions_degs, dtype=torch.float32)
    if (
        positions.ndim != 2
        or positions.shape[0] < 2
        or positions.shape[1] != 2
        or not torch.isfinite(positions).all()
    ):
        raise Stage1BuildError("cone_positions_degs must be finite [Ncone,2]")
    return positions


def _median_nearest_neighbor_spacing(positions: torch.Tensor) -> float:
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    spacing = float(distances.min(dim=1).values.median())
    if not math.isfinite(spacing) or spacing <= 0:
        raise Stage1BuildError("cone positions must be distinct")
    return spacing


def _spatial_subsample_positions(
    positions: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    if positions.shape[0] < 2 or stride < 2:
        return positions[:1]
    lower = positions.min(dim=0).values
    span = positions.max(dim=0).values - lower
    if torch.any(span <= 0):
        return positions[::stride]

    target_count = max(1, positions.shape[0] // stride)
    aspect = float((span[0] / span[1]).item())
    x_cells = max(1, round(math.sqrt(target_count * aspect)))
    y_cells = max(1, math.ceil(target_count / x_cells))
    normalized = (positions - lower) / span
    x_index = (normalized[:, 0] * x_cells).to(torch.long).clamp_max(x_cells - 1)
    y_index = (normalized[:, 1] * y_cells).to(torch.long).clamp_max(y_cells - 1)
    cell_index = x_index * y_cells + y_index
    selected: list[torch.Tensor] = []
    for cell in torch.unique(cell_index, sorted=True):
        members = torch.nonzero(cell_index == cell, as_tuple=False).flatten()
        center = torch.stack(
            (
                (cell // y_cells).to(positions.dtype) + 0.5,
                (cell % y_cells).to(positions.dtype) + 0.5,
            )
        )
        distance = torch.square(
            normalized[members] * positions.new_tensor((x_cells, y_cells)) - center
        ).sum(dim=1)
        selected.append(members[distance.argmin()])
    return positions[torch.stack(selected)]
