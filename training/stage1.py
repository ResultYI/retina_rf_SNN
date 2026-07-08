from __future__ import annotations

from dataclasses import dataclass

import torch

from configs.physiology_profiles import PhysiologyProfile, human_macaque_v1
from data.geometry import PositionArray, local_gaussian_weights, nearest_one_to_one_weights
from models.cells.amacrine import A2AmacrineLayer
from models.cells.bipolar import BipolarLayer
from models.cells.horizontal import H1HorizontalNetwork
from models.cells.rgc import RGCMosaic, RGCPopulationLayer
from models.decoder.local_decoder import DecoderTargets, LocalDecoder
from models.retina_snn import RetinaSNNCore


class Stage1BuildError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Stage1BuildConfig:
    dt_ms: float
    horizon_count: int
    parasol_stride: int = 4
    residual_stride: int = 8


@dataclass(frozen=True, slots=True)
class Stage1OptimizerConfig:
    core_lr: float = 1e-4
    decoder_lr: float = 1e-3
    weight_decay: float = 0.0


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
    )
    parasol_positions = _subsample_positions(
        positions,
        config.parasol_stride,
        "parasol_stride",
    )
    residual_positions = _subsample_positions(
        positions,
        config.residual_stride,
        "residual_stride",
    )
    if residual_positions.shape[0] > parasol_positions.shape[0]:
        residual_positions = residual_positions[: parasol_positions.shape[0]]

    mosaic = RGCMosaic(
        bipolar_positions_degs=positions,
        midget_positions_degs=positions,
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
        A2AmacrineLayer(positions, profile.a2),
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


def _subsample_positions(
    positions: torch.Tensor,
    stride: int,
    name: str,
) -> torch.Tensor:
    if stride < 2:
        raise Stage1BuildError(f"{name} must be at least 2")
    sampled = positions[::stride]
    if sampled.shape[0] == 0:
        return positions[:1]
    if sampled.shape[0] >= positions.shape[0]:
        return positions[: positions.shape[0] - 1]
    return sampled
