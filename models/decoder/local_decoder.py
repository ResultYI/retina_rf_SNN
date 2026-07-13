from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn

from data.geometry import PositionArray, local_gaussian_weights
from models.cells.rgc import RGCOutput, RGCMosaic, RGCPopulationTensors


class LocalDecoderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecoderTargets:
    fine_positions_degs: PositionArray
    coarse_positions_degs: PositionArray


@dataclass(frozen=True, slots=True)
class LocalDecoderConfig:
    horizon_count: int
    fine_radius_degs: float
    fine_sigma_degs: float
    coarse_radius_degs: float
    coarse_sigma_degs: float
    residual_weight_max: float

    def __post_init__(self) -> None:
        scales = (
            self.fine_radius_degs,
            self.fine_sigma_degs,
            self.coarse_radius_degs,
            self.coarse_sigma_degs,
            self.residual_weight_max,
        )
        if self.horizon_count < 1:
            raise LocalDecoderError("horizon_count must be positive")
        if not all(math.isfinite(value) and value > 0 for value in scales):
            raise LocalDecoderError("Decoder scales and weight bound must be positive")


@dataclass(frozen=True, slots=True)
class LocalDecoderOutput:
    target_fine: torch.Tensor
    target_coarse: torch.Tensor


class LocalDecoderDiagnostics(TypedDict):
    decoder_midget_weight_norm: torch.Tensor
    decoder_parasol_weight_norm: torch.Tensor
    decoder_residual_weight_norm: torch.Tensor
    decoder_prediction_fine_std: torch.Tensor
    decoder_prediction_coarse_std: torch.Tensor


class LocalDecoder(nn.Module):
    def __init__(
        self,
        mosaic: RGCMosaic,
        targets: DecoderTargets,
        config: LocalDecoderConfig,
    ) -> None:
        super().__init__()
        fine = targets.fine_positions_degs
        coarse = targets.coarse_positions_degs
        self.fine_midget = _LocalProjection(
            mosaic.midget_positions_degs,
            fine,
            config.fine_radius_degs,
            config.fine_sigma_degs,
            config.horizon_count,
        )
        self.fine_parasol = _LocalProjection(
            mosaic.parasol_positions_degs,
            fine,
            config.fine_radius_degs,
            config.fine_sigma_degs,
            config.horizon_count,
        )
        self.fine_residual = _LocalProjection(
            mosaic.residual_positions_degs,
            fine,
            config.fine_radius_degs,
            config.fine_sigma_degs,
            config.horizon_count,
            config.residual_weight_max,
        )
        self.coarse_midget = _LocalProjection(
            mosaic.midget_positions_degs,
            coarse,
            config.coarse_radius_degs,
            config.coarse_sigma_degs,
            config.horizon_count,
        )
        self.coarse_parasol = _LocalProjection(
            mosaic.parasol_positions_degs,
            coarse,
            config.coarse_radius_degs,
            config.coarse_sigma_degs,
            config.horizon_count,
        )
        self.coarse_residual = _LocalProjection(
            mosaic.residual_positions_degs,
            coarse,
            config.coarse_radius_degs,
            config.coarse_sigma_degs,
            config.horizon_count,
            config.residual_weight_max,
        )
        _assert_combined_coverage(
            "fine",
            (self.fine_midget, self.fine_parasol, self.fine_residual),
        )
        _assert_combined_coverage(
            "coarse",
            (self.coarse_midget, self.coarse_parasol, self.coarse_residual),
        )

    def forward(
        self,
        rgc_output: RGCOutput,
        return_diagnostics: bool = False,
    ) -> (
        LocalDecoderOutput
        | tuple[LocalDecoderOutput, LocalDecoderDiagnostics]
    ):
        rates = rgc_output.rates
        prefix = _validate_population(
            "midget",
            rates.midget,
            self.fine_midget.source_count,
        )
        if _validate_population(
            "parasol",
            rates.parasol,
            self.fine_parasol.source_count,
        ) != prefix:
            raise LocalDecoderError("RGC population batch/time dimensions must match")
        if _validate_population(
            "residual",
            rates.residual,
            self.fine_residual.source_count,
        ) != prefix:
            raise LocalDecoderError("RGC population batch/time dimensions must match")

        fine_midget = self.fine_midget(rates.midget)
        fine_parasol = self.fine_parasol(rates.parasol)
        fine_residual = self.fine_residual(rates.residual)
        coarse_midget = self.coarse_midget(rates.midget)
        coarse_parasol = self.coarse_parasol(rates.parasol)
        coarse_residual = self.coarse_residual(rates.residual)
        output = LocalDecoderOutput(
            target_fine=fine_midget + fine_parasol + fine_residual,
            target_coarse=coarse_midget + coarse_parasol + coarse_residual,
        )
        if not return_diagnostics:
            return output

        diagnostics = LocalDecoderDiagnostics(
            decoder_midget_weight_norm=_combined_norm(
                self.fine_midget,
                self.coarse_midget,
            ).detach(),
            decoder_parasol_weight_norm=_combined_norm(
                self.fine_parasol,
                self.coarse_parasol,
            ).detach(),
            decoder_residual_weight_norm=_combined_norm(
                self.fine_residual,
                self.coarse_residual,
            ).detach(),
            decoder_prediction_fine_std=output.target_fine.detach().std(
                unbiased=False
            ),
            decoder_prediction_coarse_std=output.target_coarse.detach().std(
                unbiased=False
            ),
        )
        return output, diagnostics

    def residual_weight_penalty(self) -> torch.Tensor:
        return (
            self.fine_residual.effective_weight.square().sum()
            + self.coarse_residual.effective_weight.square().sum()
        )


class _LocalProjection(nn.Module):
    def __init__(
        self,
        source_positions: PositionArray,
        target_positions: PositionArray,
        radius_degs: float,
        sigma_degs: float,
        horizon_count: int,
        weight_max: float | None = None,
    ) -> None:
        super().__init__()
        source = torch.as_tensor(source_positions, dtype=torch.float32)
        local_mask = local_gaussian_weights(
            source,
            target_positions,
            radius_degs,
            sigma_degs,
            allow_empty_rows=True,
        ).coalesce()
        self.register_buffer("local_mask", local_mask)
        self.raw_weight = nn.Parameter(torch.zeros(horizon_count, 2))
        self.source_count = source.shape[0]
        self._weight_max = weight_max

    @property
    def effective_weight(self) -> torch.Tensor:
        if self._weight_max is None:
            return self.raw_weight
        return self._weight_max * torch.tanh(self.raw_weight)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        prefix = source.shape[:-2]
        flat_source = source.reshape(-1, self.source_count)
        pooled = torch.sparse.mm(self.local_mask, flat_source.T).T.reshape(
            *prefix,
            2,
            self.local_mask.shape[0],
        )
        return torch.einsum("...pn,hp->...hn", pooled, self.effective_weight)


def _validate_population(
    name: str,
    population: torch.Tensor,
    source_count: int,
) -> tuple[int, ...]:
    if population.ndim < 3 or population.shape[-2:] != (2, source_count):
        raise LocalDecoderError(
            f"{name} rates must end with shape [2,{source_count}]"
        )
    if not torch.isfinite(population).all():
        raise LocalDecoderError(f"{name} rates contain NaN or inf")
    return population.shape[:-2]


def _combined_norm(
    first: _LocalProjection,
    second: _LocalProjection,
) -> torch.Tensor:
    return torch.sqrt(
        first.effective_weight.square().sum()
        + second.effective_weight.square().sum()
    )


def _assert_combined_coverage(
    name: str,
    projections: tuple[_LocalProjection, ...],
) -> None:
    covered = torch.zeros(projections[0].local_mask.shape[0], dtype=torch.bool)
    for projection in projections:
        row_sums = torch.sparse.sum(projection.local_mask, dim=1).to_dense()
        covered |= row_sums > 0
    if not torch.all(covered):
        raise LocalDecoderError(f"{name} decoder coverage has empty target rows")
