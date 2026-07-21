from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn

from data.geometry import PositionArray, local_gaussian_weights
from models.cells.rgc import RGCOutput, RGCMosaic


class LocalDecoderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecoderTargets:
    current_positions_degs: PositionArray


@dataclass(frozen=True, slots=True)
class LocalDecoderConfig:
    current_radius_degs: float
    current_sigma_degs: float
    current_weight_max: float

    def __post_init__(self) -> None:
        scales = (
            self.current_radius_degs,
            self.current_sigma_degs,
            self.current_weight_max,
        )
        if not all(math.isfinite(value) and value > 0 for value in scales):
            raise LocalDecoderError("Decoder scales and weight bound must be positive")


@dataclass(frozen=True, slots=True)
class LocalDecoderOutput:
    target_current: torch.Tensor


class LocalDecoderDiagnostics(TypedDict):
    decoder_midget_weight_norm: torch.Tensor
    decoder_parasol_weight_norm: torch.Tensor
    decoder_reconstruction_current_std: torch.Tensor


class LocalDecoder(nn.Module):
    def __init__(
        self,
        mosaic: RGCMosaic,
        targets: DecoderTargets,
        config: LocalDecoderConfig,
    ) -> None:
        super().__init__()
        current = targets.current_positions_degs
        self.current_midget = _LocalProjection(
            mosaic.midget_positions_degs,
            current,
            config.current_radius_degs,
            config.current_sigma_degs,
            config.current_weight_max,
        )
        self.current_parasol = _LocalProjection(
            mosaic.parasol_positions_degs,
            current,
            config.current_radius_degs,
            config.current_sigma_degs,
            config.current_weight_max,
        )
        _assert_combined_coverage(
            (self.current_midget, self.current_parasol),
        )

    def set_spatial_calibration_trainable(self) -> None:
        for projection in (self.current_midget, self.current_parasol):
            projection.raw_spatial_values.requires_grad_(True)
            projection.raw_weight.requires_grad_(False)

    def forward(
        self,
        rgc_output: RGCOutput,
        return_diagnostics: bool = False,
    ) -> LocalDecoderOutput | tuple[LocalDecoderOutput, LocalDecoderDiagnostics]:
        rates = rgc_output.rates
        prefix = _validate_population(
            "midget",
            rates.midget,
            self.current_midget.source_count,
        )
        if _validate_population(
            "parasol",
            rates.parasol,
            self.current_parasol.source_count,
        ) != prefix:
            raise LocalDecoderError("RGC population batch/time dimensions must match")

        if len(prefix) == 1:
            midget_rates = rates.midget
            parasol_rates = rates.parasol
        elif len(prefix) == 2:
            midget_rates = rates.midget[:, -1]
            parasol_rates = rates.parasol[:, -1]
        else:
            raise LocalDecoderError(
                "RGC rates must have batch or batch/time leading dimensions"
            )
        target_current = self.current_midget(midget_rates) + self.current_parasol(
            parasol_rates
        )
        output = LocalDecoderOutput(target_current=target_current)
        if not return_diagnostics:
            return output

        diagnostics = LocalDecoderDiagnostics(
            decoder_midget_weight_norm=self.current_midget.effective_weight.detach()
            .square()
            .sum()
            .sqrt(),
            decoder_parasol_weight_norm=self.current_parasol.effective_weight.detach()
            .square()
            .sum()
            .sqrt(),
            decoder_reconstruction_current_std=output.target_current.detach().std(
                unbiased=False
            ),
        )
        return output, diagnostics


class _LocalProjection(nn.Module):
    def __init__(
        self,
        source_positions: PositionArray,
        target_positions: PositionArray,
        radius_degs: float,
        sigma_degs: float,
        weight_max: float,
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
        self.raw_spatial_values = nn.Parameter(local_mask.values().log())
        self.raw_weight = nn.Parameter(torch.atanh(torch.tensor((0.1, -0.1))))
        self.source_count = source.shape[0]
        self.target_count = local_mask.shape[0]
        self._weight_max = weight_max

    @property
    def effective_weight(self) -> torch.Tensor:
        return self._weight_max * torch.tanh(self.raw_weight)

    @property
    def effective_spatial_pool(self) -> torch.Tensor:
        logits = torch.sparse_coo_tensor(
            self.local_mask.indices(),
            self.raw_spatial_values,
            self.local_mask.shape,
            device=self.raw_spatial_values.device,
        ).coalesce()
        return torch.sparse.softmax(logits, dim=1).coalesce()

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        prefix = source.shape[:-2]
        flat_source = source.reshape(-1, self.source_count)
        pooled = torch.sparse.mm(self.effective_spatial_pool, flat_source.T).T.reshape(
            *prefix,
            2,
            self.target_count,
        )
        return torch.einsum("...pn,p->...n", pooled, self.effective_weight)


def _validate_population(
    name: str,
    population: torch.Tensor,
    source_count: int,
) -> tuple[int, ...]:
    if population.ndim < 3 or population.shape[-2:] != (2, source_count):
        raise LocalDecoderError(f"{name} rates must end with shape [2,{source_count}]")
    if not torch.isfinite(population).all():
        raise LocalDecoderError(f"{name} rates contain NaN or inf")
    return population.shape[:-2]


def _assert_combined_coverage(
    projections: tuple[_LocalProjection, ...],
) -> None:
    covered = torch.zeros(projections[0].local_mask.shape[0], dtype=torch.bool)
    for projection in projections:
        row_sums = torch.sparse.sum(projection.local_mask, dim=1).to_dense()
        covered |= row_sums > 0
    if not torch.all(covered):
        raise LocalDecoderError("current decoder coverage has empty target rows")
