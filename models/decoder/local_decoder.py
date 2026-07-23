from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class LocalDecoderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecoderFit:
    unit_gain: torch.Tensor
    cone_bias: torch.Tensor
    train_mse: float
    ridge_strength: float = 0.0
    gain_clipped_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class TiedReadoutGeometry:
    spatial_weights: torch.Tensor
    prior_gain: torch.Tensor
    gain_max: float


@dataclass(frozen=True, slots=True)
class CrossFitResult:
    prediction: torch.Tensor
    loss: torch.Tensor
    fold_mse: torch.Tensor
    ridge_strength: float
    gain_clipped_fraction: float


class TiedLocalDecoder(nn.Module):
    def __init__(self, unit_count: int, cone_count: int, gain_max: float = 5.0) -> None:
        super().__init__()
        if unit_count < 1 or cone_count < 1:
            raise LocalDecoderError("unit_count and cone_count must be positive")
        if gain_max <= 0:
            raise LocalDecoderError("gain_max must be positive")
        initial_gain = torch.full((2, unit_count), 0.10 / gain_max)
        self.raw_unit_gain = nn.Parameter(torch.logit(initial_gain))
        self.cone_bias = nn.Parameter(torch.zeros(cone_count))
        self._unit_count = unit_count
        self._cone_count = cone_count
        self._gain_max = gain_max

    @property
    def unit_gain(self) -> torch.Tensor:
        return self._gain_max * torch.sigmoid(self.raw_unit_gain)

    @property
    def gain_max(self) -> float:
        return self._gain_max

    @torch.no_grad()
    def initialize(
        self,
        unit_gain: torch.Tensor,
        cone_bias: torch.Tensor,
    ) -> None:
        if unit_gain.shape != (2, self._unit_count):
            raise LocalDecoderError("unit_gain must have shape [polarity,unit]")
        if cone_bias.shape != (self._cone_count,):
            raise LocalDecoderError("cone_bias must have shape [cone]")
        bounded = unit_gain.to(self.raw_unit_gain).clamp(
            torch.finfo(self.raw_unit_gain.dtype).eps,
            self._gain_max - torch.finfo(self.raw_unit_gain.dtype).eps,
        )
        self.raw_unit_gain.copy_(
            torch.logit(bounded / self._gain_max)
        )
        self.cone_bias.copy_(cone_bias.to(self.cone_bias))

    def forward(self, rates: torch.Tensor, spatial_weights: torch.Tensor) -> torch.Tensor:
        if rates.ndim != 4 or rates.shape[-2:] != (2, self._unit_count):
            raise LocalDecoderError("rates must have shape [batch,time,polarity,unit]")
        if spatial_weights.shape != (self._unit_count, self._cone_count):
            raise LocalDecoderError("spatial_weights must have shape [unit,cone]")
        signed = (
            self.unit_gain[0] * rates[:, :, 0]
            - self.unit_gain[1] * rates[:, :, 1]
        )
        return signed @ spatial_weights + self.cone_bias


@torch.no_grad()
def fit_regularized_tied_decoder_probe(
    readout: torch.Tensor,
    target: torch.Tensor,
    geometry: TiedReadoutGeometry,
) -> DecoderFit:
    _validate_readout(readout, target, geometry)
    on_features = torch.einsum(
        "btu,uc->btcu",
        readout[:, :, 0],
        geometry.spatial_weights,
    )
    off_features = -torch.einsum(
        "btu,uc->btcu",
        readout[:, :, 1],
        geometry.spatial_weights,
    )
    solution, bias, mse, ridge_strength, clipped_fraction = _ridge_bounded_fit(
        torch.cat((on_features, off_features), dim=-1),
        target,
        geometry.gain_max,
        geometry.prior_gain.flatten(),
    )
    return DecoderFit(
        solution.reshape(2, readout.shape[-1]),
        bias,
        mse,
        ridge_strength,
        clipped_fraction,
    )


def decode_with_fit(
    readout: torch.Tensor,
    spatial_weights: torch.Tensor,
    unit_gain: torch.Tensor,
    cone_bias: torch.Tensor,
) -> torch.Tensor:
    signed = (
        unit_gain[0] * readout[:, :, 0]
        - unit_gain[1] * readout[:, :, 1]
    )
    return signed @ spatial_weights + cone_bias


def cross_fitted_tied_reconstruction(
    readout: torch.Tensor,
    target: torch.Tensor,
    geometry: TiedReadoutGeometry,
) -> CrossFitResult:
    _validate_readout(readout, target, geometry)
    if readout.shape[0] < 3:
        raise LocalDecoderError("Cross-fitted reconstruction requires three sources")
    predictions: list[torch.Tensor] = []
    fold_mse: list[torch.Tensor] = []
    ridge_strengths: list[float] = []
    clipped_fractions: list[float] = []
    for held_out_index in range(readout.shape[0]):
        training_mask = torch.arange(readout.shape[0], device=readout.device) != held_out_index
        fit = fit_regularized_tied_decoder_probe(
            readout[training_mask],
            target[training_mask],
            geometry,
        )
        prediction = decode_with_fit(
            readout[held_out_index : held_out_index + 1],
            geometry.spatial_weights,
            fit.unit_gain,
            fit.cone_bias,
        )
        predictions.append(prediction)
        fold_mse.append(
            F.mse_loss(
                prediction.detach(),
                target[held_out_index : held_out_index + 1],
            )
        )
        ridge_strengths.append(fit.ridge_strength)
        clipped_fractions.append(fit.gain_clipped_fraction)
    prediction = torch.cat(predictions)
    return CrossFitResult(
        prediction=prediction,
        loss=F.mse_loss(prediction, target),
        fold_mse=torch.stack(fold_mse),
        ridge_strength=sum(ridge_strengths) / len(ridge_strengths),
        gain_clipped_fraction=sum(clipped_fractions) / len(clipped_fractions),
    )


def _ridge_bounded_fit(
    features: torch.Tensor,
    target: torch.Tensor,
    gain_max: float,
    prior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    feature_mean = features.mean(dim=(0, 1), keepdim=True)
    target_mean = target.mean(dim=(0, 1), keepdim=True)
    design = (features - feature_mean).reshape(-1, features.shape[-1]).double()
    response = (target - target_mean).reshape(-1).double()
    prior_double = prior.reshape(-1).double()
    gram = design.T @ design
    mean_feature_energy = (
        torch.trace(gram) / max(1, design.shape[1])
    ).clamp_min(torch.finfo(design.dtype).eps)
    ridge = 1e-3 * mean_feature_energy
    system = gram + ridge * torch.eye(
        design.shape[1],
        device=design.device,
        dtype=design.dtype,
    )
    solution = torch.linalg.solve(
        system,
        design.T @ response + ridge * prior_double,
    )
    bounded_solution = solution.clamp(0.0, gain_max)
    clipped_fraction = float((bounded_solution != solution).float().mean())
    bounded_solution = bounded_solution.to(target.dtype)
    bias = (
        target
        - (features * bounded_solution).sum(dim=-1)
    ).mean(dim=(0, 1))
    prediction = (features * bounded_solution).sum(dim=-1) + bias
    return (
        bounded_solution,
        bias,
        float((prediction - target).square().mean().detach()),
        float(ridge.detach()),
        clipped_fraction,
    )


def _validate_readout(
    readout: torch.Tensor,
    target: torch.Tensor,
    geometry: TiedReadoutGeometry,
) -> None:
    if readout.ndim != 4 or readout.shape[2] != 2:
        raise LocalDecoderError(
            "readout must have shape [batch,time,polarity,unit]"
        )
    if target.shape != (*readout.shape[:2], geometry.spatial_weights.shape[1]):
        raise LocalDecoderError("target shape is incompatible with readout")
    if (
        geometry.spatial_weights.shape[0] != readout.shape[-1]
        or geometry.prior_gain.shape != (2, readout.shape[-1])
        or geometry.gain_max <= 0
    ):
        raise LocalDecoderError("decoder geometry or gain bound is invalid")


__all__ = [
    "CrossFitResult",
    "DecoderFit",
    "LocalDecoderError",
    "TiedLocalDecoder",
    "TiedReadoutGeometry",
    "cross_fitted_tied_reconstruction",
    "decode_with_fit",
    "fit_regularized_tied_decoder_probe",
]
