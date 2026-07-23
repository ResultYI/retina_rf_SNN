from __future__ import annotations

from dataclasses import dataclass

import torch


class DecoderDiagnosticError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecoderFit:
    unit_gain: torch.Tensor
    cone_bias: torch.Tensor
    train_mse: float


@dataclass(frozen=True, slots=True)
class DecoderCoverage:
    minimum: float
    maximum: float
    mean: float
    coefficient_of_variation: float
    edge_to_center_ratio: float


@torch.no_grad()
def fit_global_decoder(
    rates: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
    *,
    gain_max: float,
) -> DecoderFit:
    _validate_inputs(rates, target, spatial_weights, gain_max)
    features = torch.stack(
        (
            rates[:, :, 0] @ spatial_weights,
            -(rates[:, :, 1] @ spatial_weights),
        ),
        dim=-1,
    )
    solution, bias, mse = _bounded_fit(features, target, gain_max)
    unit_gain = solution[:, None].expand(2, rates.shape[-1]).clone()
    return DecoderFit(unit_gain, bias, mse)


@torch.no_grad()
def fit_tied_decoder_ceiling(
    rates: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
    *,
    gain_max: float,
) -> DecoderFit:
    _validate_inputs(rates, target, spatial_weights, gain_max)
    on_features = torch.einsum(
        "btu,uc->btcu",
        rates[:, :, 0],
        spatial_weights,
    )
    off_features = -torch.einsum(
        "btu,uc->btcu",
        rates[:, :, 1],
        spatial_weights,
    )
    solution, bias, mse = _bounded_fit(
        torch.cat((on_features, off_features), dim=-1),
        target,
        gain_max,
    )
    return DecoderFit(
        solution.reshape(2, rates.shape[-1]),
        bias,
        mse,
    )


def decode_with_fit(
    rates: torch.Tensor,
    spatial_weights: torch.Tensor,
    unit_gain: torch.Tensor,
    cone_bias: torch.Tensor,
) -> torch.Tensor:
    signed = unit_gain[0] * rates[:, :, 0] - unit_gain[1] * rates[:, :, 1]
    return signed @ spatial_weights + cone_bias


def decoder_coverage(
    spatial_weights: torch.Tensor,
    positions_degs: torch.Tensor,
) -> DecoderCoverage:
    if spatial_weights.ndim != 2:
        raise DecoderDiagnosticError("spatial_weights must have shape [unit,cone]")
    if positions_degs.shape != (spatial_weights.shape[1], 2):
        raise DecoderDiagnosticError("positions_degs must have shape [cone,2]")
    coverage = spatial_weights.sum(dim=0)
    radius = (positions_degs - positions_degs.mean(dim=0)).norm(dim=1)
    edge_count = max(1, positions_degs.shape[0] // 4)
    edge_indices = torch.topk(radius, edge_count).indices
    edge_mask = torch.zeros_like(radius, dtype=torch.bool)
    edge_mask[edge_indices] = True
    center_mask = ~edge_mask
    center_mean = (
        coverage[center_mask].mean()
        if center_mask.any()
        else coverage.mean()
    )
    mean = coverage.mean()
    return DecoderCoverage(
        minimum=float(coverage.min()),
        maximum=float(coverage.max()),
        mean=float(mean),
        coefficient_of_variation=float(
            coverage.std(unbiased=False) / mean.clamp_min(1e-12)
        ),
        edge_to_center_ratio=float(
            coverage[edge_mask].mean() / center_mean.clamp_min(1e-12)
        ),
    )


def _bounded_fit(
    features: torch.Tensor,
    target: torch.Tensor,
    gain_max: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    feature_mean = features.mean(dim=(0, 1), keepdim=True)
    target_mean = target.mean(dim=(0, 1), keepdim=True)
    design = (features - feature_mean).reshape(-1, features.shape[-1]).double()
    response = (target - target_mean).reshape(-1, 1).double()
    solution = torch.linalg.lstsq(design, response).solution.squeeze(-1)
    solution = solution.clamp(0.0, gain_max).to(target.dtype)
    bias = (
        target
        - (features * solution).sum(dim=-1)
    ).mean(dim=(0, 1))
    prediction = (features * solution).sum(dim=-1) + bias
    return solution, bias, float((prediction - target).square().mean().detach())


def _validate_inputs(
    rates: torch.Tensor,
    target: torch.Tensor,
    spatial_weights: torch.Tensor,
    gain_max: float,
) -> None:
    if rates.ndim != 4 or rates.shape[2] != 2:
        raise DecoderDiagnosticError(
            "rates must have shape [batch,time,polarity,unit]"
        )
    if target.shape != (*rates.shape[:2], spatial_weights.shape[1]):
        raise DecoderDiagnosticError("target shape is incompatible with rates")
    if spatial_weights.shape[0] != rates.shape[-1] or gain_max <= 0:
        raise DecoderDiagnosticError("decoder geometry or gain bound is invalid")


__all__ = [
    "DecoderCoverage",
    "DecoderDiagnosticError",
    "DecoderFit",
    "decode_with_fit",
    "decoder_coverage",
    "fit_global_decoder",
    "fit_tied_decoder_ceiling",
]
