from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from models.cells.rgc import RGCOutput
from models.decoder.local_decoder import LocalDecoderOutput

if TYPE_CHECKING:
    from training.hybrid import RetinaTargets


class RetinaLossError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetinaLossConfig:
    fine_weight: float = 1.0
    coarse_weight: float = 1.0
    rate_weight: float = 1e-3
    homeostasis_weight: float = 1e-3
    decorrelation_weight: float = 1e-4
    residual_activity_weight: float = 1e-3
    residual_decoder_weight: float = 1e-3
    target_rate: float = 0.05

    def __post_init__(self) -> None:
        weights = (
            self.fine_weight,
            self.coarse_weight,
            self.rate_weight,
            self.homeostasis_weight,
            self.decorrelation_weight,
            self.residual_activity_weight,
            self.residual_decoder_weight,
        )
        if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
            raise RetinaLossError("Loss weights must be finite and non-negative")
        if not math.isfinite(self.target_rate) or not 0 <= self.target_rate <= 1:
            raise RetinaLossError("target_rate must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class RetinaLosses:
    total: torch.Tensor
    prediction_fine: torch.Tensor
    prediction_coarse: torch.Tensor
    rate_regularization: torch.Tensor
    homeostasis: torch.Tensor
    decorrelation: torch.Tensor
    residual_activity: torch.Tensor
    residual_decoder_weight: torch.Tensor

    def detached(self) -> RetinaLosses:
        return RetinaLosses(
            total=self.total.detach(),
            prediction_fine=self.prediction_fine.detach(),
            prediction_coarse=self.prediction_coarse.detach(),
            rate_regularization=self.rate_regularization.detach(),
            homeostasis=self.homeostasis.detach(),
            decorrelation=self.decorrelation.detach(),
            residual_activity=self.residual_activity.detach(),
            residual_decoder_weight=self.residual_decoder_weight.detach(),
        )


class RetinaObjective(nn.Module):
    def __init__(self, config: RetinaLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        prediction: LocalDecoderOutput,
        targets: RetinaTargets,
        rgc_history: RGCOutput,
        residual_decoder_penalty: torch.Tensor,
    ) -> RetinaLosses:
        if prediction.target_fine.shape != targets.fine.shape:
            raise RetinaLossError("Fine prediction and target shapes must match")
        if prediction.target_coarse.shape != targets.coarse.shape:
            raise RetinaLossError("Coarse prediction and target shapes must match")
        if not torch.isfinite(targets.fine).all() or not torch.isfinite(
            targets.coarse
        ).all():
            raise RetinaLossError("Prediction targets must be finite")

        rates = rgc_history.rates
        prediction_fine = F.mse_loss(prediction.target_fine, targets.fine)
        prediction_coarse = F.mse_loss(prediction.target_coarse, targets.coarse)
        rate_regularization = (
            rates.midget.square().mean()
            + rates.parasol.square().mean()
            + rates.residual.square().mean()
        ) / 3
        homeostasis = (
            (rates.midget.mean() - self.config.target_rate).square()
            + (rates.parasol.mean() - self.config.target_rate).square()
        ) / 2
        decorrelation = _squared_trace_correlation(
            rates.midget,
            rates.parasol,
        )
        residual_activity = rates.residual.abs().mean()
        total = (
            self.config.fine_weight * prediction_fine
            + self.config.coarse_weight * prediction_coarse
            + self.config.rate_weight * rate_regularization
            + self.config.homeostasis_weight * homeostasis
            + self.config.decorrelation_weight * decorrelation
            + self.config.residual_activity_weight * residual_activity
            + self.config.residual_decoder_weight * residual_decoder_penalty
        )
        return RetinaLosses(
            total=total,
            prediction_fine=prediction_fine,
            prediction_coarse=prediction_coarse,
            rate_regularization=rate_regularization,
            homeostasis=homeostasis,
            decorrelation=decorrelation,
            residual_activity=residual_activity,
            residual_decoder_weight=residual_decoder_penalty,
        )


def _squared_trace_correlation(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    first_trace = first.mean(dim=(-2, -1)).flatten()
    second_trace = second.mean(dim=(-2, -1)).flatten()
    first_centered = first_trace - first_trace.mean()
    second_centered = second_trace - second_trace.mean()
    denominator = (
        torch.linalg.vector_norm(first_centered)
        * torch.linalg.vector_norm(second_centered)
    ).clamp_min(torch.finfo(first.dtype).eps)
    correlation = torch.dot(first_centered, second_centered) / denominator
    return correlation.square()
