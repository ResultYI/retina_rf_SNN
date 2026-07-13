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
    fine_prediction_scale: float = 1.0
    coarse_prediction_scale: float = 1.0
    rate_weight: float = 1e-3
    homeostasis_weight: float = 1e-3
    residual_activity_weight: float = 1e-3
    residual_decoder_weight: float = 1e-3
    homeostasis_rate_min: float = 0.01
    homeostasis_rate_max: float = 0.20

    def __post_init__(self) -> None:
        weights = (
            self.fine_weight,
            self.coarse_weight,
            self.rate_weight,
            self.homeostasis_weight,
            self.residual_activity_weight,
            self.residual_decoder_weight,
        )
        if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
            raise RetinaLossError("Loss weights must be finite and non-negative")
        scales = (self.fine_prediction_scale, self.coarse_prediction_scale)
        if not all(math.isfinite(scale) and scale > 0 for scale in scales):
            raise RetinaLossError("Prediction scales must be finite and positive")
        if not (
            math.isfinite(self.homeostasis_rate_min)
            and math.isfinite(self.homeostasis_rate_max)
            and 0 <= self.homeostasis_rate_min < self.homeostasis_rate_max <= 1
        ):
            raise RetinaLossError("Homeostasis rate band must lie inside [0,1]")


@dataclass(frozen=True, slots=True)
class RetinaLosses:
    total: torch.Tensor
    prediction_fine: torch.Tensor
    prediction_coarse: torch.Tensor
    rate_regularization: torch.Tensor
    homeostasis: torch.Tensor
    residual_activity: torch.Tensor
    residual_decoder_weight: torch.Tensor

    def detached(self) -> RetinaLosses:
        return RetinaLosses(
            total=self.total.detach(),
            prediction_fine=self.prediction_fine.detach(),
            prediction_coarse=self.prediction_coarse.detach(),
            rate_regularization=self.rate_regularization.detach(),
            homeostasis=self.homeostasis.detach(),
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
        population_rates = torch.stack((rates.midget.mean(), rates.parasol.mean()))
        homeostasis = (
            torch.relu(self.config.homeostasis_rate_min - population_rates).square()
            + torch.relu(population_rates - self.config.homeostasis_rate_max).square()
        ).mean()
        residual_activity = rates.residual.abs().mean()
        total = (
            self.config.fine_weight
            * prediction_fine
            / self.config.fine_prediction_scale
            + self.config.coarse_weight
            * prediction_coarse
            / self.config.coarse_prediction_scale
            + self.config.rate_weight * rate_regularization
            + self.config.homeostasis_weight * homeostasis
            + self.config.residual_activity_weight * residual_activity
            + self.config.residual_decoder_weight * residual_decoder_penalty
        )
        return RetinaLosses(
            total=total,
            prediction_fine=prediction_fine,
            prediction_coarse=prediction_coarse,
            rate_regularization=rate_regularization,
            homeostasis=homeostasis,
            residual_activity=residual_activity,
            residual_decoder_weight=residual_decoder_penalty,
        )
