from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.decoder.local_decoder import (
    TiedReadoutGeometry,
    cross_fitted_tied_reconstruction,
)
from training.state import BootstrapState


class BootstrapError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BootstrapReadouts:
    rate_readout: torch.Tensor
    generator_readout: torch.Tensor
    target: torch.Tensor
    persistent_prediction: torch.Tensor


@dataclass(frozen=True, slots=True)
class BootstrapContext:
    geometry: TiedReadoutGeometry
    reconstruction_scale: float
    generator_auxiliary_scale: float


@dataclass(frozen=True, slots=True)
class BootstrapRuntime:
    state: BootstrapState
    parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True, slots=True)
class BootstrapApplication:
    prediction: torch.Tensor
    generator_auxiliary: torch.Tensor


def apply_crossfit_bootstrap(
    readouts: BootstrapReadouts,
    context: BootstrapContext,
    runtime: BootstrapRuntime,
) -> BootstrapApplication:
    rate = cross_fitted_tied_reconstruction(
        readouts.rate_readout,
        readouts.target,
        context.geometry,
    )
    generator = cross_fitted_tied_reconstruction(
        readouts.generator_readout,
        readouts.target,
        context.geometry,
    )
    normalized_generator = generator.loss / context.reconstruction_scale
    if runtime.state.generator_auxiliary_base_weight is None:
        runtime.state.generator_auxiliary_base_weight = (
            calibrate_generator_auxiliary_weight(
                rate.loss / context.reconstruction_scale,
                normalized_generator,
                runtime.parameters,
            )
        )
        runtime.state.generator_auxiliary_calibrated_step = 0
    base_weight = runtime.state.generator_auxiliary_base_weight or 0.0
    auxiliary_weight = (
        base_weight * context.generator_auxiliary_scale
    )
    runtime.state.persistent_reconstruction = float(
        F.mse_loss(
            readouts.persistent_prediction,
            readouts.target,
        ).detach()
    )
    runtime.state.rate_reconstruction = float(rate.loss.detach())
    runtime.state.generator_reconstruction = float(generator.loss.detach())
    runtime.state.generator_auxiliary_weight = auxiliary_weight
    runtime.state.rate_ridge_strength = rate.ridge_strength
    runtime.state.generator_ridge_strength = generator.ridge_strength
    runtime.state.rate_gain_clipped_fraction = (
        rate.gain_clipped_fraction
    )
    runtime.state.generator_gain_clipped_fraction = (
        generator.gain_clipped_fraction
    )
    return BootstrapApplication(
        prediction=rate.prediction,
        generator_auxiliary=auxiliary_weight * normalized_generator,
    )


def calibrate_generator_auxiliary_weight(
    rate_loss: torch.Tensor,
    generator_loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
) -> float:
    active_parameters = tuple(
        parameter for parameter in parameters if parameter.requires_grad
    )
    rate_norm = _loss_gradient_norm(
        rate_loss,
        active_parameters,
    )
    generator_norm = _loss_gradient_norm(
        generator_loss,
        active_parameters,
    )
    if generator_norm <= 0 or not math.isfinite(generator_norm):
        raise BootstrapError(
            "Generator auxiliary calibration requires a finite gradient"
        )
    weight = 0.5 * rate_norm / generator_norm
    return min(10.0, max(0.1, weight))


def _loss_gradient_norm(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt())


__all__ = [
    "BootstrapApplication",
    "BootstrapContext",
    "BootstrapError",
    "BootstrapReadouts",
    "BootstrapRuntime",
    "apply_crossfit_bootstrap",
    "calibrate_generator_auxiliary_weight",
]
