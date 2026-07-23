from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from training.state import BootstrapState


class BootstrapError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MultiViewReadouts:
    generator_readout: torch.Tensor
    target: torch.Tensor
    persistent_prediction: torch.Tensor


@dataclass(frozen=True, slots=True)
class MultiViewBootstrapContext:
    reconstruction_scale: float
    view_consistency_scale: float
    generator_variance_weight: float


@dataclass(frozen=True, slots=True)
class MultiViewBootstrapRuntime:
    state: BootstrapState
    parameters: tuple[nn.Parameter, ...]
    optimizer_step: int = 0


@dataclass(frozen=True, slots=True)
class MultiViewBootstrapApplication:
    prediction: torch.Tensor
    auxiliary_loss: torch.Tensor


def apply_multiview_bootstrap(
    readouts: MultiViewReadouts,
    context: MultiViewBootstrapContext,
    runtime: MultiViewBootstrapRuntime,
) -> MultiViewBootstrapApplication:
    if readouts.generator_readout.shape[0] % 2 != 0:
        raise BootstrapError("Multi-view bootstrap requires paired batch rows")
    first, second = readouts.generator_readout.chunk(2)
    combined = torch.cat((first, second))
    mean = combined.mean(dim=(0, 1), keepdim=True).detach()
    standard_deviation = (
        combined.std(dim=(0, 1), unbiased=False, keepdim=True)
        .clamp_min(torch.finfo(combined.dtype).eps)
        .detach()
    )
    consistency = F.mse_loss(
        (first - mean) / standard_deviation,
        (second - mean) / standard_deviation,
    )
    current_variance = combined.std(dim=(0, 1), unbiased=False)
    if runtime.state.initial_generator_variance_reference is None:
        runtime.state.initial_generator_variance_reference = (
            current_variance.detach().cpu()
        )
    reference = runtime.state.initial_generator_variance_reference.to(
        current_variance
    )
    variance_guard = F.relu(0.5 * reference - current_variance).square().mean()
    retention = (
        current_variance / reference.clamp_min(torch.finfo(reference.dtype).eps)
    ).clamp(max=1.0).mean()
    primary = F.mse_loss(
        readouts.persistent_prediction,
        readouts.target,
    ) / context.reconstruction_scale
    if runtime.state.view_consistency_base_weight is None:
        runtime.state.view_consistency_base_weight = (
            calibrate_view_consistency_weight(
                primary,
                consistency,
                runtime.parameters,
            )
        )
        runtime.state.view_consistency_calibrated_step = runtime.optimizer_step
    base_weight = runtime.state.view_consistency_base_weight or 0.0
    consistency_weight = base_weight * context.view_consistency_scale
    auxiliary_loss = (
        consistency_weight * consistency
        + context.generator_variance_weight * variance_guard
    )
    runtime.state.persistent_reconstruction = float(primary.detach())
    runtime.state.view_consistency = float(consistency.detach())
    runtime.state.view_consistency_weight = consistency_weight
    runtime.state.generator_variance_guard = float(variance_guard.detach())
    runtime.state.generator_variance_retention = float(retention.detach())
    return MultiViewBootstrapApplication(
        prediction=readouts.persistent_prediction,
        auxiliary_loss=auxiliary_loss,
    )


def calibrate_view_consistency_weight(
    primary_loss: torch.Tensor,
    consistency_loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
) -> float:
    active_parameters = tuple(
        parameter for parameter in parameters if parameter.requires_grad
    )
    primary_norm = _loss_gradient_norm(primary_loss, active_parameters)
    consistency_norm = _loss_gradient_norm(
        consistency_loss,
        active_parameters,
    )
    if consistency_norm <= 0 or not math.isfinite(consistency_norm):
        raise BootstrapError(
            "View consistency calibration requires a finite gradient"
        )
    weight = 0.25 * primary_norm / consistency_norm
    return min(5.0, max(0.05, weight))


def bootstrap_metrics(state: BootstrapState) -> dict[str, float]:
    return {
        "bootstrap_persistent_reconstruction": state.persistent_reconstruction,
        "bootstrap_view_consistency": state.view_consistency,
        "bootstrap_view_consistency_weight": state.view_consistency_weight,
        "bootstrap_generator_variance_guard": state.generator_variance_guard,
        "bootstrap_generator_variance_retention": (
            state.generator_variance_retention
        ),
    }


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
    "BootstrapError",
    "MultiViewBootstrapApplication",
    "MultiViewBootstrapContext",
    "MultiViewBootstrapRuntime",
    "MultiViewReadouts",
    "apply_multiview_bootstrap",
    "bootstrap_metrics",
    "calibrate_view_consistency_weight",
]
