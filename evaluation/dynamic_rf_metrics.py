from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from models.retina_snn import RetinaModel, RetinaState


@dataclass(frozen=True, slots=True)
class FiniteDifferenceResult:
    autodiff_directional: float
    finite_difference_directional: float
    relative_error: float | None
    status: str


def continuous_kernel(
    model: RetinaModel,
    probe: torch.Tensor,
    state: RetinaState,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
) -> torch.Tensor:
    differentiable_probe = probe.detach().clone().requires_grad_(True)
    output, _ = model.forward_sequence(
        differentiable_probe,
        state,
        spatial_weights=spatial_weights,
        probe_continuous_output=True,
    )
    continuous = getattr(output, readout)[0, -1, polarity, unit]
    return torch.autograd.grad(continuous, differentiable_probe)[0][0].detach()


def finite_difference_check(
    model: RetinaModel,
    probe: torch.Tensor,
    state: RetinaState,
    spatial_weights: torch.Tensor,
    polarity: int,
    unit: int,
    readout: str,
    kernel: torch.Tensor,
    epsilons: Sequence[float],
) -> FiniteDifferenceResult:
    direction = local_probe_direction(model, probe, polarity=polarity, unit=unit)
    autodiff = float((kernel * direction[0]).sum())
    last_finite_difference = 0.0
    with torch.no_grad():
        for epsilon in sorted(epsilons, reverse=True):
            outputs = []
            events = []
            for signed_epsilon in (epsilon, -epsilon):
                output, _ = model.forward_sequence(
                    probe + signed_epsilon * direction,
                    state,
                    spatial_weights=spatial_weights,
                    probe_continuous_output=True,
                )
                outputs.append(getattr(output, readout)[0, -1, polarity, unit])
                events.append(output.hard_spikes[0, :, polarity, unit])
            last_finite_difference = float(
                (outputs[0] - outputs[1]) / (2.0 * epsilon)
            )
            if torch.equal(events[0], events[1]):
                relative_error = abs(autodiff - last_finite_difference) / max(
                    abs(autodiff), abs(last_finite_difference), 1e-12
                )
                return FiniteDifferenceResult(
                    autodiff,
                    last_finite_difference,
                    relative_error,
                    "local_continuous_check",
                )
    return FiniteDifferenceResult(
        autodiff,
        last_finite_difference,
        None,
        "threshold_crossing_not_local",
    )


def local_probe_direction(
    model: RetinaModel,
    probe: torch.Tensor,
    *,
    polarity: int,
    unit: int,
) -> torch.Tensor:
    generator = torch.Generator(device=probe.device).manual_seed(
        polarity * model.rgc.unit_count + unit
    )
    support = model.rgc.support_mask[unit].to(device=probe.device)
    direction = torch.zeros_like(probe)
    local_shape = (*probe.shape[:-1], int(support.sum()))
    direction[..., support] = torch.randn(
        local_shape,
        generator=generator,
        device=probe.device,
        dtype=probe.dtype,
    )
    return direction / direction.norm().clamp_min(1e-12)


def temporal_metrics(kernel: torch.Tensor, dt_ms: float) -> tuple[float, float]:
    temporal = kernel.square().sum(dim=1).sqrt()
    peak_index = int(temporal.argmax())
    peak = temporal[peak_index].clamp_min(1e-12)
    width = int((temporal >= 0.5 * peak).sum())
    return (kernel.shape[0] - 1 - peak_index) * dt_ms, width * dt_ms


def spatial_metrics(
    kernel: torch.Tensor,
    cone_positions: torch.Tensor,
    unit_center: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    spatial = kernel.abs().sum(dim=0)
    normalized = spatial / spatial.sum().clamp_min(1e-12)
    center = (normalized[:, None] * cone_positions).sum(dim=0)
    second_moment = (
        normalized * (cone_positions - unit_center).square().sum(dim=1)
    ).sum()
    return center, float((center - unit_center).norm()), float(second_moment)


__all__ = [
    "FiniteDifferenceResult",
    "continuous_kernel",
    "finite_difference_check",
    "local_probe_direction",
    "spatial_metrics",
    "temporal_metrics",
]
