from __future__ import annotations

from dataclasses import dataclass

import torch

from models.response_snn import ResponseRetinaModel


class StaticRFError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StaticRFResult:
    kernels: torch.Tensor
    finite_difference_relative_error: float
    identifiable: bool


def extract_static_rf(
    model: ResponseRetinaModel,
    sequence: torch.Tensor,
    *,
    lag_steps: int,
    epsilon: float = 1e-3,
) -> StaticRFResult:
    if sequence.shape[0] != 1 or sequence.shape[1] < lag_steps:
        raise StaticRFError(
            "Static RF sequence must be [1,time,cone] with enough lags"
        )
    stimulus = sequence.detach().clone().requires_grad_(True)
    output, _ = model.forward_sequence(stimulus)
    kernels = []
    for cell in range(output.spike_logits.shape[-1]):
        gradient = torch.autograd.grad(
            output.spike_logits[0, -1, cell],
            stimulus,
            retain_graph=True,
        )[0]
        kernels.append(gradient[0, -lag_steps:])
    stacked = torch.stack(kernels)
    relative_error = _finite_difference_check(
        model,
        sequence,
        stacked,
        epsilon,
    )
    identifiable = bool(torch.isfinite(stacked).all() and stacked.norm() > 1e-8)
    return StaticRFResult(stacked.detach(), relative_error, identifiable)


def compare_rf_kernels(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    lag = min(predicted.shape[1], reference.shape[1])
    cone = min(predicted.shape[2], reference.shape[2])
    left = predicted[:, -lag:, :cone].flatten(1)
    right = torch.flip(reference[:, :lag, :cone], dims=(1,)).flatten(1)
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=1)
    return {
        "mean_kernel_correlation": float(cosine.mean()),
        "mean_kernel_norm": float(left.norm(dim=1).mean()),
    }


def _finite_difference_check(
    model: ResponseRetinaModel,
    sequence: torch.Tensor,
    kernel: torch.Tensor,
    epsilon: float,
) -> float:
    cone = int(model.rgc.support_mask[0].nonzero()[0])
    lag_index = sequence.shape[1] - 1
    plus = sequence.detach().clone()
    minus = sequence.detach().clone()
    plus[0, lag_index, cone] += epsilon
    minus[0, lag_index, cone] -= epsilon
    with torch.no_grad():
        plus_output, _ = model.forward_sequence(plus)
        minus_output, _ = model.forward_sequence(minus)
    finite = (
        plus_output.spike_logits[0, -1, 0]
        - minus_output.spike_logits[0, -1, 0]
    ) / (2 * epsilon)
    automatic = kernel[0, -1, cone]
    return float(
        (finite - automatic).abs()
        / torch.maximum(finite.abs(), automatic.abs()).clamp_min(1e-8)
    )


__all__ = [
    "StaticRFError",
    "StaticRFResult",
    "compare_rf_kernels",
    "extract_static_rf",
]
