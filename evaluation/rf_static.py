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
    observed_counts: torch.Tensor | None = None,
    epsilon: float = 1e-3,
    finite_difference_tolerance: float | None = 0.05,
) -> StaticRFResult:
    if sequence.shape[0] != 1 or sequence.shape[1] < lag_steps:
        raise StaticRFError(
            "Static RF sequence must be [1,time,cone] with enough lags"
        )
    if observed_counts is not None and (
        observed_counts.ndim != 3
        or observed_counts.shape[:2] != sequence.shape[:2]
        or observed_counts.shape[2] != model.rgc.support_mask.shape[0]
    ):
        raise StaticRFError(
            "Static RF observed counts must have shape [1,time,cell]"
        )
    stimulus = sequence.detach().clone().requires_grad_(True)
    if observed_counts is None:
        output, _ = model.forward_sequence(stimulus)
    else:
        output, _ = model.forward_sequence(
            stimulus,
            observed_counts=observed_counts.to(stimulus.device),
        )
    kernels = []
    for cell in range(output.spike_logits.shape[-1]):
        gradient = torch.autograd.grad(
            output.spike_logits[0, -1, cell],
            stimulus,
            retain_graph=True,
        )[0]
        kernels.append(gradient[0, -lag_steps:])
    stacked = torch.stack(kernels)
    relative_error = (
        0.0
        if finite_difference_tolerance is None
        else _finite_difference_check(
            model,
            sequence,
            stacked,
            epsilon,
            observed_counts,
        )
    )
    identifiable = bool(
        torch.isfinite(stacked).all()
        and stacked.norm() > 1e-8
        and (
            finite_difference_tolerance is None
            or relative_error <= finite_difference_tolerance
        )
    )
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
    observed_counts: torch.Tensor | None,
) -> float:
    lag_index = sequence.shape[1] - 1
    errors = []
    for cell in range(kernel.shape[0]):
        supported = kernel[cell, -1].abs().masked_fill(
            ~model.rgc.support_mask[cell],
            -1.0,
        )
        cone = int(supported.argmax())
        plus = sequence.detach().clone()
        minus = sequence.detach().clone()
        plus[0, lag_index, cone] += epsilon
        minus[0, lag_index, cone] -= epsilon
        with torch.no_grad():
            if observed_counts is None:
                plus_output, _ = model.forward_sequence(plus)
                minus_output, _ = model.forward_sequence(minus)
            else:
                history = observed_counts.to(sequence.device)
                plus_output, _ = model.forward_sequence(
                    plus,
                    observed_counts=history,
                )
                minus_output, _ = model.forward_sequence(
                    minus,
                    observed_counts=history,
                )
        finite = (
            plus_output.spike_logits[0, -1, cell]
            - minus_output.spike_logits[0, -1, cell]
        ) / (2 * epsilon)
        automatic = kernel[cell, -1, cone]
        errors.append(
            (finite - automatic).abs()
            / torch.maximum(finite.abs(), automatic.abs()).clamp_min(1e-8)
        )
    return float(torch.stack(errors).max())


__all__ = [
    "StaticRFError",
    "StaticRFResult",
    "compare_rf_kernels",
    "extract_static_rf",
]
