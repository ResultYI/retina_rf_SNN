from __future__ import annotations

import torch

from evaluation.rf_static import StaticRFResult, extract_static_rf
from models.response_snn import ResponseRetinaModel
from training.response_data import masked_history_counts


def conditioned_rf(
    model: ResponseRetinaModel,
    sequence: torch.Tensor,
    counts: torch.Tensor,
    mask: torch.Tensor,
    lag_steps: int,
    *,
    condition_on_observed: bool,
    finite_difference_tolerance: float | None = 0.05,
) -> StaticRFResult:
    if not condition_on_observed:
        return extract_static_rf(
            model,
            sequence,
            lag_steps=lag_steps,
            finite_difference_tolerance=finite_difference_tolerance,
        )
    kernels: list[torch.Tensor] = []
    errors: list[float] = []
    identifiable = True
    for trial in range(counts.shape[0]):
        trial_mask = mask[trial : trial + 1]
        if not bool(trial_mask.any()):
            continue
        rf = extract_static_rf(
            model,
            sequence,
            lag_steps=lag_steps,
            observed_counts=masked_history_counts(
                counts[trial : trial + 1],
                trial_mask,
            ),
            finite_difference_tolerance=finite_difference_tolerance,
        )
        kernels.append(rf.kernels)
        errors.append(rf.finite_difference_relative_error)
        identifiable = identifiable and rf.identifiable
    if not kernels:
        support = model.rgc.support_mask
        return StaticRFResult(
            torch.zeros(
                support.shape[0],
                lag_steps,
                support.shape[1],
                device=sequence.device,
            ),
            float("inf"),
            False,
        )
    kernel = torch.stack(kernels).mean(dim=0)
    return StaticRFResult(
        kernel,
        max(errors),
        identifiable and bool(torch.isfinite(kernel).all() and kernel.norm() > 1e-8),
    )


__all__ = ["conditioned_rf"]
