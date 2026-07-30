from __future__ import annotations

import torch
from torch.nn import functional as F

from evaluation.rf_dynamic_conditioning import conditioned_rf
from evaluation.rf_static import StaticRFResult
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


def kernel_metrics(low: torch.Tensor, high: torch.Tensor) -> tuple[float, float]:
    shape = 1 - F.cosine_similarity(
        low.flatten(1),
        high.flatten(1),
        dim=1,
    ).mean()
    gain = signed_log_gains(low, high).abs().mean()
    return float(shape), float(gain)


def signed_log_gains(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (
        (high.norm(dim=(1, 2)) + 1e-8).log()
        - (low.norm(dim=(1, 2)) + 1e-8).log()
    )


def bootstrap_ci(
    values: list[float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        len(values),
        (max(1, iterations), len(values)),
        generator=generator,
    )
    means = tensor[indices].mean(dim=1)
    quantiles = torch.quantile(
        means,
        torch.tensor([0.025, 0.975], dtype=means.dtype),
    )
    return float(quantiles[0]), float(quantiles[1])


def teacher_errors(
    shapes: list[float],
    gains: list[float],
    kernels: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[float | None, float | None]:
    if kernels is None:
        return None, None
    teacher_shape, teacher_gain = kernel_metrics(*kernels)
    return (
        sum(abs(value - teacher_shape) for value in shapes) / len(shapes),
        sum(abs(value - teacher_gain) for value in gains) / len(gains),
    )


def context_pairs(split: ResponseSplit) -> tuple[tuple[int, int], ...]:
    by_source: dict[str, dict[str, int]] = {}
    for index, (source, context) in enumerate(
        zip(split.source_ids, split.context_ids, strict=True)
    ):
        by_source.setdefault(source, {})[context] = index
    return tuple(
        (contexts["low"], contexts["high"])
        for contexts in by_source.values()
        if "low" in contexts and "high" in contexts
    )


def trial_conditioned_rf(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    index: int,
    lag_steps: int,
    *,
    condition_on_observed: bool = True,
) -> StaticRFResult:
    device = next(model.parameters()).device
    sequence = split.cone_response[index : index + 1].to(device)
    counts = split.spike_counts[index].to(device)
    mask = split.valid_mask[index].to(device)
    return conditioned_rf(
        model,
        sequence,
        counts,
        mask,
        lag_steps,
        condition_on_observed=condition_on_observed,
    )


__all__ = [
    "bootstrap_ci",
    "context_pairs",
    "kernel_metrics",
    "signed_log_gains",
    "teacher_errors",
    "trial_conditioned_rf",
]
