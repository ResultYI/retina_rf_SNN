from __future__ import annotations

import torch
from torch.nn import functional as F

from evaluation.rf_static import extract_static_rf
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


def kernel_metrics(low: torch.Tensor, high: torch.Tensor) -> tuple[float, float]:
    shape = 1 - F.cosine_similarity(
        low.flatten(1),
        high.flatten(1),
        dim=1,
    ).mean()
    gain = (
        (high.norm(dim=(1, 2)) + 1e-8).log()
        - (low.norm(dim=(1, 2)) + 1e-8).log()
    ).abs().mean()
    return float(shape), float(gain)


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


def reset_distance(
    model: ResponseRetinaModel,
    pair: tuple[torch.Tensor, torch.Tensor],
    lag_steps: int,
) -> float:
    low_probe = pair[0][:, -lag_steps:]
    high_probe = pair[1][:, -lag_steps:]
    low_rf = extract_static_rf(
        model,
        low_probe,
        lag_steps=lag_steps,
        finite_difference_tolerance=None,
    )
    high_rf = extract_static_rf(
        model,
        high_probe,
        lag_steps=lag_steps,
        finite_difference_tolerance=None,
    )
    return kernel_metrics(low_rf.kernels, high_rf.kernels)[0]


def recovery_distance(
    model: ResponseRetinaModel,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    lag_steps: int,
    delay_ms: int,
    dt_ms: float,
) -> float:
    delay_steps = max(0, round(delay_ms / dt_ms))
    distances = []
    for low, high in pairs:
        if delay_steps:
            recovery = torch.zeros(
                low.shape[0],
                delay_steps,
                low.shape[2],
                device=low.device,
                dtype=low.dtype,
            )
            low = torch.cat((low[:, :-lag_steps], recovery, low[:, -lag_steps:]), dim=1)
            high = torch.cat(
                (high[:, :-lag_steps], recovery, high[:, -lag_steps:]),
                dim=1,
            )
        low_rf = extract_static_rf(
            model,
            low,
            lag_steps=lag_steps,
            finite_difference_tolerance=None,
        )
        high_rf = extract_static_rf(
            model,
            high,
            lag_steps=lag_steps,
            finite_difference_tolerance=None,
        )
        distances.append(kernel_metrics(low_rf.kernels, high_rf.kernels)[0])
    return sum(distances) / len(distances)


__all__ = [
    "bootstrap_ci",
    "context_pairs",
    "kernel_metrics",
    "recovery_distance",
    "reset_distance",
    "teacher_errors",
]
