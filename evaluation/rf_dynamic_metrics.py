from __future__ import annotations

from typing import assert_never

import torch
from torch.nn import functional as F

from evaluation.rf_dynamic_conditioning import conditioned_rf
from evaluation.rf_dynamic_result import DynamicRFError
from evaluation.rf_history_contracts import RFHistoryContract
from evaluation.rf_static import StaticRFResult
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit, masked_history_counts


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


def context_pairs(
    split: ResponseSplit,
    *,
    require_complete: bool = False,
) -> tuple[tuple[int, int], ...]:
    by_source: dict[str, dict[str, int]] = {}
    for index, (source, context) in enumerate(
        zip(split.source_ids, split.context_ids, strict=True)
    ):
        by_source.setdefault(source, {})[context] = index
    if require_complete:
        missing = tuple(
            source
            for source, contexts in by_source.items()
            if "low" not in contexts or "high" not in contexts
        )
        if missing:
            raise DynamicRFError(
                "Dynamic RF history contracts require complete low/high context pairs"
            )
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
    history_mode: RFHistoryContract | None = None,
    matched_history_index: int | None = None,
    standard_history_counts: torch.Tensor | None = None,
    finite_difference_tolerance: float | None = 0.05,
) -> StaticRFResult:
    device = next(model.parameters()).device
    _require_trial_index(split, index)
    sequence = split.cone_response[index : index + 1].to(device)
    if not condition_on_observed:
        return conditioned_rf(
            model,
            sequence,
            split.spike_counts[index].to(device),
            split.valid_mask[index].to(device),
            lag_steps,
            condition_on_observed=False,
            finite_difference_tolerance=finite_difference_tolerance,
        )
    if history_mode is not None:
        return conditioned_rf(
            model,
            sequence,
            _history_counts(
                split,
                index,
                history_mode,
                matched_history_index,
                standard_history_counts,
            ).to(device),
            torch.ones_like(split.valid_mask[index : index + 1, 0].to(device)),
            lag_steps,
            condition_on_observed=True,
            finite_difference_tolerance=finite_difference_tolerance,
        )
    counts = split.spike_counts[index].to(device)
    mask = split.valid_mask[index].to(device)
    return conditioned_rf(
        model,
        sequence,
        counts,
        mask,
        lag_steps,
        condition_on_observed=condition_on_observed,
        finite_difference_tolerance=finite_difference_tolerance,
    )


def _history_counts(
    split: ResponseSplit,
    index: int,
    history_mode: RFHistoryContract,
    matched_history_index: int | None,
    standard_history_counts: torch.Tensor | None,
) -> torch.Tensor:
    _require_history_index(split, index)
    match history_mode:
        case "zero":
            return torch.zeros_like(split.spike_counts[index : index + 1, 0])
        case "matched_observed":
            if matched_history_index is None:
                raise DynamicRFError("matched_observed RF history needs a source index")
            _require_history_index(split, matched_history_index)
            return masked_history_counts(
                split.spike_counts[matched_history_index : matched_history_index + 1, 0],
                split.valid_mask[matched_history_index : matched_history_index + 1, 0],
            )
        case "standard_train_rate":
            if standard_history_counts is None:
                raise DynamicRFError("standard_train_rate RF history needs train rates")
            expected = split.spike_counts[index : index + 1, 0].shape
            if standard_history_counts.shape != expected:
                raise DynamicRFError(
                    "standard_train_rate RF history must match [1,time,cell]"
                )
            return standard_history_counts.to(
                device=split.spike_counts.device,
                dtype=split.spike_counts.dtype,
            )
        case unreachable:
            assert_never(unreachable)


def _require_trial_index(split: ResponseSplit, index: int) -> None:
    if index < 0 or index >= split.cone_response.shape[0]:
        raise DynamicRFError("RF history contract stimulus index is out of range")
    if index >= split.spike_counts.shape[0] or index >= split.valid_mask.shape[0]:
        raise DynamicRFError("RF history contract response index is out of range")


def _require_history_index(split: ResponseSplit, index: int) -> None:
    _require_trial_index(split, index)
    if split.spike_counts.shape[1] < 1 or split.valid_mask.shape[1] < 1:
        raise DynamicRFError("RF history contracts require at least one trial")


__all__ = [
    "bootstrap_ci",
    "context_pairs",
    "kernel_metrics",
    "signed_log_gains",
    "teacher_errors",
    "trial_conditioned_rf",
]
