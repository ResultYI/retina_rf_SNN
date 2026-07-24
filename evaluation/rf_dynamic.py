from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from evaluation.rf_static import extract_static_rf
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


class DynamicRFError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DynamicRFResult:
    pair_count: int
    mean_shape_distance: float
    mean_log_gain_shift: float
    reset_shape_distance: float
    status: str


def evaluate_dynamic_rf(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    *,
    lag_steps: int,
) -> DynamicRFResult:
    pairs = _context_pairs(split)
    shape_distances: list[torch.Tensor] = []
    gain_shifts: list[torch.Tensor] = []
    for low_index, high_index in pairs:
        low = split.cone_response[low_index : low_index + 1].to(
            next(model.parameters()).device
        )
        high = split.cone_response[high_index : high_index + 1].to(
            next(model.parameters()).device
        )
        if not torch.equal(low[:, -lag_steps:], high[:, -lag_steps:]):
            raise DynamicRFError(
                "Dynamic RF context pairs need an identical final probe"
            )
        low_kernel = extract_static_rf(model, low, lag_steps=lag_steps).kernels
        high_kernel = extract_static_rf(model, high, lag_steps=lag_steps).kernels
        shape_distances.append(
            1
            - F.cosine_similarity(
                low_kernel.flatten(1),
                high_kernel.flatten(1),
                dim=1,
            ).mean()
        )
        gain_shifts.append(
            (
                (high_kernel.norm(dim=(1, 2)) + 1e-8).log()
                - (low_kernel.norm(dim=(1, 2)) + 1e-8).log()
            ).abs().mean()
        )
    if not pairs:
        return DynamicRFResult(0, 0.0, 0.0, 0.0, "not_identifiable")
    shape = float(torch.stack(shape_distances).mean())
    gain = float(torch.stack(gain_shifts).mean())
    status = classify_dynamic_rf(len(pairs), shape, gain)
    return DynamicRFResult(len(pairs), shape, gain, 0.0, status)


def classify_dynamic_rf(
    pair_count: int,
    shape_distance: float,
    log_gain_shift: float,
) -> str:
    if pair_count < 3:
        return "not_identifiable"
    return (
        "supported"
        if shape_distance > 1e-3 or log_gain_shift > 1e-3
        else "not_supported"
    )


def _context_pairs(split: ResponseSplit) -> tuple[tuple[int, int], ...]:
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


__all__ = [
    "DynamicRFError",
    "DynamicRFResult",
    "classify_dynamic_rf",
    "evaluate_dynamic_rf",
]
