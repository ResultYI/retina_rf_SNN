from __future__ import annotations

from dataclasses import dataclass

import torch

from evaluation.rf_dynamic_conditioning import conditioned_rf
from evaluation.rf_dynamic_metrics import kernel_metrics, signed_log_gains
from evaluation.rf_static import StaticRFResult
from models.response_snn import ResponseRetinaModel
from training.response_data import ResponseSplit


@dataclass(frozen=True, slots=True)
class RFDistance:
    shape_distance: float
    signed_gain_shifts: tuple[float, ...]

    @property
    def mean_absolute_gain_shift(self) -> float:
        if not self.signed_gain_shifts:
            return 0.0
        return sum(abs(value) for value in self.signed_gain_shifts) / len(
            self.signed_gain_shifts
        )


def reset_distance(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    pair: tuple[int, int],
    lag_steps: int,
    *,
    condition_on_observed: bool,
) -> RFDistance:
    low_rf = _reset_rf(model, split, pair[0], lag_steps, condition_on_observed)
    high_rf = _reset_rf(model, split, pair[1], lag_steps, condition_on_observed)
    return rf_distance(low_rf, high_rf)


def _reset_rf(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    index: int,
    lag_steps: int,
    condition_on_observed: bool,
) -> StaticRFResult:
    device = next(model.parameters()).device
    return conditioned_rf(
        model,
        split.cone_response[index : index + 1, -lag_steps:].to(device),
        split.spike_counts[index, :, -lag_steps:].to(device),
        split.valid_mask[index, :, -lag_steps:].to(device),
        lag_steps,
        condition_on_observed=condition_on_observed,
    )


def recovery_distance(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    pairs: tuple[tuple[int, int], ...],
    lag_steps: int,
    delay_ms: int,
    dt_ms: float,
    *,
    condition_on_observed: bool,
) -> RFDistance:
    delay_steps = max(0, round(delay_ms / dt_ms))
    distances: list[RFDistance] = []
    device = next(model.parameters()).device
    for low_index, high_index in pairs:
        low = _recovery_rf(
            model,
            split,
            low_index,
            lag_steps,
            delay_steps,
            device,
            condition_on_observed,
        )
        high = _recovery_rf(
            model,
            split,
            high_index,
            lag_steps,
            delay_steps,
            device,
            condition_on_observed,
        )
        distances.append(rf_distance(low, high))
    return mean_distances(tuple(distances))


def _recovery_rf(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    index: int,
    lag_steps: int,
    delay_steps: int,
    device: torch.device,
    condition_on_observed: bool,
) -> StaticRFResult:
    sequence, counts, mask = _recovery_inputs(split, index, lag_steps, delay_steps, device)
    return conditioned_rf(
        model,
        sequence,
        counts,
        mask,
        lag_steps,
        condition_on_observed=condition_on_observed,
    )


def _recovery_inputs(
    split: ResponseSplit,
    index: int,
    lag_steps: int,
    delay_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = split.cone_response[index : index + 1].to(device)
    counts = split.spike_counts[index].to(device)
    mask = split.valid_mask[index].to(device)
    if not delay_steps:
        return sequence, counts, mask
    recovery = torch.zeros(
        1,
        delay_steps,
        sequence.shape[2],
        device=device,
        dtype=sequence.dtype,
    )
    count_recovery = torch.zeros(
        counts.shape[0],
        delay_steps,
        counts.shape[2],
        device=device,
        dtype=counts.dtype,
    )
    mask_recovery = torch.ones(
        mask.shape[0],
        delay_steps,
        mask.shape[2],
        device=device,
        dtype=mask.dtype,
    )
    return (
        torch.cat((sequence[:, :-lag_steps], recovery, sequence[:, -lag_steps:]), dim=1),
        torch.cat((counts[:, :-lag_steps], count_recovery, counts[:, -lag_steps:]), dim=1),
        torch.cat((mask[:, :-lag_steps], mask_recovery, mask[:, -lag_steps:]), dim=1),
    )


def recovery_distances_by_source(
    model: ResponseRetinaModel,
    split: ResponseSplit,
    pairs: tuple[tuple[int, int], ...],
    lag_steps: int,
    recovery_delays_ms: tuple[int, ...],
    dt_ms: float,
    *,
    condition_on_observed: bool,
) -> tuple[tuple[RFDistance, ...], ...]:
    return tuple(
        tuple(
            recovery_distance(
                model,
                split,
                (pair,),
                lag_steps,
                delay,
                dt_ms,
                condition_on_observed=condition_on_observed,
            )
            for delay in recovery_delays_ms
        )
        for pair in pairs
    )


def rf_distance(low: StaticRFResult, high: StaticRFResult) -> RFDistance:
    return RFDistance(
        kernel_metrics(low.kernels, high.kernels)[0],
        tuple(
            float(value)
            for value in signed_log_gains(low.kernels, high.kernels).detach().cpu()
        ),
    )


def mean_distances(distances: tuple[RFDistance, ...]) -> RFDistance:
    if not distances:
        return RFDistance(0.0, ())
    cell_count = len(distances[0].signed_gain_shifts)
    return RFDistance(
        sum(value.shape_distance for value in distances) / len(distances),
        tuple(
            sum(value.signed_gain_shifts[cell] for value in distances)
            / len(distances)
            for cell in range(cell_count)
        ),
    )


__all__ = [
    "RFDistance",
    "mean_distances",
    "recovery_distance",
    "recovery_distances_by_source",
    "reset_distance",
]
