from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch

from data.rgc_response import ResponseTargetKind
from loss.rgc_response import expected_response, response_nll, response_nll_elements


class ResponseMetricError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponseMetrics:
    nll: float
    micro_bits_per_spike: float
    macro_bits_per_spike: float
    psth_correlation: float
    explained_variance: float
    per_cell_nll: tuple[float, ...]


def compute_response_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    target_kind: ResponseTargetKind,
    baseline_rates: torch.Tensor,
) -> ResponseMetrics:
    if logits.ndim != 4:
        raise ResponseMetricError(
            "Response metrics require [stimulus,trial,time,cell] tensors"
        )
    prediction = expected_response(logits, target_kind)
    mask = valid_mask.to(logits.dtype)
    per_cell = _per_cell_nll(logits, targets, mask, target_kind)
    nll = response_nll(logits, targets, valid_mask, target_kind)
    baseline_logits = _baseline_logits(baseline_rates, target_kind)
    expanded_baseline = baseline_logits.view(1, 1, 1, -1).expand_as(logits)
    model_elements = response_nll_elements(logits, targets, target_kind)
    baseline_elements = response_nll_elements(
        expanded_baseline,
        targets,
        target_kind,
    )
    improvement = (baseline_elements - model_elements) * mask
    log_two = torch.log(torch.tensor(2.0, device=logits.device))
    spike_count = (targets * mask).sum()
    micro_bits = improvement.sum() / spike_count.clamp_min(1) / log_two
    cell_spikes = (targets * mask).sum(dim=(0, 1, 2))
    cell_bits = improvement.sum(dim=(0, 1, 2)) / cell_spikes.clamp_min(1) / log_two
    active = cell_spikes > 0
    macro_bits = (
        cell_bits[active].mean()
        if active.any()
        else torch.zeros((), device=logits.device)
    )
    psth_target, psth_mask = _trial_mean(targets, mask)
    psth_prediction, _ = _trial_mean(prediction, mask)
    correlation = _macro_correlation(psth_prediction, psth_target, psth_mask)
    residual = ((psth_prediction - psth_target).square() * psth_mask).sum()
    temporal_mean = (psth_target * psth_mask).sum(dim=1, keepdim=True)
    temporal_mean /= psth_mask.sum(dim=1, keepdim=True).clamp_min(1)
    centered = psth_target - temporal_mean
    total = (centered.square() * psth_mask).sum().clamp_min(1e-12)
    return ResponseMetrics(
        nll=float(nll.detach()),
        micro_bits_per_spike=float(micro_bits.detach()),
        macro_bits_per_spike=float(macro_bits.detach()),
        psth_correlation=float(correlation.detach()),
        explained_variance=float((1 - residual / total).detach()),
        per_cell_nll=tuple(float(value) for value in per_cell.detach().cpu()),
    )


def training_baseline_rates(
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.to(targets.dtype)
    dims = tuple(range(targets.ndim - 1))
    return (targets * mask).sum(dim=dims) / mask.sum(dim=dims).clamp_min(1)


def _per_cell_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    target_kind: ResponseTargetKind,
) -> torch.Tensor:
    values = []
    for cell in range(logits.shape[-1]):
        values.append(
            response_nll(
                logits[..., cell : cell + 1],
                targets[..., cell : cell + 1],
                mask[..., cell : cell + 1].bool(),
                target_kind,
            )
        )
    return torch.stack(values)


def _baseline_logits(
    rates: torch.Tensor,
    target_kind: ResponseTargetKind,
) -> torch.Tensor:
    match target_kind:
        case ResponseTargetKind.BERNOULLI:
            return torch.logit(rates.clamp(1e-5, 1 - 1e-5))
        case ResponseTargetKind.POISSON:
            return torch.log(torch.expm1(rates.clamp_min(1e-5)))
        case _ as unreachable:
            assert_never(unreachable)


def _trial_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dim=1)
    return (values * mask).sum(dim=1) / count.clamp_min(1), (count > 0).to(values.dtype)


def _macro_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    correlations = []
    for stimulus in range(left.shape[0]):
        for cell in range(left.shape[-1]):
            valid = mask[stimulus, :, cell].bool()
            left_values = left[stimulus, valid, cell]
            right_values = right[stimulus, valid, cell]
            left_centered = left_values - left_values.mean()
            right_centered = right_values - right_values.mean()
            denominator = (
                left_centered.square().sum().sqrt()
                * right_centered.square().sum().sqrt()
            )
            if denominator > 1e-12:
                correlations.append(
                    (left_centered * right_centered).sum() / denominator
                )
    if not correlations:
        return torch.zeros((), device=left.device)
    return torch.stack(correlations).mean()


__all__ = [
    "ResponseMetricError",
    "ResponseMetrics",
    "compute_response_metrics",
    "training_baseline_rates",
]
