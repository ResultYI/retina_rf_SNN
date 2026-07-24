from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch

from data.rgc_response import ResponseTargetKind
from loss.rgc_response import expected_response, response_nll


@dataclass(frozen=True, slots=True)
class ResponseMetrics:
    nll: float
    bits_per_spike: float
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
    prediction = expected_response(logits, target_kind)
    mask = valid_mask.to(logits.dtype)
    per_cell = _per_cell_nll(logits, targets, mask, target_kind)
    nll = response_nll(logits, targets, valid_mask, target_kind)
    baseline_logits = _baseline_logits(baseline_rates, target_kind)
    baseline = response_nll(
        baseline_logits.view(1, 1, -1).expand_as(logits),
        targets,
        valid_mask,
        target_kind,
    )
    spike_count = (targets * mask).sum().clamp_min(1)
    bits_per_spike = (baseline - nll) * mask.sum() / spike_count / torch.log(
        torch.tensor(2.0, device=logits.device)
    )
    psth_target = _masked_time_mean(targets, mask)
    psth_prediction = _masked_time_mean(prediction, mask)
    correlation = _correlation(psth_prediction, psth_target)
    residual = ((prediction - targets).square() * mask).sum()
    centered = targets - (targets * mask).sum() / mask.sum().clamp_min(1)
    total = (centered.square() * mask).sum().clamp_min(1e-12)
    return ResponseMetrics(
        nll=float(nll.detach()),
        bits_per_spike=float(bits_per_spike.detach()),
        psth_correlation=float(correlation.detach()),
        explained_variance=float((1 - residual / total).detach()),
        per_cell_nll=tuple(float(value) for value in per_cell.detach().cpu()),
    )


def training_baseline_rates(
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    mask = valid_mask.to(targets.dtype)
    return (targets * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1)


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


def _masked_time_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten() - left.mean()
    right = right.flatten() - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return (left * right).sum() / denominator.clamp_min(1e-12)


__all__ = [
    "ResponseMetrics",
    "compute_response_metrics",
    "training_baseline_rates",
]
