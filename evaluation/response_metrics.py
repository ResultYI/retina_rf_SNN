from __future__ import annotations

from dataclasses import dataclass
from typing import Final, assert_never

import torch

from data.rgc_response import ResponseTargetKind
from loss.rgc_response import expected_response, response_nll, response_nll_elements

CALIBRATION_BIN_COUNT: Final = 10


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
    constant_rate_nll: float = 0.0
    calibration_error: float | None = None


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
    if targets.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ResponseMetricError("logits, targets, and valid_mask must share a shape")
    if baseline_rates.shape != (logits.shape[-1],):
        raise ResponseMetricError("baseline_rates must have one value per cell")
    if not torch.isfinite(baseline_rates).all():
        raise ResponseMetricError("baseline_rates must be finite")
    match target_kind:
        case ResponseTargetKind.BERNOULLI:
            if not torch.all((baseline_rates >= 0) & (baseline_rates <= 1)):
                raise ResponseMetricError(
                    "Bernoulli baseline_rates must be probabilities in [0, 1]"
                )
        case ResponseTargetKind.POISSON:
            if not torch.all(baseline_rates >= 0):
                raise ResponseMetricError("Poisson baseline_rates must be non-negative")
        case _ as unreachable:
            assert_never(unreachable)
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ResponseMetricError("Response metric tensors must be finite")
    mask_bool = valid_mask.bool()
    if not torch.any(mask_bool):
        match target_kind:
            case ResponseTargetKind.BERNOULLI:
                empty_calibration_error = 0.0
            case ResponseTargetKind.POISSON:
                empty_calibration_error = None
            case _ as unreachable:
                assert_never(unreachable)
        return ResponseMetrics(
            nll=0.0,
            micro_bits_per_spike=0.0,
            macro_bits_per_spike=0.0,
            psth_correlation=0.0,
            explained_variance=0.0,
            per_cell_nll=(0.0,) * logits.shape[-1],
            constant_rate_nll=0.0,
            calibration_error=empty_calibration_error,
        )
    if not torch.all(mask_bool.sum(dim=(0, 1, 2)) > 0):
        raise ResponseMetricError("Every cell needs at least one valid target")
    prediction = expected_response(logits, target_kind)
    mask = mask_bool.to(logits.dtype)
    per_cell = _per_cell_nll(logits, targets, mask, target_kind)
    nll = response_nll(logits, targets, mask_bool, target_kind)
    baseline_logits = _baseline_logits(baseline_rates, target_kind)
    expanded_baseline = baseline_logits.view(1, 1, 1, -1).expand_as(logits)
    model_elements = response_nll_elements(logits, targets, target_kind)
    baseline_elements = response_nll_elements(
        expanded_baseline,
        targets,
        target_kind,
    )
    constant_rate_nll = (baseline_elements * mask).sum() / mask.sum().clamp_min(1)
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
    match target_kind:
        case ResponseTargetKind.BERNOULLI:
            calibration_error = _calibration_error(
                torch.sigmoid(logits),
                targets,
                mask_bool,
            )
        case ResponseTargetKind.POISSON:
            calibration_error = None
        case _ as unreachable:
            assert_never(unreachable)
    return ResponseMetrics(
        nll=float(nll.detach()),
        micro_bits_per_spike=float(micro_bits.detach()),
        macro_bits_per_spike=float(macro_bits.detach()),
        psth_correlation=float(correlation.detach()),
        explained_variance=float((1 - residual / total).detach()),
        per_cell_nll=tuple(float(value) for value in per_cell.detach().cpu()),
        constant_rate_nll=float(constant_rate_nll.detach()),
        calibration_error=None
        if calibration_error is None
        else float(calibration_error.detach()),
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


def _calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid_probabilities = probabilities[valid_mask]
    observations = targets[valid_mask]
    error = torch.zeros((), device=probabilities.device)
    for bin_index in range(CALIBRATION_BIN_COUNT):
        lower = bin_index / CALIBRATION_BIN_COUNT
        upper = (bin_index + 1) / CALIBRATION_BIN_COUNT
        if bin_index == CALIBRATION_BIN_COUNT - 1:
            in_bin = (valid_probabilities >= lower) & (valid_probabilities <= upper)
        else:
            in_bin = (valid_probabilities >= lower) & (valid_probabilities < upper)
        if in_bin.any():
            weight = in_bin.to(probabilities.dtype).mean()
            error += weight * (
                valid_probabilities[in_bin].mean() - observations[in_bin].mean()
            ).abs()
    return error.clamp(0.0, 1.0)


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
