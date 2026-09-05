from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CellPerturbationRequest:
    normal_logits: torch.Tensor
    clamped_logits: torch.Tensor
    normal_probability: torch.Tensor
    clamped_probability: torch.Tensor
    normal_temporal_rf: torch.Tensor
    clamped_temporal_rf: torch.Tensor
    baseline_steps: int
    dt_ms: float


@dataclass(frozen=True, slots=True)
class CellPerturbationMetrics:
    mean_absolute_logit_change: torch.Tensor
    mean_logit_change: torch.Tensor
    mean_absolute_probability_change: torch.Tensor
    mean_probability_change: torch.Tensor
    normal_logit_peak_magnitude: torch.Tensor
    clamped_logit_peak_magnitude: torch.Tensor
    logit_peak_magnitude_change: torch.Tensor
    normal_logit_peak_latency_ms: torch.Tensor
    clamped_logit_peak_latency_ms: torch.Tensor
    logit_peak_latency_change_ms: torch.Tensor
    logit_peak_latency_absolute_shift_ms: torch.Tensor
    normal_probability_peak_magnitude: torch.Tensor
    clamped_probability_peak_magnitude: torch.Tensor
    probability_peak_magnitude_change: torch.Tensor
    normal_probability_peak_latency_ms: torch.Tensor
    clamped_probability_peak_latency_ms: torch.Tensor
    probability_peak_latency_change_ms: torch.Tensor
    probability_peak_latency_absolute_shift_ms: torch.Tensor
    temporal_rf_normal_norm: torch.Tensor
    temporal_rf_clamped_norm: torch.Tensor
    temporal_rf_norm_change: torch.Tensor
    temporal_rf_change_norm: torch.Tensor
    temporal_rf_cosine: torch.Tensor


@dataclass(frozen=True, slots=True)
class CellPerturbationError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def cell_perturbation_metrics(
    request: CellPerturbationRequest,
) -> CellPerturbationMetrics:
    response_shape = request.normal_logits.shape
    if not (
        len(response_shape) == 3
        and request.clamped_logits.shape == response_shape
        and request.normal_probability.shape == response_shape
        and request.clamped_probability.shape == response_shape
        and request.normal_temporal_rf.shape == request.clamped_temporal_rf.shape
        and request.normal_temporal_rf.shape[0] == response_shape[-1]
        and 0 < request.baseline_steps < response_shape[1]
        and request.dt_ms > 0
    ):
        raise CellPerturbationError("per-cell perturbation inputs are invalid")
    logit_delta = request.clamped_logits - request.normal_logits
    probability_delta = request.clamped_probability - request.normal_probability
    normal_logit_peak = _cell_peaks(
        request.normal_logits, request.baseline_steps, request.dt_ms
    )
    clamped_logit_peak = _cell_peaks(
        request.clamped_logits, request.baseline_steps, request.dt_ms
    )
    normal_probability_peak = _cell_peaks(
        request.normal_probability, request.baseline_steps, request.dt_ms
    )
    clamped_probability_peak = _cell_peaks(
        request.clamped_probability, request.baseline_steps, request.dt_ms
    )
    normal_rf_norm = torch.linalg.vector_norm(request.normal_temporal_rf, dim=1)
    clamped_rf_norm = torch.linalg.vector_norm(request.clamped_temporal_rf, dim=1)
    rf_delta = request.clamped_temporal_rf - request.normal_temporal_rf
    denominator = normal_rf_norm * clamped_rf_norm
    rf_cosine = torch.where(
        denominator > 0,
        (request.normal_temporal_rf * request.clamped_temporal_rf).sum(dim=1)
        / denominator,
        torch.where(normal_rf_norm == clamped_rf_norm, 1.0, 0.0),
    )
    return CellPerturbationMetrics(
        logit_delta.abs().mean(dim=(0, 1)),
        logit_delta.mean(dim=(0, 1)),
        probability_delta.abs().mean(dim=(0, 1)),
        probability_delta.mean(dim=(0, 1)),
        normal_logit_peak[0],
        clamped_logit_peak[0],
        clamped_logit_peak[0] - normal_logit_peak[0],
        normal_logit_peak[1],
        clamped_logit_peak[1],
        clamped_logit_peak[1] - normal_logit_peak[1],
        (clamped_logit_peak[2] - normal_logit_peak[2]).abs().float().mean(dim=0)
        * request.dt_ms,
        normal_probability_peak[0],
        clamped_probability_peak[0],
        clamped_probability_peak[0] - normal_probability_peak[0],
        normal_probability_peak[1],
        clamped_probability_peak[1],
        clamped_probability_peak[1] - normal_probability_peak[1],
        (clamped_probability_peak[2] - normal_probability_peak[2])
        .abs()
        .float()
        .mean(dim=0)
        * request.dt_ms,
        normal_rf_norm,
        clamped_rf_norm,
        clamped_rf_norm - normal_rf_norm,
        torch.linalg.vector_norm(rf_delta, dim=1),
        rf_cosine,
    )


def _cell_peaks(
    response: torch.Tensor,
    baseline_steps: int,
    dt_ms: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline = response[:, :baseline_steps].mean(dim=1, keepdim=True)
    deviation = (response[:, baseline_steps:] - baseline).abs()
    peak, peak_index = deviation.max(dim=1)
    return peak.mean(dim=0), peak_index.float().mean(dim=0) * dt_ms, peak_index


__all__ = [
    "CellPerturbationError",
    "CellPerturbationMetrics",
    "CellPerturbationRequest",
    "cell_perturbation_metrics",
]
