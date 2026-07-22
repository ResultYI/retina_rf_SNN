from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from models.retina_snn import RetinaModel


@dataclass(frozen=True, slots=True)
class TemporalProbeFeatures:
    preferred_polarity: torch.Tensor
    valid_response_mask: torch.Tensor
    impulse_peak: torch.Tensor
    impulse_time_to_peak_ms: torch.Tensor
    impulse_width_ms: torch.Tensor
    step_sustained_index: torch.Tensor
    flicker_response: torch.Tensor
    hard_evoked_spike_count: torch.Tensor


@torch.no_grad()
def run_temporal_probes(
    model: RetinaModel,
    baseline_cone: torch.Tensor,
    *,
    sequence_steps: int,
    onset_step: int,
    dt_ms: float,
    amplitude: float = 1.0,
    flicker_frequency_hz: float = 4.0,
    minimum_continuous_response: float = 1e-4,
) -> TemporalProbeFeatures:
    cone_count = model.rgc.cone_positions_degs.shape[0]
    if baseline_cone.ndim != 1 or baseline_cone.shape[0] != cone_count:
        raise ValueError("baseline_cone must have shape [cone]")
    if not 0 < onset_step < sequence_steps:
        raise ValueError("onset_step must lie inside the probe sequence")
    scales = (dt_ms, amplitude, flicker_frequency_hz, minimum_continuous_response)
    if not all(math.isfinite(value) and value > 0 for value in scales):
        raise ValueError("temporal probe scales must be finite and positive")

    device = next(model.parameters()).device
    baseline_cone = baseline_cone.to(device=device, dtype=torch.float32)
    centers = model.rgc.unit_center_indices
    unit_count = int(centers.numel())
    half_period_steps = max(
        1,
        round(500.0 / (flicker_frequency_hz * dt_ms)),
    )
    flicker_onset_step = min(onset_step, 2 * sequence_steps // 5)
    baseline, impulse, step, flicker = _probe_sequences(
        baseline_cone,
        centers,
        sequence_steps=sequence_steps,
        onset_step=onset_step,
        amplitude=amplitude,
        half_period_steps=half_period_steps,
    )

    was_training = model.training
    model.eval()
    try:
        baseline_output, _ = model.forward_sequence(baseline)
        impulse_output, _ = model.forward_sequence(impulse)
        step_output, _ = model.forward_sequence(step)
        flicker_output, _ = model.forward_sequence(flicker)
    finally:
        model.train(was_training)

    continuous = []
    hard_counts = []
    probe_outputs = (impulse_output, step_output, flicker_output)
    response_onsets = (onset_step, onset_step, flicker_onset_step)
    for output, response_onset in zip(probe_outputs, response_onsets, strict=True):
        continuous.append(
            _matched_unit_trace(
                output.spike_probability,
                baseline_output.spike_probability,
                unit_count,
            )
        )
        hard_counts.append(
            _matched_unit_trace(
                output.hard_spikes,
                torch.zeros_like(baseline_output.hard_spikes),
                unit_count,
            )[:, response_onset:].sum(dim=1)
        )
    impulse_response = continuous[0][:, :, onset_step:].relu()
    step_response = continuous[1][:, :, onset_step:].relu()
    flicker_response = continuous[2][:, :, flicker_onset_step:]

    impulse_peak_by_polarity = impulse_response.amax(dim=2)
    step_peak_by_polarity = step_response.amax(dim=2)
    response_strength = impulse_peak_by_polarity + step_peak_by_polarity
    preferred_polarity = response_strength.argmax(dim=0)
    index = preferred_polarity.view(1, -1, 1)

    preferred_impulse = impulse_response.gather(
        0, index.expand(1, unit_count, impulse_response.shape[2])
    ).squeeze(0)
    preferred_step = step_response.gather(
        0, index.expand(1, unit_count, step_response.shape[2])
    ).squeeze(0)
    preferred_flicker = flicker_response.gather(
        0, index.expand(1, unit_count, flicker_response.shape[2])
    ).squeeze(0)
    impulse_peak, peak_index = preferred_impulse.max(dim=1)
    step_peak = preferred_step.max(dim=1).values
    valid_response_mask = response_strength.max(dim=0).values >= minimum_continuous_response
    half_peak = 0.5 * impulse_peak
    impulse_width_ms = (
        (preferred_impulse >= half_peak[:, None]).sum(dim=1).to(torch.float32) * dt_ms
    )
    sustained_steps = max(1, preferred_step.shape[1] // 4)
    response_scale = torch.maximum(impulse_peak, step_peak).clamp_min(
        minimum_continuous_response
    )
    hard_evoked_spike_count = torch.stack(hard_counts).sum(dim=0).gather(
        0, preferred_polarity.view(1, -1)
    ).squeeze(0)

    zeros = torch.zeros_like(impulse_peak)
    return TemporalProbeFeatures(
        preferred_polarity=preferred_polarity,
        valid_response_mask=valid_response_mask,
        impulse_peak=torch.where(valid_response_mask, impulse_peak, zeros),
        impulse_time_to_peak_ms=torch.where(
            valid_response_mask,
            peak_index.to(torch.float32) * dt_ms,
            zeros,
        ),
        impulse_width_ms=torch.where(valid_response_mask, impulse_width_ms, zeros),
        step_sustained_index=torch.where(
            valid_response_mask,
            preferred_step[:, -sustained_steps:].mean(dim=1)
            / step_peak.clamp_min(minimum_continuous_response),
            zeros,
        ),
        flicker_response=torch.where(
            valid_response_mask,
            _f1_amplitude(
                preferred_flicker,
                dt_ms=dt_ms,
                frequency_hz=flicker_frequency_hz,
            )
            / response_scale,
            zeros,
        ),
        hard_evoked_spike_count=hard_evoked_spike_count,
    )


def _probe_sequences(
    baseline_cone: torch.Tensor,
    center_indices: torch.Tensor,
    *,
    sequence_steps: int,
    onset_step: int,
    amplitude: float,
    half_period_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    unit_count = int(center_indices.numel())
    baseline = baseline_cone.view(1, 1, -1).expand(1, sequence_steps, -1).clone()
    sequences = [
        baseline_cone.view(1, 1, -1)
        .expand(2 * unit_count, sequence_steps, -1)
        .clone()
        for _ in range(3)
    ]
    impulse, step, flicker = sequences
    rows = torch.arange(2 * unit_count, device=baseline_cone.device)
    centers = center_indices.repeat(2)
    signs = torch.cat(
        (
            torch.ones(unit_count, device=baseline_cone.device),
            -torch.ones(unit_count, device=baseline_cone.device),
        )
    )
    impulse[rows, onset_step, centers] += signs * amplitude
    step[rows[:, None], torch.arange(onset_step, sequence_steps, device=baseline_cone.device), centers[:, None]] += signs[:, None] * amplitude
    flicker_onset_step = min(onset_step, 2 * sequence_steps // 5)
    phase = (
        torch.arange(sequence_steps - flicker_onset_step, device=baseline_cone.device)
        .div(half_period_steps, rounding_mode="floor")
        .remainder(2)
        .mul(-2)
        .add(1)
    )
    flicker[
        rows[:, None],
        torch.arange(flicker_onset_step, sequence_steps, device=baseline_cone.device),
        centers[:, None],
    ] += signs[:, None] * phase[None, :] * amplitude
    return baseline, impulse, step, flicker


def _matched_unit_trace(
    probe: torch.Tensor,
    baseline: torch.Tensor,
    unit_count: int,
) -> torch.Tensor:
    units = torch.arange(unit_count, device=probe.device)
    positive = probe[units, :, 0, units] - baseline[0, :, 0, :].transpose(0, 1)
    negative_rows = units + unit_count
    negative = probe[negative_rows, :, 1, units] - baseline[0, :, 1, :].transpose(0, 1)
    return torch.stack((positive, negative))


def _f1_amplitude(
    trace: torch.Tensor,
    *,
    dt_ms: float,
    frequency_hz: float,
) -> torch.Tensor:
    sample_count = trace.shape[1]
    time_seconds = (
        torch.arange(sample_count, device=trace.device, dtype=trace.dtype) * dt_ms / 1000.0
    )
    phase = 2.0 * math.pi * frequency_hz * time_seconds
    cosine = (trace * phase.cos()).sum(dim=1)
    sine = (trace * phase.sin()).sum(dim=1)
    return 2.0 * torch.sqrt(cosine.square() + sine.square()) / sample_count


__all__ = ["TemporalProbeFeatures", "run_temporal_probes"]
