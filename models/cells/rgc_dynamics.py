from __future__ import annotations

from dataclasses import dataclass

import torch

from models.cells.rgc_types import (
    RGCConfigurationError,
    RGCState,
    RGCStepOutput,
)


@dataclass(frozen=True, slots=True)
class RGCDynamicsParameters:
    cone_count: int
    unit_count: int
    dt_ms: float
    surrogate_slope: float
    debug_checks: bool
    subunit_tau_ms: torch.Tensor
    subunit_gain: torch.Tensor
    sustained_mix: torch.Tensor
    amacrine_gain: torch.Tensor
    membrane_tau_ms: torch.Tensor
    adaptation_gain: torch.Tensor
    threshold: torch.Tensor
    adaptation_tau_ms: torch.Tensor
    readout_rate_tau_ms: torch.Tensor


@dataclass(frozen=True, slots=True)
class RGCDynamicsRequest:
    bipolar_output: torch.Tensor
    amacrine_output: torch.Tensor
    previous: RGCState
    spatial_weights: torch.Tensor
    parameters: RGCDynamicsParameters


def step_rgc_dynamics(
    request: RGCDynamicsRequest,
) -> tuple[RGCStepOutput, RGCState]:
    parameters = request.parameters
    expected = (
        request.bipolar_output.shape[0],
        2,
        2,
        parameters.cone_count,
    )
    if (
        request.bipolar_output.shape != expected
        or request.amacrine_output.shape != expected
    ):
        raise RGCConfigurationError(
            "RGC inputs must have shape [batch,2,2,Ncone]"
        )
    if request.spatial_weights.shape != (
        parameters.unit_count,
        parameters.cone_count,
    ):
        raise RGCConfigurationError(
            "spatial_weights must have shape [unit,cone]"
        )
    if parameters.debug_checks and not all(
        torch.isfinite(value).all()
        for value in (
            request.bipolar_output,
            request.amacrine_output,
            request.spatial_weights,
            request.previous.membrane,
            request.previous.adaptation,
            request.previous.rate,
            request.previous.subunit_energy,
        )
    ):
        raise RGCConfigurationError("RGC inputs and state must be finite")

    pooled_bipolar = torch.einsum(
        "uc,bpkc->bpku",
        request.spatial_weights,
        request.bipolar_output,
    )
    pooled_amacrine = torch.einsum(
        "uc,bpkc->bpku",
        request.spatial_weights,
        request.amacrine_output,
    )
    subunit_leak = torch.exp(
        -parameters.dt_ms / parameters.subunit_tau_ms
    ).view(1, 1, 1, -1)
    subunit_energy = (
        subunit_leak * request.previous.subunit_energy
        + (1.0 - subunit_leak) * pooled_bipolar.square()
    )
    adapted = pooled_bipolar / (
        1.0
        + parameters.subunit_gain.view(1, 1, 1, -1) * subunit_energy
    )
    mix = parameters.sustained_mix.view(1, 1, -1)
    bipolar_drive = mix * adapted[:, :, 0] + (1.0 - mix) * adapted[:, :, 1]
    amacrine_drive = (
        mix * pooled_amacrine[:, :, 0]
        + (1.0 - mix) * pooled_amacrine[:, :, 1]
    )
    current = (
        bipolar_drive
        - parameters.amacrine_gain.view(1, 1, -1) * amacrine_drive
    )
    membrane_leak = torch.exp(
        -parameters.dt_ms / parameters.membrane_tau_ms
    ).view(1, 1, -1)
    pre_reset = (
        membrane_leak * request.previous.membrane
        + (1.0 - membrane_leak)
        * (
            current
            - parameters.adaptation_gain.view(1, 1, -1)
            * request.previous.adaptation
        )
    )
    threshold = parameters.threshold.view(1, 1, -1)
    probability = torch.sigmoid(
        parameters.surrogate_slope * (pre_reset - threshold)
    )
    hard = (pre_reset >= threshold).to(pre_reset.dtype)
    spikes = hard + (probability - probability.detach())
    hard_event = hard.detach()
    membrane = pre_reset * (1.0 - hard_event)
    adaptation_leak = torch.exp(
        -parameters.dt_ms / parameters.adaptation_tau_ms
    ).view(1, 1, -1)
    adaptation = (
        adaptation_leak * request.previous.adaptation
        + (1.0 - adaptation_leak) * hard_event
    )
    rate_leak = torch.exp(
        -parameters.dt_ms / parameters.readout_rate_tau_ms
    )
    rate = rate_leak * request.previous.rate + (1.0 - rate_leak) * spikes
    output = RGCStepOutput(
        hard_spikes=hard,
        surrogate_spikes=spikes,
        spike_probability=probability,
        rates=rate,
        generator_potential=pre_reset,
    )
    return output, RGCState(
        membrane,
        adaptation,
        rate,
        subunit_energy,
    )


__all__ = [
    "RGCDynamicsParameters",
    "RGCDynamicsRequest",
    "step_rgc_dynamics",
]
