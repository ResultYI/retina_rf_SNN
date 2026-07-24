from __future__ import annotations

import math

import torch

from models.cells.rgc_types import RGCState


class RGCRuntimeError(ValueError):
    pass


def bounded(raw: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return minimum + (maximum - minimum) * torch.sigmoid(raw)


def raw_from_bounded(value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    fraction = ((value - minimum) / (maximum - minimum)).clamp(1e-5, 1.0 - 1e-5)
    return torch.logit(fraction)


def causal_filter(
    events: torch.Tensor,
    *,
    dt_ms: float,
    tau_ms: float,
) -> torch.Tensor:
    if events.ndim != 4:
        raise RGCRuntimeError("events must have shape [batch,time,polarity,unit]")
    if dt_ms <= 0 or tau_ms <= 0:
        raise RGCRuntimeError("filter time constants must be positive")
    leak = math.exp(-dt_ms / tau_ms)
    state = torch.zeros_like(events[:, 0])
    history: list[torch.Tensor] = []
    for event_t in events.unbind(dim=1):
        state = leak * state + (1.0 - leak) * event_t
        history.append(state)
    return torch.stack(history, dim=1)


def detach_rgc_state(state: RGCState) -> RGCState:
    return RGCState(
        membrane=state.membrane.detach(),
        adaptation=state.adaptation.detach(),
        rate=state.rate.detach(),
        subunit_energy=state.subunit_energy.detach(),
    )


def rgc_state_to_tensors(state: RGCState) -> tuple[torch.Tensor, ...]:
    return state.membrane, state.adaptation, state.rate, state.subunit_energy


def rgc_state_from_tensors(tensors: tuple[torch.Tensor, ...]) -> RGCState:
    if len(tensors) != 4:
        raise RGCRuntimeError("RGC state requires four tensors")
    return RGCState(*tensors)
