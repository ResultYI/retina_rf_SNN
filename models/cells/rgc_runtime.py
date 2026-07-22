from __future__ import annotations

import torch

from models.cells.rgc_types import RGCState


def bounded(raw: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return minimum + (maximum - minimum) * torch.sigmoid(raw)


def raw_from_bounded(value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    fraction = ((value - minimum) / (maximum - minimum)).clamp(1e-5, 1.0 - 1e-5)
    return torch.logit(fraction)


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
        raise ValueError("RGC state requires four tensors")
    return RGCState(*tensors)

