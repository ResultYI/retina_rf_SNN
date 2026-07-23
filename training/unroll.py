from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from models.cells.rgc_types import RGCOutput
from models.retina_snn import (
    RetinaModel,
    RetinaState,
    state_from_tensors,
    state_to_tensors,
)


@dataclass(frozen=True, slots=True)
class ForwardRegionRequest:
    model: RetinaModel
    region: torch.Tensor
    state: RetinaState
    spatial_weights: torch.Tensor
    checkpointed: bool
    block_steps: int


def forward_region(
    request: ForwardRegionRequest,
) -> tuple[RGCOutput, RetinaState]:
    if not request.checkpointed:
        return request.model.forward_sequence(
            request.region,
            request.state,
            spatial_weights=request.spatial_weights,
        )
    flat_state = state_to_tensors(request.state)
    histories: list[list[torch.Tensor]] = [[], [], [], [], []]
    for start in range(0, request.region.shape[1], request.block_steps):
        block = request.region[:, start : start + request.block_steps]

        def run_block(
            block_input: torch.Tensor,
            cached_weights: torch.Tensor,
            *state_values: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            output, next_state = request.model.forward_sequence(
                block_input,
                state_from_tensors(tuple(state_values)),
                spatial_weights=cached_weights,
            )
            return (
                *state_to_tensors(next_state),
                output.hard_spikes,
                output.surrogate_spikes,
                output.spike_probability,
                output.rates,
                output.generator_potential,
            )

        values = checkpoint(
            run_block,
            block,
            request.spatial_weights,
            *flat_state,
            use_reentrant=False,
        )
        flat_state = tuple(values[:8])
        for target, value in zip(histories, values[8:], strict=True):
            target.append(value)
    output = RGCOutput(
        *(torch.cat(values, dim=1) for values in histories)
    )
    return output, state_from_tensors(flat_state)


__all__ = ["ForwardRegionRequest", "forward_region"]
