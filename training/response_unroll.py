from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from models.cells.typed_rgc import TypedRGCOutput
from models.response_snn import (
    ResponseRetinaModel,
    ResponseRetinaState,
    detach_response_state,
    response_state_from_tensors,
    response_state_to_tensors,
)


class ResponseUnrollError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponseUnrollRequest:
    model: ResponseRetinaModel
    cone_response: torch.Tensor
    observed_counts: torch.Tensor
    burn_in_steps: int
    differentiable_steps: int
    checkpoint_block_steps: int
    checkpointed: bool


def unroll_response(
    request: ResponseUnrollRequest,
) -> tuple[TypedRGCOutput, ResponseRetinaState]:
    total = request.burn_in_steps + request.differentiable_steps
    if request.cone_response.shape[1] != total:
        raise ResponseUnrollError(
            "Response sequence length does not match the unroll contract"
        )
    if request.observed_counts.shape[:2] != request.cone_response.shape[:2]:
        raise ResponseUnrollError(
            "Observed response history must match batch and time"
        )
    weights = request.model.rgc.compute_spatial_weights()
    with torch.no_grad():
        _, state = request.model.forward_sequence(
            request.cone_response[:, : request.burn_in_steps],
            observed_counts=request.observed_counts[:, : request.burn_in_steps],
            spatial_weights=weights,
        )
    state = detach_response_state(state)
    cones = request.cone_response[:, request.burn_in_steps :]
    counts = request.observed_counts[:, request.burn_in_steps :]
    if not request.checkpointed:
        return request.model.forward_sequence(
            cones,
            state,
            observed_counts=counts,
            spatial_weights=weights,
        )
    return _checkpointed_region(
        request.model,
        cones,
        counts,
        state,
        weights,
        request.checkpoint_block_steps,
    )


def _checkpointed_region(
    model: ResponseRetinaModel,
    cones: torch.Tensor,
    counts: torch.Tensor,
    state: ResponseRetinaState,
    weights: torch.Tensor,
    block_steps: int,
) -> tuple[TypedRGCOutput, ResponseRetinaState]:
    flat_state = response_state_to_tensors(state)
    histories: list[list[torch.Tensor]] = [[], [], [], [], []]
    for start in range(0, cones.shape[1], block_steps):
        cone_block = cones[:, start : start + block_steps]
        count_block = counts[:, start : start + block_steps]

        def run_block(
            block_cones: torch.Tensor,
            block_counts: torch.Tensor,
            cached_weights: torch.Tensor,
            *state_values: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            output, next_state = model.forward_sequence(
                block_cones,
                response_state_from_tensors(tuple(state_values)),
                observed_counts=block_counts,
                spatial_weights=cached_weights,
            )
            return (
                *response_state_to_tensors(next_state),
                output.spike_logits,
                output.spike_probability,
                output.hard_spikes,
                output.filtered_rate,
                output.generator_potential,
            )

        values = checkpoint(
            run_block,
            cone_block,
            count_block,
            weights,
            *flat_state,
            use_reentrant=False,
        )
        flat_state = tuple(values[:8])
        for history, value in zip(histories, values[8:], strict=True):
            history.append(value)
    return (
        TypedRGCOutput(*(torch.cat(history, dim=1) for history in histories)),
        response_state_from_tensors(flat_state),
    )


__all__ = ["ResponseUnrollError", "ResponseUnrollRequest", "unroll_response"]
