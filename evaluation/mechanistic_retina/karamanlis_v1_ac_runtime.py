from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import torch

from evaluation.mechanistic_retina.rf_effective import effective_rf
from models.mechanistic_retina.contracts import MechanisticRetinaOutput, PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina

AC_CLAMPS: Final = frozenset(
    {PathwayClamp.AMACRINE_LOCAL, PathwayClamp.AMACRINE_TRANSIENT}
)


@dataclass(frozen=True, slots=True)
class ResponseTensors:
    logits: torch.Tensor
    probability: torch.Tensor
    ac_local: torch.Tensor
    ac_transient: torch.Tensor


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    model: MechanisticGraphTemporalRetina
    cones: torch.Tensor
    histories: torch.Tensor
    batch_size: int


@dataclass(frozen=True, slots=True)
class CollectedResponses:
    normal: ResponseTensors
    clamped: ResponseTensors
    upstream_outputs_unchanged: bool


@dataclass(frozen=True, slots=True)
class ACClampVerification:
    local_exact_zero: bool
    transient_exact_zero: bool
    local_max_abs: float
    transient_max_abs: float


@dataclass(frozen=True, slots=True)
class V1ACRuntimeError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def collect_responses(request: EvaluationRequest) -> CollectedResponses:
    normal_parts: list[ResponseTensors] = []
    clamped_parts: list[ResponseTensors] = []
    upstream_unchanged = True
    for start in range(0, request.cones.shape[0], request.batch_size):
        stop = start + request.batch_size
        cones = request.cones[start:stop]
        histories = request.histories[start:stop]
        with torch.no_grad():
            normal = request.model.forward_sequence(cones, observed_counts=histories)
            clamped = request.model.forward_sequence(
                cones, observed_counts=histories, clamps=AC_CLAMPS
            )
        upstream_unchanged &= all(
            torch.equal(left, right)
            for left, right in zip(
                _upstream_tensors(normal), _upstream_tensors(clamped), strict=True
            )
        )
        normal_parts.append(_response_tensors(normal))
        clamped_parts.append(_response_tensors(clamped))
    return CollectedResponses(
        _concatenate_responses(normal_parts),
        _concatenate_responses(clamped_parts),
        upstream_unchanged,
    )


def mean_temporal_rf(
    request: EvaluationRequest,
    indices: Sequence[int],
    clamps: frozenset[PathwayClamp],
) -> torch.Tensor:
    total = torch.zeros(
        request.histories.shape[-1], request.model.config.lag_steps, dtype=torch.float64
    )
    for start in range(0, len(indices), request.batch_size):
        batch = torch.as_tensor(indices[start : start + request.batch_size])
        rf = effective_rf(
            request.model,
            request.cones[batch],
            request.histories[batch],
            clamps=clamps,
        )
        total += rf.sum(dim=-1).double().sum(dim=0)
    return torch.flip((total / len(indices)).float(), dims=(1,))


def validate_ac_clamp(
    responses: CollectedResponses,
    *,
    state_unchanged: bool,
) -> ACClampVerification:
    local = responses.clamped.ac_local
    transient = responses.clamped.ac_transient
    verification = ACClampVerification(
        torch.count_nonzero(local).item() == 0,
        torch.count_nonzero(transient).item() == 0,
        float(local.abs().max()),
        float(transient.abs().max()),
    )
    if not (
        verification.local_exact_zero
        and verification.transient_exact_zero
        and responses.upstream_outputs_unchanged
        and state_unchanged
    ):
        raise V1ACRuntimeError("AC-only structural clamp contract failed")
    return verification


def stimulus_onset_step(cones: torch.Tensor) -> int:
    active = cones.abs().amax(dim=(0, 2)) > 0
    indices = torch.nonzero(active, as_tuple=False).flatten()
    if indices.numel() == 0 or int(indices[0]) == 0:
        raise V1ACRuntimeError("held-out stimuli lack a pre-flash baseline")
    return int(indices[0])


def _response_tensors(output: MechanisticRetinaOutput) -> ResponseTensors:
    return ResponseTensors(
        output.logits.detach(),
        output.spike_probability.detach(),
        output.amacrine_local_current.detach(),
        output.amacrine_transient_current.detach(),
    )


def _concatenate_responses(parts: Sequence[ResponseTensors]) -> ResponseTensors:
    return ResponseTensors(
        torch.cat(tuple(part.logits for part in parts)),
        torch.cat(tuple(part.probability for part in parts)),
        torch.cat(tuple(part.ac_local for part in parts)),
        torch.cat(tuple(part.ac_transient for part in parts)),
    )


def _upstream_tensors(output: MechanisticRetinaOutput) -> tuple[torch.Tensor, ...]:
    return output.tensors()[:13] + (
        output.amacrine_local_state,
        output.amacrine_transient_state,
    )


__all__ = [
    "AC_CLAMPS",
    "ACClampVerification",
    "CollectedResponses",
    "EvaluationRequest",
    "ResponseTensors",
    "V1ACRuntimeError",
    "collect_responses",
    "mean_temporal_rf",
    "stimulus_onset_step",
    "validate_ac_clamp",
]
