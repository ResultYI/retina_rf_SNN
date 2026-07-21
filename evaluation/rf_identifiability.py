from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RFGLMIdentifiabilityRequest:
    sequence_count: int
    input_steps: int
    source_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class RFGLMIdentifiabilityResult:
    sufficient: bool
    required_sequences: dict[str, int]
    reason: str


def rf_glm_identifiability(
    request: RFGLMIdentifiabilityRequest,
) -> RFGLMIdentifiabilityResult:
    if request.sequence_count < 1 or request.input_steps < 1:
        raise ValueError("sequence_count and input_steps must be positive")
    if not request.source_counts or any(count < 1 for count in request.source_counts.values()):
        raise ValueError("source_counts must contain positive local support sizes")
    required = {
        population: request.input_steps * count + 1
        for population, count in request.source_counts.items()
    }
    failures = tuple(
        f"{population}:{need}"
        for population, need in required.items()
        if request.sequence_count < need
    )
    if failures:
        return RFGLMIdentifiabilityResult(
            sufficient=False,
            required_sequences=required,
            reason=(
                "aggregate_count_glm_requires_sequences>="
                + ",".join(failures)
                + f";available={request.sequence_count}"
            ),
        )
    return RFGLMIdentifiabilityResult(
        sufficient=True,
        required_sequences=required,
        reason="aggregate_count_glm_identifiable",
    )
