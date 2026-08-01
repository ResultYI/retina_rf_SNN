from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn

from evaluation.parameter_audit import (
    ParameterRole,
    TypeDifferential,
    parameter_role,
    type_differential,
)
from training.response_trainer import ResponseTrainer


NamedParameters: TypeAlias = tuple[tuple[str, nn.Parameter], ...]
TensorValues: TypeAlias = tuple[torch.Tensor, ...]
PARAMETER_ROLES: tuple[ParameterRole, ...] = (
    "rgc_type_base",
    "rgc_cell_residual",
    "polarity_pathway",
    "other",
)


class GradientDecompositionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NamedNorm:
    name: str
    value: float


@dataclass(frozen=True, slots=True)
class TypeVector:
    type_id: str
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GradientSegmentAudit:
    likelihood: float
    raw_gradient_norm: float
    raw_group_norms: tuple[NamedNorm, ...]
    raw_type_vectors: tuple[TypeVector, ...]
    raw_type_differential: TypeDifferential
    effective_update_norm: float
    effective_group_norms: tuple[NamedNorm, ...]
    effective_type_vectors: tuple[TypeVector, ...]
    effective_type_differential: TypeDifferential


@dataclass(frozen=True, slots=True)
class AdamStateAudit:
    exp_avg_norm: float
    exp_avg_sq_norm: float
    exp_avg_group_norms: tuple[NamedNorm, ...]
    exp_avg_sq_group_norms: tuple[NamedNorm, ...]
    exp_avg_type_vectors: tuple[TypeVector, ...]
    exp_avg_sq_type_vectors: tuple[TypeVector, ...]


def effective_updates(
    trainer: ResponseTrainer,
    named_parameters: NamedParameters,
    gradients: TensorValues,
) -> TensorValues:
    raw_norm = tensor_norm(gradients)
    clip_norm = trainer.config.training.gradient_clip_norm
    clip_scale = min(1.0, clip_norm / (raw_norm + 1e-6))
    updates = []
    for (_, parameter), gradient in zip(named_parameters, gradients, strict=True):
        group = _optimizer_group(trainer.optimizer, parameter)
        state = trainer.optimizer.state[parameter]
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter))
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter))
        beta1, beta2 = group["betas"]
        step = int(torch.as_tensor(state.get("step", 0)).item()) + 1
        clipped = gradient * clip_scale
        next_avg = beta1 * exp_avg + (1.0 - beta1) * clipped
        next_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * clipped.square()
        corrected_avg = next_avg / (1.0 - beta1**step)
        corrected_avg_sq = next_avg_sq / (1.0 - beta2**step)
        learning_rate = float(group["lr"])
        adam_update = learning_rate * corrected_avg / (
            corrected_avg_sq.sqrt() + float(group["eps"])
        )
        decay_update = learning_rate * float(group["weight_decay"]) * parameter
        updates.append(-(adam_update + decay_update).detach())
    return tuple(updates)


def adam_moments(
    trainer: ResponseTrainer,
    named_parameters: NamedParameters,
) -> tuple[TensorValues, TensorValues]:
    exp_avg = []
    exp_avg_sq = []
    for _, parameter in named_parameters:
        state = trainer.optimizer.state[parameter]
        exp_avg.append(state.get("exp_avg", torch.zeros_like(parameter)).detach())
        exp_avg_sq.append(
            state.get("exp_avg_sq", torch.zeros_like(parameter)).detach()
        )
    return tuple(exp_avg), tuple(exp_avg_sq)


def group_norms(
    named_parameters: NamedParameters,
    values: TensorValues,
) -> tuple[NamedNorm, ...]:
    return tuple(
        NamedNorm(
            role,
            tensor_norm(
                tuple(
                    value
                    for (name, _), value in zip(
                        named_parameters,
                        values,
                        strict=True,
                    )
                    if parameter_role(name) == role
                )
            ),
        )
        for role in PARAMETER_ROLES
    )


def type_vectors(
    named_parameters: NamedParameters,
    values: TensorValues,
    type_ids: tuple[str, ...],
) -> tuple[TypeVector, ...]:
    vectors: list[TypeVector] = []
    for type_index, type_id in enumerate(type_ids):
        components = tuple(
            float(value.flatten()[type_index])
            for (name, _), value in zip(named_parameters, values, strict=True)
            if parameter_role(name) == "rgc_type_base"
        )
        if not components:
            raise GradientDecompositionError("No RGC type-base parameters were found")
        vectors.append(TypeVector(type_id, components))
    return tuple(vectors)


def differential(vectors: tuple[TypeVector, ...]) -> TypeDifferential:
    by_type = {vector.type_id: torch.tensor(vector.values) for vector in vectors}
    if "midget" not in by_type or "parasol" not in by_type:
        raise GradientDecompositionError(
            "Gradient decomposition requires midget and parasol type bases"
        )
    return type_differential(by_type["midget"], by_type["parasol"])


def tensor_norm(values: TensorValues) -> float:
    if not values:
        return 0.0
    return float(
        torch.linalg.vector_norm(torch.cat(tuple(value.flatten() for value in values)))
    )


def _optimizer_group(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
):
    for group in optimizer.param_groups:
        if any(candidate is parameter for candidate in group["params"]):
            return group
    raise GradientDecompositionError("Trainable parameter is missing from optimizer")


__all__ = [
    "AdamStateAudit",
    "GradientDecompositionError",
    "GradientSegmentAudit",
    "NamedNorm",
    "NamedParameters",
    "TensorValues",
    "TypeVector",
    "adam_moments",
    "differential",
    "effective_updates",
    "group_norms",
    "tensor_norm",
    "type_vectors",
]
