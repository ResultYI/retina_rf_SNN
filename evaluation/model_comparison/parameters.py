from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ParameterInventory:
    total: int
    requires_grad: int
    optimizer_listed: int
    nonzero_gradient: int | None
    actually_updated: int | None
    rf_bearing: int
    state_dynamics: int
    gates: int


@dataclass(frozen=True, slots=True)
class ParameterInventoryError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def parameter_inventory(
    model: nn.Module,
    optimizer_parameters: Iterable[nn.Parameter],
    *,
    initial_parameters: Mapping[str, torch.Tensor] | None = None,
) -> ParameterInventory:
    named = tuple(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in named)
    trainable = sum(parameter.numel() for _, parameter in named if parameter.requires_grad)
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    optimizer_listed = sum(
        parameter.numel() for _, parameter in named if id(parameter) in optimizer_ids
    )
    gradients = tuple(parameter.grad for _, parameter in named)
    nonzero_gradient = (
        sum(
            int(torch.count_nonzero(gradient))
            for gradient in gradients
            if gradient is not None
        )
        if any(gradient is not None for gradient in gradients)
        else None
    )
    actually_updated = (
        _updated_element_count(named, initial_parameters)
        if initial_parameters is not None
        else None
    )
    gates = sum(parameter.numel() for name, parameter in named if name.startswith("gates."))
    state = sum(parameter.numel() for name, parameter in named if name.startswith("rgc."))
    rf_bearing = sum(
        parameter.numel()
        for name, parameter in named
        if name.startswith(
            (
                "h1.",
                "feature_bank.",
                "shared_subunits.",
                "bipolar.",
                "amacrine.",
                "operator.",
            )
        )
    )
    return ParameterInventory(
        total,
        trainable,
        optimizer_listed,
        nonzero_gradient,
        actually_updated,
        rf_bearing,
        state,
        gates,
    )


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _updated_element_count(
    named: tuple[tuple[str, nn.Parameter], ...],
    initial_parameters: Mapping[str, torch.Tensor],
) -> int:
    current_names = {name for name, _ in named}
    if current_names != set(initial_parameters):
        raise ParameterInventoryError(
            "initial parameter snapshot names differ from the model"
        )
    count = 0
    for name, parameter in named:
        initial = initial_parameters[name]
        if initial.shape != parameter.shape:
            raise ParameterInventoryError(
                f"initial parameter shape differs for {name}"
            )
        count += int(torch.count_nonzero(parameter.detach() != initial.to(parameter.device)))
    return count


__all__ = [
    "ParameterInventory",
    "ParameterInventoryError",
    "parameter_inventory",
    "parameter_snapshot",
]
