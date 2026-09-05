from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from models.mechanistic_retina.contracts import PathwayClamp


@dataclass(frozen=True, slots=True)
class GateValues:
    h1: torch.Tensor
    ac_local: torch.Tensor
    ac_transient: torch.Tensor
    history: torch.Tensor


@dataclass(frozen=True, slots=True)
class PathwayGateError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class PathwayGates(nn.Module):
    def __init__(
        self,
        initial: float,
        *,
        group_index: torch.Tensor,
        trainable: bool,
        h1_amplitude: float = 0.01,
        h1_amplitude_bounds: tuple[float, float] = (0.0, 0.2),
    ) -> None:
        super().__init__()
        if not 0.0 <= initial <= 1.0:
            raise PathwayGateError("pathway gate initial value must be in [0, 1]")
        if group_index.ndim != 1 or group_index.numel() == 0:
            raise PathwayGateError("AC gate group index must be a nonempty vector")
        if group_index.dtype != torch.long or bool((group_index < 0).any()):
            raise PathwayGateError("AC gate group index must contain nonnegative integers")
        group_count = int(group_index.max()) + 1
        expected_groups = torch.arange(group_count, device=group_index.device)
        if not torch.equal(torch.unique(group_index), expected_groups):
            raise PathwayGateError("AC gate group index must be contiguous from zero")
        lower, upper = h1_amplitude_bounds
        if (
            not all(math.isfinite(item) for item in (h1_amplitude, lower, upper))
            or lower < 0
            or not lower < h1_amplitude < upper
        ):
            raise PathwayGateError(
                "H1 amplitude must lie strictly inside nonnegative bounds"
            )
        value = torch.tensor(float(initial))
        amplitude_fraction = (h1_amplitude - lower) / (upper - lower)
        self.raw_h1_amplitude = nn.Parameter(
            torch.logit(torch.tensor(amplitude_fraction)),
            requires_grad=trainable,
        )
        group_value = value.repeat(group_count)
        self.ac_local = nn.Parameter(group_value.clone(), requires_grad=trainable)
        self.ac_transient = nn.Parameter(group_value.clone(), requires_grad=trainable)
        self.history = nn.Parameter(value.clone(), requires_grad=trainable)
        self.register_buffer("group_index", group_index.clone())
        self.register_buffer(
            "h1_amplitude_bounds", torch.tensor(h1_amplitude_bounds)
        )

    @property
    def h1(self) -> torch.Tensor:
        lower, upper = self.h1_amplitude_bounds.unbind()
        return lower + (upper - lower) * torch.sigmoid(self.raw_h1_amplitude)

    def set_h1_amplitude_(self, value: float) -> None:
        lower, upper = self.h1_amplitude_bounds.unbind()
        amplitude = self.raw_h1_amplitude.new_tensor(value)
        if not bool((lower < amplitude) & (amplitude < upper)):
            raise PathwayGateError("H1 amplitude must lie strictly inside bounds")
        fraction = (amplitude - lower) / (upper - lower)
        with torch.no_grad():
            self.raw_h1_amplitude.copy_(torch.logit(fraction))

    def values(self, clamps: frozenset[PathwayClamp]) -> GateValues:
        ac_by_group = torch.softmax(
            torch.stack((self.ac_local, self.ac_transient), dim=1), dim=1
        )
        ac = ac_by_group[self.group_index]
        return GateValues(
            self._value(self.h1, PathwayClamp.H1, clamps),
            self._value(ac[:, 0], PathwayClamp.AMACRINE_LOCAL, clamps),
            self._value(ac[:, 1], PathwayClamp.AMACRINE_TRANSIENT, clamps),
            self._value(self.history, PathwayClamp.RGC_HISTORY, clamps),
        )

    def project_(self) -> None:
        with torch.no_grad():
            self.history.clamp_(0, 1)

    @staticmethod
    def _value(
        parameter: torch.Tensor,
        clamp: PathwayClamp,
        clamps: frozenset[PathwayClamp],
    ) -> torch.Tensor:
        return torch.zeros_like(parameter) if clamp in clamps else parameter


__all__ = ["GateValues", "PathwayGateError", "PathwayGates"]
