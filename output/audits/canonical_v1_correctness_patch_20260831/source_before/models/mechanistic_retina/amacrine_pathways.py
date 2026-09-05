from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.state import causal_fractional_delay, causal_lowpass
from models.mechanistic_retina.delay_parameters import ordered_bounded_delay_ms, raw_ordered_delay_from_ms
from models.mechanistic_retina.temporal_parameters import (
    ordered_bounded_tau_ms,
    raw_ordered_tau_from_ms,
)


@dataclass(frozen=True, slots=True)
class AmacrineOutput:
    local_state: torch.Tensor
    local_current: torch.Tensor
    transient_state: torch.Tensor
    transient_current: torch.Tensor


class AmacrinePathways(nn.Module):
    def __init__(
        self,
        config: MechanisticRetinaConfig,
        cell_types: tuple[str, ...],
        polarities: tuple[str, ...],
    ) -> None:
        super().__init__()
        shared = (
            ArchitectureMode(config.architecture_mode)
            is ArchitectureMode.MECHANISM_IDENTIFIABLE
        )
        group_index, _ = _group_indices(cell_types, polarities, shared=shared)
        initial_tau = torch.tensor(config.amacrine_tau_ms, dtype=torch.float32)
        tau_bounds = torch.tensor(config.amacrine_tau_bounds_ms, dtype=torch.float32)
        self.raw_tau = nn.Parameter(raw_ordered_tau_from_ms(initial_tau, tau_bounds))
        self.register_buffer("group_index", group_index)
        self.register_buffer("tau_bounds_ms", tau_bounds)
        delay_bounds = torch.tensor(config.ac_delay_bounds_ms, dtype=torch.float32)
        self.raw_delay = nn.Parameter(raw_ordered_delay_from_ms(
            torch.tensor(config.ac_delay_ms, dtype=torch.float32), delay_bounds
        ))
        self.register_buffer("delay_bounds_ms", delay_bounds)
        self._dt_ms = config.dt_ms

    @property
    def tau_ms(self) -> torch.Tensor:
        return ordered_bounded_tau_ms(self.raw_tau, self.tau_bounds_ms)

    @property
    def decay(self) -> torch.Tensor:
        return torch.exp(-self._dt_ms / self.tau_ms)

    @property
    def delay_ms(self) -> torch.Tensor:
        return ordered_bounded_delay_ms(self.raw_delay, self.delay_bounds_ms)

    def presynaptic_states(self, bc_presynaptic: torch.Tensor) -> torch.Tensor:
        batch, time, cells, paths = bc_presynaptic.shape
        if paths != 2:
            raise AmacrineInputError("AC input must have two BC sustained/transient channels")
        delayed = causal_fractional_delay(
            bc_presynaptic.reshape(batch, time, cells * paths),
            self.delay_ms.repeat(cells), dt_ms=self._dt_ms,
        )
        return causal_lowpass(delayed, self.decay.repeat(cells)).reshape(batch, time, cells, paths)

    def forward(
        self,
        bc_presynaptic: torch.Tensor,
        *,
        local_gate: torch.Tensor,
        transient_gate: torch.Tensor,
    ) -> AmacrineOutput:
        states = self.presynaptic_states(bc_presynaptic)
        local_state = states[..., 0]
        transient_state = states[..., 1]
        local = -local_gate * local_state
        transient = -transient_gate * transient_state
        return AmacrineOutput(local_state, local, transient_state, transient)


class AmacrineInputError(ValueError):
    pass


def _group_indices(
    cell_types: tuple[str, ...],
    polarities: tuple[str, ...],
    *,
    shared: bool,
) -> tuple[torch.Tensor, int]:
    if not shared:
        return torch.arange(len(cell_types)), len(cell_types)
    keys = tuple(zip(cell_types, polarities, strict=True))
    groups = tuple(dict.fromkeys(keys))
    return torch.tensor(tuple(groups.index(key) for key in keys)), len(groups)


__all__ = ["AmacrineInputError", "AmacrineOutput", "AmacrinePathways"]
