from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch
from torch import nn

from models.mechanistic_retina.contracts import (
    ArchitectureMode,
    MechanisticRetinaConfig,
)
from models.mechanistic_retina.delay_parameters import (
    bounded_delay_ms,
    raw_delay_from_ms,
)
from models.mechanistic_retina.graph import ConeGraph
from models.mechanistic_retina.state import causal_fractional_delay, causal_lowpass
from models.mechanistic_retina.temporal_parameters import (
    bounded_tau_ms,
    raw_tau_from_ms,
)


@dataclass(frozen=True, slots=True)
class H1Output:
    graph_drive: torch.Tensor
    state: torch.Tensor
    surround: torch.Tensor
    modulated_cones: torch.Tensor


class H1Pathway(nn.Module):
    def __init__(
        self,
        config: MechanisticRetinaConfig,
        cone_positions: torch.Tensor,
    ) -> None:
        super().__init__()
        mode = ArchitectureMode(config.architecture_mode)
        match mode:
            case ArchitectureMode.MECHANISM_IDENTIFIABLE:
                radius = config.h1_radius_deg
            case ArchitectureMode.LEGACY:
                radius = config.graph_radius_deg
            case unreachable:
                assert_never(unreachable)
        self.graph = ConeGraph(cone_positions, radius)
        initial_tau = torch.tensor(config.h1_tau_ms, dtype=torch.float32)
        tau_bounds = torch.tensor(config.h1_tau_bounds_ms, dtype=torch.float32)
        self.raw_tau = nn.Parameter(raw_tau_from_ms(initial_tau, tau_bounds))
        self.register_buffer("tau_bounds_ms", tau_bounds)
        initial_delay = torch.tensor(config.h1_delay_ms, dtype=torch.float32)
        delay_bounds = torch.tensor(config.h1_delay_bounds_ms, dtype=torch.float32)
        self.raw_delay = nn.Parameter(raw_delay_from_ms(initial_delay, delay_bounds))
        self.register_buffer("delay_bounds_ms", delay_bounds)
        self._dt_ms = config.dt_ms

    @property
    def tau_ms(self) -> torch.Tensor:
        return bounded_tau_ms(self.raw_tau, self.tau_bounds_ms)

    @property
    def decay(self) -> torch.Tensor:
        return torch.exp(-self._dt_ms / self.tau_ms)

    @property
    def delay_ms(self) -> torch.Tensor:
        return bounded_delay_ms(self.raw_delay, self.delay_bounds_ms)

    def forward(
        self,
        cones: torch.Tensor,
        *,
        amplitude: torch.Tensor | None = None,
        clamped: bool = False,
    ) -> H1Output:
        graph_drive = self.graph(cones)
        delayed_drive = causal_fractional_delay(
            graph_drive, self.delay_ms, dt_ms=self._dt_ms
        )
        state = causal_lowpass(delayed_drive, self.decay)
        active_amplitude = cones.new_zeros(()) if clamped else amplitude
        if active_amplitude is None:
            active_amplitude = cones.new_ones(())
        surround = active_amplitude * self.graph.transpose_apply(state)
        return H1Output(graph_drive, state, surround, cones - surround)


__all__ = ["H1Output", "H1Pathway"]
