from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CellSpecificGainError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


class CellSpecificPathwayGains(nn.Module):
    def __init__(self, cell_count: int) -> None:
        super().__init__()
        if cell_count < 1:
            raise CellSpecificGainError("cell-specific gains require at least one cell")
        self.log_bc = nn.Parameter(torch.zeros(cell_count))
        self.log_ac = nn.Parameter(torch.zeros(cell_count))

    @property
    def bc(self) -> torch.Tensor:
        return torch.exp(self.log_bc)

    @property
    def ac(self) -> torch.Tensor:
        return torch.exp(self.log_ac)

    @property
    def pathway_values(self) -> torch.Tensor:
        return torch.stack((self.bc, self.bc, self.ac, self.ac), dim=1)

    @property
    def pathway_names(self) -> tuple[str, ...]:
        return ("BC", "AC")

    @property
    def raw_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.log_bc, self.log_ac)

    @property
    def audit_values(self) -> torch.Tensor:
        return torch.stack((self.bc, self.ac), dim=1)

    @property
    def bc_current_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        gain = self.bc[None, None]
        return gain, gain

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        return currents * self.pathway_values[None, None]

    def scale_basis(self, basis: torch.Tensor) -> torch.Tensor:
        return basis * self.pathway_values[None, None, :, :, None, None]


class CellSpecificPathwayMixtureGains(nn.Module):
    def __init__(self, cell_count: int) -> None:
        super().__init__()
        if cell_count < 1:
            raise CellSpecificGainError("cell-specific gains require at least one cell")
        self.log_bc_sustained = nn.Parameter(torch.zeros(cell_count))
        self.log_bc_transient = nn.Parameter(torch.zeros(cell_count))
        self.log_ac_local = nn.Parameter(torch.zeros(cell_count))
        self.log_ac_transient = nn.Parameter(torch.zeros(cell_count))

    @property
    def pathway_names(self) -> tuple[str, ...]:
        return ("BC_sustained", "BC_transient", "AC_local", "AC_transient")

    @property
    def raw_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.log_bc_sustained,
            self.log_bc_transient,
            self.log_ac_local,
            self.log_ac_transient,
        )

    @property
    def pathway_values(self) -> torch.Tensor:
        return torch.stack(tuple(torch.exp(raw) for raw in self.raw_parameters), dim=1)

    @property
    def audit_values(self) -> torch.Tensor:
        return self.pathway_values

    @property
    def bc_current_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.pathway_values
        return values[None, None, :, 0], values[None, None, :, 1]

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        return currents * self.pathway_values[None, None]

    def scale_basis(self, basis: torch.Tensor) -> torch.Tensor:
        return basis * self.pathway_values[None, None, :, :, None, None]


def build_cell_specific_gains(
    cell_count: int,
    aggregate_enabled: bool,
    pathway_mixture_enabled: bool,
) -> CellSpecificPathwayGains | CellSpecificPathwayMixtureGains | None:
    if pathway_mixture_enabled:
        return CellSpecificPathwayMixtureGains(cell_count)
    if aggregate_enabled:
        return CellSpecificPathwayGains(cell_count)
    return None


__all__ = [
    "CellSpecificGainError",
    "CellSpecificPathwayGains",
    "CellSpecificPathwayMixtureGains",
    "build_cell_specific_gains",
]
