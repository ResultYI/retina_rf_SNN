from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class RGCConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RGCConfig:
    units_per_center: int
    support_radius_degs: float
    sigma_min_degs: float
    sigma_initial_degs: float
    sigma_max_degs: float
    dt_ms: float
    readout_rate_tau_ms: float
    max_tau_ms: float
    surrogate_slope: float
    adaptation_gain_max: float = 1.0
    amacrine_gain_max: float = 1.0
    subunit_gain_max: float = 3.0
    initialization_seed: int = 0
    debug_checks: bool = True

    def __post_init__(self) -> None:
        values = (
            self.support_radius_degs,
            self.sigma_min_degs,
            self.sigma_initial_degs,
            self.sigma_max_degs,
            self.dt_ms,
            self.readout_rate_tau_ms,
            self.max_tau_ms,
            self.surrogate_slope,
            self.adaptation_gain_max,
            self.amacrine_gain_max,
            self.subunit_gain_max,
        )
        if self.units_per_center < 1:
            raise RGCConfigurationError("units_per_center must be positive")
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise RGCConfigurationError("RGC scales must be finite and positive")
        if not self.sigma_min_degs < self.sigma_initial_degs < self.sigma_max_degs:
            raise RGCConfigurationError("Initial sigma must lie inside its bounds")
        if self.max_tau_ms <= self.dt_ms:
            raise RGCConfigurationError("max_tau_ms must exceed dt_ms")
        if self.initialization_seed < 0:
            raise RGCConfigurationError("initialization_seed must be non-negative")


@dataclass(frozen=True, slots=True)
class RGCState:
    membrane: torch.Tensor
    adaptation: torch.Tensor
    rate: torch.Tensor
    subunit_energy: torch.Tensor


@dataclass(frozen=True, slots=True)
class RGCStepOutput:
    hard_spikes: torch.Tensor
    surrogate_spikes: torch.Tensor
    spike_probability: torch.Tensor
    rates: torch.Tensor
    generator_potential: torch.Tensor


@dataclass(frozen=True, slots=True)
class RGCOutput:
    hard_spikes: torch.Tensor
    surrogate_spikes: torch.Tensor
    spike_probability: torch.Tensor
    rates: torch.Tensor
    generator_potential: torch.Tensor
