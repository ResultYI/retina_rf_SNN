from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NotRequired, TypedDict

import torch

from data.geometry import PositionArray


class RGCConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RGCMosaic:
    bipolar_positions_degs: PositionArray
    midget_positions_degs: PositionArray
    parasol_positions_degs: PositionArray
    residual_positions_degs: PositionArray


@dataclass(frozen=True, slots=True)
class RGCConfig:
    midget_radius_degs: float
    midget_sigma_degs: float
    parasol_radius_degs: float
    parasol_sigma_degs: float
    residual_radius_degs: float
    residual_sigma_degs: float
    dt_ms: float
    membrane_tau_ms: float
    membrane_tau_min_ms: float
    membrane_tau_max_ms: float
    adaptation_tau_ms: float
    adaptation_tau_min_ms: float
    adaptation_tau_max_ms: float
    readout_rate_tau_ms: float
    threshold: float
    surrogate_slope: float
    adaptation_strength: float
    initial_g_ag_midget: float
    g_ag_midget_max: float
    initial_g_ag_parasol: float
    g_ag_parasol_max: float
    initial_g_ag_residual: float
    g_ag_residual_max: float
    residual_drive_scale: float
    debug_checks: bool = True

    def __post_init__(self) -> None:
        values = (
            self.midget_radius_degs,
            self.midget_sigma_degs,
            self.parasol_radius_degs,
            self.parasol_sigma_degs,
            self.residual_radius_degs,
            self.residual_sigma_degs,
            self.dt_ms,
            self.membrane_tau_ms,
            self.membrane_tau_min_ms,
            self.membrane_tau_max_ms,
            self.adaptation_tau_ms,
            self.adaptation_tau_min_ms,
            self.adaptation_tau_max_ms,
            self.readout_rate_tau_ms,
            self.threshold,
            self.surrogate_slope,
            self.adaptation_strength,
            self.initial_g_ag_midget,
            self.g_ag_midget_max,
            self.initial_g_ag_parasol,
            self.g_ag_parasol_max,
            self.initial_g_ag_residual,
            self.g_ag_residual_max,
            self.residual_drive_scale,
        )
        if not all(math.isfinite(value) for value in values):
            raise RGCConfigurationError("RGC parameters must be finite")
        if min(values[:17]) <= 0:
            raise RGCConfigurationError(
                "RGC spatial and temporal parameters must be positive"
            )
        if not (
            self.membrane_tau_min_ms
            < self.membrane_tau_ms
            < self.membrane_tau_max_ms
        ):
            raise RGCConfigurationError("Membrane tau must lie inside its bounds")
        if not (
            self.adaptation_tau_min_ms
            < self.adaptation_tau_ms
            < self.adaptation_tau_max_ms
        ):
            raise RGCConfigurationError("Adaptation tau must lie inside its bounds")
        if self.membrane_tau_ms >= self.adaptation_tau_ms:
            raise RGCConfigurationError(
                "Membrane tau must be less than adaptation tau"
            )
        if not 0 < self.initial_g_ag_midget < self.g_ag_midget_max:
            raise RGCConfigurationError("Midget g_AG must lie inside its bounds")
        if not 0 < self.initial_g_ag_parasol < self.g_ag_parasol_max:
            raise RGCConfigurationError("Parasol g_AG must lie inside its bounds")
        if not 0 < self.initial_g_ag_residual < self.g_ag_residual_max:
            raise RGCConfigurationError("Residual g_AG must lie inside its bounds")
        if self.g_ag_parasol_max <= self.g_ag_midget_max:
            raise RGCConfigurationError("Parasol g_AG bound must exceed midget")
        if not 0 < self.residual_drive_scale <= 1:
            raise RGCConfigurationError("residual_drive_scale must lie in (0,1]")


@dataclass(frozen=True, slots=True)
class RGCPopulationTensors:
    midget: torch.Tensor
    parasol: torch.Tensor
    residual: torch.Tensor


@dataclass(frozen=True, slots=True)
class RGCState:
    membrane: RGCPopulationTensors
    adaptation: RGCPopulationTensors
    rate: RGCPopulationTensors


@dataclass(frozen=True, slots=True)
class RGCOutput:
    spikes: RGCPopulationTensors
    rates: RGCPopulationTensors


class RGCDiagnostics(TypedDict):
    rgc_g_ag: torch.Tensor
    rgc_kinetic_mix: NotRequired[torch.Tensor]
    rgc_tau_ms: NotRequired[torch.Tensor]
    rgc_readout_rate_tau_ms: NotRequired[torch.Tensor]
    rgc_midget_current_mean: torch.Tensor
    rgc_midget_current_min: torch.Tensor
    rgc_midget_current_max: torch.Tensor
    rgc_midget_current_negative_fraction: torch.Tensor
    rgc_parasol_current_mean: torch.Tensor
    rgc_parasol_current_min: torch.Tensor
    rgc_parasol_current_max: torch.Tensor
    rgc_parasol_current_negative_fraction: torch.Tensor
    rgc_residual_current_mean: torch.Tensor
    rgc_residual_current_min: torch.Tensor
    rgc_residual_current_max: torch.Tensor
    rgc_residual_current_negative_fraction: torch.Tensor
    rgc_midget_spike_mean: torch.Tensor
    rgc_parasol_spike_mean: torch.Tensor
    rgc_residual_spike_mean: torch.Tensor
    rgc_midget_rate_mean: torch.Tensor
    rgc_parasol_rate_mean: torch.Tensor
    rgc_residual_rate_mean: torch.Tensor
    rgc_midget_adaptation_mean: torch.Tensor
    rgc_parasol_adaptation_mean: torch.Tensor
    rgc_residual_adaptation_mean: torch.Tensor
    rgc_midget_membrane_max_abs: torch.Tensor
    rgc_parasol_membrane_max_abs: torch.Tensor
    rgc_residual_membrane_max_abs: torch.Tensor
    rgc_parasol_mean_neighbor_count: torch.Tensor
    rgc_residual_mean_neighbor_count: torch.Tensor
