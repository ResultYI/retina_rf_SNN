from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch


class LocalAmacrineConfigurationError(ValueError):
    pass


class LocalAmacrineDiagnostics(TypedDict):
    amacrine_tau_ms: torch.Tensor
    amacrine_leak: torch.Tensor
    amacrine_g_ba: torch.Tensor
    amacrine_self_weight_mean: torch.Tensor
    amacrine_self_weight_max: torch.Tensor
    amacrine_mean_neighbor_count: torch.Tensor
    amacrine_pooled_mean: torch.Tensor
    amacrine_pooled_mean_abs: torch.Tensor
    amacrine_pooled_max: torch.Tensor
    amacrine_state_mean: torch.Tensor
    amacrine_state_mean_abs: torch.Tensor
    amacrine_state_max: torch.Tensor
    amacrine_silent_fraction: torch.Tensor
    amacrine_on_mean: torch.Tensor
    amacrine_off_mean: torch.Tensor
    amacrine_sustained_mean: torch.Tensor
    amacrine_transient_mean: torch.Tensor


@dataclass(frozen=True, slots=True)
class LocalAmacrineConfig:
    radius_degs: float
    sigma_degs: float
    dt_ms: float
    initial_tau_sustained_ms: float
    tau_sustained_min_ms: float
    tau_sustained_max_ms: float
    initial_tau_transient_ms: float
    tau_transient_min_ms: float
    tau_transient_max_ms: float
    initial_g_ba_sustained: float
    g_ba_sustained_max: float
    initial_g_ba_transient: float
    g_ba_transient_max: float
    debug_checks: bool = True

    def __post_init__(self) -> None:
        values = (
            self.radius_degs,
            self.sigma_degs,
            self.dt_ms,
            self.initial_tau_sustained_ms,
            self.tau_sustained_min_ms,
            self.tau_sustained_max_ms,
            self.initial_tau_transient_ms,
            self.tau_transient_min_ms,
            self.tau_transient_max_ms,
            self.initial_g_ba_sustained,
            self.g_ba_sustained_max,
            self.initial_g_ba_transient,
            self.g_ba_transient_max,
        )
        if not all(math.isfinite(value) for value in values):
            raise LocalAmacrineConfigurationError(
                "Local amacrine parameters must be finite"
            )
        if self.radius_degs <= 0 or self.sigma_degs <= 0 or self.dt_ms <= 0:
            raise LocalAmacrineConfigurationError(
                "Local amacrine spatial scales and dt_ms must be positive"
            )
        if not (
            self.tau_sustained_min_ms
            < self.initial_tau_sustained_ms
            < self.tau_sustained_max_ms
        ):
            raise LocalAmacrineConfigurationError(
                "Sustained tau must lie inside its bounds"
            )
        if not (
            self.tau_transient_min_ms
            < self.initial_tau_transient_ms
            < self.tau_transient_max_ms
        ):
            raise LocalAmacrineConfigurationError(
                "Transient tau must lie inside its bounds"
            )
        if self.initial_tau_transient_ms >= self.initial_tau_sustained_ms:
            raise LocalAmacrineConfigurationError(
                "Transient tau must be less than sustained tau"
            )
        if not 0 < self.initial_g_ba_sustained < self.g_ba_sustained_max:
            raise LocalAmacrineConfigurationError(
                "Sustained g_BA must lie inside its bounds"
            )
        if not 0 < self.initial_g_ba_transient < self.g_ba_transient_max:
            raise LocalAmacrineConfigurationError(
                "Transient g_BA must lie inside its bounds"
            )
