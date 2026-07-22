from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import TypedDict

import torch


class BipolarConfigurationError(ValueError):
    pass


class BipolarPolarity(IntEnum):
    ON = 0
    OFF = 1


class BipolarKinetics(IntEnum):
    SUSTAINED = 0
    TRANSIENT = 1


class BipolarDiagnostics(TypedDict):
    bipolar_tau_ms: torch.Tensor
    bipolar_leak: torch.Tensor
    bipolar_g_ab: torch.Tensor
    bipolar_pre_mean: torch.Tensor
    bipolar_pre_min: torch.Tensor
    bipolar_pre_max: torch.Tensor
    bipolar_state_mean: torch.Tensor
    bipolar_state_max: torch.Tensor
    bipolar_silent_fraction: torch.Tensor
    bipolar_on_mean: torch.Tensor
    bipolar_off_mean: torch.Tensor
    bipolar_sustained_mean: torch.Tensor
    bipolar_transient_mean: torch.Tensor
    bipolar_transient_baseline_mean: torch.Tensor
    bipolar_transient_drive_mean: torch.Tensor
    bipolar_polarity_gain: torch.Tensor
    bipolar_polarity_threshold: torch.Tensor
    bipolar_rectifier_softness: torch.Tensor


@dataclass(frozen=True, slots=True)
class BipolarState:
    output: torch.Tensor
    transient_baseline: torch.Tensor


@dataclass(frozen=True, slots=True)
class BipolarConfig:
    dt_ms: float
    initial_tau_sustained_ms: float
    tau_sustained_min_ms: float
    tau_sustained_max_ms: float
    initial_tau_transient_ms: float
    tau_transient_min_ms: float
    tau_transient_max_ms: float
    initial_g_ab_sustained: float
    g_ab_sustained_max: float
    initial_g_ab_transient: float
    g_ab_transient_max: float
    initial_polarity_gain_on: float = 1.0
    initial_polarity_gain_off: float = 1.0
    polarity_gain_min: float = 0.25
    polarity_gain_max: float = 4.0
    initial_polarity_threshold_on: float = 0.0
    initial_polarity_threshold_off: float = 0.0
    polarity_threshold_min: float = -1.0
    polarity_threshold_max: float = 1.0
    initial_rectifier_softness: float = 0.05
    rectifier_softness_min: float = 0.01
    rectifier_softness_max: float = 0.50
    debug_checks: bool = True

    def __post_init__(self) -> None:
        values = (
            self.dt_ms,
            self.initial_tau_sustained_ms,
            self.tau_sustained_min_ms,
            self.tau_sustained_max_ms,
            self.initial_tau_transient_ms,
            self.tau_transient_min_ms,
            self.tau_transient_max_ms,
            self.initial_g_ab_sustained,
            self.g_ab_sustained_max,
            self.initial_g_ab_transient,
            self.g_ab_transient_max,
            self.initial_polarity_gain_on,
            self.initial_polarity_gain_off,
            self.polarity_gain_min,
            self.polarity_gain_max,
            self.initial_polarity_threshold_on,
            self.initial_polarity_threshold_off,
            self.polarity_threshold_min,
            self.polarity_threshold_max,
            self.initial_rectifier_softness,
            self.rectifier_softness_min,
            self.rectifier_softness_max,
        )
        if not all(math.isfinite(value) for value in values):
            raise BipolarConfigurationError("Bipolar parameters must be finite")
        if self.dt_ms <= 0:
            raise BipolarConfigurationError("dt_ms must be positive")
        if not (
            self.tau_sustained_min_ms
            < self.initial_tau_sustained_ms
            < self.tau_sustained_max_ms
        ):
            raise BipolarConfigurationError("Sustained tau must lie inside its bounds")
        if not (
            self.tau_transient_min_ms
            < self.initial_tau_transient_ms
            < self.tau_transient_max_ms
        ):
            raise BipolarConfigurationError("Transient tau must lie inside its bounds")
        if self.initial_tau_transient_ms >= self.initial_tau_sustained_ms:
            raise BipolarConfigurationError(
                "Transient tau must be less than sustained tau"
            )
        if not 0 < self.initial_g_ab_sustained < self.g_ab_sustained_max:
            raise BipolarConfigurationError("Sustained g_AB must lie inside its bounds")
        if not 0 < self.initial_g_ab_transient < self.g_ab_transient_max:
            raise BipolarConfigurationError("Transient g_AB must lie inside its bounds")
        if self.g_ab_transient_max <= self.g_ab_sustained_max:
            raise BipolarConfigurationError(
                "Transient g_AB upper bound must exceed sustained"
            )
        if not (
            0 < self.polarity_gain_min
            < min(self.initial_polarity_gain_on, self.initial_polarity_gain_off)
            <= max(self.initial_polarity_gain_on, self.initial_polarity_gain_off)
            < self.polarity_gain_max
        ):
            raise BipolarConfigurationError("Polarity gains must lie inside bounds")
        if not (
            self.polarity_threshold_min
            < min(
                self.initial_polarity_threshold_on,
                self.initial_polarity_threshold_off,
            )
            <= max(
                self.initial_polarity_threshold_on,
                self.initial_polarity_threshold_off,
            )
            < self.polarity_threshold_max
        ):
            raise BipolarConfigurationError("Polarity thresholds must lie inside bounds")
        if not (
            0 < self.rectifier_softness_min
            < self.initial_rectifier_softness
            < self.rectifier_softness_max
        ):
            raise BipolarConfigurationError("Rectifier softness must lie inside bounds")
