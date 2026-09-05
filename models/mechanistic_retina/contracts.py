from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
import math
from typing import Final

import torch

from models.mechanistic_retina.phase1_run_contract import (
    Phase1RunConfig,
    load_phase1_run_config,
)
from models.mechanistic_retina.spatial_contract import CANONICAL_SPATIAL_CONTRACT
from models.mechanistic_retina.causal_contract import CANONICAL_CAUSAL_CONTRACT


MECHANISTIC_MODEL_REVISION: Final = 4


@unique
class PathwayClamp(StrEnum):
    H1 = "no-H1"
    DIRECT_BC_SUSTAINED = "no-direct-BC-sustained"
    DIRECT_BC_TRANSIENT = "no-direct-BC-transient"
    AMACRINE_LOCAL = "no-amacrine-local"
    AMACRINE_TRANSIENT = "no-amacrine-transient"
    RGC_ADAPTATION = "no-RGC-adaptation"
    RGC_HISTORY = "no-RGC-history"


@unique
class ArchitectureMode(StrEnum):
    LEGACY = "legacy"
    MECHANISM_IDENTIFIABLE = "mechanism_identifiable"


class MechanisticConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MechanisticRetinaConfig:
    architecture_mode: ArchitectureMode = ArchitectureMode.LEGACY
    cell_specific_gains: bool = False
    cell_specific_pathway_mixture: bool = False
    lag_steps: int = 16
    dt_ms: float = 5.0
    graph_radius_deg: float = 0.11
    h1_radius_deg: float = 0.18
    shared_subunit_radius_deg: float = 0.08
    h1_tau_ms: float = 50.0
    h1_tau_bounds_ms: tuple[float, float] = (10.0, 200.0)
    h1_delay_ms: float = 5.0
    h1_delay_bounds_ms: tuple[float, float] = (0.0, 20.0)
    h1_amplitude: float = 0.01
    h1_amplitude_bounds: tuple[float, float] = (0.0, 0.2)
    bc_basis_tau_ms: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (35.0, 60.0, 100.0),
        (10.0, 20.0, 35.0),
    )
    bc_basis_tau_bounds_ms: tuple[tuple[float, float], tuple[float, float]] = (
        (20.0, 200.0),
        (5.0, 120.0),
    )
    bc_delay_ms: tuple[float, float] = (10.0, 5.0)
    bc_delay_bounds_ms: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 30.0),
        (0.0, 20.0),
    )
    ac_delay_ms: tuple[float, float] = (15.0, 7.5)
    ac_delay_bounds_ms: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 40.0),
        (0.0, 25.0),
    )
    amacrine_tau_ms: tuple[float, float] = (35.0, 18.0)
    amacrine_tau_bounds_ms: tuple[tuple[float, float], tuple[float, float]] = (
        (20.0, 250.0),
        (15.0, 180.0),
    )
    divisive_tau_ms: float = 40.0
    divisive_gain: float = 0.01
    membrane_tau_ms: float = 5.0
    adaptation_tau_ms: float = 80.0
    adaptation_gain: float = 0.01
    history_tau_ms: float = 30.0
    history_gain: float = 0.02
    logit_slope: float = 1.0
    threshold: float = 0.0
    operator_epsilon: float = 0.10
    spatial_contract: str = CANONICAL_SPATIAL_CONTRACT
    causal_contract: str = CANONICAL_CAUSAL_CONTRACT

    def __post_init__(self) -> None:
        if self.causal_contract != CANONICAL_CAUSAL_CONTRACT:
            raise MechanisticConfigError(
                f"unsupported Canonical V1 causal contract: {self.causal_contract}"
            )
        if self.spatial_contract != CANONICAL_SPATIAL_CONTRACT:
            raise MechanisticConfigError(
                f"unsupported Canonical V1 spatial contract: {self.spatial_contract}"
            )
        if self.cell_specific_gains and self.cell_specific_pathway_mixture:
            raise MechanisticConfigError("cell-specific gain modes are mutually exclusive")
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0:
            raise MechanisticConfigError("dt_ms must be positive and finite")
        _validate_tau_values("H1", (self.h1_tau_ms,), (self.h1_tau_bounds_ms,))
        _validate_delay_values(
            "H1 delay", (self.h1_delay_ms,), (self.h1_delay_bounds_ms,)
        )
        _validate_amplitude(
            "H1 amplitude", self.h1_amplitude, self.h1_amplitude_bounds
        )
        _validate_tau_values(
            "BC basis",
            tuple(value for row in self.bc_basis_tau_ms for value in row),
            tuple(
                bound
                for row_bound in self.bc_basis_tau_bounds_ms
                for bound in (row_bound,) * 3
            ),
        )
        _validate_tau_values(
            "AC state", self.amacrine_tau_ms, self.amacrine_tau_bounds_ms
        )
        _validate_delay_values("BC delay", self.bc_delay_ms, self.bc_delay_bounds_ms)
        _validate_delay_values("AC delay", self.ac_delay_ms, self.ac_delay_bounds_ms)
        _validate_tau_order("BC basis", self.bc_basis_tau_ms)
        _validate_tau_order("AC state", self.amacrine_tau_ms)
        _validate_delay_order("BC delay", self.bc_delay_ms)
        _validate_delay_order("AC delay", self.ac_delay_ms)


def _validate_tau_values(
    name: str,
    values: tuple[float, ...],
    bounds: tuple[tuple[float, float], ...],
) -> None:
    _validate_bounded_values(name, values, bounds, positive_lower=True)


def _validate_delay_values(
    name: str,
    values: tuple[float, ...],
    bounds: tuple[tuple[float, float], ...],
) -> None:
    _validate_bounded_values(name, values, bounds, positive_lower=False)


def _validate_amplitude(
    name: str, value: float, bounds: tuple[float, float]
) -> None:
    lower, upper = bounds
    if not all(math.isfinite(item) for item in (value, lower, upper)):
        raise MechanisticConfigError(f"{name} and bounds must be finite")
    if lower < 0 or not lower < value < upper:
        raise MechanisticConfigError(
            f"{name} must lie strictly inside nonnegative bounds"
        )


def _validate_bounded_values(
    name: str,
    values: tuple[float, ...],
    bounds: tuple[tuple[float, float], ...],
    *,
    positive_lower: bool,
) -> None:
    if len(values) != len(bounds):
        raise MechanisticConfigError(f"{name} values and bounds must align")
    for value, (lower, upper) in zip(values, bounds, strict=True):
        if not all(math.isfinite(item) for item in (value, lower, upper)):
            raise MechanisticConfigError(f"{name} values and bounds must be finite")
        lower_is_valid = lower > 0 if positive_lower else lower >= 0
        if not lower_is_valid or not lower < value < upper:
            raise MechanisticConfigError(
                f"{name} temporal value must lie strictly inside bounds"
            )


def _validate_tau_order(
    name: str,
    values: tuple[float, ...] | tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
    _validate_order(name, values, quantity="tau")


def _validate_order(
    name: str,
    values: tuple[float, ...] | tuple[tuple[float, ...], tuple[float, ...]],
    *,
    quantity: str,
) -> None:
    sustained, transient = values
    sustained_values = sustained if isinstance(sustained, tuple) else (sustained,)
    transient_values = transient if isinstance(transient, tuple) else (transient,)
    if len(sustained_values) != len(transient_values) or any(
        slow <= fast
        for slow, fast in zip(sustained_values, transient_values, strict=True)
    ):
        raise MechanisticConfigError(
            f"{name} sustained/local {quantity} must exceed transient {quantity}"
        )


def _validate_delay_order(name: str, values: tuple[float, float]) -> None:
    _validate_order(name, values, quantity="delay")


@dataclass(frozen=True, slots=True)
class MechanisticRetinaOutput:
    cone_graph_drive: torch.Tensor
    h1_state: torch.Tensor
    h1_surround_contribution: torch.Tensor
    on_sustained_state: torch.Tensor
    on_sustained_current: torch.Tensor
    on_transient_state: torch.Tensor
    on_transient_current: torch.Tensor
    off_sustained_state: torch.Tensor
    off_sustained_current: torch.Tensor
    off_transient_state: torch.Tensor
    off_transient_current: torch.Tensor
    bc_sustained_current: torch.Tensor
    bc_transient_current: torch.Tensor
    amacrine_local_state: torch.Tensor
    amacrine_local_current: torch.Tensor
    amacrine_transient_state: torch.Tensor
    amacrine_transient_current: torch.Tensor
    total_current: torch.Tensor
    rgc_divisive_state: torch.Tensor
    rgc_membrane: torch.Tensor
    rgc_adaptation: torch.Tensor
    rgc_history_state: torch.Tensor
    logits: torch.Tensor
    spike_probability: torch.Tensor
    bc_direct_presynaptic: torch.Tensor
    bc_broad_presynaptic: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.cone_graph_drive,
            self.h1_state,
            self.h1_surround_contribution,
            self.on_sustained_state,
            self.on_sustained_current,
            self.on_transient_state,
            self.on_transient_current,
            self.off_sustained_state,
            self.off_sustained_current,
            self.off_transient_state,
            self.off_transient_current,
            self.bc_sustained_current,
            self.bc_transient_current,
            self.amacrine_local_state,
            self.amacrine_local_current,
            self.amacrine_transient_state,
            self.amacrine_transient_current,
            self.total_current,
            self.rgc_divisive_state,
            self.rgc_membrane,
            self.rgc_adaptation,
            self.rgc_history_state,
            self.logits,
            self.spike_probability,
            self.bc_direct_presynaptic,
            self.bc_broad_presynaptic,
        )


__all__ = [
    "ArchitectureMode",
    "MechanisticConfigError",
    "MECHANISTIC_MODEL_REVISION",
    "MechanisticRetinaConfig",
    "MechanisticRetinaOutput",
    "PathwayClamp",
    "Phase1RunConfig",
    "load_phase1_run_config",
]
