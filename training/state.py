from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from training.config import ExperimentConfig


@dataclass(slots=True)  # noqa: MUTABLE_OK
class EnergyBudgetState:
    """Mutable accumulator for the online energy constraint."""

    reference_energy: float | None = None
    ema_energy: float | None = None
    current_budget: float | None = None
    target_budget: float | None = None
    dual: float = 0.0

    def observe(
        self,
        energy: float,
        optimizer_step: int,
        config: ExperimentConfig,
    ) -> None:
        objective = config.objective
        training = config.training
        self.ema_energy = (
            energy
            if self.ema_energy is None
            else 0.95 * self.ema_energy + 0.05 * energy
        )
        if optimizer_step <= training.reconstruction_bootstrap_steps:
            self.reference_energy = self.ema_energy
            self.current_budget = None
            self.target_budget = None
            self.dual = 0.0
            return
        if self.reference_energy is None:
            self.reference_energy = self.ema_energy
        if self.target_budget is None:
            self.target_budget = max(
                self.reference_energy * objective.energy_budget_ratio,
                1e-12,
            )
        ramp_width = max(
            1,
            training.budget_ramp_end_step
            - training.reconstruction_bootstrap_steps,
        )
        ramp = min(
            1.0,
            (optimizer_step - training.reconstruction_bootstrap_steps)
            / ramp_width,
        )
        self.current_budget = max(
            self.reference_energy
            + ramp * (self.target_budget - self.reference_energy),
            1e-12,
        )
        violation = max(0.0, self.ema_energy / self.current_budget - 1.0)
        self.dual = min(
            objective.dual_max,
            max(0.0, self.dual + objective.dual_lr * violation),
        )


@dataclass(slots=True)  # noqa: MUTABLE_OK
class BootstrapState:
    """Mutable state persisted across bootstrap optimizer steps."""

    view_consistency_base_weight: float | None = None
    view_consistency_calibrated_step: int | None = None
    initial_generator_variance_reference: torch.Tensor | None = None
    persistent_reconstruction: float = 0.0
    view_consistency: float = 0.0
    view_consistency_weight: float = 0.0
    generator_variance_guard: float = 0.0
    generator_variance_retention: float = 1.0


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    metrics: dict[str, float]
    gradient_norm: float
    temporal_gradient_norm: float
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class RepresentationSelectionMetrics:
    rate_source_cv_ratio: float
    generator_source_cv_ratio: float
    fixed_validation_ratio: float


@dataclass(slots=True)  # noqa: MUTABLE_OK
class ValidationState:
    """Mutable best-checkpoint statistics."""

    count: int = 0
    best_reconstruction_mse: float = math.inf
    best_feasible_mse: float = math.inf
    best_representation_rate_ratio: float = 1.0

    def observe(
        self,
        optimizer_step: int,
        reconstruction_mse: float,
        target_energy_ratio: float | None,
        config: ExperimentConfig,
    ) -> tuple[bool, bool]:
        self.count += 1
        best_reconstruction = (
            reconstruction_mse < self.best_reconstruction_mse
        )
        if best_reconstruction:
            self.best_reconstruction_mse = reconstruction_mse
        feasible = (
            optimizer_step >= config.training.budget_ramp_end_step
            and target_energy_ratio is not None
            and math.isfinite(target_energy_ratio)
            and target_energy_ratio
            <= config.evaluation.maximum_energy_budget_ratio
        )
        best_feasible = (
            feasible and reconstruction_mse < self.best_feasible_mse
        )
        if best_feasible:
            self.best_feasible_mse = reconstruction_mse
        return best_reconstruction, best_feasible

    def observe_representation(
        self,
        metrics: RepresentationSelectionMetrics,
    ) -> bool:
        guarded = (
            metrics.generator_source_cv_ratio <= 1.01
            and metrics.fixed_validation_ratio <= 1.02
        )
        improved = (
            guarded
            and metrics.rate_source_cv_ratio
            < self.best_representation_rate_ratio
        )
        if improved:
            self.best_representation_rate_ratio = (
                metrics.rate_source_cv_ratio
            )
        return improved


__all__ = [
    "BootstrapState",
    "EnergyBudgetState",
    "OptimizerStepResult",
    "RepresentationSelectionMetrics",
    "ValidationState",
]
