from __future__ import annotations

import math
from dataclasses import dataclass

from training.config import ExperimentConfig


@dataclass(slots=True)
class EnergyBudgetState:
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


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    metrics: dict[str, float]
    gradient_norm: float
    temporal_gradient_norm: float
    peak_memory_bytes: int


@dataclass(slots=True)
class ValidationState:
    count: int = 0
    best_reconstruction_mse: float = math.inf
    best_feasible_mse: float = math.inf

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


__all__ = ["EnergyBudgetState", "OptimizerStepResult", "ValidationState"]
