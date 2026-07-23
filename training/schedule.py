from __future__ import annotations

from dataclasses import dataclass

from training.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    energy: float
    wiring: float
    variance: float
    phenotype_repulsion: float
    homeostasis: float


def objective_weights(
    optimizer_step: int,
    config: ExperimentConfig,
) -> ObjectiveWeights:
    training = config.training
    objective = config.objective
    bootstrap = max(1, training.reconstruction_bootstrap_steps)
    repulsion_scale = max(0.0, 1.0 - optimizer_step / bootstrap)
    ramp_width = max(
        1,
        training.budget_ramp_end_step - training.reconstruction_bootstrap_steps,
    )
    wiring_scale = min(
        1.0,
        max(
            0.0,
            (optimizer_step - training.reconstruction_bootstrap_steps)
            / ramp_width,
        ),
    )
    return ObjectiveWeights(
        energy=1.0,
        wiring=wiring_scale * objective.wiring_weight,
        variance=objective.variance_weight,
        phenotype_repulsion=(
            repulsion_scale * objective.phenotype_repulsion_weight
        ),
        homeostasis=objective.homeostasis_weight,
    )


__all__ = ["ObjectiveWeights", "objective_weights"]
