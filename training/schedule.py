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
    generator_auxiliary_scale: float


def objective_weights(
    optimizer_step: int,
    config: ExperimentConfig,
) -> ObjectiveWeights:
    training = config.training
    objective = config.objective
    bootstrap = max(1, training.reconstruction_bootstrap_steps)
    auxiliary_horizon = max(
        1,
        training.decoder_freeze_steps or training.reconstruction_bootstrap_steps,
    )
    auxiliary_progress = optimizer_step / auxiliary_horizon
    if auxiliary_progress < 0.6:
        generator_auxiliary_scale = 1.0
    elif auxiliary_progress < 0.8:
        generator_auxiliary_scale = (0.8 - auxiliary_progress) / 0.2
    else:
        generator_auxiliary_scale = 0.0
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
        phenotype_repulsion=0.0,
        homeostasis=objective.homeostasis_weight,
        generator_auxiliary_scale=generator_auxiliary_scale,
    )


__all__ = ["ObjectiveWeights", "objective_weights"]
