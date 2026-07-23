from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from loss.retina import RetinaLosses
from models.cells.rgc_types import RGCOutput


def loss_metrics(
    losses: RetinaLosses,
    output: RGCOutput,
) -> dict[str, float]:
    return {
        "loss_total": float(losses.total.detach()),
        "reconstruction": float(losses.reconstruction.detach()),
        "normalized_reconstruction": float(
            losses.normalized_reconstruction.detach()
        ),
        "hard_energy": float(losses.energy.detach()),
        "surrogate_budget_energy": float(losses.budget_energy.detach()),
        "energy_penalty": float(losses.energy_penalty.detach()),
        "energy_violation": float(losses.energy_violation.detach()),
        "wiring": float(losses.wiring.detach()),
        "variance_floor": float(losses.variance_floor.detach()),
        "phenotype_repulsion": float(losses.phenotype_repulsion.detach()),
        "homeostasis": float(losses.homeostasis.detach()),
        "mean_rate": float(output.rates.mean().detach()),
        "hard_active_fraction_on": float(
            (output.hard_spikes[:, :, 0] > 0).float().mean().detach()
        ),
        "hard_active_fraction_off": float(
            (output.hard_spikes[:, :, 1] > 0).float().mean().detach()
        ),
    }


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


__all__ = ["gradient_norm", "loss_metrics"]
