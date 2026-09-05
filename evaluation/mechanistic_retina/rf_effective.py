from __future__ import annotations

import torch

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def effective_rf(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    observed_counts: torch.Tensor,
    *,
    clamps: frozenset[PathwayClamp] = frozenset(),
) -> torch.Tensor:
    stimulus = cones.detach().clone().requires_grad_(True)
    logits = model.forward_sequence(
        stimulus,
        observed_counts=observed_counts,
        clamps=clamps,
    ).logits[:, -1]
    kernels = []
    for cell in range(logits.shape[-1]):
        gradient = torch.autograd.grad(
            logits[:, cell].sum(),
            stimulus,
            retain_graph=cell + 1 < logits.shape[-1],
        )[0]
        kernels.append(gradient[:, -model.config.lag_steps :])
    return torch.stack(kernels, dim=1).detach()


__all__ = ["effective_rf"]
