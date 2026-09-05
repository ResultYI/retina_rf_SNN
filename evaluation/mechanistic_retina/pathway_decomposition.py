from __future__ import annotations

from collections.abc import Mapping

import torch

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def pathway_base_rf(
    model: MechanisticGraphTemporalRetina,
    *,
    clamps: frozenset[PathwayClamp] = frozenset(),
) -> Mapping[str, torch.Tensor]:
    sustained, transient, local, ac_transient = model.pathway_base_rfs(clamps=clamps)
    return {
        "BC-sustained": sustained,
        "BC-transient": transient,
        "AC-local": local,
        "AC-transient": ac_transient,
    }


def effective_pathway_rf(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    observed_counts: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    stimulus = cones.detach().clone().requires_grad_(True)
    output = model.forward_sequence(stimulus, observed_counts=observed_counts)
    currents = {
        "BC-sustained": output.bc_sustained_current,
        "BC-transient": output.bc_transient_current,
        "AC-local": output.amacrine_local_current,
        "AC-transient": output.amacrine_transient_current,
    }
    kernels = {name: [] for name in currents}
    for cell in range(output.logits.shape[-1]):
        sensitivity = torch.autograd.grad(
            output.logits[:, -1, cell].sum(),
            output.total_current,
            retain_graph=True,
        )[0].detach()
        for name, current in currents.items():
            gradient = torch.autograd.grad(
                (sensitivity * current).sum(),
                stimulus,
                retain_graph=True,
            )[0]
            kernels[name].append(gradient[:, -model.config.lag_steps :])
    return {
        name: torch.stack(values, dim=1).detach() for name, values in kernels.items()
    }


def pathway_output_sensitivity(
    model: MechanisticGraphTemporalRetina,
    cones: torch.Tensor,
    observed_counts: torch.Tensor,
    *,
    time_index: int,
) -> Mapping[str, torch.Tensor]:
    stimulus = cones.detach().clone().requires_grad_(True)
    output = model.forward_sequence(stimulus, observed_counts=observed_counts)
    currents = {
        "BC-sustained": output.bc_sustained_current,
        "BC-transient": output.bc_transient_current,
        "AC-local": output.amacrine_local_current,
        "AC-transient": output.amacrine_transient_current,
    }
    sensitivities = {}
    for name, current in currents.items():
        cells = []
        for cell in range(output.logits.shape[-1]):
            gradient = torch.autograd.grad(
                output.logits[:, time_index, cell].sum(),
                current,
                retain_graph=True,
            )[0]
            cells.append(gradient[:, time_index, cell])
        sensitivities[name] = torch.stack(cells, dim=1).detach()
    return sensitivities


__all__ = [
    "effective_pathway_rf",
    "pathway_base_rf",
    "pathway_output_sensitivity",
]
