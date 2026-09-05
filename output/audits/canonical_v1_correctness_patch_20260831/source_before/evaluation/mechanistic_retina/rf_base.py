from __future__ import annotations

import torch

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def base_rf(
    model: MechanisticGraphTemporalRetina,
    *,
    clamps: frozenset[PathwayClamp] = frozenset(),
) -> torch.Tensor:
    pathways = model.pathway_base_rfs(clamps=clamps)
    return torch.stack(pathways).sum(dim=0)


def signed_basis_kernels(
    model: MechanisticGraphTemporalRetina,
    *,
    clamps: frozenset[PathwayClamp] = frozenset(),
) -> torch.Tensor:
    return model.pathway_basis_rfs(clamps=clamps)


__all__ = ["base_rf", "signed_basis_kernels"]
