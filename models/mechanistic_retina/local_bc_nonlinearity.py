"""Experimental effective local BC subunits for the matched N=1 pilot."""

from __future__ import annotations

from enum import StrEnum
import math
from typing import assert_never

import torch
from torch import nn

from models.mechanistic_retina.bipolar_subunits import BipolarOutput
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina, MechanisticModelError


class LocalBCPlacement(StrEnum):
    PRE_POOL = "pre_pool"
    POST_POOL = "post_pool"


def pool_local_bc(
    states: torch.Tensor, alpha: torch.Tensor, placement: LocalBCPlacement,
    *, pooled_reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply shared leaky rectification before or after summing input locations."""
    pooled = states.sum(dim=-1) if pooled_reference is None else pooled_reference
    match placement:
        case LocalBCPlacement.PRE_POOL:
            negative = torch.where(states < 0, states, 0.0).sum(dim=-1)
            return pooled + (alpha - 1.0) * negative
        case LocalBCPlacement.POST_POOL:
            return torch.where(pooled >= 0, pooled, alpha * pooled)
        case unreachable:
            assert_never(unreachable)


class LocalBCNonlinearRetina(MechanisticGraphTemporalRetina):
    """One learned negative-side slope shared across all four BC branches."""

    def __init__(
        self,
        config: MechanisticRetinaConfig,
        cone_positions: torch.Tensor,
        cell_positions: torch.Tensor,
        cell_types: tuple[str, ...],
        polarities: tuple[str, ...],
        *,
        placement: LocalBCPlacement,
    ) -> None:
        if cell_positions.shape[0] != 1:
            raise MechanisticModelError("local BC pilot requires one fitted biological cell")
        if ArchitectureMode(config.architecture_mode) is not ArchitectureMode.MECHANISM_IDENTIFIABLE:
            raise MechanisticModelError("local BC pilot requires Canonical V1")
        super().__init__(config, cone_positions, cell_positions, cell_types, polarities)
        self.placement = LocalBCPlacement(placement)
        self.theta = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor:
        return self.theta.exp()

    def _local_states(self, modulated_cones: torch.Tensor) -> torch.Tensor:
        features = self.feature_bank.local_features(
            modulated_cones, mixer=self.shared_subunits
        )
        weights = self.bipolar.positive_weights().repeat(1, 2, 1, 1)
        return (features * weights[None, None, :, :, :, :, None]).sum(dim=(-3, -2))

    def local_bc_states(self, cones: torch.Tensor) -> torch.Tensor:
        """Return pre-nonlinearity u [B,T,N,4,C], including the existing H1 path."""
        gates = self.gates.values(frozenset())
        return self._local_states(self.h1(cones, amplitude=gates.h1).modulated_cones)

    def pool_local_states(
        self, local_states: torch.Tensor, pooled_reference: torch.Tensor | None = None
    ) -> torch.Tensor:
        return pool_local_bc(
            local_states, self.alpha, self.placement, pooled_reference=pooled_reference
        )

    def _bipolar_outputs(
        self, cones: torch.Tensor, operators_enabled: bool
    ) -> tuple[BipolarOutput, BipolarOutput]:
        if operators_enabled:
            raise MechanisticModelError("local BC pilot requires disabled pathway operators")
        direct, broad = super()._bipolar_outputs(cones, operators_enabled)
        pooled = torch.stack(
            (direct.sustained, direct.transient, broad.sustained, broad.transient), dim=-1
        )
        # Preserve Canonical's floating-point reduction while retaining the exact
        # identity sum(phi(u)) = sum(u) + (alpha - 1) * sum(u[u < 0]).
        states = self.pool_local_states(self._local_states(cones), pooled)
        return self._output(states[..., :2]), self._output(states[..., 2:])

    def _output(self, states: torch.Tensor) -> BipolarOutput:
        sustained, transient = states.unbind(-1)
        on = self.bipolar.on_mask.view(1, 1, -1)
        off = self.bipolar.off_mask.view(1, 1, -1)
        return BipolarOutput(
            sustained, transient, sustained * on, transient * on,
            sustained * off, transient * off,
        )

    def project_mechanism_parameters(self) -> None:
        super().project_mechanism_parameters()
        with torch.no_grad():
            self.theta.clamp_(math.log(0.05), math.log(2.0))
