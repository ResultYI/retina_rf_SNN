from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.cells.rgc import HeterogeneousRGCPool
from models.cells.rgc_types import RGCOutput


class RetinaLossError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetinaLosses:
    total: torch.Tensor
    reconstruction: torch.Tensor
    normalized_reconstruction: torch.Tensor
    energy: torch.Tensor
    budget_energy: torch.Tensor
    energy_penalty: torch.Tensor
    energy_violation: torch.Tensor
    wiring: torch.Tensor
    variance_floor: torch.Tensor
    phenotype_repulsion: torch.Tensor
    homeostasis: torch.Tensor


class RetinaObjective(nn.Module):
    def __init__(
        self,
        *,
        rho_energy: float,
        variance_floor: float,
        phenotype_temperature: float,
        homeostasis_rate_min: float,
    ) -> None:
        super().__init__()
        self.rho_energy = rho_energy
        self.variance_floor_target = variance_floor
        self.phenotype_temperature = phenotype_temperature
        self.homeostasis_rate_min = homeostasis_rate_min

    def forward(
        self,
        prediction: torch.Tensor,
        clean_target: torch.Tensor,
        rgc_output: RGCOutput,
        rgc: HeterogeneousRGCPool,
        spatial_weights: torch.Tensor,
        *,
        reconstruction_scale: float,
        energy_budget: float | None,
        energy_dual: float,
        energy_weight: float,
        wiring_weight: float,
        diversity_weight: float,
        supervised_steps: int,
    ) -> RetinaLosses:
        if prediction.shape != clean_target.shape or prediction.ndim != 3:
            raise RetinaLossError("prediction and clean_target must match [batch,time,cone]")
        supervised_prediction = prediction[:, -supervised_steps:]
        supervised_target = clean_target[:, -supervised_steps:]
        reconstruction = F.mse_loss(supervised_prediction, supervised_target)
        normalized_reconstruction = reconstruction / reconstruction_scale
        energy = rgc_output.hard_spikes.sum() / (
            rgc_output.hard_spikes.shape[0]
            * rgc_output.hard_spikes.shape[1]
            * clean_target.shape[-1]
        )
        budget_energy = rgc_output.surrogate_spikes.sum() / (
            rgc_output.surrogate_spikes.shape[0]
            * rgc_output.surrogate_spikes.shape[1]
            * clean_target.shape[-1]
        )
        if energy_budget is None:
            energy_violation = budget_energy.new_zeros(())
        else:
            energy_violation = torch.relu(budget_energy / energy_budget - 1.0)
        energy_penalty = (
            energy_dual * energy_violation
            + 0.5 * self.rho_energy * energy_violation.square()
        )
        wiring = self.wiring_cost(rgc, spatial_weights)
        continuous = rgc_output.spike_probability
        unit_std = continuous.std(dim=(0, 1), unbiased=False)
        variance_floor = torch.relu(self.variance_floor_target - unit_std).square().mean()
        phenotype_repulsion = self.phenotype_repulsion(rgc)
        unit_rates = rgc_output.rates.mean(dim=(0, 1))
        homeostasis = torch.relu(self.homeostasis_rate_min - unit_rates).square().mean()
        total = (
            normalized_reconstruction
            + energy_weight * energy_penalty
            + wiring_weight * wiring
            + diversity_weight * (variance_floor + phenotype_repulsion + homeostasis)
        )
        return RetinaLosses(
            total=total,
            reconstruction=reconstruction,
            normalized_reconstruction=normalized_reconstruction,
            energy=energy,
            budget_energy=budget_energy,
            energy_penalty=energy_penalty,
            energy_violation=energy_violation,
            wiring=wiring,
            variance_floor=variance_floor,
            phenotype_repulsion=phenotype_repulsion,
            homeostasis=homeostasis,
        )

    @staticmethod
    def wiring_cost(
        rgc: HeterogeneousRGCPool,
        spatial_weights: torch.Tensor,
    ) -> torch.Tensor:
        radius_sq = rgc.distance_sq_degs.masked_select(rgc.support_mask).max().clamp_min(1e-12)
        return (spatial_weights * rgc.distance_sq_degs / radius_sq).sum(dim=1).mean()

    def phenotype_repulsion(self, rgc: HeterogeneousRGCPool) -> torch.Tensor:
        phenotype = rgc.phenotype_features()
        centers = torch.unique(rgc.unit_center_indices)
        penalties: list[torch.Tensor] = []
        for center in centers:
            members = phenotype[rgc.unit_center_indices == center]
            if members.shape[0] < 2:
                continue
            differences = members.unsqueeze(1) - members.unsqueeze(0)
            distances = differences.square().sum(dim=-1)
            pair_mask = torch.triu(
                torch.ones_like(distances, dtype=torch.bool), diagonal=1
            )
            penalties.append(
                torch.exp(-distances[pair_mask] / self.phenotype_temperature).mean()
            )
        return torch.stack(penalties).mean() if penalties else phenotype.new_zeros(())


__all__ = ["RetinaLossError", "RetinaLosses", "RetinaObjective"]
