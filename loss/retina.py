from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from models.cells.rgc import RGCOutput, RGCPopulationTensors
from models.decoder.local_decoder import LocalDecoderOutput

if TYPE_CHECKING:
    from training.hybrid import RetinaTargets


class RetinaLossError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RetinaLossConfig:
    reconstruction_mse_scale: float = 1.0
    energy_weight: float = 0.10
    homeostasis_weight: float = 1e-3
    homeostasis_rate_min: float = 0.01
    # Engineering spike/bin budget, not a fitted human RGC firing rate.
    midget_spike_budget: float = 0.10
    parasol_spike_budget: float = 0.10

    def __post_init__(self) -> None:
        weights = (self.energy_weight, self.homeostasis_weight)
        if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
            raise RetinaLossError("Loss weights must be finite and non-negative")
        if not (
            math.isfinite(self.reconstruction_mse_scale)
            and self.reconstruction_mse_scale > 0
        ):
            raise RetinaLossError(
                "Reconstruction MSE scale must be finite and positive"
            )
        if not (
            math.isfinite(self.homeostasis_rate_min)
            and math.isfinite(self.midget_spike_budget)
            and math.isfinite(self.parasol_spike_budget)
            and 0 <= self.homeostasis_rate_min
            < min(self.midget_spike_budget, self.parasol_spike_budget)
            <= max(self.midget_spike_budget, self.parasol_spike_budget)
            <= 1
        ):
            raise RetinaLossError("Rate floor and spike budget must lie inside [0,1]")


@dataclass(frozen=True, slots=True)
class RetinaLosses:
    total: torch.Tensor
    reconstruction_current: torch.Tensor
    spike_energy: torch.Tensor
    energy_cost: torch.Tensor
    homeostasis: torch.Tensor

    def detached(self) -> RetinaLosses:
        return RetinaLosses(
            total=self.total.detach(),
            reconstruction_current=self.reconstruction_current.detach(),
            spike_energy=self.spike_energy.detach(),
            energy_cost=self.energy_cost.detach(),
            homeostasis=self.homeostasis.detach(),
        )


class RetinaObjective(nn.Module):
    def __init__(self, config: RetinaLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        reconstruction: LocalDecoderOutput,
        targets: RetinaTargets,
        rgc_history: RGCOutput,
    ) -> RetinaLosses:
        if reconstruction.target_current.shape != targets.target_current.shape:
            raise RetinaLossError("Current reconstruction and target shapes must match")
        if not (
            torch.isfinite(reconstruction.target_current).all()
            and torch.isfinite(targets.target_current).all()
        ):
            raise RetinaLossError("Current reconstruction and targets must be finite")

        rates = rgc_history.rates
        spikes = rgc_history.spikes
        reconstruction_current = F.mse_loss(
            reconstruction.target_current,
            targets.target_current,
        )
        spike_energy, energy_cost = _shared_spike_energy(
            spikes,
            targets.target_current,
            self.config,
        )
        population_rates = torch.stack((rates.midget.mean(), rates.parasol.mean()))
        homeostasis = torch.relu(
            self.config.homeostasis_rate_min - population_rates
        ).square().mean()
        total = (
            reconstruction_current / self.config.reconstruction_mse_scale
            + self.config.energy_weight * energy_cost
            + self.config.homeostasis_weight * homeostasis
        )
        return RetinaLosses(
            total=total,
            reconstruction_current=reconstruction_current,
            spike_energy=spike_energy,
            energy_cost=energy_cost,
            homeostasis=homeostasis,
        )


def _shared_spike_energy(
    spikes: RGCPopulationTensors,
    target_current: torch.Tensor,
    config: RetinaLossConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_current.ndim != 2:
        raise RetinaLossError("Current targets must have shape [batch,Ncone]")
    if spikes.midget.ndim != 4 or spikes.parasol.ndim != 4:
        raise RetinaLossError("Spike history must have shape [batch,time,2,N]")
    batch_size, time_steps = spikes.midget.shape[:2]
    if spikes.parasol.shape[:2] != (batch_size, time_steps):
        raise RetinaLossError("Population spike histories must share batch/time axes")
    if target_current.shape[0] != batch_size:
        raise RetinaLossError("Targets and spike history must share batch size")
    target_count = target_current.shape[1]
    if target_count < 1 or spikes.midget.shape[2] != 2 or spikes.parasol.shape[2] != 2:
        raise RetinaLossError("Spike history must include two ON/OFF polarities")

    per_example_energy = (
        spikes.midget.flatten(start_dim=1).sum(dim=1)
        + spikes.parasol.flatten(start_dim=1).sum(dim=1)
    ) / (time_steps * target_count)
    shared_budget = target_current.new_tensor(
        2.0
        * (
            config.midget_spike_budget * spikes.midget.shape[-1]
            + config.parasol_spike_budget * spikes.parasol.shape[-1]
        )
        / target_count
    )
    spike_energy = per_example_energy.mean()
    energy_cost = (per_example_energy / shared_budget).square().mean()
    return spike_energy, energy_cost
