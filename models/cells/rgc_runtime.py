from __future__ import annotations

import math

import torch
from torch import nn

from data.geometry import PositionArray
from models.cells.rgc_types import (
    RGCConfig,
    RGCConfigurationError,
    RGCDiagnostics,
    RGCOutput,
    RGCPopulationTensors,
    RGCState,
)
from models.cells.temporal import ordered_taus, raw_ordered_taus


class RGCAdaptiveLIF(nn.Module):
    def __init__(self, config: RGCConfig) -> None:
        super().__init__()
        tau_bounds = torch.tensor(
            (
                (config.adaptation_tau_min_ms, config.adaptation_tau_max_ms),
                (config.membrane_tau_min_ms, config.membrane_tau_max_ms),
            ),
            dtype=torch.float32,
        )
        self.register_buffer("tau_bounds_ms", tau_bounds)
        self.raw_adaptation_tau, self.raw_membrane_tau = raw_ordered_taus(
            torch.tensor((config.adaptation_tau_ms, config.membrane_tau_ms)),
            tau_bounds,
        )
        self.register_buffer(
            "rate_leak",
            torch.tensor(math.exp(-config.dt_ms / config.readout_rate_tau_ms)),
        )
        self.register_buffer(
            "readout_rate_tau_ms",
            torch.tensor(config.readout_rate_tau_ms),
        )
        self._dt_ms = config.dt_ms
        self._threshold = config.threshold
        self._surrogate_slope = config.surrogate_slope
        self._adaptation_strength = config.adaptation_strength

    @property
    def tau_ms(self) -> torch.Tensor:
        return ordered_taus(
            self.raw_adaptation_tau,
            self.raw_membrane_tau,
            self.tau_bounds_ms,
        )

    @property
    def membrane_leak(self) -> torch.Tensor:
        return torch.exp(-self._dt_ms / self.tau_ms[1])

    @property
    def adaptation_leak(self) -> torch.Tensor:
        return torch.exp(-self._dt_ms / self.tau_ms[0])

    def forward(
        self,
        current: torch.Tensor,
        membrane_prev: torch.Tensor,
        adaptation_prev: torch.Tensor,
        rate_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_reset = (
            self.membrane_leak * membrane_prev
            + current
            - adaptation_prev
        )
        hard = (pre_reset >= self._threshold).to(pre_reset.dtype)
        soft = torch.sigmoid(
            self._surrogate_slope * (pre_reset - self._threshold)
        )
        spikes = hard + (soft - soft.detach())
        membrane = pre_reset * (1.0 - spikes.detach())
        adaptation = (
            self.adaptation_leak * adaptation_prev
            + (1.0 - self.adaptation_leak)
            * self._adaptation_strength
            * spikes
        )
        rate = self.rate_leak * rate_prev + (1.0 - self.rate_leak) * spikes
        return membrane, adaptation, rate, spikes


def positions_tensor(name: str, positions: PositionArray) -> torch.Tensor:
    tensor = torch.as_tensor(positions, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != 2:
        raise RGCConfigurationError(f"{name} must have shape [N,2]")
    if not torch.isfinite(tensor).all():
        raise RGCConfigurationError(f"{name} must be finite")
    return tensor


def pool_spatial(weights: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    pooled = torch.sparse.mm(weights, source.reshape(-1, source.shape[-1]).T).T
    return pooled.reshape(source.shape[0], source.shape[1], weights.shape[0])


def mean_neighbor_count(weights: torch.Tensor) -> torch.Tensor:
    rows = weights.coalesce().indices()[0]
    return torch.bincount(rows, minlength=weights.shape[0]).float().mean()


def assert_row_stochastic(name: str, weights: torch.Tensor) -> None:
    row_sums = torch.sparse.sum(weights.coalesce(), dim=1).to_dense()
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        raise RGCConfigurationError(f"{name} rows must sum to one")


def raw_gain(initial: float, maximum: float) -> nn.Parameter:
    return nn.Parameter(torch.logit(torch.tensor(initial / maximum)))


def population_zeros(
    batch_size: int,
    population_counts: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> RGCPopulationTensors:
    midget_count, parasol_count, residual_count = population_counts
    return RGCPopulationTensors(
        torch.zeros(batch_size, 2, midget_count, device=device, dtype=dtype),
        torch.zeros(batch_size, 2, parasol_count, device=device, dtype=dtype),
        torch.zeros(batch_size, 2, residual_count, device=device, dtype=dtype),
    )


def step_populations(
    dynamics: RGCAdaptiveLIF,
    currents: RGCPopulationTensors,
    previous: RGCState,
) -> tuple[RGCOutput, RGCState]:
    steps = tuple(
        dynamics(current, membrane, adaptation, rate)
        for current, membrane, adaptation, rate in zip(
            (currents.midget, currents.parasol, currents.residual),
            (
                previous.membrane.midget,
                previous.membrane.parasol,
                previous.membrane.residual,
            ),
            (
                previous.adaptation.midget,
                previous.adaptation.parasol,
                previous.adaptation.residual,
            ),
            (previous.rate.midget, previous.rate.parasol, previous.rate.residual),
            strict=True,
        )
    )
    midget, parasol, residual = steps
    next_state = RGCState(
        membrane=RGCPopulationTensors(midget[0], parasol[0], residual[0]),
        adaptation=RGCPopulationTensors(midget[1], parasol[1], residual[1]),
        rate=RGCPopulationTensors(midget[2], parasol[2], residual[2]),
    )
    output = RGCOutput(
        spikes=RGCPopulationTensors(midget[3], parasol[3], residual[3]),
        rates=next_state.rate,
    )
    return output, next_state


def state_shapes_are_valid(
    state: RGCState,
    batch_size: int,
    population_counts: tuple[int, int, int],
) -> bool:
    expected = tuple(
        (batch_size, 2, population_count)
        for population_count in population_counts
    )
    groups = (state.membrane, state.adaptation, state.rate)
    return all(
        (group.midget.shape, group.parasol.shape, group.residual.shape) == expected
        for group in groups
    )


def state_is_finite(state: RGCState) -> bool:
    groups = (state.membrane, state.adaptation, state.rate)
    return all(
        torch.isfinite(tensor).all()
        for group in groups
        for tensor in (group.midget, group.parasol, group.residual)
    )


def build_diagnostics(
    output: RGCOutput,
    state: RGCState,
    currents: RGCPopulationTensors,
    g_ag: torch.Tensor,
    parasol_neighbor_count: torch.Tensor,
    residual_neighbor_count: torch.Tensor,
) -> RGCDiagnostics:
    spikes = output.spikes
    rates = output.rates
    adaptation = state.adaptation
    membrane = state.membrane
    current_stats = tuple(
        (
            current.detach().mean(),
            current.detach().min(),
            current.detach().max(),
            (current.detach() < 0).float().mean(),
        )
        for current in (currents.midget, currents.parasol, currents.residual)
    )
    midget_current, parasol_current, residual_current = current_stats
    return RGCDiagnostics(
        rgc_g_ag=g_ag.detach(),
        rgc_midget_current_mean=midget_current[0],
        rgc_midget_current_min=midget_current[1],
        rgc_midget_current_max=midget_current[2],
        rgc_midget_current_negative_fraction=midget_current[3],
        rgc_parasol_current_mean=parasol_current[0],
        rgc_parasol_current_min=parasol_current[1],
        rgc_parasol_current_max=parasol_current[2],
        rgc_parasol_current_negative_fraction=parasol_current[3],
        rgc_residual_current_mean=residual_current[0],
        rgc_residual_current_min=residual_current[1],
        rgc_residual_current_max=residual_current[2],
        rgc_residual_current_negative_fraction=residual_current[3],
        rgc_midget_spike_mean=spikes.midget.detach().mean(),
        rgc_parasol_spike_mean=spikes.parasol.detach().mean(),
        rgc_residual_spike_mean=spikes.residual.detach().mean(),
        rgc_midget_rate_mean=rates.midget.detach().mean(),
        rgc_parasol_rate_mean=rates.parasol.detach().mean(),
        rgc_residual_rate_mean=rates.residual.detach().mean(),
        rgc_midget_adaptation_mean=adaptation.midget.detach().mean(),
        rgc_parasol_adaptation_mean=adaptation.parasol.detach().mean(),
        rgc_residual_adaptation_mean=adaptation.residual.detach().mean(),
        rgc_midget_membrane_max_abs=membrane.midget.detach().abs().max(),
        rgc_parasol_membrane_max_abs=membrane.parasol.detach().abs().max(),
        rgc_residual_membrane_max_abs=membrane.residual.detach().abs().max(),
        rgc_parasol_mean_neighbor_count=parasol_neighbor_count.detach(),
        rgc_residual_mean_neighbor_count=residual_neighbor_count.detach(),
    )
