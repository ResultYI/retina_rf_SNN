from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn

from data.geometry import PositionArray, local_gaussian_weights
from models.cells.bipolar_types import BipolarKinetics


class A2ConfigurationError(ValueError):
    pass


class A2Diagnostics(TypedDict):
    a2_tau_ms: torch.Tensor
    a2_leak: torch.Tensor
    a2_g_ba: torch.Tensor
    a2_self_weight_mean: torch.Tensor
    a2_self_weight_max: torch.Tensor
    a2_mean_neighbor_count: torch.Tensor
    a2_pooled_mean: torch.Tensor
    a2_pooled_mean_abs: torch.Tensor
    a2_pooled_max: torch.Tensor
    a2_state_mean: torch.Tensor
    a2_state_mean_abs: torch.Tensor
    a2_state_max: torch.Tensor
    a2_silent_fraction: torch.Tensor
    a2_on_mean: torch.Tensor
    a2_off_mean: torch.Tensor
    a2_sustained_mean: torch.Tensor
    a2_transient_mean: torch.Tensor


@dataclass(frozen=True, slots=True)
class A2AmacrineConfig:
    radius_degs: float
    sigma_degs: float
    dt_ms: float
    initial_tau_sustained_ms: float
    tau_sustained_min_ms: float
    tau_sustained_max_ms: float
    initial_tau_transient_ms: float
    tau_transient_min_ms: float
    tau_transient_max_ms: float
    initial_g_ba_sustained: float
    g_ba_sustained_max: float
    initial_g_ba_transient: float
    g_ba_transient_max: float
    debug_checks: bool = True

    def __post_init__(self) -> None:
        values = (
            self.radius_degs,
            self.sigma_degs,
            self.dt_ms,
            self.initial_tau_sustained_ms,
            self.tau_sustained_min_ms,
            self.tau_sustained_max_ms,
            self.initial_tau_transient_ms,
            self.tau_transient_min_ms,
            self.tau_transient_max_ms,
            self.initial_g_ba_sustained,
            self.g_ba_sustained_max,
            self.initial_g_ba_transient,
            self.g_ba_transient_max,
        )
        if not all(math.isfinite(value) for value in values):
            raise A2ConfigurationError("A2 parameters must be finite")
        if self.radius_degs <= 0 or self.sigma_degs <= 0 or self.dt_ms <= 0:
            raise A2ConfigurationError("A2 spatial scales and dt_ms must be positive")
        if not (
            self.tau_sustained_min_ms
            < self.initial_tau_sustained_ms
            < self.tau_sustained_max_ms
        ):
            raise A2ConfigurationError("Sustained tau must lie inside its bounds")
        if not (
            self.tau_transient_min_ms
            < self.initial_tau_transient_ms
            < self.tau_transient_max_ms
        ):
            raise A2ConfigurationError("Transient tau must lie inside its bounds")
        if not 0 < self.initial_g_ba_sustained < self.g_ba_sustained_max:
            raise A2ConfigurationError("Sustained g_BA must lie inside its bounds")
        if not 0 < self.initial_g_ba_transient < self.g_ba_transient_max:
            raise A2ConfigurationError("Transient g_BA must lie inside its bounds")


class A2AmacrineLayer(nn.Module):
    def __init__(
        self,
        positions_degs: PositionArray,
        config: A2AmacrineConfig,
    ) -> None:
        super().__init__()
        spatial_pool = local_gaussian_weights(
            positions_degs, positions_degs, config.radius_degs, config.sigma_degs
        ).coalesce()
        row_sums = torch.sparse.sum(spatial_pool, dim=1).to_dense()
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            raise A2ConfigurationError("A2 spatial_pool rows must sum to one")
        self.register_buffer("spatial_pool", spatial_pool)
        indices = spatial_pool.indices()
        self_weights = spatial_pool.values()[indices[0] == indices[1]]
        self_weight_mean = torch.nan_to_num(self_weights.mean())
        self_weight_max = self_weights.max() if self_weights.numel() else self_weight_mean
        neighbor_counts = torch.bincount(
            indices[0], minlength=spatial_pool.shape[0]
        ).to(spatial_pool.dtype)
        self.register_buffer("self_weight_mean", self_weight_mean)
        self.register_buffer("self_weight_max", self_weight_max)
        self.register_buffer("mean_neighbor_count", neighbor_counts.mean())
        self.raw_tau_sustained = _raw_parameter(
            config.initial_tau_sustained_ms,
            config.tau_sustained_min_ms,
            config.tau_sustained_max_ms,
        )
        self.raw_tau_transient = _raw_parameter(
            config.initial_tau_transient_ms,
            config.tau_transient_min_ms,
            config.tau_transient_max_ms,
        )
        self.raw_g_ba_sustained = _raw_parameter(
            config.initial_g_ba_sustained,
            0.0,
            config.g_ba_sustained_max,
        )
        self.raw_g_ba_transient = _raw_parameter(
            config.initial_g_ba_transient,
            0.0,
            config.g_ba_transient_max,
        )
        self._dt_ms = config.dt_ms
        self._tau_sustained_bounds = (
            config.tau_sustained_min_ms,
            config.tau_sustained_max_ms,
        )
        self._tau_transient_bounds = (
            config.tau_transient_min_ms,
            config.tau_transient_max_ms,
        )
        self._g_ba_max = (
            config.g_ba_sustained_max,
            config.g_ba_transient_max,
        )
        self._debug_checks = config.debug_checks

    @property
    def tau_ms(self) -> torch.Tensor:
        return torch.stack(
            (
                _bounded(self.raw_tau_sustained, self._tau_sustained_bounds),
                _bounded(self.raw_tau_transient, self._tau_transient_bounds),
            )
        )

    @property
    def temporal_leak(self) -> torch.Tensor:
        return torch.exp(-self._dt_ms / self.tau_ms)

    @property
    def g_ba(self) -> torch.Tensor:
        return torch.stack(
            (
                self._g_ba_max[BipolarKinetics.SUSTAINED]
                * torch.sigmoid(self.raw_g_ba_sustained),
                self._g_ba_max[BipolarKinetics.TRANSIENT]
                * torch.sigmoid(self.raw_g_ba_transient),
            )
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            2,
            2,
            self.spatial_pool.shape[0],
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        bipolar_output: torch.Tensor,
        a2_prev: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, A2Diagnostics]:
        cone_count = self.spatial_pool.shape[1]
        if (
            bipolar_output.ndim != 4
            or bipolar_output.shape[1:3] != (2, 2)
            or bipolar_output.shape[3] != cone_count
        ):
            raise A2ConfigurationError(
                "bipolar_output must have shape [batch,2,2,Ncone]"
            )
        if self._debug_checks and not torch.isfinite(bipolar_output).all():
            raise A2ConfigurationError("bipolar_output contains NaN or inf")
        if a2_prev is None:
            a2_prev = self.initial_state(
                bipolar_output.shape[0],
                bipolar_output.device,
                bipolar_output.dtype,
            )
        elif a2_prev.shape != bipolar_output.shape:
            raise A2ConfigurationError(
                "A2 previous state must match bipolar_output shape"
            )
        if self._debug_checks and not torch.isfinite(a2_prev).all():
            raise A2ConfigurationError("A2 previous state contains NaN or inf")

        flat_output = bipolar_output.reshape(-1, cone_count)
        pooled = torch.sparse.mm(self.spatial_pool, flat_output.T).T.reshape_as(
            bipolar_output
        )
        tau_ms = self.tau_ms
        leak_values = torch.exp(-self._dt_ms / tau_ms)
        g_ba_values = self.g_ba
        leak = leak_values.view(1, 1, 2, 1)
        g_ba = g_ba_values.view(1, 1, 2, 1)
        next_state = (
            leak * a2_prev
            + (1.0 - leak) * g_ba * torch.relu(pooled)
        )
        if not return_diagnostics:
            return next_state

        pooled_detached = pooled.detach()
        state_detached = next_state.detach()
        diagnostics = A2Diagnostics(
            a2_tau_ms=tau_ms.detach(),
            a2_leak=leak_values.detach(),
            a2_g_ba=g_ba_values.detach(),
            a2_self_weight_mean=self.self_weight_mean.detach(),
            a2_self_weight_max=self.self_weight_max.detach(),
            a2_mean_neighbor_count=self.mean_neighbor_count.detach(),
            a2_pooled_mean=pooled_detached.mean(),
            a2_pooled_mean_abs=pooled_detached.abs().mean(),
            a2_pooled_max=pooled_detached.max(),
            a2_state_mean=state_detached.mean(),
            a2_state_mean_abs=state_detached.abs().mean(),
            a2_state_max=state_detached.max(),
            a2_silent_fraction=(state_detached <= 0).float().mean(),
            a2_on_mean=state_detached[:, 0].mean(),
            a2_off_mean=state_detached[:, 1].mean(),
            a2_sustained_mean=state_detached[:, :, 0].mean(),
            a2_transient_mean=state_detached[:, :, 1].mean(),
        )
        return next_state, diagnostics


def _raw_parameter(initial: float, minimum: float, maximum: float) -> nn.Parameter:
    fraction = (initial - minimum) / (maximum - minimum)
    return nn.Parameter(torch.logit(torch.tensor(fraction)))


def _bounded(
    raw: torch.Tensor,
    bounds: tuple[float, float],
) -> torch.Tensor:
    minimum, maximum = bounds
    return minimum + (maximum - minimum) * torch.sigmoid(raw)
