from __future__ import annotations

import torch
from torch import nn

from data.geometry import PositionArray, local_gaussian_weights
from models.cells.amacrine_types import (
    LocalAmacrineConfig,
    LocalAmacrineConfigurationError,
    LocalAmacrineDiagnostics,
)
from models.cells.bipolar_types import BipolarKinetics
from models.cells.temporal import ordered_taus, raw_ordered_taus


class LocalAmacrineLayer(nn.Module):
    def __init__(
        self,
        positions_degs: PositionArray,
        config: LocalAmacrineConfig,
    ) -> None:
        super().__init__()
        spatial_pool = local_gaussian_weights(
            positions_degs, positions_degs, config.radius_degs, config.sigma_degs
        ).coalesce()
        row_sums = torch.sparse.sum(spatial_pool, dim=1).to_dense()
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
            raise LocalAmacrineConfigurationError(
                "Local amacrine spatial_pool rows must sum to one"
            )
        self.register_buffer("spatial_pool", spatial_pool)
        indices = spatial_pool.indices()
        self_weights = spatial_pool.values()[indices[0] == indices[1]]
        self_weight_mean = torch.nan_to_num(self_weights.mean())
        self_weight_max = (
            self_weights.max() if self_weights.numel() else self_weight_mean
        )
        neighbor_counts = torch.bincount(
            indices[0], minlength=spatial_pool.shape[0]
        ).to(spatial_pool.dtype)
        self.register_buffer("self_weight_mean", self_weight_mean)
        self.register_buffer("self_weight_max", self_weight_max)
        self.register_buffer("mean_neighbor_count", neighbor_counts.mean())
        tau_bounds = torch.tensor(
            (
                (config.tau_sustained_min_ms, config.tau_sustained_max_ms),
                (config.tau_transient_min_ms, config.tau_transient_max_ms),
            ),
            dtype=torch.float32,
        )
        self.register_buffer("tau_bounds_ms", tau_bounds)
        self.raw_tau_sustained, self.raw_tau_transient = raw_ordered_taus(
            torch.tensor(
                (
                    config.initial_tau_sustained_ms,
                    config.initial_tau_transient_ms,
                )
            ),
            tau_bounds,
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
        self._g_ba_max = (
            config.g_ba_sustained_max,
            config.g_ba_transient_max,
        )
        self._debug_checks = config.debug_checks

    @property
    def tau_ms(self) -> torch.Tensor:
        return ordered_taus(
            self.raw_tau_sustained,
            self.raw_tau_transient,
            self.tau_bounds_ms,
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
        amacrine_prev: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LocalAmacrineDiagnostics]:
        cone_count = self.spatial_pool.shape[1]
        if (
            bipolar_output.ndim != 4
            or bipolar_output.shape[1:3] != (2, 2)
            or bipolar_output.shape[3] != cone_count
        ):
            raise LocalAmacrineConfigurationError(
                "bipolar_output must have shape [batch,2,2,Ncone]"
            )
        if self._debug_checks and not torch.isfinite(bipolar_output).all():
            raise LocalAmacrineConfigurationError(
                "bipolar_output contains NaN or inf"
            )
        if amacrine_prev is None:
            amacrine_prev = self.initial_state(
                bipolar_output.shape[0],
                bipolar_output.device,
                bipolar_output.dtype,
            )
        elif amacrine_prev.shape != bipolar_output.shape:
            raise LocalAmacrineConfigurationError(
                "Local amacrine previous state must match bipolar_output shape"
            )
        if self._debug_checks and not torch.isfinite(amacrine_prev).all():
            raise LocalAmacrineConfigurationError(
                "Local amacrine previous state contains NaN or inf"
            )

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
            leak * amacrine_prev
            + (1.0 - leak) * g_ba * torch.relu(pooled)
        )
        if not return_diagnostics:
            return next_state

        pooled_detached = pooled.detach()
        state_detached = next_state.detach()
        diagnostics = LocalAmacrineDiagnostics(
            amacrine_tau_ms=tau_ms.detach(),
            amacrine_leak=leak_values.detach(),
            amacrine_g_ba=g_ba_values.detach(),
            amacrine_self_weight_mean=self.self_weight_mean.detach(),
            amacrine_self_weight_max=self.self_weight_max.detach(),
            amacrine_mean_neighbor_count=self.mean_neighbor_count.detach(),
            amacrine_pooled_mean=pooled_detached.mean(),
            amacrine_pooled_mean_abs=pooled_detached.abs().mean(),
            amacrine_pooled_max=pooled_detached.max(),
            amacrine_state_mean=state_detached.mean(),
            amacrine_state_mean_abs=state_detached.abs().mean(),
            amacrine_state_max=state_detached.max(),
            amacrine_silent_fraction=(state_detached <= 0).float().mean(),
            amacrine_on_mean=state_detached[:, 0].mean(),
            amacrine_off_mean=state_detached[:, 1].mean(),
            amacrine_sustained_mean=state_detached[:, :, 0].mean(),
            amacrine_transient_mean=state_detached[:, :, 1].mean(),
        )
        return next_state, diagnostics


def _raw_parameter(initial: float, minimum: float, maximum: float) -> nn.Parameter:
    fraction = (initial - minimum) / (maximum - minimum)
    return nn.Parameter(torch.logit(torch.tensor(fraction)))
