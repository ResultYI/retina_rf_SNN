from __future__ import annotations

# noqa: SIZE_OK - explicit H1 geometry and dynamics stay jointly inspectable

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import nn

from data.geometry import PositionArray, local_gaussian_weights


class H1ConfigurationError(ValueError):
    pass


class H1Diagnostics(TypedDict):
    h1_gain: torch.Tensor
    h1_tau_ms: torch.Tensor
    h1_leak: torch.Tensor
    h1_node_count: torch.Tensor
    h1_node_ratio: torch.Tensor
    h1_cone_to_h1_mean_neighbor_count: torch.Tensor
    h1_h1_to_cone_mean_neighbor_count: torch.Tensor
    h1_input_surround_corr: torch.Tensor
    h1_input_std: torch.Tensor
    h1_modulated_std: torch.Tensor
    h1_modulated_to_input_std_ratio: torch.Tensor
    h1_pooled_spatial_std: torch.Tensor
    h1_pooled_mean_abs: torch.Tensor
    h1_pooled_max_abs: torch.Tensor
    h1_surround_mean_abs: torch.Tensor
    h1_surround_max_abs: torch.Tensor
    h1_state_mean_abs: torch.Tensor
    h1_state_max_abs: torch.Tensor
    h1_modulated_mean_abs: torch.Tensor
    h1_modulated_max_abs: torch.Tensor


@dataclass(frozen=True, slots=True)
class H1HorizontalConfig:
    radius_degs: float
    sigma_degs: float
    feedback_radius_degs: float
    feedback_sigma_degs: float
    h1_spacing_degs: float
    dt_ms: float
    initial_tau_ms: float
    tau_min_ms: float
    tau_max_ms: float
    initial_gain: float
    gain_max: float
    debug_checks: bool = True

    def __post_init__(self) -> None:
        values = (
            self.radius_degs,
            self.sigma_degs,
            self.feedback_radius_degs,
            self.feedback_sigma_degs,
            self.h1_spacing_degs,
            self.dt_ms,
            self.initial_tau_ms,
            self.tau_min_ms,
            self.tau_max_ms,
            self.initial_gain,
            self.gain_max,
        )
        if not all(math.isfinite(value) for value in values):
            raise H1ConfigurationError("H1 parameters must be finite")
        spatial_scales = (
            self.radius_degs,
            self.sigma_degs,
            self.feedback_radius_degs,
            self.feedback_sigma_degs,
            self.h1_spacing_degs,
            self.dt_ms,
        )
        if any(value <= 0 for value in spatial_scales):
            raise H1ConfigurationError("H1 spatial scales and dt_ms must be positive")
        if not self.tau_min_ms < self.initial_tau_ms < self.tau_max_ms:
            raise H1ConfigurationError("initial_tau_ms must lie inside its bounds")
        if not 0 < self.initial_gain < self.gain_max:
            raise H1ConfigurationError("initial_gain must lie between zero and gain_max")


class H1HorizontalNetwork(nn.Module):
    def __init__(
        self,
        cone_positions_degs: PositionArray,
        config: H1HorizontalConfig,
    ) -> None:
        super().__init__()
        cone_positions = torch.as_tensor(cone_positions_degs, dtype=torch.float32)
        h1_positions = _make_h1_grid_positions(
            cone_positions,
            config.h1_spacing_degs,
        )
        h1_positions = _filter_supported_h1_positions(
            h1_positions,
            cone_positions,
            config.radius_degs,
        )
        cone_to_h1 = local_gaussian_weights(
            cone_positions,
            h1_positions,
            config.radius_degs,
            config.sigma_degs,
        ).coalesce()
        h1_to_cone = local_gaussian_weights(
            h1_positions,
            cone_positions_degs,
            config.feedback_radius_degs,
            config.feedback_sigma_degs,
        ).coalesce()
        _assert_row_stochastic("cone_to_h1", cone_to_h1)
        _assert_row_stochastic("h1_to_cone", h1_to_cone)
        self.register_buffer("h1_positions_degs", h1_positions)
        self.register_buffer("cone_to_h1", cone_to_h1)
        self.register_buffer("h1_to_cone", h1_to_cone)
        self.register_buffer(
            "cone_to_h1_mean_neighbor_count",
            _mean_neighbor_count(cone_to_h1),
        )
        self.register_buffer(
            "h1_to_cone_mean_neighbor_count",
            _mean_neighbor_count(h1_to_cone),
        )
        self.raw_gain = nn.Parameter(
            torch.logit(torch.tensor(config.initial_gain / config.gain_max))
        )
        tau_fraction = (config.initial_tau_ms - config.tau_min_ms) / (
            config.tau_max_ms - config.tau_min_ms
        )
        self.raw_tau = nn.Parameter(torch.logit(torch.tensor(tau_fraction)))
        self._gain_max = config.gain_max
        self._tau_min_ms = config.tau_min_ms
        self._tau_max_ms = config.tau_max_ms
        self._dt_ms = config.dt_ms
        self._debug_checks = config.debug_checks

    @property
    def gain(self) -> torch.Tensor:
        return self._gain_max * torch.sigmoid(self.raw_gain)

    @property
    def tau_ms(self) -> torch.Tensor:
        tau_range = self._tau_max_ms - self._tau_min_ms
        return self._tau_min_ms + tau_range * torch.sigmoid(self.raw_tau)

    @property
    def temporal_leak(self) -> torch.Tensor:
        return torch.exp(-self._dt_ms / self.tau_ms)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.cone_to_h1.shape[0],
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        cone_drive: torch.Tensor,
        state: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, H1Diagnostics]
    ):
        cone_count = self.cone_to_h1.shape[1]
        h1_count = self.cone_to_h1.shape[0]
        if cone_drive.ndim != 2 or cone_drive.shape[1] != cone_count:
            raise H1ConfigurationError("cone_drive must have shape [batch,Ncone]")
        if self._debug_checks and not torch.isfinite(cone_drive).all():
            raise H1ConfigurationError("cone_drive contains NaN or inf")
        expected_state_shape = (cone_drive.shape[0], h1_count)
        if state is None:
            state = self.initial_state(
                cone_drive.shape[0],
                cone_drive.device,
                cone_drive.dtype,
            )
        elif state.shape != expected_state_shape:
            raise H1ConfigurationError("H1 state must have shape [batch,NH]")
        if self._debug_checks and not torch.isfinite(state).all():
            raise H1ConfigurationError("H1 state contains NaN or inf")

        pooled_drive = torch.sparse.mm(self.cone_to_h1, cone_drive.T).T
        tau_ms = self.tau_ms
        leak = torch.exp(-self._dt_ms / tau_ms)
        gain = self.gain
        next_state = leak * state + (1.0 - leak) * pooled_drive
        surround = torch.sparse.mm(self.h1_to_cone, next_state.T).T
        modulated_drive = cone_drive - gain * surround
        if not return_diagnostics:
            return modulated_drive, next_state

        cone_detached = cone_drive.detach()
        surround_detached = surround.detach()
        centered_cone = cone_detached - cone_detached.mean()
        centered_surround = surround_detached - surround_detached.mean()
        correlation_denominator = (
            torch.linalg.vector_norm(centered_cone)
            * torch.linalg.vector_norm(centered_surround)
        ).clamp_min(torch.finfo(cone_detached.dtype).eps)
        input_surround_corr = (
            centered_cone * centered_surround
        ).sum() / correlation_denominator
        pooled_detached = pooled_drive.detach()
        pooled_abs = pooled_detached.abs()
        surround_abs = surround_detached.abs()
        state_abs = next_state.detach().abs()
        modulated_detached = modulated_drive.detach()
        modulated_abs = modulated_detached.abs()
        input_std = cone_detached.std(dim=-1, unbiased=False).mean()
        modulated_std = modulated_detached.std(dim=-1, unbiased=False).mean()
        std_ratio = modulated_std / input_std.clamp_min(
            torch.finfo(cone_detached.dtype).eps
        )
        diagnostics = H1Diagnostics(
            h1_gain=gain.detach(),
            h1_tau_ms=tau_ms.detach(),
            h1_leak=leak.detach(),
            h1_node_count=cone_drive.new_tensor(h1_count).detach(),
            h1_node_ratio=cone_drive.new_tensor(h1_count / cone_count).detach(),
            h1_cone_to_h1_mean_neighbor_count=(
                self.cone_to_h1_mean_neighbor_count.detach()
            ),
            h1_h1_to_cone_mean_neighbor_count=(
                self.h1_to_cone_mean_neighbor_count.detach()
            ),
            h1_input_surround_corr=input_surround_corr,
            h1_input_std=input_std,
            h1_modulated_std=modulated_std,
            h1_modulated_to_input_std_ratio=std_ratio,
            h1_pooled_spatial_std=pooled_detached.std(dim=-1, unbiased=False).mean(),
            h1_pooled_mean_abs=pooled_abs.mean(),
            h1_pooled_max_abs=pooled_abs.max(),
            h1_surround_mean_abs=surround_abs.mean(),
            h1_surround_max_abs=surround_abs.max(),
            h1_state_mean_abs=state_abs.mean(),
            h1_state_max_abs=state_abs.max(),
            h1_modulated_mean_abs=modulated_abs.mean(),
            h1_modulated_max_abs=modulated_abs.max(),
        )
        return modulated_drive, next_state, diagnostics


def _make_h1_grid_positions(
    cone_positions_degs: PositionArray,
    spacing_degs: float,
) -> torch.Tensor:
    cone_positions = torch.as_tensor(cone_positions_degs, dtype=torch.float32)
    if (
        cone_positions.ndim != 2
        or cone_positions.shape[0] < 2
        or cone_positions.shape[1] != 2
        or not torch.isfinite(cone_positions).all()
    ):
        raise H1ConfigurationError(
            "cone_positions_degs must be finite with shape [Ncone,2] and Ncone >= 2"
        )
    axes = tuple(
        _centered_grid_axis(cone_positions[:, dimension], spacing_degs)
        for dimension in range(2)
    )
    grid_x, grid_y = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)


def _filter_supported_h1_positions(
    h1_positions: torch.Tensor,
    cone_positions: torch.Tensor,
    radius_degs: float,
) -> torch.Tensor:
    distances = torch.cdist(h1_positions, cone_positions)
    filtered = h1_positions[(distances <= radius_degs).any(dim=1)]
    if filtered.shape[0] == 0:
        raise H1ConfigurationError("No H1 nodes have cone support")
    if filtered.shape[0] >= cone_positions.shape[0]:
        raise H1ConfigurationError(
            "Supported H1 nodes must remain fewer than cones"
        )
    return filtered


def _centered_grid_axis(values: torch.Tensor, spacing: float) -> torch.Tensor:
    lower = values.min()
    span = values.max() - lower
    count = max(1, int(torch.floor(span / spacing).item()) + 1)
    start = lower + 0.5 * (span - spacing * (count - 1))
    return start + spacing * torch.arange(
        count,
        device=values.device,
        dtype=values.dtype,
    )


def _assert_row_stochastic(name: str, weights: torch.Tensor) -> None:
    row_sums = torch.sparse.sum(weights, dim=1).to_dense()
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        raise H1ConfigurationError(f"H1 {name} rows must sum to one")


def _mean_neighbor_count(weights: torch.Tensor) -> torch.Tensor:
    rows = weights.indices()[0]
    return torch.bincount(rows, minlength=weights.shape[0]).float().mean()
