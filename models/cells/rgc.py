from __future__ import annotations

import math

import torch
from torch import nn

from data.geometry import PositionArray
from models.cells.rgc_dynamics import (
    RGCDynamicsParameters,
    RGCDynamicsRequest,
    step_rgc_dynamics,
)
from models.cells.rgc_runtime import bounded, raw_from_bounded
from models.cells.rgc_types import (
    RGCConfig,
    RGCConfigurationError,
    RGCState,
    RGCStepOutput,
)


class HeterogeneousRGCPool(nn.Module):
    def __init__(self, cone_positions_degs: PositionArray, config: RGCConfig) -> None:
        super().__init__()
        positions = torch.as_tensor(cone_positions_degs, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise RGCConfigurationError("cone_positions_degs must have shape [Ncone,2]")
        if not torch.isfinite(positions).all() or positions.shape[0] < 1:
            raise RGCConfigurationError("cone positions must be non-empty and finite")

        center_indices = torch.arange(positions.shape[0]).repeat_interleave(
            config.units_per_center
        )
        centers = positions.index_select(0, center_indices)
        distance_sq = torch.cdist(centers, positions).square()
        support_mask = distance_sq <= config.support_radius_degs**2
        support_mask[torch.arange(center_indices.numel()), center_indices] = True

        self.register_buffer("cone_positions_degs", positions)
        self.register_buffer("unit_center_indices", center_indices)
        self.register_buffer("unit_centers_degs", centers)
        self.register_buffer("support_mask", support_mask)
        self.register_buffer("distance_sq_degs", distance_sq)
        self.register_buffer(
            "readout_rate_tau_ms",
            torch.tensor(config.readout_rate_tau_ms, dtype=torch.float32),
        )

        unit_count = center_indices.numel()
        generator = torch.Generator().manual_seed(config.initialization_seed)

        def centered_noise(scale: float) -> torch.Tensor:
            noise = torch.randn(unit_count, generator=generator).reshape(
                -1, config.units_per_center
            )
            noise = noise - noise.mean(dim=1, keepdim=True)
            return scale * noise.flatten()

        self.raw_spatial_sigma = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), config.sigma_initial_degs)
                * (1.0 + centered_noise(0.03)),
                config.sigma_min_degs,
                config.sigma_max_degs,
            )
        )
        self.raw_sustained_mix = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 0.5) + centered_noise(0.03), 0.0, 1.0
            )
        )
        self.raw_membrane_tau = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 20.0) * (1.0 + centered_noise(0.03)),
                config.dt_ms,
                config.max_tau_ms,
            )
        )
        self.raw_adaptation_tau = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 80.0) * (1.0 + centered_noise(0.03)),
                config.dt_ms,
                config.max_tau_ms,
            )
        )
        self.raw_adaptation_gain = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 0.10) * (1.0 + centered_noise(0.03)),
                0.0,
                config.adaptation_gain_max,
            )
        )
        self.raw_amacrine_gain = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 0.05) * (1.0 + centered_noise(0.03)),
                0.0,
                config.amacrine_gain_max,
            )
        )
        self.raw_threshold = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 0.20) * (1.0 + centered_noise(0.03)),
                0.02,
                2.0,
            )
        )
        self.raw_subunit_tau = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 50.0) * (1.0 + centered_noise(0.03)),
                config.dt_ms,
                config.max_tau_ms,
            )
        )
        self.raw_subunit_gain = nn.Parameter(
            raw_from_bounded(
                torch.full((unit_count,), 0.50) * (1.0 + centered_noise(0.03)),
                0.0,
                config.subunit_gain_max,
            )
        )

        self._sigma_bounds = (config.sigma_min_degs, config.sigma_max_degs)
        self._tau_bounds = (config.dt_ms, config.max_tau_ms)
        self._adaptation_gain_max = config.adaptation_gain_max
        self._amacrine_gain_max = config.amacrine_gain_max
        self._subunit_gain_max = config.subunit_gain_max
        self._dt_ms = config.dt_ms
        self._surrogate_slope = config.surrogate_slope
        self._debug_checks = config.debug_checks
        self.register_buffer(
            "initial_phenotype_features", self.phenotype_features().detach().clone()
        )

    @property
    def unit_count(self) -> int:
        return int(self.unit_center_indices.numel())

    @property
    def spatial_sigma(self) -> torch.Tensor:
        return bounded(self.raw_spatial_sigma, *self._sigma_bounds)

    @property
    def sustained_mix(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_sustained_mix)

    @property
    def membrane_tau_ms(self) -> torch.Tensor:
        return bounded(self.raw_membrane_tau, *self._tau_bounds)

    @property
    def adaptation_tau_ms(self) -> torch.Tensor:
        return bounded(self.raw_adaptation_tau, *self._tau_bounds)

    @property
    def adaptation_gain(self) -> torch.Tensor:
        return bounded(self.raw_adaptation_gain, 0.0, self._adaptation_gain_max)

    @property
    def amacrine_gain(self) -> torch.Tensor:
        return bounded(self.raw_amacrine_gain, 0.0, self._amacrine_gain_max)

    @property
    def threshold(self) -> torch.Tensor:
        return bounded(self.raw_threshold, 0.02, 2.0)

    @property
    def subunit_tau_ms(self) -> torch.Tensor:
        return bounded(self.raw_subunit_tau, *self._tau_bounds)

    @property
    def subunit_gain(self) -> torch.Tensor:
        return bounded(self.raw_subunit_gain, 0.0, self._subunit_gain_max)

    @property
    def sigma_bounds(self) -> tuple[float, float]:
        return self._sigma_bounds

    @property
    def tau_bounds(self) -> tuple[float, float]:
        return self._tau_bounds

    @property
    def adaptation_gain_max(self) -> float:
        return self._adaptation_gain_max

    @property
    def amacrine_gain_max(self) -> float:
        return self._amacrine_gain_max

    @property
    def subunit_gain_max(self) -> float:
        return self._subunit_gain_max

    def phenotype_features(self) -> torch.Tensor:
        sigma_min, sigma_max = self._sigma_bounds
        tau_min, tau_max = self._tau_bounds
        return torch.stack(
            (
                (self.spatial_sigma.log() - math.log(sigma_min))
                / (math.log(sigma_max) - math.log(sigma_min)),
                self.sustained_mix,
                (self.membrane_tau_ms.log() - math.log(tau_min))
                / (math.log(tau_max) - math.log(tau_min)),
                (self.adaptation_tau_ms.log() - math.log(tau_min))
                / (math.log(tau_max) - math.log(tau_min)),
                self.adaptation_gain / self._adaptation_gain_max,
            ),
            dim=1,
        )

    def compute_spatial_weights(self) -> torch.Tensor:
        sigma_sq = self.spatial_sigma.square().unsqueeze(1)
        logits = -self.distance_sq_degs / (2.0 * sigma_sq)
        logits = logits.masked_fill(~self.support_mask, -torch.inf)
        return torch.softmax(logits, dim=-1)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RGCState:
        unit_shape = (batch_size, 2, self.unit_count)
        return RGCState(
            membrane=torch.zeros(unit_shape, device=device, dtype=dtype),
            adaptation=torch.zeros(unit_shape, device=device, dtype=dtype),
            rate=torch.zeros(unit_shape, device=device, dtype=dtype),
            subunit_energy=torch.zeros(
                batch_size, 2, 2, self.unit_count, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        bipolar_output: torch.Tensor,
        amacrine_output: torch.Tensor,
        previous: RGCState | None,
        spatial_weights: torch.Tensor,
        *,
        probe_continuous_output: bool = False,
    ) -> tuple[RGCStepOutput, RGCState]:
        del probe_continuous_output
        if previous is None:
            previous = self.initial_state(
                bipolar_output.shape[0], bipolar_output.device, bipolar_output.dtype
            )
        return step_rgc_dynamics(
            RGCDynamicsRequest(
                bipolar_output=bipolar_output,
                amacrine_output=amacrine_output,
                previous=previous,
                spatial_weights=spatial_weights,
                parameters=RGCDynamicsParameters(
                    cone_count=int(self.cone_positions_degs.shape[0]),
                    unit_count=self.unit_count,
                    dt_ms=self._dt_ms,
                    surrogate_slope=self._surrogate_slope,
                    debug_checks=self._debug_checks,
                    subunit_tau_ms=self.subunit_tau_ms,
                    subunit_gain=self.subunit_gain,
                    sustained_mix=self.sustained_mix,
                    amacrine_gain=self.amacrine_gain,
                    membrane_tau_ms=self.membrane_tau_ms,
                    adaptation_gain=self.adaptation_gain,
                    threshold=self.threshold,
                    adaptation_tau_ms=self.adaptation_tau_ms,
                    readout_rate_tau_ms=self.readout_rate_tau_ms,
                ),
            )
        )


__all__ = [
    "HeterogeneousRGCPool",
    "RGCConfig",
    "RGCConfigurationError",
    "RGCState",
    "RGCStepOutput",
]
