from __future__ import annotations

import torch
from torch import nn

from models.cells.rgc_connectivity import build_population_pools
from models.cells.rgc_runtime import (
    RGCAdaptiveLIF,
    build_diagnostics,
    mean_neighbor_count,
    normalized_sparse_pool,
    pool_spatial,
    population_zeros,
    raw_gain,
    state_is_finite,
    state_shapes_are_valid,
    step_populations,
)
from models.cells.rgc_types import (
    RGCConfig,
    RGCConfigurationError,
    RGCDiagnostics,
    RGCMosaic,
    RGCOutput,
    RGCPopulationTensors,
    RGCState,
)


class RGCPopulationLayer(nn.Module):
    def __init__(self, mosaic: RGCMosaic, config: RGCConfig) -> None:
        super().__init__()
        (
            midget_positions,
            parasol_positions,
            midget_pool,
            parasol_pool,
        ) = build_population_pools(mosaic, config)
        self.register_buffer("midget_positions_degs", midget_positions)
        self.register_buffer("parasol_positions_degs", parasol_positions)
        self.register_buffer("_midget_pool_support", midget_pool)
        self.register_buffer("_parasol_pool_support", parasol_pool)
        self.raw_pool_values = nn.ParameterDict(
            {
                "midget": nn.Parameter(midget_pool.values().log()),
                "parasol": nn.Parameter(parasol_pool.values().log()),
            }
        )
        self._population_counts = (midget_pool.shape[0], parasol_pool.shape[0])
        self.register_buffer(
            "parasol_mean_neighbor_count",
            mean_neighbor_count(parasol_pool),
        )
        self.raw_g_ag_midget = raw_gain(config.initial_g_ag_midget, config.g_ag_midget_max)
        self.raw_g_ag_parasol = raw_gain(config.initial_g_ag_parasol, config.g_ag_parasol_max)
        tau_fraction = (
            config.subunit_adaptation_tau_ms - config.subunit_adaptation_tau_min_ms
        ) / (
            config.subunit_adaptation_tau_max_ms
            - config.subunit_adaptation_tau_min_ms
        )
        self.raw_subunit_adaptation_tau = nn.Parameter(
            torch.logit(torch.tensor(tau_fraction))
        )
        self.raw_subunit_gain_midget = raw_gain(
            config.initial_subunit_gain, config.subunit_gain_max
        )
        self.raw_subunit_gain_parasol = raw_gain(
            config.initial_subunit_gain, config.subunit_gain_max
        )
        # Zero raw values initialize the mathematical midpoint 0.75; this is
        # an engineering start point, not a fitted physiological estimate.
        self.raw_midget_sustained_preference = nn.Parameter(torch.zeros(()))
        self.raw_parasol_transient_preference = nn.Parameter(torch.zeros(()))
        self.midget_dynamics = RGCAdaptiveLIF(config)
        self.parasol_dynamics = RGCAdaptiveLIF(config)
        self._g_ag_max = (
            config.g_ag_midget_max,
            config.g_ag_parasol_max,
        )
        self._subunit_tau_bounds_ms = (
            config.subunit_adaptation_tau_min_ms,
            config.subunit_adaptation_tau_max_ms,
        )
        self._subunit_gain_max = config.subunit_gain_max
        self._dt_ms = config.dt_ms
        self._debug_checks = config.debug_checks

    @property
    def g_ag(self) -> torch.Tensor:
        raw = torch.stack((self.raw_g_ag_midget, self.raw_g_ag_parasol))
        return raw.new_tensor(self._g_ag_max) * torch.sigmoid(raw)

    @property
    def kinetic_mix(self) -> torch.Tensor:
        raw = torch.stack((self.raw_midget_sustained_preference, self.raw_parasol_transient_preference))
        eps = torch.finfo(raw.dtype).eps
        dominant = raw.new_tensor(0.5 + eps) + raw.new_tensor(0.5 - 2.0 * eps) * torch.sigmoid(raw)
        return torch.stack(
            (
                torch.stack((dominant[0], 1.0 - dominant[0])),
                torch.stack((1.0 - dominant[1], dominant[1])),
            )
        )

    @property
    def subunit_adaptation_tau_ms(self) -> torch.Tensor:
        lower, upper = self._subunit_tau_bounds_ms
        return lower + (upper - lower) * torch.sigmoid(
            self.raw_subunit_adaptation_tau
        )

    @property
    def subunit_gain(self) -> torch.Tensor:
        raw = torch.stack(
            (self.raw_subunit_gain_midget, self.raw_subunit_gain_parasol)
        )
        return self._subunit_gain_max * torch.sigmoid(raw)

    @property
    def midget_pool(self) -> torch.Tensor:
        return normalized_sparse_pool(
            self._midget_pool_support, self.raw_pool_values["midget"]
        )

    @property
    def parasol_pool(self) -> torch.Tensor:
        return normalized_sparse_pool(
            self._parasol_pool_support, self.raw_pool_values["parasol"]
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RGCState:
        population_counts = self._population_counts
        return RGCState(
            membrane=population_zeros(batch_size, population_counts, device, dtype),
            adaptation=population_zeros(batch_size, population_counts, device, dtype),
            rate=population_zeros(batch_size, population_counts, device, dtype),
            subunit_energy=torch.zeros(
                batch_size,
                2,
                2,
                self._midget_pool_support.shape[1],
                device=device,
                dtype=dtype,
            ),
        )

    def forward(
        self,
        bipolar_output: torch.Tensor,
        amacrine_output: torch.Tensor,
        rgc_prev: RGCState | None = None,
        return_diagnostics: bool = False,
    ) -> (
        tuple[RGCOutput, RGCState]
        | tuple[RGCOutput, RGCState, RGCDiagnostics]
    ):
        midget_pool = self.midget_pool
        parasol_pool = self.parasol_pool
        source_count = midget_pool.shape[1]
        expected_input_shape = (bipolar_output.shape[0], 2, 2, source_count)
        if bipolar_output.shape != expected_input_shape:
            raise RGCConfigurationError(
                "bipolar_output must have shape [batch,2,2,Ncone]"
            )
        if amacrine_output.shape != expected_input_shape:
            raise RGCConfigurationError(
                "amacrine_output must match bipolar_output shape"
            )
        if self._debug_checks and (
            not torch.isfinite(bipolar_output).all()
            or not torch.isfinite(amacrine_output).all()
        ):
            raise RGCConfigurationError("RGC inputs contain NaN or inf")
        if rgc_prev is None:
            rgc_prev = self.initial_state(
                bipolar_output.shape[0],
                bipolar_output.device,
                bipolar_output.dtype,
            )
        elif not state_shapes_are_valid(
            rgc_prev,
            bipolar_output.shape[0],
            self._population_counts,
            source_count,
        ):
            raise RGCConfigurationError("RGC previous state shapes are invalid")
        if self._debug_checks and not state_is_finite(rgc_prev):
            raise RGCConfigurationError("RGC previous state contains NaN or inf")

        subunit_leak = torch.exp(-self._dt_ms / self.subunit_adaptation_tau_ms)
        subunit_energy = subunit_leak * rgc_prev.subunit_energy + (
            1.0 - subunit_leak
        ) * bipolar_output.square()
        subunit_gain = self.subunit_gain
        midget_drive = bipolar_output / (1.0 + subunit_gain[0] * subunit_energy)
        parasol_drive = bipolar_output / (1.0 + subunit_gain[1] * subunit_energy)
        kinetic_mix = self.kinetic_mix
        midget_bipolar = _mix_kinetics(midget_drive, kinetic_mix[0])
        parasol_bipolar = _mix_kinetics(parasol_drive, kinetic_mix[1])
        midget_amacrine = _mix_kinetics(amacrine_output, kinetic_mix[0])
        parasol_amacrine = _mix_kinetics(amacrine_output, kinetic_mix[1])
        g_ag = self.g_ag
        currents = RGCPopulationTensors(
            midget=pool_spatial(midget_pool, midget_bipolar)
            - g_ag[0] * pool_spatial(midget_pool, midget_amacrine),
            parasol=pool_spatial(parasol_pool, parasol_bipolar)
            - g_ag[1] * pool_spatial(parasol_pool, parasol_amacrine),
        )
        output, next_state = step_populations(
            (self.midget_dynamics, self.parasol_dynamics),
            currents,
            rgc_prev,
            subunit_energy,
        )
        if not return_diagnostics:
            return output, next_state
        diagnostics = build_diagnostics(
            output,
            next_state,
            currents,
            g_ag,
            self.parasol_mean_neighbor_count,
            subunit_gain,
            subunit_energy,
        )
        diagnostics["rgc_kinetic_mix"] = kinetic_mix.detach()
        diagnostics["rgc_tau_ms"] = torch.stack(
            (
                self.midget_dynamics.tau_ms,
                self.parasol_dynamics.tau_ms,
            )
        ).detach()
        diagnostics["rgc_readout_rate_tau_ms"] = (
            torch.stack(
                (
                    self.midget_dynamics.readout_rate_tau_ms,
                    self.parasol_dynamics.readout_rate_tau_ms,
                )
            ).detach()
        )
        diagnostics["rgc_subunit_adaptation_tau_ms"] = (
            self.subunit_adaptation_tau_ms.detach()
        )
        return output, next_state, diagnostics


def _mix_kinetics(source: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("k,bpkn->bpn", weights, source)
