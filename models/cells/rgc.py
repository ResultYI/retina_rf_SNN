from __future__ import annotations

import torch
from torch import nn

from data.geometry import (
    PositionArray,
    local_gaussian_weights,
    nearest_one_to_one_weights,
)
from models.cells.bipolar_types import BipolarKinetics
from models.cells.rgc_runtime import (
    RGCAdaptiveLIF,
    assert_row_stochastic,
    build_diagnostics,
    mean_neighbor_count,
    pool_spatial,
    population_zeros,
    positions_tensor,
    raw_gain,
    state_is_finite,
    state_shapes_are_valid,
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
        bipolar_positions = positions_tensor(
            "bipolar_positions_degs",
            mosaic.bipolar_positions_degs,
        )
        midget_positions = positions_tensor(
            "midget_positions_degs",
            mosaic.midget_positions_degs,
        )
        parasol_positions = positions_tensor(
            "parasol_positions_degs",
            mosaic.parasol_positions_degs,
        )
        residual_positions = positions_tensor(
            "residual_positions_degs",
            mosaic.residual_positions_degs,
        )
        if (
            midget_positions.shape != bipolar_positions.shape
            or not torch.allclose(midget_positions, bipolar_positions, atol=1e-6)
        ):
            raise RGCConfigurationError(
                "Midget mosaic must align one-to-one with bipolar positions"
            )
        if not (
            residual_positions.shape[0]
            <= parasol_positions.shape[0]
            < midget_positions.shape[0]
        ):
            raise RGCConfigurationError(
                "Expected residual_count <= parasol_count < midget_count"
            )
        self.register_buffer("midget_positions_degs", midget_positions)
        self.register_buffer("parasol_positions_degs", parasol_positions)
        self.register_buffer("residual_positions_degs", residual_positions)

        midget_pool = nearest_one_to_one_weights(
            bipolar_positions,
            midget_positions,
        )
        parasol_pool = local_gaussian_weights(
            bipolar_positions,
            parasol_positions,
            config.parasol_radius_degs,
            config.parasol_sigma_degs,
        )
        residual_pool = local_gaussian_weights(
            bipolar_positions,
            residual_positions,
            config.residual_radius_degs,
            config.residual_sigma_degs,
        ).coalesce()
        midget_pool = midget_pool.coalesce()
        parasol_pool = parasol_pool.coalesce()
        assert_row_stochastic("midget_pool", midget_pool)
        assert_row_stochastic("parasol_pool", parasol_pool)
        assert_row_stochastic("residual_pool", residual_pool)
        self.register_buffer("midget_pool", midget_pool)
        self.register_buffer("parasol_pool", parasol_pool)
        self.register_buffer("residual_pool", residual_pool)
        self._population_counts = (
            midget_pool.shape[0], parasol_pool.shape[0], residual_pool.shape[0]
        )
        self.register_buffer(
            "parasol_mean_neighbor_count",
            mean_neighbor_count(parasol_pool),
        )
        self.register_buffer(
            "residual_mean_neighbor_count",
            mean_neighbor_count(residual_pool),
        )
        self.raw_g_ag_midget = raw_gain(
            config.initial_g_ag_midget,
            config.g_ag_midget_max,
        )
        self.raw_g_ag_parasol = raw_gain(
            config.initial_g_ag_parasol,
            config.g_ag_parasol_max,
        )
        self.raw_g_ag_residual = raw_gain(
            config.initial_g_ag_residual,
            config.g_ag_residual_max,
        )
        self.dynamics = RGCAdaptiveLIF(config)
        self._g_ag_max = (
            config.g_ag_midget_max,
            config.g_ag_parasol_max,
            config.g_ag_residual_max,
        )
        self._residual_drive_scale = config.residual_drive_scale
        self._debug_checks = config.debug_checks

    @property
    def g_ag(self) -> torch.Tensor:
        raw = torch.stack(
            (self.raw_g_ag_midget, self.raw_g_ag_parasol, self.raw_g_ag_residual)
        )
        return raw.new_tensor(self._g_ag_max) * torch.sigmoid(raw)

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
        source_count = self.midget_pool.shape[1]
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
        ):
            raise RGCConfigurationError("RGC previous state shapes are invalid")
        if self._debug_checks and not state_is_finite(rgc_prev):
            raise RGCConfigurationError("RGC previous state contains NaN or inf")

        sustained = bipolar_output[:, :, BipolarKinetics.SUSTAINED]
        transient = bipolar_output[:, :, BipolarKinetics.TRANSIENT]
        a2_sustained = amacrine_output[:, :, BipolarKinetics.SUSTAINED]
        a2_transient = amacrine_output[:, :, BipolarKinetics.TRANSIENT]
        g_ag = self.g_ag
        currents = RGCPopulationTensors(
            midget=pool_spatial(self.midget_pool, sustained)
            - g_ag[0] * pool_spatial(self.midget_pool, a2_sustained),
            parasol=pool_spatial(self.parasol_pool, transient)
            - g_ag[1] * pool_spatial(self.parasol_pool, a2_transient),
            residual=self._residual_drive_scale
            * (
                pool_spatial(self.residual_pool, bipolar_output.mean(dim=2))
                - g_ag[2]
                * pool_spatial(self.residual_pool, amacrine_output.mean(dim=2))
            ),
        )
        midget_step = self.dynamics(
            currents.midget,
            rgc_prev.membrane.midget,
            rgc_prev.adaptation.midget,
            rgc_prev.rate.midget,
        )
        parasol_step = self.dynamics(
            currents.parasol,
            rgc_prev.membrane.parasol,
            rgc_prev.adaptation.parasol,
            rgc_prev.rate.parasol,
        )
        residual_step = self.dynamics(
            currents.residual,
            rgc_prev.membrane.residual,
            rgc_prev.adaptation.residual,
            rgc_prev.rate.residual,
        )
        next_state = RGCState(
            membrane=RGCPopulationTensors(
                midget_step[0],
                parasol_step[0],
                residual_step[0],
            ),
            adaptation=RGCPopulationTensors(
                midget_step[1],
                parasol_step[1],
                residual_step[1],
            ),
            rate=RGCPopulationTensors(
                midget_step[2],
                parasol_step[2],
                residual_step[2],
            ),
        )
        output = RGCOutput(
            spikes=RGCPopulationTensors(
                midget_step[3],
                parasol_step[3],
                residual_step[3],
            ),
            rates=next_state.rate,
        )
        if not return_diagnostics:
            return output, next_state
        diagnostics = build_diagnostics(
            output,
            next_state,
            currents,
            g_ag,
            self.parasol_mean_neighbor_count,
            self.residual_mean_neighbor_count,
        )
        return output, next_state, diagnostics
