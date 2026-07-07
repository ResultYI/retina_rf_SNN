from __future__ import annotations

import torch
from torch import nn

from data.geometry import PositionArray
from models.cells.bipolar_types import (
    BipolarConfig,
    BipolarConfigurationError,
    BipolarDiagnostics,
    BipolarKinetics,
    BipolarPolarity,
    BipolarState,
)


class BipolarLayer(nn.Module):
    def __init__(
        self,
        positions_degs: PositionArray,
        config: BipolarConfig,
    ) -> None:
        super().__init__()
        positions = torch.as_tensor(positions_degs, dtype=torch.float32)
        if (
            positions.ndim != 2
            or positions.shape[1] != 2
            or not torch.isfinite(positions).all()
        ):
            raise BipolarConfigurationError("positions_degs must be finite [Ncone,2]")
        self.register_buffer(
            "private_source_index",
            torch.arange(positions.shape[0], device=positions.device),
        )
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
        self.raw_g_ab_sustained = _raw_parameter(
            config.initial_g_ab_sustained,
            0.0,
            config.g_ab_sustained_max,
        )
        self.raw_g_ab_transient = _raw_parameter(
            config.initial_g_ab_transient,
            0.0,
            config.g_ab_transient_max,
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
        self._g_ab_max = (
            config.g_ab_sustained_max,
            config.g_ab_transient_max,
        )

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
    def g_ab(self) -> torch.Tensor:
        return torch.stack(
            (
                self._g_ab_max[BipolarKinetics.SUSTAINED]
                * torch.sigmoid(self.raw_g_ab_sustained),
                self._g_ab_max[BipolarKinetics.TRANSIENT]
                * torch.sigmoid(self.raw_g_ab_transient),
            )
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> BipolarState:
        cone_count = self.private_source_index.shape[0]
        return BipolarState(
            output=torch.zeros(
                batch_size,
                2,
                2,
                cone_count,
                device=device,
                dtype=dtype,
            ),
            transient_baseline=torch.zeros(
                batch_size,
                2,
                cone_count,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(
        self,
        modulated_drive: torch.Tensor,
        state: BipolarState | None = None,
        amacrine_prev: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> BipolarState | tuple[BipolarState, BipolarDiagnostics]:
        cone_count = self.private_source_index.shape[0]
        if modulated_drive.ndim != 2 or modulated_drive.shape[1] != cone_count:
            raise BipolarConfigurationError(
                "modulated_drive must have shape [batch,Ncone]"
            )

        polarized_drive = torch.stack(
            (torch.relu(modulated_drive), torch.relu(-modulated_drive)),
            dim=1,
        )
        private_drive = polarized_drive.index_select(
            -1,
            self.private_source_index,
        )
        output_shape = (modulated_drive.shape[0], 2, 2, cone_count)
        baseline_shape = (modulated_drive.shape[0], 2, cone_count)
        if state is None:
            state = self.initial_state(
                modulated_drive.shape[0],
                modulated_drive.device,
                modulated_drive.dtype,
            )
        elif (
            state.output.shape != output_shape
            or state.transient_baseline.shape != baseline_shape
        ):
            raise BipolarConfigurationError(
                "Bipolar state output/baseline shapes are invalid"
            )
        transient_drive = torch.relu(private_drive - state.transient_baseline)
        channel_drive = torch.stack((private_drive, transient_drive), dim=2)
        if amacrine_prev is None:
            amacrine_prev = torch.zeros_like(channel_drive)
        elif amacrine_prev.shape != output_shape:
            raise BipolarConfigurationError(
                "Amacrine previous state must have shape [batch,2,2,Ncone]"
            )

        tau_ms = self.tau_ms
        leak_values = torch.exp(-self._dt_ms / tau_ms)
        g_ab_values = self.g_ab
        leak = leak_values.view(1, 1, 2, 1)
        g_ab = g_ab_values.view(1, 1, 2, 1)
        pre_activation = (
            leak * state.output
            + (1.0 - leak) * channel_drive
            - g_ab * amacrine_prev
        )
        next_output = torch.relu(pre_activation)
        baseline_leak = leak_values[BipolarKinetics.SUSTAINED]
        next_baseline = (
            baseline_leak * state.transient_baseline
            + (1.0 - baseline_leak) * private_drive
        )
        next_state = BipolarState(next_output, next_baseline)
        if not return_diagnostics:
            return next_state

        pre_detached = pre_activation.detach()
        state_detached = next_output.detach()
        diagnostics = BipolarDiagnostics(
            bipolar_tau_ms=tau_ms.detach(),
            bipolar_leak=leak_values.detach(),
            bipolar_g_ab=g_ab_values.detach(),
            bipolar_pre_mean=pre_detached.mean(),
            bipolar_pre_min=pre_detached.min(),
            bipolar_pre_max=pre_detached.max(),
            bipolar_state_mean=state_detached.mean(),
            bipolar_state_max=state_detached.max(),
            bipolar_silent_fraction=(state_detached <= 0).float().mean(),
            bipolar_on_mean=state_detached[:, BipolarPolarity.ON].mean(),
            bipolar_off_mean=state_detached[:, BipolarPolarity.OFF].mean(),
            bipolar_sustained_mean=state_detached[:, :, BipolarKinetics.SUSTAINED].mean(),
            bipolar_transient_mean=state_detached[:, :, BipolarKinetics.TRANSIENT].mean(),
            bipolar_transient_baseline_mean=next_baseline.detach().mean(),
            bipolar_transient_drive_mean=transient_drive.detach().mean(),
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
