from __future__ import annotations

import torch
from torch import nn

from data.geometry import PositionArray
from models.cells.bipolar_nonlinearity import (
    bounded,
    raw_pair_parameter,
    raw_parameter,
    smooth_rectify,
)
from models.cells.bipolar_types import (
    BipolarConfig,
    BipolarConfigurationError,
    BipolarDiagnostics,
    BipolarKinetics,
    BipolarPolarity,
    BipolarState,
)
from models.cells.temporal import ordered_taus, raw_ordered_taus


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
        self.raw_g_ab_sustained = raw_parameter(
            config.initial_g_ab_sustained,
            0.0,
            config.g_ab_sustained_max,
        )
        self.raw_g_ab_transient = raw_parameter(
            config.initial_g_ab_transient,
            0.0,
            config.g_ab_transient_max,
        )
        self.raw_polarity_gain = raw_pair_parameter(
            config.initial_polarity_gain_on,
            config.initial_polarity_gain_off,
            config.polarity_gain_min,
            config.polarity_gain_max,
        )
        self.raw_polarity_threshold = raw_pair_parameter(
            config.initial_polarity_threshold_on,
            config.initial_polarity_threshold_off,
            config.polarity_threshold_min,
            config.polarity_threshold_max,
        )
        self.raw_rectifier_softness = raw_parameter(
            config.initial_rectifier_softness,
            config.rectifier_softness_min,
            config.rectifier_softness_max,
        )
        self._dt_ms = config.dt_ms
        self._g_ab_max = (
            config.g_ab_sustained_max,
            config.g_ab_transient_max,
        )
        self._polarity_gain_bounds = (config.polarity_gain_min, config.polarity_gain_max)
        self._polarity_threshold_bounds = (config.polarity_threshold_min, config.polarity_threshold_max)
        self._rectifier_softness_bounds = (
            config.rectifier_softness_min, config.rectifier_softness_max
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
    def g_ab(self) -> torch.Tensor:
        return torch.stack(
            (
                self._g_ab_max[BipolarKinetics.SUSTAINED]
                * torch.sigmoid(self.raw_g_ab_sustained),
                self._g_ab_max[BipolarKinetics.TRANSIENT]
                * torch.sigmoid(self.raw_g_ab_transient),
            )
        )

    @property
    def polarity_gain(self) -> torch.Tensor:
        return bounded(self.raw_polarity_gain, self._polarity_gain_bounds)

    @property
    def polarity_threshold(self) -> torch.Tensor:
        return bounded(
            self.raw_polarity_threshold,
            self._polarity_threshold_bounds,
        )

    @property
    def rectifier_softness(self) -> torch.Tensor:
        return bounded(
            self.raw_rectifier_softness,
            self._rectifier_softness_bounds,
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> BipolarState:
        cone_count = self.private_source_index.shape[0]
        output = torch.zeros(
            (batch_size, 2, 2, cone_count), device=device, dtype=dtype
        )
        baseline = torch.zeros(
            (batch_size, 2, cone_count), device=device, dtype=dtype
        )
        return BipolarState(output=output, transient_baseline=baseline)

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
        if self._debug_checks and not torch.isfinite(modulated_drive).all():
            raise BipolarConfigurationError("modulated_drive must be finite")

        gain = self.polarity_gain
        threshold = self.polarity_threshold
        signed_drive = torch.stack((modulated_drive, -modulated_drive), dim=1)
        polarized_drive = smooth_rectify(
            gain.view(1, 2, 1) * signed_drive - threshold.view(1, 2, 1),
            self.rectifier_softness,
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
        if self._debug_checks and (
            not torch.isfinite(state.output).all()
            or not torch.isfinite(state.transient_baseline).all()
        ):
            raise BipolarConfigurationError("Bipolar state must be finite")
        transient_drive = smooth_rectify(
            private_drive - state.transient_baseline,
            self.rectifier_softness,
        )
        channel_drive = torch.stack((private_drive, transient_drive), dim=2)
        if amacrine_prev is None:
            amacrine_prev = torch.zeros_like(channel_drive)
        elif amacrine_prev.shape != output_shape:
            raise BipolarConfigurationError(
                "Amacrine previous state must have shape [batch,2,2,Ncone]"
            )
        if self._debug_checks and not torch.isfinite(amacrine_prev).all():
            raise BipolarConfigurationError("Amacrine previous state must be finite")

        tau_ms = self.tau_ms
        leak_values = torch.exp(-self._dt_ms / tau_ms)
        g_ab_values = self.g_ab
        leak = leak_values.view(1, 1, 2, 1)
        g_ab = g_ab_values.view(1, 1, 2, 1)
        pre_activation = (
            leak * state.output
            + (1.0 - leak) * (channel_drive - g_ab * amacrine_prev)
        )
        next_output = smooth_rectify(pre_activation, self.rectifier_softness)
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
            bipolar_sustained_mean=state_detached[
                :, :, BipolarKinetics.SUSTAINED
            ].mean(),
            bipolar_transient_mean=state_detached[
                :, :, BipolarKinetics.TRANSIENT
            ].mean(),
            bipolar_transient_baseline_mean=next_baseline.detach().mean(),
            bipolar_transient_drive_mean=transient_drive.detach().mean(),
            bipolar_polarity_gain=gain.detach(),
            bipolar_polarity_threshold=threshold.detach(),
            bipolar_rectifier_softness=self.rectifier_softness.detach(),
        )
        return next_state, diagnostics
