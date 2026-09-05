from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.state import (
    causal_lowpass,
    decay_from_tau,
    fixed_one_bin_history_state,
)


@dataclass(frozen=True, slots=True)
class RGCStateOutput:
    divisive: torch.Tensor
    membrane: torch.Tensor
    adaptation: torch.Tensor
    history: torch.Tensor
    logits: torch.Tensor
    probability: torch.Tensor


class RGCStateDynamics(nn.Module):
    def __init__(self, config: MechanisticRetinaConfig, cell_count: int) -> None:
        super().__init__()
        self.response_bias = nn.Parameter(torch.full((cell_count,), -2.0))
        self.register_buffer(
            "divisive_decay",
            torch.tensor(decay_from_tau(config.dt_ms, config.divisive_tau_ms)),
        )
        self.register_buffer("divisive_gain", torch.tensor(config.divisive_gain))
        self.register_buffer(
            "membrane_decay",
            torch.tensor(decay_from_tau(config.dt_ms, config.membrane_tau_ms)),
        )
        self.register_buffer(
            "adaptation_decay",
            torch.tensor(decay_from_tau(config.dt_ms, config.adaptation_tau_ms)),
        )
        self.register_buffer("adaptation_gain", torch.tensor(config.adaptation_gain))
        self.register_buffer(
            "history_decay",
            torch.tensor(decay_from_tau(config.dt_ms, config.history_tau_ms)),
        )
        self.register_buffer("history_gain", torch.tensor(config.history_gain))
        self.register_buffer(
            "logit_slope", torch.tensor(config.logit_slope).clamp_min(1e-8)
        )
        self.register_buffer("threshold", torch.full((cell_count,), config.threshold))

    def forward(
        self,
        total_current: torch.Tensor,
        observed_counts: torch.Tensor,
        *,
        adaptation_clamped: bool,
        history_gate: torch.Tensor,
    ) -> RGCStateOutput:
        divisive = causal_lowpass(total_current.abs(), self.divisive_decay)
        normalized = total_current / (1 + self.divisive_gain * divisive)
        membrane = causal_lowpass(normalized, self.membrane_decay)
        adaptation = causal_lowpass(membrane, self.adaptation_decay)
        history = fixed_one_bin_history_state(
            observed_counts, float(self.history_decay)
        )
        adaptation_term = 0 if adaptation_clamped else self.adaptation_gain * adaptation
        history_term = history_gate * self.history_gain * history
        logits = (
            self.logit_slope * (membrane - self.threshold)
            - adaptation_term
            - history_term
            + self.response_bias
        )
        return RGCStateOutput(
            divisive,
            membrane,
            adaptation,
            history,
            logits,
            torch.sigmoid(logits),
        )


__all__ = ["RGCStateDynamics", "RGCStateOutput"]
