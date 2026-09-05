from __future__ import annotations

from torch import nn
from torch.optim import Adam, Optimizer

from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


def phase1_parameters(
    model: MechanisticGraphTemporalRetina,
) -> tuple[nn.Parameter, ...]:
    gain_parameters = (
        ()
        if model.cell_gains is None
        else model.cell_gains.raw_parameters
    )
    intended = (
        model.h1.raw_tau,
        model.h1.raw_delay,
        model.feature_bank.raw_tau,
        model.feature_bank.raw_delay,
        model.shared_subunits.raw_connections,
        model.bipolar.raw_weights,
        model.amacrine.raw_tau,
        model.amacrine.raw_delay,
        model.rgc.response_bias,
        model.gates.raw_h1_amplitude,
        model.gates.ac_local,
        model.gates.ac_transient,
        model.gates.history,
        *gain_parameters,
    )
    return tuple(parameter for parameter in intended if parameter.requires_grad)


def build_phase1_optimizer(
    model: MechanisticGraphTemporalRetina,
    *,
    learning_rate: float,
) -> Optimizer:
    return Adam(phase1_parameters(model), lr=learning_rate)


__all__ = ["build_phase1_optimizer", "phase1_parameters"]
