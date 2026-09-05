from __future__ import annotations

from collections.abc import Mapping

import torch

from models.mechanistic_retina.contracts import PathwayClamp
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.losses import expected_bernoulli_nll
from training.mechanistic_retina.stages import MechanisticSeedData


def legacy_closure(
    model: MechanisticGraphTemporalRetina,
    data: MechanisticSeedData,
) -> Mapping[str, bool | float]:
    stimulus = data.validation_cones[:1, :32].detach().clone().requires_grad_(True)
    history = torch.ones_like(data.validation_probability[:1, 0, :32])
    clamps = frozenset(
        {
            PathwayClamp.H1,
            PathwayClamp.AMACRINE_LOCAL,
            PathwayClamp.AMACRINE_TRANSIENT,
            PathwayClamp.RGC_HISTORY,
        }
    )
    output = model.forward_sequence(stimulus, observed_counts=history, clamps=clamps)
    zero_history = model.forward_sequence(
        stimulus,
        observed_counts=torch.zeros_like(history),
        clamps=clamps,
    )
    h1_rf = torch.autograd.grad(
        output.h1_surround_contribution.sum(),
        stimulus,
        retain_graph=True,
    )[0]
    ac_rf = torch.autograd.grad(
        output.amacrine_local_current.sum() + output.amacrine_transient_current.sum(),
        stimulus,
    )[0]
    ac_current = max(
        float(output.amacrine_local_current.abs().max()),
        float(output.amacrine_transient_current.abs().max()),
    )
    return {
        "parameterizable_absence": False,
        "h1_amplitude_forced_positive": float(model.gates.h1) > 0,
        "ac_mixture_gates_positive": bool(
            (model.gates.values(frozenset()).ac_local > 0).all()
            and (model.gates.values(frozenset()).ac_transient > 0).all()
        ),
        "history_gain_forced_positive": float(model.rgc.history_gain) > 0,
        "h1_zero_current": float(output.h1_surround_contribution.abs().max()) <= 1e-8,
        "history_zero_effect": float((output.logits - zero_history.logits).abs().max())
        <= 1e-8,
        "ac_local_zero_current": float(output.amacrine_local_current.abs().max()) <= 1e-8,
        "ac_transient_zero_current": float(output.amacrine_transient_current.abs().max())
        <= 1e-8,
        "ac_zero_current": ac_current <= 1e-8,
        "h1_zero_rf": float(h1_rf.abs().max()) <= 1e-8,
        "ac_zero_rf": float(ac_rf.abs().max()) <= 1e-8,
    }


def bias_ce(
    train_probability: torch.Tensor,
    validation_probability: torch.Tensor,
    data: MechanisticSeedData,
) -> float:
    count = data.train_mask[:, 0].sum(dim=(0, 1)).clamp_min(1)
    rate = (train_probability * data.train_mask[:, 0]).sum(dim=(0, 1)) / count
    logits = torch.logit(rate.clamp(1e-6, 1 - 1e-6)).view(1, 1, -1)
    return float(
        expected_bernoulli_nll(
            logits.expand_as(validation_probability),
            validation_probability,
            data.validation_mask[:, 0],
        )
    )


__all__ = ["bias_ce", "legacy_closure"]
