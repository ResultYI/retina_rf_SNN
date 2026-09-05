from __future__ import annotations

from typing import Final

import torch

from evaluation.mechanistic_retina.mechanism_teacher_support import (
    set_teacher_parameters,
)
from models.mechanistic_retina.delay_parameters import (
    raw_delay_from_ms,
    raw_ordered_delay_from_ms,
)
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina


_AC_LOCAL_MIXTURE_BY_GROUP: Final = (0.75, 0.60, 0.40, 0.25)


def configure_clean_teacher(model: MechanisticGraphTemporalRetina) -> None:
    set_teacher_parameters(model)
    with torch.no_grad():
        model.h1.raw_delay.copy_(
            raw_delay_from_ms(
                model.h1.raw_delay.new_tensor(9.0),
                model.h1.delay_bounds_ms,
            )
        )
        model.feature_bank.raw_delay.copy_(
            raw_ordered_delay_from_ms(
                model.feature_bank.raw_delay.new_tensor((14.0, 4.0)),
                model.feature_bank.delay_bounds_ms,
            )
        )
        model.amacrine.raw_delay.copy_(
            raw_ordered_delay_from_ms(
                model.amacrine.raw_delay.new_tensor((20.0, 6.0)),
                model.amacrine.delay_bounds_ms,
            )
        )
        model.gates.set_h1_amplitude_(0.164)
        local_mixture = model.gates.ac_local.new_tensor(
            _AC_LOCAL_MIXTURE_BY_GROUP
        )
        model.gates.ac_local.copy_(torch.log(local_mixture))
        model.gates.ac_transient.copy_(torch.log1p(-local_mixture))
        model.gates.history.fill_(0.25)
        model.rgc.response_bias.fill_(-2.0)


__all__ = ["configure_clean_teacher"]
