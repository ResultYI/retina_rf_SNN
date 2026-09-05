from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from typing import assert_never

import torch

from evaluation.mechanistic_retina.mechanism_teacher_support import (
    preregistered_probe_inputs,
)
from models.mechanistic_retina.model import MechanisticGraphTemporalRetina
from training.mechanistic_retina.stages import MechanisticSeedData


@unique
class HeldoutPathway(StrEnum):
    H1 = "H1"
    AC = "AC"


@dataclass(frozen=True, slots=True)
class HeldoutProbe:
    name: str
    preregistered_name: str
    stimulus: torch.Tensor
    history: torch.Tensor
    cell: int


def heldout_probes(
    model: MechanisticGraphTemporalRetina,
    data: MechanisticSeedData,
    pathway: HeldoutPathway,
) -> tuple[HeldoutProbe, ...]:
    match pathway:
        case HeldoutPathway.H1:
            h1 = True
            selected = (
                ("diagnostic-h1-03", "diagnostic probe H1-03"),
                ("diagnostic-h1-05", "diagnostic probe H1-05"),
                ("diagnostic-h1-04", "diagnostic probe H1-04"),
            )
        case HeldoutPathway.AC:
            h1 = False
            selected = (
                ("diagnostic-ac-05", "diagnostic probe AC-05"),
                ("diagnostic-ac-04", "diagnostic probe AC-04"),
                ("diagnostic-ac-03", "diagnostic probe AC-03"),
            )
        case unreachable:
            assert_never(unreachable)
    source = {
        probe.preregistered_name: probe
        for probe in preregistered_probe_inputs(model, data, h1)
    }
    return tuple(
        HeldoutProbe(
            name,
            preregistered,
            source[preregistered].stimulus,
            source[preregistered].history,
            source[preregistered].cell,
        )
        for preregistered, name in selected
    )


__all__ = ["HeldoutPathway", "HeldoutProbe", "heldout_probes"]
