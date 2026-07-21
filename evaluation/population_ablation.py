from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, assert_never

import torch
from torch.nn import functional as F

from models.cells.rgc import RGCOutput, RGCPopulationTensors
from models.decoder.local_decoder import LocalDecoderOutput
from training.hybrid import RetinaTargets


PopulationName = Literal["midget", "parasol"]


class CurrentDecoder(Protocol):
    def __call__(self, output: RGCOutput) -> LocalDecoderOutput: ...


@dataclass(frozen=True, slots=True)
class PopulationAblationReport:
    population: PopulationName
    current_on_mse: float
    current_off_mse: float
    current_mse_delta: float
    current_contribution_abs: float


def population_ablation_report(
    decoder: CurrentDecoder,
    rgc_output: RGCOutput,
    population: PopulationName,
    targets: RetinaTargets,
) -> PopulationAblationReport:
    with torch.no_grad():
        on = decoder(rgc_output).target_current
        off = decoder(_zero_population(rgc_output, population)).target_current
    on_mse = float(F.mse_loss(on, targets.target_current))
    off_mse = float(F.mse_loss(off, targets.target_current))
    contribution = float((on - off).abs().mean())
    return PopulationAblationReport(
        population=population,
        current_on_mse=on_mse,
        current_off_mse=off_mse,
        current_mse_delta=off_mse - on_mse,
        current_contribution_abs=contribution,
    )


def _zero_population(output: RGCOutput, population: PopulationName) -> RGCOutput:
    return RGCOutput(
        spikes=_zero_population_tensors(output.spikes, population),
        rates=_zero_population_tensors(output.rates, population),
    )


def _zero_population_tensors(
    tensors: RGCPopulationTensors,
    population: PopulationName,
) -> RGCPopulationTensors:
    match population:
        case "midget":
            return RGCPopulationTensors(
                midget=torch.zeros_like(tensors.midget),
                parasol=tensors.parasol,
            )
        case "parasol":
            return RGCPopulationTensors(
                midget=tensors.midget,
                parasol=torch.zeros_like(tensors.parasol),
            )
        case unreachable:
            assert_never(unreachable)
