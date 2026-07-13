from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

from models.cells.rgc import RGCOutput, RGCPopulationTensors
from models.decoder.local_decoder import LocalDecoder, LocalDecoderOutput
from training.hybrid import RetinaTargets


@dataclass(frozen=True, slots=True)
class ResidualAblationReport:
    fine_on_mse: float
    fine_off_mse: float
    coarse_on_mse: float
    coarse_off_mse: float
    fine_residual_contribution: float
    coarse_residual_contribution: float


@dataclass(frozen=True, slots=True)
class PopulationAblationReport:
    population: str
    fine_on_mse: float
    fine_off_mse: float
    coarse_on_mse: float
    coarse_off_mse: float
    fine_contribution: float
    coarse_contribution: float


def residual_ablation_report(
    decoder: LocalDecoder,
    rgc_output: RGCOutput,
    targets: RetinaTargets | None = None,
) -> ResidualAblationReport:
    report = population_ablation_report(
        decoder,
        rgc_output,
        "residual",
        targets,
    )
    return ResidualAblationReport(
        fine_on_mse=report.fine_on_mse,
        fine_off_mse=report.fine_off_mse,
        coarse_on_mse=report.coarse_on_mse,
        coarse_off_mse=report.coarse_off_mse,
        fine_residual_contribution=report.fine_contribution,
        coarse_residual_contribution=report.coarse_contribution,
    )


def population_ablation_report(
    decoder: LocalDecoder,
    rgc_output: RGCOutput,
    population: Literal["midget", "parasol", "residual"],
    targets: RetinaTargets | None = None,
) -> PopulationAblationReport:
    if population not in {"midget", "parasol", "residual"}:
        raise ValueError(f"Unsupported RGC population: {population}")
    with torch.no_grad():
        on = decoder(rgc_output)
        off = decoder(_zero_population(rgc_output, population))
    return PopulationAblationReport(
        population=population,
        fine_on_mse=_mse(on.target_fine, None if targets is None else targets.fine),
        fine_off_mse=_mse(off.target_fine, None if targets is None else targets.fine),
        coarse_on_mse=_mse(
            on.target_coarse,
            None if targets is None else targets.coarse,
        ),
        coarse_off_mse=_mse(
            off.target_coarse,
            None if targets is None else targets.coarse,
        ),
        fine_contribution=float((on.target_fine - off.target_fine).abs().mean()),
        coarse_contribution=float(
            (on.target_coarse - off.target_coarse).abs().mean()
        ),
    )


def _zero_population(
    output: RGCOutput,
    population: Literal["midget", "parasol", "residual"],
) -> RGCOutput:
    return RGCOutput(
        spikes=_zero_population_tensors(output.spikes, population),
        rates=_zero_population_tensors(output.rates, population),
    )


def _zero_population_tensors(
    tensors: RGCPopulationTensors,
    population: Literal["midget", "parasol", "residual"],
) -> RGCPopulationTensors:
    return RGCPopulationTensors(
        midget=(
            torch.zeros_like(tensors.midget)
            if population == "midget"
            else tensors.midget
        ),
        parasol=(
            torch.zeros_like(tensors.parasol)
            if population == "parasol"
            else tensors.parasol
        ),
        residual=(
            torch.zeros_like(tensors.residual)
            if population == "residual"
            else tensors.residual
        ),
    )


def _mse(prediction: torch.Tensor, target: torch.Tensor | None) -> float:
    if target is None:
        return float("nan")
    return float(F.mse_loss(prediction, target))
