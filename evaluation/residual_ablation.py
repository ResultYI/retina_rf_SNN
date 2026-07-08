from __future__ import annotations

from dataclasses import dataclass

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


def residual_ablation_report(
    decoder: LocalDecoder,
    rgc_output: RGCOutput,
    targets: RetinaTargets | None = None,
) -> ResidualAblationReport:
    with torch.no_grad():
        on = decoder(rgc_output)
        off = decoder(_zero_residual(rgc_output))
    return ResidualAblationReport(
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
        fine_residual_contribution=float((on.target_fine - off.target_fine).abs().mean()),
        coarse_residual_contribution=float(
            (on.target_coarse - off.target_coarse).abs().mean()
        ),
    )


def _zero_residual(output: RGCOutput) -> RGCOutput:
    return RGCOutput(
        spikes=RGCPopulationTensors(
            midget=output.spikes.midget,
            parasol=output.spikes.parasol,
            residual=torch.zeros_like(output.spikes.residual),
        ),
        rates=RGCPopulationTensors(
            midget=output.rates.midget,
            parasol=output.rates.parasol,
            residual=torch.zeros_like(output.rates.residual),
        ),
    )


def _mse(prediction: torch.Tensor, target: torch.Tensor | None) -> float:
    if target is None:
        return float("nan")
    return float(F.mse_loss(prediction, target))
