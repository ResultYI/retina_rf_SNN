from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from evaluation.decoder_diagnostics import (
    DecoderCoverage,
    DecoderFit,
    decode_with_fit,
    decoder_coverage,
    fit_global_decoder,
    fit_tied_decoder_ceiling,
)
from evaluation.reconstruction import ReconstructionMetrics, reconstruction_metrics
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel
from training.augmentation import AugmentedClip


@dataclass(frozen=True, slots=True)
class DecoderExamples:
    rates: torch.Tensor
    target: torch.Tensor
    noisy_input: torch.Tensor


@dataclass(frozen=True, slots=True)
class RepresentationDiagnostics:
    calibrated_decoder: ReconstructionMetrics
    decoder_ceiling: ReconstructionMetrics
    decoder_ceiling_train_mse: float
    ema_alpha: float
    coverage: DecoderCoverage


@torch.no_grad()
def collect_decoder_examples(
    model: RetinaModel,
    clips: Sequence[AugmentedClip],
    supervised_steps: int,
) -> DecoderExamples:
    batch = AugmentedClip.stack(clips)
    model.eval()
    spatial_weights = model.rgc.compute_spatial_weights()
    output, _ = model.forward_sequence(
        batch.noisy_input.float(),
        spatial_weights=spatial_weights,
    )
    return DecoderExamples(
        rates=output.rates[:, -supervised_steps:],
        target=batch.clean_target[:, -supervised_steps:].float(),
        noisy_input=batch.noisy_input[:, -supervised_steps:].float(),
    )


def calibrate_decoder(
    decoder: TiedLocalDecoder,
    examples: DecoderExamples,
    spatial_weights: torch.Tensor,
) -> DecoderFit:
    fit = fit_global_decoder(
        examples.rates,
        examples.target,
        spatial_weights,
        gain_max=decoder.gain_max,
    )
    decoder.initialize(fit.unit_gain, fit.cone_bias)
    return fit


def write_decoder_calibration(output_dir: Path, calibration: DecoderFit) -> None:
    (output_dir / "decoder_calibration.json").write_text(
        json.dumps(
            {
                "unit_gain": calibration.unit_gain.detach().cpu().tolist(),
                "cone_bias": calibration.cone_bias.detach().cpu().tolist(),
                "train_mse": calibration.train_mse,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@torch.no_grad()
def representation_diagnostics(
    decoder: TiedLocalDecoder,
    train_examples: DecoderExamples,
    validation_examples: DecoderExamples,
    spatial_weights: torch.Tensor,
    positions_degs: torch.Tensor,
    train_mean: torch.Tensor,
    ema_alpha: float,
) -> RepresentationDiagnostics:
    actual_prediction = decoder(validation_examples.rates, spatial_weights)
    actual_metrics = reconstruction_metrics(
        actual_prediction,
        validation_examples.target,
        train_mean,
        validation_examples.noisy_input,
        ema_alpha,
    )
    ceiling_fit = fit_tied_decoder_ceiling(
        train_examples.rates,
        train_examples.target,
        spatial_weights,
        gain_max=decoder.gain_max,
    )
    ceiling_prediction = decode_with_fit(
        validation_examples.rates,
        spatial_weights,
        ceiling_fit.unit_gain,
        ceiling_fit.cone_bias,
    )
    ceiling_metrics = reconstruction_metrics(
        ceiling_prediction,
        validation_examples.target,
        train_mean,
        validation_examples.noisy_input,
        ema_alpha,
    )
    return RepresentationDiagnostics(
        calibrated_decoder=actual_metrics,
        decoder_ceiling=ceiling_metrics,
        decoder_ceiling_train_mse=ceiling_fit.train_mse,
        ema_alpha=ema_alpha,
        coverage=decoder_coverage(spatial_weights, positions_degs),
    )


__all__ = [
    "DecoderExamples",
    "RepresentationDiagnostics",
    "calibrate_decoder",
    "collect_decoder_examples",
    "representation_diagnostics",
    "write_decoder_calibration",
]
