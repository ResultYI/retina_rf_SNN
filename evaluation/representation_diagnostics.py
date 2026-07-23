from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from evaluation.decoder_diagnostics import (
    DecoderCoverage,
    DecoderFit,
    decoder_coverage,
    fit_global_decoder,
)
from evaluation.reconstruction import ReconstructionMetrics, reconstruction_metrics
from evaluation.representation_probe import fit_probe_prediction, source_mse_by_id
from evaluation.representation_comparison import (
    RepresentationComparison,
    SourceRepresentationDelta,
    compare_representation_diagnostics,
)
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel
from training.augmentation import AugmentedClip


@dataclass(frozen=True, slots=True)
class DecoderExamples:
    rates: torch.Tensor
    generator_potential: torch.Tensor
    target: torch.Tensor
    noisy_input: torch.Tensor
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceDecoderMetrics:
    source_id: str
    current_decoder_mse: float
    fixed_calibrated_decoder_mse: float
    posthoc_tied_decoder_probe_mse: float
    posthoc_generator_probe_mse: float


@dataclass(frozen=True, slots=True)
class RepresentationDiagnostics:
    current_decoder: ReconstructionMetrics
    fixed_calibrated_decoder: ReconstructionMetrics
    posthoc_tied_decoder_probe: ReconstructionMetrics
    posthoc_generator_probe: ReconstructionMetrics
    posthoc_tied_decoder_probe_train_mse: float
    posthoc_tied_decoder_probe_source_cv_mse: float
    posthoc_generator_probe_train_mse: float
    posthoc_generator_probe_source_cv_mse: float
    ema_alpha: float
    coverage: DecoderCoverage
    source_metrics: tuple[SourceDecoderMetrics, ...]


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
        source_ids=tuple(str(clip.metadata["source_id"]) for clip in clips),
        generator_potential=output.generator_potential[:, -supervised_steps:],
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
    current_decoder: TiedLocalDecoder,
    fixed_calibrated_decoder: TiedLocalDecoder,
    train_examples: DecoderExamples,
    validation_examples: DecoderExamples,
    spatial_weights: torch.Tensor,
    positions_degs: torch.Tensor,
    train_mean: torch.Tensor,
    ema_alpha: float,
) -> RepresentationDiagnostics:
    current_prediction = current_decoder(
        validation_examples.rates,
        spatial_weights,
    )
    current_metrics = reconstruction_metrics(
        current_prediction,
        validation_examples.target,
        train_mean,
        validation_examples.noisy_input,
        ema_alpha,
    )
    fixed_prediction = fixed_calibrated_decoder(
        validation_examples.rates,
        spatial_weights,
    )
    fixed_metrics = reconstruction_metrics(
        fixed_prediction,
        validation_examples.target,
        train_mean,
        validation_examples.noisy_input,
        ema_alpha,
    )
    rate_probe = fit_probe_prediction(
        train_examples.rates,
        train_examples.target,
        train_examples.source_ids,
        validation_examples.rates,
        spatial_weights,
        gain_max=current_decoder.gain_max,
        prior_gain=fixed_calibrated_decoder.unit_gain,
    )
    generator_probe = fit_probe_prediction(
        train_examples.generator_potential,
        train_examples.target,
        train_examples.source_ids,
        validation_examples.generator_potential,
        spatial_weights,
        gain_max=current_decoder.gain_max,
        prior_gain=fixed_calibrated_decoder.unit_gain,
    )
    probe_metrics = reconstruction_metrics(
        rate_probe.validation_prediction,
        validation_examples.target,
        train_mean,
        validation_examples.noisy_input,
        ema_alpha,
    )
    generator_metrics = reconstruction_metrics(
        generator_probe.validation_prediction,
        validation_examples.target,
        train_mean,
        validation_examples.noisy_input,
        ema_alpha,
    )
    return RepresentationDiagnostics(
        current_decoder=current_metrics,
        fixed_calibrated_decoder=fixed_metrics,
        posthoc_tied_decoder_probe=probe_metrics,
        posthoc_tied_decoder_probe_train_mse=rate_probe.fit.train_mse,
        posthoc_tied_decoder_probe_source_cv_mse=rate_probe.source_cv_mse,
        ema_alpha=ema_alpha,
        coverage=decoder_coverage(spatial_weights, positions_degs),
        source_metrics=_source_decoder_metrics(
            validation_examples,
            current_prediction,
            fixed_prediction,
            rate_probe.validation_prediction,
            generator_probe.validation_prediction,
        ),
        posthoc_generator_probe=generator_metrics,
        posthoc_generator_probe_train_mse=generator_probe.fit.train_mse,
        posthoc_generator_probe_source_cv_mse=generator_probe.source_cv_mse,
    )


def _source_decoder_metrics(
    examples: DecoderExamples,
    current_prediction: torch.Tensor,
    fixed_prediction: torch.Tensor,
    probe_prediction: torch.Tensor,
    generator_prediction: torch.Tensor,
) -> tuple[SourceDecoderMetrics, ...]:
    current_mses = source_mse_by_id(
        current_prediction, examples.target, examples.source_ids
    )
    fixed_mses = source_mse_by_id(
        fixed_prediction, examples.target, examples.source_ids
    )
    probe_mses = source_mse_by_id(
        probe_prediction, examples.target, examples.source_ids
    )
    generator_mses = source_mse_by_id(
        generator_prediction, examples.target, examples.source_ids
    )
    rows: list[SourceDecoderMetrics] = []
    for source_id in dict.fromkeys(examples.source_ids):
        rows.append(
            SourceDecoderMetrics(
                source_id=source_id,
                current_decoder_mse=current_mses[source_id],
                fixed_calibrated_decoder_mse=fixed_mses[source_id],
                posthoc_tied_decoder_probe_mse=probe_mses[source_id],
                posthoc_generator_probe_mse=generator_mses[source_id],
            )
        )
    return tuple(rows)


__all__ = [
    "DecoderExamples",
    "RepresentationComparison",
    "RepresentationDiagnostics",
    "SourceDecoderMetrics",
    "SourceRepresentationDelta",
    "calibrate_decoder",
    "collect_decoder_examples",
    "compare_representation_diagnostics",
    "representation_diagnostics",
    "write_decoder_calibration",
]
