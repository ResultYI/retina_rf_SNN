from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from evaluation import representation_diagnostics, representation_selector
from evaluation.reconstruction import fit_causal_ema_alpha
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel
from training.augmentation import AugmentedClip
from training.config import ExperimentConfig
from training.data import PreparedData
from training.runtime import (
    InitialReference,
    diagnostic_training_clips,
    ensure_initial_reference,
    training_mean,
)
from training.validation_clips import fixed_validation_clips


@dataclass(frozen=True, slots=True)
class ExperimentSetupRequest:
    model: RetinaModel
    decoder: TiedLocalDecoder
    prepared: PreparedData
    config: ExperimentConfig
    device: torch.device
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ExperimentSetup:
    validation_clips: tuple[AugmentedClip, ...]
    calibration_clips: tuple[AugmentedClip, ...]
    train_mean: torch.Tensor
    ema_alpha: float
    initial_diagnostics: representation_diagnostics.RepresentationDiagnostics
    initial_reference: InitialReference
    selector: representation_selector.RepresentationSelector


def initialize_experiment(
    request: ExperimentSetupRequest,
) -> ExperimentSetup:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    validation_clips = fixed_validation_clips(
        request.prepared.validation,
        request.config.data,
        request.config.seed + 10_000,
        request.device,
    )
    calibration_clips = diagnostic_training_clips(
        request.prepared,
        request.config,
        request.device,
    )
    train_examples = representation_diagnostics.collect_decoder_examples(
        request.model,
        calibration_clips,
        request.config.training.supervised_steps,
    )
    validation_examples = representation_diagnostics.collect_decoder_examples(
        request.model,
        validation_clips,
        request.config.training.supervised_steps,
    )
    spatial_weights = request.model.rgc.compute_spatial_weights()
    calibration = representation_diagnostics.calibrate_decoder(
        request.decoder,
        train_examples,
        spatial_weights,
    )
    ema_alpha = fit_causal_ema_alpha(
        train_examples.noisy_input,
        train_examples.target,
    )
    mean = training_mean(request.prepared).to(request.device)
    initial_diagnostics = (
        representation_diagnostics.representation_diagnostics(
            request.decoder,
            request.decoder,
            train_examples,
            validation_examples,
            spatial_weights,
            torch.as_tensor(
                request.prepared.positions_degs,
                device=request.device,
                dtype=spatial_weights.dtype,
            ),
            mean,
            ema_alpha,
        )
    )
    representation_diagnostics.write_decoder_calibration(
        request.output_dir,
        calibration,
    )
    (request.output_dir / "representation_initial.json").write_text(
        json.dumps(asdict(initial_diagnostics), indent=2),
        encoding="utf-8",
    )
    selector = representation_selector.RepresentationSelector(
        representation_selector.RepresentationSelectionRequest(
            model=request.model,
            decoder=request.decoder,
            training_clips=calibration_clips,
            validation_clips=validation_clips,
            supervised_steps=request.config.training.supervised_steps,
        )
    )
    selector.write_baseline(request.output_dir)
    return ExperimentSetup(
        validation_clips=validation_clips,
        calibration_clips=calibration_clips,
        train_mean=mean,
        ema_alpha=ema_alpha,
        initial_diagnostics=initial_diagnostics,
        initial_reference=ensure_initial_reference(
            request.output_dir,
            request.model,
            request.decoder,
            request.config,
        ),
        selector=selector,
    )


__all__ = [
    "ExperimentSetup",
    "ExperimentSetupRequest",
    "initialize_experiment",
]
