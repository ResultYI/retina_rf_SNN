from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from configs.physiology_profiles import human_macaque
from evaluation.reconstruction import ReconstructionMetrics, reconstruction_metrics
from models.cells.rgc_types import RGCConfig, RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel, build_retina_model
from training.augmentation import AugmentedClip, augment_clip
from training.config import ExperimentConfig
from training.data import PreparedData
from training.trainer import RetinaTrainer


class ExperimentRuntimeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidationContext:
    clips: tuple[AugmentedClip, ...]
    train_mean: torch.Tensor
    ema_alpha: float


def build_network(
    config: ExperimentConfig,
    prepared: PreparedData,
    device: torch.device,
) -> tuple[RetinaModel, TiedLocalDecoder]:
    spacing = median_nearest_spacing(prepared.positions_degs)
    model_config = config.model
    profile = human_macaque(
        dt_ms=prepared.dt_ms,
        cone_spacing_deg=spacing,
        eccentricity_deg=prepared.eccentricity_deg,
        debug_checks=model_config.debug_checks,
    )
    rgc_config = RGCConfig(
        units_per_center=model_config.units_per_center,
        support_radius_degs=(
            model_config.support_radius_spacing_multiplier * spacing
        ),
        sigma_min_degs=model_config.sigma_min_spacing_multiplier * spacing,
        sigma_initial_degs=(
            model_config.sigma_initial_spacing_multiplier * spacing
        ),
        sigma_max_degs=model_config.sigma_max_spacing_multiplier * spacing,
        dt_ms=prepared.dt_ms,
        readout_rate_tau_ms=model_config.readout_rate_tau_ms,
        max_tau_ms=model_config.max_tau_ms,
        surrogate_slope=model_config.surrogate_slope,
        adaptation_gain_max=model_config.adaptation_gain_max,
        amacrine_gain_max=model_config.amacrine_gain_max,
        subunit_gain_max=model_config.subunit_gain_max,
        initialization_seed=config.seed,
        debug_checks=model_config.debug_checks,
    )
    model = build_retina_model(
        prepared.positions_degs,
        profile,
        rgc_config,
    ).to(device)
    decoder = TiedLocalDecoder(
        model.rgc.unit_count,
        prepared.positions_degs.shape[0],
        model_config.decoder_gain_max,
    ).to(device)
    return model, decoder


def diagnostic_training_clips(
    prepared: PreparedData,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[AugmentedClip, ...]:
    result: list[AugmentedClip] = []
    generator = torch.Generator().manual_seed(config.seed + 3)
    for clip in prepared.train[:4]:
        augmented = augment_clip(clip, config.data, generator)
        result.append(
            AugmentedClip(
                noisy_input=augmented.noisy_input.unsqueeze(0).to(device),
                clean_target=augmented.clean_target.unsqueeze(0).to(device),
                metadata=augmented.metadata,
            )
        )
    return tuple(result)


def ensure_initial_reference(
    output_dir: Path,
    model: RetinaModel,
    decoder: TiedLocalDecoder,
    config: ExperimentConfig,
) -> dict[str, object]:
    path = output_dir / "initial_reference.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "retina_rf_snn_initial_reference"
        ):
            raise ExperimentRuntimeError("Initial reference schema is incompatible")
        if payload.get("resolved_config") != config.resolved():
            raise ExperimentRuntimeError(
                "Initial reference configuration does not match"
            )
        if not isinstance(payload.get("model_state"), dict) or not isinstance(
            payload.get("decoder_state"),
            dict,
        ):
            raise ExperimentRuntimeError("Initial reference state is missing")
        return payload
    payload: dict[str, object] = {
        "schema": "retina_rf_snn_initial_reference",
        "resolved_config": config.resolved(),
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "decoder_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in decoder.state_dict().items()
        },
    }
    torch.save(payload, path)
    return payload


@torch.no_grad()
def evaluate_validation(
    trainer: RetinaTrainer,
    context: ValidationContext,
    config: ExperimentConfig,
) -> tuple[ReconstructionMetrics, RGCOutput, float | None]:
    trainer.model.eval()
    trainer.decoder.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    noisy_inputs: list[torch.Tensor] = []
    outputs: list[RGCOutput] = []
    energies: list[float] = []
    spatial_weights = trainer.model.rgc.compute_spatial_weights()
    for clip in context.clips:
        losses, output, _ = trainer.forward_clip(
            clip.noisy_input,
            clip.clean_target,
            checkpointed=False,
        )
        supervised = config.training.supervised_steps
        predictions.append(
            trainer.decoder(output.rates, spatial_weights)[:, -supervised:]
        )
        targets.append(clip.clean_target[:, -supervised:])
        noisy_inputs.append(clip.noisy_input[:, -supervised:])
        outputs.append(output)
        energies.append(float(losses.energy))
    metrics = reconstruction_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        context.train_mean,
        torch.cat(noisy_inputs),
        context.ema_alpha,
    )
    target_budget = trainer.energy_state.target_budget
    target_energy_ratio = (
        float(np.mean(energies)) / target_budget
        if target_budget is not None
        else None
    )
    return metrics, concatenate_outputs(outputs), target_energy_ratio


def training_mean(prepared: PreparedData) -> torch.Tensor:
    return torch.cat([clip.clean for clip in prepared.train]).mean(dim=0)


def concatenate_outputs(outputs: Sequence[RGCOutput]) -> RGCOutput:
    if not outputs:
        raise ExperimentRuntimeError("Validation requires at least one output")
    return RGCOutput(
        hard_spikes=torch.cat([output.hard_spikes for output in outputs]),
        surrogate_spikes=torch.cat(
            [output.surrogate_spikes for output in outputs]
        ),
        spike_probability=torch.cat(
            [output.spike_probability for output in outputs]
        ),
        rates=torch.cat([output.rates for output in outputs]),
        generator_potential=torch.cat(
            [output.generator_potential for output in outputs]
        ),
    )


def median_nearest_spacing(positions_degs: np.ndarray) -> float:
    positions = torch.as_tensor(positions_degs, dtype=torch.float32)
    if (
        positions.ndim != 2
        or positions.shape[0] < 2
        or positions.shape[1] != 2
    ):
        raise ExperimentRuntimeError(
            "At least two finite cone positions are required"
        )
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(torch.inf)
    return float(distances.min(dim=1).values.median())


__all__ = [
    "ExperimentRuntimeError",
    "ValidationContext",
    "build_network",
    "diagnostic_training_clips",
    "ensure_initial_reference",
    "evaluate_validation",
    "training_mean",
]
