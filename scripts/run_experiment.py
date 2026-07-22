from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.physiology_profiles import human_macaque
from evaluation.dynamic_rf import build_matched_context_pairs, evaluate_dynamic_rf
from evaluation.parameter_audit import audit_parameters
from evaluation.reconstruction import (
    ReconstructionMetrics,
    fit_reconstruction_scale,
    reconstruction_metrics,
)
from evaluation.reporting import summarize_evaluation, write_evaluation_report
from evaluation.rgc_types import identify_rgc_types
from loss.retina import RetinaObjective
from models.cells.rgc_types import RGCConfig, RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import build_retina_model
from training.checkpointing import load_checkpoint, save_checkpoint
from training.config import ExperimentConfig, load_config
from training.data import (
    AugmentedClip,
    PreparedData,
    augment_clip,
    fixed_validation_clips,
    prepare_data,
)
from training.trainer import RetinaTrainer


class ExperimentError(RuntimeError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    _seed_everything(config.seed)
    device = torch.device(args.device)
    prepared = prepare_data(config.data)
    spacing = _median_nearest_spacing(prepared.positions_degs)
    profile = human_macaque(
        dt_ms=prepared.dt_ms,
        cone_spacing_deg=spacing,
        eccentricity_deg=prepared.eccentricity_deg,
    )
    model_config = config.model
    rgc_config = RGCConfig(
        units_per_center=model_config.units_per_center,
        support_radius_degs=model_config.support_radius_spacing_multiplier * spacing,
        sigma_min_degs=model_config.sigma_min_spacing_multiplier * spacing,
        sigma_initial_degs=model_config.sigma_initial_spacing_multiplier * spacing,
        sigma_max_degs=model_config.sigma_max_spacing_multiplier * spacing,
        dt_ms=prepared.dt_ms,
        readout_rate_tau_ms=model_config.readout_rate_tau_ms,
        max_tau_ms=model_config.max_tau_ms,
        surrogate_slope=model_config.surrogate_slope,
    )
    model = build_retina_model(prepared.positions_degs, profile, rgc_config).to(device)
    decoder = TiedLocalDecoder(
        model.rgc.unit_count, prepared.positions_degs.shape[0]
    ).to(device)
    objective = RetinaObjective(
        rho_energy=config.objective.rho_energy,
        variance_floor=config.objective.variance_floor,
        phenotype_temperature=config.objective.phenotype_temperature,
        homeostasis_rate_min=config.objective.homeostasis_rate_min,
    )
    trainer = RetinaTrainer(
        model,
        decoder,
        objective,
        config,
        fit_reconstruction_scale(prepared.train),
    )
    augmentation_generator = torch.Generator().manual_seed(config.seed)
    if args.resume is not None:
        payload = load_checkpoint(args.resume, device)
        if payload["resolved_config"] != config.resolved():
            raise ExperimentError("Resume configuration does not match checkpoint")
        trainer.restore(payload, augmentation_generator)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    while trainer.optimizer_step < config.training.max_optimizer_steps:
        batch: list[AugmentedClip] = []
        for _ in range(config.training.gradient_accumulation_steps):
            source = random.choice(prepared.train)
            clip = augment_clip(source, config.data, augmentation_generator)
            batch.append(
                AugmentedClip(
                    noisy_input=clip.noisy_input.unsqueeze(0).to(device),
                    clean_target=clip.clean_target.unsqueeze(0).to(device),
                    metadata=clip.metadata,
                )
            )
        result = trainer.train_optimizer_step(batch)
        if trainer.optimizer_step % config.training.validation_interval_steps == 0:
            _write_training_row(output_dir, trainer.optimizer_step, result.metrics)
            save_checkpoint(
                output_dir / "checkpoint_last.pt",
                trainer.checkpoint_payload(augmentation_generator),
            )

    validation = fixed_validation_clips(
        prepared.validation, config.data, config.seed + 10_000, device
    )
    reconstruction, output, energy_ratio = _evaluate_validation(
        trainer, validation, prepared, config
    )
    pairs = build_matched_context_pairs(
        prepared.validation, config.data, config.evaluation
    )
    dynamic_rf = evaluate_dynamic_rf(
        model, pairs, config.evaluation, dt_ms=prepared.dt_ms
    )
    type_report = identify_rgc_types(model.rgc, output, seed=config.seed)
    summary = summarize_evaluation(reconstruction, energy_ratio, dynamic_rf, config)
    write_evaluation_report(output_dir, summary, dynamic_rf, type_report, config)
    audit = [asdict(entry) for entry in audit_parameters(model, decoder)]
    (output_dir / "parameter_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    save_checkpoint(
        output_dir / "checkpoint_last.pt",
        trainer.checkpoint_payload(augmentation_generator),
    )
    return 0


@torch.no_grad()
def _evaluate_validation(
    trainer: RetinaTrainer,
    clips: Sequence[AugmentedClip],
    prepared: PreparedData,
    config: ExperimentConfig,
) -> tuple[ReconstructionMetrics, RGCOutput, float]:
    trainer.model.eval()
    trainer.decoder.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    outputs: list[RGCOutput] = []
    energies: list[float] = []
    spatial_weights = trainer.model.rgc.compute_spatial_weights()
    for clip in clips:
        losses, output, _ = trainer.forward_clip(
            clip.noisy_input,
            clip.clean_target,
            checkpointed=False,
        )
        predictions.append(trainer.decoder(output.rates, spatial_weights))
        targets.append(clip.clean_target[:, config.training.burn_in_steps :])
        outputs.append(output)
        energies.append(float(losses.energy))
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    train_mean = torch.cat([clip.clean for clip in prepared.train]).mean(dim=0).to(
        prediction.device, prediction.dtype
    )
    metrics = reconstruction_metrics(prediction, target, train_mean)
    energy_budget = trainer.energy_state.budget
    energy_ratio = float(np.mean(energies)) / max(energy_budget or 1e-12, 1e-12)
    return metrics, _concatenate_outputs(outputs), energy_ratio


def _concatenate_outputs(outputs: Sequence[RGCOutput]) -> RGCOutput:
    if not outputs:
        raise ExperimentError("Validation requires at least one output")
    return RGCOutput(
        hard_spikes=torch.cat([output.hard_spikes for output in outputs]),
        spike_probability=torch.cat([output.spike_probability for output in outputs]),
        rates=torch.cat([output.rates for output in outputs]),
        generator_potential=torch.cat(
            [output.generator_potential for output in outputs]
        ),
    )


def _median_nearest_spacing(positions_degs: np.ndarray) -> float:
    positions = torch.as_tensor(positions_degs, dtype=torch.float32)
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] != 2:
        raise ExperimentError("At least two finite cone positions are required")
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(torch.inf)
    return float(distances.min(dim=1).values.median())


def _write_training_row(
    output_dir: Path,
    optimizer_step: int,
    metrics: dict[str, float],
) -> None:
    row = {"optimizer_step": optimizer_step, **metrics}
    with (output_dir / "training.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the canonical retina model")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/experiment"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
