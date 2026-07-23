from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.experiment_pipeline import (
    FinalEvaluationRequest,
    run_final_evaluation,
)
from evaluation.reconstruction import (
    fit_causal_ema_alpha,
    fit_augmented_reconstruction_scale,
)
from evaluation.representation_diagnostics import (
    calibrate_decoder,
    collect_decoder_examples,
    representation_diagnostics,
    write_decoder_calibration,
)
from loss.retina import RetinaObjective
from training.checkpointing import load_checkpoint, save_checkpoint
from training.checkpoint_reporting import (
    write_checkpoint_summaries,
    write_training_row,
)
from training.augmentation import (
    AugmentedClip,
    augment_clip,
    fixed_validation_clips,
)
from training.config import load_config
from training.data import prepare_data
from training.runtime import (
    ValidationContext,
    build_network,
    diagnostic_training_clips,
    ensure_initial_reference,
    evaluate_validation,
    training_mean,
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
    model, decoder = build_network(config, prepared, device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_clips = fixed_validation_clips(
        prepared.validation,
        config.data,
        config.seed + 10_000,
        device,
    )
    calibration_clips = diagnostic_training_clips(prepared, config, device)
    train_examples = collect_decoder_examples(
        model,
        calibration_clips,
        config.training.supervised_steps,
    )
    validation_examples = collect_decoder_examples(
        model,
        validation_clips,
        config.training.supervised_steps,
    )
    spatial_weights = model.rgc.compute_spatial_weights()
    calibration = calibrate_decoder(decoder, train_examples, spatial_weights)
    ema_alpha = fit_causal_ema_alpha(
        train_examples.noisy_input,
        train_examples.target,
    )
    train_mean = training_mean(prepared).to(device)
    initial_diagnostics = representation_diagnostics(
        decoder,
        train_examples,
        validation_examples,
        spatial_weights,
        torch.as_tensor(
            prepared.positions_degs,
            device=device,
            dtype=spatial_weights.dtype,
        ),
        train_mean,
        ema_alpha,
    )
    write_decoder_calibration(output_dir, calibration)
    (output_dir / "representation_initial.json").write_text(
        json.dumps(asdict(initial_diagnostics), indent=2),
        encoding="utf-8",
    )
    initial_reference = ensure_initial_reference(
        output_dir,
        model,
        decoder,
        config,
    )
    if args.diagnostics_only:
        return 0
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
        fit_augmented_reconstruction_scale(
            prepared.train,
            config.data,
            seed=config.seed + 3,
        ),
    )
    sampling_generator = torch.Generator().manual_seed(config.seed + 1)
    augmentation_generator = torch.Generator().manual_seed(config.seed + 2)
    if args.resume is not None:
        payload = load_checkpoint(args.resume, device)
        if payload["resolved_config"] != config.resolved():
            raise ExperimentError("Resume configuration does not match checkpoint")
        trainer.restore(payload, sampling_generator, augmentation_generator)

    execution_limit = _execution_limit(
        config.training.max_optimizer_steps,
        args.stop_after_steps,
    )
    validation = ValidationContext(
        clips=validation_clips,
        train_mean=train_mean,
        ema_alpha=ema_alpha,
    )
    while trainer.optimizer_step < execution_limit:
        batch: list[AugmentedClip] = []
        for _ in range(config.training.batch_size):
            source_index = int(
                torch.randint(
                    len(prepared.train), (1,), generator=sampling_generator
                ).item()
            )
            source = prepared.train[source_index]
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
            reconstruction, _, target_energy_ratio = evaluate_validation(
                trainer,
                validation,
                config,
            )
            best_reconstruction, best_feasible = trainer.record_validation(
                trainer.optimizer_step,
                reconstruction.mse,
                target_energy_ratio,
            )
            metrics = {
                **result.metrics,
                "gradient_norm": result.gradient_norm,
                "temporal_gradient_norm": result.temporal_gradient_norm,
                "peak_memory_bytes": result.peak_memory_bytes,
                "reference_energy": trainer.energy_state.reference_energy,
                "current_budget": trainer.energy_state.current_budget,
                "target_budget": trainer.energy_state.target_budget,
                "validation_mse": reconstruction.mse,
                "validation_representation_skill": reconstruction.representation_skill,
                "validation_target_energy_ratio": target_energy_ratio,
                "best_reconstruction_event": best_reconstruction,
                "best_feasible_event": best_feasible,
            }
            write_training_row(output_dir, trainer.optimizer_step, metrics)
            payload = trainer.checkpoint_payload(
                sampling_generator, augmentation_generator
            )
            save_checkpoint(
                output_dir / "checkpoint_last.pt",
                payload,
            )
            if best_reconstruction:
                save_checkpoint(
                    output_dir / "checkpoint_best_reconstruction.pt", payload
                )
            if best_feasible:
                save_checkpoint(output_dir / "checkpoint_best_feasible.pt", payload)

    selected_checkpoint = output_dir / "checkpoint_best_feasible.pt"
    if not selected_checkpoint.exists():
        selected_checkpoint = output_dir / "checkpoint_best_reconstruction.pt"
    if not selected_checkpoint.exists():
        raise ExperimentError("Training produced no validation checkpoint")
    write_checkpoint_summaries(output_dir, selected_checkpoint, device)
    trainer.restore(
        load_checkpoint(selected_checkpoint, device),
        sampling_generator,
        augmentation_generator,
    )
    run_final_evaluation(
        FinalEvaluationRequest(
            trainer=trainer,
            prepared=prepared,
            config=config,
            validation=validation,
            calibration_clips=calibration_clips,
            initial_reference=initial_reference,
            output_dir=output_dir,
            device=device,
        )
    )
    return 0


def _seed_everything(seed: int) -> None:
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
    parser.add_argument("--stop-after-steps", type=_positive_int)
    parser.add_argument("--diagnostics-only", action="store_true")
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _execution_limit(configured_limit: int, stop_after_steps: int | None) -> int:
    return min(configured_limit, stop_after_steps) if stop_after_steps is not None else configured_limit


if __name__ == "__main__":
    raise SystemExit(main())
