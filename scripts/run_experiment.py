from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.experiment_pipeline import (
    FinalEvaluationRequest,
    run_final_evaluation,
)
from evaluation.reconstruction import (
    fit_augmented_reconstruction_scale,
)
from loss.retina import RetinaObjective
from training import (
    checkpoint_reporting,
    experiment_cli,
    experiment_setup,
    runtime,
    training_batch,
)
from training.checkpointing import load_checkpoint, save_checkpoint
from training.config import load_config
from training.data import prepare_data
from training.trainer import RetinaTrainer


class ExperimentError(RuntimeError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    args = experiment_cli.parse_experiment_args(argv)
    config = experiment_cli.apply_invocation_overrides(
        load_config(args.config),
        args,
    )
    experiment_cli.seed_everything(config.seed)
    device = torch.device(args.device)
    prepared = prepare_data(config.data)
    model, decoder = runtime.build_network(config, prepared, device)
    output_dir = args.output
    setup = experiment_setup.initialize_experiment(
        experiment_setup.ExperimentSetupRequest(
            model=model,
            decoder=decoder,
            prepared=prepared,
            config=config,
            device=device,
            output_dir=output_dir,
        )
    )
    if args.diagnostics_only and args.resume is None:
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
    generators = training_batch.TrainingGenerators(
        sampling=torch.Generator().manual_seed(config.seed + 1),
        augmentation=torch.Generator().manual_seed(config.seed + 2),
    )
    if args.resume is not None:
        payload = load_checkpoint(args.resume, device)
        if payload["resolved_config"] != config.resolved():
            raise ExperimentError("Resume configuration does not match checkpoint")
        trainer.restore(payload, generators.sampling, generators.augmentation)

    execution_limit = experiment_cli.execution_limit(
        config.training.max_optimizer_steps,
        args.stop_after_steps,
        args.representation_diagnostic_steps,
    )
    validation = runtime.ValidationContext(
        clips=setup.validation_clips,
        train_mean=setup.train_mean,
        ema_alpha=setup.ema_alpha,
    )
    evaluation_request = FinalEvaluationRequest(
        trainer=trainer,
        prepared=prepared,
        config=config,
        validation=validation,
        calibration_clips=setup.calibration_clips,
        initial_reference=setup.initial_reference,
        initial_diagnostics=setup.initial_diagnostics,
        output_dir=output_dir,
        device=device,
    )
    if args.diagnostics_only:
        run_final_evaluation(evaluation_request)
        return 0
    diagnostic_validation_mses: list[float] = []
    while trainer.optimizer_step < execution_limit:
        batch = training_batch.build_training_batch(
            training_batch.TrainingBatchRequest(
                sources=prepared.train,
                config=config,
                device=device,
                generators=generators,
                optimizer_step=trainer.optimizer_step,
            )
        )
        result = trainer.train_optimizer_step(batch)
        if trainer.optimizer_step % config.training.validation_interval_steps == 0:
            reconstruction, _, target_energy_ratio = runtime.evaluate_validation(
                trainer,
                validation,
                config,
            )
            best_reconstruction, best_feasible = trainer.record_validation(
                trainer.optimizer_step,
                reconstruction.mse,
                target_energy_ratio,
            )
            best_representation = False
            selection_metrics = None
            if trainer.optimizer_step <= config.training.reconstruction_bootstrap_steps:
                best_representation, selection_metrics = setup.selector.observe(
                    trainer.validation_state,
                    reconstruction.mse,
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
            if selection_metrics is not None:
                metrics.update(asdict(selection_metrics))
            checkpoint_reporting.write_training_row(
                output_dir,
                trainer.optimizer_step,
                metrics,
            )
            payload = trainer.checkpoint_payload(
                generators.sampling,
                generators.augmentation,
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
            if best_representation:
                save_checkpoint(
                    output_dir / "checkpoint_best_representation.pt",
                    payload,
                )
            diagnostic_validation_mses.append(reconstruction.mse)
            if experiment_cli.diagnostic_should_stop(
                setup.initial_diagnostics.current_decoder.mse,
                diagnostic_validation_mses,
                args,
            ):
                break

    selected_checkpoint = checkpoint_reporting.select_checkpoint(output_dir)
    checkpoint_reporting.write_checkpoint_summaries(
        output_dir,
        selected_checkpoint,
        device,
    )
    trainer.restore(
        load_checkpoint(selected_checkpoint, device),
        generators.sampling,
        generators.augmentation,
    )
    run_final_evaluation(evaluation_request)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
