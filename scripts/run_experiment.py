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

from configs.physiology_profiles import human_macaque
from evaluation.dynamic_rf import (
    build_matched_context_pairs,
    evaluate_dynamic_rf,
    select_dynamic_rf_units,
)
from evaluation.dynamic_rf_summary import (
    compare_dynamic_rf,
    not_run_dynamic_rf_summary,
)
from evaluation.parameter_audit import audit_parameters
from evaluation.reconstruction import (
    ReconstructionMetrics,
    fit_augmented_reconstruction_scale,
    reconstruction_metrics,
)
from evaluation.reporting import summarize_evaluation, write_evaluation_report
from evaluation.rgc_types import identify_rgc_types
from evaluation.temporal_probes import run_temporal_probes
from loss.retina import RetinaObjective
from models.cells.rgc_types import RGCConfig, RGCOutput
from models.decoder.local_decoder import TiedLocalDecoder
from models.retina_snn import RetinaModel, build_retina_model
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
    model, decoder = _build_network(config, prepared, device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_reference = _ensure_initial_reference(output_dir, model, config)
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

    validation = fixed_validation_clips(
        prepared.validation, config.data, config.seed + 10_000, device
    )
    while trainer.optimizer_step < config.training.max_optimizer_steps:
        batch: list[AugmentedClip] = []
        for _ in range(config.training.gradient_accumulation_steps):
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
            reconstruction, _, target_energy_ratio = _evaluate_validation(
                trainer, validation, prepared, config
            )
            best_reconstruction, best_feasible = trainer.record_validation(
                trainer.optimizer_step,
                reconstruction.mse,
                target_energy_ratio,
            )
            metrics = {
                **result.metrics,
                "validation_mse": reconstruction.mse,
                "validation_representation_skill": reconstruction.representation_skill,
                "validation_target_energy_ratio": target_energy_ratio or 0.0,
            }
            _write_training_row(output_dir, trainer.optimizer_step, metrics)
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
    trainer.restore(
        load_checkpoint(selected_checkpoint, device),
        sampling_generator,
        augmentation_generator,
    )
    reconstruction, output, target_energy_ratio = _evaluate_validation(
        trainer, validation, prepared, config
    )
    representation_passed = (
        reconstruction.representation_skill
        >= config.evaluation.minimum_representation_skill
    )
    energy_passed = (
        trainer.optimizer_step >= config.training.budget_ramp_end_step
        and target_energy_ratio is not None
        and target_energy_ratio <= config.evaluation.maximum_energy_budget_ratio
    )
    initialized_model, initialized_decoder = _build_network(config, prepared, device)
    initialized_model.load_state_dict(initial_reference["model_state"])
    if representation_passed and energy_passed:
        pairs = build_matched_context_pairs(
            prepared.validation, config.data, config.evaluation
        )
        selection_plan = select_dynamic_rf_units(model, pairs, config.evaluation)
        trained_dynamic_rf = evaluate_dynamic_rf(
            model,
            pairs,
            config.evaluation,
            dt_ms=prepared.dt_ms,
            selection_plan=selection_plan,
        )
        initialized_dynamic_rf = evaluate_dynamic_rf(
            initialized_model,
            pairs,
            config.evaluation,
            dt_ms=prepared.dt_ms,
            selection_plan=selection_plan,
        )
        probes = run_temporal_probes(
            model,
            _training_mean(prepared),
            sequence_steps=config.data.sequence_steps,
            onset_step=(
                config.training.burn_in_steps + config.training.context_only_steps
            ),
            dt_ms=prepared.dt_ms,
        )
        initialized_probes = run_temporal_probes(
            initialized_model,
            _training_mean(prepared),
            sequence_steps=config.data.sequence_steps,
            onset_step=(
                config.training.burn_in_steps + config.training.context_only_steps
            ),
            dt_ms=prepared.dt_ms,
        )
        type_report = identify_rgc_types(
            model.rgc,
            output,
            probes=probes,
            initialized_rgc=initialized_model.rgc,
            initialized_probes=initialized_probes,
            config=config.evaluation,
            seed=config.seed,
        )
        dynamic_rf_summary, dynamic_rf_sources = compare_dynamic_rf(
            trained_dynamic_rf,
            initialized_dynamic_rf,
            config.evaluation,
            seed=config.seed,
        )
    else:
        selection_plan = ()
        trained_dynamic_rf = ()
        initialized_dynamic_rf = ()
        dynamic_rf_sources = ()
        dynamic_rf_summary = not_run_dynamic_rf_summary()
        type_report = None
    summary = summarize_evaluation(
        reconstruction,
        target_energy_ratio,
        trained_dynamic_rf,
        config,
        dynamic_rf_status=dynamic_rf_summary.status,
        rgc_type_status=type_report.status if type_report is not None else "not_run",
        budget_ramp_complete=(
            trainer.optimizer_step >= config.training.budget_ramp_end_step
        ),
    )
    write_evaluation_report(
        output_dir,
        summary,
        trained_dynamic_rf,
        initialized_dynamic_rf,
        selection_plan,
        dynamic_rf_summary,
        dynamic_rf_sources,
        type_report,
        config,
    )
    audit = [
        asdict(entry)
        for entry in audit_parameters(
            model,
            decoder,
            initialized_model=initialized_model,
            initialized_decoder=initialized_decoder,
        )
    ]
    (output_dir / "parameter_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return 0


def _build_network(
    config: ExperimentConfig,
    prepared: PreparedData,
    device: torch.device,
) -> tuple[RetinaModel, TiedLocalDecoder]:
    spacing = _median_nearest_spacing(prepared.positions_degs)
    model_config = config.model
    profile = human_macaque(
        dt_ms=prepared.dt_ms,
        cone_spacing_deg=spacing,
        eccentricity_deg=prepared.eccentricity_deg,
        debug_checks=model_config.debug_checks,
    )
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
        adaptation_gain_max=model_config.adaptation_gain_max,
        amacrine_gain_max=model_config.amacrine_gain_max,
        subunit_gain_max=model_config.subunit_gain_max,
        initialization_seed=config.seed,
        debug_checks=model_config.debug_checks,
    )
    model = build_retina_model(prepared.positions_degs, profile, rgc_config).to(device)
    decoder = TiedLocalDecoder(
        model.rgc.unit_count,
        prepared.positions_degs.shape[0],
        model_config.decoder_gain_max,
    ).to(device)
    return model, decoder


def _ensure_initial_reference(
    output_dir: Path,
    model: RetinaModel,
    config: ExperimentConfig,
) -> dict[str, object]:
    path = output_dir / "initial_reference.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema") != "retina_rf_snn_initial_reference":
            raise ExperimentError("Initial reference schema is incompatible")
        if payload.get("resolved_config") != config.resolved():
            raise ExperimentError("Initial reference configuration does not match")
        if not isinstance(payload.get("model_state"), dict):
            raise ExperimentError("Initial reference model state is missing")
        return payload
    payload: dict[str, object] = {
        "schema": "retina_rf_snn_initial_reference",
        "resolved_config": config.resolved(),
        "model_state": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, path)
    return payload


@torch.no_grad()
def _evaluate_validation(
    trainer: RetinaTrainer,
    clips: Sequence[AugmentedClip],
    prepared: PreparedData,
    config: ExperimentConfig,
) -> tuple[ReconstructionMetrics, RGCOutput, float | None]:
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
        predictions.append(
            trainer.decoder(output.rates, spatial_weights)[
                :, -config.training.supervised_steps :
            ]
        )
        targets.append(clip.clean_target[:, -config.training.supervised_steps :])
        outputs.append(output)
        energies.append(float(losses.energy))
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    train_mean = _training_mean(prepared).to(
        prediction.device, prediction.dtype
    )
    metrics = reconstruction_metrics(prediction, target, train_mean)
    target_budget = trainer.energy_state.target_budget
    target_energy_ratio = (
        float(np.mean(energies)) / target_budget
        if target_budget is not None
        else None
    )
    return metrics, _concatenate_outputs(outputs), target_energy_ratio


def _concatenate_outputs(outputs: Sequence[RGCOutput]) -> RGCOutput:
    if not outputs:
        raise ExperimentError("Validation requires at least one output")
    return RGCOutput(
        hard_spikes=torch.cat([output.hard_spikes for output in outputs]),
        surrogate_spikes=torch.cat([output.surrogate_spikes for output in outputs]),
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


def _training_mean(prepared: PreparedData) -> torch.Tensor:
    return torch.cat([clip.clean for clip in prepared.train]).mean(dim=0)


def _write_training_row(
    output_dir: Path,
    optimizer_step: int,
    metrics: dict[str, float],
) -> None:
    row = {"optimizer_step": optimizer_step, **metrics}
    with (output_dir / "training.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
