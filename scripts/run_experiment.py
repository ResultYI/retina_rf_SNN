from __future__ import annotations
# noqa: SIZE_OK — CLI orchestration remains a single explicit experiment lifecycle.

import argparse
import copy
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.physiology_profiles import macaque_photopic
from configs.rgc_type_priors import load_type_priors
from data.input_identity import validate_experiment_input
from evaluation.response_pipeline import evaluate_and_report_response_experiment
from models.response_snn import build_response_retina_model
from training.response_checkpointing import (
    ResponseCheckpointState,
    inspect_response_checkpoint,
    load_response_checkpoint,
)
from training.response_config import load_response_config
from training.response_data import prepare_response_data, sample_response_batch
from training.response_trainer import ResponseTrainer


class ResponseExperimentError(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit known recorded-RGC responses with a stateful retinal SNN."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument("--resume")
    args = parser.parse_args()
    output = _prepare_output(args)
    try:
        _run(args, output)
    except (OSError, RuntimeError, ValueError) as exc:
        (output / "run_status.json").write_text(
            json.dumps(
                {"status": "FAILED_WITH_REPORT", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output / "final_report_zh.md").write_text(
            f"# RGC 响应拟合失败\n\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise


def _run(args: argparse.Namespace, output: Path) -> None:
    config = load_response_config(args.config)
    torch.manual_seed(config.seed)
    data = prepare_response_data(config.data)
    validate_experiment_input(data.input_identity, data.dt_ms)
    priors = load_type_priors(
        config.model.type_prior_path,
        required_type_ids=tuple(sorted(set(data.cells.type_ids))),
    )
    spacing = _cone_spacing(data.cone_positions_degs)
    profile = macaque_photopic(
        dt_ms=data.dt_ms,
        cone_spacing_deg=spacing,
        eccentricity_deg=float(np.mean(data.cells.eccentricities_deg)),
    )
    model = build_response_retina_model(
        torch.as_tensor(data.cone_positions_degs),
        data.cells,
        profile,
        priors,
        support_radius_degs=config.model.support_radius_degs,
        readout_rate_tau_ms=config.model.readout_rate_tau_ms,
        surrogate_slope=config.model.surrogate_slope,
        parameter_sharing_mode=config.model.parameter_sharing_mode,
        parameter_sharing_seed=config.seed,
        matched_initialization=config.model.matched_initialization,
    ).to(torch.device(args.device))
    initialized_model = copy.deepcopy(model)
    trainer = ResponseTrainer(model, config, data, torch.device(args.device))
    best_path = output / "checkpoint_best_nll.pt"
    last_path = output / "checkpoint_last.pt"
    if args.checkpoint:
        checkpoint_state = load_response_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=None,
            generator=None,
            fingerprint=data.fingerprint,
            target_kind=data.target_kind.value,
            config=config,
        )
        _restore_trainer_lineage(trainer, checkpoint_state)
    elif args.resume:
        resume_state = inspect_response_checkpoint(args.resume)
        _restore_resume_best(Path(args.resume), best_path, resume_state.run_id)
        checkpoint_state = load_response_checkpoint(
            args.resume,
            model=model,
            optimizer=trainer.optimizer,
            generator=trainer.sampling_generator,
            fingerprint=data.fingerprint,
            target_kind=data.target_kind.value,
            config=config,
            expected_run_id=resume_state.run_id,
        )
        _restore_trainer_lineage(trainer, checkpoint_state)
        if not args.diagnostics_only:
            _train(
                trainer,
                data,
                config,
                output,
                best_path,
                last_path,
                args.stop_after_steps,
            )
    elif not args.diagnostics_only:
        _train(trainer, data, config, output, best_path, last_path, args.stop_after_steps)
    if not best_path.exists() and not (args.checkpoint or args.resume):
        raise ResponseExperimentError("No best-NLL checkpoint was produced")
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else best_path
        if best_path.exists()
        else Path(args.resume)
    )
    load_response_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        generator=None,
        fingerprint=data.fingerprint,
        target_kind=data.target_kind.value,
        config=config,
        expected_run_id=trainer.run_id,
    )
    evaluate_and_report_response_experiment(
        output,
        model=model,
        initialized_model=initialized_model,
        trainer=trainer,
        data=data,
        config=config,
        checkpoint=checkpoint,
        evaluation_split="test" if args.final_test else "validation",
    )
    _write_parameter_sharing_manifest(output, model, initialized_model)


def _train(
    trainer: ResponseTrainer,
    data,
    config,
    output: Path,
    best_path: Path,
    last_path: Path,
    stop_after_steps: int | None,
) -> None:
    steps = config.training.max_optimizer_steps
    if stop_after_steps is not None:
        steps = min(steps, stop_after_steps)
    log_path = output / "training.jsonl"
    while trainer.optimizer_step < steps:
        batch = sample_response_batch(
            data.train,
            batch_size=config.training.batch_size,
            generator=trainer.sampling_generator,
            device=trainer.device,
        )
        result = trainer.train_step(*batch)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"optimizer_step": trainer.optimizer_step, **asdict(result)}
                )
                + "\n"
            )
        validate = (
            trainer.optimizer_step % config.training.validation_interval_steps == 0
            or trainer.optimizer_step == steps
        )
        if validate:
            metrics = trainer.evaluate(data.validation)
            if metrics.nll < trainer.best_nll:
                trainer.best_nll = metrics.nll
                trainer.best_checkpoint_step = trainer.optimizer_step
                trainer.save(best_path, "best")
            trainer.save(last_path, "last")


def _cone_spacing(positions: np.ndarray) -> float:
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    )
    distances[distances == 0] = np.inf
    return float(np.median(distances.min(axis=1)))


def _restore_resume_best(
    resume_path: Path,
    output_best_path: Path,
    run_id: str,
) -> None:
    if output_best_path.exists():
        output_state = inspect_response_checkpoint(output_best_path)
        if output_state.run_id != run_id or output_state.checkpoint_kind != "best":
            raise ResponseExperimentError(
                "Output best checkpoint belongs to a foreign run lineage"
            )
        return
    previous_best = (
        resume_path
        if resume_path.name == output_best_path.name
        else resume_path.with_name(output_best_path.name)
    )
    if not previous_best.exists():
        raise ResponseExperimentError(
            "Resume requires the historical checkpoint_best_nll.pt"
        )
    previous_state = inspect_response_checkpoint(previous_best)
    if previous_state.run_id != run_id or previous_state.checkpoint_kind != "best":
        raise ResponseExperimentError(
            "Historical best checkpoint belongs to a foreign run lineage"
        )
    if previous_best.resolve() != output_best_path.resolve():
        shutil.copy2(previous_best, output_best_path)


def _restore_trainer_lineage(
    trainer: ResponseTrainer,
    state: ResponseCheckpointState,
) -> None:
    trainer.optimizer_step = state.optimizer_step
    trainer.best_nll = state.best_nll
    trainer.best_checkpoint_step = state.best_checkpoint_step
    trainer.run_id = state.run_id
    trainer.parent_run_id = state.parent_run_id


def _write_parameter_sharing_manifest(output: Path, model, initialized_model=None) -> None:
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_rgc = model.rgc if initialized_model is None else initialized_model.rgc
    manifest["parameter_sharing"] = {
        "mode": model.rgc.parameter_sharing_mode,
        "matched_initialization": model.rgc.matched_initialization,
        "shuffle_contract": model.rgc.shuffle_contract,
        "observed_type_labels": model.rgc.observed_type_labels,
        "cell_polarities": model.rgc.cell_polarities.detach().cpu().tolist(),
        "effective_type_labels": model.rgc.effective_type_labels,
        "parameter_group_labels": model.rgc.parameter_group_labels,
        "initial_effective_parameters": {
            name: getattr(initial_rgc, name)().detach().cpu().tolist()
            for name in initial_rgc.parameter_names
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prepare_output(args: argparse.Namespace) -> Path:
    if args.final_test and not args.diagnostics_only:
        raise ResponseExperimentError(
            "--final-test requires --diagnostics-only with --checkpoint or --resume"
        )
    if args.diagnostics_only and not (args.checkpoint or args.resume):
        raise ResponseExperimentError(
            "--diagnostics-only requires --checkpoint or --resume"
        )
    if args.overwrite and args.resume:
        raise ResponseExperimentError("--overwrite cannot be combined with --resume")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        if not args.overwrite:
            raise ResponseExperimentError(
                "Fresh run requires an empty output directory; use --overwrite"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


if __name__ == "__main__":
    main()
