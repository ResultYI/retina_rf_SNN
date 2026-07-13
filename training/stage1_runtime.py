from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

import torch

from training.epoch_metrics import weighted_mean_row
from training.hybrid import (
    HybridRetinaTrainer,
    RetinaTargets,
    RetinaTrainingBatch,
    TrainingStage,
)
from training.stage1_reporting import (
    baseline_loader,
    stage1_log_row,
    write_log_row,
)
from training.stage1_types import (
    CheckpointProgress,
    Stage1Loaders,
    TrainStage1Config,
    TrainStage1Error,
)


def train_epochs(
    config: TrainStage1Config,
    trainer: HybridRetinaTrainer,
    loaders: Stage1Loaders,
    writer: csv.DictWriter[TextIO],
    jsonl_handle: TextIO,
    progress: CheckpointProgress,
) -> None:
    step = progress.step
    best_loss = progress.best_loss
    for epoch in range(progress.epoch, config.epochs):
        for batch in loaders.train:
            step += 1
            result = trainer.train_batch(
                batch_to_device(batch, config.device),
                config.stage,
            )
            write_log_row(
                writer,
                jsonl_handle,
                stage1_log_row("train", epoch, step, result),
            )
        summaries: dict[str, dict[str, str]] = {}
        for split, loader in (
            ("train_eval", baseline_loader(loaders.train, config)),
            ("val", loaders.val),
        ):
            if loader is not None:
                rows = []
                for batch in loader:
                    result = trainer.evaluate_batch(
                        batch_to_device(batch, config.device)
                    )
                    rows.append(
                        (
                            batch.x_cone.shape[0],
                            stage1_log_row(split, epoch, step, result),
                        )
                    )
                summary = weighted_mean_row(rows)
                summaries[split] = summary
                write_log_row(writer, jsonl_handle, summary)
        selected_summary = (
            summaries["val"] if loaders.val is not None else summaries["train_eval"]
        )
        selected_loss = float(selected_summary["loss_total"])
        is_best = selected_loss < best_loss
        if is_best:
            best_loss = selected_loss
        write_checkpoint(config, trainer, epoch, step, best_loss, is_best)


def batch_to_device(
    batch: RetinaTrainingBatch,
    device: torch.device,
) -> RetinaTrainingBatch:
    return RetinaTrainingBatch(
        x_cone=batch.x_cone.to(device),
        targets=RetinaTargets(
            fine=batch.targets.fine.to(device),
            coarse=batch.targets.coarse.to(device),
        ),
    )


def write_checkpoint(
    config: TrainStage1Config,
    trainer: HybridRetinaTrainer,
    epoch: int,
    step: int,
    best_loss: float,
    is_best: bool,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "core": trainer.core.state_dict(),
        "decoder": trainer.decoder.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "stage": config.stage.value,
    }
    torch.save(checkpoint, config.output_dir / "checkpoint.pt")
    if is_best:
        torch.save(checkpoint, config.output_dir / "best_checkpoint.pt")


def restore_checkpoint(
    config: TrainStage1Config,
    trainer: HybridRetinaTrainer,
) -> CheckpointProgress:
    if config.resume is None:
        return CheckpointProgress(0, 0, float("inf"))
    checkpoint = load_checkpoint(config.resume, config.device)
    checkpoint_stage = checkpoint.get("stage")
    if checkpoint_stage == TrainingStage.DECODER_WARMUP.value and (
        config.stage == TrainingStage.CORE_FINETUNE
    ):
        trainer.core.load_state_dict(checkpoint["core"])
        trainer.decoder.load_state_dict(checkpoint["decoder"])
        return CheckpointProgress(0, 0, float("inf"))
    if checkpoint_stage != config.stage.value:
        raise TrainStage1Error("Checkpoint stage does not match --stage")
    trainer.core.load_state_dict(checkpoint["core"])
    trainer.decoder.load_state_dict(checkpoint["decoder"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state(trainer.optimizer, config.device)
    return CheckpointProgress(
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["step"]),
        float(checkpoint.get("best_loss", float("inf"))),
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TrainStage1Error("Checkpoint must contain a mapping")
    checkpoint = dict(payload)
    required = {"epoch", "step", "core", "decoder", "optimizer", "stage"}
    if not required <= checkpoint.keys():
        raise TrainStage1Error("Checkpoint is missing required fields")
    if not isinstance(checkpoint["epoch"], int) or not isinstance(
        checkpoint["step"], int
    ):
        raise TrainStage1Error("Checkpoint epoch and step must be integers")
    if not isinstance(checkpoint["stage"], str):
        raise TrainStage1Error("Checkpoint stage must be text")
    if not all(
        isinstance(checkpoint[key], Mapping)
        for key in ("core", "decoder", "optimizer")
    ):
        raise TrainStage1Error("Checkpoint state entries must be mappings")
    return checkpoint


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config: TrainStage1Config) -> None:
    if config.epochs < 1 or config.batch_size < 1 or config.t_bptt < 1:
        raise TrainStage1Error("epochs, batch_size, and t_bptt must be positive")
    if not all(
        math.isfinite(rate) and rate > 0
        for rate in (config.lr_core, config.lr_decoder)
    ):
        raise TrainStage1Error("Learning rates must be positive and finite")
    if config.num_workers < 0:
        raise TrainStage1Error("num_workers must be non-negative")
    if not 0 <= config.max_clip_fraction <= 1:
        raise TrainStage1Error("max_clip_fraction must lie in [0, 1]")
    if config.resume is not None and not config.resume.is_file():
        raise TrainStage1Error(f"Checkpoint does not exist: {config.resume}")


def validate_clip_fractions(
    fractions: Sequence[tuple[Path, float]],
    *,
    maximum: float,
) -> None:
    for path, fraction in fractions:
        if fraction > maximum:
            raise TrainStage1Error(
                f"{path}: clip_fraction={fraction:.6g} exceeds "
                f"--max-clip-fraction={maximum:.6g}"
            )


def validate_formal_video_paths(config: TrainStage1Config) -> None:
    from data.cone_response import load_cone_response, validate_natural_video_splits

    if not config.val_h5:
        raise TrainStage1Error(
            "--formal-evidence requires a held-out validation split"
        )
    validate_natural_video_splits(
        tuple(load_cone_response(path) for path in config.train_h5),
        tuple(load_cone_response(path) for path in config.val_h5),
    )
