from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prediction_baselines import baseline_mse, fit_global_change_baseline
from loss.retina import RetinaLossConfig, RetinaObjective
from training.hybrid import (
    HybridRetinaTrainer,
    HybridTrainingConfig,
    RetinaTrainingBatch,
    TrainingStepResult,
    TrainingStage,
)
from training.epoch_metrics import weighted_mean_row
from training.stage1 import (
    Stage1BuildConfig,
    Stage1Components,
    Stage1OptimizerConfig,
    build_stage1_components,
    build_stage1_optimizer,
)


class TrainStage1Error(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainStage1Config:
    train_h5: tuple[Path, ...]
    val_h5: tuple[Path, ...]
    output_dir: Path
    epochs: int
    batch_size: int
    input_steps: int
    horizons: tuple[int, ...]
    stage: TrainingStage
    device: torch.device
    seed: int
    t_bptt: int
    lr_core: float
    lr_decoder: float
    num_workers: int
    max_clip_fraction: float
    resume: Path | None


@dataclass(frozen=True, slots=True)
class Stage1Loaders:
    train: DataLoader[RetinaTrainingBatch]
    val: DataLoader[RetinaTrainingBatch] | None


@dataclass(frozen=True, slots=True)
class CheckpointProgress:
    epoch: int
    step: int
    best_loss: float


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    _seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    components = _build_components(config)
    loaders = _build_loaders(config, components)
    train_baselines = _write_baseline_summary(config, loaders)
    optimizer = build_stage1_optimizer(
        components.core,
        components.decoder,
        Stage1OptimizerConfig(config.lr_core, config.lr_decoder),
    )
    trainer = HybridRetinaTrainer(
        components.core.to(config.device),
        components.decoder.to(config.device),
        RetinaObjective(
            _loss_config_from_train_metrics(train_baselines)
        ).to(config.device),
        optimizer,
        HybridTrainingConfig(t_bptt=config.t_bptt),
    )

    progress = _restore_checkpoint(config, trainer)
    csv_path = config.output_dir / "stage1_log.csv"
    jsonl_path = config.output_dir / "stage1_log.jsonl"
    csv_mode = "a" if progress.epoch else "w"
    jsonl_mode = "a" if progress.epoch else "w"
    with csv_path.open(csv_mode, newline="", encoding="utf-8") as csv_handle:
        with jsonl_path.open(jsonl_mode, encoding="utf-8") as jsonl_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=_fieldnames())
            if not progress.epoch:
                writer.writeheader()
            _train_epochs(
                config,
                trainer,
                loaders,
                writer,
                jsonl_handle,
                progress,
            )
    return 0


def _parse_args(argv: Sequence[str] | None) -> TrainStage1Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5", nargs="+", type=Path, required=True)
    parser.add_argument("--val-h5", nargs="*", type=Path, default=())
    parser.add_argument("--output-dir", type=Path, default=Path("runs/stage1"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-steps", type=int, default=16)
    parser.add_argument("--horizons", default="1,2,4")
    parser.add_argument(
        "--stage",
        choices=tuple(stage.value for stage in TrainingStage),
        default=TrainingStage.DECODER_WARMUP.value,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t-bptt", type=int, default=8)
    parser.add_argument("--lr-core", type=float, default=1e-4)
    parser.add_argument("--lr-decoder", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-clip-fraction", type=float, default=0.01)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    config = TrainStage1Config(
        train_h5=tuple(args.train_h5),
        val_h5=tuple(args.val_h5),
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_steps=args.input_steps,
        horizons=_parse_horizons(args.horizons),
        stage=TrainingStage(args.stage),
        device=torch.device(args.device),
        seed=args.seed,
        t_bptt=args.t_bptt,
        lr_core=args.lr_core,
        lr_decoder=args.lr_decoder,
        num_workers=args.num_workers,
        max_clip_fraction=args.max_clip_fraction,
        resume=args.resume,
    )
    _validate_config(config)
    return config


def _build_components(config: TrainStage1Config) -> Stage1Components:
    from configs.physiology_profiles import dt_ms_from_time_axis_seconds
    from data.cone_response import load_cone_response

    export = load_cone_response(config.train_h5[0])
    return build_stage1_components(
        export.positions_degs,
        Stage1BuildConfig(
            dt_ms=dt_ms_from_time_axis_seconds(export.time_axis_seconds),
            horizon_count=len(config.horizons),
            eccentricity_deg=export.eccentricity_deg,
        ),
    )


def _build_loaders(
    config: TrainStage1Config,
    components: Stage1Components,
) -> Stage1Loaders:
    from data.dataset import (
        fit_log_cone_stats,
        save_log_cone_stats,
        validate_compatible_cone_exports,
    )
    from datasets.isetbio_h5_dataset import (
        ConeNormalizationStats,
        ISETBioH5Dataset,
        ISETBioH5DatasetConfig,
        collate_isetbio_h5_batch,
    )

    validate_compatible_cone_exports((*config.train_h5, *config.val_h5))
    mean, scale = fit_log_cone_stats(config.train_h5)
    save_log_cone_stats(config.output_dir / "normalization_stats.npz", mean, scale)
    stats = ConeNormalizationStats(mean, scale)
    train_datasets = [
        ISETBioH5Dataset(_dataset_config(path, config, components), stats)
        for path in config.train_h5
    ]
    train = ConcatDataset(train_datasets)
    val = None
    val_datasets = []
    if config.val_h5:
        val_datasets = [
            ISETBioH5Dataset(_dataset_config(path, config, components), stats)
            for path in config.val_h5
        ]
        val = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_isetbio_h5_batch,
        )
    _write_data_summary(config.output_dir, train_datasets, val_datasets)
    _validate_clip_fractions(
        tuple(
            (path, dataset.clip_fraction)
            for path, dataset in zip(config.train_h5, train_datasets)
        )
        + tuple(
            (path, dataset.clip_fraction)
            for path, dataset in zip(config.val_h5, val_datasets)
        ),
        maximum=config.max_clip_fraction,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return Stage1Loaders(
        train=DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_isetbio_h5_batch,
            generator=generator,
        ),
        val=val,
    )


def _dataset_config(
    path: Path,
    config: TrainStage1Config,
    components: Stage1Components,
):
    from datasets.isetbio_h5_dataset import ISETBioH5DatasetConfig

    return ISETBioH5DatasetConfig(
        h5_path=path,
        input_steps=config.input_steps,
        horizons=config.horizons,
        target_fine_pool=components.target_pools.fine,
        target_coarse_pool=components.target_pools.coarse,
    )


def _train_epochs(
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
                _batch_to_device(batch, config.device),
                config.stage,
            )
            _write_row(writer, jsonl_handle, _row("train", epoch, step, result))
        summaries: dict[str, dict[str, str]] = {}
        for split, loader in (
            ("train_eval", _baseline_loader(loaders.train, config)),
            ("val", loaders.val),
        ):
            if loader is not None:
                rows = []
                for batch in loader:
                    result = trainer.evaluate_batch(
                        _batch_to_device(batch, config.device)
                    )
                    rows.append(
                        (
                            batch.x_cone.shape[0],
                            _row(split, epoch, step, result),
                        )
                    )
                summary = weighted_mean_row(rows)
                summaries[split] = summary
                _write_row(writer, jsonl_handle, summary)
        selected_summary = (
            summaries["val"] if loaders.val is not None else summaries["train_eval"]
        )
        selected_loss = float(selected_summary["loss_total"])
        is_best = selected_loss < best_loss
        if is_best:
            best_loss = selected_loss
        _write_checkpoint(config, trainer, epoch, step, best_loss, is_best)


def _batch_to_device(
    batch: RetinaTrainingBatch,
    device: torch.device,
) -> RetinaTrainingBatch:
    return RetinaTrainingBatch(
        x_cone=batch.x_cone.to(device),
        targets=type(batch.targets)(
            fine=batch.targets.fine.to(device),
            coarse=batch.targets.coarse.to(device),
        ),
    )


def _write_checkpoint(
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


def _restore_checkpoint(
    config: TrainStage1Config,
    trainer: HybridRetinaTrainer,
) -> CheckpointProgress:
    if config.resume is None:
        return CheckpointProgress(0, 0, float("inf"))
    checkpoint = torch.load(config.resume, map_location=config.device, weights_only=False)
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
    _move_optimizer_state(trainer.optimizer, config.device)
    return CheckpointProgress(
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["step"]),
        float(checkpoint.get("best_loss", float("inf"))),
    )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _row(
    split: str,
    epoch: int,
    step: int,
    result: TrainingStepResult,
) -> dict[str, str]:
    losses = result.losses
    rgc = result.core_diagnostics["rgc"]
    h1 = result.core_diagnostics["h1"]
    a2 = result.core_diagnostics["a2"]
    decoder = result.decoder_diagnostics
    return {
        "split": split,
        "epoch": str(epoch),
        "step": str(step),
        "loss_total": _format(losses.total),
        "loss_fine": _format(losses.prediction_fine),
        "loss_coarse": _format(losses.prediction_coarse),
        "loss_residual_activity": _format(losses.residual_activity),
        "loss_residual_decoder": _format(losses.residual_decoder_weight),
        "rgc_midget_rate": _format(rgc["rgc_midget_rate_mean"]),
        "rgc_parasol_rate": _format(rgc["rgc_parasol_rate_mean"]),
        "rgc_residual_rate": _format(rgc["rgc_residual_rate_mean"]),
        "rgc_midget_spike_per_bin": _format(rgc["rgc_midget_spike_mean"]),
        "rgc_parasol_spike_per_bin": _format(rgc["rgc_parasol_spike_mean"]),
        "rgc_residual_spike_per_bin": _format(rgc["rgc_residual_spike_mean"]),
        "h1_gain": _format(h1["h1_gain"]),
        "a2_g_ba_sustained": _format(a2["a2_g_ba"][0]),
        "a2_g_ba_transient": _format(a2["a2_g_ba"][1]),
        "rgc_g_ag_midget": _format(rgc["rgc_g_ag"][0]),
        "rgc_g_ag_parasol": _format(rgc["rgc_g_ag"][1]),
        "rgc_g_ag_residual": _format(rgc["rgc_g_ag"][2]),
        "decoder_residual_weight_norm": _format(
            decoder["decoder_residual_weight_norm"]
        ),
        "decoder_midget_weight_norm": _format(
            decoder["decoder_midget_weight_norm"]
        ),
        "decoder_parasol_weight_norm": _format(
            decoder["decoder_parasol_weight_norm"]
        ),
    }


def _fieldnames() -> tuple[str, ...]:
    return (
        "split",
        "epoch",
        "step",
        "loss_total",
        "loss_fine",
        "loss_coarse",
        "loss_residual_activity",
        "loss_residual_decoder",
        "rgc_midget_rate",
        "rgc_parasol_rate",
        "rgc_residual_rate",
        "rgc_midget_spike_per_bin",
        "rgc_parasol_spike_per_bin",
        "rgc_residual_spike_per_bin",
        "h1_gain",
        "a2_g_ba_sustained",
        "a2_g_ba_transient",
        "rgc_g_ag_midget",
        "rgc_g_ag_parasol",
        "rgc_g_ag_residual",
        "decoder_residual_weight_norm",
        "decoder_midget_weight_norm",
        "decoder_parasol_weight_norm",
    )


def _parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(int(value) for value in raw.split(",") if value)
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise TrainStage1Error("horizons must be comma-separated positive integers")
    return horizons


def _write_data_summary(
    output_dir: Path,
    train_datasets,
    val_datasets,
) -> None:
    def summary(dataset) -> dict[str, float | int]:
        return {
            "samples": len(dataset),
            "cone_count": int(dataset.positions_degs.shape[0]),
            "dt_ms": dataset.dt_ms,
            "eccentricity_deg": dataset.eccentricity_deg,
            "clip_fraction": dataset.clip_fraction,
        }

    payload = {
        "train": [summary(dataset) for dataset in train_datasets],
        "val": [summary(dataset) for dataset in val_datasets],
    }
    (output_dir / "data_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_baseline_summary(
    config: TrainStage1Config,
    loaders: Stage1Loaders,
) -> dict[str, float]:
    baseline = fit_global_change_baseline(_baseline_loader(loaders.train, config))
    train_metrics = _baseline_metrics(
        _baseline_loader(loaders.train, config),
        baseline,
    )
    payload = {
        "train": train_metrics,
        "global_change_mean": {
            "fine": baseline.fine_mean.tolist(),
            "coarse": baseline.coarse_mean.tolist(),
        },
    }
    if loaders.val is not None:
        payload["val"] = _baseline_metrics(
            _baseline_loader(loaders.val, config),
            baseline,
        )
    (config.output_dir / "prediction_baselines.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return train_metrics


def _loss_config_from_train_metrics(
    train_metrics: dict[str, float],
) -> RetinaLossConfig:
    fine = train_metrics["zero_change_mse_fine"]
    coarse = train_metrics["zero_change_mse_coarse"]
    if not all(torch.isfinite(torch.tensor((fine, coarse)))) or min(fine, coarse) <= 0:
        raise TrainStage1Error(
            "Train zero-change MSE must be finite and positive for both scales"
        )
    return RetinaLossConfig(
        fine_prediction_scale=fine,
        coarse_prediction_scale=coarse,
    )


def _baseline_loader(
    loader: DataLoader[RetinaTrainingBatch],
    config: TrainStage1Config,
) -> DataLoader[RetinaTrainingBatch]:
    return DataLoader(
        loader.dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=loader.collate_fn,
    )


def _baseline_metrics(loader, baseline) -> dict[str, float]:
    totals = {
        "zero_change_mse_fine": 0.0,
        "zero_change_mse_coarse": 0.0,
        "global_change_mse_fine": 0.0,
        "global_change_mse_coarse": 0.0,
    }
    counts = {"fine": 0, "coarse": 0}
    for batch in loader:
        mse = baseline_mse(baseline, batch.targets)
        fine_count = batch.targets.fine.numel()
        coarse_count = batch.targets.coarse.numel()
        totals["zero_change_mse_fine"] += float(mse.zero_fine) * fine_count
        totals["global_change_mse_fine"] += float(mse.global_fine) * fine_count
        totals["zero_change_mse_coarse"] += float(mse.zero_coarse) * coarse_count
        totals["global_change_mse_coarse"] += float(mse.global_coarse) * coarse_count
        counts["fine"] += fine_count
        counts["coarse"] += coarse_count
    return {
        "zero_change_mse_fine": totals["zero_change_mse_fine"] / counts["fine"],
        "zero_change_mse_coarse": totals["zero_change_mse_coarse"] / counts["coarse"],
        "global_change_mse_fine": totals["global_change_mse_fine"] / counts["fine"],
        "global_change_mse_coarse": totals["global_change_mse_coarse"] / counts["coarse"],
    }


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: TrainStage1Config) -> None:
    if config.epochs < 1 or config.batch_size < 1 or config.t_bptt < 1:
        raise TrainStage1Error("epochs, batch_size, and t_bptt must be positive")
    if config.lr_core <= 0 or config.lr_decoder <= 0:
        raise TrainStage1Error("Learning rates must be positive")
    if config.num_workers < 0:
        raise TrainStage1Error("num_workers must be non-negative")
    if not 0 <= config.max_clip_fraction <= 1:
        raise TrainStage1Error("max_clip_fraction must lie in [0, 1]")
    if config.resume is not None and not config.resume.is_file():
        raise TrainStage1Error(f"Checkpoint does not exist: {config.resume}")


def _validate_clip_fractions(
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


def _format(value: torch.Tensor) -> str:
    return f"{float(value):.8g}"


def _write_row(
    writer: csv.DictWriter[TextIO],
    jsonl_handle: TextIO,
    row: dict[str, str],
) -> None:
    writer.writerow(row)
    jsonl_handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
