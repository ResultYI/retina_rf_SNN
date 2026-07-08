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

from loss.retina import RetinaLossConfig, RetinaObjective
from training.hybrid import (
    HybridRetinaTrainer,
    HybridTrainingConfig,
    RetinaTrainingBatch,
    TrainingStage,
)
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


@dataclass(frozen=True, slots=True)
class Stage1Loaders:
    train: DataLoader[RetinaTrainingBatch]
    val: DataLoader[RetinaTrainingBatch] | None


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    components = _build_components(config)
    loaders = _build_loaders(config, components)
    optimizer = build_stage1_optimizer(
        components.core,
        components.decoder,
        Stage1OptimizerConfig(config.lr_core, config.lr_decoder),
    )
    trainer = HybridRetinaTrainer(
        components.core.to(config.device),
        components.decoder.to(config.device),
        RetinaObjective(RetinaLossConfig()).to(config.device),
        optimizer,
        HybridTrainingConfig(t_bptt=config.t_bptt),
    )

    csv_path = config.output_dir / "stage1_log.csv"
    jsonl_path = config.output_dir / "stage1_log.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=_fieldnames())
            writer.writeheader()
            _train_epochs(config, trainer, loaders, writer, jsonl_handle)
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
    args = parser.parse_args(argv)
    return TrainStage1Config(
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
    )


def _build_components(config: TrainStage1Config) -> Stage1Components:
    from configs.physiology_profiles import dt_ms_from_time_axis_seconds
    from data.cone_response import load_cone_response

    export = load_cone_response(config.train_h5[0])
    return build_stage1_components(
        export.positions_degs,
        Stage1BuildConfig(
            dt_ms=dt_ms_from_time_axis_seconds(export.time_axis_seconds),
            horizon_count=len(config.horizons),
        ),
    )


def _build_loaders(
    config: TrainStage1Config,
    components: Stage1Components,
) -> Stage1Loaders:
    from data.dataset import fit_log_cone_stats, save_log_cone_stats
    from datasets.isetbio_h5_dataset import (
        ConeNormalizationStats,
        ISETBioH5Dataset,
        ISETBioH5DatasetConfig,
        collate_isetbio_h5_batch,
    )

    mean, scale = fit_log_cone_stats(config.train_h5)
    save_log_cone_stats(config.output_dir / "normalization_stats.npz", mean, scale)
    stats = ConeNormalizationStats(mean, scale)
    train = ConcatDataset(
        [
            ISETBioH5Dataset(_dataset_config(path, config, components), stats)
            for path in config.train_h5
        ]
    )
    val = None
    if config.val_h5:
        val = DataLoader(
            ConcatDataset(
                [
                    ISETBioH5Dataset(_dataset_config(path, config, components), stats)
                    for path in config.val_h5
                ]
            ),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_isetbio_h5_batch,
        )
    return Stage1Loaders(
        train=DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_isetbio_h5_batch,
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
) -> None:
    step = 0
    for epoch in range(config.epochs):
        for batch in loaders.train:
            step += 1
            result = trainer.train_batch(_batch_to_device(batch, config.device), config.stage)
            row = _row("train", epoch, step, result.losses.total)
            writer.writerow(row)
            jsonl_handle.write(json.dumps(row) + "\n")
        _write_checkpoint(config, trainer, epoch, step)


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
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "core": trainer.core.state_dict(),
            "decoder": trainer.decoder.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
        },
        config.output_dir / "checkpoint.pt",
    )


def _row(split: str, epoch: int, step: int, loss: torch.Tensor) -> dict[str, str]:
    return {
        "split": split,
        "epoch": str(epoch),
        "step": str(step),
        "loss_total": f"{float(loss):.8g}",
    }


def _fieldnames() -> tuple[str, ...]:
    return ("split", "epoch", "step", "loss_total")


def _parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(int(value) for value in raw.split(",") if value)
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise TrainStage1Error("horizons must be comma-separated positive integers")
    return horizons


if __name__ == "__main__":
    raise SystemExit(main())
