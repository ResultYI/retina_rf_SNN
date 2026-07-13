from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loss.retina import RetinaObjective
from training.hybrid import (
    HybridRetinaTrainer,
    HybridTrainingConfig,
    TrainingStage,
)
from training.stage1 import (
    MidgetSamplingMode,
    Stage1BuildConfig,
    Stage1Components,
    Stage1OptimizerConfig,
    build_stage1_components,
    build_stage1_optimizer,
)
from training.stage1_reporting import (
    loss_config_from_train_metrics as _loss_config_from_train_metrics,
    stage1_log_fieldnames as _fieldnames,
    write_baseline_summary as _write_baseline_summary,
    write_data_summary as _write_data_summary,
    write_parameter_audit as _write_parameter_audit,
)
from training.stage1_runtime import (
    restore_checkpoint as _restore_checkpoint,
    seed_everything as _seed_everything,
    train_epochs as _train_epochs,
    validate_clip_fractions as _validate_clip_fractions,
    validate_config as _validate_config,
    validate_formal_video_paths as _validate_formal_video_paths,
)
from training.stage1_types import (
    CheckpointProgress,
    Stage1Loaders,
    TrainStage1Config,
    TrainStage1Error,
)


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    _seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.formal_evidence:
        _validate_formal_video_paths(config)

    components = _build_components(config)
    loaders = _build_loaders(config, components)
    train_baselines = _write_baseline_summary(config, loaders, components)
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
    _write_parameter_audit(config.output_dir, components)
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
    parser.add_argument("--formal-evidence", action="store_true")
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
        formal_evidence=args.formal_evidence,
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
            midget_sampling=(
                MidgetSamplingMode.FOVEAL_PRIVATE_LINE
                if export.eccentricity_deg == 0
                else MidgetSamplingMode.CONVERGENT
            ),
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


def _parse_horizons(raw: str) -> tuple[int, ...]:
    horizons = tuple(int(value) for value in raw.split(",") if value)
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise TrainStage1Error("horizons must be comma-separated positive integers")
    return horizons


if __name__ == "__main__":
    raise SystemExit(main())
