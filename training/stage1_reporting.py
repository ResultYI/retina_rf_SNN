from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TextIO

import torch
from torch.utils.data import DataLoader

from evaluation.parameter_audit import audit_stage1_parameters
from evaluation.prediction_baselines import (
    GlobalChangeBaseline,
    LocalARBaseline,
    LocalARSupports,
    baseline_mse,
    fit_global_change_baseline,
    fit_local_ar_baseline,
    predict_local_ar,
)
from loss.retina import RetinaLossConfig
from training.hybrid import RetinaTrainingBatch, TrainingStepResult
from training.stage1 import Stage1Components
from training.stage1_types import Stage1Loaders, TrainStage1Config, TrainStage1Error


class Stage1DatasetSummary(Protocol):
    positions_degs: torch.Tensor
    dt_ms: float
    eccentricity_deg: float
    clip_fraction: float

    def __len__(self) -> int: ...


def stage1_log_row(
    split: str,
    epoch: int,
    step: int,
    result: TrainingStepResult,
) -> dict[str, str]:
    losses = result.losses
    rgc = result.core_diagnostics["rgc"]
    h1 = result.core_diagnostics["h1"]
    amacrine = result.core_diagnostics["amacrine"]
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
        "amacrine_g_ba_sustained": _format(amacrine["amacrine_g_ba"][0]),
        "amacrine_g_ba_transient": _format(amacrine["amacrine_g_ba"][1]),
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


def stage1_log_fieldnames() -> tuple[str, ...]:
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
        "amacrine_g_ba_sustained",
        "amacrine_g_ba_transient",
        "rgc_g_ag_midget",
        "rgc_g_ag_parasol",
        "rgc_g_ag_residual",
        "decoder_residual_weight_norm",
        "decoder_midget_weight_norm",
        "decoder_parasol_weight_norm",
    )


def write_data_summary(
    output_dir: Path,
    train_datasets: Sequence[Stage1DatasetSummary],
    val_datasets: Sequence[Stage1DatasetSummary],
) -> None:
    def summary(dataset: Stage1DatasetSummary) -> dict[str, float | int]:
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


def write_baseline_summary(
    config: TrainStage1Config,
    loaders: Stage1Loaders,
    components: Stage1Components,
) -> dict[str, float]:
    train_loader = baseline_loader(loaders.train, config)
    baseline = fit_global_change_baseline(train_loader)
    local_ar = fit_local_ar_baseline(
        train_loader,
        LocalARSupports(
            components.target_pools.fine,
            components.target_pools.coarse,
        ),
    )
    train_metrics = baseline_metrics(
        baseline_loader(loaders.train, config),
        baseline,
        local_ar,
    )
    payload = {
        "train": train_metrics,
        "global_change_mean": {
            "fine": baseline.fine_mean.tolist(),
            "coarse": baseline.coarse_mean.tolist(),
        },
    }
    if loaders.val is not None:
        payload["val"] = baseline_metrics(
            baseline_loader(loaders.val, config),
            baseline,
            local_ar,
        )
    (config.output_dir / "prediction_baselines.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return train_metrics


def loss_config_from_train_metrics(
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


def baseline_loader(
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


def baseline_metrics(
    loader: DataLoader[RetinaTrainingBatch],
    baseline: GlobalChangeBaseline,
    local_ar: LocalARBaseline,
) -> dict[str, float]:
    totals = {
        "zero_change_mse_fine": 0.0,
        "zero_change_mse_coarse": 0.0,
        "global_change_mse_fine": 0.0,
        "global_change_mse_coarse": 0.0,
        "local_ar_mse_fine": 0.0,
        "local_ar_mse_coarse": 0.0,
    }
    counts = {"fine": 0, "coarse": 0}
    for batch in loader:
        mse = baseline_mse(baseline, batch.targets)
        local_prediction = predict_local_ar(local_ar, batch.x_cone)
        fine_count = batch.targets.fine.numel()
        coarse_count = batch.targets.coarse.numel()
        totals["zero_change_mse_fine"] += float(mse.zero_fine) * fine_count
        totals["global_change_mse_fine"] += float(mse.global_fine) * fine_count
        totals["zero_change_mse_coarse"] += float(mse.zero_coarse) * coarse_count
        totals["global_change_mse_coarse"] += float(mse.global_coarse) * coarse_count
        totals["local_ar_mse_fine"] += float(
            (local_prediction.fine - batch.targets.fine).square().mean()
        ) * fine_count
        totals["local_ar_mse_coarse"] += float(
            (local_prediction.coarse - batch.targets.coarse).square().mean()
        ) * coarse_count
        counts["fine"] += fine_count
        counts["coarse"] += coarse_count
    return {
        "zero_change_mse_fine": totals["zero_change_mse_fine"] / counts["fine"],
        "zero_change_mse_coarse": totals["zero_change_mse_coarse"] / counts["coarse"],
        "global_change_mse_fine": totals["global_change_mse_fine"] / counts["fine"],
        "global_change_mse_coarse": (
            totals["global_change_mse_coarse"] / counts["coarse"]
        ),
        "local_ar_mse_fine": totals["local_ar_mse_fine"] / counts["fine"],
        "local_ar_mse_coarse": totals["local_ar_mse_coarse"] / counts["coarse"],
    }


def write_parameter_audit(output_dir: Path, components: Stage1Components) -> None:
    payload = [
        {
            "name": item.name,
            "value": item.value,
            "lower": item.lower,
            "upper": item.upper,
            "boundary_fraction": item.boundary_fraction,
            "near_boundary": item.near_boundary,
        }
        for item in audit_stage1_parameters(components)
    ]
    (output_dir / "parameter_audit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_log_row(
    writer: csv.DictWriter[TextIO],
    jsonl_handle: TextIO,
    row: dict[str, str],
) -> None:
    writer.writerow(row)
    jsonl_handle.write(json.dumps(row) + "\n")


def _format(value: torch.Tensor) -> str:
    return f"{float(value):.8g}"
