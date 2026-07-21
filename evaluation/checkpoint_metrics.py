from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, assert_never

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from evaluation.checkpoint_tensors import last_output
from evaluation.humret import smoothed_spike_probability_to_hz
from evaluation.reconstruction_baselines import (
    GlobalMeanBaseline,
    LocalLinearBaseline,
    LocalLinearSupport,
    fit_global_mean_baseline,
    fit_local_linear_baseline,
)
from models.cells.rgc import RGCOutput, RGCPopulationTensors
from training.hybrid import RetinaTrainingBatch
from training.stage1 import Stage1Components
from training.stage1_reporting import baseline_metrics
from training.stage1_runtime import batch_to_device


class CheckpointMetricsError(RuntimeError):
    pass


class ReconstructionMetrics(TypedDict):
    model_mse_current: float
    zero_contrast_mse_current: float
    global_mean_mse_current: float
    local_linear_mse_current: float
    best_baseline_mse_current: float
    skill_current: float


class BaselineMetrics(TypedDict):
    zero_contrast_mse_current: float
    global_mean_mse_current: float
    local_linear_mse_current: float
    best_baseline_mse_current: float


class PopulationUsageMetrics(TypedDict):
    population: str
    spike_per_bin: float
    smoothed_rate_per_bin: float
    smoothed_rate_hz: float


class PopulationAblationMetrics(TypedDict):
    population: str
    current_on_mse: float
    current_off_mse: float
    current_mse_delta: float
    current_contribution_abs: float


@dataclass(frozen=True, slots=True)
class FittedBaselines:
    global_mean: GlobalMeanBaseline
    local_linear: LocalLinearBaseline
    metrics: BaselineMetrics


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationRequest:
    components: Stage1Components
    loader: DataLoader[RetinaTrainingBatch]
    baselines: FittedBaselines
    device: torch.device


@dataclass(frozen=True, slots=True)
class HeldOutEvaluation:
    reconstruction: ReconstructionMetrics
    population_usage: tuple[PopulationUsageMetrics, ...]
    population_ablation: tuple[PopulationAblationMetrics, ...]


def fit_evaluation_baselines(
    train_loader: DataLoader[RetinaTrainingBatch],
    eval_loader: DataLoader[RetinaTrainingBatch],
    support: LocalLinearSupport,
) -> FittedBaselines:
    global_mean = fit_global_mean_baseline(train_loader)
    local_linear = fit_local_linear_baseline(train_loader, support)
    metrics = baseline_metrics(eval_loader, global_mean, local_linear)
    return FittedBaselines(
        global_mean=global_mean,
        local_linear=local_linear,
        metrics=BaselineMetrics(**metrics),
    )


def evaluate_held_out(request: HeldOutEvaluationRequest) -> HeldOutEvaluation:
    components = request.components
    components.core.eval()
    components.decoder.eval()
    model_sse = 0.0
    target_weight = 0.0
    usage = {name: [0.0, 0, 0.0, 0] for name in _POPULATION_NAMES}
    ablation = {name: [0.0] * 3 for name in _POPULATION_NAMES}

    with torch.no_grad():
        for batch in request.loader:
            device_batch = batch_to_device(batch, request.device)
            history, _ = components.core.forward_sequence(device_batch.x_cone)
            final = last_output(history)
            reconstruction = components.decoder(final)
            current_weight = float(device_batch.targets.target_current.numel())
            model_sse += float(
                F.mse_loss(
                    reconstruction.target_current,
                    device_batch.targets.target_current,
                )
            ) * current_weight
            target_weight += current_weight
            _accumulate_usage(usage, history)
            _accumulate_ablation(ablation, components, final, device_batch)

    if target_weight <= 0:
        raise CheckpointMetricsError("Held-out loader contains no evaluation samples")
    model_mse = model_sse / target_weight
    reconstruction_metrics = _reconstruction_metrics(model_mse, request.baselines.metrics)
    return HeldOutEvaluation(
        reconstruction=reconstruction_metrics,
        population_usage=_usage_metrics(
            usage,
            components.profile.rgc.dt_ms,
        ),
        population_ablation=_ablation_metrics(
            ablation,
            target_weight,
        ),
    )


def _reconstruction_metrics(
    model_mse: float,
    baseline: BaselineMetrics,
) -> ReconstructionMetrics:
    best = min(
        baseline["zero_contrast_mse_current"],
        baseline["global_mean_mse_current"],
    )
    if best <= 0:
        raise CheckpointMetricsError("Best baseline MSE must be positive")
    return ReconstructionMetrics(
        **baseline,
        model_mse_current=model_mse,
        skill_current=1.0 - model_mse / best,
    )


def _accumulate_usage(
    totals: dict[str, list[float | int]],
    output: RGCOutput,
) -> None:
    for name, spikes, rates in _population_tensors(output):
        values = totals[name]
        values[0] += spikes.sum().item()
        values[1] += spikes.numel()
        values[2] += rates.sum().item()
        values[3] += rates.numel()


def _accumulate_ablation(
    totals: dict[str, list[float]],
    components: Stage1Components,
    final: RGCOutput,
    batch: RetinaTrainingBatch,
) -> None:
    weight = float(batch.targets.target_current.numel())
    on = components.decoder(final).target_current
    for name in _POPULATION_NAMES:
        off = components.decoder(_zero_population(final, name)).target_current
        values = totals[name]
        values[0] += (
            float(
                F.mse_loss(
                    on,
                    batch.targets.target_current,
                )
            )
            * weight
        )
        values[1] += (
            float(
                F.mse_loss(
                    off,
                    batch.targets.target_current,
                )
            )
            * weight
        )
        values[2] += float(
            (on - off).abs().sum()
        )


def _usage_metrics(
    totals: dict[str, list[float | int]],
    dt_ms: float,
) -> tuple[PopulationUsageMetrics, ...]:
    results = []
    for name in _POPULATION_NAMES:
        spike_sum, spike_count, rate_sum, rate_count = totals[name]
        spike_mean = float(spike_sum) / int(spike_count)
        rate_mean = float(rate_sum) / int(rate_count)
        rate_hz = smoothed_spike_probability_to_hz(
            torch.tensor(rate_mean),
            dt_ms,
        ).item()
        results.append(
            PopulationUsageMetrics(
                population=name,
                spike_per_bin=spike_mean,
                smoothed_rate_per_bin=rate_mean,
                smoothed_rate_hz=rate_hz,
            )
        )
    return tuple(results)


def _ablation_metrics(
    totals: dict[str, list[float]],
    weight: float,
) -> tuple[PopulationAblationMetrics, ...]:
    results = []
    for name in _POPULATION_NAMES:
        values = totals[name]
        current = tuple(value / weight for value in values)
        results.append(
            PopulationAblationMetrics(
                population=name,
                current_on_mse=current[0],
                current_off_mse=current[1],
                current_mse_delta=current[1] - current[0],
                current_contribution_abs=current[2],
            )
        )
    return tuple(results)


PopulationName = Literal["midget", "parasol"]
_POPULATION_NAMES: tuple[PopulationName, ...] = ("midget", "parasol")


def _population_tensors(output: RGCOutput) -> tuple[
    tuple[PopulationName, torch.Tensor, torch.Tensor], ...
]:
    return (
        ("midget", output.spikes.midget, output.rates.midget),
        ("parasol", output.spikes.parasol, output.rates.parasol),
    )


def _zero_population(output: RGCOutput, population: PopulationName) -> RGCOutput:
    return RGCOutput(
        spikes=_zero_population_tensors(output.spikes, population),
        rates=_zero_population_tensors(output.rates, population),
    )


def _zero_population_tensors(
    tensors: RGCPopulationTensors,
    population: PopulationName,
) -> RGCPopulationTensors:
    match population:
        case "midget":
            return RGCPopulationTensors(
                midget=torch.zeros_like(tensors.midget),
                parasol=tensors.parasol,
            )
        case "parasol":
            return RGCPopulationTensors(
                midget=tensors.midget,
                parasol=torch.zeros_like(tensors.parasol),
            )
        case unreachable:
            assert_never(unreachable)
