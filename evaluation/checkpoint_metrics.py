from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from torch.utils.data import DataLoader

from evaluation.checkpoint_tensors import concat_outputs, last_output, slice_output
from evaluation.humret import smoothed_spike_probability_to_hz
from evaluation.prediction_baselines import (
    GlobalChangeBaseline,
    LocalARBaseline,
    LocalARSupports,
    fit_global_change_baseline,
    fit_local_ar_baseline,
)
from evaluation.residual_ablation import population_ablation_report
from models.cells.rgc import RGCOutput
from training.hybrid import RetinaTrainingBatch
from training.stage1 import Stage1Components
from training.stage1_reporting import baseline_metrics
from training.stage1_runtime import batch_to_device


class CheckpointMetricsError(RuntimeError):
    pass


class PredictionMetrics(TypedDict):
    model_mse_fine: float
    model_mse_coarse: float
    zero_change_mse_fine: float
    zero_change_mse_coarse: float
    global_change_mse_fine: float
    global_change_mse_coarse: float
    local_ar_mse_fine: float
    local_ar_mse_coarse: float
    best_baseline_mse_fine: float
    best_baseline_mse_coarse: float
    skill_fine: float
    skill_coarse: float


class BaselineMetrics(TypedDict):
    zero_change_mse_fine: float
    zero_change_mse_coarse: float
    global_change_mse_fine: float
    global_change_mse_coarse: float
    local_ar_mse_fine: float
    local_ar_mse_coarse: float


class PopulationUsageMetrics(TypedDict):
    population: str
    spike_per_bin: float
    smoothed_rate_per_bin: float
    smoothed_rate_hz: float


class PopulationAblationMetrics(TypedDict):
    population: str
    fine_on_mse: float
    fine_off_mse: float
    fine_mse_delta: float
    fine_contribution_abs: float
    coarse_on_mse: float
    coarse_off_mse: float
    coarse_mse_delta: float
    coarse_contribution_abs: float


@dataclass(frozen=True, slots=True)
class FittedBaselines:
    global_change: GlobalChangeBaseline
    local_ar: LocalARBaseline
    metrics: BaselineMetrics


@dataclass(frozen=True, slots=True)
class HeldOutEvaluationRequest:
    components: Stage1Components
    loader: DataLoader[RetinaTrainingBatch]
    baselines: FittedBaselines
    device: torch.device
    probe_sample_count: int


@dataclass(frozen=True, slots=True)
class HeldOutEvaluation:
    prediction: PredictionMetrics
    population_usage: tuple[PopulationUsageMetrics, ...]
    population_ablation: tuple[PopulationAblationMetrics, ...]
    probe_stimuli: torch.Tensor
    probe_output: RGCOutput


def fit_evaluation_baselines(
    train_loader: DataLoader[RetinaTrainingBatch],
    eval_loader: DataLoader[RetinaTrainingBatch],
    supports: LocalARSupports,
) -> FittedBaselines:
    global_change = fit_global_change_baseline(train_loader)
    local_ar = fit_local_ar_baseline(train_loader, supports)
    metrics = baseline_metrics(eval_loader, global_change, local_ar)
    return FittedBaselines(
        global_change=global_change,
        local_ar=local_ar,
        metrics=BaselineMetrics(**metrics),
    )


def evaluate_held_out(request: HeldOutEvaluationRequest) -> HeldOutEvaluation:
    components = request.components
    components.core.eval()
    components.decoder.eval()
    model_sse = [0.0, 0.0]
    target_counts = [0, 0]
    usage = {name: [0.0, 0, 0.0, 0] for name in _POPULATION_NAMES}
    ablation = {name: [0.0] * 6 for name in _POPULATION_NAMES}
    probe_stimuli: list[torch.Tensor] = []
    probe_outputs: list[RGCOutput] = []
    probe_count = 0

    with torch.no_grad():
        for batch in request.loader:
            device_batch = batch_to_device(batch, request.device)
            history, _ = components.core.forward_sequence(device_batch.x_cone)
            final = last_output(history)
            prediction = components.decoder(final)
            fine_error = prediction.target_fine - device_batch.targets.fine
            coarse_error = prediction.target_coarse - device_batch.targets.coarse
            model_sse[0] += fine_error.square().sum().item()
            model_sse[1] += coarse_error.square().sum().item()
            target_counts[0] += fine_error.numel()
            target_counts[1] += coarse_error.numel()
            _accumulate_usage(usage, history)
            _accumulate_ablation(ablation, components, final, device_batch)
            remaining = request.probe_sample_count - probe_count
            if remaining > 0:
                take = min(remaining, device_batch.x_cone.shape[0])
                probe_stimuli.append(device_batch.x_cone[:take].detach())
                probe_outputs.append(slice_output(history, take))
                probe_count += take

    if min(target_counts) == 0 or probe_count == 0:
        raise CheckpointMetricsError("Held-out loader contains no evaluation samples")
    model_mse = (model_sse[0] / target_counts[0], model_sse[1] / target_counts[1])
    prediction_metrics = _prediction_metrics(model_mse, request.baselines.metrics)
    return HeldOutEvaluation(
        prediction=prediction_metrics,
        population_usage=_usage_metrics(
            usage,
            components.profile.rgc.dt_ms,
        ),
        population_ablation=_ablation_metrics(
            ablation,
            target_counts,
        ),
        probe_stimuli=torch.cat(probe_stimuli, dim=0),
        probe_output=concat_outputs(probe_outputs),
    )


def _prediction_metrics(
    model_mse: tuple[float, float],
    baseline: BaselineMetrics,
) -> PredictionMetrics:
    best_fine = min(
        baseline["zero_change_mse_fine"],
        baseline["global_change_mse_fine"],
        baseline["local_ar_mse_fine"],
    )
    best_coarse = min(
        baseline["zero_change_mse_coarse"],
        baseline["global_change_mse_coarse"],
        baseline["local_ar_mse_coarse"],
    )
    if min(best_fine, best_coarse) <= 0:
        raise CheckpointMetricsError("Best baseline MSE must be positive")
    return PredictionMetrics(
        **baseline,
        model_mse_fine=model_mse[0],
        model_mse_coarse=model_mse[1],
        best_baseline_mse_fine=best_fine,
        best_baseline_mse_coarse=best_coarse,
        skill_fine=1.0 - model_mse[0] / best_fine,
        skill_coarse=1.0 - model_mse[1] / best_coarse,
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
    fine_count = batch.targets.fine.numel()
    coarse_count = batch.targets.coarse.numel()
    for name in _POPULATION_NAMES:
        report = population_ablation_report(
            components.decoder,
            final,
            name,
            batch.targets,
        )
        values = totals[name]
        values[0] += report.fine_on_mse * fine_count
        values[1] += report.fine_off_mse * fine_count
        values[2] += report.fine_contribution * fine_count
        values[3] += report.coarse_on_mse * coarse_count
        values[4] += report.coarse_off_mse * coarse_count
        values[5] += report.coarse_contribution * coarse_count


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
    counts: list[int],
) -> tuple[PopulationAblationMetrics, ...]:
    results = []
    for name in _POPULATION_NAMES:
        values = totals[name]
        fine = tuple(value / counts[0] for value in values[:3])
        coarse = tuple(value / counts[1] for value in values[3:])
        results.append(
            PopulationAblationMetrics(
                population=name,
                fine_on_mse=fine[0],
                fine_off_mse=fine[1],
                fine_mse_delta=fine[1] - fine[0],
                fine_contribution_abs=fine[2],
                coarse_on_mse=coarse[0],
                coarse_off_mse=coarse[1],
                coarse_mse_delta=coarse[1] - coarse[0],
                coarse_contribution_abs=coarse[2],
            )
        )
    return tuple(results)


PopulationName = Literal["midget", "parasol", "residual"]
_POPULATION_NAMES: tuple[PopulationName, ...] = ("midget", "parasol", "residual")


def _population_tensors(output: RGCOutput) -> tuple[
    tuple[PopulationName, torch.Tensor, torch.Tensor], ...
]:
    return (
        ("midget", output.spikes.midget, output.rates.midget),
        ("parasol", output.spikes.parasol, output.rates.parasol),
        ("residual", output.spikes.residual, output.rates.residual),
    )
