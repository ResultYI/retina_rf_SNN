from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from evaluation.global_probe import (
    GlobalProbeInputs,
    GlobalProbeResult,
    GlobalReadoutGeometry,
    fit_global_probe,
)
from models.cells.rgc_runtime import causal_filter
from models.cells.rgc_types import RGCOutput
from models.retina_snn import RetinaModel
from training.augmentation import AugmentedClip


class ReadoutLadderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReadoutOperatingPoint:
    margin_mean: float
    margin_standard_deviation: float
    margin_quantile_05: float
    margin_median: float
    margin_quantile_95: float
    probability_below_001_fraction: float
    probability_above_099_fraction: float
    probability_saturated_fraction: float
    probability_variance: float
    hard_spike_fraction: float
    zero_spike_unit_fraction: float
    filtered_rate_variance: float


@dataclass(frozen=True, slots=True)
class ReadoutExamples:
    generator_potential: torch.Tensor
    spike_probability: torch.Tensor
    probability_rate: torch.Tensor
    hard_rate_10_ms: torch.Tensor
    hard_rate_20_ms: torch.Tensor
    hard_rate_50_ms: torch.Tensor
    hard_rate_100_ms: torch.Tensor
    filtered_rate: torch.Tensor
    target: torch.Tensor
    source_ids: Sequence[str]
    operating_point: ReadoutOperatingPoint


@dataclass(frozen=True, slots=True)
class ReadoutExampleRequest:
    output: RGCOutput
    target: torch.Tensor
    source_ids: Sequence[str]
    threshold: torch.Tensor
    supervised_steps: int
    dt_ms: float


@dataclass(frozen=True, slots=True)
class ModelReadoutLadderRequest:
    model: RetinaModel
    training_clips: Sequence[AugmentedClip]
    validation_clips: Sequence[AugmentedClip]
    supervised_steps: int
    dt_ms: float
    gain_max: float


@dataclass(frozen=True, slots=True)
class ReadoutLadderResult:
    generator_potential: GlobalProbeResult
    spike_probability: GlobalProbeResult
    probability_rate: GlobalProbeResult
    hard_rate_10_ms: GlobalProbeResult
    hard_rate_20_ms: GlobalProbeResult
    hard_rate_50_ms: GlobalProbeResult
    hard_rate_100_ms: GlobalProbeResult
    filtered_rate: GlobalProbeResult
    operating_point: ReadoutOperatingPoint


def build_readout_examples(request: ReadoutExampleRequest) -> ReadoutExamples:
    if request.supervised_steps < 1:
        raise ReadoutLadderError("supervised_steps must be positive")
    if len(request.source_ids) != request.output.rates.shape[0]:
        raise ReadoutLadderError("source IDs do not match batch rows")
    if request.target.shape[:2] != request.output.rates.shape[:2]:
        raise ReadoutLadderError("target and readout time axes must match")
    supervised = slice(-request.supervised_steps, None)
    probability_rate = causal_filter(
        request.output.spike_probability,
        dt_ms=request.dt_ms,
        tau_ms=50.0,
    )
    hard_rates = tuple(
        causal_filter(
            request.output.hard_spikes,
            dt_ms=request.dt_ms,
            tau_ms=tau_ms,
        )
        for tau_ms in (10.0, 20.0, 50.0, 100.0)
    )
    probability = request.output.spike_probability[:, supervised]
    hard = request.output.hard_spikes[:, supervised]
    filtered_rate = request.output.rates[:, supervised]
    margin = (
        request.output.generator_potential[:, supervised]
        - request.threshold.view(1, 1, 1, -1)
    )
    quantiles = torch.quantile(
        margin.detach().float(),
        torch.tensor(
            [0.05, 0.5, 0.95],
            device=margin.device,
            dtype=torch.float32,
        ),
    )
    saturated = (probability < 0.01) | (probability > 0.99)
    zero_spike_units = hard.sum(dim=(0, 1)) == 0
    operating_point = ReadoutOperatingPoint(
        margin_mean=float(margin.mean()),
        margin_standard_deviation=float(
            margin.std(unbiased=False)
        ),
        margin_quantile_05=float(quantiles[0]),
        margin_median=float(quantiles[1]),
        margin_quantile_95=float(quantiles[2]),
        probability_below_001_fraction=float((probability < 0.01).float().mean()),
        probability_above_099_fraction=float((probability > 0.99).float().mean()),
        probability_saturated_fraction=float(saturated.float().mean()),
        probability_variance=float(probability.var(unbiased=False)),
        hard_spike_fraction=float(hard.mean()),
        zero_spike_unit_fraction=float(zero_spike_units.float().mean()),
        filtered_rate_variance=float(filtered_rate.var(unbiased=False)),
    )
    return ReadoutExamples(
        generator_potential=request.output.generator_potential[:, supervised],
        spike_probability=probability,
        probability_rate=probability_rate[:, supervised],
        hard_rate_10_ms=hard_rates[0][:, supervised],
        hard_rate_20_ms=hard_rates[1][:, supervised],
        hard_rate_50_ms=hard_rates[2][:, supervised],
        hard_rate_100_ms=hard_rates[3][:, supervised],
        filtered_rate=filtered_rate,
        target=request.target[:, supervised],
        source_ids=request.source_ids,
        operating_point=operating_point,
    )


def fit_readout_ladder(
    training: ReadoutExamples,
    validation: ReadoutExamples,
    geometry: GlobalReadoutGeometry,
) -> ReadoutLadderResult:
    def fit(
        training_readout: torch.Tensor,
        validation_readout: torch.Tensor,
    ) -> GlobalProbeResult:
        return fit_global_probe(
            GlobalProbeInputs(
                train_readout=training_readout,
                train_target=training.target,
                train_source_ids=training.source_ids,
                validation_readout=validation_readout,
                validation_target=validation.target,
                validation_source_ids=validation.source_ids,
            ),
            geometry,
        )

    return ReadoutLadderResult(
        generator_potential=fit(
            training.generator_potential,
            validation.generator_potential,
        ),
        spike_probability=fit(
            training.spike_probability,
            validation.spike_probability,
        ),
        probability_rate=fit(
            training.probability_rate,
            validation.probability_rate,
        ),
        hard_rate_10_ms=fit(
            training.hard_rate_10_ms,
            validation.hard_rate_10_ms,
        ),
        hard_rate_20_ms=fit(
            training.hard_rate_20_ms,
            validation.hard_rate_20_ms,
        ),
        hard_rate_50_ms=fit(
            training.hard_rate_50_ms,
            validation.hard_rate_50_ms,
        ),
        hard_rate_100_ms=fit(
            training.hard_rate_100_ms,
            validation.hard_rate_100_ms,
        ),
        filtered_rate=fit(
            training.filtered_rate,
            validation.filtered_rate,
        ),
        operating_point=validation.operating_point,
    )


@torch.no_grad()
def evaluate_model_readout_ladder(
    request: ModelReadoutLadderRequest,
) -> ReadoutLadderResult:
    spatial_weights = request.model.rgc.compute_spatial_weights()

    def collect(clips: Sequence[AugmentedClip]) -> ReadoutExamples:
        batch = AugmentedClip.stack(clips)
        request.model.eval()
        output, _ = request.model.forward_sequence(
            batch.noisy_input.float(),
            spatial_weights=spatial_weights,
        )
        return build_readout_examples(
            ReadoutExampleRequest(
                output=output,
                target=batch.clean_target.float(),
                source_ids=tuple(
                    str(clip.metadata["source_id"]) for clip in clips
                ),
                threshold=request.model.rgc.threshold,
                supervised_steps=request.supervised_steps,
                dt_ms=request.dt_ms,
            )
        )

    return fit_readout_ladder(
        collect(request.training_clips),
        collect(request.validation_clips),
        GlobalReadoutGeometry(
            spatial_weights=spatial_weights,
            gain_max=request.gain_max,
        ),
    )


__all__ = [
    "ModelReadoutLadderRequest",
    "ReadoutExampleRequest",
    "ReadoutExamples",
    "ReadoutLadderError",
    "ReadoutLadderResult",
    "ReadoutOperatingPoint",
    "build_readout_examples",
    "causal_filter",
    "evaluate_model_readout_ladder",
    "fit_readout_ladder",
]
