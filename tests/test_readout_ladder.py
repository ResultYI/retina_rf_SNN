from __future__ import annotations

import math

import pytest
import torch

from evaluation import readout_ladder
from evaluation.global_probe import GlobalReadoutGeometry
from models.cells.rgc_types import RGCOutput


def test_causal_filter_matches_rgc_rate_recurrence() -> None:
    # Given
    events = torch.tensor([[[[1.0]], [[0.0]], [[1.0]]]])
    dt_ms = 5.0
    tau_ms = 20.0
    leak = math.exp(-dt_ms / tau_ms)
    expected = torch.tensor(
        [
            [
                [[[1.0 - leak]]],
                [[[leak * (1.0 - leak)]]],
                [[[leak**2 * (1.0 - leak) + 1.0 - leak]]],
            ]
        ]
    ).reshape_as(events)

    # When
    filtered = readout_ladder.causal_filter(events, dt_ms=dt_ms, tau_ms=tau_ms)

    # Then
    assert torch.allclose(filtered, expected)


def test_readout_examples_preserve_the_model_rate_and_build_soft_rate() -> None:
    # Given
    probability = torch.tensor([[[[0.2]], [[0.8]], [[0.4]]]])
    hard = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
    saved_rate = readout_ladder.causal_filter(hard, dt_ms=5.0, tau_ms=50.0)
    output = RGCOutput(
        hard_spikes=hard,
        surrogate_spikes=hard,
        spike_probability=probability,
        rates=saved_rate,
        generator_potential=torch.tensor([[[[0.1]], [[0.4]], [[0.2]]]]),
    )

    # When
    examples = readout_ladder.build_readout_examples(
        readout_ladder.ReadoutExampleRequest(
            output=output,
            target=torch.zeros(1, 3, 1),
            source_ids=("source-a",),
            threshold=torch.tensor([0.25]),
            supervised_steps=2,
            dt_ms=5.0,
        )
    )

    # Then
    assert torch.equal(examples.filtered_rate, saved_rate[:, -2:])
    assert torch.allclose(
        examples.probability_rate,
        readout_ladder.causal_filter(
            probability,
            dt_ms=5.0,
            tau_ms=50.0,
        )[:, -2:],
    )
    assert examples.operating_point.probability_saturated_fraction == pytest.approx(
        0.0
    )


def test_ladder_separates_probability_information_from_hard_rate() -> None:
    # Given
    probability = torch.tensor(
        [
            [[[0.2], [0.0]], [[0.8], [0.0]], [[0.4], [0.0]]],
            [[[0.3], [0.0]], [[0.7], [0.0]], [[0.5], [0.0]]],
        ]
    )
    weights = torch.tensor([[0.5, 0.5]])
    target = probability[:, :, 0] @ weights
    zeros = torch.zeros_like(probability)
    operating = readout_ladder.ReadoutOperatingPoint(
        margin_mean=0.0,
        margin_standard_deviation=1.0,
        margin_quantile_05=-1.0,
        margin_median=0.0,
        margin_quantile_95=1.0,
        probability_below_001_fraction=0.0,
        probability_above_099_fraction=0.0,
        probability_saturated_fraction=0.0,
        probability_variance=float(probability.var(unbiased=False)),
        hard_spike_fraction=0.0,
        zero_spike_unit_fraction=1.0,
        filtered_rate_variance=0.0,
    )
    examples = readout_ladder.ReadoutExamples(
        generator_potential=probability,
        spike_probability=probability,
        probability_rate=probability,
        hard_rate_10_ms=zeros,
        hard_rate_20_ms=zeros,
        hard_rate_50_ms=zeros,
        hard_rate_100_ms=zeros,
        filtered_rate=zeros,
        target=target,
        source_ids=("source-a", "source-b"),
        operating_point=operating,
    )

    # When
    result = readout_ladder.fit_readout_ladder(
        examples,
        examples,
        GlobalReadoutGeometry(spatial_weights=weights, gain_max=5.0),
    )

    # Then
    assert result.spike_probability.validation_mse < 1e-8
    assert result.filtered_rate.validation_mse > 0.01
