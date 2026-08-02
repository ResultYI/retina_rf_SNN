from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest
import torch

from data.rgc_response import ResponseTargetKind
from evaluation.response_metrics import (
    ResponseMetricError,
    compute_response_metrics,
    training_baseline_rates,
)
from loss.rgc_response import response_nll, response_nll_elements


def test_response_metrics_preserve_existing_masked_nll_and_bits_behavior() -> None:
    # Given
    probabilities = torch.tensor(
        [[[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.2, 0.1]]]]
    )
    targets = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]]]
    )
    valid_mask = torch.tensor(
        [[[[True, True], [True, False]], [[True, True], [False, True]]]]
    )
    logits = torch.logit(probabilities)
    baseline_rates = training_baseline_rates(targets, valid_mask)

    # When
    metrics = compute_response_metrics(
        logits,
        targets,
        valid_mask,
        ResponseTargetKind.BERNOULLI,
        baseline_rates,
    )

    # Then
    mask = valid_mask.to(logits.dtype)
    model_elements = response_nll_elements(
        logits,
        targets,
        ResponseTargetKind.BERNOULLI,
    )
    baseline_logits = torch.logit(baseline_rates.clamp(1e-5, 1 - 1e-5))
    baseline_elements = response_nll_elements(
        baseline_logits.view(1, 1, 1, -1).expand_as(logits),
        targets,
        ResponseTargetKind.BERNOULLI,
    )
    denominators = mask.sum(dim=(0, 1, 2))
    per_cell_nll = (model_elements * mask).sum(dim=(0, 1, 2)) / denominators
    improvement = (baseline_elements - model_elements) * mask
    spike_count = (targets * mask).sum()
    log_two = torch.log(torch.tensor(2.0))
    cell_spikes = (targets * mask).sum(dim=(0, 1, 2))
    cell_bits = improvement.sum(dim=(0, 1, 2)) / cell_spikes.clamp_min(1) / log_two

    assert metrics.nll == torch.mean(per_cell_nll).item()
    assert metrics.per_cell_nll == tuple(float(value) for value in per_cell_nll)
    assert metrics.micro_bits_per_spike == (
        improvement.sum() / spike_count.clamp_min(1) / log_two
    ).item()
    assert metrics.macro_bits_per_spike == cell_bits[cell_spikes > 0].mean().item()


def test_response_metrics_report_constant_rate_nll_and_masked_bernoulli_ece() -> None:
    # Given
    probabilities = torch.tensor([[[[0.05], [0.15], [0.85], [0.95]]]])
    targets = torch.tensor([[[[0.0], [1.0], [1.0], [1.0]]]])
    valid_mask = torch.tensor([[[[True], [False], [True], [True]]]])
    logits = torch.logit(probabilities)
    baseline_rates = training_baseline_rates(targets, valid_mask)

    # When
    metrics = compute_response_metrics(
        logits,
        targets,
        valid_mask,
        ResponseTargetKind.BERNOULLI,
        baseline_rates,
    )

    # Then
    baseline_logits = torch.logit(baseline_rates.clamp(1e-5, 1 - 1e-5))
    expected_nll = response_nll(
        baseline_logits.view(1, 1, 1, -1).expand_as(logits),
        targets,
        valid_mask,
        ResponseTargetKind.BERNOULLI,
    )
    expected_ece = (0.05 + 0.15 + 0.05) / 3
    assert metrics.constant_rate_nll == expected_nll.item()
    assert metrics.calibration_error is not None
    assert metrics.calibration_error == pytest.approx(expected_ece)


def test_response_metrics_constant_rate_nll_uses_global_masked_element_mean() -> None:
    # Given
    logits = torch.zeros(1, 1, 4, 2)
    targets = torch.tensor(
        [[[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]]
    )
    valid_mask = torch.tensor(
        [[[[True, True], [True, False], [True, False], [True, False]]]]
    )
    baseline_rates = training_baseline_rates(targets, valid_mask)

    # When
    metrics = compute_response_metrics(
        logits,
        targets,
        valid_mask,
        ResponseTargetKind.BERNOULLI,
        baseline_rates,
    )

    # Then
    mask = valid_mask.to(logits.dtype)
    baseline_logits = torch.logit(baseline_rates.clamp(1e-5, 1 - 1e-5))
    baseline_elements = response_nll_elements(
        baseline_logits.view(1, 1, 1, -1).expand_as(logits),
        targets,
        ResponseTargetKind.BERNOULLI,
    )
    global_baseline_nll = (baseline_elements * mask).sum() / mask.sum()
    macro_baseline_nll = response_nll(
        baseline_logits.view(1, 1, 1, -1).expand_as(logits),
        targets,
        valid_mask,
        ResponseTargetKind.BERNOULLI,
    )

    assert global_baseline_nll != macro_baseline_nll
    assert metrics.constant_rate_nll == pytest.approx(global_baseline_nll.item())


def test_response_metrics_poisson_calibration_serializes_as_explicit_na() -> None:
    # Given
    logits = torch.zeros(1, 1, 2, 1)
    targets = torch.tensor([[[[0.0], [1.0]]]])
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    baseline_rates = training_baseline_rates(targets, valid_mask)

    # When
    metrics = compute_response_metrics(
        logits,
        targets,
        valid_mask,
        ResponseTargetKind.POISSON,
        baseline_rates,
    )

    # Then
    payload = json.dumps(asdict(metrics), allow_nan=False)
    assert metrics.calibration_error is None
    assert '"calibration_error": null' in payload


def test_response_metrics_keep_all_zero_spike_outputs_finite() -> None:
    # Given
    logits = torch.full((1, 2, 3, 2), -2.0)
    targets = torch.zeros_like(logits)
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    baseline_rates = training_baseline_rates(targets, valid_mask)

    # When
    metrics = compute_response_metrics(
        logits,
        targets,
        valid_mask,
        ResponseTargetKind.BERNOULLI,
        baseline_rates,
    )

    # Then
    assert math.isfinite(metrics.nll)
    assert math.isfinite(metrics.constant_rate_nll)
    assert math.isfinite(metrics.micro_bits_per_spike)
    assert math.isfinite(metrics.macro_bits_per_spike)
    assert metrics.calibration_error is not None
    assert 0.0 <= metrics.calibration_error <= 1.0


def test_response_metrics_all_masked_inputs_return_defined_finite_metrics() -> None:
    # Given
    logits = torch.zeros(1, 1, 2, 1)
    targets = torch.zeros_like(logits)
    valid_mask = torch.zeros_like(logits, dtype=torch.bool)
    baseline_rates = torch.tensor([0.0])

    # When
    metrics = compute_response_metrics(
        logits,
        targets,
        valid_mask,
        ResponseTargetKind.BERNOULLI,
        baseline_rates,
    )

    # Then
    assert metrics.nll == 0.0
    assert metrics.constant_rate_nll == 0.0
    assert metrics.micro_bits_per_spike == 0.0
    assert metrics.macro_bits_per_spike == 0.0
    assert metrics.psth_correlation == 0.0
    assert metrics.explained_variance == 0.0
    assert metrics.per_cell_nll == (0.0,)
    assert metrics.calibration_error == 0.0


def test_response_metrics_reject_partially_unobserved_cells() -> None:
    # Given
    logits = torch.zeros(1, 1, 2, 2)
    targets = torch.zeros_like(logits)
    valid_mask = torch.tensor([[[[True, False], [True, False]]]])
    baseline_rates = torch.tensor([0.0, 0.0])

    # When / Then
    with pytest.raises(ResponseMetricError, match="Every cell needs at least one valid target"):
        compute_response_metrics(
            logits,
            targets,
            valid_mask,
            ResponseTargetKind.BERNOULLI,
            baseline_rates,
        )


def test_response_metrics_reject_mismatched_mask_shape() -> None:
    # Given
    logits = torch.zeros(1, 1, 2, 1)
    targets = torch.zeros_like(logits)
    valid_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    baseline_rates = torch.tensor([0.0])

    # When / Then
    with pytest.raises(ResponseMetricError, match="share a shape"):
        compute_response_metrics(
            logits,
            targets,
            valid_mask,
            ResponseTargetKind.BERNOULLI,
            baseline_rates,
        )


def test_response_metrics_reject_nan_baseline_rates_at_boundary() -> None:
    # Given
    logits = torch.zeros(1, 1, 2, 1)
    targets = torch.zeros_like(logits)
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    baseline_rates = torch.tensor([float("nan")])

    # When / Then
    with pytest.raises(ResponseMetricError, match="baseline_rates must be finite"):
        compute_response_metrics(
            logits,
            targets,
            valid_mask,
            ResponseTargetKind.BERNOULLI,
            baseline_rates,
        )
