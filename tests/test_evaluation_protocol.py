from __future__ import annotations

import pytest
import torch

from evaluation.dynamics import (
    TemporalMetricsRequest,
    TemporalProbeKind,
    TemporalProbeSpec,
    build_temporal_probe,
    temporal_response_metrics,
)
from evaluation.rf_probe import (
    LocalPoissonGLMRequest,
    compare_rf_maps,
    compare_temporal_rfs,
    fit_local_poisson_glm,
)


def test_temporal_probe_family_and_metrics_are_explicit() -> None:
    # Given
    spec = TemporalProbeSpec(
        cone_count=2,
        time_steps=12,
        dt_ms=5.0,
        onset_step=2,
        offset_step=10,
        amplitude=1.0,
        flicker_hz=10.0,
        chirp_start_hz=1.0,
        chirp_end_hz=8.0,
    )

    # When
    probes = tuple(build_temporal_probe(kind, spec) for kind in TemporalProbeKind)
    metrics = temporal_response_metrics(
        TemporalMetricsRequest(
            response=torch.tensor([0.0, 0.0, 1.0, 0.2, -0.5, 0.0, 0.0]),
            dt_ms=5.0,
            onset_step=2,
            offset_step=5,
        )
    )

    # Then
    assert all(probe.shape == (12, 2) for probe in probes)
    assert all(torch.isfinite(probe).all() for probe in probes)
    assert torch.count_nonzero(probes[0]) == 2
    assert metrics.time_to_peak_ms == 0.0
    assert metrics.crossover_ms is not None
    assert metrics.transience_index == 1.0


def test_sustained_step_is_not_misclassified_as_transient_or_crossover() -> None:
    # Given / When
    metrics = temporal_response_metrics(
        TemporalMetricsRequest(
            response=torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            dt_ms=5.0,
            onset_step=2,
            offset_step=5,
        )
    )

    # Then
    assert metrics.transience_index == 0.0
    assert metrics.crossover_ms is None


def test_local_poisson_glm_keeps_rf_outside_support_zero() -> None:
    # Given
    generator = torch.Generator().manual_seed(9)
    stimulus = torch.randn((48, 3, 3), generator=generator)
    rate = torch.exp(0.3 * stimulus[:, :, 0].sum(dim=1))
    spikes = torch.poisson(rate, generator=generator)

    # When
    result = fit_local_poisson_glm(
        LocalPoissonGLMRequest(
            stimulus=stimulus,
            spike_counts=spikes,
            source_indices=torch.tensor([0]),
            l2_weight=1e-3,
            max_steps=25,
        )
    )

    # Then
    assert result.rf.shape == (3, 3)
    assert torch.count_nonzero(result.rf[:, 1:]) == 0
    assert torch.isfinite(result.nll)
    assert result.rf.device == stimulus.device


def test_rf_map_agreement_reports_sign_centroid_and_waveform_consistency() -> None:
    # Given
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    first = torch.tensor([[0.0, 1.0, 0.0], [0.0, -0.5, 0.0]])
    second = torch.tensor([[0.0, 0.8, 0.0], [0.0, -0.4, 0.0]])

    # When
    agreement = compare_rf_maps(first, second, positions)

    # Then
    assert agreement.center_sign_match
    assert agreement.centroid_distance_degs == 0.0
    assert agreement.cosine_similarity == pytest.approx(1.0)


def test_temporal_rf_comparison_reports_condition_shift_without_pass_threshold(
) -> None:
    # Given
    reference = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [-0.5, 0.0], [0.0, 0.0]]
    )
    condition = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [-0.25, 0.0]]
    )

    # When
    comparison = compare_temporal_rfs(reference, condition, dt_ms=5.0)

    # Then
    assert comparison.reference_ttp_ms == 10.0
    assert comparison.condition_ttp_ms == 5.0
    assert comparison.ttp_shift_ms == -5.0
    assert comparison.peak_gain_ratio == 1.0
    assert comparison.reference_biphasic_index == 0.5
    assert comparison.condition_biphasic_index == 0.25
