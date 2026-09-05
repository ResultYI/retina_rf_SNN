from __future__ import annotations

import importlib

import pytest
import torch

from evaluation.mechanistic_retina.temporal_center_surround import (
    CenterSurroundProbeConfig,
    build_center_surround_probe,
    summarize_response,
)


def test_probe_uses_disjoint_center_surround_supports_and_signed_offsets() -> None:
    # Given
    center = torch.tensor([1.0, 1.0, 0.0, 0.0])
    surround = torch.tensor([0.0, 0.0, 1.0, 1.0])
    config = CenterSurroundProbeConfig(
        total_ms=500.0,
        center_onset_ms=200.0,
        pulse_duration_ms=100.0,
        contrast=0.25,
    )

    # When
    probe = build_center_surround_probe(center, surround, 10.0, 1.0, config)

    # Then
    assert probe.names == (
        "center_only",
        "surround_only",
        "surround_then_center_100ms",
        "surround_then_center_50ms",
        "center_surround_simultaneous",
        "center_then_surround_50ms",
        "center_then_surround_100ms",
    )
    assert torch.isnan(probe.offset_ms[:2]).all()
    assert probe.offset_ms[2:].tolist() == [-100, -50, 0, 50, 100]
    simultaneous = probe.cone_drive[4]
    torch.testing.assert_close(simultaneous[:, :2], simultaneous[:, 2:])
    assert torch.count_nonzero(probe.cone_drive[0, :, 2:]).item() == 0
    assert torch.count_nonzero(probe.cone_drive[1, :, :2]).item() == 0


def test_response_summary_reports_signed_peak_integral_and_event_windows() -> None:
    # Given
    response = torch.zeros(50)
    response[20:30] = torch.tensor(
        [0.1, 0.2, 0.4, 0.8, 0.5, 0.3, 0.2, 0.1, 0.0, -0.1]
    )

    # When
    summary = summarize_response(
        response,
        dt_ms=10.0,
        center_onset_ms=200.0,
        surround_onset_ms=None,
        pulse_duration_ms=100.0,
        event_window_ms=30.0,
    )

    # Then
    assert summary.peak_response == pytest.approx(0.8)
    assert summary.peak_latency_ms == pytest.approx(230.0)
    assert summary.response_integral == pytest.approx(0.025)
    assert summary.center_onset_response == pytest.approx((0.1 + 0.2 + 0.4) / 3)
    assert summary.center_offset_response == pytest.approx(0.0)
    assert summary.surround_onset_response is None
    assert summary.surround_offset_response is None


def test_runner_import_does_not_require_removed_candidate_lineage() -> None:
    # Given the cleaned Canonical V1 repository without Candidate0 types.
    module_name = (
        "evaluation.mechanistic_retina.schottdorf_temporal_center_surround"
    )

    # When the frozen-analysis runner is imported.
    module = importlib.import_module(module_name)

    # Then its public entry point is available without legacy imports.
    assert callable(module.run_frozen_temporal_center_surround)
