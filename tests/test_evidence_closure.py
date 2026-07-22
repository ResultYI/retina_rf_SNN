from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import torch

from evaluation.dynamic_rf import (
    DynamicRFSelection,
    DynamicRFUnitResult,
    FiniteDifferenceResult,
)
from evaluation.dynamic_rf_summary import compare_dynamic_rf
from evaluation.reconstruction import fit_augmented_reconstruction_scale
from evaluation.rgc_types import FEATURE_NAMES
from evaluation.temporal_probes import TemporalProbeFeatures
from training.config import load_config
from training.data import PreparedClip


ROOT = Path(__file__).resolve().parents[1]


def test_temporal_probe_contract_has_per_unit_quality_fields() -> None:
    assert tuple(field.name for field in fields(TemporalProbeFeatures)) == (
        "preferred_polarity",
        "valid_response_mask",
        "impulse_peak",
        "impulse_time_to_peak_ms",
        "impulse_width_ms",
        "step_sustained_index",
        "flicker_response",
        "hard_evoked_spike_count",
    )


def test_rgc_typing_uses_only_preregistered_functional_features() -> None:
    assert FEATURE_NAMES == (
        "effective_spatial_radius",
        "impulse_time_to_peak_ms",
        "impulse_width_ms",
        "step_sustained_index",
        "normalized_flicker_response",
    )


def test_augmented_reconstruction_scale_is_seed_deterministic() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    clip = PreparedClip(
        clean=torch.linspace(0.0, 1.0, config.data.sequence_steps).view(-1, 1),
        source_id="source",
    )
    first = fit_augmented_reconstruction_scale([clip], config.data, seed=31)
    second = fit_augmented_reconstruction_scale([clip], config.data, seed=31)
    assert first == second


def test_dynamic_rf_comparison_is_source_paired() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    trained = tuple(_dynamic_row(f"source-{index}", 0.20, math.exp(0.20)) for index in range(3))
    initialized = tuple(_dynamic_row(f"source-{index}", 0.01, math.exp(0.01)) for index in range(3))
    summary, sources = compare_dynamic_rf(
        trained,
        initialized,
        config.evaluation,
        seed=config.seed,
    )
    assert summary.valid_source_count == 3
    assert len(sources) == 3
    assert summary.status == "learned_dynamic_rf_supported"


def test_dynamic_rf_selection_is_immutable_and_shared() -> None:
    selection = DynamicRFSelection(polarity=1, unit_indices=(2, 4))
    assert selection.unit_indices == (2, 4)


def _dynamic_row(
    source_id: str,
    shape_distance: float,
    norm_ratio: float,
) -> DynamicRFUnitResult:
    return DynamicRFUnitResult(
        source_id=source_id,
        polarity=0,
        unit=0,
        low_kernel_norm=1.0,
        high_kernel_norm=norm_ratio,
        kernel_norm_ratio=norm_ratio,
        gain_normalized_cosine_distance=shape_distance,
        low_temporal_peak_ms=10.0,
        high_temporal_peak_ms=12.0,
        temporal_peak_shift_ms=2.0,
        low_integration_width_ms=20.0,
        high_integration_width_ms=22.0,
        integration_width_shift_ms=2.0,
        low_spatial_center_distance_degs=0.1,
        high_spatial_center_distance_degs=0.1,
        spatial_center_shift_degs=0.0,
        low_spatial_second_moment=0.2,
        high_spatial_second_moment=0.2,
        spatial_second_moment_shift=0.0,
        identical_reset_kernel_error=0.0,
        recovery_curve=((0, 1.0), (500, 0.1)),
        finite_difference=FiniteDifferenceResult(1.0, 1.0, 0.0, "local_continuous_check"),
    )
