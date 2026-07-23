from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from evaluation.dynamic_rf import (
    DynamicRFSelection,
    DynamicRFUnitResult,
    FiniteDifferenceResult,
)
from evaluation.dynamic_rf_summary import compare_dynamic_rf
from evaluation.reconstruction import fit_augmented_reconstruction_scale
from evaluation.rgc_types import (
    FEATURE_NAMES,
    eligible_rgc_units,
    separation_supports_learning,
)
from evaluation.temporal_probes import TemporalProbeFeatures
from training.augmentation import augment_clip
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
        "encoder_pooling_radius",
        "impulse_time_to_peak_ms",
        "impulse_width_ms",
        "step_sustained_index",
        "normalized_flicker_response",
    )


def test_rgc_eligibility_uses_trained_response_quality_only() -> None:
    probes = _temporal_features(torch.tensor([True, False]))
    trained_features = np.ones((2, len(FEATURE_NAMES)))
    initialized_features = np.zeros_like(trained_features)
    assert eligible_rgc_units(
        probes,
        trained_features,
        initialized_features,
    ).tolist() == [True, False]


def test_rgc_separation_requires_absolute_and_relative_gain() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml").evaluation
    assert not separation_supports_learning(0.06, 0.09, config)
    assert not separation_supports_learning(0.20, 0.26, config)
    assert separation_supports_learning(0.20, 0.31, config)
    assert separation_supports_learning(0.01, 0.07, config)


def test_augmented_reconstruction_scale_is_seed_deterministic() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    clip = PreparedClip(
        clean=torch.linspace(0.0, 1.0, config.data.sequence_steps).view(-1, 1),
        source_id="source",
    )
    first = fit_augmented_reconstruction_scale([clip], config.data, seed=31)
    second = fit_augmented_reconstruction_scale([clip], config.data, seed=31)
    assert first == second


def test_augmentation_transitions_finish_before_supervised_onset() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    clip = PreparedClip(
        clean=torch.ones(config.data.sequence_steps, 2),
        source_id="source",
    )
    supervised_onset = (
        config.training.burn_in_steps + config.training.context_only_steps
    )
    for seed in range(32):
        augmented = augment_clip(
            clip,
            config.data,
            torch.Generator().manual_seed(seed),
        )
        if augmented.metadata["transition_step"] >= 0:
            assert (
                augmented.metadata["transition_step"]
                <= config.data.context_transition_latest_step
            )
            assert (
                augmented.metadata["transition_step"]
                + augmented.metadata["transition_width_steps"]
                < supervised_onset
            )


def test_dynamic_rf_comparison_is_source_paired() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    trained = tuple(
        _dynamic_row(f"source-{source}", unit, 0.20, math.exp(0.20))
        for source in range(3)
        for unit in range(3)
    )
    initialized = tuple(
        _dynamic_row(f"source-{source}", unit, 0.01, math.exp(0.01))
        for source in range(3)
        for unit in range(3)
    )
    summary, sources = compare_dynamic_rf(
        trained,
        initialized,
        config.evaluation,
        seed=config.seed,
    )
    assert summary.valid_source_count == 3
    assert len(sources) == 3
    assert summary.status == "learned_dynamic_rf_supported"
    assert summary.trained_finite_difference_valid_fraction == 1.0
    assert summary.initialized_finite_difference_valid_fraction == 1.0


def test_dynamic_rf_source_requires_multiple_valid_records() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    trained = (_dynamic_row("source", 0, 0.20, math.exp(0.20)),)
    initialized = (_dynamic_row("source", 0, 0.01, math.exp(0.01)),)
    summary, sources = compare_dynamic_rf(
        trained,
        initialized,
        config.evaluation,
        seed=config.seed,
    )
    assert summary.status == "not_identifiable"
    assert summary.valid_source_count == 0
    assert sources[0].valid_record_fraction == 1.0


def test_dynamic_rf_selection_is_immutable_and_shared() -> None:
    selection = DynamicRFSelection(polarity=1, unit_indices=(2, 4))
    assert selection.unit_indices == (2, 4)


def _dynamic_row(
    source_id: str,
    unit: int,
    shape_distance: float,
    norm_ratio: float,
) -> DynamicRFUnitResult:
    return DynamicRFUnitResult(
        source_id=source_id,
        polarity=0,
        unit=unit,
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


def _temporal_features(valid: torch.Tensor) -> TemporalProbeFeatures:
    count = valid.numel()
    ones = torch.ones(count)
    return TemporalProbeFeatures(
        preferred_polarity=torch.zeros(count, dtype=torch.long),
        valid_response_mask=valid,
        impulse_peak=ones,
        impulse_time_to_peak_ms=ones,
        impulse_width_ms=ones,
        step_sustained_index=ones,
        flicker_response=ones,
        hard_evoked_spike_count=ones,
    )
