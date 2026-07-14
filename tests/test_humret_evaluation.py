from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.humret import (
    HUMRET_SPATIAL_PERIODS_UM,
    HUMRET_TEMPORAL_FREQUENCIES_HZ,
    HumRetGratingCondition,
    build_humret_contrast_chirp,
    build_humret_drifting_grating,
    build_humret_flash_steps,
    build_humret_frequency_chirp,
    compare_humret_grating_population,
    humret_grating_conditions,
    parse_humret_reference,
    smoothed_spike_probability_to_hz,
)


def test_humret_grating_protocol_has_the_published_24_conditions() -> None:
    # Given / When
    conditions = humret_grating_conditions()

    # Then
    assert len(conditions) == 24
    assert tuple(dict.fromkeys(c.spatial_period_um for c in conditions)) == (
        HUMRET_SPATIAL_PERIODS_UM
    )
    assert tuple(dict.fromkeys(c.temporal_frequency_hz for c in conditions)) == (
        HUMRET_TEMPORAL_FREQUENCIES_HZ
    )
    assert conditions[0].spatial_frequency_cpd == pytest.approx(2.66)
    assert conditions[-1].spatial_frequency_cpd == pytest.approx(0.0665)


def test_humret_probe_builders_follow_the_published_waveforms() -> None:
    # Given
    dt_ms = 5.0

    # When
    frequency_chirp = build_humret_frequency_chirp(2, dt_ms)
    contrast_chirp = build_humret_contrast_chirp(2, dt_ms)
    flash = build_humret_flash_steps(2, dt_ms)
    grating = build_humret_drifting_grating(
        torch.tensor([[0.0, 0.0], [(100.0 / 266.0) / 4.0, 0.0]]),
        dt_ms,
        HumRetGratingCondition(100.0, 1.0),
    )

    # Then
    assert frequency_chirp.shape == (1600, 2)
    assert contrast_chirp.shape == (1600, 2)
    assert flash.shape == (2000, 2)
    torch.testing.assert_close(
        frequency_chirp[200],
        torch.full((2,), torch.sin(torch.tensor(1.1 * torch.pi))),
    )
    torch.testing.assert_close(flash[0], torch.zeros(2))
    torch.testing.assert_close(flash[400], -torch.ones(2))
    torch.testing.assert_close(flash[1200], torch.ones(2))
    assert grating.shape == (2400, 2)
    assert grating[0, 0] == pytest.approx(1.0)
    assert grating[0, 1] == pytest.approx(0.0, abs=1e-6)


def test_humret_mat_fields_are_parsed_without_cell_type_relabeling() -> None:
    # Given
    norm_peaks = np.zeros((2, 24, 4), dtype=np.float32)
    norm_peaks[0, :, 1] = np.arange(1, 25, dtype=np.float32)
    norm_peaks[1, :, 1] = np.arange(24, 0, -1, dtype=np.float32)
    chirp_mod = np.empty((2, 2), dtype=object)
    chirp_mod[0] = (
        np.asarray([0.2, 1.0, 0.4], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    chirp_mod[1] = (
        np.asarray([0.4, 0.2, 1.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )

    # When
    reference = parse_humret_reference(
        norm_peaks,
        np.asarray([11, 29]),
        chirp_mod,
    )

    # Then
    assert reference.grating_f1_normalized.shape == (2, 6, 4)
    torch.testing.assert_close(reference.grating_cell_ids, torch.tensor([11, 29]))
    torch.testing.assert_close(
        reference.grating_f1_normalized[0, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert reference.chirp_modulation_normalized.shape == (2, 3)
    assert not hasattr(reference, "cell_type_labels")


def test_humret_population_comparison_and_rate_units_are_explicit() -> None:
    # Given
    norm_peaks = np.zeros((2, 24, 4), dtype=np.float32)
    norm_peaks[:, :, 1] = np.asarray(
        [np.arange(1, 25), np.arange(24, 0, -1)],
        dtype=np.float32,
    )
    chirp_mod = np.empty((1, 2), dtype=object)
    chirp_mod[0] = (
        np.asarray([0.5, 1.0]),
        np.asarray([1.0, 2.0]),
    )
    reference = parse_humret_reference(norm_peaks, np.asarray([1, 2]), chirp_mod)

    # When
    agreement = compare_humret_grating_population(
        reference.grating_f1_normalized,
        reference,
    )
    firing_rate_hz = smoothed_spike_probability_to_hz(
        torch.tensor([0.0, 0.5, 1.0]),
        dt_ms=5.0,
    )

    # Then
    assert agreement.mean_tuning_cosine_similarity == pytest.approx(1.0)
    assert agreement.spatial_preference_total_variation == pytest.approx(0.0)
    assert agreement.temporal_preference_total_variation == pytest.approx(0.0)
    torch.testing.assert_close(firing_rate_hz, torch.tensor([0.0, 100.0, 200.0]))
