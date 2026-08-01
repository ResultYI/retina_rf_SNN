from __future__ import annotations

import numpy as np

from benchmarks.point_process_teacher import (
    SyntheticTeacherResult,
    generate_teacher_responses,
)
from data.synthetic_teacher import fit_teacher_input_normalization


def test_teacher_current_spike_does_not_change_current_probability() -> None:
    # Given
    result = _history_teacher()

    # When
    current_spikes = result.session.spike_counts[:, :, 0]
    current_probabilities = result.conditional_probabilities[:, :, 0]

    # Then
    assert np.unique(current_spikes).size == 2
    assert np.allclose(current_probabilities, current_probabilities[:, :1])


def test_teacher_previous_spike_suppresses_next_probability() -> None:
    # Given
    result = _history_teacher()
    previous_spikes = result.session.spike_counts[:, :, 0].astype(bool)
    base_probability = result.expected_probabilities[:, 1]

    # When
    conditional_probability = result.conditional_probabilities[:, :, 1]
    base_logit = np.log(base_probability / (1 - base_probability))
    conditional_logit = np.log(
        conditional_probability / (1 - conditional_probability)
    )
    history_offset = conditional_logit - base_logit[:, None]

    # Then
    assert np.allclose(history_offset[~previous_spikes], 0.0, atol=1e-6)
    assert np.all(history_offset[previous_spikes] < -0.5)


def _history_teacher() -> SyntheticTeacherResult:
    rng = np.random.default_rng(29)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    return generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        np.arange(80) * 0.005,
        trials=32,
        seed=31,
        adaptive=False,
        teacher_normalization=fit_teacher_input_normalization(cones),
    )
