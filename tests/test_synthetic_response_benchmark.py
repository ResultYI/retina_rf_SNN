from __future__ import annotations

import numpy as np

from benchmarks.point_process_teacher import generate_teacher_responses


def test_static_and_adaptive_teachers_declare_distinct_context_behavior() -> None:
    rng = np.random.default_rng(7)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    time_axis = np.arange(80) * 0.005

    static = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        time_axis,
        trials=2,
        seed=3,
        adaptive=False,
    )
    adaptive = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        time_axis,
        trials=2,
        seed=3,
        adaptive=True,
    )

    assert np.array_equal(
        static.kernels["context_kernel_low"],
        static.kernels["context_kernel_high"],
    )
    assert not np.array_equal(
        adaptive.kernels["context_kernel_low"],
        adaptive.kernels["context_kernel_high"],
    )
    assert np.allclose(
        static.expected_probabilities[0, -16:],
        static.expected_probabilities[1, -16:],
    )
    final_difference = (
        adaptive.expected_probabilities[1, -1]
        - adaptive.expected_probabilities[0, -1]
    )
    assert np.all(np.abs(final_difference) > 1e-5)
    envelope = adaptive.kernels["context_gain_envelope"]
    transition = envelope.shape[1] - min(64, envelope.shape[1] // 2)
    assert np.all(
        np.abs(envelope[1, transition] - 1)
        > np.abs(envelope[1, -1] - 1)
    )
    assert adaptive.session.spike_counts.shape == (4, 2, 80, 4)
