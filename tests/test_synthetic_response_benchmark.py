from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from benchmarks.point_process_teacher import generate_teacher_responses
from data.cone_response import ConeResponseExport, DataContractError
from data.dataset import validate_loaded_cone_exports
from data.input_identity import synthetic_input_identity
from data.rgc_response_export import write_rgc_response
from data.synthetic_teacher import fit_teacher_input_normalization


def test_static_and_adaptive_teachers_declare_distinct_context_behavior() -> None:
    rng = np.random.default_rng(7)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    time_axis = np.arange(80) * 0.005
    normalization = fit_teacher_input_normalization(cones)

    static = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        time_axis,
        trials=2,
        seed=3,
        adaptive=False,
        teacher_normalization=normalization,
    )
    adaptive = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        time_axis,
        trials=2,
        seed=3,
        adaptive=True,
        teacher_normalization=normalization,
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


def test_teacher_logits_use_shared_train_normalization_across_split_containers() -> None:
    base = np.linspace(-1.0, 1.0, 80 * 5, dtype=np.float32).reshape(80, 5)
    train_cones = np.stack((base, base + 4.0))
    validation_cones = np.stack((base, base - 8.0))
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    time_axis = np.arange(80) * 0.005
    normalization = fit_teacher_input_normalization(train_cones)

    train = generate_teacher_responses(
        train_cones,
        positions,
        ("shared-train", "other-train"),
        time_axis,
        trials=1,
        seed=3,
        adaptive=False,
        teacher_normalization=normalization,
    )
    validation = generate_teacher_responses(
        validation_cones,
        positions,
        ("shared-validation", "other-validation"),
        time_axis,
        trials=1,
        seed=3,
        adaptive=False,
        teacher_normalization=normalization,
    )

    assert np.allclose(
        train.expected_probabilities[0],
        validation.expected_probabilities[0],
        atol=1e-7,
    )


def test_export_persists_teacher_input_normalization(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    time_axis = np.arange(80) * 0.005
    normalization = fit_teacher_input_normalization(cones)
    result = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        time_axis,
        trials=1,
        seed=3,
        adaptive=False,
        teacher_normalization=normalization,
    )

    write_rgc_response(
        tmp_path / "synthetic.h5",
        result.session,
        teacher_kernels=result.kernels,
        teacher_normalization=result.teacher_normalization,
    )

    with h5py.File(tmp_path / "synthetic.h5", "r") as handle:
        assert "teacher/input_mean" in handle
        assert "teacher/input_std" in handle


def test_same_shape_different_cone_positions_are_rejected() -> None:
    identity = synthetic_input_identity(2, ("source",))
    reference = ConeResponseExport(
        response=np.ones((4, 2), dtype=np.float32),
        positions_degs=np.zeros((2, 2), dtype=np.float32),
        cone_types=np.asarray([1, 2], dtype=np.uint8),
        time_axis_seconds=np.arange(4) * 0.005,
        eye_trace_degs=np.zeros((4, 2), dtype=np.float32),
        units="isomerizations_per_integration_time",
        eccentricity_deg=4.0,
        input_identity=identity,
    )
    changed = replace(reference, positions_degs=np.ones((2, 2), dtype=np.float32))

    with pytest.raises(DataContractError, match="cone positions"):
        validate_loaded_cone_exports((reference, changed))


def test_static_teacher_kernel_matches_normalized_coordinate_finite_difference() -> None:
    rng = np.random.default_rng(23)
    cones = rng.random((2, 80, 5), dtype=np.float32)
    positions = np.stack((np.arange(5) * 0.05, np.zeros(5)), axis=1)
    time_axis = np.arange(80) * 0.005
    normalization = fit_teacher_input_normalization(cones)
    output_time = 70
    lag = 3
    cone_index = 2
    cell_index = 0
    epsilon = np.float32(1e-3)
    result = generate_teacher_responses(
        cones,
        positions,
        ("a", "b"),
        time_axis,
        trials=1,
        seed=3,
        adaptive=False,
        teacher_normalization=normalization,
    )
    perturbed_cones = cones.copy()
    perturbed_cones[0, output_time - lag, cone_index] += (
        epsilon * normalization.input_std[cone_index]
    )

    perturbed = generate_teacher_responses(
        perturbed_cones,
        positions,
        ("a", "b"),
        time_axis,
        trials=1,
        seed=3,
        adaptive=False,
        teacher_normalization=normalization,
    )

    base_probability = result.expected_probabilities[0, output_time, cell_index]
    perturbed_probability = perturbed.expected_probabilities[0, output_time, cell_index]
    base_logit = np.log(base_probability / (1 - base_probability)) + 2.0
    perturbed_logit = np.log(perturbed_probability / (1 - perturbed_probability)) + 2.0
    finite_difference = (perturbed_logit - base_logit) / epsilon

    assert np.isclose(
        finite_difference,
        result.kernels["static_kernel"][cell_index, lag, cone_index],
        rtol=5e-3,
        atol=5e-3,
    )
