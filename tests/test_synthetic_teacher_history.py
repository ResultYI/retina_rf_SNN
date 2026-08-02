from __future__ import annotations

import numpy as np
import pytest

from benchmarks.point_process_teacher import (
    SyntheticTeacherError,
    SyntheticTeacherResult,
    TeacherPopulationConfig,
    generate_teacher_responses,
)
from data.rgc_response import validate_response_splits
from data.synthetic_teacher import fit_teacher_input_normalization


def test_teacher_current_spike_does_not_change_current_probability() -> None:
    result = _legacy_history_teacher()

    current_spikes = result.session.spike_counts[:, :, 0]
    current_probabilities = result.conditional_probabilities[:, :, 0]

    assert np.unique(current_spikes).size == 2
    assert np.allclose(current_probabilities, current_probabilities[:, :1])


def test_teacher_previous_spike_suppresses_next_probability() -> None:
    result = _legacy_history_teacher()
    previous_spikes = result.session.spike_counts[:, :, 0].astype(bool)
    base_probability = result.expected_probabilities[:, 1]

    conditional_probability = result.conditional_probabilities[:, :, 1]
    base_logit = np.log(base_probability / (1 - base_probability))
    conditional_logit = np.log(
        conditional_probability / (1 - conditional_probability)
    )
    history_offset = conditional_logit - base_logit[:, None]

    assert np.allclose(history_offset[~previous_spikes], 0.0, atol=1e-6)
    assert np.all(history_offset[previous_spikes] < -0.5)


def test_teacher_explicit_single_replicate_preserves_legacy_four_cell_layout() -> None:
    cones, positions = _teacher_inputs(cone_count=3)

    result = _history_teacher(
        cones=cones,
        positions=positions,
        population_config=TeacherPopulationConfig(cells_per_type_polarity=1),
    )

    assert result.session.spike_counts.shape == (4, 32, 80, 4)
    assert tuple(zip(result.session.cells.type_ids, result.session.cells.polarities)) == (
        ("midget", 0),
        ("midget", 1),
        ("parasol", 0),
        ("parasol", 1),
    )


def test_teacher_default_population_has_four_replicates_per_group() -> None:
    cones, positions = _teacher_inputs()

    result = _history_teacher(cones=cones, positions=positions)

    groups = result.kernels["cell_group_id"]
    assert result.session.spike_counts.shape == (4, 32, 80, 16)
    assert len(set(result.session.cells.ids)) == 16
    assert tuple(np.unique(groups, return_counts=True)[1]) == (4, 4, 4, 4)
    assert tuple(result.kernels["component_id"]) == (
        "population",
        "type",
        "polarity",
        "cell_residual",
    )
    assert tuple(result.kernels["revision"]) == ("hierarchical-synthetic-teacher-v1",)


def test_teacher_reuses_replicate_centers_across_type_polarity_groups() -> None:
    cones, positions = _teacher_inputs()

    result = _history_teacher(cones=cones, positions=positions)

    cell_positions = result.session.cells.positions_degs.reshape(4, 4, 2)
    for group_positions in cell_positions[1:]:
        assert np.array_equal(group_positions, cell_positions[0])
    assert np.unique(cell_positions[0], axis=0).shape == (4, 2)


def test_teacher_cell_residuals_are_group_centered_and_bounded() -> None:
    cones, positions = _teacher_inputs()
    config = TeacherPopulationConfig(cells_per_type_polarity=4, residual_bound=0.03)

    result = _history_teacher(
        cones=cones,
        positions=positions,
        population_config=config,
    )

    residuals = result.kernels["context_gain_cell_residual"].reshape(4, 4)
    group_scales = result.kernels["context_high_scale"].reshape(4, 4)
    assert np.allclose(residuals.sum(axis=1), 0.0, atol=1e-8)
    assert np.max(np.abs(residuals)) <= config.residual_bound
    assert np.allclose(group_scales.mean(axis=1), (0.85, 0.90, 1.10, 1.15))


def test_teacher_population_is_seed_deterministic() -> None:
    cones, positions = _teacher_inputs()

    first = _history_teacher(cones=cones, positions=positions)
    second = _history_teacher(cones=cones, positions=positions)

    assert first.session.cells.ids == second.session.cells.ids
    assert np.array_equal(first.session.cells.positions_degs, second.session.cells.positions_degs)
    assert np.array_equal(first.kernels["context_high_scale"], second.kernels["context_high_scale"])
    assert np.array_equal(first.session.spike_counts, second.session.spike_counts)


def test_teacher_seed_changes_spikes_and_residual_metadata() -> None:
    cones, positions = _teacher_inputs()

    first = _history_teacher(cones=cones, positions=positions, seed=31)
    second = _history_teacher(cones=cones, positions=positions, seed=32)

    assert not np.array_equal(first.session.spike_counts, second.session.spike_counts)
    assert not np.array_equal(
        first.kernels["context_gain_cell_residual"],
        second.kernels["context_gain_cell_residual"],
    )
    assert tuple(first.kernels["generation_seed"]) == (31,)
    assert tuple(second.kernels["generation_seed"]) == (32,)


def test_teacher_same_modulo_seed_changes_residual_metadata() -> None:
    cones, positions = _teacher_inputs()

    first = _history_teacher(cones=cones, positions=positions, seed=41)
    second = _history_teacher(cones=cones, positions=positions, seed=45)

    assert not np.array_equal(
        first.kernels["context_gain_cell_residual"],
        second.kernels["context_gain_cell_residual"],
    )


def test_teacher_rejects_invalid_population_replicate_counts() -> None:
    cones, positions = _teacher_inputs(cone_count=3)

    with pytest.raises(SyntheticTeacherError, match="positive"):
        TeacherPopulationConfig(cells_per_type_polarity=0)
    with pytest.raises(SyntheticTeacherError, match="finite and non-negative"):
        TeacherPopulationConfig(residual_bound=-0.1)
    with pytest.raises(SyntheticTeacherError, match="distinct"):
        _history_teacher(
            cones=cones,
            positions=positions,
            population_config=TeacherPopulationConfig(cells_per_type_polarity=4),
        )


def test_teacher_population_cell_identity_matches_across_splits() -> None:
    train_cones, positions = _teacher_inputs()
    validation_cones = train_cones + np.float32(0.1)

    train = _history_teacher(
        cones=train_cones,
        positions=positions,
        source_ids=("train-a", "train-b"),
    )
    validation = _history_teacher(
        cones=validation_cones,
        positions=positions,
        source_ids=("validation-a", "validation-b"),
    )

    validate_response_splits((train.session,), (validation.session,))
    assert train.session.cells.ids == validation.session.cells.ids
    assert np.array_equal(
        train.session.cells.positions_degs,
        validation.session.cells.positions_degs,
    )


def _legacy_history_teacher() -> SyntheticTeacherResult:
    cones, positions = _teacher_inputs()
    return _history_teacher(
        cones=cones,
        positions=positions,
        population_config=TeacherPopulationConfig(cells_per_type_polarity=1),
    )


def _teacher_inputs(cone_count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(29)
    cones = rng.random((2, 80, cone_count), dtype=np.float32)
    positions = np.stack((np.arange(cone_count) * 0.05, np.zeros(cone_count)), axis=1)
    return cones, positions


def _history_teacher(
    *,
    cones: np.ndarray,
    positions: np.ndarray,
    source_ids: tuple[str, str] = ("a", "b"),
    seed: int = 31,
    population_config: TeacherPopulationConfig = TeacherPopulationConfig(),
) -> SyntheticTeacherResult:
    return generate_teacher_responses(
        cones,
        positions,
        source_ids,
        np.arange(80) * 0.005,
        trials=32,
        seed=seed,
        adaptive=False,
        teacher_normalization=fit_teacher_input_normalization(cones),
        population_config=population_config,
    )
