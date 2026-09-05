from __future__ import annotations

import torch

from evaluation.mechanistic_retina.clean_sampled_data import (
    CleanBenchmarkConfig,
    build_clean_state,
    nested_budget_state,
)
from evaluation.mechanistic_retina.noise_free_recovery_metrics import probability
from evaluation.mechanistic_retina.clean_sampled_reporting import pathway_parameters


def test_probability_uses_explicit_sampled_history_when_provided() -> None:
    # Given: a canonical teacher and a nonzero sampled causal history.
    state = build_clean_state(
        CleanBenchmarkConfig(
            train_stimuli=2,
            validation_stimuli=1,
            time_steps=16,
            trials=1,
            steps=1,
            checkpoint_steps=(0, 1),
            batch_size=1,
        )
    )
    cones = state.validation_cones
    history = state.validation_spikes[:, 0]

    # When: probability is evaluated through the robustness metric boundary.
    actual = probability(state.teacher, cones, history)
    with torch.no_grad():
        expected = state.teacher.forward_sequence(
            cones, observed_counts=history
        ).spike_probability

    # Then: the metric uses the supplied sampled history exactly.
    torch.testing.assert_close(actual, expected)


def test_pathway_snapshot_flattens_group_specific_ac_gates() -> None:
    # Given: the four-group Canonical V1 parameter contract.
    state = build_clean_state(
        CleanBenchmarkConfig(
            train_stimuli=2,
            validation_stimuli=1,
            time_steps=16,
            trials=1,
            steps=1,
            checkpoint_steps=(0, 1),
            batch_size=1,
        )
    )

    # When: the training evidence captures effective pathway parameters.
    snapshot = pathway_parameters(state.student)

    # Then: AC normalized weights and group gates share one flat audit vector.
    assert snapshot["AC"].ndim == 1
    assert snapshot["AC"].numel() == 112


def test_nested_budget_state_uses_master_repeat_prefix_and_fresh_student() -> None:
    master = build_clean_state(
        CleanBenchmarkConfig(
            train_stimuli=2,
            validation_stimuli=1,
            time_steps=16,
            trials=4,
            steps=1,
            checkpoint_steps=(0, 1),
            batch_size=1,
        )
    )

    budget_one = nested_budget_state(master, 1)
    budget_two = nested_budget_state(master, 2)

    torch.testing.assert_close(budget_one.train_spikes, master.train_spikes[:, :1])
    torch.testing.assert_close(
        budget_two.validation_spikes, master.validation_spikes[:, :2]
    )
    torch.testing.assert_close(
        budget_one.train_spikes,
        budget_two.train_spikes[:, :1],
    )
    assert budget_one.student is not budget_two.student
    for one, two in zip(
        budget_one.student.parameters(), budget_two.student.parameters(), strict=True
    ):
        torch.testing.assert_close(one, two)
