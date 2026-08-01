from __future__ import annotations

import pytest
import torch

from training.response_data import ResponseSplit
from training.response_trainer import (
    ResponseHistoryMode,
    ResponseHistoryModeError,
    ResponseHistoryTrial,
    evaluation_history_counts,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("observed", torch.tensor([[[0.0], [1.0], [0.0], [0.0]]])),
        ("zero", torch.zeros(1, 4, 1)),
        ("shuffled", torch.tensor([[[1.0], [0.0], [0.0], [1.0]]])),
    ),
)
def test_evaluation_history_counts_selects_declared_mode(
    mode: ResponseHistoryMode,
    expected: torch.Tensor | None,
) -> None:
    # Given
    split = _two_trial_split()

    # When
    history = evaluation_history_counts(ResponseHistoryTrial(split, 0, 0), mode)

    # Then
    assert torch.equal(history, expected)


def test_shuffled_history_requires_repeated_trials() -> None:
    # Given
    split = _two_trial_split()
    single_trial = ResponseSplit(
        split.cone_response,
        split.spike_counts[:, :1],
        split.valid_mask[:, :1],
        split.source_ids,
        split.context_ids,
    )

    # When / Then
    with pytest.raises(ResponseHistoryModeError, match="at least two trials"):
        evaluation_history_counts(
            ResponseHistoryTrial(single_trial, 0, 0),
            "shuffled",
        )


def _two_trial_split() -> ResponseSplit:
    counts = torch.tensor(
        [[[[0.0], [1.0], [1.0], [0.0]], [[1.0], [0.0], [0.0], [1.0]]]]
    )
    mask = torch.tensor(
        [[[[True], [True], [False], [True]], [[True], [True], [True], [True]]]]
    )
    return ResponseSplit(
        cone_response=torch.ones(1, 4, 1),
        spike_counts=counts,
        valid_mask=mask,
        source_ids=("source",),
        context_ids=("stationary",),
    )
