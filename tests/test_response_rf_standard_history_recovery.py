from __future__ import annotations

import torch

from evaluation.rf_dynamic import evaluate_dynamic_rf
from test_response_rf_history_contract import (
    _RecordingHistoryModel,
    _conditional_split,
)


def test_standard_train_rate_history_reaches_reset_and_recovery() -> None:
    # Given
    model = _RecordingHistoryModel()
    split = _conditional_split((1.0, 1.0))
    standard = torch.ones(1, split.cone_response.shape[1], 1)

    # When
    evaluate_dynamic_rf(
        model,
        split,
        lag_steps=2,
        history_mode="standard_train_rate",
        standard_history_counts=standard,
        recovery_delays_ms=(5,),
        dt_ms=5.0,
        bootstrap_iterations=1,
        finite_difference_tolerance=None,
    )

    # Then
    reset_histories = tuple(
        history for history in model.histories if history.shape == (1, 2, 1)
    )
    recovery_histories = tuple(
        history for history in model.histories if history.shape == (1, 5, 1)
    )
    recovery_expected = torch.cat(
        (
            standard[:, :-2],
            torch.zeros(1, 1, 1),
            torch.zeros_like(standard[:, -2:]),
        ),
        dim=1,
    )
    assert reset_histories
    assert recovery_histories
    assert all(torch.equal(history, standard[:, -2:]) for history in reset_histories)
    assert all(torch.equal(history, recovery_expected) for history in recovery_histories)
