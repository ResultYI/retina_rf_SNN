from __future__ import annotations

from training.epoch_metrics import weighted_mean_row


def test_weighted_mean_row_uses_batch_sizes() -> None:
    # Given
    rows = (
        (
            2,
            {
                "split": "train_eval",
                "epoch": "3",
                "step": "12",
                "loss_total": "1",
                "rgc_midget_rate": "0.2",
            },
        ),
        (
            1,
            {
                "split": "train_eval",
                "epoch": "3",
                "step": "12",
                "loss_total": "4",
                "rgc_midget_rate": "0.8",
            },
        ),
    )

    # When
    mean = weighted_mean_row(rows)

    # Then
    assert mean["split"] == "train_eval"
    assert mean["epoch"] == "3"
    assert mean["loss_total"] == "2"
    assert mean["rgc_midget_rate"] == "0.4"
